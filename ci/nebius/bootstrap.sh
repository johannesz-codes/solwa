#!/usr/bin/env bash
# Runs as root only during bootstrap, before accepting repository code.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Progress markers on the serial console, which provision prints on timeout.
step() { echo "solwa-bootstrap: $*" | tee /dev/console >/dev/null || true; }
trap 'step "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR
step "installing system packages"
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl git python3-venv libicu74 iptables
# No inbound services or SSH keys are needed. Preserve outbound HTTPS and DNS.
iptables -I INPUT 1 -i lo -j ACCEPT
iptables -I INPUT 2 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -P INPUT DROP
# Do not expose the cloud metadata service to the unprivileged job user.
iptables -A OUTPUT -d 169.254.169.254 -m owner --uid-owner runner -j REJECT
step "checking GPU"
nvidia-smi
step "downloading GitHub runner"
python3 - <<'PY'
import hashlib,json,pathlib,subprocess,urllib.request
root = pathlib.Path('/opt/solwa-ci')
settings = json.loads((root/'settings.json').read_text())
version = settings['runner_version']
url = f'https://github.com/actions/runner/releases/download/v{version}/actions-runner-linux-x64-{version}.tar.gz'
archive = root/'runner.tar.gz'
with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as out:
    while chunk := response.read(1024*1024):
        out.write(chunk)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == settings['runner_sha256'], 'Runner checksum mismatch'
runner_dir = pathlib.Path('/home/runner/actions-runner')
runner_dir.mkdir(parents=True, exist_ok=True)
subprocess.run(['tar','xzf',str(archive),'-C',str(runner_dir)],check=True)
# Keep credentials out of command-line arguments until run.sh needs its JIT input.
(runner_dir/'jit-config').write_text(settings['jit'])
(runner_dir/'jit-config').chmod(0o600)
(root/'settings.json').unlink()
archive.unlink()
PY
step "installing runner dependencies"
bash /home/runner/actions-runner/bin/installdependencies.sh
chown -R runner:runner /home/runner/actions-runner
# Register as early as possible: Python test dependencies are installed by the
# test job itself, so their progress and timing show up in the GitHub log.
cat > /opt/solwa-ci/start-runner.sh <<'START'
#!/usr/bin/env bash
set -euo pipefail
jit=$(cat jit-config)
rm jit-config
exec ./run.sh --jitconfig "$jit"
START
chmod 0755 /opt/solwa-ci/start-runner.sh
cat > /etc/systemd/system/solwa-runner.service <<'UNIT'
[Unit]
Description=SOLWA one-job GitHub runner
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=runner
WorkingDirectory=/home/runner/actions-runner
Environment=PATH=/home/runner/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/bin/bash /opt/solwa-ci/start-runner.sh
Restart=no
KillMode=control-group
# If the runner exits (e.g. registration rejected), copy its log to the console.
ExecStopPost=+/bin/sh -c 'journalctl -u solwa-runner --no-pager -n 40 | sed "s/^/solwa-runner: /" > /dev/console'
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
step "starting GitHub runner"
systemctl enable --now solwa-runner.service
step "runner service started"
