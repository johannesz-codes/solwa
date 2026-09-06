#!/usr/bin/env python3
"""Short-lived GPU runners. Infrastructure credentials never enter user data.

Uses Python's standard library and the official Nebius CLI. CLI responses and
errors are captured because instance payloads contain sensitive JIT credentials.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MANAGER = "solwa-gpu-ci-v1"
TTL_SECONDS = 5400
RUNNER_VERSION = "2.337.0"
RUNNER_SHA256 = "70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"


class CIError(RuntimeError):
    """Safe error message suitable for public workflow logs."""


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise CIError(f"Missing {name}; see ci/nebius/README.md")
    return value


def mask(value):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::add-mask::{escaped}", flush=True)


def output(key, value):
    if "\n" in str(value) or "\r" in str(value):
        raise CIError("Invalid workflow output")
    with open(required("GITHUB_OUTPUT"), "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def owner_label(repo):
    return hashlib.sha256(repo.lower().encode()).hexdigest()[:16]


def run_name(repo, run_id, attempt):
    if not re.fullmatch(r"[0-9]+", str(run_id)) or not re.fullmatch(r"[0-9]+", str(attempt)):
        raise CIError("Invalid workflow run identity")
    return f"solwa-{owner_label(repo)}-{run_id}-{attempt}"


def owned(resource, repo):
    labels = resource.get("metadata", {}).get("labels", {})
    return labels.get("managed-by") == MANAGER and labels.get("repository") == owner_label(repo)


def resource_labels(repo, name, now):
    return {"managed-by": MANAGER, "repository": owner_label(repo),
            "ci-run": name, "expires-at": str(int(now + TTL_SECONDS))}


class GitHub:
    def __init__(self, repo, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise CIError("Invalid repository")
        self.repo = repo
        self.token = token
        mask(token)

    def request(self, method, path, data=None, missing_ok=False):
        url = f"https://api.github.com/repos/{self.repo}/{path}"
        request = urllib.request.Request(
            url, data=None if data is None else json.dumps(data).encode(), method=method,
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            raise CIError(f"GitHub {method} {path.split('?')[0]} failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, TimeoutError):
            raise CIError("GitHub request failed or timed out") from None

    def runners(self):
        page = 1
        while True:
            values = self.request("GET", f"actions/runners?per_page=100&page={page}")["runners"]
            yield from values
            if len(values) < 100:
                return
            page += 1

    def resolve(self, ref):
        sha = self.request("GET", "commits/" + urllib.parse.quote(ref, safe=""))["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise CIError("GitHub returned an invalid commit SHA")
        return sha


class Nebius:
    def __init__(self, directory, project):
        self.directory = Path(directory)
        self.project = project
        self.config = self.directory / "nebius-config.yaml"
        credentials = required("NEBIUS_CREDENTIALS_JSON")
        try:
            value = json.loads(credentials)
            if not isinstance(value, dict):
                raise ValueError()
        except ValueError:
            raise CIError("NEBIUS_CREDENTIALS_JSON must contain the generated credentials JSON") from None
        mask(credentials)
        # Mask individual fields too, not only the complete JSON document.
        for field in value.values():
            if isinstance(field, str):
                mask(field)
        credential_path = self.directory / "credentials.json"
        credential_path.write_text(credentials, encoding="utf-8")
        credential_path.chmod(0o600)
        self.call("profile", "create", "solwa-ci", "--endpoint", "api.nebius.cloud",
                  "--service-account-file", str(credential_path), "--parent-id", project,
                  "--skip-auth", profile=False)

    def call(self, *args, payload=None, profile=True, timeout=180):
        command = ["nebius", "--config", str(self.config), "--format", "json",
                   "--no-progress", "--no-check-update", "--no-browser",
                   "--timeout", f"{timeout}s", "--auth-timeout", "60s"]
        if profile:
            command += ["--profile", "solwa-ci"]
        command += list(args)
        with contextlib.ExitStack() as stack:
            if payload is not None:
                handle = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", dir=self.directory))
                json.dump(payload, handle)
                handle.flush()
                command += ["--file", handle.name]
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=timeout + 30, check=False)
            except subprocess.TimeoutExpired:
                raise CIError(f"Nebius {' '.join(args[:3])} timed out; cleanup will reconcile resources") from None
            if result.returncode:
                # Do not echo stderr: it can contain cloud-init/JIT secrets.
                raise CIError(f"Nebius {' '.join(args[:3])} failed (exit {result.returncode}); check credentials, quota and configuration")
            try:
                return json.loads(result.stdout) if result.stdout.strip() else {}
            except ValueError:
                raise CIError("Nebius returned non-JSON output") from None

    def resources(self, kind):
        page = ""
        while True:
            response = self.call("compute", kind, "list", "--parent-id", self.project,
                                 "--page-size", "100", "--page-token", page)
            yield from response.get("items", [])
            page = response.get("next_page_token", "")
            if not page:
                return

    def delete(self, kind, resource_id):
        # Reconcile from a fresh list; tolerate a concurrent reaper deletion only
        # after confirming the resource has disappeared.
        try:
            self.call("compute", kind, "delete", "--id", resource_id)
        except CIError:
            if any(r["metadata"]["id"] == resource_id for r in self.resources(kind)):
                raise


def cloud_init(repo, name, jit):
    script = (ROOT / "bootstrap.sh").read_text()
    settings = {"repo": repo, "name": name, "jit": jit,
                "runner_version": RUNNER_VERSION, "runner_sha256": RUNNER_SHA256}
    # cloud-init accepts JSON as YAML. No string interpolation into shell code.
    config = {
        "users": [{"name": "runner", "shell": "/bin/bash", "lock_passwd": True,
                   "sudo": False}],
        "ssh_pwauth": False,
        "write_files": [
            {"path": "/opt/solwa-ci/bootstrap.sh", "permissions": "0700", "content": script},
            {"path": "/opt/solwa-ci/settings.json", "permissions": "0600",
             "content": json.dumps(settings)},
        ],
        "runcmd": [["bash", "/opt/solwa-ci/bootstrap.sh"]],
    }
    return "#cloud-config\n" + json.dumps(config)


def instance_spec(project, subnet, platform, preset, image, repo, name, jit, now):
    if not preset.startswith("1gpu-"):
        raise CIError("GPU CI currently requires a 1gpu-* preset")
    labels = resource_labels(repo, name, now)
    return {
        "metadata": {"parent_id": project, "name": name, "labels": labels},
        "spec": {
            "resources": {"platform": platform, "preset": preset},
            "boot_disk": {"attach_mode": "READ_WRITE", "managed_disk": {
                "name": name + "-boot", "labels": labels,
                "spec": {"type": "NETWORK_SSD", "size_gibibytes": 80,
                         "source_image_family": {"image_family": image}}}},
            "network_interfaces": [{"name": "eth0", "subnet_id": subnet,
                                    "ip_address": {}, "public_ip_address": {}}],
            "cloud_init_user_data": cloud_init(repo, name, jit),
        },
    }


def preflight(provision=True):
    names = ["NEBIUS_CREDENTIALS_JSON", "NEBIUS_PROJECT_ID", "GH_RUNNER_TOKEN", "GITHUB_REPOSITORY"]
    if provision:
        names += ["NEBIUS_SUBNET_ID"]
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise CIError("Missing configuration: " + ", ".join(missing) + "; see ci/nebius/README.md")
    try:
        credentials = json.loads(required("NEBIUS_CREDENTIALS_JSON"))
        if not isinstance(credentials, dict) or not credentials:
            raise ValueError()
    except ValueError:
        raise CIError("NEBIUS_CREDENTIALS_JSON must contain the generated credentials JSON") from None
    if provision:
        preset = os.environ.get("NEBIUS_PRESET") or "1gpu-8vcpu-32gb"
        if not preset.startswith("1gpu-"):
            raise CIError("GPU CI currently requires a 1gpu-* preset")


def provision(cloud, github, ref, run_id, attempt, ready_timeout=900):
    name = run_name(github.repo, run_id, attempt)
    sha = github.resolve(ref)
    # Reconcile before acquiring new capacity. A stale or failed earlier run
    # must not silently permit an extra GPU despite workflow concurrency.
    existing = [r for r in cloud.resources("instance") if owned(r, github.repo)]
    if existing:
        raise CIError("An existing SOLWA CI VM remains; run GPU cleanup before provisioning")
    jit_response = github.request("POST", "actions/runners/generate-jitconfig", {
        "name": name, "runner_group_id": 1,
        "labels": ["self-hosted", "linux", "x64", "solwa-gpu", name],
        "work_folder": "_work"})
    jit = jit_response["encoded_jit_config"]
    mask(jit)
    output("runner_label", name)
    output("test_sha", sha)
    deadline = time.monotonic() + ready_timeout
    try:
        payload = instance_spec(cloud.project, required("NEBIUS_SUBNET_ID"),
                                os.environ.get("NEBIUS_PLATFORM") or "gpu-l40s-a",
                                os.environ.get("NEBIUS_PRESET") or "1gpu-8vcpu-32gb",
                                os.environ.get("NEBIUS_IMAGE_FAMILY") or "ubuntu24.04-cuda13.0",
                                github.repo, name, jit, time.time())
        # Async create lets us observe the runner without blocking on a long
        # cloud operation. Resource labels exist even if the response is lost.
        cloud.call("compute", "instance", "create", "--async", payload=payload)
        while time.monotonic() < deadline:
            if any(r["name"] == name and r["status"] == "online" for r in github.runners()):
                print(f"GPU runner ready: {name}; test commit {sha}", flush=True)
                return
            time.sleep(10)
        raise CIError("GPU runner did not become ready within 15 minutes")
    except Exception:
        # Outer workflow cleanup is also required for cancellation/SIGKILL.
        try:
            cleanup(cloud, github, name=name)
        except CIError:
            print("::warning::Inline cleanup failed; workflow cleanup and reaper must retry")
        raise


def cleanup(cloud, github, name=None, now=None):
    """Delete only owned resources for an exact run or an expired TTL."""
    now = time.time() if now is None else now
    errors = []
    selected_names = set()

    def selected(resource):
        if not owned(resource, github.repo):
            return False
        labels = resource["metadata"]["labels"]
        if name is not None:
            return labels.get("ci-run") == name
        try:
            return int(labels["expires-at"]) <= now
        except (KeyError, ValueError):
            return False

    for vm in cloud.resources("instance"):
        if selected(vm):
            selected_names.add(vm["metadata"]["labels"]["ci-run"])
            try:
                cloud.delete("instance", vm["metadata"]["id"])
                print(f"Deleted CI VM {vm['metadata']['name']}", flush=True)
            except CIError as exc:
                errors.append(str(exc))
    # Managed boot disks normally disappear with their instance. Reconcile any
    # leftovers from failed creates/deletes, but never delete a live VM's disk.
    active = {r["metadata"]["labels"].get("ci-run")
              for r in cloud.resources("instance") if owned(r, github.repo)}
    for disk in cloud.resources("disk"):
        if selected(disk) and disk["metadata"]["labels"].get("ci-run") not in active:
            try:
                cloud.delete("disk", disk["metadata"]["id"])
            except CIError as exc:
                errors.append(str(exc))
    # Exact-name cleanup also covers JIT registration followed by create failure.
    # For orphan registrations with no VM, use the associated workflow run age.
    prefix = f"solwa-{owner_label(github.repo)}-"
    for runner in github.runners():
        runner_name = runner["name"]
        if runner_name in active:
            continue
        remove = runner_name == name or runner_name in selected_names
        if name is None and runner_name.startswith(prefix) and not remove:
            suffix = runner_name[len(prefix):]
            if re.fullmatch(r"[0-9]+-[0-9]+", suffix):
                run = github.request("GET", f"actions/runs/{suffix.split('-')[0]}", missing_ok=True)
                if run is None:
                    remove = not runner.get("busy", False)
                else:
                    updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00")).timestamp()
                    remove = now - updated > TTL_SECONDS and not runner.get("busy", False)
        if remove:
            try:
                github.request("DELETE", f"actions/runners/{runner['id']}", missing_ok=True)
            except CIError as exc:
                errors.append(str(exc))
    if errors:
        raise CIError("Cleanup incomplete: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "provision", "cleanup", "reap"))
    parser.add_argument("--ref", default=os.environ.get("TEST_REF") or os.environ.get("GITHUB_SHA", ""))
    args = parser.parse_args()
    preflight(provision=args.command in ("check", "provision"))
    if args.command == "check":
        print("Required CI configuration is present; no cloud resources created")
        return
    github = GitHub(required("GITHUB_REPOSITORY"), required("GH_RUNNER_TOKEN"))
    # Directory is outside the checkout and removed even on ordinary failures.
    with tempfile.TemporaryDirectory(prefix="solwa-nebius-") as directory:
        cloud = Nebius(directory, required("NEBIUS_PROJECT_ID"))
        if args.command == "provision":
            provision(cloud, github, args.ref, required("GITHUB_RUN_ID"), required("GITHUB_RUN_ATTEMPT"))
        elif args.command == "cleanup":
            cleanup(cloud, github, name=run_name(github.repo, required("GITHUB_RUN_ID"), required("GITHUB_RUN_ATTEMPT")))
        else:
            cleanup(cloud, github)


if __name__ == "__main__":
    try:
        main()
    except CIError as error:
        print(f"::error::{error}")
        raise SystemExit(1)
