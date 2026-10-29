# Nebius GPU CI

CPU checks remain on GitHub-hosted runners. GPU CI creates one fresh Nebius VM,
registers a one-job GitHub runner, runs the physics tests with `--device=cuda`,
and deletes the VM and its managed boot disk. No workstation or research cluster
is involved. No cloud resources are created just by merging this change.

## First-time setup

No source edits are required. You need **two secrets and two infrastructure IDs**.
API keys alone cannot identify which Nebius project/subnet should be billed.
The default is one L40S in `eu-north1`; another region needs a matching platform
and single-GPU preset override below. GPU quota and actual capacity must exist.

1. Use a dedicated Nebius CI project and an existing subnet with internet access.
   Give a CI service account permission to list/create/get/delete Compute
   instances and disks and use that subnet. Keep the grant scoped to this project;
   do not use a tenant administrator account. The VM itself has no service account.
2. Generate its authorized-key credentials file with the Nebius CLI:

   ```sh
   nebius iam auth-public-key generate \
     --service-account-id YOUR_CI_SERVICE_ACCOUNT_ID \
     --output nebius-ci-credentials.json
   ```

   Keep this file out of git. This is a Nebius **AI Cloud service-account key**,
   not a Nebius Token Factory/inference API key or a short-lived bearer token.
3. Create a fine-grained GitHub PAT for **only `johannesz-codes/solwa`** with
   **Administration: read/write**, **Contents: read**, and **Actions: read**.
   Administration is required by GitHub's JIT runner API; Actions read is used
   to reconcile abandoned runner registrations. The workflow's `GITHUB_TOKEN`
   cannot register runners. Set an expiration and rotate the token before expiry.
4. In SOLWA **Settings → Environments**, create **`nebius-ci`**. Leave deployment
   branches unrestricted: labelled PR runs deploy from `refs/pull/<n>/merge`, so a
   `main`-only rule would block them. Fork PRs never run GPU jobs. Do not add
   required reviewers; unattended cleanup must run without manual approval.
5. Add these **environment secrets**:

   | Name | Value |
   | --- | --- |
   | `NEBIUS_CREDENTIALS_JSON` | Complete generated credentials JSON from step 2 |
   | `GH_RUNNER_TOKEN` | Fine-grained PAT from step 3 |

6. Add these **environment variables** (non-secret IDs):

   | Name | Value |
   | --- | --- |
   | `NEBIUS_PROJECT_ID` | The dedicated CI project ID |
   | `NEBIUS_SUBNET_ID` | A subnet in that project/region |

   Find the values in the Nebius console or `nebius config get parent-id` and
   `nebius vpc subnet list --parent-id YOUR_PROJECT_ID --format json`.
7. Add **repository variable** `NEBIUS_CI_CLEANUP_ENABLED=true`. This enables the
   independent cleanup workflow and is required for GPU runs on main. Keep it
   enabled even when disabling automatic GPU tests.
8. Once this workflow is on `main`, open **Actions → GPU CI → Run workflow**.
   Leave `ref` empty to test the workflow commit, or enter a reviewed full commit
   SHA. Expect the stages `provision`, `test`, `cleanup`. The first boot downloads
   dependencies and may take several minutes; readiness (including waiting for GPU capacity) is limited to 60 minutes.
9. After the first successful run, set **repository variable**
   `NEBIUS_CI_ENABLED=true` to run GPU tests after the CPU tests on every
   **push to main**.

## Running GPU tests on a pull request

GPU tests are opt-in per PR. Add the label **`gpu-ci`** to a PR from this
repository: CI reruns, and the `gpu` job runs after the CPU `test` job passes.
Every later push to the labelled PR runs it again; remove the label to stop.
Without the label the `gpu` job is skipped. It is not a required check. PRs from
forks never run GPU jobs and never receive infrastructure secrets.

You may set the secrets from files using GitHub CLI, without pasting keys into
shell command arguments:

```sh
gh secret set NEBIUS_CREDENTIALS_JSON --repo johannesz-codes/solwa \
  --env nebius-ci < nebius-ci-credentials.json
gh secret set GH_RUNNER_TOKEN --repo johannesz-codes/solwa --env nebius-ci
```

The PAT can later be replaced with a short-lived GitHub App installation token
passed as `GH_RUNNER_TOKEN`; token minting is not part of this initial version.

## Optional environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `NEBIUS_PLATFORM` | `gpu-l40s-a` | L40S / `eu-north1` |
| `NEBIUS_PRESET` | `1gpu-8vcpu-32gb` | Only `1gpu-*` accepted |
| `NEBIUS_IMAGE_FAMILY` | `ubuntu24.04-cuda13.0` | Must provide Ubuntu 24.04 and working NVIDIA drivers |

Boot disk size is 80 GiB. It is managed by the VM and is deleted with it. Public
IPv4 is dynamically allocated for outbound access and is released with the VM.
No persistent IP, SSH key, data volume, or cloud service account is attached.

## Credentials and trust boundary

- `provision` and `cleanup` execute the workflow's own commit on GitHub-hosted
  machines. They never check out the requested test ref. Protect changes to
  workflows and `ci/nebius/` on main. The bootstrap branch is equally trusted
  while its environment access is enabled.
- Nebius credentials and PAT are passed only to the steps that need them, stored
  temporarily outside the checkout, and excluded from CLI output/error logs.
- `test` has only a read-only `GITHUB_TOKEN`, no `nebius-ci` environment, and no
  cloud credentials. It runs as an unprivileged user with no sudo or Docker access.
  Cloud metadata access is blocked for that user. Inbound traffic is blocked.
- Only the single-use GitHub JIT configuration enters cloud-init. It is consumed
  on startup. The VM is never reused for another job.
- Workflow actions and the NVIDIA image are trusted dependencies. Do not use
  `pull_request_target` to run a PR's scripts with infrastructure secrets.
- `ref` is resolved through the GitHub API to a full SHA before allocation. It is
  never interpolated into shell code. Review that SHA before testing external code.

## Tests and versions

The GPU `test` job installs pinned Python packages (before checking out code):
PyTorch 2.7.1 / cu128, NumPy 2.2.6, SciPy 1.15.3, pytest 8.3.5, using a pinned
`uv` (0.12.24) for faster downloads. The NVIDIA driver
comes from the CUDA 13.0 image and can run the older cu128 wheel. CPU CI continues
to exercise the normal package dependency range separately.

The runner download is pinned to 2.337.0 and SHA-256 checked. Automatic GitHub
runner updates remain enabled. Refresh the pinned baseline periodically to avoid
repeated updates. The official Nebius CLI installer and image family track their
current releases; a future custom image can remove repeated package downloads.

`tests/conftest.py` adds `--device=cpu|cuda` (default CPU). An explicit CUDA run
fails at session startup if CUDA is absent or a simple GPU kernel fails. The same
physics tests use the selected device. Reports and runner diagnostics are uploaded
before teardown; infrastructure cleanup failure is a separate failing job.

Infrastructure tests use only Python's standard library and fake cloud clients.
They run in the **GPU infrastructure** workflow only when `ci/nebius/` or the CI
workflows change, and as the first step of every `provision` job, so no VM is
created with a broken script. Locally:

```sh
python -m unittest discover -s ci/nebius/tests -v
bash -n ci/nebius/bootstrap.sh
bash -n ci/nebius/install-cli.sh
python -m pytest tests/ --device=cpu -v
```

## Timeouts, cancellation, and recovery

One workflow at a time is allowed; additional runs wait (GitHub may replace an
older pending run). Provisioning also refuses to create another VM while an
owned CI VM exists. No automatic capacity retry can allocate additional GPUs.

Resources are labeled with repository identity, run ID/attempt, and a **150-minute
expiry**. Normal cleanup runs even after test/provision failures. A separate
workflow handles completed/cancelled GPU workflows, and scans expired resources
four times per hour. It also removes orphan runner registrations and boot disks.
Deletion is scoped to this repository's labels; unrelated VMs/disks are preserved.

**The expiry label is not a Nebius-enforced timer.** GitHub schedules can be
delayed, and expired credentials/outages can prevent cleanup. This is a recovery
mechanism, not a guaranteed billing cap. Monitor cleanup failures and configure
Nebius budget alerts. Job execution timeouts do not bound GitHub queue time.

While waiting for the runner, `provision` logs every VM state change with the
elapsed time. `CREATING`/`STARTING` are left alone however long they take. When
the VM is `STOPPED` or `ERROR` and no Nebius operation is still running on it,
the failed operation's error (for example `NotEnoughResources`, i.e. no free GPU
host) is logged and `start` is sent again, without a fixed limit. Running out of
capacity is not treated as a failure; only the 60-minute readiness timeout is.
Never send a start while another operation runs: Nebius aborts the running one
(even the initial create). If the runner is still not online, the error says
whether the VM never reached `RUNNING` (Nebius side) or ran without the runner
coming online (bootstrap side), and the VM's boot log is printed in a collapsed
group in the `provision` log before the VM is deleted. That log is public on
this repository; known secrets are masked.

For manual recovery, use the Nebius console to delete the CI VM (not Linux
`shutdown`) and check its managed disk, or use the trusted script locally with
Nebius CLI installed and the environment secrets/variables set:

```sh
# Deletes expired owned resources only:
python3 ci/nebius/runner.py reap

# Delete a particular failed run immediately:
GITHUB_RUN_ID=123456 GITHUB_RUN_ATTEMPT=1 python3 ci/nebius/runner.py cleanup
```

Do not unset cleanup configuration or revoke its credentials until no CI resources
remain. To stop new automatic allocations, set `NEBIUS_CI_ENABLED=false` and
remove the `gpu-ci` label from open PRs.

## References

- [Nebius VM creation / managed disks](https://docs.nebius.com/compute/virtual-machines/manage)
- [Nebius service-account authorized keys](https://docs.nebius.com/iam/service-accounts/authorized-keys)
- [Nebius VM deletion](https://docs.nebius.com/compute/virtual-machines/delete)
- [GitHub ephemeral runners and autoscaling](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [GitHub JIT runner API](https://docs.github.com/en/rest/actions/self-hosted-runners#create-configuration-for-a-just-in-time-runner-for-a-repository)
