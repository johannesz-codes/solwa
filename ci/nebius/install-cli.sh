#!/usr/bin/env bash
# Official installer; no infrastructure secrets are passed to this step.
set -euo pipefail
installer=$(mktemp)
trap 'rm -f "$installer"' EXIT
curl --fail --silent --show-error --location --retry 3 \
  https://storage.eu-north1.nebius.cloud/cli/install.sh -o "$installer"
bash "$installer"
printf '%s\n' "$HOME/.nebius/bin" >> "$GITHUB_PATH"
"$HOME/.nebius/bin/nebius" version
