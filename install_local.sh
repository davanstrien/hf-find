#!/usr/bin/env bash
# Install hf-semsearch as a local hf extension WITHOUT a GitHub remote.
#
# `hf extensions install` only fetches from a public GitHub repo, so for local
# development we replicate exactly what it does: create an isolated venv under
# the hf extensions root, install this package into it, and drop a manifest.json
# so `hf semsearch ...` dispatches to it. Re-run any time to pick up code edits.
set -euo pipefail

SHORT_NAME="semsearch"
ROOT="${HOME}/.local/share/hf/extensions"
EXT_DIR="${ROOT}/hf-${SHORT_NAME}"
VENV="${EXT_DIR}/venv"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing hf-${SHORT_NAME} from ${SRC_DIR}"
rm -rf "${EXT_DIR}"
mkdir -p "${EXT_DIR}"

uv venv "${VENV}" >/dev/null
uv pip install --python "${VENV}/bin/python" "${SRC_DIR}" >/dev/null

EXE="${VENV}/bin/hf-${SHORT_NAME}"
if [[ ! -x "${EXE}" ]]; then
  echo "ERROR: console script not found at ${EXE}" >&2
  exit 1
fi

cat > "${EXT_DIR}/manifest.json" <<JSON
{
  "description": "hf CLI extension: semantic search over Hub datasets & models, with agent-friendly output.",
  "executable_name": "hf-${SHORT_NAME}",
  "executable_path": "${EXE}",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%S.000000+00:00)",
  "owner": "local",
  "repo": "hf-${SHORT_NAME}",
  "repo_id": "local/hf-${SHORT_NAME}",
  "short_name": "${SHORT_NAME}",
  "source": "${SRC_DIR}",
  "type": "python"
}
JSON

echo "Installed. Try:  hf ${SHORT_NAME} --help"
