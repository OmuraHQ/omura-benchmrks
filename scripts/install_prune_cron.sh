#!/usr/bin/env bash
set -euo pipefail

# Installs/updates a user crontab entry to run blob pruning regularly.
#
# Usage:
#   bash scripts/install_prune_cron.sh
#   bash scripts/install_prune_cron.sh "0 * * * *"
#   bash scripts/install_prune_cron.sh "*/30 * * * *" 22

CRON_SCHEDULE="${1:-0 * * * *}"
FORCED_EPOCH="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/data/logs"
mkdir -p "${LOG_DIR}"

CMD="cd \"${REPO_ROOT}\" && uv run python scripts/prune_expired_blobs.py"
if [[ -n "${FORCED_EPOCH}" ]]; then
  CMD="${CMD} --epoch ${FORCED_EPOCH}"
fi
CMD="${CMD} >> \"${LOG_DIR}/prune_expired_blobs.log\" 2>&1"

TAG="# omura-prune-expired-blobs"
CRON_LINE="${CRON_SCHEDULE} ${CMD} ${TAG}"

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

if crontab -l >/dev/null 2>&1; then
  crontab -l | grep -v "omura-prune-expired-blobs" > "${TMP_FILE}" || true
else
  : > "${TMP_FILE}"
fi

echo "${CRON_LINE}" >> "${TMP_FILE}"
crontab "${TMP_FILE}"

echo "Installed cron entry:"
echo "  ${CRON_LINE}"
echo
echo "Current crontab:"
crontab -l
