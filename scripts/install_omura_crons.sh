#!/usr/bin/env bash
set -euo pipefail

# Install/refresh all Omura background cron jobs:
#   - omura-prune-expired-blobs:   hourly  — remove epoch-expired blobs from the vector store
#   - omura-sweep-broken-blobs:    daily   — probe aggregator, drop blobs that 404 repeatedly
#   - omura-reparse-quilts:        every 15 min — expand the next batch of quilts into patches
#
# Idempotent: re-running replaces existing entries with the same tag.
#
# Usage:
#   bash scripts/install_omura_crons.sh

PRUNE_SCHEDULE="${PRUNE_SCHEDULE:-0 * * * *}"        # hourly at :00
SWEEP_SCHEDULE="${SWEEP_SCHEDULE:-30 3 * * *}"      # daily at 03:30 UTC
REPARSE_SCHEDULE="${REPARSE_SCHEDULE:-*/15 * * * *}" # every 15 minutes
SWEEP_WORKERS="${SWEEP_WORKERS:-16}"
SWEEP_PASSES="${SWEEP_PASSES:-2}"
REPARSE_LIMIT="${REPARSE_LIMIT:-30}"     # quilts per run
REPARSE_WORKERS="${REPARSE_WORKERS:-3}"  # parallel — respect public aggregators
# Sweep + reparse run in-process via admin endpoints to avoid clobbering the indexer's
# periodic save. Set OMURA_ADMIN_TOKEN in the API process's env, then mirror it here.
API_URL="${OMURA_API_URL:-http://127.0.0.1:19353}"
ADMIN_TOKEN="${OMURA_ADMIN_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/data/logs"
mkdir -p "${LOG_DIR}"

PRUNE_TAG="# omura-prune-expired-blobs"
PRUNE_CMD="cd \"${REPO_ROOT}\" && uv run python scripts/prune_expired_blobs.py >> \"${LOG_DIR}/prune_expired_blobs.log\" 2>&1"
PRUNE_LINE="${PRUNE_SCHEDULE} ${PRUNE_CMD} ${PRUNE_TAG}"

SWEEP_TAG="# omura-sweep-broken-blobs"
SWEEP_CMD="curl -sS -X POST -H 'Content-Type: application/json' -H 'X-Admin-Token: ${ADMIN_TOKEN}' --data '{\"workers\":${SWEEP_WORKERS},\"passes\":${SWEEP_PASSES}}' ${API_URL}/admin/sweep-broken-blobs >> \"${LOG_DIR}/sweep_broken_blobs.log\" 2>&1 && echo \"\" >> \"${LOG_DIR}/sweep_broken_blobs.log\""
SWEEP_LINE="${SWEEP_SCHEDULE} ${SWEEP_CMD} ${SWEEP_TAG}"

REPARSE_TAG="# omura-reparse-quilts"
REPARSE_CMD="curl -sS -X POST -H 'Content-Type: application/json' -H 'X-Admin-Token: ${ADMIN_TOKEN}' --data '{\"limit\":${REPARSE_LIMIT},\"workers\":${REPARSE_WORKERS},\"resume\":true}' --max-time 1800 ${API_URL}/admin/reparse-quilts >> \"${LOG_DIR}/reparse_quilts.log\" 2>&1 && echo \"\" >> \"${LOG_DIR}/reparse_quilts.log\""
REPARSE_LINE="${REPARSE_SCHEDULE} ${REPARSE_CMD} ${REPARSE_TAG}"

if [[ -z "${ADMIN_TOKEN}" ]]; then
  echo "WARNING: OMURA_ADMIN_TOKEN is empty. The admin endpoints refuse calls without a token."
  echo "  Generate one (e.g. openssl rand -hex 32), set it in the API process's env, then re-run this installer."
fi

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

if crontab -l >/dev/null 2>&1; then
  crontab -l | grep -vE "omura-prune-expired-blobs|omura-sweep-broken-blobs|omura-reparse-quilts" > "${TMP_FILE}" || true
else
  : > "${TMP_FILE}"
fi

printf '%s\n%s\n%s\n' "${PRUNE_LINE}" "${SWEEP_LINE}" "${REPARSE_LINE}" >> "${TMP_FILE}"
crontab "${TMP_FILE}"

echo "Installed cron entries:"
echo "  ${PRUNE_LINE}"
echo "  ${SWEEP_LINE}"
echo "  ${REPARSE_LINE}"
echo
echo "Current crontab:"
crontab -l
