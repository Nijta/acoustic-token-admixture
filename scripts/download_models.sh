#!/usr/bin/env bash
# Download the released checkpoints and speaker pool from Hugging Face and
# verify every file against SHA256SUMS.
#
# Usage: scripts/download_models.sh [target_dir]   (default: ./models)
#        ENGLISH_ONLY=1 scripts/download_models.sh  (skip the French checkpoints)
set -euo pipefail

REPO_ID="brijsri/acoustic-token-admixture"
TARGET="${1:-models}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' command was not found. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

args=(download "$REPO_ID" --local-dir "$TARGET")
if [[ "${ENGLISH_ONLY:-0}" == "1" ]]; then
  args+=(--exclude "*french*" --exclude "*_fr.pth")
fi
hf "${args[@]}"

cd "$TARGET"
sums=SHA256SUMS
if [[ "${ENGLISH_ONLY:-0}" == "1" ]]; then
  grep -vE 'french|_fr\.pth' SHA256SUMS > SHA256SUMS.english
  sums=SHA256SUMS.english
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c --quiet "$sums"
else
  shasum -a 256 -c --quiet "$sums"
fi

echo "All files verified. Now run:"
echo "  export MODELS_DIR=$(pwd)"
