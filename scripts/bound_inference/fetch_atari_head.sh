#!/bin/bash
# Download and unpack one game of Atari-HEAD (Zenodo record 3451402).
# Usage: bash fetch_atari_head.sh [GAME] [WORKSPACE_DIR]
set -euo pipefail
GAME=${1:-asterix}
WORKSPACE=${2:-/sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs}
DEST=${WORKSPACE}/Atari-HEAD
mkdir -p "${DEST}"
cd "${DEST}"
curl -fL -o "${GAME}.zip" "https://zenodo.org/records/3451402/files/${GAME}.zip"
curl -fL -o meta_data.csv "https://zenodo.org/records/3451402/files/meta_data.csv"
unzip -q -o "${GAME}.zip"
rm "${GAME}.zip"
echo "Atari-HEAD ${GAME} in ${DEST}/${GAME}"
