#!/bin/bash
# Download and unpack one game of Atari-HEAD (Zenodo record 3451402).
# Usage: GAME=asterix bash fetch_atari_head.sh
set -euo pipefail
GAME=${GAME:-asterix}
WORKSPACE_DIR=${WORKSPACE_DIR:-$(ws_find DreamerSpace)}
DEST=${WORKSPACE_DIR}/Atari-HEAD
mkdir -p "${DEST}"
cd "${DEST}"
curl -fL -o "${GAME}.zip" "https://zenodo.org/records/3451402/files/${GAME}.zip"
curl -fL -o meta_data.csv "https://zenodo.org/records/3451402/files/meta_data.csv"
unzip -q -o "${GAME}.zip"
rm "${GAME}.zip"
echo "Atari-HEAD ${GAME} in ${DEST}/${GAME}"
