#!/bin/bash
#SBATCH --job-name=fetch_atari_head
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=pot-hpi-cpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --account=sci-ebel
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --output=%x_%j.out
# Download and unpack one game of Atari-HEAD (Zenodo record 3451402).
# Not allowed on the login node; submit it instead:
#   sbatch scripts/bound_inference/fetch_atari_head.sh [GAME] [WORKSPACE_DIR]
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
