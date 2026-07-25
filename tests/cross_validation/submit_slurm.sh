#!/bin/bash -l
# Submit the full ripple-vs-LAL accuracy campaign as a single GPU job.
# Usage: bash tests/cross_validation/submit_slurm.sh  (from anywhere -- uses this script's directory)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

N_SAMPLES="${N_SAMPLES:-1000}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outdir/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTDIR}"

sbatch \
    --partition=gpu_h100 \
    --time=04:00:00 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --job-name="ripple-accuracy" \
    --output="${OUTDIR}/campaign-%j.out" \
    --wrap="cd '${REPO_DIR}' && uv run --extra cuda --group test --group cross-validation pytest -m accuracy --reference lal --n-samples ${N_SAMPLES} --outdir '${OUTDIR}' --cache-reference --plots"

echo "Submitted accuracy campaign: n-samples=${N_SAMPLES}, output -> ${OUTDIR}"
