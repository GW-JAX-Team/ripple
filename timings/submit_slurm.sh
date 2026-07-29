#!/bin/bash -l
# Submit one GPU job per (model, precision) combination, then chain postprocessing.
# Usage: bash timings/submit_slurm.sh  (from anywhere — uses this script's directory)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

DEVICE="gpu"
N_WAVEFORMS="10000"
N_RUNS="50"

PRECISIONS=("float32" "float64")
# Registry-driven: whatever ripplegw.list_waveforms() reports, so a newly
# registered CBC model is picked up automatically with no edits to this
# script. timing.py only supports CBC waveforms -- other
# source types (e.g. burst) have their own parameterisation and would fail.
MODELS=($(cd "${REPO_DIR}" && uv run python -c "import ripplegw; print(' '.join(sorted(ripplegw.list_waveforms(source_type='cbc'))))"))

mkdir -p "${SCRIPT_DIR}/outdir"

JOB_IDS=()

for PRECISION in "${PRECISIONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        JOB_ID=$(sbatch \
            --partition=gpu_h100 \
            --time=00:10:00 \
            --ntasks=1 \
            --gpus-per-task=1 \
            --job-name="ripple-${MODEL}-${PRECISION}" \
            --output="${SCRIPT_DIR}/outdir/${MODEL}_${PRECISION}-%j.out" \
            --parsable \
            --wrap="cd '${REPO_DIR}' && uv run --extra cuda ripple-benchmark '${MODEL}' --device ${DEVICE} --n-waveforms ${N_WAVEFORMS} --n-runs ${N_RUNS} --precision ${PRECISION}")
        echo "Submitted ${MODEL} (${PRECISION}): job ${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
    done
done

# Build afterok dependency string
DEPENDENCY=$(IFS=:; echo "afterany:${JOB_IDS[*]}")

sbatch \
    --partition=rome \
    --time=00:01:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --dependency="${DEPENDENCY}" \
    --job-name="ripple-postprocess" \
    --output="${SCRIPT_DIR}/outdir/postprocess-%j.out" \
    --wrap="cd '${REPO_DIR}' && uv run --group test python src/ripplegw/benchmarks/timings/postprocess.py"

echo "Postprocessing job submitted with dependency on: ${JOB_IDS[*]}"
