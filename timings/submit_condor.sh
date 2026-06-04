#!/bin/bash
# Submit ripple timing benchmarks as a HTCondor DAG.
# Usage: bash timings/submit_condor.sh  (from anywhere)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
UV="$(which uv)"
OUTDIR="${SCRIPT_DIR}/outdir"
N_WAVEFORMS="10000"
N_RUNS="50"

mkdir -p "${OUTDIR}"

PRECISIONS=("float32" "float64")
MODELS=("TaylorF2" "IMRPhenomD" "IMRPhenomD_NRTidalv2" "IMRPhenomHM" "IMRPhenomPv2" "IMRPhenomXAS" "IMRPhenomXAS_NRTidalv3" "IMRPhenomXHM" "IMRPhenomXP" "IMRPhenomXPHM")

TIMING_SUB="${OUTDIR}/timing.sub"
POSTPROCESS_SUB="${OUTDIR}/postprocess.sub"
DAG_FILE="${OUTDIR}/ripple_timings.dag"

# --- timing submit file ---
# $(MODEL) and $(PRECISION) are HTCondor macros filled per-job by DAGMan VARS.
# Adjust the requirements line to match your cluster's GPU ClassAd attribute.
cat > "${TIMING_SUB}" << EOF
executable = ${UV}
arguments  = run --extra cuda ripple-benchmark \$(MODEL) --device gpu --n-waveforms ${N_WAVEFORMS} --n-runs ${N_RUNS} --precision \$(PRECISION)
initialdir = ${REPO_DIR}
output     = ${OUTDIR}/\$(MODEL)_\$(PRECISION)-\$(ClusterId).\$(ProcId).out
error      = ${OUTDIR}/\$(MODEL)_\$(PRECISION)-\$(ClusterId).\$(ProcId).err
log        = ${OUTDIR}/condor.log
request_GPUs   = 1
request_CPUs   = 1
request_memory = 8GB
# Uncomment to target a specific GPU type:
# requirements = (CUDADeviceName == "NVIDIA H100 80GB HBM3")
queue
EOF

# --- postprocess submit file ---
cat > "${POSTPROCESS_SUB}" << EOF
executable = ${UV}
arguments  = run --group test python src/ripplegw/benchmarks/timings/postprocess.py
initialdir = ${REPO_DIR}
output     = ${OUTDIR}/postprocess-\$(ClusterId).\$(ProcId).out
error      = ${OUTDIR}/postprocess-\$(ClusterId).\$(ProcId).err
log        = ${OUTDIR}/condor.log
request_CPUs   = 1
request_memory = 4GB
queue
EOF

# --- DAG file ---
: > "${DAG_FILE}"
JOB_NAMES=()

for PRECISION in "${PRECISIONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="${MODEL}_${PRECISION}"
        JOB_NAMES+=("${JOB_NAME}")
        printf 'JOB  %s %s\n'                          "${JOB_NAME}" "${TIMING_SUB}"   >> "${DAG_FILE}"
        printf 'VARS %s MODEL="%s" PRECISION="%s"\n'   "${JOB_NAME}" "${MODEL}" "${PRECISION}" >> "${DAG_FILE}"
    done
done

printf '\nJOB  postprocess %s\n'          "${POSTPROCESS_SUB}"   >> "${DAG_FILE}"
printf 'PARENT %s CHILD postprocess\n'    "${JOB_NAMES[*]}"      >> "${DAG_FILE}"

echo "Generated DAG: ${DAG_FILE}"
condor_submit_dag "${DAG_FILE}"
