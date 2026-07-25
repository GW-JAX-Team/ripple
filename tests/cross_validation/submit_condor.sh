#!/bin/bash
# Submit the full ripple-vs-LAL accuracy campaign as a single HTCondor GPU job.
# Usage: bash tests/cross_validation/submit_condor.sh  (from anywhere)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
UV="$(which uv)"

N_SAMPLES="${N_SAMPLES:-1000}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/outdir/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTDIR}"

SUB_FILE="${OUTDIR}/campaign.sub"

# Adjust the requirements line to match your cluster's GPU ClassAd attribute.
cat > "${SUB_FILE}" << EOF
executable = ${UV}
arguments  = run --extra cuda --group test --group cross-validation pytest -m accuracy --reference lal --n-samples ${N_SAMPLES} --outdir ${OUTDIR} --cache-reference --plots
initialdir = ${REPO_DIR}
output     = ${OUTDIR}/campaign-\$(ClusterId).\$(ProcId).out
error      = ${OUTDIR}/campaign-\$(ClusterId).\$(ProcId).err
log        = ${OUTDIR}/condor.log
request_GPUs   = 1
request_CPUs   = 4
request_memory = 16GB
# Uncomment to target a specific GPU type:
# requirements = (CUDADeviceName == "NVIDIA H100 80GB HBM3")
queue
EOF

echo "Generated submit file: ${SUB_FILE}"
condor_submit "${SUB_FILE}"
