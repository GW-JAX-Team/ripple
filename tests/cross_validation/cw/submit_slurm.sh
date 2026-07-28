#!/bin/bash -l
# Submit the CW-vs-CWMakeFakeData large-scale campaign (test_makefakedata_v5_campaign.py)
# as a single CPU Slurm job.
# Usage: bash tests/cross_validation/cw/submit_slurm.sh  (from anywhere -- uses this script's directory)
#
# CPU-only, unlike cross_validation/submit_slurm.sh's GPU job: the bottleneck here is
# LALPulsar's CWMakeFakeData (a serial C call per trial) and the per-sample
# ComputeDetAMResponse loop in _lal_helpers.detector_strain_from_am_response, neither
# of which touch a GPU -- ripple's own polarizations are cheap on CPU at these array
# sizes. campaign.run_campaign parallelizes across trials with a thread pool (LAL
# releases the GIL), so this wants many CPU cores on one node, not a GPU.
#
# Requires RIPPLE_EARTH_EPHEMERIS/RIPPLE_SUN_EPHEMERIS to already point at cached
# ephemeris files reachable from the compute node (compute nodes on many clusters lack
# internet access, so ripplegw's auto-download in resolve_ephemeris_path() won't work
# there -- fetch once on the login node first, e.g.:
#   python3 -c "from ripplegw.waveforms.cw.ephemeris import resolve_ephemeris_path as r; \
#               print(r('earth00-40-DE405.dat.gz')); print(r('sun00-40-DE405.dat.gz'))"
#
# Uses the repo's existing .venv directly rather than `uv run`: `uv run` re-syncs the
# environment against the requested dependency groups, which hits the network (PyPI)
# if the venv doesn't already match exactly -- compute nodes without internet access
# would fail here. Run `uv sync --group test --group cross-validation` once on the
# login node first if .venv isn't already up to date.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")"

N_SAMPLES="${N_SAMPLES:-2000}"
CPUS="${CPUS:-64}"
PARTITION="${PARTITION:-genoa}"
# Deliberately *outside* the tests/ tree (unlike cross_validation/submit_slurm.sh's
# tests/cross_validation/outdir/): pointing --outdir at a path nested inside a
# package directory (tests/cross_validation/cw/ has its own __init__.py) triggered a
# pytest "Plugin already registered under a different name" conftest-loading error
# when the directory was created before pytest ran. A sibling directory outside
# tests/ avoids the ambiguity entirely.
OUTDIR="${OUTDIR:-${REPO_DIR}/accuracy-results/cw/$(date +%Y%m%d-%H%M%S)}"
: "${RIPPLE_EARTH_EPHEMERIS:?Set RIPPLE_EARTH_EPHEMERIS to a local ephemeris file (see script header)}"
: "${RIPPLE_SUN_EPHEMERIS:?Set RIPPLE_SUN_EPHEMERIS to a local ephemeris file (see script header)}"

mkdir -p "${OUTDIR}"

sbatch \
    --partition="${PARTITION}" \
    --time=04:00:00 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --job-name="ripple-cw-accuracy" \
    --output="${OUTDIR}/campaign-%j.out" \
    --export=ALL,RIPPLE_EARTH_EPHEMERIS="${RIPPLE_EARTH_EPHEMERIS}",RIPPLE_SUN_EPHEMERIS="${RIPPLE_SUN_EPHEMERIS}" \
    --wrap="cd '${REPO_DIR}' && source .venv/bin/activate && python3 -m pytest tests/cross_validation/cw/test_makefakedata_v5_campaign.py -v -s --n-samples ${N_SAMPLES} --outdir '${OUTDIR}' --plots"

echo "Submitted CW accuracy campaign: n-samples=${N_SAMPLES}, cpus=${CPUS}, partition=${PARTITION}, output -> ${OUTDIR}"
