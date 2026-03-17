DEVICE="cpu"
N_WAVEFORMS="1000"
PRECISION="float32"

uv run ripple_time TaylorF2 --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
uv run ripple_time IMRPhenomD --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
uv run ripple_time IMRPhenomXAS --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
uv run ripple_time IMRPhenomPv2 --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION

echo "Running postprocessing..."
uv run --group test python src/ripplegw/benchmarks/timings/postprocess.py --device $DEVICE
