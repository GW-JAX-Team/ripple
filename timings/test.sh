#!/usr/bin/env bash
DEVICE="cpu"
N_WAVEFORMS="1000"
N_RUNS="5"
PRECISION="float32"

uv run ripple-benchmark TaylorF2 --device "$DEVICE" --n-waveforms "$N_WAVEFORMS" --n-runs "$N_RUNS" --precision "$PRECISION"
uv run ripple-benchmark IMRPhenomD --device "$DEVICE" --n-waveforms "$N_WAVEFORMS" --n-runs "$N_RUNS" --precision "$PRECISION"
uv run ripple-benchmark IMRPhenomXAS --device "$DEVICE" --n-waveforms "$N_WAVEFORMS" --n-runs "$N_RUNS" --precision "$PRECISION"
uv run ripple-benchmark IMRPhenomPv2 --device "$DEVICE" --n-waveforms "$N_WAVEFORMS" --n-runs "$N_RUNS" --precision "$PRECISION"

echo "Running postprocessing..."
uv run --group test python src/ripplegw/benchmarks/timings/postprocess.py --device "$DEVICE"
