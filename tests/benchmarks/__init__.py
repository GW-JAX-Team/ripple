"""Runtime benchmarking for ripple waveforms.

This directory contains scripts for benchmarking the runtime performance of
ripple's waveform models. These benchmarks are NOT run in CI (hardware varies),
but should be run manually to track performance changes.

To run benchmarks:
    uv run python tests/benchmarks/benchmark_runtime.py

With specific waveforms:
    uv run python tests/benchmarks/benchmark_runtime.py --waveforms IMRPhenomD IMRPhenomXAS

With LAL comparison:
    uv run python tests/benchmarks/benchmark_runtime.py --with-lal

Help:
    uv run python tests/benchmarks/benchmark_runtime.py --help
"""
