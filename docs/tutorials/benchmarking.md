# Benchmarking

This guide shows how to benchmark waveform generation in ripple.

The recommended workflow is to use the maintained scripts in `timings/`, which run benchmarks and postprocessing for you.

## 1. Run a benchmark

ripple installs the `ripple-benchmark` command, which times waveform generation for a chosen approximant.

```bash
ripple-benchmark IMRPhenomXAS \
  --device gpu \
  --precision float64 \
  --n-waveforms 20000 \
  --n-runs 5 \
  --duration 4.0 \
  --f-min 20.0 \
  --f-max 2048.0 \
  --f-ref 50.0
```

By default, results are written to `timings/outdir/<waveform>_<device>_<precision>.json`.

## 2. Recommended script workflows

### Local CPU smoke test + postprocessing

From the repository root:

```bash
bash timings/test.sh
```

This runs a small benchmark sweep and then automatically runs postprocessing.

### Slurm GPU batch + postprocessing

```bash
bash timings/submit_slurm.sh
```

This submits one job per `(model, precision)` pair and then submits a dependent postprocessing job.

### HTCondor DAG + postprocessing

```bash
bash timings/submit_condor.sh
```

This generates a DAG, submits all timing jobs, and runs postprocessing after all timing jobs complete.

## 3. Understand the timing output

Each run reports:

- first-run time (includes JIT compilation)
- mean/std/min/max execution time over timed runs
- mean time per waveform (ms)
- mean waveforms per second

For fair comparisons, use the timed runs and treat the first run separately.

The postprocessing step creates bar plots for:

- time per waveform
- throughput (waveforms per second)

By default, generated figures are written to `timings/figures`.
