# Run Reference Checks

Cross-validation runs the large-scale reference sweep for one registered waveform.
It is separate from the fast suite and from the focused reference regressions listed in [Test Coverage](test_coverage.md).

## Set up the environment

From the repository root, prepare the environment before submitting work to a cluster:

```bash
uv sync --group test --group cross-validation
source .venv/bin/activate
```

For CUDA-enabled JAX, add `--extra cuda` to `uv sync`.

Continuous-wave (CW) targets also require LALSuite Earth and Sun ephemerides visible to the worker:

```bash
export RIPPLE_EARTH_EPHEMERIS=/shared/path/earth00-40-DE405.dat.gz
export RIPPLE_SUN_EPHEMERIS=/shared/path/sun00-40-DE405.dat.gz
```

## Preview and run a target

The launcher is the entry point for local, Slurm, and HTCondor runs.
Preview a target first to see the exact pytest node and resource request:

```bash
python -m tests.cross_validation.submit \
    --scheduler local --waveform IMRPhenomD --n-samples 10 \
    --outdir accuracy-results/preflight --dry-run
```

Submit the same target to Slurm:

```bash
python -m tests.cross_validation.submit \
    --scheduler slurm \
    --waveform IMRPhenomD \
    --n-samples 1000 \
    --outdir /shared/scratch/ripple-cross-validation/run-YYYYMMDD \
    --plots
```

Use `--scheduler condor` for HTCondor, or omit `--plots` when only the machine-readable result is needed.
`--waveform all` makes one independently configured submission per available adapter and reports registered waveforms that have no large-scale test.
Add `--dry-run` to inspect all submissions without sending them to the scheduler.

The launcher chooses resources by test method: frequency-domain targets request a GPU, while SineGaussian and CW targets are CPU jobs.
Override site-specific defaults with `--partition`, `--cpus`, `--gpus`, `--memory`, and `--time`.

## What the sweep compares

| Family | Comparison |
| --- | --- |
| Frequency-domain models | ripple and LALSuite on a shared frequency grid, scored by an ET-D PSD-weighted overlap loss. Non-precessing models also check the absolute phase convention. |
| Time-domain models | A waveform-specific LAL reference on aligned real samples, scored by a normalized time-domain mismatch and a relative-norm amplitude diagnostic. No FFT is introduced. |

The current waveform-to-test map, including focused regressions not selected by the launcher, is in [Test Coverage](test_coverage.md).
[Reference Comparisons and Limits](reference_comparisons_and_limits.md) records the acceptance thresholds and any limits of a reference path.

## Results and failures

`--outdir` is resolved to an absolute path.
The launcher creates one subdirectory per waveform for logs and results; `--plots` adds figures there.
Use a fresh output directory for each submission.

A failed sweep is a signal to investigate.
First confirm the selected waveform, reference dependency, ephemerides for CW, and sample configuration.
Then compare the failure with the applicable threshold and reference notes.
Do not loosen a threshold based on one run.

## Adding a sweep

Reference comparison coverage is not automatic for a new waveform.
Add a focused, waveform-appropriate comparison, its acceptance threshold, and a launcher adapter.
Frequency-domain models use the LAL overlap path when supported; time-domain models need an aligned direct comparison plus the amplitude diagnostic.
Do not transform a time-domain waveform with an FFT merely to use the frequency-domain harness.
