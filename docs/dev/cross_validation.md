# Cross-validation tests

Use cross-validation when a waveform implementation or its reference adapter changes.
Each large-scale test evaluates one named waveform against its registered reference and is intentionally separate from the fast checks run by CI.

## Before submitting a job

Prepare the environment on a login node, from the repository root:

```bash
uv sync --group test --group cross-validation
source .venv/bin/activate
```

For a CUDA-enabled JAX installation, use `uv sync --extra cuda --group test --group cross-validation` instead.
Do this before submitting: compute nodes commonly cannot reach package indexes.

Continuous-wave (CW) tests also need LALSuite's Earth and Sun ephemerides on storage visible to the worker
Set both paths before submitting:

```bash
export RIPPLE_EARTH_EPHEMERIS=/shared/path/earth00-40-DE405.dat.gz
export RIPPLE_SUN_EPHEMERIS=/shared/path/sun00-40-DE405.dat.gz
```

## Submit one waveform

`tests.cross_validation.submit` is the single entry point for local, Slurm, and HTCondor runs.
Choose a registered waveform name and scheduler, then set a sample count and an output directory on storage available to the job:

```bash
python -c 'import ripplegw; print(*ripplegw.list_waveforms(), sep="\n")'
```

```bash
python -m tests.cross_validation.submit \
    --scheduler slurm \
    --waveform IMRPhenomD \
    --n-samples 1000 \
    --outdir /shared/scratch/ripple-cross-validation/run-YYYYMMDD \
    --plots
```

For HTCondor, change `--scheduler slurm` to `--scheduler condor`.
Use `--scheduler local` to run the same selected test in the current environment before committing cluster resources.
`--plots` is optional; omit it when only machine-readable results are needed.

For a small local preflight:

```bash
python -m tests.cross_validation.submit \
    --scheduler local --waveform SineGaussian --n-samples 10 \
    --outdir accuracy-results/local-preflight
```

Use `--waveform all` to submit every waveform with a registered large-scale test.
The launcher reports models that have no test rather than silently treating them as validated.

Before a first submission, inspect the rendered job without sending it to the scheduler:

```bash
python -m tests.cross_validation.submit \
    --scheduler slurm --waveform IMRPhenomD --n-samples 1000 --dry-run
```

The built-in defaults are tuned for the development cluster: frequency-domain tests request a GPU, while time-domain tests request CPU cores.
Adapt them for your site with `--partition`, `--cpus`, `--gpus`, `--memory`, and `--time`.
`--help` documents every option.

## What is compared

| Waveform family | Comparison |
| --- | --- |
| Frequency-domain models | ripple and the reference backend are evaluated on the same frequency grid and scored with a PSD-weighted overlap loss. |
| Time-domain models | A waveform-specific adapter produces aligned real time series and scores a white, normalized time-domain mismatch plus a relative-norm amplitude diagnostic. No FFT is introduced. |

The two metrics are intentionally different: a PSD-weighted frequency-domain overlap is meaningful for frequency-domain waveforms, while a time-domain test compares directly generated samples on the same uniform time grid.
The normalized mismatch is phase- and shape-sensitive, with no time/phase maximization, but is invariant under a global positive amplitude scale.
The relative-norm diagnostic supplies the corresponding amplitude check.

Time-domain validation is a framework, not a generic CW fallback.
Each time-domain waveform needs its own reference adapter, aligned-grid construction, and acceptance criteria.
Current large-scale tests cover the three CW models and SineGaussian; the launcher runs only registered waveform-specific tests and reports any future exclusions.
See [Reference Implementations](reference_implementations.md) for thresholds, known reference limits, and coverage.

## Results and failures

The launcher resolves `--outdir` to an absolute path and creates one subdirectory per selected waveform for the job log and test results; `--plots` adds figures there.
Keep each run in a fresh directory so its parameters and outputs remain attributable to one submission.

A failed test is evidence to investigate, not a reason to loosen a threshold immediately.
First check that the reference dependency, ephemerides (for CW), waveform configuration, and selected sample range match the documented test.
Then compare the saved results with the relevant entry in [Reference Implementations](reference_implementations.md).

## Adding a test

A new waveform is covered by the fast integration suite once it is registered, but reference validation is not automatic.
Add or extend the waveform's reference comparison before advertising cross-validation support.
Frequency-domain models use the reference-backend/tolerance mechanism; time-domain models need a direct, aligned-sample time-domain comparison plus an amplitude diagnostic.
Do not convert a time-domain waveform with an FFT solely to fit the frequency-domain harness.
