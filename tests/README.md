# ripple test suite

Two tiers, selected by pytest marker rather than by directory path.
See [`docs/dev/testing.md`](https://gw-jax-team.github.io/ripple/latest/dev/testing/) for the full developer-facing writeup.
This file is the quick-reference.

## Layout

```
tests/
├── conftest.py              # x64, all pytest_addoption, compiled_model fixture
├── psds/                    # PSD data files
├── helpers/                 # importable test utilities, never collected
│   ├── grids.py             # frequency/time axis construction
│   ├── params.py            # canonical_params / random_params_batch
│   ├── metrics.py           # overlap, overlap loss, phase
│   └── config.py            # default_config() constructor kwargs for waveforms
│                             # that can't be built with zero args (CW)
├── unit/                    # fast, pure; no waveform evaluation
├── integration/             # every registered waveform: output format, jit/vmap/grad,
│                             # amplitude/phase + distance, edge cases
└── cross_validation/         # organised by *how* a family is validated, not source type
    ├── reference/            # ReferenceBackend protocol + backends (lal.py, ...) -- the
    │                         # only place LAL/lalsimulation is imported outside cw/
    ├── campaign.py           # fd/'s batch runner (not a test module)
    ├── tolerances.toml       # per-(backend, waveform) thresholds
    ├── test_reference_constants.py  # unmarked; 0 cases if no backend is installed
    ├── test_tolerance_table.py      # unmarked; no backend needed to run
    ├── fd/                   # frequency-domain, stateless-per-call models (domain="FD")
    │   ├── test_overlap.py           # the accuracy campaign, via ReferenceBackend
    │   └── test_phase_convention.py
    ├── cw/                   # continuous-wave: no ReferenceBackend (see lal_agreement.md);
    │   │                     # one file per registered class, reproducing LAL's own
    │   │                     # reference computation from its SWIG-exposed building blocks
    │   ├── _lal_helpers.py            # shared, not collected by pytest
    │   ├── test_exact_pulsar_signal.py
    │   ├── test_pulsar_signal.py
    │   └── test_binary_pulsar_signal.py
    └── submit_slurm.sh / submit_condor.sh
```

## Markers

| Marker | Meaning | Runs in CI? |
| --- | --- | --- |
| *(none)* | `unit/` + `integration/` | Every PR, every Python version |
| `accuracy` | Compares waveform output against a reference backend | Only the `smoke` subset, on PRs/pushes to `main` |
| `smoke` | The cheap `accuracy` subset (3 representative models) | Yes |

```bash
uv run pytest -m "not accuracy"   # CI's default tier
uv run pytest -m "accuracy and smoke" --reference lal --n-samples 3
uv run pytest -m accuracy --reference lal --n-samples 1000   # the real campaign
```

`cross_validation/test_reference_constants.py` is deliberately **not** marked `accuracy`: it compares numeric literals in `ripplegw.constants`, not waveform output.
It runs in the default CI tier wherever a reference backend happens to be installed, contributing zero test cases (not a failure) otherwise.

## Running the accuracy campaign

```bash
uv sync --group test --group cross-validation
uv run pytest -m accuracy --reference lal \
    --n-samples 1000 --outdir accuracy-results --cache-reference --plots
```

Useful flags (see `tests/conftest.py` for the full list): `--reference` (default `lal`), `--n-samples`, `--T` (segment duration override), `--outdir`, `--cache-reference`, `--plots`.
Results land under `accuracy-results/n<N>_T<T>/<backend>_<waveform>.json` (`accuracy-results/` is gitignored).

On a cluster: `bash tests/cross_validation/submit_slurm.sh` or `bash tests/cross_validation/submit_condor.sh`, both single-GPU jobs (`N_SAMPLES`/`OUTDIR` env vars override the defaults).

## Adding a reference backend

Currently only LAL, for frequency-domain models (`fd/`).
A future CPU-based reference for another FD family is one file in `tests/cross_validation/reference/` implementing the `ReferenceBackend` protocol (`available`, `supports`, `constants`, `generate`), registered with `@register_backend`, plus a `[<name>.<waveform>]` block per supported model in `tests/cross_validation/tolerances.toml` -- no existing test file changes, `fd/` picks it up via `--reference <name>`.
A family whose calling convention doesn't fit `ReferenceBackend.generate(name, params, grid)` (time-domain, construction-time state, etc. -- like CW) needs its own directory and its own comparison instead, same as `cw/`.

## Adding a waveform

If it introduces no new parameter name, the test suite covers it automatically -- `integration/` parametrizes off `ripplegw.list_waveforms()`, and `fd/test_overlap.py` off `ripplegw.list_waveforms(domain="FD")` if the model is frequency-domain and a reference backend supports it.
If it does, add the default to `tests/helpers/params.py` (`canonical_params`/`random_params_batch`) and a tolerance row in `tests/cross_validation/tolerances.toml` if a reference backend supports it.
See [`docs/dev/adding_a_waveform.md`](https://gw-jax-team.github.io/ripple/latest/dev/adding_a_waveform/).
