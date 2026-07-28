# Testing

ripple has two complementary test tiers.
CI runs on every pull request and provides fast feedback on the package interface and JAX compatibility.
More expensive accuracy campaigns compare waveform outputs with a reference implementation at a scale that is impractical in CI.

Two pytest markers, rather than directory paths, select the tier to run.

## The two tiers

| Marker | What it checks | Where it runs |
| --- | --- | --- |
| *(none)* | Every registered waveform's output format, JAX transformations, amplitude/phase and distance behaviour, and edge cases | Every PR, every supported Python version |
| `accuracy` | ripple's output against a reference backend (currently LAL) | The `smoke` subset on PRs and pushes to `main`; the full campaign outside CI |

```bash
uv run pytest -m "not accuracy"                                         # what CI runs
uv run pytest -m "accuracy and smoke" --reference lal --n-samples 3     # CI's accuracy check
uv run pytest -m accuracy --reference lal --n-samples 1000              # full accuracy campaign
```

`cross_validation/test_reference_constants.py` compares numeric literals in `ripplegw.constants` with a reference backend rather than waveform output.
It is inexpensive enough to run whenever a backend is installed; otherwise it contributes no test cases.

## CI tier: `unit/` and `integration/`

These tests neither import nor evaluate a reference backend.
`unit/` covers the registry and `ripplegw.conversions`.
`integration/` is parametrized directly from `ripplegw.list_waveforms()`, so newly registered models are included automatically:

- `test_output_format.py` — output keys, shape, dtype, finiteness, `repr`, registry round-trip.
- `test_jax.py` — `jit` matches eager evaluation, `vmap` over a batch of parameters, `grad` is finite for every parameter.
- `test_amplitude_and_distance.py` — `AmplitudePhaseWaveform.strain == amplitude * exp(i * phase)`; `DistanceScaledWaveform.at_unit_distance` matches `d_L=1.0` exactly, and scales as `1/d_L`.
- `test_edge_cases.py` — equal mass, zero/near-extremal spin, face-on/edge-on, zero tidal deformability, aligned-spins-only and fully-precessing limits, burst shape-parameter boundaries — each parametrized by the relevant registry metadata (`is_tidal`, `is_precessing`, `parameter_names`), not by waveform name.

Parameter dictionaries come from `tests/helpers/params.py::canonical_params`, keyed by physical regime (BBH or BNS) rather than model name.
Evaluation grids come from `tests/helpers/grids.py::grid_for`.
Both helpers raise an informative error for an unrecognised parameter name.

A session-scoped cache in `tests/conftest.py` (`compiled_model`) JIT-compiles each `(waveform, config, method)` combination once and reuses it across test modules.
This keeps the higher-mode and precessing-model tests practical to run.

## Accuracy tier: `cross_validation/`

The accuracy tests are organised by validation method rather than source type:

- **`cross_validation/fd/`** — the `ReferenceBackend` campaign for frequency-domain, stateless-per-call models (`domain="FD"`).
  Every CBC model currently qualifies, but the campaign is selected by domain because that is its actual interface requirement.
  - `test_overlap.py` draws random parameter sets (`tests/helpers/params.py::random_params_batch`), evaluates ripple and the reference backend, and checks that the noise-weighted overlap loss remains below the configured threshold.
  - `test_phase_convention.py` uses a fixed configuration to detect a constant phase offset that an overlap alone would not detect
  Both use the Einstein Telescope D-design PSD (`tests/psds/ET_D_psd.txt`); the documented thresholds use this weighting.
- **`cross_validation/cw/`** — continuous-wave models.
  These are time-domain models with an ephemeris and epoch fixed at construction (the observing site's location is a per-call parameter), so they do not fit `ReferenceBackend`.
  Each registered class has a dedicated test that reconstructs LAL's reference calculation from SWIG-exposed building blocks, plus an independent end-to-end check (`test_makefakedata_v5.py`) against the `lalpulsar_Makefakedata_v5` engine for the classes it can validate; shared helpers live in `cw/_lal_helpers.py`.
  See [Reference Implementations](reference_implementations.md).

Thresholds live in `tests/cross_validation/tolerances.toml`: one `[<backend>.<waveform>]` block per model, with `[<backend>.defaults]` providing fallback values.
`cross_validation/test_tolerance_table.py` runs without a reference backend.
It checks that each supported waveform has an entry and that the overlap-loss values match [Reference Implementations](reference_implementations.md).

`tests/cross_validation/campaign.py` provides the frequency-domain campaign's batch generation, on-disk reference cache (`--cache-reference`), OOM retry handling, and JSON/figure output (`--plots`; matplotlib remains optional).
Results land under `--outdir` (default `accuracy-results/`, gitignored) as `n<N>_T<T>/<backend>_<waveform>.json`.

Run it locally:

```bash
uv sync --group test --group cross-validation
uv run pytest -m accuracy --reference lal --n-samples 1000 \
    --outdir accuracy-results --cache-reference --plots
```

On a Slurm cluster, run `bash tests/cross_validation/submit_slurm.sh`; on an HTCondor cluster, run `bash tests/cross_validation/submit_condor.sh`.
Each submits the campaign as a single GPU job.
`N_SAMPLES` and `OUTDIR` environment variables override the defaults.

### Adding a reference backend

The `ReferenceBackend` protocol supports additional frequency-domain reference implementations.
A backend is a class in `tests/cross_validation/reference/` implementing:

```python
class ReferenceBackend(Protocol):
    name: ClassVar[str]
    @classmethod
    def available(cls) -> bool: ...  # is the dependency importable?
    def supports(self, waveform: str) -> bool: ...
    def constants(self) -> dict[str, float]: ...
    def generate(self, waveform: str, params: dict, grid: Grid) -> dict: ...  # {"p", "c"}
```

`params` is always ripple's parameter dictionary, keyed by `parameter_names`.
Each backend translates it into the reference implementation's convention, leaving `fd/test_overlap.py` and `fd/test_phase_convention.py` independent of those details.
Register with `@register_backend`, add a `[<name>.<waveform>]` block per supported model to `tolerances.toml`, and select it with `--reference <name>`.
`supports()` lets a backend declare the subset of models it can generate.
This protocol applies only to frequency-domain, stateless-per-call models.
A family whose calling convention does not fit `ReferenceBackend.generate(name, params, grid)`, such as CW, needs its own validation method.

## Adding a waveform

If a new model uses only existing parameter names, it is included automatically: `integration/` uses `ripplegw.list_waveforms()`, and `fd/test_overlap.py` uses `ripplegw.list_waveforms(domain="FD")` when the model is frequency-domain and supported by a reference backend.
For a new parameter name, add a default to `tests/helpers/params.py` and, when a reference backend supports the model, a tolerance row in `tolerances.toml`.
See step 13 of [Adding a Waveform](adding_a_waveform.md) for the full checklist.
