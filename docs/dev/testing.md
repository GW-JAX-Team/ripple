# Testing

ripple's tests run in two different places with two different goals.
CI runs on every pull request and needs to be fast: it should prove the package still works, not exhaustively re-verify waveform accuracy.
An HPC run, typically on a single high-end GPU, does the opposite: it exists specifically to check waveform accuracy against a CPU-based reference implementation, at a scale CI cannot afford.

The suite is organised around that split.
Two pytest markers, not directory paths, decide what runs where.

## The two tiers

| Marker | What it checks | Where it runs |
| --- | --- | --- |
| *(none)* | Every registered waveform's output format, `jit`/`vmap`/`grad`, amplitude/phase + distance, and edge cases | Every PR, every supported Python version |
| `accuracy` | ripple's output against a reference backend (LAL today) | The `smoke` subset on PRs/pushes to `main`; the full campaign on HPC |

```bash
uv run pytest -m "not accuracy"                                       # what CI runs
uv run pytest -m "accuracy and smoke" --reference lal --n-samples 3   # CI's accuracy check
uv run pytest -m accuracy --reference lal --n-samples 1000     # the real campaign
```

`cross_validation/test_reference_constants.py` sits outside this table on purpose.
It compares numeric literals in `ripplegw.constants` against a reference backend, not waveform output, so it is cheap enough to run unconditionally wherever a backend happens to be installed -- it lives under `cross_validation/` because it is the one CI-tier test that touches a reference backend at all, and every file that does so is kept there.
It contributes zero test cases, not a failure, when no backend is installed.

## CI tier: `unit/` and `integration/`

These never import a reference backend and never evaluate against one -- LAL/lalsimulation usage anywhere in the suite is confined to `tests/cross_validation/` (the `reference/` subpackage, its two consumers below, and `test_cw_exact_pulsar.py`, which imports `lalpulsar` directly since CW's calling convention doesn't fit the `ReferenceBackend` protocol -- see [LAL Agreement](lal_agreement.md)), so `unit/` and `integration/` work with no CPU-based reference package installed at all.
`unit/` covers the registry and `ripplegw.conversions`.
`integration/` is parametrized directly off `ripplegw.list_waveforms()`, so a newly registered model is covered the moment it appears, with no test-file edits:

- `test_output_format.py` — output keys, shape, dtype, finiteness, `repr`, registry round-trip.
- `test_transforms.py` — `jit` matches eager evaluation, `vmap` over a batch of parameters, `grad` is finite for every parameter.
- `test_amplitude_and_distance.py` — `AmplitudePhaseWaveform.strain == amplitude * exp(i * phase)`; `DistanceScaledWaveform.at_unit_distance` matches `d_L=1.0` exactly, and scales as `1/d_L`.
- `test_edge_cases.py` — equal mass, zero/near-extremal spin, face-on/edge-on, zero tidal deformability, aligned-spins-only and fully-precessing limits, burst shape-parameter boundaries — each parametrized by the relevant registry metadata (`is_tidal`, `is_precessing`, `parameter_names`), not by waveform name.

Parameter dicts come from `tests/helpers/params.py::canonical_params`, keyed by physical regime (BBH vs BNS) rather than by model name; evaluation grids come from `tests/helpers/grids.py::grid_for`.
Both raise immediately on a parameter name they don't recognise, naming themselves as the place to extend.

A session-scoped cache in `tests/conftest.py` (`compiled_model`) jit-compiles each `(waveform, config, method)` combination once and reuses it across every test module — load-bearing for the higher-mode and precessing models, whose eager evaluation is dispatch-bound and tens of times slower than the compiled call.

## Accuracy tier: `cross_validation/`

`test_overlap.py` draws random parameter sets (`tests/helpers/params.py::random_params_batch`), generates both sides — ripple via a batched `vmap` call (falling back to `jax.lax.map`, then a sequential loop, on GPU OOM) and the reference backend via a thread pool — and asserts the noise-weighted overlap loss stays under a threshold.
`test_phase_convention.py` checks a single fixed configuration to catch a constant phase offset that the overlap test alone cannot see.
Both use the Einstein Telescope D-design PSD (`tests/psds/ET_D_psd.txt`) for the noise weighting; the documented thresholds are calibrated against that specific weighting, not a flat spectrum.

Thresholds live in `tests/cross_validation/tolerances.toml`, one `[<backend>.<waveform>]` block per model, falling back to `[<backend>.defaults]` per key.
`cross_validation/test_tolerance_table.py` checks two things without needing a reference backend installed: every waveform a backend claims to support has an entry, and the overlap-loss column matches the table in [LAL Agreement](lal_agreement.md).

`tests/cross_validation/campaign.py` holds everything that is not the assertion itself: batch generation, the on-disk reference cache (`--cache-reference`), the OOM-retry ladder, and JSON/figure output (`--plots`, matplotlib-gated so it is never a hard dependency).
Results land under `--outdir` (default `accuracy-results/`, gitignored) as `n<N>_T<T>/<backend>_<waveform>.json`.

Run it locally:

```bash
uv sync --group test --group cross-validation
uv run pytest -m accuracy --reference lal --n-samples 1000 \
    --outdir accuracy-results --cache-reference --plots
```

On a cluster, `bash tests/cross_validation/submit_slurm.sh` or `bash tests/cross_validation/submit_condor.sh` submit the same campaign as a single GPU job (`N_SAMPLES`/`OUTDIR` environment variables override the defaults).

### Adding a reference backend

The comment in the issue that started this — "there is only LAL, but later on we may need other CPU-based comparisons for new waveform families" — is why this is an extension point, not a hardcoded LAL call.
A backend is a class in `tests/cross_validation/reference/` implementing the `ReferenceBackend` protocol:

```python
class ReferenceBackend(Protocol):
    name: ClassVar[str]
    @classmethod
    def available(cls) -> bool: ...             # is the dependency importable?
    def supports(self, waveform: str) -> bool: ...
    def constants(self) -> dict[str, float]: ...
    def generate(self, waveform: str, params: dict, grid: Grid) -> dict: ...  # {"p", "c"}
```

`params` is always ripple's own dict, keyed by `parameter_names` — the backend owns translating that into its own convention, so `test_overlap.py` and `test_phase_convention.py` never need to know it exists.
Register with `@register_backend`, add a `[<name>.<waveform>]` block per supported model to `tolerances.toml`, and select it with `--reference <name>`.
`supports()` gates which models a partial backend is asked to generate, so covering only some models is fine.

## Adding a waveform

If the new model introduces no parameter name the suite doesn't already know, it is covered automatically: `integration/` and `test_overlap.py` both parametrize off `ripplegw.list_waveforms()`.
If it does introduce a new name, add a default to `tests/helpers/params.py` and, if a reference backend supports the model, a tolerance row in `tolerances.toml`.
See step 13 of [Adding a Waveform](adding_a_waveform.md) for the full checklist.
