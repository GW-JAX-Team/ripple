# Test Coverage

ripple has two complementary test layers:

- The fast suite checks the registered waveform interface and JAX behaviour.
- Reference comparisons check numerical agreement against an external implementation.

Registration adds a waveform to the first layer automatically.
It does not by itself establish reference agreement.

## Run the fast suite

```bash
uv sync --group test
uv run pytest -m "not accuracy"
```

CI runs the non-accuracy suite on its PR and push workflows.
Relevant Python 3.12 main workflows also run a LAL accuracy smoke test over every frequency-domain waveform at three parameter draws each; it is not a replacement for a selected reference sweep.

## Coverage added by registration

The integration tests parametrize from `ripplegw.list_waveforms()` at test collection time.
Every registered waveform therefore receives the following baseline coverage.

| Coverage | What is checked |
| --- | --- |
| Waveform interface | Registry construction, `parameter_names`, `repr`, `{"p", "c"}` output, shape, dtype, and finite values. |
| JAX | Eager and `jax.jit` agreement, parameter batching with `jax.vmap`, and finite gradients. |
| Optional interfaces | `AmplitudePhaseWaveform` methods and `DistanceScaledWaveform` distance scaling, when implemented. |
| Declared physical features | Edge cases selected by metadata or parameters: aligned/precessing spins, tidal deformability, inclination, and time-domain shape parameters. |

For precessing and tidal models the eager/`jit` and `vmap` checks are marked `slow`: they run on `main`-targeted CI only, where the accuracy job already compares those paths against LAL.
Finite-gradient checks run on every build.

The three built-in continuous-wave models also have dedicated integration tests for their non-default configurations and batches over distinct detector sites.
Focused unit tests cover utilities and registry behaviour; they are not a per-waveform accuracy claim.

## Current reference-validation map

All models below receive the fast baseline above.
The `accuracy` tests add the reference checks in this table.
The launcher column is the one large-scale test selected by [Run Reference Checks](run_reference_checks.md); it does not also run every focused reference regression.

| Waveform(s) | Reference checks | Launcher sweep |
| --- | --- | --- |
| `TaylorF2`, `IMRPhenomD`, `IMRPhenomD_NRTidalv2`, `IMRPhenomHM`, `IMRPhenomXAS`, `IMRPhenomXAS_NRTidalv3`, `IMRPhenomXHM` | LALSuite frequency-domain overlap and absolute phase convention | The same LAL overlap and phase tests over the selected sample set |
| `IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomXP_NRTidalv3`, `IMRPhenomXPHM` | LALSuite frequency-domain overlap | The LAL overlap test over the selected sample set |
| `SineGaussian` | LALSimulation on aligned time samples | The direct time-domain sweep |
| `ExactPulsarSignal` | LALPulsar building-block reconstruction and barycenter check | A randomized LALPulsar building-block sweep |
| `PulsarSignal` | Per-sample LALPulsar barycenter check and `CWMakeFakeData` regression | A randomized `CWMakeFakeData` sweep |
| `BinaryPulsarSignal` | LALPulsar orbital-phase check and `CWMakeFakeData` regression | A randomized `CWMakeFakeData` sweep |

Frequency-domain comparisons use the ET-D PSD-weighted overlap; non-precessing models also receive the absolute phase check.
Large-scale time-domain comparisons use aligned, directly generated samples, a normalized mismatch, and a relative-norm check—never an FFT added solely for testing.
The acceptance thresholds and reference-specific limits are in [Reference Comparisons and Limits](reference_comparisons_and_limits.md).

## Inspect one waveform's sweep

Preview the exact launcher target without running it:

```bash
uv run python -m tests.cross_validation.submit \
    --scheduler local --waveform IMRPhenomD --n-samples 10 \
    --outdir accuracy-results/preflight --dry-run
```

Replace the name with any registered waveform.
`--waveform all --dry-run` prints every available target and explicitly lists registered waveforms without a large-scale adapter.

## When tests need work

- A change to any registered waveform should pass the fast suite.
- A numerical or reference-adapter change should also run that waveform's launcher sweep.
- A new parameter name needs a canonical value in `tests/helpers/params.py`; a waveform with required constructor arguments needs a test configuration in `tests/helpers/config.py`.
- A new waveform needs a focused reference comparison and a launcher adapter before it is described as cross-validated.
  See [Adding a Waveform](adding_a_waveform.md).
