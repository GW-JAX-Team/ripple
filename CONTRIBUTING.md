# Contributing

Contributions of any kind are welcome and appreciated.
See the guidelines below.

## Expectations

ripple is developed and maintained by the GW JAX Team and community contributors.
While we try to be responsive, we don’t always get to every issue immediately.
If it has been more than a week or two, feel free to ping the maintainers on the issue.

## Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/GW-JAX-Team/ripple/issues).
If you’re unable to find an open issue addressing the problem, [open a new one](https://github.com/GW-JAX-Team/ripple/issues/new).
Be sure to include a **title and clear description**, as much relevant information as possible, and the simplest possible **code sample** demonstrating the expected behaviour that is not occurring.

## Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch.
Ensure the PR description clearly describes the problem and solution.
Include the relevant issue number if applicable.

## Do you intend to add a new feature or change an existing one?

Open a new GitHub pull request with the feature or change.
Please follow these principles:

1. New features should be able to take advantage of `jax.jit` wherever possible.
2. Modular implementation is preferred.
3. If a waveform is ported from an existing implementation, it should match the original to machine precision. If that is not achievable, a clear explanation of the discrepancy must be provided. New waveforms implemented directly in ripple are not subject to this requirement.

If you are unsure whether a feature fits, open an issue first to discuss it with the maintainers.

### Adding a new waveform

When adding a new waveform approximant `NewWaveform`, you need to update several files.
The canonical ordering for all waveform lists is the same as in `waveform_preset` in `interfaces.py` — insert your waveform in that position everywhere.

**In ripple (this repository):**

- `src/ripplegw/waveforms/NewWaveform.py` — implement the waveform and expose a `gen_NewWaveform(frequency_array, ...)` function.
- `src/ripplegw/interfaces.py` — add an import of `gen_NewWaveform`, define a `NewWaveform(Waveform)` class with `parameter_names`, `__call__`, and `__repr__`, and add it to the `waveform_preset` dict at the bottom.
- `src/ripplegw/__init__.py` — add `NewWaveform` to both the `from .interfaces import (...)` block and `__all__`.
- `README.md` and `docs/index.md` — add `NewWaveform` to the list of supported approximants.
- `src/ripplegw/benchmarks/timings/timing.py` — add `"NewWaveform"` to the models list.
- `timings/submit_condor.sh` and `timings/submit_slurm.sh` — add `"NewWaveform"` to the `MODELS` array.
- `tests/integration/test_waveforms.py` — add a `TestNewWaveform` class with at least `test_basic`, `test_jit`, `test_repr`, and `test_in_waveform_preset` tests. See existing classes for the pattern.
- `tests/cross_validation/test_lal_overlap.py` *(if a LAL reference implementation exists)* — add an entry to `OVERLAP_LOSS_THRESHOLDS` and a `pytest.param` to the parametrize list.
- `tests/utils.py` *(if a LAL reference implementation exists)* — add an `elif waveform_name == "NewWaveform":` branch in `get_jitted_waveform` and a corresponding branch in the LAL waveform generation helper.

**In [Jim](https://github.com/GW-JAX-Team/jim) (the inference framework):**

Jim wraps ripple waveforms under `Ripple`-prefixed aliases.
Once your ripple PR is merged (or during development against a local ripple install), open a separate pull request in the Jim repository with the following changes:

- `src/jimgw/core/single_event/waveform.py` — add `NewWaveform` to the `from ripplegw import (...)` block and add `RippleNewWaveform = NewWaveform` plus an entry in `__all__`.
- `src/jimgw/cli/_waveform.py` — import `RippleNewWaveform` and add `"NewWaveform": RippleNewWaveform` to `_REGISTRY`.
- `src/jimgw/cli/_config.py` — add `"NewWaveform"` to the `Approximant` `Literal` type so the CLI accepts it.

## Do you intend to add an example or tutorial?

Open a new GitHub pull request with the example or tutorial.
The example should be self-contained and keep imports from other packages to a minimum.
Leave case-specific analysis details out.
