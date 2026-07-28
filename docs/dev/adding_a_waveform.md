# Adding a Waveform

Read [Architecture](architecture.md) first. This page explains what a public waveform class must provide, how it is registered, and how it is validated; use an existing nearby model as the implementation reference.

## Set up

```bash
uv sync --group test --group doc
uv run pre-commit install
```

If the proposed model changes ripple's scope, discuss it first as described in [Contributing](../contributing.md).

## Place and register the model

Put an in-tree model under `src/ripplegw/waveforms/`:

- Extend an existing CBC baseline in that baseline's subpackage.
- Put a new source family in a new top-level subpackage such as `waveforms/<source_type>/`.

Importing `ripplegw` discovers non-underscore modules below `waveforms/` and registers their decorated classes.
Do not add a central import list.
The first subpackage after `waveforms` becomes the default `source_type` metadata.

Register the concrete class with its public name and deliberate metadata:

```python
@register("NewWaveform", is_tidal=False, is_precessing=False)
class NewWaveform(...):
    ...
```

The registry name is the string accepted by `ripplegw.waveform(...)`.
`domain` comes from the base class, not from `@register`.
Metadata is public: `list_waveforms(**filters)` exposes it, so add tags only when they describe a useful, stable property.

## Choose the interface

| Model property | Base class |
| --- | --- |
| Frequency-domain waveform | `FrequencyDomainWaveform` |
| Frequency-domain waveform with a meaningful amplitude/phase split | `AmplitudePhaseWaveform` instead of `FrequencyDomainWaveform` |
| Time-domain waveform | `TimeDomainWaveform` |
| Has a `d_L` parameter | Also inherit `DistanceScaledWaveform` |

Use `DistanceScaledWaveform` with a domain base, with the domain base first:

```python
class NewWaveform(AmplitudePhaseWaveform, DistanceScaledWaveform):
    ...
```

`AmplitudePhaseWaveform` requires `amplitude()` and `phase()` in addition to `__call__`; its `strain()` method builds their complex product.

## Implement the public interface

Configuration is fixed at construction; source parameters are supplied on each call in `params`.
Do not put per-call parameters in `__init__`.

Implement:

- `parameter_names` as a property returning the parameter names in the order your implementation packs or consumes them.
  Reuse established names for the same physical quantity.
- `__call__(axis, params)`, returning a dictionary of polarization arrays. Built-in models conventionally return `{"p": plus, "c": cross}`. Frequency domain arrays are complex; time domain arrays are real; each matches `axis`.
- Any public evaluation method so it works under `jax.jit`, `jax.vmap`, and `jax.grad`.
  Use JAX control flow when a branch depends on an array value.

Time-domain models should return a time series.
Do not add an FFT merely to fit a frequency-domain interface or test.

[`SineGaussian.py`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/burst/SineGaussian.py) is the smallest complete in-tree example.
For a configurable, amplitude/phase, distance-scaled model, read [`IMRPhenomD.py`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/cbc/IMRPhenomD/IMRPhenomD.py).

## Add the right tests

Registration automatically gives a model the baseline integration coverage in [Test Coverage](test_coverage.md).
Make that generic coverage constructible:

- Add a canonical value for every new parameter name in `tests/helpers/params.py`.
- Add a minimal constructor configuration in `tests/helpers/config.py` if the model cannot be constructed without arguments.
- Add focused integration tests for model-specific configuration or physics that the registry-driven tests cannot exercise.

Reference comparison coverage is separate.
Before calling a new model cross-validated, add a waveform-appropriate reference comparison, its threshold, and a large-scale launcher adapter.
Frequency-domain models can use the LAL overlap path when it supports them; time-domain models need directly aligned samples and the relative-norm amplitude diagnostic.
Record the method and threshold in [Reference Comparisons and Limits](reference_comparisons_and_limits.md), not results from a particular run.

## Verify before a PR

```bash
uv run pytest -m "not accuracy"
uv run pre-commit run --all-files
```

Run the selected reference sweep when the model or its adapter changes; see [Run Reference Checks](run_reference_checks.md).
Update the README only for a new user-visible capability.
Timing support is usually automatic for CBC models; inspect `src/ripplegw/benchmarks/timings/timing.py` when the parameter set is unusual.
Changes needed by Jim belong in its separate repository and pull request.
