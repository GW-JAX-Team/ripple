# Testing

This page is for developers changing ripple.
The fast Tier 1 suite is run by CI on each pull request; use it as the interface and JAX-regression safety net, rather than as a manual release procedure.

## What the suite covers

| Layer | Purpose | Coverage |
| --- | --- | --- |
| Unit tests | Registry and conversion utilities | Focused, fast checks |
| Integration tests | Public waveform behaviour | Every registered waveform: output contract, JAX transformations, amplitude/phase and distance relations, and physical edge cases |
| Cross-validation | Agreement with a reference calculation | A selected waveform at a useful sample scale; launched locally or on HPC |

Integration parametrization comes from `ripplegw.list_waveforms()`, so registering a waveform adds it to the fast coverage automatically.
If it introduces a new parameter name, add a suitable default in `tests/helpers/params.py`.

## Cross-validation

Cross-validation is the developer-owned accuracy check.
It deliberately has one launcher for every supported waveform and scheduler; see [Cross-validation tests](cross_validation.md) for the command, HPC setup, output handling, and plotting.

The comparison follows the waveform's domain:

- Frequency-domain models use a reference backend on a shared frequency grid, with the ET-D PSD-weighted overlap loss and the tolerance table in `tests/cross_validation/tolerances.toml`.
- Time-domain models such as CW and SineGaussian use waveform-specific references on aligned samples.
  They use a white, normalized time-domain mismatch and a relative-norm amplitude diagnostic; no FFT is used.

The documented thresholds, known limits in the reference code, and CW validation provenance are kept in [Reference Implementations](reference_implementations.md).

## When adding a waveform

Fast integration coverage is automatic after registration.
Reference validation is not: add a supported reference adapter and tolerance for a frequency-domain model, or a waveform-specific direct time-domain reference comparison plus amplitude diagnostic for a time-domain model.
Do not convert a time-domain waveform with an FFT merely to reuse the frequency-domain test setup.

See [Adding a Waveform](adding_a_waveform.md) for the complete contribution checklist.
