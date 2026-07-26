# Architecture

This page explains why ripple is structured the way it is, so you can extend or debug it without fighting the framework.
For the practical steps, see [Adding a Waveform](adding_a_waveform.md).

## Design goals

- **Differentiable, JIT-compilable, vectorisable.**
  Every waveform is a pure function of `(axis, params)` with no hidden state, so it composes with `jax.grad`/`jax.jit`/`jax.vmap` unmodified.
- **No FFTs.**
  A frequency-domain model builds its strain analytically as $A(f) e^{i\psi(f)}$; a time-domain model returns a real time series.
  Converting between domains, if ever needed, is the caller's job — never add an FFT to `src/`.
- **No detector response, no PSDs, no likelihoods.**
  ripple's scope stops at generating polarizations.
  Detector projection, noise weighting, and inference are the caller's job — e.g. [Jim](https://github.com/GW-JAX-Team/jim), or any other downstream pipeline.
- **One construction path.**
  Concrete waveform classes are deliberately not exported — `ripplegw.waveform(name, **config)` is the only way to build one, so there's exactly one thing to keep stable across releases.

Two modules are usable but not re-exported at the top level — import them explicitly: `ripplegw.conversions` (mass and tidal-parameter conversions) and `ripplegw.constants` (physical constants).

## The registry

Waveform classes self-register with `@register` when their module is imported — there's no central list mapping names to classes to maintain.
Metadata like `domain` and `source_type` (the GW source category — `cbc`, `burst`, `cw`, ...) is inferred from the class itself — its base class, its location on disk — rather than declared, so it can't drift from what the class actually is.
See [Adding a Waveform](adding_a_waveform.md) for the registration and discovery mechanics.

## The class hierarchy

```
Waveform (ABC)                        parameter_names, __call__
├── FrequencyDomainWaveform           domain = "FD"
│   └── AmplitudePhaseWaveform (ABC)  + amplitude(), phase(), concrete strain()
├── TimeDomainWaveform                domain = "TD"
└── DistanceScaledWaveform            mixin (not a Waveform subclass); concrete at_unit_distance()
```

- **`Waveform`** — every model implements `parameter_names` (the ordered `params` keys it needs) and `__call__(axis, params)` (returns `{"p": ..., "c": ...}`, or model-specific keys).
- **`AmplitudePhaseWaveform`** — for single-mode, aligned-spin models, where amplitude and phase as functions of frequency are individually well-defined.
  Multimode and precessing models can't use it: `|hp|` beats between modes, so there's no single well-defined amplitude.
- **`DistanceScaledWaveform`** — a mixin, not a `Waveform` on its own.
  It must be combined with a domain base, or the registry rejects the class.

## Choosing where shared code lives

A helper belongs in `src/ripplegw/utils/` only if it's used by more than one waveform subpackage and contains no family-specific physics; otherwise it stays local to the one subpackage that uses it.
Check `utils/` before reimplementing something — shared spherical-harmonic and tidal-conversion helpers already live there.

## Non-goals

Detector response and antenna patterns, PSDs and noise weighting, parameter estimation and priors, and FFT-based domain conversion are all out of scope for ripple — they belong in the caller's own code, whether that's [Jim](https://github.com/GW-JAX-Team/jim) or another downstream pipeline.
