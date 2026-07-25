# Architecture

This page explains *why* ripple is organised the way it is.
[Adding a Waveform](adding_a_waveform.md) tells you which keys to type; this page explains the reasoning behind them.

## Design goals

- **Differentiable, JIT-compilable, vectorisable.**
  Every waveform is a pure function of `(axis, params)` with no hidden state, so it composes with `jax.grad`/`jax.jit`/`jax.vmap` unmodified.
- **No FFTs.**
  A frequency-domain model builds its strain analytically as $A(f) e^{i\psi(f)}$; a time-domain model returns a real time series.
  Converting between domains, if ever needed, is the caller's job — never add an FFT to `src/`.
- **No detector response, no PSDs, no likelihoods.**
  ripple's scope stops at generating polarizations.
  Detector projection, noise weighting, and inference are [Jim](https://github.com/GW-JAX-Team/jim)'s job.
- **A single, minimal public surface.**
  Concrete waveform classes are deliberately not exported — `ripplegw.waveform(name, **config)` is the only construction path, so there is exactly one thing to keep stable across releases.

## The public surface

`ripplegw.__all__` is the entire public API — 11 names: the `Waveform` class hierarchy (`Waveform`, `FrequencyDomainWaveform`, `TimeDomainWaveform`, `AmplitudePhaseWaveform`, `DistanceScaledWaveform`), the registry functions (`waveform`, `list_waveforms`, `get_waveform_metadata`, `register`), `WAVEFORM_REGISTRY`, and `__version__`.

Two submodules are user-facing but **not** re-exported at the top level — import them explicitly: `ripplegw.conversions` (mass and tidal-parameter conversions) and `ripplegw.constants` (physical constants).
`ripplegw.typing` holds internal `jaxtyping` aliases.

## Source layout

```
src/ripplegw/
  __init__.py       public API surface (see above)
  interfaces.py     the Waveform class hierarchy — no concrete-family knowledge
  registry.py       WAVEFORM_REGISTRY, register(), waveform(), list_waveforms()
  constants.py      physical constants
  conversions.py    mass / tidal parameter conversions
  typing.py         internal jaxtyping aliases
  utils/            helpers shared by 2+ waveform subpackages (see "utils/ admission rule" below)
    spherical_harmonics.py
    tidal.py
  waveforms/
    __init__.py     recursive auto-discovery (see below) — never edited when adding a family
    CBC/            compact binary coalescence waveforms
      IMRPhenomD/     IMRPhenomD, IMRPhenomPv2, IMRPhenomHM, and their private helpers
      IMRPhenomX/     IMRPhenomXAS, IMRPhenomXHM, IMRPhenomXP, IMRPhenomXPHM, and their private helpers
      IMRPhenom_NRTidal/  IMRPhenomD_NRTidalv2, IMRPhenomXAS_NRTidalv3
      TaylorF2/       TaylorF2
    burst/          SineGaussian
```

Subpackages under `waveforms/CBC/` are split by **lineage, not class name** — `IMRPhenomHM` and `IMRPhenomPv2` live under `IMRPhenomD/` because they build on the D baseline.
Non-CBC source types (e.g. `burst/`) are direct children of `waveforms/`, not `CBC/`, since they don't share the CBC parameterisation.
This isn't just a directory convention: `register()` infers `source_type` from `cls.__module__` (a class under `ripplegw.waveforms.CBC.*` gets `source_type="CBC"`, `ripplegw.waveforms.burst.*` gets `"burst"`, and so on) — the same pattern as `domain` (see below), so it's checkable at runtime via `get_waveform_metadata(name)["source_type"]` without every family needing to state it.
Every subpackage `__init__.py` is a bare one-line docstring: zero imports, zero `__all__` — the subpackage doesn't need to know what's inside it.

## Registry mechanics

`register(name=None, *, override=False, **metadata)` is a class decorator.
It validates the class is a `Waveform` subclass, then merges `**metadata` onto a *copy* of whatever `waveform_metadata` the class already inherited from its base — which is how `domain` arrives without ever being passed to the decorator (see below) — and stores the result in `WAVEFORM_REGISTRY[name]`.
If `source_type` isn't in `**metadata`, it's filled in from `cls.__module__` — a class under `ripplegw.waveforms.<type>.*` gets `source_type=<type>` for free; classes registered from outside `ripplegw.waveforms` entirely (e.g. user code) get no inferred value unless passed explicitly.
Registering an existing name without `override=True` raises `ValueError`.

`waveform(name, /, **config)` looks the class up and calls `cls(**config)`.
`list_waveforms(**filters)` returns every name whose metadata matches all the given filters (a typo'd filter key just matches nothing — there's no validation).
`get_waveform_metadata(name)` returns a **copy** of a model's metadata dict, so callers can't mutate the registry by accident.

## Auto-discovery

`waveforms/__init__.py` walks every module and subpackage under `waveforms/` with `pkgutil.walk_packages(..., onerror=_reraise)` and imports each one whose dotted path has **no leading-underscore component**.
Importing a module is what triggers its `@register` decorator, so the act of importing *is* the registration step — there's no separate list to edit.

Two consequences worth internalising:

- A `_`-prefixed module or subpackage name is the escape hatch for something you don't want auto-imported.
- This work happens at `import ripplegw`, so it's paid by every user of the package.
  Keep import-time work cheap — defer heavy data loading or optional third-party imports to `__init__`/first use.
- `onerror=_reraise` is deliberate: `pkgutil.walk_packages`'s default behaviour is to silently drop a subpackage that fails to import.
  Re-raising means a broken family fails loudly at `import ripplegw` instead of just vanishing from `list_waveforms()`.

## The class hierarchy

```
Waveform (ABC)                        parameter_names (abstract property), __call__ (abstract)
├── FrequencyDomainWaveform           waveform_metadata = {"domain": "FD"}
│   └── AmplitudePhaseWaveform (ABC)  + abstract amplitude(), abstract phase(), concrete strain()
├── TimeDomainWaveform                waveform_metadata = {"domain": "TD"}
└── DistanceScaledWaveform            bare mixin (NOT a Waveform subclass); concrete at_unit_distance()
```

Each base makes one guarantee:

- **`Waveform`** — the universal contract: `parameter_names` names the ordered `params` keys the model needs, `__call__(axis, params)` returns `{"p": ..., "c": ...}` (or model-specific keys).
- **`FrequencyDomainWaveform` / `TimeDomainWaveform`** — fix what `domain` metadata the model gets, by inheritance, so `@register(...)` calls never pass `domain=` explicitly.
- **`AmplitudePhaseWaveform`** — for single-mode, aligned-spin models where amplitude and phase as functions of frequency are individually well-defined.
  `amplitude`/`phase` describe the pre-polarization strain `h0`; `__call__` applies the inclination-dependent plus/cross prefactors on top of `strain()` (which is `amplitude * exp(1j * phase)`, concrete on the base — no subclass implements it directly).
  Multimode and precessing models never inherit this: `|hp|` beats between modes there, so there is no single well-defined amplitude.
- **`DistanceScaledWaveform`** — a bare mixin, not a `Waveform` subclass.
  `at_unit_distance(axis, params)` is exact by construction (`== __call__(axis, {**params, "d_L": 1.0})`); the reverse relation (`__call__ == at_unit_distance / d_L`) holds only to floating-point precision, since it's a full re-evaluation, not a factored cheaper path.
  A class inheriting only this mixin is rejected by `registry._check_is_waveform` with `TypeError` — it must always be combined with a domain base.

## The `utils/` admission rule

A helper module moves to `src/ripplegw/utils/` only if it has consumers in **2+ waveform subpackages** and contains **no family-specific physics**; otherwise it stays local to the one subpackage that uses it.
Today that's exactly two modules: `spherical_harmonics.py` (used by the higher-mode families across both `IMRPhenomD/` and `IMRPhenomX/`) and `tidal.py` (used by `TaylorF2/`, `IMRPhenom_NRTidal/`, and `IMRPhenomD_NRTidalv2`'s tidal-amplitude helpers).
Check `utils/` before reimplementing something — these are exactly the kind of shared physics a new tidal or higher-mode model is likely to need.

## Why internals aren't in the Reference tab

The generated API reference only documents `ripplegw`'s five top-level modules — not the ~20 family and helper modules under `waveforms/` and `utils/`.
Those are implementation details a user should never need to import directly; the [Waveform Catalogue](../guides/catalogue.md) is the user-facing surface for what each model does and takes.
If you're writing a new family, you'll read those modules as source, not as rendered documentation.

## Non-goals

Detector response and antenna patterns, PSDs and noise weighting, parameter estimation and priors, and FFT-based domain conversion are all explicitly out of scope for ripple — they belong in [Jim](https://github.com/GW-JAX-Team/jim) or the caller's own code.
