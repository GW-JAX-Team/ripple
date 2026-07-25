# Adding a Waveform

This page walks through implementing a new waveform model from scratch, end to end.
Read [Architecture](architecture.md) first if you haven't — this page assumes you know what the registry and the `Waveform` hierarchy are for.

Every code snippet below is taken from a real file in the repository, named where it comes from, so you can open the source alongside this page.

## 1. Before you start

```bash
uv sync --group test --group doc
uv run pre-commit install
```

If you're unsure whether a feature fits ripple's scope, open an issue first — see [Contributing](../contributing.md) for the three principles new features are expected to follow (JIT-friendliness, modular implementation, machine-precision agreement with any LAL counterpart or a written explanation of the discrepancy).

## 2. Where it lives

New waveforms are added in-tree: the model lives in `src/ripplegw/waveforms/`, sharing the
`Waveform` + `@register` contract described below.

## 3. Choose the subpackage

Subpackages under `src/ripplegw/waveforms/` are split by **lineage, not class name** — `IMRPhenomHM`/`IMRPhenomPv2` live under `IMRPhenomD/` because they build on the D baseline.
The existing subpackages:

| Subpackage | Contains |
| --- | --- |
| `IMRPhenomD/` | `IMRPhenomD`, `IMRPhenomPv2`, `IMRPhenomHM` |
| `IMRPhenomX/` | `IMRPhenomXAS`, `IMRPhenomXHM`, `IMRPhenomXP`, `IMRPhenomXPHM` |
| `IMRPhenom_NRTidal/` | `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3` |
| `TaylorF2/` | `TaylorF2` |
| `burst/` | `SineGaussian` |

If your model builds on an existing baseline, add it to that subpackage.
If it's a genuinely new family, create a new subpackage — it needs nothing but a one-line docstring in its `__init__.py`:

```python
"""One-line description of the family."""
```

No imports, no `__all__`.
Auto-discovery (below) finds it on its own.

## 4. The three-part module shape

Every model follows the same shape.
The complete reference is [`src/ripplegw/waveforms/burst/SineGaussian.py`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/burst/SineGaussian.py) (129 lines — read the whole file, it's the simplest complete example in the repo):

**(a) A module-level generator function taking a positional, packed `theta` array** — this is the differentiable numerical core:

```python
def gen_SineGaussian_hphc(
    t: Float[Array, " n_time"],
    theta: Float[Array, "5"],
) -> tuple[Float[Array, " n_time"], Float[Array, " n_time"]]:
    quality, frequency, hrss, phase, eccentricity = theta
    ...
    return plus, cross
```

Taking one packed array rather than separate keyword arguments keeps this function directly `jit`/`grad`/`vmap`-friendly and easy to compare line-by-line against a LAL/lalinference source if you're porting one.

**(b) A module-level `_split_params(params)` helper** converting the `params` mapping into that packed array.
This is per-module — there's no shared framework helper.
`SineGaussian` folds it directly into `__call__` since it's a single 5-element pack with no config-dependent shape; when there's real work to share between `amplitude()`/`phase()`/`__call__()` (see step 9), it's worth its own function — the pattern from [`IMRPhenomD.py:739-750`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/IMRPhenomD/IMRPhenomD.py#L739-L750):

```python
def _split_params(
    params: Mapping[str, Any],
) -> tuple[Float[Array, "4"], Float[Array, "3"]]:
    m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
    theta_intrinsic = jnp.array([m1, m2, params["s1_z"], params["s2_z"]])
    theta_extrinsic = jnp.array([params["d_L"], 0.0, params["phase_c"]])
    return theta_intrinsic, theta_extrinsic
```

**(c) A thin `@register`-decorated class** that packs `params` into `theta`, calls the generator, and returns `{"p": ..., "c": ...}`:

```python
@register("SineGaussian", is_tidal=False, is_precessing=False)
class SineGaussian(TimeDomainWaveform):
    """Sine-Gaussian time-domain burst waveform."""

    def __init__(self) -> None:
        pass

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("Q", "f_0", "hrss", "phase", "e")

    def __call__(
        self, t: Float[Array, " n_time"], params: Mapping[str, Any]
    ) -> dict[str, Float[Array, " n_time"]]:
        theta = jnp.array(
            [params["Q"], params["f_0"], params["hrss"], params["phase"], params["e"]]
        )
        hp, hc = gen_SineGaussian_hphc(t, theta)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return "SineGaussian()"
```

!!! note "The core contract"
    **Configuration** lives on `self`, set once at construction (`self.f_ref`, or nothing at all for `SineGaussian`).
    **Physics parameters** arrive fresh on every call, in the `params` mapping.
    Never read configuration out of `params`, and never accept physics parameters through `__init__`.

## 5. Choose base classes

| Question | Base class |
| --- | --- |
| Is `axis` a frequency array? | `FrequencyDomainWaveform` |
| Is `axis` a time array? | `TimeDomainWaveform` |
| FD **and** single-mode aligned-spin — is a single $A(f)$, $\psi(f)$ well-defined? | `AmplitudePhaseWaveform` **instead of** `FrequencyDomainWaveform` (it already subclasses it — never list both) |
| Multiple modes mixed into the polarizations (higher harmonics, precession)? | plain `FrequencyDomainWaveform` — `\|hp\|` beats between modes, so there's no single amplitude |
| Does `params` include `d_L`? | additionally inherit `DistanceScaledWaveform` |

Two rules that are easy to get wrong:

- `DistanceScaledWaveform` is a **bare mixin, not a `Waveform` subclass**.
  Inheriting only it is rejected by the registry with `TypeError` — always combine it with a domain base.
  All 10 built-in models that have it put the domain base first and the mixin last: `class IMRPhenomD(AmplitudePhaseWaveform, DistanceScaledWaveform):`.
- `domain` metadata comes from the base class's `ClassVar`, **not** from `@register` — never pass `domain=` to the decorator.

See the [Waveform Catalogue](../guides/catalogue.md) for how all 11 built-in models are classified — find the closest precedent to your model.

## 6. `parameter_names`

A `@property`, not a method — `wf.parameter_names`, no parentheses.
The order must match how `__call__` packs `theta`.
It can depend on construction-time configuration; from [`TaylorF2.py`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/TaylorF2/TaylorF2.py):

```python
@property
def parameter_names(self) -> tuple[str, ...]:
    return (
        "M_c", "eta", "s1_z", "s2_z",
        *(
            ("lambda_tilde", "delta_lambda_tilde")
            if self.use_lambda_tildes
            else ("lambda_1", "lambda_2")
        ),
        "d_L", "phase_c", "iota",
    )
```

Reuse the established names (`M_c`, `eta`, `s1_z`, `d_L`, `phase_c`, `iota`, ...) whenever the underlying physical quantity is the same as an existing model's — a new spelling for something that already has a name breaks every downstream consumer that switches models by name.

## 7. `@register` and metadata

```python
@register("NewWaveform", is_tidal=False, is_precessing=False)
```

The first argument is the registry key — the exact string users pass to `ripplegw.waveform(...)` (defaults to the class name if omitted).
Keyword arguments become `waveform_metadata`, which `list_waveforms(**filters)` filters on and `get_waveform_metadata(name)` returns.
`is_tidal` and `is_precessing` are the two conventional CBC tags; add your own free-form keys if useful, but each one is a new public filter, so pick deliberately.
Registering a name that already exists raises `ValueError` unless you pass `override=True`.

## 8. Auto-discovery

`waveforms/__init__.py` imports every module and subpackage under `waveforms/` whose dotted path has no leading-underscore component, at `import ripplegw`.
Practical consequences:

- You never edit a central list — adding the file is the whole registration step.
- A `_`-prefixed module or subpackage name is the escape hatch if you have a helper you don't want auto-imported.
- This cost is paid by every user of the package at import time.
  Keep it cheap — defer heavy data loading or optional third-party imports to `__init__`/first use.

## 9. The return contract

`__call__` always returns a `dict`.
The convention is `{"p": ..., "c": ...}` (plus/cross), though the base class permits any keys a model needs.
Frequency-domain models return **complex** arrays; time-domain models return **real** arrays.
Output length matches the input axis.
ripple's `src/` never performs an FFT — if your model is naturally time-domain, return a time series; don't convert to frequency domain internally.

If your model qualifies for `AmplitudePhaseWaveform` (step 5), implement `amplitude(f, params)` and `phase(f, params)` such that `amplitude(f, p) * exp(1j * phase(f, p))` reproduces the pre-polarization strain `h0` — `strain()` (concrete on the base class) computes exactly that product for you.
`__call__` then applies whatever inclination-dependent plus/cross prefactor your model needs on top of `strain()`, the way [`IMRPhenomD.py:782-825`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/IMRPhenomD/IMRPhenomD.py#L782-L825) does.

## 10. Numerical requirements

- **JIT-friendly**: no data-dependent Python branching on traced values — use `jnp.where`, `lax.cond`, or `lax.select` instead of `if traced_value > 0:`.
- **float64-aware**: assume callers enable `jax.config.update("jax_enable_x64", True)` (see [JAX Transformations](../guides/jax.md)); don't rely on float32 rounding.
- **LAL agreement**: if you're porting an existing approximant, it should match its reference to machine precision, or the discrepancy needs a written explanation — see [LAL Agreement](lal_agreement.md) for the format existing entries use.

## 11. Worked example

Read [`SineGaussian.py`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/burst/SineGaussian.py) top to bottom for the complete minimal shape (step 4).
Read [`IMRPhenomD.py:739-828`](https://github.com/GW-JAX-Team/ripple/blob/main/src/ripplegw/waveforms/IMRPhenomD/IMRPhenomD.py#L739-L828) for the same shape extended with `f_ref` configuration, `amplitude`/`phase`, and the distance mixin.

A skeleton for a new frequency-domain, single-mode, aligned-spin model with a distance parameter, combining both:

```python
from typing import Any, Mapping

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.interfaces import AmplitudePhaseWaveform, DistanceScaledWaveform
from ripplegw.registry import register


def gen_NewWaveform_hphc(f, theta, f_ref):
    ...  # your physics, following the existing generator functions' style
    return hp, hc


def _split_params(params: Mapping[str, Any]) -> tuple[Float[Array, "N"], Float[Array, "M"]]:
    m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
    theta_intrinsic = jnp.array([m1, m2, params["s1_z"], params["s2_z"]])
    theta_extrinsic = jnp.array([params["d_L"], params["phase_c"]])
    return theta_intrinsic, theta_extrinsic


@register("NewWaveform", is_tidal=False, is_precessing=False)
class NewWaveform(AmplitudePhaseWaveform, DistanceScaledWaveform):
    def __init__(self, f_ref: float = 20.0) -> None:
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("M_c", "eta", "s1_z", "s2_z", "d_L", "phase_c", "iota")

    def amplitude(self, frequency, params):
        ...

    def phase(self, frequency, params):
        ...

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, Any]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        theta = jnp.array([
            params["M_c"], params["eta"], params["s1_z"], params["s2_z"],
            params["d_L"], params["phase_c"], params["iota"],
        ])
        hp, hc = gen_NewWaveform_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"NewWaveform(f_ref={self.f_ref})"
```

## 12. Validate locally

```python
import jax
import jax.numpy as jnp
import ripplegw

jax.config.update("jax_enable_x64", True)

assert "NewWaveform" in ripplegw.list_waveforms()
wf = ripplegw.waveform("NewWaveform", f_ref=20.0)
print(wf.parameter_names)

f = jnp.arange(20.0, 1024.0, 0.25)
params = {...}  # one value per name in wf.parameter_names
h = wf(f, params)
print(h["p"].dtype, h["p"].shape)   # complex for FD, real for TD; shape == f.shape

jax.jit(wf)(f, params)
jax.grad(lambda M_c: jnp.sum(jnp.abs(wf(f, {**params, "M_c": M_c})["p"]) ** 2))(params["M_c"])
jax.vmap(lambda M_c: wf(f, {**params, "M_c": M_c})["p"])(jnp.array([params["M_c"]] * 4))

# If you added AmplitudePhaseWaveform / DistanceScaledWaveform:
assert jnp.allclose(wf.amplitude(f, params) * jnp.exp(1j * wf.phase(f, params)), wf.strain(f, params))
assert jnp.allclose(wf.at_unit_distance(f, params)["p"] / params["d_L"], h["p"], rtol=1e-6)
```

Then: `uv run ruff check src/`, `uv run pyright`, `uv run pre-commit run --all-files`.

## 13. What else to update

- **`README.md`** — its "Supported waveforms" list is a curated set of highlights, not exhaustive; update it only if your model represents a genuinely new capability.
  `docs/index.md` links to the [catalogue](../guides/catalogue.md) instead of listing models, so it needs no edit.
- **`src/ripplegw/benchmarks/timings/timing.py`** — up to **three** separate lists, depending on your model: `choices=[...]` in `main()`'s argument parser (required for every model), `bns_waveforms` in `get_waveform_type()` if it's a tidal model, `precessing_waveforms` in `run_timing()` if it takes in-plane spins.
  You may also need a new `_prepare_*_params` builder if your model's parameter set doesn't match an existing one.
- **`timings/submit_slurm.sh`** and **`timings/submit_condor.sh`** — the `MODELS=(...)` array in each.
- **[LAL Agreement](lal_agreement.md)** — add a row if you cross-validated against LALSuite; keep it in sync with `tests/cross_validation/tolerances.toml` (`unit/test_tolerance_table.py` checks the two match).
- **Jim** (separate repository, separate PR) — `src/jimgw/core/single_event/waveform.py`, `src/jimgw/cli/_waveform.py`, `src/jimgw/cli/_config.py`.
  See that repo's own `CONTRIBUTING.md`.
- **Tests** — usually nothing: `integration/` and the accuracy campaign both parametrize off `ripplegw.list_waveforms()`, so a registered model is covered automatically.
  If your model introduces a parameter name the suite doesn't already know, add a default to `tests/helpers/params.py`; if a reference backend supports it, add the tolerance row mentioned above.
  See [Testing](testing.md).

## 14. Opening the PR

See [Contributing](../contributing.md) for the process.
Reviewers will check: the model registers and constructs (step 12), the base classes match the decision table (step 5), `parameter_names` matches the packing order in `__call__`, no FFT was introduced, and — if porting an existing approximant — the machine-precision (or documented-discrepancy) requirement is met.
