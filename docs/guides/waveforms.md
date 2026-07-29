# Working with Waveforms

ripple provides a set of gravitational-wave waveform models, each identified by a registered name and constructed via `ripplegw.waveform(name, **config)`.
Every model takes an evaluation grid (a frequency or time array) and a dictionary of parameters, and returns the gravitational-wave polarizations as JAX arrays — so every waveform is **differentiable**, **JIT-compilable**, and **vectorisable**.
See [JAX Transformations](jax.md) for `jax.grad` / `jax.jit` / `jax.vmap` in depth.

## Discovering waveforms

```python
import ripplegw

ripplegw.list_waveforms()                       # every registered model name
ripplegw.list_waveforms(domain="FD")            # filter by metadata
ripplegw.list_waveforms(is_precessing=True)
ripplegw.get_waveform_metadata("IMRPhenomD")    # {"domain": "FD", "is_tidal": False, "is_precessing": False}
```

See the [Waveform Catalogue](catalogue.md) for the full list with each waveform's domain and capabilities.

!!! note
    An unrecognized filter key doesn't raise — it just matches nothing.
    If a filter returns fewer names than you expect, check `get_waveform_metadata(...)` on a known model to see the real key names.

## Constructing a waveform

```python
waveform = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
```

`ripplegw.waveform(name, **config)` is the **only** construction path.
`**config` is forwarded to the model's constructor; for most models this is just `f_ref`, the reference frequency in Hz.

!!! warning "`name` is positional-only"
    `ripplegw.waveform(name="IMRPhenomD")` raises `TypeError` — always pass the name as the first positional argument, `ripplegw.waveform("IMRPhenomD", ...)`.

## Evaluating a waveform

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # see JAX Transformations for why this matters

frequency = jnp.arange(20.0, 1024.0, 0.25)
params = {
    "M_c": 28.3,    # chirp mass [solar masses]
    "eta": 0.247,   # symmetric mass ratio
    "s1_z": 0.0,    # primary aligned spin
    "s2_z": 0.0,    # secondary aligned spin
    "d_L": 440.0,   # luminosity distance [Mpc]
    "phase_c": 0.0,
    "iota": 0.0,
}

polarizations = waveform(frequency, params)
hp, hc = polarizations["p"], polarizations["c"]
```

Every model returns a `dict` with exactly two keys: `"p"` (plus polarization, $h_+$) and `"c"` (cross polarization, $h_\times$), each the same length as the input axis.

**Frequency-domain models return complex arrays; time-domain models return real arrays**.
ripple never performs an FFT internally: a frequency-domain model builds its strain analytically as $A(f) e^{i\psi(f)}$.

See [Parameters and Conventions](parameters.md) for what each parameter name means and its units.

### Visualising the spectrum

```python
import matplotlib.pyplot as plt

plt.loglog(frequency, jnp.abs(hp), label=r"$|h_+|$")
plt.loglog(frequency, jnp.abs(hc), label=r"$|h_\times|$", ls="--")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Strain amplitude")
plt.legend()
plt.show()
```

## Amplitude and phase

For single-mode, aligned-spin models — where an amplitude and phase as a function of frequency are individually well-defined — you can evaluate them separately:

```python
wf = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
amp = wf.amplitude(frequency, params)      # includes distance scaling
phase = wf.phase(frequency, params)        # the exponent in exp(1j * phase)
strain = wf.strain(frequency, params)      # == amp * exp(1j * phase), the pre-polarization strain
```

`amplitude(f, p) * exp(1j * phase(f, p))` reproduces `strain(f, p)` exactly; `__call__` then applies the inclination-dependent plus/cross prefactors on top of `strain` to build `hp`/`hc`.
Not every model supports this — check directly via `isinstance(wf, ripplegw.AmplitudePhaseWaveform)`.

## Evaluating at a fixed distance

Any model whose parameters include `d_L` can be evaluated at 1 Mpc directly, ignoring whatever `d_L` is in `params`:

```python
h_at_1mpc = wf.at_unit_distance(frequency, params)
```

This is exact by construction — `at_unit_distance(axis, params) == wf(axis, {**params, "d_L": 1.0})`.
Check availability the same way: `isinstance(wf, ripplegw.DistanceScaledWaveform)`.

## Switching between models

All waveform models share the same calling interface, so swapping models only requires changing the registered name:

```python
for name in ["IMRPhenomD", "IMRPhenomXAS"]:
    h = ripplegw.waveform(name)(frequency, params)
    print(f"{name}: max|h+| = {jnp.max(jnp.abs(h['p'])):.3e}")
```

This only works cleanly when the models share `parameter_names` (e.g. precessing models need `s1_x`/`s1_y` in addition to `s1_z`, and tidal models need `lambda_1`/`lambda_2`).
