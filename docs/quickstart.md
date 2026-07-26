# Quick Start

## Basic Usage

To generate a gravitational-wave waveform, construct the model by name via `ripplegw.waveform(...)` and call it with a frequency array and a parameter dictionary:

```python
import jax.numpy as jnp
import ripplegw

# Frequency grid: 20–1024 Hz at 0.25 Hz resolution
frequency = jnp.arange(20.0, 1024.0, 0.25)

# GW150914-like binary black hole parameters
params = {
    "M_c": 28.3,   # chirp mass [solar masses]
    "eta": 0.247,  # symmetric mass ratio
    "s1_z": 0.0,   # primary aligned spin
    "s2_z": 0.0,   # secondary aligned spin
    "d_L": 440.0,  # luminosity distance [Mpc]
    "phase_c": 0.0,
    "iota": 0.0,
}

# Construct the waveform model by its registered name
waveform = ripplegw.waveform("IMRPhenomD", f_ref=20.0)

# Evaluate: returns a dict with keys "p" (h+) and "c" (hx)
polarizations = waveform(frequency, params)
hp = polarizations["p"]
hc = polarizations["c"]
```

`ripplegw.waveform(name, **config)` is the single entry point for every model — the `**config` keywords (e.g. `f_ref`) are forwarded to the model's constructor.
All waveform models share the same calling interface, so switching models only requires changing the name:

```python
waveform = ripplegw.waveform("IMRPhenomXAS", f_ref=20.0)   # same params dict
waveform = ripplegw.waveform("TaylorF2", f_ref=20.0)       # add lambda_1, lambda_2 for BNS
```

Discover what's available and inspect a model's metadata with:

```python
ripplegw.list_waveforms()                # every registered model name
ripplegw.list_waveforms(domain="FD")     # filter by metadata
ripplegw.get_waveform_metadata("IMRPhenomD")
```

## GPU and Gradient Support

ripple waveforms are pure JAX functions, so they work out of the box with `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
import jax

# JIT-compile for fast repeated evaluation
fast_waveform = jax.jit(waveform)

# Compute gradient w.r.t. chirp mass
def log_likelihood(M_c):
    h = waveform(frequency, {**params, "M_c": M_c})
    return -0.5 * jnp.sum(jnp.abs(h["p"]) ** 2)

grad_Mc = jax.grad(log_likelihood)(params["M_c"])
```

GPU execution requires no code changes — JAX will automatically use the GPU if one is available.
See the [Installation](installation.md) page for GPU setup.

## Next steps

- **[Working with Waveforms](guides/waveforms.md)** — what `__call__` returns, amplitude/phase evaluation, and switching between models in more depth.
- **[Parameters and Conventions](guides/parameters.md)** — what every parameter name means.
- **[Waveform Catalogue](guides/catalogue.md)** — every registered model, its parameters, and its capabilities.
- **[JAX Transformations](guides/jax.md)** — `jit`/`grad`/`vmap` patterns and precision.
