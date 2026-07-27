# JAX Transformations

Every ripple waveform is a pure JAX function of `(axis, params)`, so it composes directly with `jax.grad`, `jax.jit`, and `jax.vmap`.
This page shows the common patterns; see the [FAQ](../FAQ.md) for troubleshooting compile times and precision issues.

## Float precision

JAX defaults to 32-bit floats.
Enable 64-bit precision before doing anything else — without it you may see inaccurate waveforms, particularly at high frequencies or for long signals:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Gradients with `jax.grad`

Construct the waveform once, outside the function you differentiate — the model object is reused across every gradient evaluation rather than rebuilt each call:

```python
import jax
import jax.numpy as jnp
import ripplegw

jax.config.update("jax_enable_x64", True)

frequency = jnp.arange(20.0, 1024.0, 0.25)
params = {
    "M_c": 28.3, "eta": 0.247, "s1_z": 0.0, "s2_z": 0.0,
    "d_L": 440.0, "phase_c": 0.0, "iota": 0.0,
}
wf = ripplegw.waveform("IMRPhenomD", f_ref=20.0)

def total_power(M_c):
    h = wf(frequency, {**params, "M_c": M_c})
    return jnp.sum(jnp.abs(h["p"]) ** 2)

grad_Mc = jax.grad(total_power)(params["M_c"])
```

This same pattern differentiates with respect to any parameter, or several at once via `jax.grad` on a function that takes a pytree of parameters.

## Fast repeated evaluation with `jax.jit`

```python
fast_wf = jax.jit(wf)

_ = fast_wf(frequency, params)  # first call traces and compiles — this is slow
h = fast_wf(frequency, params)  # subsequent calls with the same shapes are fast
h["p"].block_until_ready()      # JAX dispatch is async; block before timing
```

The first call to a `jit`-compiled waveform triggers XLA compilation, which will be slow for large models.
If you're timing ripple, always discard the first call and call `.block_until_ready()` before stopping your timer — see [Benchmarking](benchmarking.md) for a ready-made CLI that already does this correctly.

## Batch evaluation with `jax.vmap`

```python
Mc_values = jnp.linspace(20.0, 40.0, 1000)

def h_plus(M_c):
    return wf(frequency, {**params, "M_c": M_c})["p"]

hp_batch = jax.jit(jax.vmap(h_plus))(Mc_values)   # shape (1000, n_freq)
```

Prefer `jax.vmap` over a Python loop when evaluating many parameter sets — a Python loop retraces (or re-dispatches) once per iteration, while `vmap` compiles a single batched kernel.

## Compilation time for large models

Some models (particularly the higher-mode and precessing families) have long JIT compile times on first call because of their size.
If you only need to evaluate once, `jax.config.update("jax_disable_jit", True)` skips compilation entirely at some runtime cost; if you need many evaluations, `jax.jit` still wins overall.
