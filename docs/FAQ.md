# FAQ

## Float precision

JAX defaults to float32, but gravitational-wave waveform computations typically require float64 precision. Always enable it at the top of your script, before any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without this, you may see unexpected numerical errors or inaccurate waveforms, particularly at high frequencies or for long signals.

## JIT compilation time

The first call to a JIT-compiled waveform (e.g. via `jax.jit`) triggers XLA compilation, which can take several seconds. This is normal — subsequent calls will be much faster. If you are timing ripple for benchmarking purposes, discard the first call.

To disable JIT for debugging:

```python
jax.config.update("jax_disable_jit", True)
```

## Compilation is slow for complex models

If you wrap a ripple waveform inside a larger likelihood with many operations or Python-level loops, JAX may take a long time to compile the full computational graph. Replacing Python loops with `jax.lax.scan` or `jax.vmap` where possible can significantly reduce compilation time.
