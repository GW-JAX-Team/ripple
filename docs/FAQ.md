# FAQ

## Float precision

JAX defaults to float32, but gravitational-wave waveform computations typically require float64 precision.
Always enable it at the top of your script, before any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without this, you may see unexpected numerical errors or inaccurate waveforms, particularly at high frequencies or for long signals.

## JIT compilation time

The first call to a JIT-compiled waveform (e.g. via `jax.jit`) triggers XLA compilation, which can take several seconds.
This is normal — subsequent calls will be much faster.
If you are timing ripple for benchmarking purposes, discard the first call.

To disable JIT for debugging:

```python
jax.config.update("jax_disable_jit", True)
```

## Compilation is slow for complex models

If you wrap a ripple waveform inside a larger likelihood with many operations or Python-level loops, JAX may take a long time to compile the full computational graph.
Replacing Python loops with `jax.lax.scan` or `jax.vmap` where possible can significantly reduce compilation time.

## `ripplegw.IMRPhenomD` (or `waveform_preset`) raises `AttributeError`

Older ripple code constructed models directly off the top-level module (`ripplegw.IMRPhenomD(f_ref=20.0)`) or looked them up in a `waveform_preset` dict.
Neither exists anymore.
The only construction path is the registry factory:

```python
waveform = ripplegw.waveform("IMRPhenomD", f_ref=20.0)
```

See [Working with Waveforms](guides/waveforms.md) for the full interface and `ripplegw.list_waveforms()` for every registered name.

## Where is `t_c` (time of coalescence)?

It isn't exposed.
Every built-in model fixes the time of coalescence internally and only exposes `phase_c` (coalescence phase) and `iota` (inclination) as extrinsic parameters — see [Parameters and Conventions](guides/parameters.md).
If your use case needs to vary `t_c`, you currently need to apply the standard linear-in-frequency phase shift ($e^{2\pi i f\,\delta t_c}$) yourself on the returned strain.

## Continuous-wave signals: ephemeris files

The continuous-wave models in `ripplegw.cw` (`ExactPulsarSignal`, `PulsarSignal`, `BinaryPulsarSignal`) need a JPL solar-system ephemeris to barycenter the signal.
You pass the path to a standard LALPulsar ephemeris file, e.g. `earth00-40-DE405.dat.gz` (and `sun00-40-DE405.dat.gz` for the full and binary models, which include the Shapiro delay).
ripple parses these files itself — it does not import `lal`/`lalpulsar` at runtime.

The `pip` `lalsuite` wheel does **not** bundle the ephemeris files.
Obtain them from an installed LALSuite (`$LALPULSAR_DATADIR`) or the [LALSuite repository](https://git.ligo.org/lscsoft/lalsuite/-/tree/master/lalpulsar/lib), and make sure your observation span lies within the file's coverage (the standard `earth00-40-*`/`sun00-40-*` files cover GPS years 2000–2040).

Two practical notes:

- The call axis is **time in seconds relative to `start_gps`** (not absolute GPS), which preserves float64 precision; absolute sample times are reconstructed internally as integer second + fraction.
- Sample times outside the ephemeris span are clamped rather than raising, so keep your time grid within the file's coverage.
