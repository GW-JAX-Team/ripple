# Validity Ranges

ripple ports the LAL implementations. Most models do not enforce their parameter
checks and extrapolate silently outside these ranges. `IMRPhenomNSBH` applies
the recoverable checks described below.
Each model links to its LAL 7.26 documentation; limits that page does not state are checks in the [LALSuite 7.26.16 source](https://git.ligo.org/lscsoft/lalsuite/-/tree/lalsuite-v7.26.16).

## Compact-binary coalescences

The table lists what LAL rejects and the calibrated range stated in LAL's reviewed model documentation, with $q = m_1/m_2 \ge 1$.
LAL documents every model as usable outside its calibrated range, with unverified accuracy there.

| Waveform | LAL rejects | Calibrated range |
| --- | --- | --- |
| [`TaylorF2`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_taylor_f2_8c.html) | — | Post-Newtonian; no calibration |
| [`IMRPhenomD`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_d_8c.html) | $\lvert\chi_i\rvert > 1$ | $q = 1, 4, 8, 18$ with spins in $[-0.95, 0.98]$, $[-0.75, 0.75]$, $[-0.85, 0.85]$, $[-0.8, 0.4]$ respectively |
| [`IMRPhenomHM`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_h_m_8c.html) | $\lvert\chi_i\rvert > 1$ | As `IMRPhenomD`; the higher modes are not calibrated to numerical relativity |
| [`IMRPhenomPv2`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_p_8c.html) | $\lvert\boldsymbol{\chi}_i\rvert > 1$ | As `IMRPhenomD`; LAL also warns for strongly anti-aligned spins, where the precession opening angle is no longer tracked |
| [`IMRPhenomD_NRTidalv2`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_d___n_r_tidal_8c.html) | $\lvert\chi_i\rvert > 1$, $\Lambda_i < 0$ | As `IMRPhenomD`; [tides](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_n_r_tuned_tides_8c.html) to $q = 1.5$ and $\kappa_2^T \in [40, 5000]$ |
| [`IMRPhenomPv2_NRTidalv2`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_p_8c.html) | $\lvert\boldsymbol{\chi}_i\rvert > 1$, $\Lambda_i < 0$ | As `IMRPhenomD_NRTidalv2`; the review found unphysical features for $q < 1.5$ with both $\Lambda_i > 2000$ |
| [`IMRPhenomNSBH`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_n_s_b_h_8c.html) | $m_\mathrm{NS} > 3\,M_\odot$, $q \notin [1, 100]$, $\Lambda_\mathrm{NS} \notin [0, 5000]$; a negative inspiral frequency derivative | Amplitude at $q = 2, 3$ with BH spin in $[-0.5, 0.75]$ and $q = 4, 5$ with BH spin in $[0, 0.75]$; non-spinning NS with $m_\mathrm{NS} \ge 1\,M_\odot$; phase as `IMRPhenomD_NRTidalv2` |
| [`IMRPhenomXAS`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_8c.html), [`IMRPhenomXHM`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_h_m_8c.html) | $q > 1000$, $\lvert\chi_i\rvert > 1$ | $q$ from 1 to 1000; LAL warns beyond the numerical-relativity data ($q > 20$) and for $\lvert\chi_i\rvert > 0.99$ |
| [`IMRPhenomXP`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_8c.html), [`IMRPhenomXPHM`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_p_h_m_8c.html) | $q > 1000$, $\lvert\boldsymbol{\chi}_i\rvert > 1$ | As `IMRPhenomXAS` |
| [`IMRPhenomXAS_NRTidalv3`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_8c.html), [`IMRPhenomXP_NRTidalv3`](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_8c.html) | As the BBH model, and $\Lambda_i < 0$ | As the BBH model; [tides](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_n_r_tuned_tides_8c.html) to $q = 2$ and $\kappa_2^T \in [40, 5000]$ |

LAL applies the `IMRPhenomNSBH` limit $m_\mathrm{NS} \le 3\,M_\odot$ to the detector-frame mass, so it also rejects ordinary systems at redshift $z \gtrsim 1.15$; ripple's model has no such limit.

### Recoverable IMRPhenomNSBH validation

`IMRPhenomNSBH` returns `NaN` for an entire sample when $q$ is outside
$[1, 100]$, $\Lambda_2$ is outside $[0, 5000]$, or the inspiral frequency
derivative is negative at any requested frequency below LAL's SPA matching
frequency. The matching frequency is $0.1$ times the **PhenomC BBH** ringdown
frequency, before the NSBH corrections. Chirp mass must also be finite and
positive, and the PhenomC remnant-spin fit must give a physical matching frequency.
The mass-ratio bound is checked in the supplied symmetric mass ratio before
conversion to component masses; this also rejects unphysical `eta > 0.25`.
The lower bound allows a few units in the last place, so `eta` converted from
masses with $q = 100$ exactly is accepted, as LAL accepts those masses.
No neutron-star mass cutoff is applied. Nonzero NS spin and NS masses below
one solar mass remain calibration caveats rather than rejection rules.

Normal evaluation uses array operations without error side effects. In a
compiled inference loop, reject nonfinite waveforms explicitly:

```python
@jax.jit
def log_likelihood(params):
    h = waveform(frequencies, params)
    valid = jnp.all(jnp.isfinite(h["p"])) & jnp.all(jnp.isfinite(h["c"]))
    safe_h = jax.tree.map(lambda x: jnp.where(valid, x, 0.0), h)
    return jnp.where(valid, evaluate_likelihood(safe_h), -jnp.inf)
```

This rejects individual samples under `jit`, `vmap`, and `lax.scan` without
throwing an exception. For scalar `jit` and `lax.scan` calls, the waveform
stops at the first failed domain guard: raw chirp/tidal bounds before mass
conversion, remnant spin before inspiral coefficients, and the frequency
derivative before remaining amplitude fits, merger properties, and phase.
Strain construction and polarization factors are inside the accepted branch.
Intermediates used by the guards are reused by the waveform.

Under ordinary per-sample `vmap`, conditionals become selections and both
branches are evaluated, so invalid samples do not save waveform work. Use
`lax.scan` or `lax.map` when per-sample early rejection is required; their
sequential execution can cost more than vectorization for valid batches.
Derivatives (`jax.grad`, `jax.jacfwd`, `jax.jvp`) evaluate the guards as
selections too, which avoids keeping every branch's intermediates across the
conditionals; values and derivatives of valid samples are unchanged. The
derivative is zero for a sample the guards reject, and for one with a
non-finite, zero or negative frequency, reference frequency or `f_max`, a zero
distance, or a non-finite distance, time or phase shift, or inclination. Such a
sample therefore does not turn the pattern above, a batch sum, or a parameter
shared with valid samples into `NaN`. This holds for inputs of ordinary scale:
finite values near the edge of the floating-point range (for example
frequencies or a reference frequency around 1e300 Hz or 1e-300 Hz, a chirp mass
of 1e-100, or time and phase shifts near 1e308) overflow inside the model and
are not covered. The zero is not a physical derivative: gradient samplers should
keep proposals in their physical domain.

For optional error messages, enable diagnostics and functionalize the checks:

```python
from jax.experimental import checkify

waveform = ripplegw.waveform("IMRPhenomNSBH", diagnostics=True)
checked = jax.jit(checkify.checkify(waveform))
err, h = checked(frequencies, params)
message = err.get()  # Host inspection; returns None for a valid sample.
# err.throw() is optional and explicitly raises on a failed check.
```

`diagnostics=True` adds `checkify.debug_check` annotations; without
`checkify.checkify` they do not report errors, while the NaN rejection still
applies. Leave diagnostics disabled in the normal sampling path to avoid
effect annotations. Host inspection of error messages synchronizes execution.

The low-level `gen_IMRPhenomNSBH` and `gen_IMRPhenomNSBH_hphc` functions accept
the same static `diagnostics` keyword.

## Bursts

[LAL](https://git.ligo.org/lscsoft/lalsuite/-/blob/lalsuite-v7.26.16/lalsimulation/lib/LALSimBurst.c) rejects a negative `Q`, `f_0` or `hrss` for `SineGaussian`, and an eccentricity `e` outside $[0, 1]$.

## Continuous waves

- All sample times must lie within the span of the ephemeris files. [LAL](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalpulsar/_l_a_l_barycenter_8c.html) rejects other times; ripple clamps to the ends of the tables and returns unphysical values.
- LAL rejects a right ascension outside $[-2\pi, 2\pi]$ or a declination outside $[-\pi/2, \pi/2]$.
- `BinaryPulsarSignal` models elliptical orbits only, $0 \le$ `ecc` $< 1$. [LAL](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalpulsar/_generate_spin_orbit_c_w_8c.html) rejects a negative eccentricity; ripple returns non-finite values for `ecc` $\ge 1$.
