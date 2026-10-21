# Validity Ranges

ripple ports the LAL implementations but does not enforce their parameter checks.
Outside these ranges the models extrapolate silently.

## Compact-binary coalescences

The table lists what LAL rejects and where it warns about, or documents, the edge of the calibration, with $q = m_1/m_2 \ge 1$.

| Waveform | LAL rejects | Outside the calibration |
| --- | --- | --- |
| `TaylorF2` | — | No range checks |
| `IMRPhenomD` | $\lvert\chi_i\rvert > 1$ | $q > 18$ |
| `IMRPhenomHM` | $\lvert\chi_i\rvert > 1$ | $q > 20$ |
| `IMRPhenomPv2` | $\lvert\vec\chi_i\rvert > 1$ | $q > 18$; strongly anti-aligned spins, where the precession opening angle is no longer tracked |
| `IMRPhenomD_NRTidalv2` | $\lvert\chi_i\rvert > 1$, $\Lambda_i < 0$ | $q > 18$; tides beyond $q = 1.5$ or $\kappa_2^T \notin [40, 5000]$ |
| `IMRPhenomPv2_NRTidalv2` | $\lvert\vec\chi_i\rvert > 1$, $\Lambda_i < 0$ | As `IMRPhenomPv2`, with the tides of `IMRPhenomD_NRTidalv2`; $q < 1.5$ with both $\Lambda_i > 2000$ |
| `IMRPhenomNSBH` | $m_\mathrm{BH} < m_\mathrm{NS}$, $q > 100$, $\Lambda_\mathrm{NS} > 5000$, $m_\mathrm{NS} > 3\,M_\odot$; a negative inspiral frequency derivative | $m_\mathrm{NS} < 1\,M_\odot$, non-zero NS spin |
| `IMRPhenomXAS`, `IMRPhenomXHM` | $q > 1000$, $\lvert\chi_i\rvert > 1$ | $q > 20$, $\lvert\chi_i\rvert > 0.99$ |
| `IMRPhenomXP`, `IMRPhenomXPHM` | $q > 1000$, $\lvert\vec\chi_i\rvert > 1$ | $q > 20$, $\lvert\vec\chi_i\rvert > 0.99$ |
| `IMRPhenomXAS_NRTidalv3`, `IMRPhenomXP_NRTidalv3` | As the BBH model, and $\Lambda_i < 0$ | As the BBH model; tides beyond $q = 2$ or $\kappa_2^T \notin [40, 5000]$ |

LAL applies the `IMRPhenomNSBH` limit $m_\mathrm{NS} \le 3\,M_\odot$ to the detector-frame mass, so it also rejects ordinary systems at redshift $z \gtrsim 1.15$; ripple's model has no such limit.

## Bursts

LAL rejects a negative `Q`, `f_0` or `hrss` for `SineGaussian`, and an eccentricity `e` outside $[0, 1]$.

## Continuous waves

- All sample times must lie within the span of the ephemeris files. LAL rejects other times; ripple clamps to the ends of the tables and returns unphysical values.
- LAL rejects a right ascension outside $[-2\pi, 2\pi]$ or a declination outside $[-\pi/2, \pi/2]$.
- `BinaryPulsarSignal` models elliptical orbits only, $0 \le$ `ecc` $< 1$. LAL rejects a negative eccentricity; ripple returns non-finite values for `ecc` $\ge 1$.
