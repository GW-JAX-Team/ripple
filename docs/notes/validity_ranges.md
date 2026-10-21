# Validity Ranges

ripple ports the LAL implementations but does not enforce their parameter checks.
Outside these ranges the models extrapolate silently.
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

## Bursts

[LAL](https://git.ligo.org/lscsoft/lalsuite/-/blob/lalsuite-v7.26.16/lalsimulation/lib/LALSimBurst.c) rejects a negative `Q`, `f_0` or `hrss` for `SineGaussian`, and an eccentricity `e` outside $[0, 1]$.

## Continuous waves

- All sample times must lie within the span of the ephemeris files. [LAL](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalpulsar/_l_a_l_barycenter_8c.html) rejects other times; ripple clamps to the ends of the tables and returns unphysical values.
- LAL rejects a right ascension outside $[-2\pi, 2\pi]$ or a declination outside $[-\pi/2, \pi/2]$.
- `BinaryPulsarSignal` models elliptical orbits only, $0 \le$ `ecc` $< 1$. [LAL](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalpulsar/_generate_spin_orbit_c_w_8c.html) rejects a negative eccentricity; ripple returns non-finite values for `ecc` $\ge 1$.
