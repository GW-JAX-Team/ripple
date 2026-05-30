# LAL Agreement and Overlap Loss Thresholds

This document records what is currently known about the overlap loss between ripple and LALSuite for each supported waveform, as tested by `tests/cross_validation/test_lal_overlap.py`.

The overlap loss (OL) is defined as `1 - Re(<h1|h2>) / sqrt(<h1|h1> * <h2|h2>)`, using the ET-D PSD noise weighting, evaluated on a 1000-sample Monte Carlo draw of BBH parameter space (seed 42, T = 32 s, f_low = 20 Hz, f_high = 2048 Hz, f_ref = 20 Hz).
A lower OL indicates better agreement.

## Summary table

| Waveform | Threshold | Mean OL | Max OL | Status |
|---|---|---|---|---|
| TaylorF2 | 1e-15 | 6.34e-17 | 3.69e-16 | machine precision |
| IMRPhenomD | 1e-15 | 5.95e-17 | 3.55e-16 | machine precision |
| IMRPhenomD_NRTidalv2 | 1e-15 | 6.57e-17 | 3.21e-16 | machine precision |
| IMRPhenomHM | 1e-15 | 3.7e-17 | 3.2e-16 | machine precision |
| IMRPhenomPv2 | 1e-4 | 8.8e-7 | 3.7e-5 | known cause (LAL-side) |
| IMRPhenomXAS | 1e-15 | 3.5e-17 | 3.3e-16 | machine precision |
| IMRPhenomXAS_NRTidalv3 | 1e-6 | — | — | under investigation |
| IMRPhenomXHM | 1e-6 | 3.6e-9 | 6.5e-7 | known cause |
| IMRPhenomXP | 1e-6 | 3.3e-11 | 3.1e-8 | under investigation |
| IMRPhenomXPHM | 1e-6 | 2.2e-9 | 4.7e-8 | under investigation |

"Machine precision" means OL values are consistent with floating-point rounding (a few times eps_machine, typically 1e-15 to 1e-16).

---

## Waveforms at machine precision

### TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM, IMRPhenomXAS

These waveforms agree with LAL to the limit of float64 arithmetic.
The mean OL values (~0 to 6e-17) are consistent with rounding noise in the ET-D noise-weighted inner product.

---

## Waveforms with a known cause

### IMRPhenomPv2

**Threshold: 1e-4 | Max OL: 3.7e-5**

This large threshold does **not** reflect a deficiency in ripple's implementation.
The overlap loss is almost entirely from a time shift (the amplitude agrees to better than 1e-9 across all samples).
The cause is a difference in how the coalescence-time correction `t0` is computed.

- **LAL** computes `t0` via a 10-point natural cubic spline over `[0.8*f_RD, 1.2*f_RD]` (function `gsl_interp_cspline`, `LALSimIMRPhenomP.c` lines 1060-1151).
  The 10-point grid has spacing of roughly 9-12 Hz, which underresolves the Lorentzian arctan feature in the merger-ringdown phase (characteristic width ~ f_damp ~ 22 Hz), introducing a derivative error of 5-12 us depending on the binary.
- **ripple** computes `t0` using exact JAX autodiff, which gives the true instantaneous derivative.

ripple's result is the more accurate one; the LAL comparison is limited by the coarse spline grid.

The amplitude agrees to better than 5.5e-10 across all tested samples.
After removing the linear phase ramp, the residual is < 1e-10 rad.
The worst-case Δt0 is ~14 µs (worst case: m1=95.1, m2=16.5 M☉).

Two further properties of this waveform pair are worth noting:

- **Continuity at sx = 0**: Both ripple and LAL are continuous as in-plane spin → 0. The phase jump between sx=0 and sx=1e-6 is < 1e-7 rad in both implementations.
- **PhenomPv2 vs PhenomD at zero in-plane spin**: Setting sx=sy=0 in PhenomPv2 does not recover PhenomD. This is expected: `gen_IMRPhenomPv2` internally swaps m1 ↔ m2 to follow the LAL convention (m1 < m2), which re-assigns chi1/chi2 to the opposite mass component. The asymmetric PN phase terms then differ from PhenomD's assignment. Both LAL and ripple exhibit this behaviour identically; it is a convention difference, not a bug.

### IMRPhenomXHM

**Threshold: 1e-6 | Max OL: 6.5e-7**

The deviation is confined almost entirely to a single corner of parameter space: extreme mass ratio (q ~ 0.07), near-extremal primary spin (chi1 ~ 0.98), and hence a near-extremal final spin (a_f ~ 0.98).
In this regime the (3,2) mode exhibits strong spheroidal-spherical mixing near the ringdown.

The (3,2) intermediate phase is determined by a 6x6 linear system, one of whose constraints is the first derivative of the spheroidal-to-spherical (S2S) phase at the intermediate-ringdown boundary `fcutRD`.
This derivative is computed via a first finite difference and evaluates to -420.63 in ripple, compared to ~-421.14 inferred from the LAL waveform.
The 0.5-unit gap appears to come from the rapidly oscillating S2S phase near `fcutRD` (a beat between the (3,2) and (2,2) QNMs): small differences in how the phase of a rapidly oscillating function is sampled in JAX vs C give a slightly different instantaneous slope.
This leads to a ~0.017 rad phase error in the 200-300 Hz intermediate region for the worst-case sample, which contributes ~6.5e-7 to the full-waveform OL.

---

## Waveforms under investigation

### IMRPhenomXAS_NRTidalv3

**Threshold: 1e-6 | Max OL: ~2-3e-8 (10-sample estimate)**

The overlap loss is above the ~1e-15 machine-precision floor.
The following has been established:

- The tidal phase terms (`get_tidal_phase`, merger frequency, etc.) agree with LAL to machine precision.
- The merger frequency `f_final = min(f_last + df, f_merger)` falls in the inspiral phase region for all tested samples.
- The OL grows with total mass (heavier systems have higher OL), suggesting the relevant error is in the high-frequency phase behaviour.
- A hypothesis that the backward finite-difference secant used for `dphiXAS` (which sets the linear phase coefficient `linb`) is the culprit was tested but did not reduce the OL, so the primary source remains unknown.

The 1e-6 threshold is a temporary holding value.

### IMRPhenomXP

**Threshold: 1e-6 | Max OL: 3.1e-8**

The mean OL (3.3e-11) is well below 1e-6 for most of parameter space, but a handful of samples reach the 1e-8 level.
The worst case found so far (m1 = 27.8 Msun, m2 = 18.0 Msun, |chi1| ~ 0.25 in-plane-dominated) does not show an obvious pattern — the spin magnitudes are moderate and the mass ratio is not extreme.
The precessing waveforms apply MSA (multiple-scale analysis) for the spin-precession angles, and any difference between ripple's MSA angles and LAL's could contribute.
The source of the occasional ~1e-8 excess has not been traced.

### IMRPhenomXPHM

**Threshold: 1e-6 | Max OL: 4.7e-8**

XPHM extends XP with higher-order modes and shows a systematically higher mean OL (2.2e-9) compared to XP (3.3e-11).
The pattern among the worst-case samples does not point to a single cause: systems with large spin magnitudes (|chi1| ~ 0.94-0.98) feature in the top few, but so do moderate-spin systems.

Two potential contributions have not yet been isolated:

1. **Inherited from XHM.** The (3,2) mode phase error in XHM (up to ~0.017 rad for extreme parameters) propagates into XPHM, which uses the same XHM co-precessing-frame modes before twisting.
2. **Precession-angle differences.** Any small differences in the MSA precession angles between ripple and LAL would modify how the co-precessing modes combine into the inertial frame.

Neither source has been confirmed as the dominant one.

---

## Notes

- All OL values are from a 1000-sample GPU run (A100, JAX 0.10.1, float64) using the ET-D PSD and the parameter bounds in `tests/cross_validation/test_lal_overlap.py`.
- A different random seed or parameter range may give different extreme values.
- The thresholds are set with >2x margin over the observed max OL to allow for variation across seeds and parameter ranges.
- The OL is not maximised over time or phase shifts; it is the raw un-maximised overlap between the ripple and LAL waveforms with identical input parameters.
