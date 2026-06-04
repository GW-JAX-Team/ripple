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
| IMRPhenomXP | 1e-6 | 1.20e-10 | 7.91e-08 | known cause |
| IMRPhenomXPHM | 1e-6 | 2.44e-09 | 9.97e-08 | known cause |

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

The cause is a difference in how the coalescence-time correction `t0` is computed:

- **LAL** computes `t0` via a 10-point natural cubic spline over `[0.8*f_RD, 1.2*f_RD]` (function `gsl_interp_cspline`, `LALSimIMRPhenomP.c` lines 1060–1151). The 10-point grid has spacing of roughly 9–12 Hz, which underresolves the Lorentzian arctan feature in the merger-ringdown phase (characteristic width ~ f_damp ~ 22 Hz), introducing a derivative error of 5–12 µs depending on the binary.
- **ripple** computes `t0` using exact JAX autodiff, which gives the true instantaneous derivative.

ripple's result is the more accurate one; the LAL comparison is limited by the coarse spline grid. The worst-case Δt0 is ~14 µs (m1=95.1, m2=16.5 M☉). After removing the linear phase ramp, the residual is < 1e-10 rad.

Two further properties:

- **Continuity at sx = 0**: Both ripple and LAL are continuous as in-plane spin → 0. The phase jump between sx=0 and sx=1e-6 is < 1e-7 rad in both implementations.
- **PhenomPv2 vs PhenomD at zero in-plane spin**: Setting sx=sy=0 in PhenomPv2 does not recover PhenomD. `gen_IMRPhenomPv2` internally swaps m1 ↔ m2 to follow the LAL convention (m1 < m2), which re-assigns chi1/chi2 to the opposite mass component. The asymmetric PN phase terms then differ from PhenomD's assignment. Both LAL and ripple exhibit this behaviour identically; it is a convention difference, not a bug.

### IMRPhenomXHM

**Threshold: 1e-6 | Max OL: 6.5e-7**

The deviation is confined to extreme mass ratio (q ~ 0.07), near-extremal primary spin (chi1 ~ 0.98). In this regime the (3,2) mode exhibits strong spheroidal-spherical mixing near the ringdown.

The (3,2) intermediate phase is determined by a 6×6 linear system, one of whose constraints is the first derivative of the spheroidal-to-spherical (S2S) phase at `fcutRD`. This derivative evaluates to −420.63 in ripple vs ~−421.14 inferred from LAL. The 0.5-unit gap comes from the rapidly oscillating S2S phase near `fcutRD` (a beat between the (3,2) and (2,2) QNMs): small differences in how the phase of a rapidly oscillating function is sampled in JAX vs C give a slightly different instantaneous slope. This leads to a ~0.017 rad phase error in the 200–300 Hz intermediate region for the worst-case sample.

### IMRPhenomXP

**Threshold: 1e-6 | Max OL: 5.0e-7**

The max OL is dominated by a single parameter combination (m1=19.75, m2=9.10 M☉, chip≈0.032, incl≈77°) where the MSA precession correction has a near-singularity in the sensitive band.

In `IMRPhenomX_Return_phiz_of_v_MSA_precav_correction_LAL`, the correction formula contains `1/sqrt(d0+d2+d4)`, which diverges when `d0+d2+d4 = 0`. This zero-crossing corresponds to the angular momentum resonance `J = L − Smi`. At f ≈ 115 Hz for this system, `J = L − Smi` to 11 significant figures, making `d0+d2+d4 ≈ −4×10⁻¹⁶` with 8 orders of catastrophic cancellation in the sum. GPU/CPU float64 differences (from different FMA patterns in the spin-evolution coefficient computation) then cause a ~3 mrad discrepancy in the precession angle at that bin, which is amplified ~14× in hc vs hp for inclination ≈ 77°.

This is a fundamental float64 limitation: both ripple and LAL use the same formula, and the error arises from irreducible GPU/CPU rounding differences near the resonance.

### IMRPhenomXPHM

**Threshold: 1e-6 | Max OL: 9.97e-8**

The worst case is the same system as XP (same MSA resonance mechanism). Beyond that, the next worst cases have high aligned spin (|chi1| ~ 0.94–0.98), where the (3,2) mode contributes a small additional error (see XHM section), diluted ~50–100× by the other four modes.

---

## Waveforms under investigation

### IMRPhenomXAS_NRTidalv3

**Threshold: 1e-6 | Max OL: ~2-3e-8 (10-sample estimate)**

The overlap loss is above the machine-precision floor. The tidal phase terms agree with LAL to machine precision; the OL grows with total mass, pointing to a high-frequency phase error whose source has not yet been identified.

The 1e-6 threshold is a temporary holding value.

---

## Notes

- OL values are from 1000-sample GPU runs.
- A different random seed or parameter range may give different extreme values.
- The thresholds are set with >2x margin over the observed max OL to allow for variation across seeds and parameter ranges.
- The OL is not maximised over time or phase shifts; it is the raw un-maximised overlap between the ripple and LAL waveforms with identical input parameters.
