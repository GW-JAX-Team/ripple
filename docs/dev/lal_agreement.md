# LAL Agreement and Overlap Loss Thresholds

This document records what is currently known about the overlap loss between ripple and LALSuite for each supported waveform, as tested by `tests/cross_validation/test_lal_overlap.py`.

The per-polarization overlap loss (OL) is `1 - Re(<h1|h2>) / sqrt(<h1|h1> * <h2|h2>)`, using the ET-D PSD noise weighting, evaluated on a Monte Carlo draw of parameter space (seed 42; BBH: T = 32 s, f in [20, 2048] Hz; BNS: T = 128 s, f in [5, 4096] Hz).
A lower OL indicates better agreement.

**The asserted metric (since 2026-08) is the SNR-weighted combined loss**

    OL_combined = (<hp|hp>*OL_+ + <hc|hc>*OL_x) / (<hp|hp> + <hc|hc>)

(`compute_polarization_weighted_overlap_loss` in `tests/utils.py`), which equals the polarization-angle-averaged mismatch of the detector strain.  Rationale: judging `hc` in isolation near edge-on divides by a vanishing signal.  A twist Euler-angle error common to alpha and epsilon cancels in the dominant `exp(2i(alpha-eps))` phase of `hp` but survives in the `O(sin beta)` sidebands that are all `hc` consists of edge-on, so the per-polarization `hc` loss is amplified by `A(iota) = [(1+cos^2 iota)/(2 cos iota)]^2` — unbounded as iota -> pi/2 — while carrying negligible detector SNR (measured on n=5000: `log10(OL_hc/OL_hp) ~ 1.84*log10 A`, R^2 = 0.93; a controlled angle-injection experiment reproduces the exponent, ~-1.9 in cos^2 iota, independent of chi_p).  The SNR weight cancels that amplification exactly; per-polarization losses remain in the CSVs (`overlap_loss_hp/hc`, `overlap_loss_max`) as diagnostics.

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
| IMRPhenomXP | 1e-6 | ~2e-16 | ~3e-16 | machine precision (combined metric) |
| IMRPhenomXP_NRTidalv3 | 2e-6 | — | see section | known cause (LAL MSA conditioning) |
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

**Threshold: 1e-6 | typical max OL ~1e-16 (n=5000, combined metric ~ per-pol here)**

XP agrees with LAL at the float64 floor across the BBH prior (median OL 1.9e-16 for both
polarizations, no inclination correlation).  Two reasons, both verified by a controlled
angle-injection experiment (`results/analyse_mismatches.ipynb`, cells 12-14):

1. At BBH spins the MSA S^2 cubic is far from its double root (median `|1+acosarg|` =
   2.9e-3), so there is no amplified angle error to expose.
2. The edge-on amplification switches off once the opening angle beta is no longer small:
   the measured coupling exponent falls from -1.9 at chi_p <= 0.03 to -0.3 at chi_p = 0.31,
   and the XP prior median is chi_p = 0.43.

Historical note: earlier versions of this section attributed rare XP outliers to float64
cancellation near the angular-momentum critical point `J = L - Smi`
(`1/sqrt(d0+d2+d4)` in the MSA correction).  That mechanism exists but is rare; the
dominant, population-wide conditioning problem is the S^2 cubic double root described
under IMRPhenomXP_NRTidalv3 below.

### IMRPhenomXP_NRTidalv3

**Threshold: 2e-6 (combined metric; expected headroom ~1000x after the 2026-08 fixes)**

The BNS prior (chi in [-0.05, 0.05]) puts ~88% of samples in a regime where the MSA S^2
cubic (Eq. B2-B4 of arXiv:1703.03967, solved as `theta = arccos(a)/3`) is within 1e-6 of
a double root: `a -> -1`, where `arccos(-1+e) = pi - sqrt(2e)` has an infinite derivative.
Half the mantissa of `Spl2/Smi2` is lost there and further amplified (~1e10-1e12 in total)
through `cm = Smi2*eta^2 - c1^2` and `adD = aw/(4*sqrt(|cp*cm|))` into `Omegaz0`, which
sets the leading `v^-3` term of BOTH `phi_z` and `zeta` — i.e. a *common* error in
alpha(f) and epsilon(f).  `cos beta` is unaffected.  `corr(log chi_p, log|1+a|) = +0.5`:
small in-plane spin drives the degeneracy, which is why mismatches correlated with chi_p.

**Both codes are unstable there.** Perturbing m1 by 1 part in 1e15 moves alpha(f) by a
median ~1e-2 rad in LAL itself.  ripple solves the same cubic by deflating the
well-separated root and recovering the colliding pair from Vieta's formulas
(`IMRPhenomX_Return_Roots_MSA`), which is stable (~1e-11 rad self-sensitivity) — ripple is
deliberately *more accurate than LAL* in this regime, so angle-level agreement there is
bounded by LAL's own noise floor and is not a ripple defect.  Where the cubic is
resolvable (`|1+a| > 1e-6`) ripple tracks LAL to ~1e-8 rad
(`tests/cross_validation/test_msa_angles.py`).

The common alpha/epsilon error cancels in `hp` and appears only in `hc`, amplified by
`A(iota)` (see the metric note at the top).  Under the combined metric the amplification
cancels and all samples — including near-edge-on and near-degenerate ones — are asserted
against the threshold; the `msa_degenerate` CSV column marks the near-degenerate ones as
a diagnostic.

Two further LAL-side facts documented here for reproducibility:

- LAL builds this approximant via `XLALSimIMRPhenomXPHM` (modes (2,+-2)); multibanding is
  ON by default there and must be disabled in comparisons
  (`PhenomXPHMThresholdMband = PhenomXHMThresholdMband = 0`, see `tests/utils.py`).
  With multibanding on, LAL disagrees with its own multibanding-off output by up to
  OL ~ 1 for edge-on BNS samples — the historical `overlap_loss ~ 1` outlier.
- The twist cutoff matches LAL bin-for-bin (`Mf <= (fCutDef/M_sec)*M_sec`, inclusive,
  fCutDef in {0.3, 0.33}); the sole corner case is chiEff > 0.99, where ripple's
  co-precessing amplitude is independently zeroed at Mf = 0.3 (`fM_CUT` in
  `IMRPhenomXAS.py`) so Mf in (0.3, 0.33] stays zero while LAL keeps it — unreachable
  within the test priors (see `mf_twist_cutoff`).
- ripple matches LAL's arithmetic bit-for-bit up to the cubic solve (eta/q mass
  conventions, vector-norm/dot reduction order, Mf and per-mode-v groupings, twist
  prefactor grouping — 0 ULP verified over 100 samples).  The irreducible residual is
  libm: glibc's `cbrt` is itself up to ~2.4 ULP off the true cube root, and XLA's
  `acos`/`cos` differ from glibc's by ~1-2 ULP; ripple uses a correctly-rounded cbrt
  (`cbrt_cr`) instead of chasing glibc's idiosyncratic last bits.

### IMRPhenomXPHM

**Threshold: 1e-6 | Max OL: 9.97e-8**

The worst case is the same system as XP (same MSA near-singularity mechanism). Beyond that, the next worst cases have high aligned spin (|chi1| ~ 0.94–0.98), where the (3,2) mode contributes a small additional error (see XHM section), diluted ~50–100× by the other four modes.

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
