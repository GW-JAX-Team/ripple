# LAL Agreement and Overlap Loss Thresholds

This document records what's known about each supported waveform's agreement with LALSuite: the accuracy threshold enforced by `tests/cross_validation/test_overlap.py`, and — for any waveform whose threshold is looser than machine precision — why.
See `tests/cross_validation/tolerances.toml` for the enforced values themselves; this page only needs to stay in sync with that file's thresholds, not with any specific measured result (see [Testing](testing.md) for how to run the campaign yourself).

The overlap loss (OL) is `1 - Re(<h1|h2>) / sqrt(<h1|h1> * <h2|h2>)`, using the ET-D PSD noise weighting.
A lower OL indicates better agreement.
"Machine precision" means a threshold consistent with floating-point rounding (a few times eps_machine, typically 1e-15 to 1e-16), not a real physical discrepancy.

## Summary table

| Waveform | Threshold | Status |
|---|---|---|
| TaylorF2 | 1e-15 | machine precision |
| IMRPhenomD | 1e-15 | machine precision |
| IMRPhenomD_NRTidalv2 | 1e-15 | machine precision |
| IMRPhenomHM | 1e-15 | machine precision |
| IMRPhenomPv2 | 1e-4 | known cause (LAL-side) |
| IMRPhenomXAS | 1e-15 | machine precision |
| IMRPhenomXAS_NRTidalv3 | 1e-12 | resolved |
| IMRPhenomXHM | 1e-6 | known cause |
| IMRPhenomXP | 1e-6 | known cause |
| IMRPhenomXPHM | 1e-6 | known cause |

The continuous-wave (CW) models below are validated by a different mechanism (see
[below](#continuous-wave-cw-models)), not `test_overlap.py`/`tolerances.toml`, so they aren't
subject to the "stay in sync with `tolerances.toml`" rule above.

| Waveform | Threshold | Status |
|---|---|---|
| ExactPulsarSignal | 1e-10 | machine precision (component-level, see below) |
| PulsarSignal | 1e-10 | machine precision (component-level, see below) |
| BinaryPulsarSignal | 1e-12 | orbital phase only, tight-Kepler regime (see below) |

---

## Waveforms at machine precision

### TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM, IMRPhenomXAS

These waveforms agree with LAL to the limit of float64 arithmetic.
The residual is consistent with floating-point rounding noise in the ET-D noise-weighted inner product, not a real discrepancy.

---

## Waveforms with a known cause

### IMRPhenomPv2

**Threshold: 1e-4**

This large threshold does **not** reflect a deficiency in ripple's implementation.
The overlap loss is almost entirely a time shift, not an amplitude error — the two implementations compute the coalescence-time correction `t0` differently:

- **LAL** computes `t0` via a 10-point natural cubic spline over `[0.8*f_RD, 1.2*f_RD]` (`gsl_interp_cspline`, `LALSimIMRPhenomP.c` lines 1060–1151).
  That grid underresolves the Lorentzian arctan feature in the merger-ringdown phase (characteristic width ~ f_damp ~ 22 Hz), introducing a derivative error.
- **ripple** computes `t0` via exact JAX autodiff, giving the true instantaneous derivative.

ripple's result is the more accurate one; the LAL comparison is limited by the coarse spline grid.
Once the resulting linear phase ramp is removed, the residual is consistent with the machine-precision models above.

Two further properties, both convention rather than bugs:

- **Continuity at sx = 0**: both ripple and LAL are continuous as in-plane spin → 0.
- **PhenomPv2 vs PhenomD at zero in-plane spin**: setting `sx=sy=0` in PhenomPv2 does not recover PhenomD.
  `gen_IMRPhenomPv2` internally swaps `m1 ↔ m2` to follow the LAL convention (`m1 < m2`), which re-assigns `chi1`/`chi2` to the opposite mass component, so the asymmetric PN phase terms differ from PhenomD's assignment.
  Both LAL and ripple exhibit this identically.

### IMRPhenomXHM

**Threshold: 1e-6**

The deviation is confined to extreme mass ratio (q ~ 0.07) and near-extremal primary spin (chi1 ~ 0.98), where the (3,2) mode exhibits strong spheroidal-spherical mixing near the ringdown.

The (3,2) intermediate phase is set by a 6×6 linear system, one of whose constraints is the first derivative of the spheroidal-to-spherical (S2S) phase at `fcutRD`.
That derivative disagrees slightly between ripple and LAL: the S2S phase oscillates rapidly near `fcutRD` (a beat between the (3,2) and (2,2) QNMs), so small differences in how JAX vs. C sample a rapidly oscillating function's instantaneous slope produce a small phase error in the 200–300 Hz intermediate region.

### IMRPhenomXP

**Threshold: 1e-6**

The dominant contribution comes from parameter combinations where the MSA precession correction has a near-singularity in the sensitive band.
In `IMRPhenomX_Return_phiz_of_v_MSA_precav_correction_LAL`, the correction formula contains `1/sqrt(d0+d2+d4)`, which diverges when `d0+d2+d4 = 0` — the angular-momentum resonance `J = L − Smi`.
Near that resonance, `d0+d2+d4` is a difference of large, nearly-equal terms (catastrophic cancellation), so GPU/CPU float64 rounding differences in the spin-evolution coefficients get amplified into a real, if small, discrepancy in the precession angle — further amplified in `hc` vs `hp` at high inclination.

This is a fundamental float64 limitation: both ripple and LAL use the same formula, and the error arises from irreducible GPU/CPU rounding differences near the resonance, not from either implementation being wrong.

### IMRPhenomXPHM

**Threshold: 1e-6**

Dominated by the same MSA resonance mechanism as XP.
Beyond that, high aligned-spin cases pick up a small additional contribution from the (3,2) mode (see IMRPhenomXHM above), diluted by the other modes.

---

## Waveforms with a resolved history

### IMRPhenomXAS_NRTidalv3

**Threshold: 1e-12**

This model previously carried a 1e-6 holding threshold under a documented high-frequency phase discrepancy that grew with total mass.
The threshold has since been tightened to 1e-12 and is no longer under active investigation.

---

## Continuous-wave (CW) models

LAL does not SWIG-wrap `PulsarSignalParams` (it has anonymous nested structs), so
`XLALSimulateExactPulsarSignal`/`XLALGeneratePulsarSignal` cannot be called directly from
Python, and CW's calling convention (a fixed detector + ephemeris + GPS epoch at
construction, a time axis, not a frequency grid) doesn't fit the batched
`ReferenceBackend`/`tolerances.toml` campaign the CBC models above use. CW agreement is
instead checked by `tests/cross_validation/test_cw_exact_pulsar.py` (`accuracy`-marked,
skipped without `lalpulsar` + an Earth/Sun ephemeris file), which reproduces LAL's own
reference computation in Python from its SWIG-exposed building blocks
(`XLALGetDetectorStates`, `XLALComputeAMCoeffs`, `XLALBarycenter`, `XLALGenerateSpinOrbitCW`):

- **`ExactPulsarSignal`** — reconstructed detector strain vs. a Python transcription of
  `SimulatePulsarSignal.c`'s exact (Roemer-only) path: overlap loss < 1e-10 (observed ~3e-13).
  The geometric delay alone agrees with `XLALBarycenter` to << 1 microsecond.
- **`PulsarSignal`** — same check using the full barycentering delay (Roemer + Earth-rotation
  with precession/nutation + Einstein − Shapiro), each sample built from `XLALBarycenter`
  directly: overlap loss < 1e-10 (observed ~1e-9).
- **`BinaryPulsarSignal`** — only the orbital source-phase model is checked automatically, against
  `XLALGenerateSpinOrbitCW` in the tight-Kepler regime (`f0=1000 Hz`): overlap loss < 1e-12.
  The full binary waveform end-to-end is not yet part of the automated suite (see below).

In all three cases the residual tracks **LAL's own reference precision, not ripple's** — the
exact/full floors come from LAL evaluating phase in REAL8 GPS time (`t ≈ 1e9`, resolving
~0.1 microsecond); ripple's int+frac GPS split is more precise than that floor.

### Supplementary: compiled-function comparison (`c_harness/`)

`tests/cross_validation/c_harness/` additionally compares against the actual *compiled*
`XLALSimulateExactPulsarSignal`/`XLALGeneratePulsarSignal` entry points (not just the
SWIG-exposed building blocks above), by declaring the unwrapped struct layout in C and calling
them directly. This is a manual, environment-dependent recipe (see its `README.md`), not
collected as a pytest test, run once to corroborate the automated result above:

| Comparison | log10 overlap loss |
|---|---|
| `ExactPulsarSignal` vs. compiled `XLALSimulateExactPulsarSignal` | −12.46 |
| `PulsarSignal` vs. compiled `XLALGeneratePulsarSignal` (fHet = 0) | −12.00 |
| `PulsarSignal` vs. compiled `XLALGeneratePulsarSignal` (fHet = 12 Hz) | −12.00 |

A 400-point random parameter sweep (`make_figs.py`, H1, `f0 ∈ [10, 500] Hz`) against the same
compiled functions gives median/max log10 overlap loss of −9.2/−7.9 (`ExactPulsarSignal`),
−11.3/−10.5 (`PulsarSignal`), and −5.5/−4.8 (`BinaryPulsarSignal`, full end-to-end incl. orbital
modulation). The binary floor is set by LAL `GenerateSpinOrbitCW`'s own Kepler solver tolerance
(`dxMax = 0.01/(f0·P)`); ripple solves Kepler to machine precision, so this is LAL's limitation,
not ripple's — at `f0 = 1000 Hz` (where LAL's tolerance is tightest) the two agree to
log10(1−O) ≈ −15.2.

---

## Notes

- A different random seed or parameter range may give different extreme values; thresholds carry margin over what's been observed to allow for that variation.
- OL is not maximised over time or phase shifts — it is the raw, un-maximised overlap between ripple and LAL waveforms with identical input parameters.
- IMRPhenomXAS_NRTidalv3's BNS-band frequency grid can exceed available host memory at large `--n-samples` on GPU; the OOM-retry ladder in `campaign.py` only handles GPU/JAX OOM, not host-side array allocation.
