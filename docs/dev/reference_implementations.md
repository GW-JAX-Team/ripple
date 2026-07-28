# Reference Implementations and Accuracy Bounds

This is the developer record of ripple's agreement with external reference implementations.
It currently covers LALSuite, the thresholds enforced by cross-validation, and the reason for any bound looser than machine precision.
To submit a test, see [Cross-validation tests](cross_validation.md).

The overlap loss (OL), using the ET-D PSD noise weighting, is

$$
\mathrm{OL} = 1 - \frac{\operatorname{Re}\langle h_1 \mid h_2 \rangle}
{\bigl(\langle h_1 \mid h_1 \rangle \langle h_2 \mid h_2 \rangle\bigr)^{1/2}}\,.
$$

A lower OL indicates better agreement.
"Machine precision" means a threshold consistent with floating-point rounding (a few times eps_machine, typically 1e-15 to 1e-16), not a real physical discrepancy.

## Summary table

| Waveform | Threshold | Status |
| --- | --- | --- |
| TaylorF2 | 1e-15 | machine precision |
| IMRPhenomD | 1e-15 | machine precision |
| IMRPhenomD_NRTidalv2 | 1e-15 | machine precision |
| IMRPhenomHM | 1e-15 | machine precision |
| IMRPhenomPv2 | 1e-4 | known cause (LAL-side) |
| IMRPhenomXAS | 1e-15 | machine precision |
| IMRPhenomXAS_NRTidalv3 | 1e-12 | under investigation |
| IMRPhenomXHM | 1e-6 | known cause |
| IMRPhenomXP | 1e-6 | known cause |
| IMRPhenomXPHM | 1e-6 | known cause |

The time-domain models below use waveform-specific direct comparisons, not the frequency-domain tolerance table.

| Waveform | Threshold | Status |
| --- | --- | --- |
| SineGaussian vs LALSimulation | mismatch < 1e-10; relative norm error < 1e-8 | direct centered time series |
| ExactPulsarSignal vs direct LALPulsar building blocks | mismatch < 1e-9; relative norm error < 1e-8 | randomized 512 Hz aligned time series |
| PulsarSignal | 1e-10 | machine precision |
| BinaryPulsarSignal | 1e-12 | orbital phase only, tight-Kepler regime |
| PulsarSignal vs `CWMakeFakeData` | `1e-4 + 3e-4·(f0/100Hz)²` | LAL reference-interpolation bound, see below |
| BinaryPulsarSignal vs `CWMakeFakeData` | `2e-3 + 4e-4·(f0/100Hz)²` | LAL reference-interpolation bound, see below |

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

## Time-domain waveform models

Time-domain validation compares directly generated, aligned real samples.
It never uses an FFT simply to reuse the frequency-domain test setup.
The white normalized mismatch

$$
m = 1 - \frac{h_1 \mathbin{\cdot} h_2}{\lVert h_1 \rVert\lVert h_2 \rVert}
$$

has no whitening or time/phase maximization.
It checks phase and shape but is unchanged by a global positive amplitude scale, so the SineGaussian, ExactPulsarSignal, and `CWMakeFakeData` tests also check a relative-norm amplitude error.

### SineGaussian

`SineGaussian` is compared with `lalsimulation.SimBurstSineGaussian`.
The adapter uses LAL's centered sample convention directly; recreating the axis from a floating-point GPS epoch would lose sub-nanosecond information and introduce an artificial high-frequency phase error.
It compares both polarizations on the reference samples with the FFT-free mismatch and relative-norm diagnostics.

### Continuous-wave (CW) models

CW uses a time axis, an ephemeris, an epoch, and a per-call detector site, so it does not fit the stateless frequency-domain reference-backend interface.
The `CWMakeFakeData` regression and large-scale test use the shared mismatch and relative-norm amplitude diagnostics above.

Two independent LALPulsar comparisons provide coverage:

- **Building-block reconstruction** compares each model with a reference built from SWIG-exposed LAL routines (`XLALGetDetectorStates`, `XLALComputeAMCoeffs`, `XLALBarycenter`, and `XLALGenerateSpinOrbitCW`).
  It covers `ExactPulsarSignal`, `PulsarSignal`, and the binary orbital source phase in its tight-Kepler regime.
- **`CWMakeFakeData`** compares `PulsarSignal` and `BinaryPulsarSignal` with the engine behind `lalpulsar_Makefakedata_v5`, including detector response, barycentering, and orbital modulation.
  `ExactPulsarSignal` is intentionally excluded because this LAL path cannot disable the Einstein/Shapiro terms.

`ExactPulsarSignal` also has a randomized large-scale test built from the first reference path.
It faithfully follows LAL's `XLALGPSGetREAL8` phase convention on a 512 Hz grid whose timestamps are exactly representable in that format, sampling 10–200 Hz without introducing a grid-created phase floor.
It checks the direct FFT-free mismatch and the relative-norm amplitude diagnostic.

### Frequency-dependent `CWMakeFakeData` bound

The `CWMakeFakeData` mismatch is not a flat numerical floor: the large-scale test finds it grows approximately as $f_0^2$.
This is a limitation of the LAL reference path.
`PulsarSimulateCoherentGW.c`, reached through `CWMakeFakeData`, uses a hard-coded 400-second delay-table half interval (800 seconds between tabulated delay values) and linearly interpolates at each output sample.
Its source comments estimate a delay error of order microseconds.
A delay-induced phase error scales with $f_0$, so the normalized mismatch naturally scales approximately as $f_0^2$.

The frequency-dependent bounds above are calibrated to that reference approximation, not evidence that ripple should add the same interpolation.
Recalibrate them after changing the sampled parameter range or LAL version.
Run the selected CW test through [Cross-validation tests](cross_validation.md); it requires LALPulsar and Earth/Sun ephemerides visible to the worker.

---

## Notes

- A different random seed or parameter range may give different extreme values; thresholds carry margin over what's been observed to allow for that variation.
- The frequency-domain overlap loss is not maximised over time or phase shifts; it is the raw comparison between ripple and LAL waveforms with identical input parameters.
- IMRPhenomXAS_NRTidalv3's BNS-band frequency grid can exceed available host memory at large `--n-samples` on GPU; the OOM-retry ladder in the frequency-domain test runner only handles GPU/JAX OOM, not host-side array allocation.
