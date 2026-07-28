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
| ExactPulsarSignal vs direct LALPulsar building blocks | mismatch < `3e-9 + 8e-9·(f0/100Hz)²`; relative norm error < 2e-7 | frequency-dependent, see below |
| PulsarSignal | 1e-10 | machine precision |
| BinaryPulsarSignal | 1e-12 | orbital phase only, tight-Kepler regime |
| PulsarSignal vs `CWMakeFakeData` | mismatch < `1e-4 + 3e-4·(f0/100Hz)²`; relative norm error < 5e-2 | LAL reference-interpolation bound, see below |
| BinaryPulsarSignal vs `CWMakeFakeData` | mismatch < `2e-3 + 4e-4·(f0/100Hz)²`; relative norm error < 5e-2 | LAL reference-interpolation bound, see below |

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

This threshold is deliberately **not** tightened to match a smaller-`n` observed max (see [Large-scale campaign results](#large-scale-campaign-results-2026-07-28)): the extreme-mass-ratio/near-extremal-spin corner responsible is rare enough that the observed max keeps growing with `--n-samples` rather than converging -- a `n=5000` run (2026-07-28) found max 9.75e-06, nearly 10x over this threshold itself.

### IMRPhenomXP

**Threshold: 1e-6**

The dominant contribution comes from parameter combinations where the MSA precession correction has a near-singularity in the sensitive band.
In `IMRPhenomX_Return_phiz_of_v_MSA_precav_correction_LAL`, the correction formula contains `1/sqrt(d0+d2+d4)`, which diverges when `d0+d2+d4 = 0` — the angular-momentum resonance `J = L − Smi`.
Near that resonance, `d0+d2+d4` is a difference of large, nearly-equal terms (catastrophic cancellation), so GPU/CPU float64 rounding differences in the spin-evolution coefficients get amplified into a real, if small, discrepancy in the precession angle — further amplified in `hc` vs `hp` at high inclination.

This is a fundamental float64 limitation: both ripple and LAL use the same formula, and the error arises from irreducible GPU/CPU rounding differences near the resonance, not from either implementation being wrong.

At the same near-extremal-spin resonance, LAL occasionally refuses to generate at all rather than returning a degraded result: a 1000-sample large-scale run (2026-07-28) hit `Internal function call failed: Input domain error` for 1/1000 random draws (`|s2| ~ 0.95`), confirmed via the local LALSuite source to originate in the same `d0+d2+d4`/`sqrt(fabs(...))` MSA quantities as above. This is a benign, expected LAL-side domain rejection near the resonance, not a ripple regression, and unrelated to the overlap-loss threshold itself.

Like XHM, this threshold is deliberately **not** tightened to a smaller-`n` observed max: the same resonance corner makes overlap loss heavy-tailed here too. `n=1000` found max 6.98e-08, but an `n=5000` run (2026-07-28) found max 1.15e-06 -- *exceeding* this threshold. See [Large-scale campaign results](#large-scale-campaign-results-2026-07-28).

### IMRPhenomXPHM

**Threshold: 1e-6**

Dominated by the same MSA resonance mechanism as XP.
Beyond that, high aligned-spin cases pick up a small additional contribution from the (3,2) mode (see IMRPhenomXHM above), diluted by the other modes.

Shares XP's occasional benign LAL domain rejection near the resonance (same 2026-07-28 run: 1/1000, max overlap loss among generated samples 8.24e-07).

Given the shared mechanism, this threshold is also kept unrounded rather than tightened from the `n=1000` max: an `n=3000` recheck (2026-07-28) found max 2.26e-07, comfortably under 1e-6, but XHM's and XP's own larger-`n` reruns showed this family's tail isn't reliably characterized by any single run of this size -- see [Large-scale campaign results](#large-scale-campaign-results-2026-07-28).

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
Like `CWMakeFakeData` below, its mismatch is not a flat floor either: a 1000-trial run (2026-07-28) found a clean $f_0^2$ scaling (log-log fit exponent ~1.97), just ~100x tighter in absolute terms (~1e-9 vs ~1e-6) since this path uses LAL's own `XLALGPSGetREAL8`/`XLALBarycenter` REAL8-precision routines directly rather than the `REAL4`-tabulated `CWMakeFakeData` pipeline. The relative norm error showed no comparable $f_0$ trend (log-log correlation ~0.13).

### Frequency-dependent `CWMakeFakeData` bound

The `CWMakeFakeData` mismatch is not a flat numerical floor: the large-scale test finds it grows approximately as $f_0^2$.
This is a limitation of the LAL reference path.
`PulsarSimulateCoherentGW.c`, reached through `CWMakeFakeData`, uses a hard-coded 400-second delay-table half interval (800 seconds between tabulated delay values) and linearly interpolates at each output sample.
Its source comments estimate a delay error of order microseconds.
A delay-induced phase error scales with $f_0$, so the normalized mismatch naturally scales approximately as $f_0^2$.

The relative-norm-error (amplitude-scale) diagnostic against `CWMakeFakeData` has a *separate* LAL-side floor, unrelated to $f_0$: a 1000-trial `PulsarSignal` run (2026-07-28) found essentially no $f_0$ correlation (log-log correlation ~-0.01) but roughly 2x higher error at short trial duration (≤100s) than long (≥1000s), plus occasional larger outliers (observed max 1.42e-2) from specific sky-position/polarization-angle configurations. The same `PulsarSimulateCoherentGW.c` path tabulates the detector antenna response (`LALComputeDetAMResponseSeries`) at a hard-coded `dtPolBy2 = 300`-second half interval (600 seconds between `REAL4`-precision table nodes), linearly interpolated — coarser for short trials, which span fewer table nodes — unlike ripple's own comparison strain, which calls `ComputeDetAMResponse` fresh at every sample (`_lal_helpers.detector_strain_from_am_response`). This is why the large-scale test's norm-error ceiling is a flat bound, not an $f_0$-scaled one, unlike the mismatch.

The frequency-dependent bounds above are calibrated to that reference approximation, not evidence that ripple should add the same interpolation.
Recalibrate them after changing the sampled parameter range or LAL version.
Run the selected CW test through [Cross-validation tests](cross_validation.md); it requires LALPulsar and Earth/Sun ephemerides visible to the worker.

---

## Notes

- A different random seed or parameter range may give different extreme values; thresholds carry margin over what's been observed to allow for that variation.
- The frequency-domain overlap loss is not maximised over time or phase shifts; it is the raw comparison between ripple and LAL waveforms with identical input parameters.
- A BNS-band frequency grid can exceed available host memory at large `--n-samples` on GPU; the OOM-retry ladder in the frequency-domain test runner only handles GPU/JAX OOM, not host-side array allocation. This isn't unique to IMRPhenomXAS_NRTidalv3: a 2026-07-28 `--n-samples 1000` run with `tests.cross_validation.submit`'s default `--memory 16G` also OOM-killed IMRPhenomD_NRTidalv2 and, unexpectedly, the non-BNS TaylorF2 -- all three passed cleanly once resubmitted with `--memory 64G`. Consider raising the frequency-domain default in `tests/cross_validation/submit.py` rather than relying on a per-run override.

---

## Large-scale campaign results (2026-07-28)

A full `--n-samples 1000 --plots` run of every registered waveform (`python -m tests.cross_validation.submit --scheduler slurm --waveform all`) was run on Slurm; results and plots are stored under `accuracy-results/20260728-192604/<waveform>/` (one `*_n1000.json` + `*_n1000.png` per waveform; the bulky `--cache-reference` raw-array caches were deleted afterward to stay under the cluster's disk quota -- rerun with `--cache-reference` locally if you need to regenerate them). `IMRPhenomD_NRTidalv2`, `IMRPhenomXAS_NRTidalv3`, and `TaylorF2` needed a `--memory 64G` rerun (the frequency-domain launcher's 16G default OOM-killed them at `n=1000`, unrelated to accuracy); `ExactPulsarSignal` and `PulsarSignal` needed the threshold fixes below and were rerun to confirm.

Every threshold in this document and in `tolerances.toml` is "just above" the max observed in this campaign, rounded up to the nearest power of ten (a `1e-N` value) -- **except** `IMRPhenomXHM`/`IMRPhenomXP`/`IMRPhenomXPHM`'s `overlap_loss`, kept at the pre-existing `1e-6` for reasons explained below.

| Waveform | Threshold | Max observed (n=1000) | Notes |
| --- | --- | --- | --- |
| TaylorF2 | overlap_loss 1e-15; phase_offset 1e-10 | 3.93e-16; 4.50e-11 | — |
| IMRPhenomD | overlap_loss 1e-15; phase_offset 1e-11 | 4.44e-16; 7.31e-12 | — |
| IMRPhenomD_NRTidalv2 | overlap_loss 1e-15; phase_offset 1e-9 | 3.82e-16; 1.00e-10 | needed `--memory 64G` |
| IMRPhenomHM | overlap_loss 1e-15; phase_offset 1e-11 | 4.39e-16; 7.51e-12 | — |
| IMRPhenomPv2 | overlap_loss 1e-4 | 3.21e-05 | — |
| IMRPhenomXAS | overlap_loss 1e-15; phase_offset 1e-12 | 4.12e-16; 7.49e-13 | — |
| IMRPhenomXAS_NRTidalv3 | overlap_loss 1e-12; phase_offset 1e-7 | 3.57e-13; 1.94e-8 | needed `--memory 64G` |
| IMRPhenomXHM | overlap_loss **1e-6** (unrounded, see below) | 5.29e-08 (n=1000); **9.75e-06 (n=5000)** | heavy-tailed -- see below |
| IMRPhenomXP | overlap_loss **1e-6** (unrounded, see below) | 6.98e-08 (n=1000); **1.15e-06 (n=5000)** | heavy-tailed; 1/1000 benign LAL domain rejection |
| IMRPhenomXPHM | overlap_loss **1e-6** (unrounded, see below) | 8.24e-07 (n=1000); 2.26e-07 (n=3000) | 1/1000 benign LAL domain rejection |
| SineGaussian | mismatch 1e-17; norm err 1e-15 | 2.42e-18; 8.88e-16 | — |
| ExactPulsarSignal | mismatch `1e-9·(f0/100)²`; norm err 1e-7 | 2.43e-09 at f0=191Hz; 1.58e-8 | — |
| PulsarSignal vs `CWMakeFakeData` | mismatch `1e-3·(f0/100)²`; norm err 1e-1 | 5.10e-03; 1.42e-02 | — |
| BinaryPulsarSignal vs `CWMakeFakeData` | mismatch `1e-2·(f0/100)²`; norm err 1e-1 | 1.20e-02; 6.17e-03 | — |

This run also calibrated every frequency-dependent CW bound and the `ExactPulsarSignal` threshold; it superseded smaller earlier CW calibration runs (n=300-500) that hadn't sampled enough of the parameter space to catch the true floor. The CW/`ExactPulsarSignal` mismatch thresholds turned out to need no additive floor term at all -- a pure `C·(f0/100Hz)²` stays above every sampled trial -- so the earlier floor-plus-power-law formulas were simplified.

**Why `IMRPhenomXHM`/`IMRPhenomXP`/`IMRPhenomXPHM` are the exception.** "Round the n=1000 max up to the nearest power of ten" assumes the statistic is reasonably well-behaved -- true for the smooth f0² power laws above, but not for these three. Their overlap loss is dominated by a rare, narrow parameter-space corner (extreme mass ratio, near-extremal spin -- see their sections above), so the observed max keeps growing as `--n-samples` grows rather than converging. Follow-up runs the same day confirmed this directly: at `n=5000`, `IMRPhenomXHM`'s max grew to 9.75e-06 and `IMRPhenomXP`'s to 1.15e-06 -- both *exceeding* the pre-existing `1e-6` threshold, not just the naively-rounded `1e-7` an n=1000-only calibration would have produced. (`IMRPhenomXPHM`'s own `n=3000` recheck stayed under `1e-6`, but given the shared mechanism and the other two's behavior, that's not enough to conclude it's safe rounded down either.) A robust threshold for this family needs a dedicated, much larger campaign -- and possibly a different methodology (e.g. explicitly characterizing or excluding the bad corner) -- rather than a quick re-derivation, so all three keep their long-standing `1e-6` unrounded.
