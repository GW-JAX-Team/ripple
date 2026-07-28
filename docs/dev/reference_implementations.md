# Reference Implementations and Overlap Loss Thresholds

This document records ripple's agreement with external reference implementations. It currently covers LALSuite: the accuracy threshold enforced by `tests/cross_validation/fd/test_overlap.py`, and — for any waveform whose threshold is looser than machine precision — why.

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

The continuous-wave (CW) models below are validated by a different mechanism (see [below](#continuous-wave-cw-models)), not `test_overlap.py`/`tolerances.toml`, so they aren't subject to the "stay in sync with `tolerances.toml`" rule above.

| Waveform | Threshold | Status |
| --- | --- | --- |
| ExactPulsarSignal | 1e-10 | machine precision |
| PulsarSignal | 1e-10 | machine precision |
| BinaryPulsarSignal | 1e-12 | orbital phase only, tight-Kepler regime |
| PulsarSignal vs `CWMakeFakeData` | `1e-4 + 3e-4·(f0/100Hz)²` | frequency-dependent, see below |
| BinaryPulsarSignal vs `CWMakeFakeData` | `2e-3 + 4e-4·(f0/100Hz)²` | frequency-dependent, see below |

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

## Continuous-wave (CW) models

LAL does not SWIG-wrap `PulsarSignalParams` (it has anonymous nested structs), so `XLALSimulateExactPulsarSignal`/`XLALGeneratePulsarSignal` cannot be called directly from Python, and CW's calling convention (an ephemeris + GPS epoch fixed at construction, plus a per-call site location and a time axis rather than a frequency grid) doesn't fit the batched `ReferenceBackend`/`tolerances.toml` campaign the frequency-domain models above use.
CW agreement is checked by `tests/cross_validation/cw/` (`accuracy`-marked, skipped without `lalpulsar` + an Earth/Sun ephemeris file), by two independent methods that deliberately do not overlap:

**Building-block reconstruction** — one file per registered class (`test_exact_pulsar_signal.py`, `test_full_pulsar_signal.py`, `test_binary_pulsar_signal.py`), each reproducing LAL's own reference computation in Python from its SWIG-exposed building blocks (`XLALGetDetectorStates`, `XLALComputeAMCoeffs`, `XLALBarycenter`, `XLALGenerateSpinOrbitCW`) -- a translation-fidelity check against the same low-level routines ripple's own `barycenter.py`/`earth.py` were ported from:

- **`ExactPulsarSignal`** — reconstructed detector strain vs. a Python transcription of `SimulatePulsarSignal.c`'s exact (Roemer-only) path: overlap loss < 1e-10 (observed ~3e-13).
  The geometric delay alone agrees with `XLALBarycenter` to << 1 microsecond.
- **`PulsarSignal`** — same check using the full barycentering delay (Roemer + Earth-rotation with precession/nutation + Einstein − Shapiro), each sample built from `XLALBarycenter` directly: overlap loss < 1e-10 (observed ~1e-9).
- **`BinaryPulsarSignal`** — only the orbital source-phase model is checked here, against `XLALGenerateSpinOrbitCW` in the tight-Kepler regime (`f0=1000 Hz`): overlap loss < 1e-12. (The full end-to-end binary waveform is covered by the independent method below instead.)

In all three cases the residual tracks **LAL's own reference precision, not ripple's** — the exact/full floors come from LAL evaluating phase in REAL8 GPS time (`t ≈ 1e9`, resolving ~0.1 microsecond); ripple's int+frac GPS split is more precise than that floor.

**`lalpulsar_Makefakedata_v5` engine** — `test_makefakedata_v5.py` instead drives `XLALCWMakeFakeData` (the literal engine behind the `lalpulsar_Makefakedata_v5` CLI real CW searches use for injections/MDCs), via its SWIG-wrapped "modern" `PulsarParams`/`CWMFDataParams` structs. `XLALCWMakeFakeData` is a thin wrapper around `XLALGeneratePulsarSignal` (the same function the anonymous-nested-struct `PulsarSignalParams` above blocks direct SWIG access to), so this is a genuinely independent code path from the building-block tests. Covers `PulsarSignal` and `BinaryPulsarSignal` (both use the full barycentering delay this pipeline implements); not `ExactPulsarSignal`, since LAL has no toggle to disable the Einstein/Shapiro terms here.

LALPulsar's other Python-native option, `lalpulsar.simulateCW.CWSimulator`, was tried first and rejected: it reaches only ~1e-3 due to its own internal interpolation, too loose to be useful here.

This `Makefakedata_v5`-based check is also the first automated, in-repo, end-to-end validation of `BinaryPulsarSignal`'s full waveform (barycentering + orbital modulation + antenna response combined) — previously this was only checked manually, off-repo, against the compiled `XLALGeneratePulsarSignal` entry point directly (all three models agreed to log10 overlap loss better than −5, limited by LAL's own Kepler solver tolerance for `BinaryPulsarSignal`, not ripple's). `CWMakeFakeData`'s SWIG-friendly wrapper structs make that same check reproducible from Python alone, in-tree, in CI.

**The overlap loss against this pipeline is not a flat floor — it scales as roughly f0².** The fixed single-point regression test above (f0=12.3 Hz) originally reported this as a flat ~1e-7 `REAL4` floor. An HPC-scale Monte Carlo campaign (`test_makefakedata_v5_campaign.py`, sky position/amplitude/f0/spindown/site/orbital-element sweep, f0 log-uniform 10–2000 Hz — the typical CW all-sky search band) instead found:

- Overlap loss vs. `f0` fits `loss(f0) ≈ C·(f0/100 Hz)^p` with `p ≈ 1.9` for isolated pulsars and `p ≈ 1.5` for binaries (500-trial run, 2026-07-28), reaching up to ~3e-3 (isolated) and ~6e-3 (binary) at f0 ≈ 2000 Hz — three to four orders of magnitude looser than the single-point f0=12.3 measurement suggested.
- Traced to source: `CWMakeFakeData.c` → `XLALGenerateCWSignalTS` → `XLALGeneratePulsarSignal` → `XLALPulsarSimulateCoherentGW`, whose source phase table comes from `XLALGenerateSpinOrbitCW` using plain REAL8 GPS-second-scale time representations (~1e9 magnitude) rather than ripple's own int+frac split — the same class of floor already noted above for the building-block tests, just far more prominent in this code path.
- Four differential experiments confirm this is a genuine LAL-side effect, not a ripple bug or test artifact: the loss is invariant to `Band`/sampling rate and to `sourceDeltaT` (60s down to 0.01s — ruling out phase-table interpolation error), tracks the pulsar's absolute f0 rather than the heterodyne/residual frequency (tested by decoupling fHet from f0: identical loss whether fHet≈f0 or fHet≪f0), and is *not* reproduced by casting ripple's own float64 output through a float32 round-trip (~1e-16, flat in f0 — twelve orders of magnitude too small to explain the observed effect).
- `_threshold()` in `test_makefakedata_v5.py` is a floor-plus-power-law fit calibrated from that 500-trial campaign, with ≥13x (binary) / ≥25x (isolated) margin over every sampled trial. A larger `--n-samples` re-run may want to re-derive these constants rather than assume the same margin holds.

See `tests/cross_validation/cw/campaign.py`'s module docstring for the full writeup, and `test_makefakedata_v5_campaign.py` / `submit_slurm.sh` to reproduce or extend the campaign.

---

## Notes

- A different random seed or parameter range may give different extreme values; thresholds carry margin over what's been observed to allow for that variation.
- OL is not maximised over time or phase shifts — it is the raw, un-maximised overlap between ripple and LAL waveforms with identical input parameters.
- IMRPhenomXAS_NRTidalv3's BNS-band frequency grid can exceed available host memory at large `--n-samples` on GPU; the OOM-retry ladder in `campaign.py` only handles GPU/JAX OOM, not host-side array allocation.
