# IMRPhenomXAS_NRTidalv3 Overlap Investigation – Handover Notes

**Date:** 2026-05-22  
**Goal:** Reduce overlap loss between ripple and LALSuite for `IMRPhenomXAS_NRTidalv3` from ~6.55e-9 down to <1e-12 (machine precision).  
**Test harness:** `tests/cross_validation/test_lal_overlap.py` with the ET PSD (`tests/psds/ET_D_psd.txt`), T=128 s, f_l=20 Hz, f_u=4096 Hz, f_ref=20 Hz, tc=0, 10 random samples (seed=42).  
**Current threshold in the test:** `1e-6` (should be tightened to `1e-12` once the fix is found).

---

## Baseline results (before any changes to this branch)

```
Mean overlap loss:    6.55e-09  (log10 ≈ -8.2)
Median overlap loss:  3.26e-09
Min overlap loss:     2.03e-10
Max overlap loss:     2.83e-08   ← sample 0 (m1=2.92, m2=2.88)
```

Full per-sample data (saved by the test):
`tests/cross_validation/results/n10_T128/overlap_loss_IMRPhenomXAS_NRTidalv3.csv`

---

## Key facts established during investigation

### 1. Tidal phase functions are numerically equivalent (ripple ↔ LAL)

- `phenomx_tidal_phase(θ, Mf)` ≡ `IMRPhenomX_TidalPhase(Mf, ...)` in LAL ✓
- `phenomx_tidal_phase_derivative(θ, Mf)` ≡ `IMRPhenomX_TidalPhaseDerivative` ✓
- `get_tidal_phase(x, ...)` ≡ LAL's `SimNRTunedTidesFDTidalPhase_v3` ✓
- `_get_merger_frequency(θ)` ≡ `XLALSimNRTunedTidesMergerFrequency_v3` ✓
- `phenomx_tidal_phase` and the waveform-loop tidal phase `psi_T + psi_QM + psi_SS` agree to machine precision (verified in `scripts/compare_tidal_phase_only.py`)

### 2. f_merger falls in the **inspiral** phase region for all 10 test cases

`scripts/check_boundaries.py` and `scripts/check_phase_at_merger.py` established:
- For all 10 samples, `Mf_merger < f1_Ms` (the inspiral-intermediate boundary)
- `f1_Ms = fMECO - 0.03*(fIMmatch - fMECO)` where fMECO = `get_cutoff_fMs(...)[2]`
- f_merger (and hence f_final = f_merger) is always in the **inspiral** region

### 3. Overlap loss scales strongly with total mass

| m_total | f_merger (Hz) | linb_error (analytic−secant) | overlap loss |
|---------|--------------|------------------------------|--------------|
| 5.80    | 482.6         | −3.49e-2                     | 2.83e-8      |
| 4.91    | 594.0         | −2.51e-2                     | 1.69e-8      |
| 4.48    | 598.4         | −3.35e-2                     | 8.22e-9      |
| 3.58    | 860.2         | −1.52e-2                     | 4.44e-9      |
| 1.84–1.91 | 1354–1770  | −6e-3 to −18e-3              | 2–3e-10      |

---

## What the alignment code does (ripple vs LAL)

### Ripple (`src/ripplegw/waveforms/IMRPhenomXAS_NRTidalv3.py`)

```python
# Step 1: compute BBH alignment constants (same as XAS)
lina, linb, psi4tostrain = calc_phaseatpeak(eta, StotR, chia, delta)
dphi22Ref = jax.grad(Phase)((fMs_RD - fMs_damp) / M_s, ...) / M_s
linb = linb - dphi22Ref - 2*PI*(500 + psi4tostrain)   # = linb_BBH

# Step 2: compute dphiXAS at f_final = min(f[-1]+df, f_merger)
dphiXAS = jax.lax.cond(
    df > 0.0,
    lambda _: (Phase(f_final,...) - Phase(f_final-df,...)) / (df * M_s),  # ← backward secant
    lambda _: PhaseDerivative(f_final,...) / M_s,
    operand=None,
)

# Step 3: update linb so that dlinb_total/dMf matches dphiT at f_final
linb = linb - (dphiXAS + linb - dphiT)   # ≡ linb_tidal = dphiT - dphiXAS

# Step 4: compute phifRef
phiTfRef = phenomx_tidal_phase(theta_intrinsic, f_ref * M_s)
phifRef = -(Phase(f_ref,...) + linb*(f_ref*M_s) + lina - phiTfRef) + PI/4 + PI
```

### LAL (`LALSimIMRPhenomX.c`, approximate)

```c
// Step 3 equivalent (LAL uses the analytic derivative):
dphi_fmerger = (1/eta) * IMRPhenomX_dPhase_22(Mf_final, ...) + linb_BBH
              - IMRPhenomX_TidalPhaseDerivative(Mf_final, ...);
linb += -dphi_fmerger;
// i.e. linb_LAL = dphiT_LAL - (1/eta)*IMRPhenomX_dPhase_22(Mf_final)
```

---

## The backward-secant hypothesis (partially verified, then contradicted)

For m_total=5.8 (f_merger=483 Hz):

| Method | dphiXAS value | diff vs secant |
|--------|--------------|----------------|
| backward secant (ripple current) | 1193.138 | 0 |
| `PhaseDerivative(f_final)/M_s` (analytic) | 1193.103 | −3.49e-2 |
| `jax.grad(Phase)(f_final)/M_s` (auto-diff) | 1193.103 | −3.49e-2 |
| central secant | 1193.103 | −3.49e-2 |

The backward secant overshoots by ~0.035 rad/Mf relative to the analytic derivative.

**Hypothesis:** This linb_error = (analytic − secant) ≈ −0.035 creates a linear phase error  
`ΔΦ(f) ≈ linb_error × (Mf − Mf_ref)` that grows to ~4.6e-4 rad at f_ref=20 Hz for the heaviest system.

**Test:** Replace backward secant with `PhaseDerivative(f_final,...)/M_s`.

**Result: overlap loss INCREASED (from 2.83e-8 to 3.85e-8 for the worst case).**

This means `PhaseDerivative/M_s` is NOT what LAL uses, OR there is another dominant error source.

---

## LAL dPhase22 extraction attempt

For m_total=5.8 (f_merger=483 Hz), extracting from the XAS LAL waveform:
- `d(angle(h_xas))/df at f_merger` ≈ 1491.9 / M_s
- This equals `(1/eta)*dPhase22/dMf + linb_BBH`, where `linb_BBH ≈ 299`
- So `(1/eta)*dPhase22/dMf` = 1491.9 − 299 ≈ 1192.9 ≈ ripple's `PhaseDerivative/M_s`

This suggests `PhaseDerivative` DOES match LAL's `IMRPhenomX_dPhase_22`. Yet using it made things worse.

---

## Open questions / what needs investigation

1. **Why does using `PhaseDerivative` (analytic, matching LAL) give WORSE overlap?**  
   Possible explanation: `PhaseDerivative` deviates from the backward secant by only 0.035, but there is another much larger error source making the total alignment worse.

2. **Is there a large-scale phase mismatch between ripple and LAL at the waveform level?**  
   `scripts/measure_phase_diff.py` shows an ~83 rad phase difference between ripple and LAL tidal waveforms for m_total=5.8, but this script may be comparing waveforms with different frequency extents (ripple has `A_P` Planck-tapering amplitude to 0 above f_merger; LAL may not zero the amplitude there). The script needs fixing to properly handle the frequency-extent difference.

3. **What is the actual frequency range where the overlap integral has weight?**  
   For m_total=5.8, both ripple and LAL should only have appreciable amplitude below ~575 Hz. The 71065 nonzero bins in ripple vs 289807 in LAL (from `measure_phase_diff.py`) suggests LAL has residual amplitude above f_merger.

4. **Is there a factor-of-2 or sign error in `phifRef` or `ext_phase_contrib`?**  
   The ext_phase_contrib = `2*PI*f*tc + 2*phi_c`. For tc=0, this is `2*phi_ref`. This needs checking against LAL's phase convention.

5. **Does the `changePhase_if_min` correction matter?**  
   For m_total=5.8, it activates at f ≈ 3828 Hz (well above f_merger and above the 1.35*f_merger taper endpoint). Since P_P = 1 there, the NRTidalv3_phase is already irrelevant (full PN taper), so this should not matter.

---

## Files created during investigation

| File | Purpose |
|------|---------|
| `scripts/check_boundaries.py` | Checks which IMRPhenomX phase region f_merger falls in |
| `scripts/check_phase_at_merger.py` | Computes backward-secant vs analytic dphiXAS for all test samples |
| `scripts/compare_tidal_phase_only.py` | Verifies phenomx_tidal_phase vs waveform-loop tidal phase |
| `scripts/diagnose_tidal_phase.py` | Full ripple vs LAL waveform phase comparison (heavy, slow to run) |
| `scripts/diagnose_tidal_phase_fast.py` | Lighter version of the above |
| `scripts/measure_phase_diff.py` | Direct phase-unwrap comparison for m_total=5.8 case (has frequency-extent bug) |
| `scripts/quick_diag.py` | Early diagnostic for dphiXAS at f_merger |
| `scripts/quick_diag2.py` | Confirmed dphiXAS_rip ≈ dphiXAS_LAL = 148.40 for m1=m2=1.4 |

---

## Current state of the code

The `PhaseDerivative`-based fix was applied to `src/ripplegw/waveforms/IMRPhenomXAS_NRTidalv3.py` but made things **worse**. **The change should be reverted** before continuing:

```python
# In _gen_IMRPhenomXAS_NRTidalv3, around line 120, REVERT to:
dphiXAS = jax.lax.cond(
    df > 0.0,
    lambda _: (
        Phase(f_final, theta_intrinsic[:4], bbh_phase_coeffs)
        - Phase(f_final - df, theta_intrinsic[:4], bbh_phase_coeffs)
    ) / (df * M_s),
    lambda _: PhaseDerivative(f_final, theta_intrinsic[:4], bbh_phase_coeffs) / M_s,
    operand=None,
)
```

---

## Suggested next steps

1. **Revert the failed PhaseDerivative fix** (restore backward secant).

2. **Fix `scripts/measure_phase_diff.py`** to properly compare waveforms:
   - Zero out both waveforms above `1.35 * f_merger * (1 + epsilon)` before comparing
   - Use proper nyquist mask as in the test
   - Verify the phase difference is truly small (< 1e-4 rad) in the overlap-relevant band

3. **Isolate the dominant error by computing:**
   - The total ripple tidal waveform phase at each frequency
   - The total LAL tidal waveform phase at each frequency
   - Their difference, after removing any constant offset
   - Whether this difference is dominated by a linear term (linb error) or a nonlinear term

4. **Check `IMRPhenomX_dPhase_22` source in LAL** to see if it uses a different formula:
   - It might use a C2-corrected derivative near the boundary
   - Or it might account for the linb term differently
   - LAL source: `lalsimulation/lib/LALSimIMRPhenomX_internals.c` (function `IMRPhenomX_dPhase_22`)

5. **Consider alternative fix: use `PhaseDerivative` at `f_final - df/2` (midpoint)**  
   The central secant `(Phase(f_final+df/2) - Phase(f_final-df/2)) / (df*M_s)` matches the analytic derivative. But this shifts f_final by half a bin, which might not be what LAL does.

6. **Check if LAL uses `freqs->data[freqs->length-1]` as f_final and whether that equals the exact last frequency bin or `f_max`.**

---

## Summary table of waveform function matching status

| Function (ripple) | Matches LAL? | Notes |
|---|---|---|
| `phenomx_tidal_phase` | ✓ Algebraically verified | Used for phiTfRef alignment |
| `phenomx_tidal_phase_derivative` | ✓ Algebraically verified | Used for dphiT alignment |
| `get_tidal_phase` | ✓ Algebraically verified | Core NRTidalv3 Padé formula |
| `_get_merger_frequency` | ✓ Algebraically verified | |
| `general_planck_taper` | ✓ Algebraically verified | |
| `IMRPhenomXAS.Phase` | ✓ Numerically verified (XAS overlap 1e-15) | |
| `IMRPhenomXAS.PhaseDerivative` | ≈ matches `jax.grad(Phase)` | Unclear if it matches LAL's `IMRPhenomX_dPhase_22` exactly |
| dphiXAS (backward secant) | ≈ matches analytic to 3.5e-2 | Secant gives slightly BETTER overlap than analytic! |
