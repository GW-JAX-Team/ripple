# The MSA Precession Instability

This note explains the numerical mechanism behind the "known cause (LAL MSA conditioning)" entry for `IMRPhenomXP_NRTidalv3` in [Reference Comparisons and Limits](reference_comparisons_and_limits.md), and why it appears as an overlap-loss *tail* at low spin and small opening angle rather than as a uniform error.
A separate equal-mass singularity is noted at the end.

Equation numbers refer to Chatziioannou, Klein, Yunes & Cornish, *PRD 95, 104004 (2017)*, arXiv:1703.03967 ("the MSA paper"), which `IMRPhenomX_Initialize_MSA_System` (`src/ripplegw/waveforms/cbc/IMRPhenomX/initialize_MSA_system.py`) implements.

## 1. S, L, and the S² cubic

MSA ("multi-scale analysis") splits binary precession into orbital, precession, and radiation-reaction timescales.
The quantities that matter here are the orbital angular momentum **L**, the total spin **S = S1 + S2**, and their norms `L0norm`, `J0norm` (with **J = L + S**), and `S_0_norm`, all built at `f_ref` exactly as LAL builds them.

Over one precession cycle `S² = |S|²` oscillates between two turning points `Smi2` and `Spl2`, the two smaller roots of a monic cubic in `S²` (Eq. B2–B4, `IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA`).
The cubic is solved in depressed form with the trigonometric three-real-roots formula (`IMRPhenomX_Return_Roots_MSA`):

```
p  = C - B²/3
qc = (2/27) B³ - B C/3 + D
acosarg = 1.5 qc / (p sqrt(-p/3))        # in [-1, 1] for three real roots
theta   = arccos(acosarg) / 3
root_k  = 2 sqrt(-p/3) cos(theta - 2πk/3) - B/3,   k = 0, 1, 2
```

The formula fixes no positional order for the three roots, so the code sorts them and identifies the colliding pair generically, by comparing the gap above and below the middle root.
It then keeps the well-separated isolated root directly and **reconstructs the colliding pair from Vieta's formulas on the deflated quadratic** `(x³ + Bx² + Cx + D) / (x - r_iso)`, rather than differencing two nearly-equal trigonometric values.
This keeps ripple's own `Smi2`/`Spl2` self-consistent even when the pair is numerically degenerate.

### 1.1 Where the double root comes from

The precession cone opening angle is set by `Spl2 - Smi2`.
When the total spin is small or **L** and **S** are nearly parallel — a weakly precessing binary — the two turning points are close, and in the exact aligned-spin limit they coincide: a double root of the cubic at `acosarg = -1`.

`arccos` has an infinite derivative there:

```
arccos(-1 + e) = π - sqrt(2e) + O(e^1.5)
```

So as `acosarg → -1` an input perturbation `δ(acosarg)` produces an unbounded `δ(theta) ~ δ(acosarg) / sqrt(2(1 + acosarg))`, and hence unbounded perturbations in the colliding pair, while the isolated root stays accurate.
Across the BNS prior, low spins put `acosarg` close to `-1` for a large fraction of samples, so this is the *typical* configuration there rather than a corner case; the BBH prior reaches it only rarely.

## 2. `c_1`, `cp`, `cm`, and why `cm` is the danger point

`c_1` (Eq. 41) ties the roots to the angular-momentum geometry:

```
c_1 = eta (J0norm² - L0norm² - SAv2) / (2 L0norm),   SAv2 = (Spl2 + Smi2)/2
```

Evaluating the quadratic form `S² eta² - c_1²` at both turning points (Eq. D2–D3) gives

```
cp = Spl2 eta² - c_1²        # at the upper turning point
cm = Smi2 eta² - c_1²        # at the lower turning point
```

`cp` and `cm` are bookkeeping values for a partial-fraction decomposition of the precession-averaged precession frequency `Omega_z(S²)`.
They feed the `Omegaz0..5` PN coefficients of `d(alpha)/dv` (Eq. 65) and an analogous `Omegazeta0..5` chain for `d(zeta)/dv`, through shared intermediates `adD, hdD, cdD, fdD`.
Both chains are integrated in closed form to give `alpha(f)` and `epsilon(f) = -gamma(f)`, the Euler angles that twist the co-precessing `(2,2)` mode into the inertial frame.

Near the double root `cp ≈ cm`, and two steps compound the precision already lost at the `arccos` branch point:

1. `sqrt(cp cm)` roughly halves the surviving precision again.
2. `dw = 4 cp - 4 D2RmSq eta²` subtracts two comparable terms derived from `cp`/`cm`, and `adD, hdD, cdD, fdD` all divide by `dw`; near the double root `dw` is itself anomalously small.

The corrupted digits are therefore **common to `alpha` and `epsilon`** (same `adD, hdD, cdD, fdD`), while `cos(beta)` does not pass through this chain and is unaffected.
This is an ill-conditioning of the MSA formulation itself: a one-ULP change in an input mass moves LAL's own `alpha(f)` by far more than the ripple–LAL difference.
ripple's root deflation gives a well-defined, reproducible answer rather than amplified noise, but cannot recover information the double-precision inputs do not contain.
ripple and LAL agree to ~1e-8 rad wherever the cubic is resolvable (`|1 + acosarg|` not tiny).

## 3. Why the tail is edge-on and `hc`-only

The co-precessing `(2,2)` mode is twisted into polarizations by a sum over `m = -2..2` of `e^{∓i m alpha} d²_{m,±2}(beta) Y_{2m}(theta_JN)` (`twist_22`, `src/ripplegw/waveforms/cbc/IMRPhenomX/IMRPhenomXPHM.py`; Eq. 3.5–3.7 of arXiv:2004.06503).
For a weakly precessing binary `beta` is small, so every term except the one carrying `e^{-2i(alpha - epsilon)}` is suppressed by a power of `sin(beta)`.

- **`hp`** keeps the `beta`-independent leading term, and because the corrupted digits are common to `alpha` and `epsilon` they cancel in `alpha - epsilon`, so `hp` is protected from the instability at leading order.
- **`hc`** has no such leading term once `beta` is small; it is built entirely from the `O(sin beta)` sidebands, whose *relative* `alpha`/`epsilon` phase does not reproduce that cancellation, so the absolute angle error invisible in `hp` appears at full strength in `hc`.

For a source at inclination `iota` (with `theta_JN ≈ iota` at low spin), the cross polarization's own signal content scales with `2 cos(iota) / (1 + cos²iota)` and vanishes edge-on.
A fixed absolute angle error divided by that vanishing amplitude becomes an arbitrarily large *relative* per-polarization overlap loss for `hc` as `iota → π/2` — a geometric amplification of an error that carries negligible detector SNR there, not a growing absolute defect.

This is why `tests/cross_validation/fd/test_overlap.py` asserts the SNR-weighted combination of the two polarizations (`combined_overlap_loss` in `tests/helpers/metrics.py`), which weights each polarization by its own noise-weighted power and divides the edge-on amplification back out.
The per-polarization losses are kept only as diagnostics.

## 4. The separate equal-mass singularity (q → 1)

`S1L_pav`/`S2L_pav` (`initialize_MSA_system.py`, Eq. A9–A14), which feed the precessing final spin `afinal_prec` used for `fRING`/`fDAMP` in `src/ripplegw/waveforms/cbc/IMRPhenomX/IMRPhenomXHM.py`, are

```
S1L_pav =  (c_1 (1 + q) - q eta Seff) / (eta (1 - q²))
S2L_pav = -q (c_1 (1 + q) -   eta Seff) / (eta (1 - q²))
```

a genuine `0/0` at `q = 1`, where the numerator vanishes too.
ripple guards the exact floating-point equality `1 - q² == 0` and returns `0`, which is not the L'Hôpital limit; LAL's `(1 - q²) + 1e-16` padding is no better right at `q = 1`.
This only produces an O(1) relative error within `|1 - q| ≲ 1e-9`, far tighter than continuous mass sampling reaches, so it is not what the observed low-spin tail is made of.
It would matter for a deliberately equal-mass system.
