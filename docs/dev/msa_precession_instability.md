# The MSA Precession Instability: What `cm` Is and Why It Drives the Mismatch Tail

This document derives, step by step, the numerical mechanism behind the "known cause
(LAL MSA conditioning)" entry for `IMRPhenomXP_NRTidalv3` in
[`lal_agreement.md`](lal_agreement.md), and explains why it shows up as a mismatch
*tail* concentrated at low spin and small opening angle rather than as a uniform
error. It also covers the companion (and separate) equal-mass singularity. The
reproducing script is
`tests/cross_validation/results/analyse_msa_failure_modes.py`.

Equation numbers below refer to Chatziioannou, Klein, Yunes & Cornish, *PRD 95,
104004 (2017)*, arXiv:1703.03967 ("the MSA paper"), which is what
`IMRPhenomX_Initialize_MSA_System` (`src/ripplegw/waveforms/initialize_MSA_system.py`)
implements.

---

## 1. Where S and L come from

MSA ("multi-scale analysis") describes the precession of a compact binary's spins by
splitting the dynamics into three timescales: orbital, precession, and radiation
reaction. The two vectors that matter here are the orbital angular momentum **L** and
the total spin **S = S1 + S2**.

ripple builds these exactly as LAL does
(`IMRPhenomX_Initialize_MSA_System`, lines 188-244):

```
Lhat = (0, 0, 1)                          # source frame: L defines z at f_ref

S1v  = chi1 * (eta / q)                   # dimensionful spin vectors
S2v  = chi2 * (eta * q)                   #   q = m2/m1 < 1, eta = m1*m2/M^2

L_0  = Lhat * (eta / v_0)                 # 3PN orbital angular momentum at f_ref
                                           #   v_0 = (pi * M * f_ref)^(1/3)  [cbrt_cr]

S0   = S1v + S2v                          # total spin vector
J_0  = L_0 + S0                           # total angular momentum vector
```

`S1v`/`S2v` are dimensionful (mass-rescaled) spin vectors, not the unit dimensionless
spins `chi1`/`chi2` — note the `eta/q` and `eta*q` prefactors, which come from
expressing each body's spin magnitude `chi_i * m_i^2` in the `M=1`, `q = m2/m1`
convention MSA uses internally. `L_0` is the leading-order (Newtonian, here refined to
3PN) orbital angular momentum at the reference frequency; the whole point of MSA is
that this vector precesses around the (nearly-conserved) direction of **J** on the
precession timescale while **S** rotates around **L** on a much shorter timescale.

The three scalar invariants that drive everything below are the norms:

```
L0norm = |L_0|,   J0norm = |J_0|,   S_0_norm = |S0|
```

## 2. The S² cubic

Over one precession cycle, `S = |S|` oscillates between two turning points, `S_-`
and `S_+` (`Smi`, `Spl` in the code), determined by a cubic polynomial in `S²`
(Eq. B2-B4 of the MSA paper):

```
B(L, J, S1², S2², q, eta, delta, Seff)
C(L, J, S1², S2², q, eta, delta, Seff)
D(L, J, S1², S2², q, eta, delta, Seff)

x^3 + B x^2 + C x + D = 0        where x = S^2
```

(`IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA`). This monic cubic is solved in
its depressed (trace-removed) form via the standard trigonometric formula for three
real roots (Viète, `IMRPhenomX_Return_Roots_MSA`):

```
p  = C - B^2/3
qc = (2/27) B^3 - B C / 3 + D

acosarg = 1.5 * qc / (p * sqrt(-p/3))            # in [-1, 1] for 3 real roots
theta   = arccos(acosarg) / 3

tmp1 = 2*sqrt(-p/3)*cos(theta - 4pi/3) - B/3
tmp2 = 2*sqrt(-p/3)*cos(theta - 2pi/3) - B/3
tmp3 = 2*sqrt(-p/3)*cos(theta)         - B/3
```

**Nothing about the formula guarantees `tmp1 <= tmp2 <= tmp3`** (or any other fixed
positional order) — which branch ends up largest/smallest depends on `theta`, i.e.
on the binary's parameters. The code does not assume an order; it sorts explicitly:

```
hi  = max(tmp1, tmp2, tmp3)
lo  = min(tmp1, tmp2, tmp3)
mid = tmp1 + tmp2 + tmp3 - hi - lo     # whichever of the three wasn't hi or lo
```

`lo`, `mid`, `hi` are **not** simply relabeled `S32`, `Smi2`, `Spl2`. §2.1 shows the
double root relevant to this document always occurs at the *upper* pair
(`hi ≈ mid`); the code detects this generically by comparing `gap_hi = hi - mid` to
`gap_lo = mid - lo` and treats whichever gap is smaller as the colliding pair. It
then takes the *isolated*, well-separated root directly (insensitive to the
precision lost at the collision — see §2.1) and **reconstructs the colliding pair**
from Vieta's formulas applied to the deflated quadratic
(`(x^3+Bx^2+Cx+D) / (x - r_iso)`), rather than reading it off the raw `mid`/`hi` (or
`mid`/`lo`) trigonometric values, which is exactly where the cancellation lives. So
`S32` is the raw isolated root *only* when the upper pair collides; when the lower
pair collides instead, `S32` is itself a reconstructed value and the isolated root
becomes `Spl2`. §3 picks this back up for why the reconstruction matters for `cm`.

### 2.1 Where the double root comes from physically

The precession *cone opening angle* is controlled by how far apart `Smi2` and `Spl2`
are. When the total spin is small and/or L and S are nearly parallel — i.e. the
binary is only weakly precessing — the two turning points of `S²(t)` are close
together: the system barely oscillates because there's almost no room for **S** to
swing around **L**. In the exact aligned-spin limit (`chi1x=chi1y=chi2x=chi2y=0`,
no precession at all) they coincide exactly: `Smi2 = Spl2`, a double root of the
cubic.

The double root specifically sits at `acosarg = -1`, not `+1` — this follows from
the trigonometric solution itself, not just from this instability's phenomenology.
`theta = arccos(acosarg)/3` sweeps `[0, pi/3]` as `acosarg` sweeps `[+1, -1]`.
Substituting the endpoints directly into `tmp1, tmp2, tmp3` above:

- At `theta = 0` (`acosarg = +1`): `cos(-4pi/3) = cos(-2pi/3) = -1/2`, so
  `tmp1 = tmp2` — the *lower* pair collides (`lo ≈ mid`), and `tmp3` (`= 2*sqrt(-p/3)*1
  - B/3`) is the isolated, larger root.
- At `theta = pi/3` (`acosarg = -1`): `cos(pi/3 - 4pi/3) = cos(-pi) = -1`, so
  `tmp1 = -2*sqrt(-p/3) - B/3` is the isolated, *smaller* root, while
  `cos(pi/3 - 2pi/3) = cos(-pi/3) = 1/2 = cos(pi/3) = cos(theta)`, so `tmp2 = tmp3` —
  the *upper* pair collides (`mid ≈ hi`).

Only the second case (`acosarg -> -1`, upper pair colliding) is the physically
relevant one here: it is what happens as spins approach alignment. Verified directly
against the code for the exactly-aligned example above (`m1=20, m2=12, chi1z=0.3,
chi2z=-0.15, f_ref=20`): `acosarg = -1.0000000000`, `hi - mid = -6.9e-18` (zero to
float64 precision) while `mid - lo = 0.189` — the upper pair, and only the upper
pair, has collided.

`arccos` has an infinite derivative at `-1`:

```
arccos(-1 + e) = pi - sqrt(2e) + O(e^1.5)
```

so as `acosarg` approaches `-1`, an input perturbation `delta(acosarg)` produces an
*unbounded* perturbation `delta(theta) ~ delta(acosarg) / sqrt(2(1+acosarg))` in
`theta`, and hence in `tmp2 = cos(theta - 2pi/3)` and `tmp3 = cos(theta)` — the
upper pair that is colliding (§2 above; `tmp1` is the well-separated isolated root
here and stays accurate). Measured over the BNS prior (`chi <~ 0.05`), **88%** of samples have
`|1 + acosarg| <= 1e-6` (`test_msa_angle_conditioning_is_reported` in
`test_msa_angles.py`) — this is not a rare corner case, it is the *typical* low-spin
configuration.

ripple mitigates the immediate loss of precision in `Smi2`/`Spl2` themselves by
deflating the well-conditioned isolated root and reading the colliding pair off the
resulting quadratic via Vieta's formulas rather than differencing two nearly-equal
numbers directly (see the long comment block in `IMRPhenomX_Return_Roots_MSA`). This
makes ripple's *own* `Smi2`/`Spl2` self-consistent to ~1e-11 rad in the resulting
angles. **It does not fix the conditioning of the physics itself** — see §4.

## 3. `c_1`, and what `cm` actually is

`c_1` (Eq. 41) is the conserved combination that ties the cubic's roots to the
angular-momentum geometry:

```
c_1 = eta * (J0norm^2 - L0norm^2 - SAv2) / (2 * L0norm)     where SAv2 = (Spl2+Smi2)/2
```

Then (Eq. D2-D3), evaluate the same quadratic form `S² eta² - c_1²` at *both*
turning points:

```
cp = Spl2 * eta^2 - c_1^2        # at the upper turning point S_+
cm = Smi2 * eta^2 - c_1^2        # at the lower turning point S_-   <-- this is `cm`
```

`cp` and `cm` are not independent physical quantities so much as bookkeeping
values used to build a *partial-fraction decomposition* of the precession-averaged
precession frequency `Omega_z(S²)` (the rate of change of the spin-precession
phase) as a rational function of `S²` with poles related to `Smi2`/`Spl2`. They feed
directly into (Eq. D4-D21, `IMRPhenomX_Initialize_MSA_System` lines 483-563):

```
Rm       = Spl2 - Smi2
cpcm     = |cp * cm|
sqrt_cpcm = sqrt(cpcm)

D2RmSq = (cp - sqrt_cpcm) / eta^2
D4RmSq = -0.5*Rm*sqrt_cpcm/eta^2 - cp/eta^4*(sqrt_cpcm - cp)

dw = 4*cp - 4*D2RmSq*eta^2
hw = -2*(2*D2RmSq - Rm)*c_1
fw = Rm*D2RmSq - D4RmSq - 0.25*Rm^2

adD = aw/dw,  hdD = hw/dw,  cdD = cw/dw,  fdD = fw/dw   # dw is the shared denominator

Omegaz0 = a1dD + adD
Omegaz1 = a2dD - adD*Seff - adD*hdD
Omegaz2 = adD*hdD*Seff + cdD - adD*fdD + adD*hdD^2
Omegaz3, Omegaz4, Omegaz5 = ... (products/sums of the same adD, hdD, cdD, fdD)
```

`Omegaz0..5` are then the PN-expansion coefficients of `d(alpha)/dv` (Eq. 65), and an
analogous `Omegazeta0..5` chain (reusing the same `adD, hdD, cdD, fdD, gdD`) gives
`d(zeta)/dv`. Both are integrated in closed form over `v` to give `phi_z(v)`/`zeta(v)`,
which set `alpha(f)` and `epsilon(f) = -gamma(f)` — the two Euler angles that twist the
co-precessing `(2,2)` mode into the inertial frame.

### 3.1 Why `cm` is the danger point

Near the double root (`Smi2 ~= Spl2`), both `cp` and `cm` are evaluated at *nearly the
same* `S²` value, so `cp ~= cm` and both already carry whatever precision loss
`Smi2`/`Spl2` picked up from the `arccos` branch point in §2.1. Two things then make
it worse rather than better:

1. **`sqrt(cp*cm)` roughly halves the remaining precision again** (a relative error
   `delta` in the product becomes `delta/2` in the square root — the standard
   amplitude/precision trade of a square root — but only *after* `cp*cm` has already
   inherited the `arccos` error, so the combined loss compounds).
2. **`dw = 4*cp - 4*D2RmSq*eta²` is a subtraction of two comparable-sized terms**
   that both derive from `cp`/`cm`, and `adD, hdD, cdD, fdD` all *divide by* `dw`.
   Near the double root this denominator can itself be anomalously small, so the
   division amplifies whatever error survived the square root above.

The net effect, measured directly: perturbing `m1` by 1 part in `1e15` (a single
ULP) moves LAL's own `alpha(f)` by a **median ~1e-2 rad**, i.e. LAL does not agree
with itself to better than that in this regime — this is not a ripple-vs-LAL
discrepancy, it is the underlying MSA formulation being numerically ill-conditioned
at low spin, on both sides. The total measured amplification from the `arccos`
input to `Omegaz0` is roughly **1e10-1e12**.

Because `cm`/`cp` feed *both* the `Omegaz*` chain (sets `alpha`) and the
`Omegazeta*` chain (sets `zeta`, hence `epsilon`) through the *same* `adD, hdD, cdD,
fdD` intermediates, the corrupted digits are **common to `alpha` and `epsilon`** —
this is the key fact that determines how the error shows up in the waveform (§5).
`cos(beta)` does not go through this chain and is unaffected.

## 4. What ripple does differently from LAL here — and what it doesn't fix

ripple's root-deflation trick (§2.1) keeps `Smi2`/`Spl2` internally self-consistent,
so ripple's `cm`, `Omegaz0..5`, `alpha(f)`, `epsilon(f)` are reproducible to ~1e-11
rad under input perturbations that move LAL's own output by ~1e-2 rad. **This does
not mean ripple's angles are more physically correct** — the S² cubic genuinely does
have (numerically) a double root there, and *no* algorithm can extract more
information about the true precession geometry than the double-precision inputs
contain. What ripple's stability buys is a well-defined, reproducible answer rather
than amplified noise; the two only need to (and only can) be compared where the
cubic is resolvable, `|1 + acosarg| > 1e-6`, where ripple tracks LAL to ~1e-8 rad
(`test_msa_angles.py`).

## 5. Why the error is amplified for edge-on inclinations: the Wigner-d matrices

The co-precessing `(2,2)` mode `h0` gets twisted into the inertial-frame polarizations
via a sum over `m = -2..2` (`twist_22`, `IMRPhenomXPHM.py`; Eq. 3.5-3.7 of the
precessing-waveform paper, arXiv:2004.06503):

```
hp_sum = sum_m [ e^{-i m alpha} d^2_{m,-2}(beta) Y_{2m}(theta_JN)
               + e^{+i m alpha} d^2_{m, 2}(beta) Y*_{2m}(theta_JN) ]

hc_sum = i * (same sum, antisymmetric combination)

hp, hc = eps_phase * hp_sum, eps_phase * hc_sum,   eps_phase = exp(-2i*epsilon)/2 * h0
```

`d^2_{m,\pm2}(beta)` are the Wigner-d matrix elements — in the code, `d22`/`d2m2`
built from `beta_powers` (`cBetah = cos(beta/2)`, `sBetah = sin(beta/2)`,
`IMRPhenomXWignerdCoefficients_cosbeta`):

```
d^2_{2,2}   = cBetah^4
d^2_{1,2}   = 2 cBetah^3 sBetah
d^2_{0,2}   = sqrt(6) cBetah^2 sBetah^2
d^2_{-1,2}  = 2 cBetah sBetah^3
d^2_{-2,2}  = sBetah^4
```

For a weakly precessing binary — exactly the low-spin regime where §2-3's `cm`
instability is common — `beta` (the opening angle between L and J) is small, so
`sBetah = sin(beta/2) ~ beta/2` is small and `cBetah ~= 1`. Every term in the sum
*except* the `m=2` (in `hp_sum`'s first half) / `m=-2`-conjugate (in the second half)
term is then suppressed by a power of `sin(beta)`: the dominant contribution reduces
to the single term carrying `e^{-2i*(alpha - epsilon)}` — this is where the earlier
"common error in alpha and epsilon" from §3.1 matters. **Because the corrupted
`Omegaz0`/`Omegazeta0` digits are common to both `alpha` and `epsilon`, the leading-
order error cancels identically in this dominant `exp(2i(alpha-epsilon))` phase.**
`hp`'s accuracy is therefore essentially unaffected by the MSA instability at leading
order — it is "protected."

The error does **not** cancel in the `O(sin(beta))` sub-leading terms of the sum
(the `d^2_{\pm1,2}` and `d^2_{0,2}` terms), because those come with a *relative*
`alpha`/`epsilon` phase (`e^{\mp i alpha}` combined with a different sign in the
`hp_sum`/`hc_sum` antisymmetric combination) that does not reproduce the same
cancellation. These sidebands are what `hc` (the cross polarization) is *entirely*
built from once `beta` is small — `hc_sum` has no `beta`-independent leading term the
way `hp_sum` does. So the same absolute angle error that is invisible in `hp` shows
up at full strength, relatively speaking, in `hc`.

### 5.1 From opening angle to inclination

`beta` above is the *intrinsic* precession opening angle, not the observer's
inclination `iota` — but the two set how much power ends up in each polarization
together. For a source viewed at inclination `iota` (with `theta_JN ~= iota` in the
low-spin limit), the standard non-precessing amplitude ratio is

```
|h_+| / |h_x| = (1 + cos^2(iota)) / (2 cos(iota))
```

which diverges as `iota -> pi/2` (edge-on): `h_x`'s *own* signal content vanishes
there, even though the absolute angle error injected into it does not. So the same
absolute `alpha`/`epsilon` error, divided by a vanishing `h_x` amplitude, produces an
arbitrarily large *relative* error — i.e. an arbitrarily large per-polarization
overlap loss for `h_x` — as the source approaches edge-on. Empirically (fit over
n=5000 low-spin samples):

```
log10(overlap_loss_hc / overlap_loss_hp) ~ 1.84 * log10( [(1+cos^2 iota)/(2 cos iota)]^2 )
```

with R² = 0.93, and a controlled angle-injection experiment (inject a known
`delta_alpha = delta_epsilon` into an otherwise-exact waveform and sweep `iota`)
reproduces the same exponent independent of `chi_p`
(`tests/cross_validation/results/analyse_mismatches.ipynb`, cells 17-18). This is
exactly why `overlap_loss_hc` alone is not the metric asserted in
`test_lal_overlap.py`: it measures an edge-on *geometric amplification* of an error
that carries negligible detector SNR there, not a growing absolute defect. The
SNR-weighted combined metric (`compute_polarization_weighted_overlap_loss`,
documented in `lal_agreement.md`) divides that amplification back out.

## 6. Putting it together: why the mismatch tail tracks degeneracy, not q -> 1

Two distinct MSA failure modes are named in the IMRPhenomXP paper (Pratten et al,
arXiv:2004.06503): the equal-mass singularity in `S1L_pav`/`S2L_pav` (feeds the
final-spin estimate, §7) and the S/L-near-alignment instability derived above.
Correlating `lal_msa_cubic_degeneracy` (`tests/utils.py`) against the observed
mismatch in low-spin cross-validation runs
(`analyse_msa_failure_modes.py --csv ...`) shows the S² cubic degeneracy is the one
that matters here: binning n=4246 low-spin/low-mass `IMRPhenomXP` samples by
`|1+acosarg|` decile gives a clean monotonic trend from median `log10(mismatch)
= -11.2` (worst decile) down to `-15.5` (best decile), Spearman rho = -0.55
(p ~ 0). Mass ratio does *not* independently correlate — `1-q` only tracks mismatch
because it happens to anti-correlate with the degeneracy metric itself (rho =
-0.77..-0.82) in this parameter draw, not because any sample gets close enough to
q=1 for §7's mechanism to activate (closest sample in 5000 draws: `|1-q| = 3e-5`, far
outside the `|1-q| <~ 1e-9` window where that mechanism produces an O(1) relative
error).

## 7. The other failure mode, briefly: q -> 1

Separately, `S1L_pav`/`S2L_pav` (`initialize_MSA_system.py` lines 312-321, feeding
the final-spin estimate `afinal_prec = sqrt(SAv2 + Lfinal^2 + 2*Lfinal*(S1L_pav +
S2L_pav))` in `IMRPhenomXHM.py`) are computed as

```
S1L_pav = (c_1*(1+q) - q*eta*Seff) / (eta*(1-q^2))
S2L_pav = -q*(c_1*(1+q) - eta*Seff) / (eta*(1-q^2))
```

which is a genuine `0/0` at `q=1` (the numerator vanishes there too — verified
numerically, `analyse_msa_failure_modes.py --q-sweep`). ripple guards only the exact
floating-point equality `1-q^2 == 0.0` and returns `0`, which is *not* the true
L'Hopital limit (empirically ~8e-3/1.7e-3 for a representative case, not 0); LAL's
own `(1-q^2)+1e-16` padding is, if anything, worse right at `q=1` (returns O(0.1)
amplified roundoff noise in the same test case). Both are real defects worth fixing,
but — per the `--q-sweep` output — the instability only produces an O(1) relative
error for `|1-q| <~ 1e-9`, far tighter than continuous mass sampling reaches; §6's
correlation confirms this is not what's in the observed tail. It would matter for a
deliberately-injected exactly-equal-mass system.
