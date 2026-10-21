"""
JAX implementation of LAL's ``IMRPhenomNSBH`` (``LALSimIMRPhenomNSBH.c``).

The model (Thompson et al., arXiv:1905.07179; LIGO-T1900729) consists of

* an amplitude based on IMRPhenomC (Santamaria et al. 1005.3306) with the NSBH merger-type corrections of
  Pannarale et al. (1509.00512). The corrections depend on the tidal disruption frequency, the torus mass,
  the remnant mass/spin (Zappa et al. 1903.11622 on top of the UIB2016 BBH fits, 1611.00332) and the
  l=m=2 QNM frequency (1810.03550).
* a phase given by IMRPhenomD (1508.07253), evaluated for the NSBH remnant spin, plus the NRTidalv2 tidal
  phase (1905.06011) with only the NS being deformable.

Conventions: ``m1`` is the black hole (the heavier body, ``m1 >= m2``), ``m2`` the neutron star. ``s1_z`` /
``s2_z`` are the aligned BH / NS spins and ``lambda_2`` the NS dimensionless tidal deformability (the BH
deformability is zero by definition).

Notes on LAL conventions that are reproduced on purpose:

* LAL's NSBH amplitude is negative (``amp = -(...)``); here the amplitude is ``|amp|`` and the sign is carried
  by an additional ``pi`` in the phase.
* The time shift ``t0`` is the derivative of the merger-ringdown phase at LAL's upper frequency ``f_max``
  (not at the amplitude peak as in IMRPhenomD); without ``f_max``, LAL's default ``Mf = 0.2``.
* There is no frequency cutoff: every requested frequency is filled, as LAL fills its band up to ``f_max``.
"""

from collections.abc import Mapping
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jaxtyping import Array, Complex, Float

from ripplegw.constants import EULERGAMMA, MPC, MRSUN, MTSUN, PI
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.interfaces import AmplitudePhaseWaveform, DistanceScaledWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.utils.tidal import get_kappa, universal_relation
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    get_tidal_phase,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import (
    get_IIa_raw_phase,
    get_IIb_raw_phase,
    get_inspiral_phase,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import (
    QNMData_fdamp,
    QNMData_fRD,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import (
    EradRational0815,
    _qnm_interp,
    get_coeffs,
    get_transition_frequencies_from_fRD_fdamp,
)
from ripplegw.waveforms.cbc.Taylor.TaylorF2 import (
    get_4PNQM2SCoeff,
    get_4PNQM2SOCoeff,
    get_6PNQM2SCoeff,
)

#######################################################################################################
### Small helpers
#######################################################################################################


def _component_masses(mc: FloatLike, eta: FloatLike, diagnostics: bool = False):
    """Validate chirp coordinates before conversion can hide an unphysical eta.

    The LAL bound ``1 <= mBH/mNS <= 100`` is equivalent to
    ``100/101**2 <= eta <= 1/4``. No neutron-star mass cut is applied.
    """
    valid_eta = (eta >= 100.0 / 101.0**2) & (eta <= 0.25)
    valid_mc = jnp.isfinite(mc) & (mc > 0.0)
    if diagnostics:
        checkify.debug_check(
            valid_eta,
            "IMRPhenomNSBH requires 1 <= q <= 100 (eta in [100/101**2, 0.25])",
        )
        checkify.debug_check(valid_mc, "IMRPhenomNSBH requires a finite positive M_c")
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    valid = valid_eta & valid_mc
    return jnp.where(valid, m1, jnp.nan), jnp.where(valid, m2, jnp.nan)


def _window_plus(x: FloatLike, x0: FloatLike, d: FloatLike) -> FloatLike:
    """LAL ``XLALSimIMRPhenomNSBH_window_plus``: ``0.25 (1 + tanh(4 (x - x0) / d))``."""
    return 0.25 * (1.0 + jnp.tanh(4.0 * (x - x0) / d))


def _window_minus(x: FloatLike, x0: FloatLike, d: FloatLike) -> FloatLike:
    """LAL ``XLALSimIMRPhenomNSBH_window_minus``: ``0.25 (1 - tanh(4 (x - x0) / d))``."""
    return 0.25 * (1.0 - jnp.tanh(4.0 * (x - x0) / d))


def _w_plus(Mf: FloatLike, Mf0: FloatLike, d: FloatLike) -> FloatLike:
    """IMRPhenomC ``wPlus`` window, in geometric frequency units."""
    return 0.5 * (1.0 + jnp.tanh(4.0 * (Mf - Mf0) / d))


def _w_minus(Mf: FloatLike, Mf0: FloatLike, d: FloatLike) -> FloatLike:
    """IMRPhenomC ``wMinus`` window, in geometric frequency units."""
    return 0.5 * (1.0 - jnp.tanh(4.0 * (Mf - Mf0) / d))


def _sqrt_nonneg(x: FloatLike) -> FloatLike:
    """``sqrt(max(x, 0))`` (guards round-off such as ``1 - 4 eta = -1e-17`` at equal mass)."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


#######################################################################################################
### Neutron-star / Kerr helpers (LALSimNSBHProperties.c)
#######################################################################################################


def compactness_from_lambda(lam: FloatLike) -> FloatLike:
    """NS compactness from the tidal deformability (``XLALSimNSBH_compactness_from_lambda``).

    Yagi & Yunes universal relation for ``lam > 1``; for ``lam <= 1`` a C^1 extrapolation to the BH limit.
    """
    a0, a1, a2 = 0.360, -0.0355, 0.000705
    log_lam = jnp.log(jnp.maximum(lam, 1.0))  # only used where lam > 1
    high = a0 + a1 * log_lam + a2 * log_lam**2
    L2 = lam * lam
    L3 = L2 * lam
    low = 0.5 + (3 * a0 - a1 - 1.5) * L2 + (-2 * a0 + a1 + 1) * L3
    return jnp.where(lam > 1.0, high, low)


def baryonic_mass_from_C(C: FloatLike, Mg: FloatLike) -> FloatLike:
    """NS baryonic mass from gravitational mass and compactness (``XLALSimIMRPhenomNSBH_baryonic_mass_from_C``)."""
    d1, d2 = 6.19e-1, 1.359e-1
    return Mg + Mg * (d1 * C + d2 * C**2)


def kerr_isco_radius(a: FloatLike) -> FloatLike:
    """Kerr ISCO radius in units of the BH mass (``XLALSimNSBH_rKerrISCO``)."""
    # The radicand vanishes like a^2 at a = 0 so d(sqrt)/da is 0/0; nudge exactly-zero spins by 1e-9
    # (value change < 1e-9, gradient w.r.t. ``a`` kept).
    a = a + jnp.where(jnp.abs(a) < 1e-9, 1e-9, 0.0)
    Z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    Z2 = jnp.sqrt(3.0 * a**2 + Z1**2)
    root = jnp.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    return jnp.where(a > 0.0, 3.0 + Z2 - root, 3.0 + Z2 + root)


def _xi_poly(z: FloatLike, q: FloatLike, a: FloatLike, mu: FloatLike) -> FloatLike:
    """The degree-10 polynomial of ``XLALSimNSBH_xi_tide`` (Eq. 8 of 1509.00512), in ``z = sqrt(xi)``."""
    c0 = -3.0 * q * mu**2 * a**2
    c2 = 6.0 * q * mu
    c4 = -3.0 * q
    c7 = 2.0 * a * mu**1.5
    c8 = -3.0 * mu
    # Horner, ascending coefficients [c0, 0, c2, 0, c4, 0, 0, c7, c8, 0, 1]
    out = 1.0
    for c in (0.0, c8, c7, 0.0, 0.0, c4, 0.0, c2, 0.0, c0):
        out = out * z + c
    return out


_XI_GRID = 2048
_XI_BISECT_ITERS = 64


def xi_tide(q: FloatLike, a: FloatLike, mu: FloatLike) -> FloatLike:
    """Relativistic correction to the orbital radius at mass shedding (``XLALSimNSBH_xi_tide``).

    LAL returns the square of the largest positive real root of a degree-10 polynomial in ``z = sqrt(xi)``
    (0 if there is none). Here the root is bracketed with a scan on a uniform grid up to a Fujiwara bound,
    refined by bisection, and made differentiable with one implicit-function Newton step.

    Args:
        q: Mass ratio ``M_BH / M_NS``.
        a: Dimensionless BH spin.
        mu: ``q * C`` with ``C`` the NS compactness.
    """
    # Fujiwara upper bound on |roots| of the monic polynomial: 2 max_k |a_{n-k}|^(1/k) (constant term halved)
    bound = 2.0 * jnp.max(
        jnp.stack(
            [
                (3.0 * mu) ** (1.0 / 2.0),
                (2.0 * jnp.abs(a) * mu**1.5) ** (1.0 / 3.0),
                (3.0 * q) ** (1.0 / 6.0),
                (6.0 * q * mu) ** (1.0 / 8.0),
                (1.5 * q * mu**2 * a**2) ** (1.0 / 10.0),
            ]
        )
    )
    bound = jax.lax.stop_gradient(bound)
    q_, a_, mu_ = jax.lax.stop_gradient((q, a, mu))

    z_grid = bound * jnp.arange(1, _XI_GRID + 1) / _XI_GRID
    nonpos = jnp.asarray(_xi_poly(z_grid, q_, a_, mu_) <= 0.0)
    has_root = jnp.any(nonpos)
    # last grid point with P <= 0; the largest root lies in (z[idx], z[idx+1]]
    idx = _XI_GRID - 1 - jnp.argmax(nonpos[::-1])
    lo = z_grid[idx]
    hi = z_grid[jnp.minimum(idx + 1, _XI_GRID - 1)]

    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        below = _xi_poly(mid, q_, a_, mu_) <= 0.0
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _XI_BISECT_ITERS, bisect, (lo, hi))
    z0 = jax.lax.stop_gradient(0.5 * (lo + hi))

    # implicit-function step: z = z0 - P(z0; theta) / P'(z0); exact value (P(z0) ~ 0), correct first derivatives
    dP = jax.lax.stop_gradient(jax.grad(_xi_poly)(z0, q_, a_, mu_))
    safe_dP = jnp.where(has_root, dP, 1.0)
    z = z0 - _xi_poly(z0, q, a, mu) / safe_dP
    return jnp.where(has_root, z**2, 0.0)


def torus_mass_fit(q: FloatLike, a: FloatLike, C: FloatLike) -> FloatLike:
    """Baryonic torus mass in units of the NS baryonic mass (``XLALSimNSBH_torus_mass_fit``, Eq. 11 of 1509.00512)."""
    mu = q * C
    alpha, beta = 0.296, 0.171
    xi = xi_tide(q, a, mu)
    Mtorus = alpha * xi * (1.0 - 2.0 * C) - beta * mu * kerr_isco_radius(a)
    return jnp.where(Mtorus < 0.0, 0.0, Mtorus)


def f_gw_in_kerr(r: FloatLike, M: FloatLike, a: FloatLike) -> FloatLike:
    """GW frequency of a particle on a circular Kerr orbit (``XLALSimNSBH_fGWinKerr``)."""
    return 1.0 / (PI * (a * M + jnp.sqrt(jnp.maximum(r**3 / M, 1e-300))))


#######################################################################################################
### BH-NS remnant fits (LALSimBHNSRemnantFits.c)
#######################################################################################################


def bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, chi2):
    """``XLALbbh_final_spin_non_precessing_UIB2016`` (1611.00332v2, Eq. 16)."""
    m = m1 + m2
    msq = m * m
    m1sq = m1 * m1
    m2sq = m2 * m2
    eta = jnp.clip(m1 * m2 / msq, 0.0, 0.25)
    eta2 = eta * eta
    eta3 = eta2 * eta
    S1 = chi1 * m1sq / msq
    S2 = chi2 * m2sq / msq
    Stot = S1 + S2
    Shat = (chi1 * m1sq + chi2 * m2sq) / (m1sq + m2sq)
    Shat2 = Shat * Shat
    Shat3 = Shat2 * Shat
    chidiff = jnp.where(m2 > m1, -(chi1 - chi2), chi1 - chi2)
    chidiff2 = chidiff * chidiff
    sqrt3 = 3.0**0.5
    sqrt1m4eta = _sqrt_nonneg(1.0 - 4.0 * eta)

    a20, a30, a50 = 5.24, 1.3, 2.88
    b10, b20, b30, b50 = -0.194, 0.0851, 0.00954, -0.579
    a2, a3, a5 = 3.8326341618708577, -9.487364155598392, 2.5134875145648374
    b1, b2, b3, b5 = (
        1.0009563702914628,
        0.7877509372255369,
        0.6540138407185817,
        0.8396665722805308,
    )
    f21, f31, f50, f11, f52 = (
        8.77367320110712,
        22.830033250479833,
        1.8804718791591157,
        4.409160174224525,
        0.0,
    )
    d10, d11, d20, d30, d31 = (
        0.3223660562764661,
        9.332575956437443,
        -0.059808322561702126,
        2.3170397514509933,
        -3.2624649875884852,
    )
    f12, f22, f32, f51 = (
        0.5118334706832706,
        -32.060648277652994,
        -153.83722669033995,
        -4.770246856212403,
    )

    Lorb = (
        (2.0 * sqrt3 * eta + a20 * a2 * eta2 + a30 * a3 * eta3) / (1.0 + a50 * a5 * eta)
        + (
            b10
            * b1
            * Shat
            * (f11 * eta + f12 * eta2 + (64.0 - 16.0 * f11 - 4.0 * f12) * eta3)
            + b20
            * b2
            * Shat2
            * (f21 * eta + f22 * eta2 + (64.0 - 16.0 * f21 - 4.0 * f22) * eta3)
            + b30
            * b3
            * Shat3
            * (f31 * eta + f32 * eta2 + (64.0 - 16.0 * f31 - 4.0 * f32) * eta3)
        )
        / (
            1.0
            + b50
            * b5
            * Shat
            * (
                f50
                + f51 * eta
                + f52 * eta2
                + (64.0 - 64.0 * f50 - 16.0 * f51 - 4.0 * f52) * eta3
            )
        )
        + d10 * sqrt1m4eta * eta2 * (1.0 + d11 * eta) * chidiff
        + d30 * Shat * sqrt1m4eta * eta3 * (1.0 + d31 * eta) * chidiff
        + d20 * eta3 * chidiff2
    )
    return Lorb + Stot


def bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2):
    """``XLALbbh_final_mass_non_precessing_UIB2016`` (1611.00332v2, Eq. 27)."""
    m = m1 + m2
    msq = m * m
    m1sq = m1 * m1
    m2sq = m2 * m2
    eta = jnp.clip(m1 * m2 / msq, 0.0, 0.25)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    Shat = (chi1 * m1sq + chi2 * m2sq) / (m1sq + m2sq)
    Shat2 = Shat * Shat
    Shat3 = Shat2 * Shat
    chidiff = jnp.where(m2 > m1, -(chi1 - chi2), chi1 - chi2)
    chidiff2 = chidiff * chidiff
    sqrt2 = 2.0**0.5
    sqrt1m4eta = _sqrt_nonneg(1.0 - 4.0 * eta)

    b10, b20, b30, b50 = 0.346, 0.211, 0.128, -0.212
    a2, a3, a4 = 0.5609904135313374, -0.84667563764404, 3.145145224278187
    b1, b2, b3, b5 = (
        -0.2091189048177395,
        -0.19709136361080587,
        -0.1588185739358418,
        2.9852925538232014,
    )
    f20, f30, f50, f10, f21 = (
        4.271313308472851,
        31.08987570280556,
        1.5673498395263061,
        1.8083565298668276,
        0.0,
    )
    d10, d11, d20, d30, d31 = (
        -0.09803730445895877,
        -3.2283713377939134,
        0.01118530335431078,
        -0.01978238971523653,
        -4.91667749015812,
    )
    f11, f31, f51 = 15.738082204419655, -243.6299258830685, -0.5808669012986468

    Erad = (
        (
            ((1.0 + -2.0 / 3.0 * sqrt2) * eta + a2 * eta2 + a3 * eta3 + a4 * eta4)
            * (
                1.0
                + b10
                * b1
                * Shat
                * (f10 + f11 * eta + (16.0 - 16.0 * f10 - 4.0 * f11) * eta2)
                + b20
                * b2
                * Shat2
                * (f20 + f21 * eta + (16.0 - 16.0 * f20 - 4.0 * f21) * eta2)
                + b30
                * b3
                * Shat3
                * (f30 + f31 * eta + (16.0 - 16.0 * f30 - 4.0 * f31) * eta2)
            )
        )
        / (
            1.0
            + b50
            * b5
            * Shat
            * (f50 + f51 * eta + (16.0 - 16.0 * f50 - 4.0 * f51) * eta2)
        )
        + d10 * sqrt1m4eta * eta2 * (1.0 + d11 * eta) * chidiff
        + d30 * Shat * sqrt1m4eta * eta * (1.0 + d31 * eta) * chidiff
        + d20 * eta3 * chidiff2
    )
    return m * (1.0 - Erad)


_BHNS_MASS_COEFFS = (
    -1.83417425e-03, 2.39226041e-03, 4.29407902e-03, 9.79775571e-03,
    2.33868869e-07, -8.28090025e-07, -1.64315549e-06, 8.08340931e-06,
    -2.00726981e-02, 1.31986011e-01, 6.50754064e-02, -1.42749961e-01,
)  # fmt: skip
_BHNS_SPIN_COEFFS = (
    -5.44187381e-03, 7.91165608e-03, 2.33362046e-02, 2.47764497e-02,
    -8.56844797e-07, -2.81727682e-06, 6.61290966e-06, 4.28979016e-05,
    -3.04174272e-02, 2.54889050e-01, 1.47549350e-01, -4.27905832e-01,
)  # fmt: skip


def _bhns_model3a(nu, ai, lam, par):
    """Template 3D NSBH remnant model (Eq. 4-6 of 1903.11622)."""
    p110, p111, p120, p121, p210, p211, p220, p221, p310, p311, p320, p321 = par
    p11 = p110 * ai + p111
    p12 = p120 * ai + p121
    p21 = p210 * ai + p211
    p22 = p220 * ai + p221
    p31 = p310 * ai + p311
    p32 = p320 * ai + p321
    p0 = (p11 + p12 * nu) * nu
    p1 = (p21 + p22 * nu) * nu
    p2 = (p31 + p32 * nu) * nu
    return (1.0 + lam * p0 + lam**2 * p1) / (1.0 + lam * p2**2) ** 2


def _bhns_clamp(model, nu, chi1):
    model = jnp.where((chi1 < 0) & (nu < 0.188), 1.0, model)
    model = jnp.where(chi1 < -0.5, 1.0, model)
    return jnp.where(model > 1.0, 1.0, model)


def bhns_final_spin_aligned(m1, m2, chi1, lam):
    """``XLALBHNS_spin_aligned``: final spin of an aligned-spin BH-NS merger."""
    nu = m1 * m2 / (m1 + m2) ** 2
    model = _bhns_clamp(_bhns_model3a(nu, chi1, lam, _BHNS_SPIN_COEFFS), nu, chi1)
    return bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, 0.0) * model


def bhns_final_mass_aligned(m1, m2, chi1, lam):
    """``XLALBHNS_mass_aligned``: final mass (solar masses) of an aligned-spin BH-NS merger."""
    nu = m1 * m2 / (m1 + m2) ** 2
    model = _bhns_clamp(_bhns_model3a(nu, chi1, lam, _BHNS_MASS_COEFFS), nu, chi1)
    return bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, 0.0) * model


def omega_tilde(a: FloatLike) -> Complex[Array, ""]:
    """Dimensionless 220 QNM frequency fit (``XLALSimIMRPhenomNSBH_omega_tilde``, Eq. 22 of 1810.03550)."""
    kappa = jnp.sqrt(jnp.log(2.0 - a) / jnp.log(3.0))
    return 1.0 + kappa * (
        1.5578 * jnp.exp(1j * 2.9031)
        + 1.9510 * jnp.exp(1j * 5.9210) * kappa
        + 2.0997 * jnp.exp(1j * 2.7606) * kappa**2
        + 1.4109 * jnp.exp(1j * 5.9143) * kappa**3
        + 0.4106 * jnp.exp(1j * 2.7952) * kappa**4
    )


#######################################################################################################
### IMRPhenomC amplitude coefficients (LALSimIMRPhenomC_internals.c)
#######################################################################################################


def phenomC_amp_coeffs(
    eta: FloatLike, chi: FloatLike, diagnostics: bool = False
) -> dict[str, FloatLike]:
    """IMRPhenomC PN amplitude coefficients and phenomenological fits ``g1, del1, del2``.

    Args:
        eta: Symmetric mass ratio.
        chi: Reduced spin of the binary, ``(m1 chi1 + m2 chi2) / M`` (IMRPhenomB ``chi``).
        diagnostics: Add optional checkify annotations (static Python flag).

    The clamps applied by ``IMRPhenomNSBH_Core`` (``g1 >= 0``, ``del1 >= 0``, ``del2 >= 1e-4``) are included.
    """
    pi, pi2, gam = PI, PI * PI, EULERGAMMA
    eta2 = eta * eta
    chisum = chi + chi
    chiprod = chi * chi
    chi2 = chi * chi

    xdotaN = 64.0 * eta / 5.0
    xdota2 = -7.43 / 3.36 - 11.0 * eta / 4.0
    xdota3 = 4.0 * pi - 11.3 * chi / 1.2 + 19.0 * eta * chisum / 6.0
    xdota4 = (
        3.4103 / 1.8144
        + 5 * chi2
        + eta * (13.661 / 2.016 - chiprod / 8.0)
        + 5.9 * eta2 / 1.8
    )
    xdota5 = (
        -pi * (41.59 / 6.72 + 189.0 * eta / 8.0)
        - chi * (31.571 / 1.008 - 116.5 * eta / 2.4)
        + chisum * (21.863 * eta / 1.008 - 79.0 * eta2 / 6.0)
        - 3 * chi * chi2 / 4.0
        + 9.0 * eta * chi * chiprod / 4.0
    )
    xdota6 = (
        164.47322263 / 1.39708800
        - 17.12 * gam / 1.05
        + 16.0 * pi2 / 3
        - 8.56 * jnp.log(16.0) / 1.05
        + eta * (45.1 * pi2 / 4.8 - 561.98689 / 2.17728)
        + 5.41 * eta2 / 8.96
        - 5.605 * eta * eta2 / 2.592
        - 80.0 * pi * chi / 3.0
        + eta * chisum * (20.0 * pi / 3.0 - 113.5 * chi / 3.6)
        + chi2 * (64.153 / 1.008 - 45.7 * eta / 3.6)
        - chiprod * (7.87 * eta / 1.44 - 30.37 * eta2 / 1.44)
    )
    xdota6log = -856.0 / 105.0
    xdota7 = (
        -pi * (4.415 / 4.032 - 358.675 * eta / 6.048 - 91.495 * eta2 / 1.512)
        - chi * (252.9407 / 2.7216 - 845.827 * eta / 6.048 + 415.51 * eta2 / 8.64)
        + chisum
        * (
            158.0239 * eta / 5.4432
            - 451.597 * eta2 / 6.048
            + 20.45 * eta2 * eta / 4.32
            + 107.0 * eta * chi2 / 6.0
            - 5.0 * eta2 * chiprod / 24.0
        )
        + 12.0 * pi * chi2
        - chi2 * chi * (150.5 / 2.4 + eta / 8.0)
        + chi * chiprod * (10.1 * eta / 2.4 + 3.0 * eta2 / 8.0)
    )

    AN = 8.0 * eta * jnp.sqrt(pi / 5.0)
    A2 = (-107.0 + 55.0 * eta) / 42.0
    A3 = 2.0 * pi - 4.0 * chi / 3.0 + 2.0 * eta * chisum / 3.0
    A4 = -2.173 / 1.512 - eta * (10.69 / 2.16 - 2.0 * chiprod) + 2.047 * eta2 / 1.512
    A5 = -10.7 * pi / 2.1 + eta * (3.4 * pi / 2.1)
    A5imag = -24.0 * eta
    A6 = (
        270.27409 / 6.46800
        - 8.56 * gam / 1.05
        + 2.0 * pi2 / 3.0
        + eta * (4.1 * pi2 / 9.6 - 27.8185 / 3.3264)
        - 20.261 * eta2 / 2.772
        + 11.4635 * eta * eta2 / 9.9792
        - 4.28 * jnp.log(16.0) / 1.05
    )
    A6log = -428.0 / 105.0
    A6imag = 4.28 * pi / 1.05

    # Table II / Eq. 5.14 of 1005.3306: gamma_1, delta_1, delta_2
    g1 = (
        4.149e00 * chi
        - 4.070e00 * chi2
        - 8.752e01 * eta * chi
        - 4.897e01 * eta
        + 6.665e02 * eta2
    )
    del1 = (
        -5.472e-02 * chi
        + 2.094e-02 * chi2
        + 3.554e-01 * eta * chi
        + 1.151e-01 * eta
        + 9.640e-01 * eta2
    )
    del2 = (
        -1.235e00 * chi
        + 3.423e-01 * chi2
        + 6.062e00 * eta * chi
        + 5.949e00 * eta
        - 1.069e01 * eta2
    )

    # clamps in IMRPhenomNSBH_Core
    g1 = jnp.maximum(g1, 0.0)
    del1 = jnp.maximum(del1, 0.0)
    del2 = jnp.maximum(del2, 1.0e-4)

    # LAL ComputeIMRPhenomCParams: this BBH fit sets the SPA matching
    # frequency used by the negative-xdot guard, before NSBH corrections.
    etachi = eta * chi
    afin = (
        chi
        - 0.129 * chi * etachi
        - 0.384 * etachi * eta
        - 2.686 * etachi
        + 2.0 * jnp.sqrt(3.0) * eta
        - 3.454 * eta2
        + 2.353 * eta2 * eta
    )
    if diagnostics:
        checkify.debug_check(
            jnp.abs(afin) <= 1.0, "IMRPhenomNSBH PhenomC remnant spin exceeds unity"
        )
    Mf1 = 0.1 * (1.5251 - 1.1568 * (1.0 - jnp.abs(afin)) ** 0.1292) / (2.0 * PI)

    return {
        "xdotaN": xdotaN, "xdota2": xdota2, "xdota3": xdota3, "xdota4": xdota4,
        "xdota5": xdota5, "xdota6": xdota6, "xdota6log": xdota6log, "xdota7": xdota7,
        "AN": AN, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A5imag": A5imag,
        "A6": A6, "A6log": A6log, "A6imag": A6imag,
        "g1": g1, "del1": del1, "del2": del2, "d0": 0.015, "Mf1": Mf1,
    }  # fmt: skip


#######################################################################################################
### NSBH merger properties (ComputeIMRPhenomNSBHParams)
#######################################################################################################


def nsbh_merger_properties(
    mBH: FloatLike,
    mNS: FloatLike,
    chi_BH: FloatLike,
    lam: FloatLike,
    chi_NS: FloatLike = 0.0,
    del2: Optional[FloatLike] = None,
) -> dict[str, FloatLike]:
    """Port of ``ComputeIMRPhenomNSBHParams``.

    Args:
        mBH, mNS: Component masses (any unit, only the ratio matters).
        chi_BH: Aligned BH spin.
        lam: NS dimensionless tidal deformability.
        chi_NS: Aligned NS spin; only enters through the IMRPhenomC ``del2`` used when ``lam <= 1``.
        del2: IMRPhenomC ``delta_2`` (clamped). Computed from the masses and spins if not given.

    Returns:
        Everything ``ComputeIMRPhenomNSBHParams`` computes. Frequencies are dimensionless (``M f``), masses
        in units of the total initial mass (``final_mass``) or of the NS baryonic mass (``Mtorus``).
        ``merger_type`` is 0 disruptive, 1 mildly disruptive without torus, 2 mildly disruptive with
        torus, 3 non-disruptive.
    """
    M = mBH + mNS
    q = mBH / mNS
    if del2 is None:
        eta = mBH * mNS / (M * M)
        chi_eff = (mBH * chi_BH + mNS * chi_NS) / M
        del2 = phenomC_amp_coeffs(eta, chi_eff)["del2"]

    C = compactness_from_lambda(lam)
    Mtorus = torus_mass_fit(q, chi_BH, C)
    chif = bhns_final_spin_aligned(mBH, mNS, chi_BH, lam)
    final_mass = bhns_final_mass_aligned(mBH, mNS, chi_BH, lam) / M

    om = omega_tilde(chif)
    f_RD = jnp.real(om) / 2.0 / PI / final_mass
    mu = q * C
    xi = xi_tide(q, chi_BH, mu)
    rtide = xi * (1.0 - 2.0 * C) / mu  # follows arXiv:1207.6304v1 (in M_BH units)
    q_factor = jnp.real(om) / jnp.imag(om) / 2.0
    # a spinning BH with vanishing rtide (BH limit) gives an infinite tidal frequency; keep it finite so
    # that comparisons/windows saturate exactly like inf does in LAL
    f_tide = jnp.minimum(
        jnp.abs(f_gw_in_kerr(rtide, 1.0, chi_BH) * (1.0 + 1.0 / q)), 1e6
    )

    f_RD_tilde = 0.99 * 0.98 * f_RD
    lam2 = lam * lam
    gamma_correction = jnp.where(lam > 1.0, 1.25, 1.0 + 0.5 * lam - 0.25 * lam2)

    # disruptive-merger ingredients
    delta_2_prime_hi = 1.62496 * _window_plus(
        (f_tide - f_RD_tilde) / f_RD_tilde, 0.0188092, 0.338737
    )
    c_2 = del2 - 0.81248
    delta_2_prime_lo = del2 - 2 * c_2 * lam + c_2 * lam2
    delta_2_prime = jnp.where(lam > 1.0, delta_2_prime_hi, delta_2_prime_lo)
    sigma = delta_2_prime * f_RD / q_factor

    x_D = (
        Mtorus
        + 0.424912 * C
        + 0.363604 * jnp.sqrt(q / (1.0 + q) ** 2)
        - 0.0605591 * chi_BH
    )
    eps_ins_torus = 1.29971 - 1.61724 * x_D
    x_D_prime = (
        Mtorus
        - 0.132754 * C
        + 0.576669 * jnp.sqrt(q / (1.0 + q) ** 2)
        - 0.0603749 * chi_BH
        - 0.0601185 * chi_BH**2
        - 0.0729134 * chi_BH**3
    )
    sigma_tide_torus = 0.137722 - 0.293237 * x_D_prime

    # non-disruptive-merger ingredients
    ratio2 = ((f_tide - f_RD_tilde) / f_RD_tilde) ** 2
    x_ND = ratio2 - 0.571505 * C - 0.00508451 * chi_BH
    x_ND_prime = ratio2 - 0.657424 * C - 0.0259977 * chi_BH
    epsilon_tide_ND = 2.0 * _window_plus(x_ND, -0.0796251, 0.0801192)
    sigma_tide_ND = 2.0 * _window_minus(x_ND_prime, -0.206465, 0.226844)

    has_torus = Mtorus > 0.0
    disruptive = f_tide < f_RD

    # --- f_tide < f_RD (mildly disruptive or totally disruptive)
    eps_ins_D = jnp.minimum(eps_ins_torus, 1.0)
    f1_md = (1.0 - 1.0 / q) * f_RD_tilde + eps_ins_D * f_tide / q
    f2_md = (1.0 - 1.0 / q) * f_RD_tilde + f_tide / q
    f0_PN_D = jnp.where(has_torus, f_tide, f1_md)
    f0_PM_D = jnp.where(has_torus, f_tide, f2_md)
    sigma_tide_D = jnp.where(
        has_torus, sigma_tide_torus, 0.5 * (sigma_tide_torus + sigma_tide_ND)
    )

    # --- f_tide >= f_RD (mildly disruptive or non-disruptive)
    f0_ND = jnp.where(
        lam > 1.0, f_RD_tilde, (1.0 - 0.02 * lam + 0.01 * lam2) * 0.98 * f_RD
    )
    eps_ins_ND = jnp.where(has_torus, eps_ins_torus, 1.0)

    epsilon_tide = jnp.where(disruptive, 0.0, epsilon_tide_ND)
    epsilon_ins = jnp.where(disruptive, eps_ins_D, eps_ins_ND)
    sigma_tide = jnp.where(disruptive, sigma_tide_D, sigma_tide_ND)
    f0_PN = jnp.where(disruptive, f0_PN_D, f0_ND)
    f0_PM = jnp.where(disruptive, f0_PM_D, f0_ND)
    f0_RD = jnp.where(disruptive, 0.0, f0_ND)
    merger_type = jnp.where(
        disruptive,
        jnp.where(has_torus, 0.0, 1.0),
        jnp.where(has_torus, 2.0, 3.0),
    )

    return {
        "compactness": C, "Mtorus": Mtorus, "chif": chif, "final_mass": final_mass,
        "f_RD": f_RD, "q_factor": q_factor, "f_tide": f_tide,
        "gamma_correction": gamma_correction, "delta_2_prime": delta_2_prime,
        "sigma": sigma, "sigma_tide": sigma_tide, "epsilon_ins": epsilon_ins,
        "epsilon_tide": epsilon_tide, "f0_PN": f0_PN, "f0_PM": f0_PM, "f0_RD": f0_RD,
        "merger_type": merger_type,
    }  # fmt: skip


#######################################################################################################
### Amplitude
#######################################################################################################


def nsbh_amplitude(
    Mf: Float[Array, " n_freq"],
    cc: dict[str, FloatLike],
    props: dict[str, FloatLike],
    diagnostics: bool = False,
) -> Float[Array, " n_freq"]:
    """``|PhenomNSBHAmplitudeOneFrequency|`` (dimensionless, before ``amp0 * ylm_fac``) at geometric frequency ``Mf``.

    LAL's amplitude is ``-(aPN wMinus + aPM wMinus + aRD wPlus)``, i.e. non-positive; the sign is returned
    separately through the phase (see module docstring).
    """
    if Mf.size == 0:
        raise ValueError("IMRPhenomNSBH requires a nonempty frequency array")
    v = (PI * Mf) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v5 = v3 * v2
    v6 = v3 * v3
    v7 = v4 * v3
    v10 = v5 * v5
    logv2 = jnp.log(v2)

    xdot = (
        1.0
        + cc["xdota2"] * v2
        + cc["xdota3"] * v3
        + cc["xdota4"] * v4
        + cc["xdota5"] * v5
        + (cc["xdota6"] + cc["xdota6log"] * logv2) * v6
        + cc["xdota7"] * v7
    )
    xdot = xdot * (cc["xdotaN"] * v10)
    valid = ~jnp.any((xdot < 0.0) & (Mf < cc["Mf1"])) & jnp.isfinite(cc["Mf1"])
    if diagnostics:
        checkify.debug_check(
            valid,
            "IMRPhenomNSBH omegaDot < 0 below the SPA matching frequency, or invalid matching frequency",
        )
    omgdot = 1.5 * v * xdot
    ampfac = jnp.sqrt(jnp.abs(PI / omgdot))
    AmpPNre = (
        ampfac
        * cc["AN"]
        * v2
        * (
            1.0
            + cc["A2"] * v2
            + cc["A3"] * v3
            + cc["A4"] * v4
            + cc["A5"] * v5
            + (cc["A6"] + cc["A6log"] * logv2) * v6
        )
    )
    AmpPNim = ampfac * cc["AN"] * v2 * (cc["A5imag"] * v5 + cc["A6imag"] * v6)
    aPN = jnp.sqrt(AmpPNre * AmpPNre + AmpPNim * AmpPNim)

    aPM = props["gamma_correction"] * cc["g1"] * Mf ** (5.0 / 6.0)

    sig2 = props["sigma"] * props["sigma"]
    L = sig2 / ((Mf - props["f_RD"]) ** 2 + sig2 / 4.0)
    aRD = props["epsilon_tide"] * cc["del1"] * L * Mf ** (-7.0 / 6.0)

    d = cc["d0"] + props["sigma_tide"]
    wMinus_PN = _w_minus(Mf, props["epsilon_ins"] * props["f0_PN"], d)
    wMinus_PM = _w_minus(Mf, props["f0_PM"], d)
    wPlus_RD = _w_plus(Mf, props["f0_RD"], d)
    amp = aPN * wMinus_PN + aPM * wMinus_PM + aRD * wPlus_RD
    return jnp.where(valid, amp, jnp.nan)


#######################################################################################################
### Phase
#######################################################################################################


def quadmon_from_lambda(lam: FloatLike) -> FloatLike:
    """NS spin-induced quadrupole parameter, LAL's ``XLALSimInspiralEOSQfromLambda`` (used by NSBH, 1303.1528).

    Not the same fit as ``XLALSimUniversalRelationQuadMonVSlambda2Tidal`` used by NRTidalv2 (cf. ``utils.tidal``).
    """
    coeffs = jnp.array([0.194, 0.0936, 0.0474, -0.00421, 0.000123])
    loglam = jnp.log(jnp.maximum(lam, 0.5))
    return jnp.where(lam < 0.5, 1.0, jnp.exp(universal_relation(coeffs, loglam)))


def _qm_phase_correction(fM_s, m1, m2, chi1, chi2, lambda2):
    """Phase shift from the NS spin-induced quadrupole (``dQuadMon2 = Q(lambda2) - 1``; BH has ``dQuadMon1 = 0``)
    on top of IMRPhenomD's BBH inspiral phase, from the 2PN and 3PN terms of ``XLALSimInspiralPNPhasing_F2``."""
    m1_s, m2_s = m1 * MTSUN, m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    X2 = m2_s / M_s
    dquadmon2 = quadmon_from_lambda(lambda2) - 1.0

    delta_phi4 = (
        (get_4PNQM2SOCoeff(X2) + get_4PNQM2SCoeff(X2)) * dquadmon2 * chi2 * chi2
    )
    delta_phi6 = get_6PNQM2SCoeff(X2) * dquadmon2 * chi2 * chi2
    v = (PI * fM_s) ** (1.0 / 3.0)
    return 3.0 / (128.0 * eta) * (delta_phi4 / v + delta_phi6 * v)


def _fRD_fdamp_from_chif(m1, m2, chi1, chi2, chif):
    """IMRPhenomD ``fring`` / ``fdamp`` evaluated at an externally supplied final spin (Hz)."""
    m1_s, m2_s = m1 * MTSUN, m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    Erad = EradRational0815(eta, chi1, chi2)
    f_RD = _qnm_interp(chif, QNMData_fRD) / (1.0 - Erad)
    f_damp = _qnm_interp(chif, QNMData_fdamp) / (1.0 - Erad)
    return f_RD / M_s, f_damp / M_s


def _phenomD_phase_at(f, theta_bbh, coeffs, transition_freqs, lambda2):
    """IMRPhenomD phase (inspiral + NS quadrupole-monopole shift, intermediate, merger-ringdown), C^1 glued."""
    m1, m2, chi1, chi2 = theta_bbh
    M_s = (m1 + m2) * MTSUN
    f1, f2, _, _, f_RD, f_damp = transition_freqs

    def inspiral_phase(fM_s):
        return get_inspiral_phase(fM_s, theta_bbh, coeffs) + _qm_phase_correction(
            fM_s, m1, m2, chi1, chi2, lambda2
        )

    phi_Ins = inspiral_phase(f * M_s)
    phi_Ins_f1, dphi_Ins_f1 = jax.value_and_grad(inspiral_phase)(f1 * M_s)
    phi_IIa_f1, dphi_IIa_f1 = jax.value_and_grad(get_IIa_raw_phase)(
        f1 * M_s, theta_bbh, coeffs
    )
    beta1_correction = dphi_Ins_f1 - dphi_IIa_f1
    beta0 = phi_Ins_f1 - beta1_correction * (f1 * M_s) - phi_IIa_f1

    def phi_IIa_func(fM_s):
        return get_IIa_raw_phase(fM_s, theta_bbh, coeffs) + beta1_correction * fM_s

    phi_IIa = phi_IIa_func(f * M_s) + beta0
    phi_IIa_f2, dphi_IIa_f2 = jax.value_and_grad(phi_IIa_func)(f2 * M_s)
    phi_IIb_f2, dphi_IIb_f2 = jax.value_and_grad(get_IIb_raw_phase)(
        f2 * M_s, theta_bbh, coeffs, f_RD, f_damp
    )
    a1_correction = dphi_IIa_f2 - dphi_IIb_f2
    a0 = phi_IIa_f2 + beta0 - a1_correction * (f2 * M_s) - phi_IIb_f2
    phi_IIb = (
        get_IIb_raw_phase(f * M_s, theta_bbh, coeffs, f_RD, f_damp)
        + a0
        + a1_correction * (f * M_s)
    )

    return (
        phi_Ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_IIa * jnp.heaviside(f2 - f, 0.5)
        + phi_IIb * jnp.heaviside(f - f2, 0.5)
    )


#######################################################################################################
### Master functions
#######################################################################################################


def _amp_phase(
    f: Float[Array, " n_freq"],
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    lambda2: FloatLike,
    d_L: FloatLike,
    tc: FloatLike,
    phase_c: FloatLike,
    f_ref: float,
    f_max: Optional[float],
    diagnostics: bool = False,
) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
    """Amplitude ``A >= 0`` and phase of ``h0 = A exp(i phase)`` (see module docstring for the sign convention)."""
    if f.size == 0:
        raise ValueError("IMRPhenomNSBH requires a nonempty frequency array")
    M = m1 + m2
    M_s = M * MTSUN
    eta = m1 * m2 / (M * M)
    valid_tides = (lambda2 >= 0.0) & (lambda2 <= 5000.0)
    if diagnostics:
        checkify.debug_check(
            valid_tides, "IMRPhenomNSBH requires lambda_2 in [0, 5000]"
        )

    # ---- amplitude
    chi_eff = (m1 * chi1 + m2 * chi2) / M
    cc = phenomC_amp_coeffs(eta, chi_eff, diagnostics=diagnostics)
    props = nsbh_merger_properties(m1, m2, chi1, lambda2, chi_NS=chi2, del2=cc["del2"])
    amp = nsbh_amplitude(f * M_s, cc, props, diagnostics=diagnostics)
    # amp0 * ylm_fac, with the distance in metres
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * MTSUN / (d_L * MPC)
    A = amp0 * amp

    # ---- phase
    theta_bbh = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(theta_bbh)
    f_RD, f_damp = _fRD_fdamp_from_chif(m1, m2, chi1, chi2, props["chif"])
    transition_freqs = get_transition_frequencies_from_fRD_fdamp(
        theta_bbh, coeffs[5], coeffs[6], f_RD, f_damp
    )

    Psi = _phenomD_phase_at(f, theta_bbh, coeffs, transition_freqs, lambda2)
    Psi_ref = _phenomD_phase_at(
        jnp.array([f_ref]), theta_bbh, coeffs, transition_freqs, lambda2
    )[0]
    # time shift: derivative of the merger-ringdown ansatz at f_max (NOT at the peak, cf. PhenomD);
    # without f_max, LAL's default upper frequency Mf = 0.2
    Mf_max = 0.2 if f_max is None else f_max * M_s
    t0 = jax.grad(get_IIb_raw_phase)(Mf_max, theta_bbh, coeffs, f_RD, f_damp)
    Mf_ref = f_ref * M_s

    x = (PI * M_s * f) ** (2.0 / 3.0)
    # only the NS is deformable: lambda_1 = 0
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, 0.0, lambda2])
    psi_T = get_tidal_phase(x, theta_intrinsic, get_kappa(theta_intrinsic))

    Psi = Psi + psi_T - t0 * (f * M_s - Mf_ref) - Psi_ref
    Psi = Psi + 2.0 * PI * f * tc - 2.0 * phase_c
    # +pi carries the sign of LAL's negative amplitude
    valid = valid_tides & jnp.all(jnp.isfinite(amp))
    return jnp.where(valid, A, jnp.nan), jnp.where(valid, -Psi + PI, jnp.nan)


def gen_IMRPhenomNSBH(
    f: Float[Array, " n_freq"],
    params: Float[Array, "8"],
    f_ref: float,
    f_max: Optional[float] = None,
    *,
    diagnostics: bool = False,
) -> Complex[Array, " n_freq"]:
    """
    Generate the IMRPhenomNSBH strain ``h0``.

    params = [Mchirp, eta, chi_BH, chi_NS, lambda_NS, D, tc, phic]; body 1 is the black
    hole (the heavier body), body 2 the neutron star.

    Args:
        f (Array): Frequencies in Hz.
        params (Array): Parameters, ordered as above (D in Mpc).
        f_ref (float): Reference frequency in Hz.
        f_max (Optional[float]): Frequency in Hz at which the time shift is anchored;
            ``None`` uses LAL's default upper frequency ``Mf = 0.2``.
        diagnostics (bool): Add optional checkify annotations. Keep this static
            under ``jit``; wrap in ``checkify.checkify`` to collect error messages.

    Returns:
        h0 (array): Strain.
    """
    m1, m2 = _component_masses(params[0], params[1], diagnostics=diagnostics)
    chi1, chi2, lambda2, d_L, tc, phase_c = params[2:]
    A, phi = _amp_phase(
        f,
        m1,
        m2,
        chi1,
        chi2,
        lambda2,
        d_L,
        tc,
        phase_c,
        f_ref,
        f_max,
        diagnostics=diagnostics,
    )
    return A * jnp.cos(phi) + 1j * (A * jnp.sin(phi))


def gen_IMRPhenomNSBH_hphc(
    f: Float[Array, " n_freq"],
    params: Float[Array, "9"],
    f_ref: float,
    f_max: Optional[float] = None,
    *,
    diagnostics: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Generate the IMRPhenomNSBH plus and cross polarizations.

    params = [Mchirp, eta, chi_BH, chi_NS, lambda_NS, D, tc, phic, inclination]; see
    ``gen_IMRPhenomNSBH`` for the other arguments.

    Returns:
        hp (array): Strain of the plus polarization
        hc (array): Strain of the cross polarization

    ``diagnostics`` is a static flag with the same meaning as in
    ``gen_IMRPhenomNSBH``.
    """
    iota = params[-1]
    h0 = gen_IMRPhenomNSBH(f, params[:-1], f_ref, f_max=f_max, diagnostics=diagnostics)
    hp = h0 * (0.5 * (1.0 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)
    return hp, hc


def _split_params(params: Mapping[str, FloatLike], diagnostics: bool = False):
    # Mc_eta_to_ms returns m1 >= m2: m1 is the BH, m2 the NS
    m1, m2 = _component_masses(params["M_c"], params["eta"], diagnostics=diagnostics)
    return (
        m1,
        m2,
        params["s1_z"],
        params["s2_z"],
        params["lambda_2"],
        params["d_L"],
        params["phase_c"],
    )


@register("IMRPhenomNSBH", is_tidal=True, is_precessing=False, highest_mode=2)
class IMRPhenomNSBH(AmplitudePhaseWaveform, DistanceScaledWaveform):
    """IMRPhenomNSBH frequency-domain waveform (non-precessing, single deformable body: the NS).

    Invalid mass ratios (outside ``[1, 100]``), tidal deformabilities (outside
    ``[0, 5000]``), or a negative inspiral frequency derivative below LAL's
    SPA matching frequency return NaN for the entire sample. Evaluation remains
    compatible with ``jit``, ``vmap`` and ``scan``; other batch samples continue.
    Set ``diagnostics=True`` and wrap evaluation in ``checkify.checkify`` for
    optional error diagnostics. Normal evaluation has no error effects.
    Errors are returned as values; throwing them is the caller's choice.
    There is no neutron-star mass cut. Nonzero NS spin and NS masses below one
    solar mass are outside LAL's calibration, but are not rejected here.

    Attributes:
        f_ref (float): Reference frequency in Hz.
        f_max (Optional[float]): Frequency in Hz at which the coalescence-time shift is anchored
            (LAL's upper frequency ``f_max``). ``None`` uses LAL's default, ``Mf = 0.2``.
        diagnostics (bool): Enable optional checkify annotations. Defaults to False.
    """

    f_ref: float
    f_max: Optional[float]

    def __init__(
        self,
        f_ref: float = 20.0,
        f_max: Optional[float] = None,
        *,
        diagnostics: bool = False,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            f_max (Optional[float]): Time-shift anchor frequency in Hz. Defaults to None
                (``Mf = 0.2``).
            diagnostics (bool): Add optional checkify annotations. Defaults to False.
        """
        self.f_ref = f_ref
        self.f_max = f_max
        self.diagnostics = diagnostics

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("M_c", "eta", "s1_z", "s2_z", "lambda_2", "d_L", "phase_c", "iota")

    def amplitude(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Amplitude of ``h0`` (non-negative; includes distance scaling)."""
        m1, m2, chi1, chi2, lam, d_L, phase_c = _split_params(params, self.diagnostics)
        A, _ = _amp_phase(
            frequency,
            m1,
            m2,
            chi1,
            chi2,
            lam,
            d_L,
            0.0,
            phase_c,
            self.f_ref,
            self.f_max,
            diagnostics=self.diagnostics,
        )
        return A

    def phase(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> Float[Array, " n_freq"]:
        """Phase of ``h0`` (includes the ``pi`` that carries LAL's negative amplitude)."""
        m1, m2, chi1, chi2, lam, d_L, phase_c = _split_params(params, self.diagnostics)
        _, phi = _amp_phase(
            frequency,
            m1,
            m2,
            chi1,
            chi2,
            lam,
            d_L,
            0.0,
            phase_c,
            self.f_ref,
            self.f_max,
            diagnostics=self.diagnostics,
        )
        return phi

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomNSBH waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params: Source parameters with keys ``M_c``, ``eta``, ``s1_z`` (BH),
                ``s2_z`` (NS), ``lambda_2`` (NS), ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["lambda_2"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomNSBH_hphc(
            frequency, theta, self.f_ref, f_max=self.f_max, diagnostics=self.diagnostics
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomNSBH(f_ref={self.f_ref}, f_max={self.f_max}, diagnostics={self.diagnostics})"
