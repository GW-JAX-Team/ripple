"""
SpinTaylor Euler angle computation for IMRPhenomXPrecVersion=320.

Pipeline overview:
  Stage 1 — SpinTaylor ODE integration (inspiral)
  Stage 2 — Euler angle extraction + spline construction
  Stage 3 — MRD analytical continuation coefficients
  Stage 4 — Gamma integration via Boole's rule (continuous through MRD)
  Stage 5 — Reference-point offsets and final angle output

Units throughout: geometric (G = c = M_total = 1) unless noted as _SI.
"""

import math
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from ..constants import EULERGAMMA, G, C, PI
import numpy as np

# G * M_sun / c^3  in seconds  (= LAL_MTSUN_SI)
_MTSUN_SI = G * 1.988409870698050731911960804878414216e30 / C**3


# =============================================================================
# PN coefficient helpers
# (mirrors LALSimInspiralPNCoefficients.c and LALSimInspiralSpinTaylor.c)
# =============================================================================


def _wdot_0PN(eta):
    return 96.0 / 5.0 * eta


def _wdot_2PN(eta):
    return -(743.0 + 924.0 * eta) / 336.0


def _wdot_3PN(_eta):
    return 4.0 * math.pi


def _wdot_3PN_SO(mByM):
    """1.5PN spin-orbit coefficient for domega (T4). LALSimInspiralPNCoefficients.c:1841"""
    return -19.0 / 6.0 - 25.0 / (4.0 * mByM)


def _wdot_4PN(eta):
    return (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0


def _wdot_4PN_S1S2_avg(eta):
    """LALSimInspiralPNCoefficients.c:1876"""
    return -247.0 / (48.0 * eta)


def _wdot_4PN_S1OS2O_avg(eta):
    """LALSimInspiralPNCoefficients.c:1883"""
    return 721.0 / (48.0 * eta)


def _wdot_4PN_S1S1_avg(mByM):
    """LALSimInspiralPNCoefficients.c:1904"""
    return 7.0 / (96.0 * mByM**2)


def _wdot_4PN_S1OS1O_avg(mByM):
    """LALSimInspiralPNCoefficients.c:1911"""
    return -1.0 / (96.0 * mByM**2)


def _wdot_4PN_QMS1S1_avg(mByM):
    """LALSimInspiralPNCoefficients.c:1939"""
    return -2.5 / mByM**2


def _wdot_4PN_QMS1OS1O_avg(mByM):
    """LALSimInspiralPNCoefficients.c:1946"""
    return 7.5 / mByM**2


def _wdot_5PN(eta):
    return -(4159.0 + 15876.0 * eta) * math.pi / 672.0


def _wdot_5PN_SO(mByM):
    """LALSimInspiralPNCoefficients.c:1969"""
    return (
        -809.0 / (84.0 * mByM)
        + 13.795 / 1.008
        - 527.0 * mByM / 24.0
        - 79.0 * mByM**2 / 6.0
    )


def _wdot_6PN(eta):
    return (
        16447.322263 / 139.7088
        - 1712.0 / 105.0 * EULERGAMMA
        - 561.98689 / 2.17728 * eta
        + math.pi**2 * (16.0 / 3.0 + 451.0 / 48.0 * eta)
        + 541.0 / 896.0 * eta**2
        - 5605.0 / 2592.0 * eta**3
        - 856.0 / 105.0 * math.log(16.0)
    )


def _wdot_6PN_log(_eta):
    """Coefficient of log(v) at 3PN. LALSimInspiralPNCoefficients.c:1987"""
    return -1712.0 / 105.0


def _wdot_6PN_SO(mByM):
    """LALSimInspiralPNCoefficients.c:1994"""
    return math.pi * (-37.0 / 3.0 - 151.0 / (6.0 * mByM))


def _wdot_6PN_S1S2_avg(eta):
    """LALSimInspiralPNCoefficients.c:2029"""
    return 108.79 / (6.72 * eta) + 75.25 / 2.88


def _wdot_6PN_S1OS2O_avg(eta):
    """LALSimInspiralPNCoefficients.c:2036"""
    return 162.25 / (2.24 * eta) - 129.31 / 2.88


def _wdot_6PN_S1S1_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2071"""
    return 101.9 / (6.4 * mByM**2) + 2.51 / (5.76 * mByM) + 13.33 / 5.76


def _wdot_6PN_S1OS1O_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2078"""
    return -49.3 / (6.4 * mByM**2) + 197.47 / (5.76 * mByM) + 56.45 / 5.76


def _wdot_6PN_QMS1S1_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2106"""
    return -6.59 / (2.24 * mByM**2) + 7.3 / (4.8 * mByM) - 43.0 / 4.0


def _wdot_6PN_QMS1OS1O_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2113"""
    return 19.77 / (2.24 * mByM**2) - 7.3 / (1.6 * mByM) + 129.0 / 4.0


def _wdot_7PN(eta):
    """LALSimInspiralPNCoefficients.c:2120"""
    return math.pi / 12096.0 * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2)


# --- spin precession coefficients ---


def _Sdot_3PN(mByM):
    """Leading spin precession. LALSimInspiralPNCoefficients.c:2328"""
    return 1.5 - mByM - mByM**2 / 2.0


# Orbit-averaged S1S2 constants (LALSimInspiralPNCoefficients.c:2344, 2347)
_SDOT_4PN_S2_AVG = 0.5
_SDOT_4PN_S2O_AVG = -1.5


def _Sdot_4PN_QMS1O_avg(mByM):
    """QM self-spin precession. LALSimInspiralPNCoefficients.c:2353"""
    return 1.5 * (1.0 - 1.0 / mByM)


def _Sdot_5PN(mByM):
    """NNLO spin precession. LALSimInspiralPNCoefficients.c:2370"""
    return (
        9.0 / 8.0
        - mByM / 2.0
        + 7.0 * mByM**2 / 12.0
        - 7.0 * mByM**3 / 6.0
        - mByM**4 / 24.0
    )


def _Sdot_6PN_S2_avg(mByM):
    """
    Orbit-averaged S1S2 term in dS1 (coefficient of S2 x S1).
    LALSimInspiralPNCoefficients.c:2416
    = S2Coeff + 0.5*(S2nCoeff + S2vCoeff)
    = (-1.5 - mByM) + 0.5*((1.5 + 2*mByM + mByM^2) + (1.5 + mByM))
    """
    return 0.5 * mByM + 0.5 * mByM**2


def _Sdot_6PN_S1O_avg(mByM):
    """
    Coefficient of (LN.S1)(LN x S1) in dS1.
    LALSimInspiralPNCoefficients.c:2423
    = -0.5*(S1nCoeff + S1vCoeff)
    = -0.5*((3.5 - 3/mByM - 0.5*mByM^2) + (3 - 1.5*mByM - 1.5/mByM))
    """
    return -3.25 + 2.25 / mByM + 0.75 * mByM + 0.25 * mByM**2


def _Sdot_6PN_S2O_avg(mByM):
    """
    Coefficient of (LN.S2)(LN x S1) in dS1.
    LALSimInspiralPNCoefficients.c:2430
    = -0.5*(S2nCoeff + S2vCoeff)
    = -0.5*((1.5 + 2*mByM + mByM^2) + (1.5 + mByM))
    """
    return -1.5 - 1.5 * mByM - 0.5 * mByM**2


def _Sdot_6PN_QMSO_avg(mByM):
    """
    QM coefficient of (LN.S1)(LN x S1) in dS1 at 3PN.
    LALSimInspiralPNCoefficients.c:2455
    = -0.5*(QMSnCoeff + QMSvCoeff)
    = -0.5*(3*(0.5/mByM + 1 - mByM - 0.5*mByM^2) + 3*(1/mByM - 1))
    """
    return -2.25 / mByM + 1.5 * mByM + 0.75 * mByM**2


def _LDot_3PN_SO(mByM):
    """Coefficient for omegashift. LALSimInspiralPNCoefficients.c:2274"""
    return 0.5 + 1.5 / mByM


def _L_2PN(eta):
    """LALSimInspiralPNCoefficients.c:2181"""
    return 1.5 + eta / 6.0


def _L_4PN(eta):
    """LALSimInspiralPNCoefficients.c:2253"""
    return 27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0


def _L_3PN_Si_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2214"""
    return -0.75 - 0.25 / mByM


def _L_3PN_SiL_avg(mByM):
    """LALSimInspiralPNCoefficients.c:2221"""
    return -(1.0 / 3.0 + 9.0 / mByM) / 4.0


# =============================================================================
# Parameter struct
# =============================================================================


@dataclass
class SpinTaylorT4Params:
    """
    Pre-computed PN coefficients for SpinTaylorT4.

    All quantities are dimensionless (geometric units, M=1).
    Build with build_spin_taylor_params().
    """

    # mass ratios
    eta: float
    m1M: float  # m1 / M
    m2M: float  # m2 / M

    # domega: non-spinning PN series coefficients (indices 0..7 = 0PN..3.5PN)
    wdotnewt: float  # Newtonian prefactor (96/5 * eta)
    wdotcoeff: np.ndarray  # shape (8,); multiplied by v^(2*index)
    wdotlogcoeff: float  # coefficient of log(v) at 3PN

    # domega: tidal (zero for BBH)
    wdottidal10: float
    wdottidal12: float

    # domega: spin-orbit
    wdot3S1O: float  # 1.5PN SO, body 1
    wdot3S2O: float  # 1.5PN SO, body 2
    wdot5S1O: float  # 2.5PN SO, body 1
    wdot5S2O: float  # 2.5PN SO, body 2
    wdot6S1O: float  # 3PN SO, body 1
    wdot6S2O: float  # 3PN SO, body 2

    # domega: spin-spin (2PN, orbit-averaged)
    wdot4S1S2Avg: float
    wdot4S1OS2OAvg: float
    wdot4S1S1Avg: float
    wdot4S2S2Avg: float
    wdot4S1OS1OAvg: float
    wdot4S2OS2OAvg: float
    wdot4QMS1S1Avg: float
    wdot4QMS2S2Avg: float
    wdot4QMS1OS1OAvg: float
    wdot4QMS2OS2OAvg: float

    # domega: spin-spin (3PN, orbit-averaged)
    wdot6S1S2Avg: float
    wdot6S1OS2OAvg: float
    wdot6S1S1Avg: float
    wdot6S2S2Avg: float
    wdot6S1OS1OAvg: float
    wdot6S2OS2OAvg: float
    wdot6QMS1S1Avg: float
    wdot6QMS2S2Avg: float
    wdot6QMS1OS1OAvg: float
    wdot6QMS2OS2OAvg: float

    # spin precession (dS1, dS2)
    S1dot3: float  # leading (1.5PN), body 1
    S2dot3: float  # leading (1.5PN), body 2
    S1dot4S2Avg: float  # NLO S1S2, avg
    S1dot4S2OAvg: float  # NLO (LN.S2)(LNxS1), avg
    S1dot4QMS1OAvg: float  # NLO QM self-spin, body 1
    S2dot4QMS2OAvg: float  # NLO QM self-spin, body 2
    S1dot5: float  # NNLO, body 1
    S2dot5: float  # NNLO, body 2
    S1dot6S2Avg: float  # N3L S1S2 in dS1
    S2dot6S1Avg: float  # N3L S1S2 in dS2
    S1dot6S2OAvg: float  # N3L (LN.S2)(LNxS1)
    S1dot6S1OAvg: float  # N3L (LN.S1)(LNxS1)
    S2dot6S1OAvg: float  # N3L (LN.S1)(LNxS2)
    S2dot6S2OAvg: float  # N3L (LN.S2)(LNxS2)
    S1dot6QMS1OAvg: float  # N3L QM body 1
    S2dot6QMS2OAvg: float  # N3L QM body 2

    # phase correction (omegashift at 1.5PN)
    omegashiftS1: float
    omegashiftS2: float

    # flags
    lscorr: int  # 1 = include spin-dependent L corrections to dLNhat; 0 = off


def build_spin_taylor_params(m1M, eta, quadparam1=1.0, quadparam2=1.0, lscorr=0):
    """
    Pre-compute all PN coefficients for SpinTaylorT4.

    Args:
        m1M:        m1 / (m1 + m2)  (dimensionless)
        eta:        symmetric mass ratio
        quadparam1: quadrupole parameter body 1 (1 for BH, ~2-12 for NS)
        quadparam2: quadrupole parameter body 2
        lscorr:     include spin-dependent L corrections to dLNhat (default 0)

    Returns:
        SpinTaylorT4Params
    """
    m2M = 1.0 - m1M

    wdotcoeff = np.array(
        [
            1.0,  # 0PN
            0.0,  # 0.5PN (vanishes)
            _wdot_2PN(eta),  # 1PN
            _wdot_3PN(eta),  # 1.5PN (non-spinning)
            _wdot_4PN(eta),  # 2PN
            _wdot_5PN(eta),  # 2.5PN (non-spinning)
            _wdot_6PN(eta),  # 3PN
            _wdot_7PN(eta),  # 3.5PN
        ]
    )

    return SpinTaylorT4Params(
        eta=eta,
        m1M=m1M,
        m2M=m2M,
        wdotnewt=_wdot_0PN(eta),
        wdotcoeff=wdotcoeff,
        wdotlogcoeff=_wdot_6PN_log(eta),
        wdottidal10=0.0,
        wdottidal12=0.0,
        # spin-orbit in domega
        wdot3S1O=_wdot_3PN_SO(m1M),
        wdot3S2O=_wdot_3PN_SO(m2M),
        wdot5S1O=_wdot_5PN_SO(m1M),
        wdot5S2O=_wdot_5PN_SO(m2M),
        wdot6S1O=_wdot_6PN_SO(m1M),
        wdot6S2O=_wdot_6PN_SO(m2M),
        # spin-spin domega (2PN)
        wdot4S1S2Avg=_wdot_4PN_S1S2_avg(eta),
        wdot4S1OS2OAvg=_wdot_4PN_S1OS2O_avg(eta),
        wdot4S1S1Avg=_wdot_4PN_S1S1_avg(m1M),
        wdot4S2S2Avg=_wdot_4PN_S1S1_avg(m2M),
        wdot4S1OS1OAvg=_wdot_4PN_S1OS1O_avg(m1M),
        wdot4S2OS2OAvg=_wdot_4PN_S1OS1O_avg(m2M),
        wdot4QMS1S1Avg=quadparam1 * _wdot_4PN_QMS1S1_avg(m1M),
        wdot4QMS2S2Avg=quadparam2 * _wdot_4PN_QMS1S1_avg(m2M),
        wdot4QMS1OS1OAvg=quadparam1 * _wdot_4PN_QMS1OS1O_avg(m1M),
        wdot4QMS2OS2OAvg=quadparam2 * _wdot_4PN_QMS1OS1O_avg(m2M),
        # spin-spin domega (3PN)
        wdot6S1S2Avg=_wdot_6PN_S1S2_avg(eta),
        wdot6S1OS2OAvg=_wdot_6PN_S1OS2O_avg(eta),
        wdot6S1S1Avg=_wdot_6PN_S1S1_avg(m1M),
        wdot6S2S2Avg=_wdot_6PN_S1S1_avg(m2M),
        wdot6S1OS1OAvg=_wdot_6PN_S1OS1O_avg(m1M),
        wdot6S2OS2OAvg=_wdot_6PN_S1OS1O_avg(m2M),
        wdot6QMS1S1Avg=quadparam1 * _wdot_6PN_QMS1S1_avg(m1M),
        wdot6QMS2S2Avg=quadparam2 * _wdot_6PN_QMS1S1_avg(m2M),
        wdot6QMS1OS1OAvg=quadparam1 * _wdot_6PN_QMS1OS1O_avg(m1M),
        wdot6QMS2OS2OAvg=quadparam2 * _wdot_6PN_QMS1OS1O_avg(m2M),
        # spin precession
        S1dot3=_Sdot_3PN(m1M),
        S2dot3=_Sdot_3PN(m2M),
        S1dot4S2Avg=_SDOT_4PN_S2_AVG,
        S1dot4S2OAvg=_SDOT_4PN_S2O_AVG,
        S1dot4QMS1OAvg=quadparam1 * _Sdot_4PN_QMS1O_avg(m1M),
        S2dot4QMS2OAvg=quadparam2 * _Sdot_4PN_QMS1O_avg(m2M),
        S1dot5=_Sdot_5PN(m1M),
        S2dot5=_Sdot_5PN(m2M),
        S1dot6S2Avg=_Sdot_6PN_S2_avg(m1M),
        S2dot6S1Avg=_Sdot_6PN_S2_avg(m2M),
        S1dot6S2OAvg=_Sdot_6PN_S2O_avg(m1M),
        S1dot6S1OAvg=_Sdot_6PN_S1O_avg(m1M),
        S2dot6S1OAvg=_Sdot_6PN_S2O_avg(m2M),
        S2dot6S2OAvg=_Sdot_6PN_S1O_avg(m2M),
        S1dot6QMS1OAvg=quadparam1 * _Sdot_6PN_QMSO_avg(m1M),
        S2dot6QMS2OAvg=quadparam2 * _Sdot_6PN_QMSO_avg(m2M),
        omegashiftS1=_LDot_3PN_SO(m1M),
        omegashiftS2=_LDot_3PN_SO(m2M),
        lscorr=lscorr,
    )


# =============================================================================
# Stage 1: SpinTaylor ODE
# =============================================================================


def _spin_derivatives_avg(v, LNhat, E1, S1, S2, LNdotS1, LNdotS2, p):
    """
    Compute d/dt of {LNhat, E1, S1, S2} at PN orders up to 3PN (spinO=-1).

    Mirrors XLALSimInspiralSpinDerivativesAvg in
    LALSimInspiralSpinTaylor.c:483.

    All time derivatives are with respect to dimensionless time t_hat = t/M.
    Spin vectors S1, S2 are dimensionful: S_i = chi_i * m_i^2 (in units M=1).

    Args:
        v:       PN velocity parameter v = omega^(1/3)
        LNhat:   (3,) unit orbital angular momentum
        E1:      (3,) reference direction in orbital plane
        S1:      (3,) spin of body 1
        S2:      (3,) spin of body 2
        LNdotS1: LNhat . S1
        LNdotS2: LNhat . S2
        p:       SpinTaylorT4Params

    Returns:
        dLNhat, dE1, dS1, dS2  each (3,)
    """
    omega = v**3
    v2 = v * v
    v5 = omega * v2
    v7 = omega * omega * v
    v8 = omega * omega * v2

    LNhcS1 = np.cross(LNhat, S1)  # LNhat x S1
    LNhcS2 = np.cross(LNhat, S2)  # LNhat x S2
    S1cS2 = np.cross(S1, S2)  # S1 x S2

    # ---- 1.5PN leading spin precession (spinO >= 3) ----
    dS1 = p.S1dot3 * v5 * LNhcS1
    dS2 = p.S2dot3 * v5 * LNhcS2

    # dL/dt at leading order = -(dS1 + dS2)
    dLNhat_raw = -(dS1 + dS2)

    # ---- 2PN NLO spin precession (spinO >= 4) ----
    omega2 = omega * omega

    # S1S2 terms (eq. 4.17 of gr-qc/9506022)
    dS1_NL = omega2 * (-p.S1dot4S2Avg * S1cS2 + p.S1dot4S2OAvg * LNdotS2 * LNhcS1)
    dS2_NL = omega2 * (+p.S1dot4S2Avg * S1cS2 + p.S1dot4S2OAvg * LNdotS1 * LNhcS2)

    # QM self-spin terms
    dS1_NL += omega2 * p.S1dot4QMS1OAvg * LNdotS1 * LNhcS1
    dS2_NL += omega2 * p.S2dot4QMS2OAvg * LNdotS2 * LNhcS2

    dS1 += dS1_NL
    dS2 += dS2_NL
    dLNhat_raw += -(dS1_NL + dS2_NL)

    # ---- 2.5PN NNLO spin precession (spinO >= 5) ----
    # eq. 7.8 of gr-qc/0605140
    L1PN = _L_2PN(p.eta)
    LNmag = p.eta / v * (1.0 + v2 * L1PN)  # corrected LN magnitude

    dS1_NNL = p.S1dot5 * v7 * LNhcS1
    dS2_NNL = p.S2dot5 * v7 * LNhcS2

    dS1 += dS1_NNL
    dS2 += dS2_NNL
    dLNhat_raw -= dS1_NNL + dS2_NNL

    if p.lscorr:
        cS1 = _L_3PN_Si_avg(p.m1M)
        cS2 = _L_3PN_Si_avg(p.m2M)
        dS1_L = p.S1dot3 * v5 * LNhcS1  # leading spin deriv
        dS2_L = p.S2dot3 * v5 * LNhcS2
        dLNhat_raw -= p.eta * v2 * (cS1 * dS1_L + cS2 * dS2_L)

    # ---- 3PN N3L spin precession (spinO >= 6, phenomtp=False) ----
    # eq. A.2 of arXiv:1501.01529
    dS1_N3L = v8 * (
        -p.S1dot6S2Avg * S1cS2
        + (p.S1dot6S1OAvg * LNdotS1 + p.S1dot6S2OAvg * LNdotS2) * LNhcS1
    )
    dS2_N3L = v8 * (
        +p.S2dot6S1Avg * S1cS2
        + (p.S2dot6S1OAvg * LNdotS1 + p.S2dot6S2OAvg * LNdotS2) * LNhcS2
    )

    # QM terms
    dS1_N3L += v8 * p.S1dot6QMS1OAvg * LNdotS1 * LNhcS1
    dS2_N3L += v8 * p.S2dot6QMS2OAvg * LNdotS2 * LNhcS2

    dS1 += dS1_N3L
    dS2 += dS2_N3L
    dLNhat_raw -= dS1_N3L + dS2_N3L

    if p.lscorr:
        # spin-dependent L contributions at v8 order
        cS1 = _L_3PN_Si_avg(p.m1M)
        cS2 = _L_3PN_Si_avg(p.m2M)
        cS1L = _L_3PN_SiL_avg(p.m1M)
        cS2L = _L_3PN_SiL_avg(p.m2M)
        LN0mag = p.eta / v
        dS1_L = p.S1dot3 * v5 * LNhcS1
        dS2_L = p.S2dot3 * v5 * LNhcS2
        dL_L = -(dS1_L + dS2_L)
        dLNhat_raw -= (
            p.eta
            * v2
            * (
                cS1 * dS1_NL
                + cS2 * dS2_NL
                + (cS1L * LNdotS1 + cS2L * LNdotS2) * dL_L / LN0mag
            )
        )

    # ---- Normalize and project dLNhat ----
    # dLNhat_raw / LNmag gives the precession contribution
    dLNhat_raw /= LNmag

    # Precession vector: Om = LNhat x dLNhat
    Om = np.cross(LNhat, dLNhat_raw)

    # Project: dLNhat = Om x LNhat  (perpendicular to LNhat)
    dLNhat = np.cross(Om, LNhat)

    # E1 precesses at the same rate
    dE1 = np.cross(Om, E1)

    return dLNhat, dE1, dS1, dS2


def spinTaylorT4_derivatives(t, y, p):
    """
    14-component ODE RHS for SpinTaylorT4.

    State vector layout (all dimensionless, geometric units M=1):
      y[0]      = phi        orbital phase
      y[1]      = omega      orbital angular frequency (M * omega_SI)
      y[2:5]    = LNhat      unit orbital angular momentum (x,y,z)
      y[5:8]    = S1         spin of body 1 = chi1 * m1^2
      y[8:11]   = S2         spin of body 2 = chi2 * m2^2
      y[11:14]  = E1         reference direction in orbital plane

    All time derivatives are w.r.t. t_hat = t / M.

    Mirrors XLALSimInspiralSpinTaylorT4DerivativesAvg in
    LALSimInspiralSpinTaylor.c:2540, with spin orders up to 3PN (spinO=-1)
    and tidal terms set to zero (BBH).

    Args:
        t:  dimensionless time (unused, equations are autonomous)
        y:  (14,) state vector
        p:  SpinTaylorT4Params from build_spin_taylor_params()

    Returns:
        dy/dt as (14,) array
    """
    omega = y[1]
    LNhat = y[2:5]
    S1 = y[5:8]
    S2 = y[8:11]
    E1 = y[11:14]

    if omega <= 0.0:
        raise ValueError("omega must be positive")

    v = omega ** (1.0 / 3.0)
    v2 = v * v
    v11 = omega**3 * v2  # = omega^(11/3)

    LNdotS1 = np.dot(LNhat, S1)
    LNdotS2 = np.dot(LNhat, S2)
    S1dotS2 = np.dot(S1, S2)
    S1sq = np.dot(S1, S1)
    S2sq = np.dot(S2, S2)

    # --- spin contributions to domega ---

    # 1.5PN SO
    wspin3 = p.wdot3S1O * LNdotS1 + p.wdot3S2O * LNdotS2

    # 2PN SS (orbit-averaged)
    wspin4Avg = (
        p.wdot4S1S2Avg * S1dotS2
        + p.wdot4S1OS2OAvg * LNdotS1 * LNdotS2
        + (p.wdot4S1S1Avg + p.wdot4QMS1S1Avg) * S1sq
        + (p.wdot4S2S2Avg + p.wdot4QMS2S2Avg) * S2sq
        + (p.wdot4S1OS1OAvg + p.wdot4QMS1OS1OAvg) * LNdotS1**2
        + (p.wdot4S2OS2OAvg + p.wdot4QMS2OS2OAvg) * LNdotS2**2
    )

    # 2.5PN SO
    wspin5 = p.wdot5S1O * LNdotS1 + p.wdot5S2O * LNdotS2

    # 3PN SS (orbit-averaged)
    wspin6Avg = (
        p.wdot6S1O * LNdotS1
        + p.wdot6S2O * LNdotS2
        + p.wdot6S1OS2OAvg * LNdotS1 * LNdotS2
        + p.wdot6S1S2Avg * S1dotS2
        + (p.wdot6S1S1Avg + p.wdot6QMS1S1Avg) * S1sq
        + (p.wdot6S2S2Avg + p.wdot6QMS2S2Avg) * S2sq
        + (p.wdot6S1OS1OAvg + p.wdot6QMS1OS1OAvg) * LNdotS1**2
        + (p.wdot6S2OS2OAvg + p.wdot6QMS2OS2OAvg) * LNdotS2**2
    )

    # full domega (eq. 1-7 of gr-qc/0405090)
    # wdotcoeff[i] multiplies v^(2i), stacked as a Horner polynomial in v
    c = p.wdotcoeff
    domega = (
        p.wdotnewt
        * v11
        * (
            c[0]
            + v
            * (
                c[1]
                + v
                * (
                    c[2]
                    + v
                    * (
                        c[3]
                        + wspin3
                        + v
                        * (
                            c[4]
                            + wspin4Avg
                            + v
                            * (
                                c[5]
                                + wspin5
                                + v
                                * (
                                    c[6]
                                    + wspin6Avg
                                    + p.wdotlogcoeff * math.log(v)
                                    + v
                                    * (
                                        c[7]
                                        + omega * (p.wdottidal10 + v2 * p.wdottidal12)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    # --- spin and LN derivatives ---
    dLNhat, dE1, dS1, dS2 = _spin_derivatives_avg(
        v, LNhat, E1, S1, S2, LNdotS1, LNdotS2, p
    )

    # --- orbital phase derivative ---
    # d phi / d t_hat = omega * (1 + omega^3 * omegashift)
    # omegashift accounts for the shift in orbital frequency due to spin (1.5PN)
    # eq. (2.5) of gr-qc/0405090
    omegashift = -0.25 * (
        p.omegashiftS1**2 * (S1sq - LNdotS1**2)
        + p.omegashiftS2**2 * (S2sq - LNdotS2**2)
        + 2.0 * p.omegashiftS1 * p.omegashiftS2 * (S1dotS2 - LNdotS1 * LNdotS2)
    )
    dphi = omega * (1.0 + omega**3 * omegashift)

    dy = np.empty(14)
    dy[0] = dphi
    dy[1] = domega
    dy[2:5] = dLNhat
    dy[5:8] = dS1
    dy[8:11] = dS2
    dy[11:14] = dE1

    return dy


# =============================================================================
# Stage 2: Euler angle extraction and spline construction
# =============================================================================


def extract_alpha_cosbeta(V_PN, LNhat_PN, phiJ_Sf, thetaJN, kappa, piGM):
    """
    Rotate LNhat into the J-frame and extract Euler angles alpha and cosbeta.

    Rotation sequence: R_z(-phiJ_Sf) -> R_y(-thetaJN) -> R_z(-kappa)

    alpha_raw[i] = atan2(LNy_rot[i], LNx_rot[i])   — phase-unwrapped
    cosbeta[i]   = LNz_rot[i]
    Mf[i]        = V_PN[i]^3 / pi                   — geometric GW frequency

    Mirrors IMRPhenomX_InterpolateAlphaBeta_SpinTaylor in
    LALSimIMRPhenomX_precession.c:4291.

    Args:
        V_PN:     (N,) array of v values from ODE
        LNhat_PN: (N, 3) array of LNhat in source frame
        phiJ_Sf:  azimuthal angle of J in source frame
        thetaJN:  polar angle of J w.r.t. line of sight
        kappa:    azimuthal angle of LN in J-frame at reference point
        piGM:     pi * G * M_total (SI units)

    Returns:
        Mf:      (N,) geometric GW frequency array
        alpha:   (N,) unwrapped alpha angle
        cosbeta: (N,) cosine of beta angle
    """
    raise NotImplementedError


# =============================================================================
# Stage 3: MRD analytical continuation
# =============================================================================


def alphaMRD_coeff(Mf_nodes, alpha_nodes, fmax_inspiral):
    """
    Fit the 3-parameter rational function alpha_MRD(Mf) = -(a + b/Mf^4 + c/Mf^2)
    by matching the inspiral spline at two nodes near fmax_inspiral.

    Matching conditions (from LALSimIMRPhenomX_precession.c ~line 3919):
      - alpha and dalpha/dMf at f1 = 0.97 * fmax_inspiral
      - alpha               at f2 = 0.99 * fmax_inspiral

    Args:
        Mf_nodes:       (N,) Mf grid from stage 2
        alpha_nodes:    (N,) unwrapped alpha on that grid
        fmax_inspiral:  upper edge of the inspiral spline range

    Returns:
        aRD, bRD, cRD: float coefficients
    """
    raise NotImplementedError


def alphaMRD(Mf, aRD, bRD, cRD):
    """
    Evaluate alpha in the MRD regime.

    alpha_MRD(Mf) = -(aRD + bRD / Mf^4 + cRD / Mf^2)

    Args:
        Mf:         geometric GW frequency (scalar or array)
        aRD, bRD, cRD: coefficients from alphaMRD_coeff

    Returns:
        alpha (same shape as Mf)
    """
    raise NotImplementedError


def dalphaMRD(Mf, aRD, bRD, cRD):
    """
    Derivative of alphaMRD with respect to Mf.

    d(alpha_MRD)/dMf = 4*bRD / Mf^5 + 2*cRD / Mf^3

    Args:
        Mf:         geometric GW frequency (scalar or array)
        aRD, bRD, cRD: coefficients from alphaMRD_coeff

    Returns:
        d(alpha)/dMf (same shape as Mf)
    """
    raise NotImplementedError


def betaMRD_coeff(Mf_nodes, cosbeta_nodes, fmax_inspiral, fDAMP, dfdamp):
    """
    Fit the damped-exponential ansatz for beta in the MRD regime.

    Ansatz (from LALSimIMRPhenomX_precession.c ~line 4016):
      beta_MRD(Mf) = exp(-kappa*Mf)/Mf * (aRD/Mf + bRD/Mf^2 + cRD/Mf^3) + dRD
      where kappa = 2*pi*dfdamp

    Matching conditions:
      - cosbeta and d(cosbeta)/dMf at f1 = 0.97 * fmax_inspiral
      - cosbeta and d(cosbeta)/dMf at f2 = 0.98 * fmax_inspiral

    Sets flat_RD=True and returns cosbeta_sign if cosbeta values are unphysical.

    Args:
        Mf_nodes:       (N,) Mf grid from stage 2
        cosbeta_nodes:  (N,) cosbeta array from stage 2
        fmax_inspiral:  upper edge of the inspiral spline range
        fDAMP:          QNM damping frequency (geometric units)
        dfdamp:         difference between (2,1) and (2,2) QNM damping freqs

    Returns:
        aRD, bRD, cRD, dRD: float coefficients
        flat_RD:            bool — use flat fallback if True
        cosbeta_sign:       sign of cosbeta at fmax (used in flat fallback)
    """
    raise NotImplementedError


def betaMRD(Mf, aRD, bRD, cRD, dRD, dfdamp, flat_RD, cosbeta_sign):
    """
    Evaluate beta (as an angle) in the MRD regime.

    Uses the damped-exponential fit, or falls back to acos(cosbeta_sign)
    if flat_RD is True.

    Args:
        Mf:           geometric GW frequency (scalar or array)
        aRD..dRD:     coefficients from betaMRD_coeff
        dfdamp:       QNM damping freq difference (sets kappa = 2*pi*dfdamp)
        flat_RD:      bool flag
        cosbeta_sign: +1 or -1 for flat fallback

    Returns:
        beta angle (same shape as Mf)
    """
    raise NotImplementedError


# =============================================================================
# Stage 4: Gamma integration via Boole's rule
# =============================================================================


def gamma_integrand(
    Mf, alpha_spline, cosbeta_spline, alpha_params, beta_params, ftrans_MRD
):
    """
    Evaluate the integrand g(Mf) = -cosbeta(Mf) * d(alpha)/dMf(Mf).

    Dispatches to spline evaluators for Mf < ftrans_MRD,
    and to alphaMRD/betaMRD for Mf >= ftrans_MRD (version 320 behaviour).

    Args:
        Mf:             scalar frequency
        alpha_spline:   cubic spline object for alpha (inspiral)
        cosbeta_spline: cubic spline object for cosbeta (inspiral)
        alpha_params:   (aRD, bRD, cRD) from alphaMRD_coeff
        beta_params:    output of betaMRD_coeff
        ftrans_MRD:     transition frequency

    Returns:
        g(Mf): scalar
    """
    raise NotImplementedError


def integrate_gamma_boole(
    Mf_grid, alpha_spline, cosbeta_spline, alpha_params, beta_params, ftrans_MRD
):
    """
    Integrate d(gamma)/dMf = -cosbeta * d(alpha)/dMf over Mf_grid
    using 5-point Boole's rule at each step.

    Unlike version 310 (which freezes gamma at ftrans_MRD), version 320
    continues the integration through the MRD using the analytical forms.

    Boole's rule (h = deltaMF/4):
      delta_gamma = (2h/45) * (7*g0 + 32*g1 + 12*g2 + 32*g3 + 7*g4)

    Mirrors IMRPhenomX_InterpolateGamma_SpinTaylor in
    LALSimIMRPhenomX_precession.c:4450.

    Args:
        Mf_grid:        uniform (N,) frequency array
        alpha_spline:   cubic spline for alpha (inspiral)
        cosbeta_spline: cubic spline for cosbeta (inspiral)
        alpha_params:   MRD alpha coefficients
        beta_params:    MRD beta coefficients
        ftrans_MRD:     inspiral/MRD transition frequency

    Returns:
        gamma: (N,) cumulative gamma array (not yet offset-corrected)
    """
    raise NotImplementedError


# =============================================================================
# Stage 5: Reference-point offsets and top-level entry point
# =============================================================================


def apply_angle_offsets(
    Mf_array,
    alpha_raw,
    cosbeta,
    gamma,
    MfRef,
    alpha_params,
    beta_params,
    gamma_spline,
    ftrans_MRD,
    alpha0,
    epsilon0,
):
    """
    Subtract reference-point values and add frame-convention offsets.

    For version 320, alpha_ref is evaluated via alphaMRD if MfRef > ftrans_MRD,
    otherwise from the inspiral spline.

    Offsets applied:
      alpha   += alpha0 - alpha_ref       where alpha0 = pi - kappa
      gamma   -= gamma_ref + epsilon0

    Args:
        Mf_array:    output frequency grid
        alpha_raw:   un-offset alpha (N,)
        cosbeta:     (N,)
        gamma:       un-offset gamma (N,)
        MfRef:       reference geometric frequency
        alpha_params: MRD alpha coefficients
        beta_params:  MRD beta coefficients
        gamma_spline: spline of gamma vs Mf (for gamma_ref)
        ftrans_MRD:  transition frequency
        alpha0:      = pi - kappa
        epsilon0:    reference epsilon offset

    Returns:
        alpha:   (N,) offset-corrected
        cosbeta: (N,) unchanged
        gamma:   (N,) offset-corrected
    """
    raise NotImplementedError


def integrate_spin_taylor(
    chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, fmin, fRef, m1_SI, m2_SI, params, fCut_Hz
):
    """
    Integrate the SpinTaylorT4 equations from fMin to fCut = fRING + 8*fDAMP.

    Mirrors IMRPhenomX_InspiralAngles_SpinTaylor in
    LALSimIMRPhenomX_precession.c:4583, using scipy.integrate.solve_ivp
    in place of XLALSimInspiralSpinTaylorPNEvolveOrbit.

    If fRef > fmin, performs two passes and concatenates:
      - backward  : integrate from fRef down to fMin (reversed at output)
      - forward   : integrate from fRef up to fCut
    If fRef == fmin, a single forward pass is run from fMin to fCut.

    Initial conditions (LALSimInspiralSpinTaylor.c:3692):
      phi   = 0
      omega = pi * (G*M/c^3) * fStart        [dimensionless]
      LNhat = (0, 0, 1)                       [z-axis]
      S_i   = (mi/M)^2 * chi_i               [geometric, M_total = 1]
      E1    = (1, 0, 0)                       [x-axis]

    Args:
        chi1x/y/z:  dimensionless spin components of body 1 at fRef
                    expressed in the frame where LNhat = (0,0,1)
        chi2x/y/z:  same for body 2
        fmin:       starting GW frequency [Hz]
        fRef:       reference GW frequency [Hz];  must be >= fmin
        m1_SI:      mass of body 1 [kg]
        m2_SI:      mass of body 2 [kg]
        params:     SpinTaylorT4Params from build_spin_taylor_params()
        fCut_Hz:    upper integration limit [Hz]; typically fRING + 8*fDAMP
                    converted from geometric units via XLALSimIMRPhenomXUtilsMftoHz

    Returns:
        V_PN:     (N,) array of v = omega^(1/3)
        LNhat_PN: (N, 3) unit orbital angular momentum direction
        S1_PN:    (N, 3) spin of body 1  (= (m1/M)^2 * chi1_evolved)
        S2_PN:    (N, 3) spin of body 2
    """
    M_SI = m1_SI + m2_SI
    m1M = m1_SI / M_SI
    m2M = m2_SI / M_SI
    # G*M/c^3 in seconds — same as LAL_MTSUN_SI * (M / M_sun)
    M_sec = _MTSUN_SI * (M_SI / 1.988409870698050731911960804878414216e30)

    # Dimensionless orbital frequencies  omega_hat = pi * M_sec * f_Hz
    omega_ref = PI * M_sec * fRef
    omega_min = PI * M_sec * fmin
    omega_cut = PI * M_sec * fCut_Hz

    # ── initial state at fRef ─────────────────────────────────────────────
    # y = [phi, omega, LNhat(3), S1(3), S2(3), E1(3)]
    y0 = np.array(
        [
            0.0,  # phi
            omega_ref,  # omega
            0.0,
            0.0,
            1.0,  # LNhat = z-axis
            m1M**2 * chi1x,
            m1M**2 * chi1y,
            m1M**2 * chi1z,  # S1
            m2M**2 * chi2x,
            m2M**2 * chi2y,
            m2M**2 * chi2z,  # S2
            1.0,
            0.0,
            0.0,  # E1 = x-axis
        ]
    )

    # ODE tolerance (matches LAL_ST4_ABSOLUTE/RELATIVE_TOLERANCE = 1e-12 / 1e-10)
    rtol = 1e-10
    atol = 1e-12

    def _run(t_end, y_start, omega_stop, stop_direction):
        """
        Single ODE segment.
        t_end > 0 → forward  (omega increases → stop when omega >= omega_stop)
        t_end < 0 → backward (omega decreases → stop when omega <= omega_stop)
        """

        def _event(t, y, p):
            return y[1] - omega_stop

        _event.terminal = True  # type: ignore[reportFunctionMemberAccess]
        _event.direction = stop_direction  # +1 for forward, -1 for backward # type: ignore[reportFunctionMemberAccess]

        sol = solve_ivp(
            spinTaylorT4_derivatives,
            t_span=(0.0, t_end),
            y0=y_start,
            args=(params,),
            events=_event,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            max_step=np.inf,
        )
        # include terminal event point if captured
        _t_arr = sol.t
        y_arr = sol.y.T  # (N, 14)
        if sol.t_events[0].size > 0 and sol.status == 1:
            y_evt = sol.y_events[0][0]  # (14,)
            y_arr = np.vstack([y_arr, y_evt])
        return y_arr

    # large time budget: Newtonian chirp time from fmin to infinity
    # t_Newt ~ (5/256) * M_sec * (pi*M_sec*fmin)^{-8/3}  in units of M_sec
    # Use 1e7 M_sec as a safe upper bound
    T_MAX = 1e8

    # ── single-pass (fRef == fmin) ────────────────────────────────────────
    if abs(fRef - fmin) < 1e-4:  # ~0.1 mHz tolerance, mirrors LAL_REAL4_EPS check
        Y = _run(+T_MAX, y0, omega_cut, stop_direction=+1)

    # ── two-pass (fRef > fmin) ────────────────────────────────────────────
    else:
        # backward: start at fRef, stop when omega reaches omega_min
        Y_bwd = _run(-T_MAX, y0, omega_min, stop_direction=-1)

        if len(Y_bwd) <= 1:
            raise RuntimeError(
                "SpinTaylor backward integration produced ≤ 1 point; "
                "try reducing fRef or increasing fmin."
            )

        # forward: start at fRef, stop when omega reaches omega_cut
        Y_fwd = _run(+T_MAX, y0, omega_cut, stop_direction=+1)

        # reversed backward (oldest first) + forward (skip duplicate fRef point)
        # Y_bwd[0] is at fRef, Y_bwd[-1] is near fmin  →  reverse to get fmin first
        Y = np.vstack([Y_bwd[::-1], Y_fwd[1:]])

    # ── extract outputs ───────────────────────────────────────────────────
    V_PN = Y[:, 1] ** (1.0 / 3.0)  # v = omega^{1/3}
    LNhat_PN = Y[:, 2:5].copy()
    # ODE stores S_i = (mi/M)^2 * chi_i; divide back to match LALSim convention
    # (XLALSimInspiralSpinTaylorPNEvolveOrbit divides by norm_i at output, line 4575)
    S1_PN = Y[:, 5:8] / m1M**2
    S2_PN = Y[:, 8:11] / m2M**2

    return V_PN, LNhat_PN, S1_PN, S2_PN


def compute_angles_320(
    Mf_array,
    m1_SI,
    m2_SI,
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    fRef,
    fRING,
    fDAMP,
    fMECO,
    phiJ_Sf,
    thetaJN,
    kappa,
    alpha0,
    epsilon0,
    params,
):
    """
    Top-level function: compute Euler angles (alpha, cosbeta, gamma)
    for IMRPhenomXPrecVersion=320 on the given frequency array.

    Orchestrates stages 1-5:
      1. Integrate SpinTaylorT4 ODE
      2. Extract alpha, cosbeta, build cubic splines
      3. Fit MRD analytical continuation coefficients
      4. Integrate gamma via Boole's rule (continuous through MRD)
      5. Apply reference-point offsets

    Args:
        Mf_array:        (N,) geometric GW frequency output grid
        m1_SI, m2_SI:    component masses (kg)
        chi1x/y/z:       spin components of body 1 at fRef
        chi2x/y/z:       spin components of body 2 at fRef
        fRef:            reference GW frequency (Hz)
        fRING:           (2,2) ringdown frequency (geometric units)
        fDAMP:           (2,2) damping frequency (geometric units)
        fMECO:           MECO frequency (geometric units)
        phiJ_Sf:         azimuthal angle of J in source frame
        thetaJN:         inclination angle
        kappa:           azimuthal angle of LN in J-frame at reference point
        alpha0:          = pi - kappa
        epsilon0:        reference epsilon offset
        params:          SpinTaylorT4Params

    Returns:
        alpha:   (N,) Euler angle alpha
        cosbeta: (N,) cos(beta)
        gamma:   (N,) Euler angle gamma
    """
    raise NotImplementedError
