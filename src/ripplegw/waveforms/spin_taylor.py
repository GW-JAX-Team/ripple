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
import equinox as eqx
import diffrax
import jax.numpy as jnp
from ..constants import EULERGAMMA, G, C, PI

# G * M_sun / c^3  in seconds  (= LAL_MTSUN_SI)

_MTSUN_SI = G * 1.988409870698050731911960804878414216e30 / C**3


# =============================================================================
# PN coefficient helpers  (evaluated once at build-time with Python scalars)
# =============================================================================


def _wdot_0PN(eta):
    return 96.0 / 5.0 * eta


def _wdot_2PN(eta):
    return -(743.0 + 924.0 * eta) / 336.0


def _wdot_3PN(_eta):
    return 4.0 * math.pi


def _wdot_3PN_SO(mByM):
    return -19.0 / 6.0 - 25.0 / (4.0 * mByM)


def _wdot_4PN(eta):
    return (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0


def _wdot_4PN_S1S2_avg(eta):
    return -247.0 / (48.0 * eta)


def _wdot_4PN_S1OS2O_avg(eta):
    return 721.0 / (48.0 * eta)


def _wdot_4PN_S1S1_avg(mByM):
    return 7.0 / (96.0 * mByM**2)


def _wdot_4PN_S1OS1O_avg(mByM):
    return -1.0 / (96.0 * mByM**2)


def _wdot_4PN_QMS1S1_avg(mByM):
    return -2.5 / mByM**2


def _wdot_4PN_QMS1OS1O_avg(mByM):
    return 7.5 / mByM**2


def _wdot_5PN(eta):
    return -(4159.0 + 15876.0 * eta) * math.pi / 672.0


def _wdot_5PN_SO(mByM):
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
        - 856.0 / 105.0 * jnp.log(16.0)
    )


def _wdot_6PN_log(_eta):
    return -1712.0 / 105.0


def _wdot_6PN_SO(mByM):
    return math.pi * (-37.0 / 3.0 - 151.0 / (6.0 * mByM))


def _wdot_6PN_S1S2_avg(eta):
    return 108.79 / (6.72 * eta) + 75.25 / 2.88


def _wdot_6PN_S1OS2O_avg(eta):
    return 162.25 / (2.24 * eta) - 129.31 / 2.88


def _wdot_6PN_S1S1_avg(mByM):
    return 101.9 / (6.4 * mByM**2) + 2.51 / (5.76 * mByM) + 13.33 / 5.76


def _wdot_6PN_S1OS1O_avg(mByM):
    return -49.3 / (6.4 * mByM**2) + 197.47 / (5.76 * mByM) + 56.45 / 5.76


def _wdot_6PN_QMS1S1_avg(mByM):
    return -6.59 / (2.24 * mByM**2) + 7.3 / (4.8 * mByM) - 43.0 / 4.0


def _wdot_6PN_QMS1OS1O_avg(mByM):
    return 19.77 / (2.24 * mByM**2) - 7.3 / (1.6 * mByM) + 129.0 / 4.0


def _wdot_7PN(eta):
    return math.pi / 12096.0 * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2)


def _Sdot_3PN(mByM):
    return 1.5 - mByM - mByM**2 / 2.0


_SDOT_4PN_S2_AVG = 0.5
_SDOT_4PN_S2O_AVG = -1.5


def _Sdot_4PN_QMS1O_avg(mByM):
    return 1.5 * (1.0 - 1.0 / mByM)


def _Sdot_5PN(mByM):
    return (
        9.0 / 8.0
        - mByM / 2.0
        + 7.0 * mByM**2 / 12.0
        - 7.0 * mByM**3 / 6.0
        - mByM**4 / 24.0
    )


def _Sdot_6PN_S2_avg(mByM):
    return 0.5 * mByM + 0.5 * mByM**2


def _Sdot_6PN_S1O_avg(mByM):
    return -3.25 + 2.25 / mByM + 0.75 * mByM + 0.25 * mByM**2


def _Sdot_6PN_S2O_avg(mByM):
    return -1.5 - 1.5 * mByM - 0.5 * mByM**2


def _Sdot_6PN_QMSO_avg(mByM):
    return -2.25 / mByM + 1.5 * mByM + 0.75 * mByM**2


def _LDot_3PN_SO(mByM):
    return 0.5 + 1.5 / mByM


def _L_2PN(eta):
    return 1.5 + eta / 6.0


def _L_3PN_Si_avg(mByM):
    return -0.75 - 0.25 / mByM


def _L_3PN_SiL_avg(mByM):
    return -(1.0 / 3.0 + 9.0 / mByM) / 4.0


# =============================================================================
# Parameter struct
# =============================================================================


class SpinTaylorT4Params(eqx.Module):
    """
    Pre-computed PN coefficients for SpinTaylorT4 (JAX version).

    Uses equinox.Module so that diffrax/equinox JIT can hash it correctly.
    All float/array fields are traced JAX leaves.
    lscorr is marked static: it controls Python-level if-branches at trace
    time and causes retracing only when its value changes.
    """

    eta: float
    m1M: float
    m2M: float
    wdotnewt: float
    wdotcoeff: jnp.ndarray  # shape (8,)
    wdotlogcoeff: float
    wdottidal10: float
    wdottidal12: float
    wdot3S1O: float
    wdot3S2O: float
    wdot5S1O: float
    wdot5S2O: float
    wdot6S1O: float
    wdot6S2O: float
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
    S1dot3: float
    S2dot3: float
    S1dot4S2Avg: float
    S1dot4S2OAvg: float
    S1dot4QMS1OAvg: float
    S2dot4QMS2OAvg: float
    S1dot5: float
    S2dot5: float
    S1dot6S2Avg: float
    S2dot6S1Avg: float
    S1dot6S2OAvg: float
    S1dot6S1OAvg: float
    S2dot6S1OAvg: float
    S2dot6S2OAvg: float
    S1dot6QMS1OAvg: float
    S2dot6QMS2OAvg: float
    omegashiftS1: float
    omegashiftS2: float
    lscorr: int = eqx.field(static=True)


def build_spin_taylor_params(m1M, eta, quadparam1=1.0, quadparam2=1.0, lscorr=0):
    """
    Pre-compute all PN coefficients for SpinTaylorT4 (JAX version).
    Same interface as spin_taylor.build_spin_taylor_params.
    wdotcoeff is stored as a jnp.array.
    """
    m2M = 1.0 - m1M
    wdotcoeff = jnp.array(
        [
            1.0,
            0.0,
            _wdot_2PN(eta),
            _wdot_3PN(eta),
            _wdot_4PN(eta),
            _wdot_5PN(eta),
            _wdot_6PN(eta),
            _wdot_7PN(eta),
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
        wdot3S1O=_wdot_3PN_SO(m1M),
        wdot3S2O=_wdot_3PN_SO(m2M),
        wdot5S1O=_wdot_5PN_SO(m1M),
        wdot5S2O=_wdot_5PN_SO(m2M),
        wdot6S1O=_wdot_6PN_SO(m1M),
        wdot6S2O=_wdot_6PN_SO(m2M),
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
    JAX version of _spin_derivatives_avg.

    All numpy ops replaced with jnp; in-place mutations replaced with
    functional accumulation. The lscorr branch is a Python-level condition
    evaluated at JAX trace time (static).
    """
    omega = v**3
    v2 = v * v
    v5 = omega * v2
    v7 = omega * omega * v
    v8 = omega * omega * v2

    LNhcS1 = jnp.cross(LNhat, S1)
    LNhcS2 = jnp.cross(LNhat, S2)
    S1cS2 = jnp.cross(S1, S2)

    # ---- 1.5PN leading spin precession ----
    dS1 = p.S1dot3 * v5 * LNhcS1
    dS2 = p.S2dot3 * v5 * LNhcS2
    dLNhat_raw = -(dS1 + dS2)

    # ---- 2PN NLO spin precession ----
    omega2 = omega * omega
    dS1_NL = omega2 * (-p.S1dot4S2Avg * S1cS2 + p.S1dot4S2OAvg * LNdotS2 * LNhcS1)
    dS2_NL = omega2 * (+p.S1dot4S2Avg * S1cS2 + p.S1dot4S2OAvg * LNdotS1 * LNhcS2)
    dS1_NL = dS1_NL + omega2 * p.S1dot4QMS1OAvg * LNdotS1 * LNhcS1
    dS2_NL = dS2_NL + omega2 * p.S2dot4QMS2OAvg * LNdotS2 * LNhcS2
    dS1 = dS1 + dS1_NL
    dS2 = dS2 + dS2_NL
    dLNhat_raw = dLNhat_raw - (dS1_NL + dS2_NL)

    # ---- 2.5PN NNLO spin precession ----
    L1PN = _L_2PN(p.eta)
    LNmag = p.eta / v * (1.0 + v2 * L1PN)
    dS1_NNL = p.S1dot5 * v7 * LNhcS1
    dS2_NNL = p.S2dot5 * v7 * LNhcS2
    dS1 = dS1 + dS1_NNL
    dS2 = dS2 + dS2_NNL
    dLNhat_raw = dLNhat_raw - (dS1_NNL + dS2_NNL)

    # Python-level branch: lscorr is static (aux) in the pytree
    if p.lscorr:
        cS1 = _L_3PN_Si_avg(p.m1M)
        cS2 = _L_3PN_Si_avg(p.m2M)
        dS1_L = p.S1dot3 * v5 * LNhcS1
        dS2_L = p.S2dot3 * v5 * LNhcS2
        dLNhat_raw = dLNhat_raw - p.eta * v2 * (cS1 * dS1_L + cS2 * dS2_L)

    # ---- 3PN N3L spin precession ----
    dS1_N3L = v8 * (
        -p.S1dot6S2Avg * S1cS2
        + (p.S1dot6S1OAvg * LNdotS1 + p.S1dot6S2OAvg * LNdotS2) * LNhcS1
    )
    dS2_N3L = v8 * (
        +p.S2dot6S1Avg * S1cS2
        + (p.S2dot6S1OAvg * LNdotS1 + p.S2dot6S2OAvg * LNdotS2) * LNhcS2
    )
    dS1_N3L = dS1_N3L + v8 * p.S1dot6QMS1OAvg * LNdotS1 * LNhcS1
    dS2_N3L = dS2_N3L + v8 * p.S2dot6QMS2OAvg * LNdotS2 * LNhcS2
    dS1 = dS1 + dS1_N3L
    dS2 = dS2 + dS2_N3L
    dLNhat_raw = dLNhat_raw - (dS1_N3L + dS2_N3L)

    if p.lscorr:
        cS1 = _L_3PN_Si_avg(p.m1M)
        cS2 = _L_3PN_Si_avg(p.m2M)
        cS1L = _L_3PN_SiL_avg(p.m1M)
        cS2L = _L_3PN_SiL_avg(p.m2M)
        LN0mag = p.eta / v
        dS1_L = p.S1dot3 * v5 * LNhcS1
        dS2_L = p.S2dot3 * v5 * LNhcS2
        dL_L = -(dS1_L + dS2_L)
        dLNhat_raw = dLNhat_raw - (
            p.eta
            * v2
            * (
                cS1 * dS1_NL
                + cS2 * dS2_NL
                + (cS1L * LNdotS1 + cS2L * LNdotS2) * dL_L / LN0mag
            )
        )

    # ---- Normalize and project dLNhat ----
    dLNhat_raw = dLNhat_raw / LNmag
    Om = jnp.cross(LNhat, dLNhat_raw)
    dLNhat = jnp.cross(Om, LNhat)
    dE1 = jnp.cross(Om, E1)

    return dLNhat, dE1, dS1, dS2


def spinTaylorT4_derivatives(t, y, p):
    """
    JAX version of the 14-component ODE RHS for SpinTaylorT4.

    State vector layout: [phi, omega, LNhat(3), S1(3), S2(3), E1(3)]

    Differences from spin_taylor.py:
      - Uses jnp instead of np
      - No ValueError guard on omega (incompatible with JIT)
      - Returns jnp.concatenate([...]) instead of mutating np.empty(14)
    """
    omega = y[1]
    LNhat = y[2:5]
    S1 = y[5:8]
    S2 = y[8:11]
    E1 = y[11:14]

    v = omega ** (1.0 / 3.0)
    v2 = v * v
    v11 = omega**3 * v2  # = omega^(11/3)

    LNdotS1 = jnp.dot(LNhat, S1)
    LNdotS2 = jnp.dot(LNhat, S2)
    S1dotS2 = jnp.dot(S1, S2)
    S1sq = jnp.dot(S1, S1)
    S2sq = jnp.dot(S2, S2)

    # --- spin contributions to domega ---
    wspin3 = p.wdot3S1O * LNdotS1 + p.wdot3S2O * LNdotS2

    wspin4Avg = (
        p.wdot4S1S2Avg * S1dotS2
        + p.wdot4S1OS2OAvg * LNdotS1 * LNdotS2
        + (p.wdot4S1S1Avg + p.wdot4QMS1S1Avg) * S1sq
        + (p.wdot4S2S2Avg + p.wdot4QMS2S2Avg) * S2sq
        + (p.wdot4S1OS1OAvg + p.wdot4QMS1OS1OAvg) * LNdotS1**2
        + (p.wdot4S2OS2OAvg + p.wdot4QMS2OS2OAvg) * LNdotS2**2
    )

    wspin5 = p.wdot5S1O * LNdotS1 + p.wdot5S2O * LNdotS2

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

    # Horner-form PN series for domega
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
                                    + p.wdotlogcoeff * jnp.log(v)
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

    dLNhat, dE1, dS1, dS2 = _spin_derivatives_avg(
        v, LNhat, E1, S1, S2, LNdotS1, LNdotS2, p
    )

    # orbital phase derivative with 1.5PN omegashift
    omegashift = -0.25 * (
        p.omegashiftS1**2 * (S1sq - LNdotS1**2)
        + p.omegashiftS2**2 * (S2sq - LNdotS2**2)
        + 2.0 * p.omegashiftS1 * p.omegashiftS2 * (S1dotS2 - LNdotS1 * LNdotS2)
    )
    dphi = omega * (1.0 + omega**3 * omegashift)

    return jnp.concatenate(
        [
            jnp.array([dphi, domega]),
            dLNhat,
            dS1,
            dS2,
            dE1,
        ]
    )


# =============================================================================
# Stage 1: ODE integration via diffrax
# =============================================================================


class _VectorField(eqx.Module):
    """
    Wraps spinTaylorT4_derivatives as an eqx.Module so that diffrax can
    split it into static structure (direction) and dynamic leaves (params
    arrays) without needing to hash the JAX arrays.

    Passing params here — rather than via diffeqsolve's `args` — avoids the
    error where diffrax treats `args` as a static (hashable) argument.
    """

    params: SpinTaylorT4Params
    direction: int = eqx.field(static=True)
    omega_stop: float

    def __call__(self, t, y, args):
        return self.direction * spinTaylorT4_derivatives(t, y, self.params)

    def event_cond(self, t, y, args, **kwargs):
        # Forward: stop when omega >= omega_stop (y[1] - omega_stop crosses 0+)
        # Backward: stop when omega <= omega_stop (omega_stop - y[1] crosses 0+)
        return jnp.where(
            self.direction > 0, y[1] - self.omega_stop, self.omega_stop - y[1]
        )


def _run_segment(
    y_start: jnp.ndarray,
    omega_stop: float,
    direction: int,
    params: SpinTaylorT4Params,
    T_MAX: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> jnp.ndarray:
    """
    Integrate one ODE segment (forward or backward in physical time).

    Args:
        y_start:    (14,) initial state
        omega_stop: terminal value of omega
        direction:  +1 = forward (omega increasing), -1 = backward (omega decreasing)
        params:     SpinTaylorT4Params
        T_MAX:      maximum integration time (geometric units)
        rtol/atol:  tolerances
        max_steps:  maximum allowed steps (output is padded to this length)

    Returns:
        ys: (max_steps, 14) array — valid rows have finite values; the rest are NaN.
    """
    vf = _VectorField(params=params, direction=direction, omega_stop=omega_stop)
    term = diffrax.ODETerm(vf)
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    event = diffrax.Event(vf.event_cond)

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=T_MAX,
        dt0=None,
        y0=y_start,
        args=None,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(steps=True),
        max_steps=max_steps,
        event=event,
    )
    return sol.ys  # (max_steps, 14), NaN-padded beyond actual steps


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

    Note:
        Assumes V_PN contains no NaN values. Strip NaN-padded rows from the
        diffrax output before calling (e.g. mask = jnp.isfinite(V_PN)).
        piGM is accepted for API consistency but is not used internally;
        Mf = V^3 / pi is exact.
    """
    x, y, z = LNhat_PN[:, 0], LNhat_PN[:, 1], LNhat_PN[:, 2]

    # Step 1: R_z(-phiJ_Sf)
    cp, sp = jnp.cos(phiJ_Sf), jnp.sin(phiJ_Sf)
    x1 = x * cp + y * sp
    y1 = -x * sp + y * cp
    z1 = z

    # Step 2: R_y(-thetaJN)
    ct, st = jnp.cos(thetaJN), jnp.sin(thetaJN)
    x2 = x1 * ct - z1 * st
    y2 = y1
    z2 = x1 * st + z1 * ct

    # Step 3: R_z(-kappa)
    ck, sk = jnp.cos(kappa), jnp.sin(kappa)
    x3 = x2 * ck + y2 * sk
    y3 = -x2 * sk + y2 * ck
    z3 = z2

    alpha_raw = jnp.arctan2(y3, x3)
    cosbeta = z3
    alpha = jnp.unwrap(alpha_raw)
    Mf = V_PN**3 / jnp.pi

    return Mf, alpha, cosbeta


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
    f1 = 0.97 * fmax_inspiral
    f2 = 0.99 * fmax_inspiral

    # Evaluate inspiral alpha (and its derivative) at f1, f2.
    # The C code negates the spline values: alpha1 = -spline(f1), etc.
    alpha1 = -jnp.interp(f1, Mf_nodes, alpha_nodes)
    alpha2 = -jnp.interp(f2, Mf_nodes, alpha_nodes)
    dalpha_grid = jnp.gradient(alpha_nodes, Mf_nodes)
    dalpha1 = -jnp.interp(f1, Mf_nodes, dalpha_grid)

    f1sq = f1 * f1
    f2sq = f2 * f2
    f1cu = f1sq * f1
    f1q = f1sq * f1sq
    f2q = f2sq * f2sq
    denom = 2.0 * (f1sq - f2sq) ** 2

    aRD = (
        f1cu * (f1 - f2) * (f1 + f2) * dalpha1
        + 2.0 * (f1q - 2.0 * f1sq * f2sq) * alpha1
        + 2.0 * f2q * alpha2
    ) / denom

    bRD = (
        f1q
        * f2sq
        * (f1 * (f1 - f2) * (f1 + f2) * dalpha1 + 2.0 * f2sq * (-alpha1 + alpha2))
    ) / denom

    cRD = (f1sq * (f1 * (-f1q + f2q) * dalpha1 + 4.0 * f2q * (alpha1 - alpha2))) / denom

    return aRD, bRD, cRD


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
    Mf2 = Mf * Mf
    Mf4 = Mf2 * Mf2
    return -(aRD + bRD / Mf4 + cRD / Mf2)


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
    Mf2 = Mf * Mf
    Mf3 = Mf2 * Mf
    Mf5 = Mf2 * Mf3
    return 4.0 * bRD / Mf5 + 2.0 * cRD / Mf3


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
    kappa = 2.0 * jnp.pi * dfdamp
    f1 = 0.97 * fmax_inspiral
    f2 = 0.98 * fmax_inspiral
    f1sq = f1 * f1
    f2sq = f2 * f2

    dcosbeta_grid = jnp.gradient(cosbeta_nodes, Mf_nodes)
    cosbeta1 = jnp.interp(f1, Mf_nodes, cosbeta_nodes)
    cosbeta2 = jnp.interp(f2, Mf_nodes, cosbeta_nodes)
    dcosbeta2 = jnp.interp(f2, Mf_nodes, dcosbeta_grid)
    cosbetamax = jnp.interp(fmax_inspiral, Mf_nodes, cosbeta_nodes)

    flat_RD = (
        (jnp.abs(cosbeta1) > 1.0)
        | (jnp.abs(cosbeta2) > 1.0)
        | (jnp.abs(cosbetamax) > 1.0)
    )

    # Convert cosbeta → beta and d(cosbeta)/dMf → d(beta)/dMf at f2.
    # Clip to [-1,1] so arccos/sqrt are safe even in the flat_RD branch
    # (those values are discarded by jnp.where at the end).
    beta1 = jnp.arccos(jnp.clip(cosbeta1, -1.0, 1.0))
    beta2 = jnp.arccos(jnp.clip(cosbeta2, -1.0, 1.0))
    sqrtarg = 1.0 - cosbeta2 * cosbeta2
    dbeta2 = -dcosbeta2 / jnp.sqrt(jnp.where(sqrtarg > 0.0, sqrtarg, 1.0))

    # Offset: pi if cosbetamax < 0 else 0  (PrecVersion 320, not 330)
    off = jnp.where(cosbetamax < 0.0, jnp.pi, 0.0)

    ef1 = jnp.exp(kappa * f1)
    ef2 = jnp.exp(kappa * f2)
    denom = (f1 - f2) ** 2

    aC = (
        -ef1 * f1**4 * (off - beta1)
        + ef2
        * f2**3
        * (
            f2 * (-f1 + f2) * dbeta2
            + (-(f2 * (3.0 + f2 * kappa)) + f1 * (4.0 + f2 * kappa)) * (off - beta2)
        )
    ) / denom

    bC = (
        2.0 * ef1 * f1**4 * f2 * (off - beta1)
        + ef2
        * f2**3
        * (
            (f1 - f2) * f2 * (f1 + f2) * dbeta2
            - (-(f2sq * (2.0 + f2 * kappa)) + f1sq * (4.0 + f2 * kappa)) * (off - beta2)
        )
    ) / denom

    cC = (
        -ef1 * f1**4 * f2sq * (off - beta1)
        + ef2
        * f1
        * f2**4
        * (
            f2 * (-f1 + f2) * dbeta2
            + (-(f2 * (2.0 + f2 * kappa)) + f1 * (3.0 + f2 * kappa)) * (off - beta2)
        )
    ) / denom

    dC = off

    aRD = jnp.where(flat_RD, 0.0, aC)
    bRD = jnp.where(flat_RD, 0.0, bC)
    cRD = jnp.where(flat_RD, 0.0, cC)
    dRD = jnp.where(flat_RD, 0.0, dC)
    cosbeta_sign = jnp.sign(cosbetamax)

    return aRD, bRD, cRD, dRD, flat_RD, cosbeta_sign


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
    kappa = 2.0 * jnp.pi * dfdamp
    Mf2 = Mf * Mf
    Mf3 = Mf2 * Mf
    beta_fit = jnp.exp(-Mf * kappa) / Mf * (aRD / Mf + bRD / Mf2 + cRD / Mf3) + dRD
    beta_flat = jnp.arccos(cosbeta_sign)
    return jnp.where(flat_RD, beta_flat, beta_fit)


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
        alpha_spline:   3-tuple (Mf_nodes, alpha_nodes, dalpha_nodes) where
                        dalpha_nodes = jnp.gradient(alpha_nodes, Mf_nodes)
        cosbeta_spline: 2-tuple (Mf_nodes, cosbeta_nodes)
        alpha_params:   (aRD, bRD, cRD) from alphaMRD_coeff
        beta_params:    7-tuple (aRD, bRD, cRD, dRD, dfdamp, flat_RD, cosbeta_sign)
                        from betaMRD_coeff (plus dfdamp passed separately)
        ftrans_MRD:     transition frequency

    Returns:
        g(Mf): scalar
    """
    aRD_a, bRD_a, cRD_a = alpha_params
    aRD_b, bRD_b, cRD_b, dRD_b, dfdamp_b, flat_RD, cosbeta_sign = beta_params

    # Inspiral branch: interpolate from PN arrays
    cb_insp = jnp.interp(Mf, cosbeta_spline[0], cosbeta_spline[1])
    da_insp = jnp.interp(Mf, alpha_spline[0], alpha_spline[2])

    # MRD branch: analytical forms
    cb_mrd = jnp.cos(
        betaMRD(Mf, aRD_b, bRD_b, cRD_b, dRD_b, dfdamp_b, flat_RD, cosbeta_sign)
    )
    da_mrd = dalphaMRD(Mf, aRD_a, bRD_a, cRD_a)

    cb = jnp.where(Mf <= ftrans_MRD, cb_insp, cb_mrd)
    da = jnp.where(Mf <= ftrans_MRD, da_insp, da_mrd)

    return -cb * da


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
    deltaMF = Mf_grid[1] - Mf_grid[0]
    h = deltaMF / 4.0  # sub-step for Boole's rule

    # Build all quadrature points in one shot: shape (N-1, 5)
    # pts[i, k] = Mf_grid[i] + k*h  (left endpoint of each step + offset)
    left = Mf_grid[:-1]
    pts = left[:, None] + jnp.arange(5) * h  # (N-1, 5)

    # gamma_integrand is element-wise in Mf, so we can pass the full 2D array
    g = gamma_integrand(
        pts, alpha_spline, cosbeta_spline, alpha_params, beta_params, ftrans_MRD
    )

    weights = jnp.array([7.0, 32.0, 12.0, 32.0, 7.0])
    delta_gamma = (2.0 * h / 45.0) * (g @ weights)  # (N-1,)

    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(delta_gamma)])


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
    aRD_a, bRD_a, cRD_a = alpha_params

    # alpha_ref: use MRD analytical form if MfRef is past the inspiral,
    # otherwise interpolate from the PN spline (version 320 behaviour)
    alpha_ref_insp = jnp.interp(MfRef, alpha_spline[0], alpha_spline[1])
    alpha_ref_mrd = alphaMRD(MfRef, aRD_a, bRD_a, cRD_a)
    alpha_ref = jnp.where(MfRef <= ftrans_MRD, alpha_ref_insp, alpha_ref_mrd)

    alpha_offset = -alpha_ref + alpha0
    alpha_out = alpha_raw + alpha_offset

    # gamma_ref: interpolate the already-integrated gamma spline at MfRef,
    # then subtract (gamma_ref + epsilon0) from every point
    gamma_ref = jnp.interp(MfRef, gamma_spline[0], gamma_spline[1])
    gamma_out = gamma - gamma_ref - epsilon0

    return alpha_out, cosbeta, gamma_out


def integrate_spin_taylor(
    chi1x,
    chi1y,
    chi1z,
    chi2x,
    chi2y,
    chi2z,
    fmin,
    fRef,
    m1_SI,
    m2_SI,
    params: SpinTaylorT4Params,
    fCut_Hz: float,
    max_steps: int = 100_000,
):
    """
    JAX/diffrax version of integrate_spin_taylor.

    Integrates SpinTaylorT4 from fMin to fCut, with an optional backward pass
    if fRef > fmin.

    The output arrays are NaN-padded to max_steps per segment.
    Strip NaN rows before use:
        mask = jnp.isfinite(V_PN)
        V_PN, LNhat_PN, S1_PN, S2_PN = V_PN[mask], LNhat_PN[mask], ...

    For GPU/vmap usage, fix fRef == fmin (single forward pass) so the function
    is fully JIT-compilable over binary parameters.

    Args:
        chi1x/y/z:  spin components of body 1 at fRef (LNhat = z-axis frame)
        chi2x/y/z:  spin components of body 2 at fRef
        fmin:       starting GW frequency [Hz]
        fRef:       reference GW frequency [Hz];  must be >= fmin
        m1_SI:      mass of body 1 [kg]
        m2_SI:      mass of body 2 [kg]
        params:     SpinTaylorT4Params from build_spin_taylor_params()
        fCut_Hz:    upper integration limit [Hz]
        max_steps:  maximum ODE steps per segment (sets output array size)

    Returns:
        V_PN:     (N,) array of v = omega^(1/3)      [NaN-padded]
        LNhat_PN: (N, 3) unit orbital angular momentum [NaN-padded]
        S1_PN:    (N, 3) chi1 spin vector (evolved)    [NaN-padded]
        S2_PN:    (N, 3) chi2 spin vector (evolved)    [NaN-padded]
    """
    M_SI = m1_SI + m2_SI
    m1M = m1_SI / M_SI
    m2M = m2_SI / M_SI
    M_sec = _MTSUN_SI * (M_SI / 1.988409870698050731911960804878414216e30)

    omega_ref = PI * M_sec * fRef
    omega_min = PI * M_sec * fmin
    omega_cut = PI * M_sec * fCut_Hz

    y0 = jnp.array(
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

    rtol = 1e-10
    atol = 1e-12
    T_MAX = 1e8

    if abs(fRef - fmin) < 1e-4:
        # ── single forward pass ──────────────────────────────────────────
        Y = _run_segment(y0, omega_cut, +1, params, T_MAX, rtol, atol, max_steps)
    else:
        # ── two-pass: backward then forward ─────────────────────────────
        Y_bwd = _run_segment(y0, omega_min, -1, params, T_MAX, rtol, atol, max_steps)
        Y_fwd = _run_segment(y0, omega_cut, +1, params, T_MAX, rtol, atol, max_steps)

        # Strip NaN padding from both, reverse backward (fmin-first), concatenate.
        # Note: this uses numpy-level logic so it is NOT JIT-able end-to-end.
        import numpy as _np

        mask_bwd = _np.isfinite(_np.array(Y_bwd[:, 0]))
        mask_fwd = _np.isfinite(_np.array(Y_fwd[:, 0]))
        Y_bwd_valid = Y_bwd[mask_bwd][::-1]  # reverse: fRef→fmin becomes fmin→fRef
        Y_fwd_valid = Y_fwd[mask_fwd][1:]  # skip duplicate fRef point
        Y = jnp.concatenate([Y_bwd_valid, Y_fwd_valid], axis=0)

    # ── extract outputs ───────────────────────────────────────────────────
    V_PN = Y[:, 1] ** (1.0 / 3.0)
    LNhat_PN = Y[:, 2:5]
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

    Note:
        dfdamp (difference between (2,1) and (2,2) QNM damping frequencies,
        geometric units) is needed by betaMRD_coeff.  Pass it as a keyword
        argument; defaults to 0.0 if omitted.
    """
    import numpy as _np

    _MSUN = 1.988409870698050731911960804878414216e30  # kg
    M_sec = _MTSUN_SI * (m1_SI + m2_SI) / _MSUN  # total mass in seconds
    piGM = PI * M_sec  # pi * G * M / C^3

    # ------------------------------------------------------------------
    # Stage 1: SpinTaylorT4 ODE integration
    # ------------------------------------------------------------------
    fCut_Hz = (fRING + 8.0 * fDAMP) / M_sec
    fmin_Hz = float(Mf_array[0]) / M_sec

    V_PN_raw, LNhat_PN_raw, _, _ = integrate_spin_taylor(
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        fmin_Hz,
        fRef,
        m1_SI,
        m2_SI,
        params,
        fCut_Hz=fCut_Hz,
    )

    # Strip NaN padding (requires concrete evaluation; not JIT-able)
    n_valid = int(_np.sum(_np.isfinite(_np.array(V_PN_raw))))
    V_PN = V_PN_raw[:n_valid]
    LNhat_PN = LNhat_PN_raw[:n_valid]

    # ------------------------------------------------------------------
    # Stage 2: Extract inspiral Euler angles
    # ------------------------------------------------------------------
    Mf_PN, alpha_PN, cosbeta_PN = extract_alpha_cosbeta(
        V_PN, LNhat_PN, phiJ_Sf, thetaJN, kappa, piGM
    )

    # Transition frequency from end of PN integration
    deltaMF = Mf_PN[-1] - Mf_PN[-2]
    fmax_inspiral = Mf_PN[-1] - deltaMF
    fmax_inspiral = jnp.where(
        fmax_inspiral > fRING - fDAMP, 1.020 * fMECO, fmax_inspiral
    )
    ftrans_MRD = 0.98 * fmax_inspiral

    # Pre-compute derivative array once for reuse in all spline evaluations
    dalpha_PN = jnp.gradient(alpha_PN, Mf_PN)
    alpha_spline = (Mf_PN, alpha_PN, dalpha_PN)  # 3-tuple for gamma_integrand
    cosbeta_spline = (Mf_PN, cosbeta_PN)

    # ------------------------------------------------------------------
    # Stage 3: MRD analytical continuation coefficients
    # ------------------------------------------------------------------
    alpha_params = alphaMRD_coeff(Mf_PN, alpha_PN, fmax_inspiral)  # (aRD, bRD, cRD)

    aRD_b, bRD_b, cRD_b, dRD_b, flat_RD, cosbeta_sign = betaMRD_coeff(
        Mf_PN, cosbeta_PN, fmax_inspiral, fDAMP, dfdamp=0.0
    )
    beta_params = (aRD_b, bRD_b, cRD_b, dRD_b, 0.0, flat_RD, cosbeta_sign)

    # ------------------------------------------------------------------
    # Stage 4: Integrate gamma via Boole's rule
    # ------------------------------------------------------------------
    gamma_PN = integrate_gamma_boole(
        Mf_PN, alpha_spline, cosbeta_spline, alpha_params, beta_params, ftrans_MRD
    )
    gamma_spline = (Mf_PN, gamma_PN)

    # ------------------------------------------------------------------
    # Evaluate on the output frequency grid Mf_array
    # ------------------------------------------------------------------
    aRD_a, bRD_a, cRD_a = alpha_params

    # alpha: inspiral spline / MRD analytical
    alpha_insp = jnp.interp(Mf_array, Mf_PN, alpha_PN)
    alpha_mrd = alphaMRD(Mf_array, aRD_a, bRD_a, cRD_a)
    alpha_out = jnp.where(Mf_array <= ftrans_MRD, alpha_insp, alpha_mrd)

    # cosbeta: inspiral spline / MRD analytical
    cb_insp = jnp.interp(Mf_array, Mf_PN, cosbeta_PN)
    cb_mrd = jnp.cos(
        betaMRD(Mf_array, aRD_b, bRD_b, cRD_b, dRD_b, 0.0, flat_RD, cosbeta_sign)
    )
    cosbeta_out = jnp.where(Mf_array <= ftrans_MRD, cb_insp, cb_mrd)

    # gamma: interpolate the already-integrated PN+MRD array
    gamma_out = jnp.interp(Mf_array, Mf_PN, gamma_PN)

    # ------------------------------------------------------------------
    # Stage 5: Reference-point offsets
    # ------------------------------------------------------------------
    MfRef = fRef * M_sec
    alpha_out, cosbeta_out, gamma_out = apply_angle_offsets(
        Mf_array,
        alpha_out,
        cosbeta_out,
        gamma_out,
        MfRef,
        alpha_params,
        beta_params,
        gamma_spline,
        ftrans_MRD,
        alpha0,
        epsilon0,
    )

    return alpha_out, cosbeta_out, gamma_out
