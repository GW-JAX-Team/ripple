"""
JAX implementation of LALSimInspiralSpinTaylor.c

Authors: Harsh Narola (h.b.narola@uu.nl)
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, Event
from jax import jit, lax, tree_util
from dataclasses import dataclass

# Import coefficient functions
from . import LALSimInspiralPNCoefficients as pn_coeffs

# ============================================================================
# PHYSICAL CONSTANTS AND TOLERANCES
# ============================================================================

# Solar mass and related quantities (exact LAL values)
LAL_MSUN_SI = 1.988409870698050731911960804878414216e30  # kg
LAL_MTSUN_SI = 4.925490947641266978197229498498379006e-6  # seconds
LAL_G_SI = 6.67430e-11  # Gravitational constant (m³/kg/s²)
LAL_C_SI = 299792458.0  # Speed of light (m/s)
LAL_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant

# Integration tolerances
LAL_ST4_ABSOLUTE_TOLERANCE = 1.0e-12
LAL_ST4_RELATIVE_TOLERANCE = 1.0e-12
LAL_NUM_ST4_VARIABLES = 14

# Numerical precision
LAL_REAL4_EPS = jnp.float32(2.0**-23)

# Spin order constants
LAL_SIM_INSPIRAL_SPIN_ORDER_ALL = -1
LAL_SIM_INSPIRAL_SPIN_ORDER_35PN = 7
LAL_SIM_INSPIRAL_SPIN_ORDER_3PN = 6
LAL_SIM_INSPIRAL_SPIN_ORDER_25PN = 5
LAL_SIM_INSPIRAL_SPIN_ORDER_2PN = 4
LAL_SIM_INSPIRAL_SPIN_ORDER_15PN = 3
LAL_SIM_INSPIRAL_SPIN_ORDER_1PN = 2
LAL_SIM_INSPIRAL_SPIN_ORDER_05PN = 1
LAL_SIM_INSPIRAL_SPIN_ORDER_0PN = 0

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class REAL8TimeSeries:
    """JAX equivalent of LAL REAL8TimeSeries."""

    data: jax.Array
    deltaT: float
    epoch: float = 0.0


@dataclass
class EvolutionParameters:
    """Parameters for SpinTaylor evolution."""

    # Basic binary parameters
    m1_SI: float
    m2_SI: float
    fStart: float
    fEnd: float

    # Tidal and quadrupole parameters
    lambda1: float
    lambda2: float
    quadparam1: float
    quadparam2: float

    # PN orders
    spinO: int
    tideO: int
    phaseO: int
    lscorr: int
    phenomtp: bool

    # Derived quantities
    m1sec: float
    m2sec: float
    Msec: float
    Mcsec: float
    eta: float
    norm1: float
    norm2: float

    # Coefficient dictionaries
    wdot_coeffs: Dict[str, float]
    spin_coeffs: Dict[str, float]
    energy_coeffs: Dict[str, float]

    # Single coefficients
    wdotnewt: float
    omegashiftS1: float
    omegashiftS2: float

    def tree_flatten(self):
        # Extract all the values (children) that JAX should trace
        children = (
            self.m1_SI,
            self.m2_SI,
            self.fStart,
            self.fEnd,
            self.lambda1,
            self.lambda2,
            self.quadparam1,
            self.quadparam2,
            self.spinO,
            self.tideO,
            self.phaseO,
            self.lscorr,
            self.phenomtp,
            self.m1sec,
            self.m2sec,
            self.Msec,
            self.Mcsec,
            self.eta,
            self.norm1,
            self.norm2,
            self.wdot_coeffs,
            self.spin_coeffs,
            self.energy_coeffs,
            self.wdotnewt,
            self.omegashiftS1,
            self.omegashiftS2,
        )
        # No auxiliary data needed since all fields should be JAX-traceable
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


tree_util.register_pytree_node(
    EvolutionParameters,
    EvolutionParameters.tree_flatten,
    EvolutionParameters.tree_unflatten,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


@jit
def normsq(x: float, y: float, z: float) -> float:
    """Compute squared magnitude of 3-vector."""
    return x * x + y * y + z * z


@jit
def cdot(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    """Compute dot product of two 3-vectors."""
    return ax * bx + ay * by + az * bz


@jit
def cross_vec(a: jax.Array, b: jax.Array) -> jax.Array:
    """Compute cross product of two 3-vectors."""
    return jnp.array(
        [
            a[1] * b[2] - a[2] * b[1],  # x-component
            a[2] * b[0] - a[0] * b[2],  # y-component
            a[0] * b[1] - a[1] * b[0],  # z-component
        ]
    )


@jit
def omegashift(
    S1sq: float,
    S2sq: float,
    S1S2: float,
    LNhS1: float,
    LNhS2: float,
    OmS1: float,
    OmS2: float,
) -> float:
    """
    Compute spin-orbit corrections to orbital frequency.

    Args:
        S1sq, S2sq: Squared spin magnitudes |S₁|², |S₂|²
        S1S2: Spin-spin dot product S₁ · S₂
        LNhS1, LNhS2: Orbital angular momentum projections L̂N · S₁, L̂N · S₂
        OmS1, OmS2: Spin-orbit coupling coefficients

    Returns:
        Fractional frequency shift Δω/ω
    """
    return -0.25 * (
        OmS1 * OmS1 * (S1sq - LNhS1 * LNhS1)
        + OmS2 * OmS2 * (S2sq - LNhS2 * LNhS2)
        + 2.0 * OmS1 * OmS2 * (S1S2 - LNhS1 * LNhS2)
    )


def _get(params: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely extract parameter from dictionary with default."""
    if isinstance(params, dict):
        return params.get(key, default)
    else:
        return getattr(params, key, default)


# ============================================================================
# SETUP AND COEFFICIENT COMPUTATION
# ============================================================================


@jit
def XLALSimInspiralSpinTaylorT4Setup(
    m1_SI: float,
    m2_SI: float,
    fStart: float,
    fEnd: float,
    lambda1: float,
    lambda2: float,
    quadparam1: float,
    quadparam2: float,
    spinO: int,
    tideO: int,
    phaseO: int,
    lscorr: int,
    phenomtp: bool,
) -> EvolutionParameters:
    """
    Setup all parameters and coefficients for SpinTaylorT4 evolution.

    Returns:
        EvolutionParameters object with all computed coefficients
    """
    # Basic derived quantities
    m1sec = m1_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    m2sec = m2_SI / LAL_MSUN_SI * LAL_MTSUN_SI
    Msec = m1sec + m2sec
    eta = m1sec * m2sec / (Msec * Msec)
    Mcsec = Msec * jnp.power(eta, 0.6)
    norm1 = m1sec * m1sec / Msec / Msec
    norm2 = m2sec * m2sec / Msec / Msec

    # Mass ratios
    m1 = m1_SI / LAL_MSUN_SI
    m2 = m2_SI / LAL_MSUN_SI
    M = m1 + m2
    m1M = m1 / M
    m2M = m2 / M

    # Newtonian wdot coefficient
    wdotnewt = 96.0 / 5.0 * eta

    # Compute PN coefficients for wdot
    wdot_coeffs = XLALSimInspiralTaylorT4SetupWdotCoeffs(
        eta, m1M, m2M, lambda1, lambda2
    )

    # Compute spin coefficients
    spin_coeffs = pn_coeffs.add_wdot_spin_coefficients_to_setup(
        m1M, m2M, eta, quadparam1, quadparam2
    )

    # Add additional spin coefficients
    additional_spin_coeffs = XLALSimInspiralSpinTaylorSetupSpinCoeffs(
        m1M, m2M, quadparam1, quadparam2
    )
    spin_coeffs.update(additional_spin_coeffs)

    # Omega shift coefficients
    omegashiftS1 = pn_coeffs.XLALSimInspiralLDot_3PNSOCoeff(m1M)
    omegashiftS2 = pn_coeffs.XLALSimInspiralLDot_3PNSOCoeff(m2M)

    # Energy coefficients
    energy_coeffs = XLALSimInspiralSetEnergyPNTermsAvg(
        m1M, m2M, eta, lambda1, lambda2, quadparam1, quadparam2
    )

    return EvolutionParameters(
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        fStart=fStart,
        fEnd=fEnd,
        lambda1=lambda1,
        lambda2=lambda2,
        quadparam1=quadparam1,
        quadparam2=quadparam2,
        spinO=spinO,
        tideO=tideO,
        phaseO=phaseO,
        lscorr=lscorr,
        phenomtp=phenomtp,
        m1sec=m1sec,
        m2sec=m2sec,
        Msec=Msec,
        Mcsec=Mcsec,
        eta=eta,
        norm1=norm1,
        norm2=norm2,
        wdot_coeffs=wdot_coeffs,
        spin_coeffs=spin_coeffs,
        energy_coeffs=energy_coeffs,
        wdotnewt=wdotnewt,
        omegashiftS1=omegashiftS1,
        omegashiftS2=omegashiftS2,
    )


@jit
def XLALSimInspiralTaylorT4SetupWdotCoeffs(
    eta: float, m1M: float, m2M: float, lambda1: float, lambda2: float
) -> Dict[str, float]:
    """Compute PN coefficients for domega/dt."""
    coeffs = {}

    # Standard PN coefficients
    coeffs["0PN"] = 1.0
    coeffs["1PN"] = 0.0
    coeffs["2PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_2PNCoeff(eta)
    coeffs["3PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_3PNCoeff(eta)
    coeffs["4PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_4PNCoeff(eta)
    coeffs["5PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_5PNCoeff(eta)
    coeffs["6PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_6PNCoeff(eta)
    coeffs["6PN_log"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_6PNLogCoeff(eta)
    coeffs["7PN"] = pn_coeffs.XLALSimInspiralTaylorT4wdot_7PNCoeff(eta)

    # Tidal coefficients
    coeffs["10PN_tidal"] = (
        lambda1 * pn_coeffs.XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(m1M)
        + lambda2 * pn_coeffs.XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(m2M)
    )
    coeffs["12PN_tidal"] = (
        lambda1 * pn_coeffs.XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(m1M)
        + lambda2 * pn_coeffs.XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(m2M)
    )

    return coeffs


@jit
def XLALSimInspiralSpinTaylorSetupSpinCoeffs(
    m1M: float, m2M: float, quadparam1: float, quadparam2: float
) -> Dict[str, float]:
    """Compute additional spin-related coefficients."""
    coeffs = {}

    # 6PN spin coefficients
    coeffs["S1dot6S2Avg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS2CoeffAvg(m1M)
    coeffs["S2dot6S1Avg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS2CoeffAvg(m2M)
    coeffs["S1dot6S2OAvg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS2OCoeffAvg(m1M)
    coeffs["S1dot6S1OAvg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS1OCoeffAvg(m1M)
    coeffs["S2dot6S1OAvg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS2OCoeffAvg(m2M)
    coeffs["S2dot6S2OAvg"] = pn_coeffs.XLALSimInspiralSpinDot_6PNS1OCoeffAvg(m2M)
    coeffs["S1dot6QMS1OAvg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(m1M)
    )
    coeffs["S2dot6QMS2OAvg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(m2M)
    )

    # 3PN and 5PN spin coefficients
    coeffs["S1dot3"] = pn_coeffs.XLALSimInspiralSpinDot_3PNCoeff(m1M)
    coeffs["S2dot3"] = pn_coeffs.XLALSimInspiralSpinDot_3PNCoeff(m2M)
    coeffs["S1dot5"] = pn_coeffs.XLALSimInspiralSpinDot_5PNCoeff(m1M)
    coeffs["S2dot5"] = pn_coeffs.XLALSimInspiralSpinDot_5PNCoeff(m2M)

    # 4PN spin coefficients
    coeffs["S1dot4S2Avg"] = pn_coeffs.XLALSimInspiralSpinDot_4PNS2CoeffAvg
    coeffs["S1dot4S2OAvg"] = pn_coeffs.XLALSimInspiralSpinDot_4PNS2OCoeffAvg
    coeffs["S1dot4QMS1OAvg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(m1M)
    )
    coeffs["S2dot4QMS2OAvg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(m2M)
    )

    # L coefficients for lscorr
    coeffs["cS1"] = pn_coeffs.XLALSimInspiralL_3PNSicoeffAvg(m1M)
    coeffs["cS2"] = pn_coeffs.XLALSimInspiralL_3PNSiLcoeffAvg(m1M)
    coeffs["cS1L"] = pn_coeffs.XLALSimInspiralL_3PNSicoeffAvg(m2M)
    coeffs["cS2L"] = pn_coeffs.XLALSimInspiralL_3PNSiLcoeffAvg(m2M)

    return coeffs


@jit
def XLALSimInspiralSetEnergyPNTermsAvg(
    m1M: float,
    m2M: float,
    eta: float,
    lambda1: float,
    lambda2: float,
    quadparam1: float,
    quadparam2: float,
) -> Dict[str, float]:
    """Compute energy-related PN coefficients."""
    coeffs = {}

    # Basic energy coefficients
    coeffs["Ecoeff"] = jnp.array(
        [
            1.0,  # 0PN
            0.0,  # 0.5PN
            pn_coeffs.XLALSimInspiralPNEnergy_2PNCoeff(eta),  # 1PN
            0.0,  # 1.5PN
            pn_coeffs.XLALSimInspiralPNEnergy_4PNCoeff(eta),  # 2PN
            0.0,  # 2.5PN
            pn_coeffs.XLALSimInspiralPNEnergy_6PNCoeff(eta),  # 3PN
            0.0,  # 3.5PN
        ]
    )

    # Spin-orbit energy coefficients
    coeffs["E3S1O"] = pn_coeffs.XLALSimInspiralPNEnergy_3PNSOCoeff(m1M)
    coeffs["E3S2O"] = pn_coeffs.XLALSimInspiralPNEnergy_3PNSOCoeff(m2M)
    coeffs["E5S1O"] = pn_coeffs.XLALSimInspiralPNEnergy_5PNSOCoeff(m1M)
    coeffs["E5S2O"] = pn_coeffs.XLALSimInspiralPNEnergy_5PNSOCoeff(m2M)
    coeffs["E7S1O"] = pn_coeffs.XLALSimInspiralPNEnergy_7PNSOCoeff(m1M)
    coeffs["E7S2O"] = pn_coeffs.XLALSimInspiralPNEnergy_7PNSOCoeff(m2M)

    # Spin-spin energy coefficients
    coeffs["E4S1S2Avg"] = pn_coeffs.XLALSimInspiralPNEnergy_4PNS1S2CoeffAvg(eta)
    coeffs["E4S1OS2OAvg"] = pn_coeffs.XLALSimInspiralPNEnergy_4PNS1OS2OCoeffAvg(eta)
    coeffs["E6S1S2Avg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1S2CoeffAvg(eta)
    coeffs["E6S1OS2OAvg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1OS2OCoeffAvg(eta)

    # Quadrupole-monopole energy coefficients
    coeffs["E4QMS1S1Avg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(m1M)
    )
    coeffs["E4QMS2S2Avg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(m2M)
    )
    coeffs["E4QMS1OS1OAvg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(m1M)
    )
    coeffs["E4QMS2OS2OAvg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(m2M)
    )

    # Higher-order spin-spin energy coefficients
    coeffs["E6S1S1Avg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(m1M)
    coeffs["E6S2S2Avg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(m2M)
    coeffs["E6QMS1S1Avg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(m1M)
    )
    coeffs["E6QMS2S2Avg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(m2M)
    )
    coeffs["E6S1OS1OAvg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(m1M)
    coeffs["E6S2OS2OAvg"] = pn_coeffs.XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(m2M)
    coeffs["E6QMS1OS1OAvg"] = (
        quadparam1 * pn_coeffs.XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(m1M)
    )
    coeffs["E6QMS2OS2OAvg"] = (
        quadparam2 * pn_coeffs.XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(m2M)
    )

    # Tidal energy coefficients
    coeffs["Etidal10"] = lambda1 * pn_coeffs.XLALSimInspiralPNEnergy_10PNTidalCoeff(
        m1M
    ) + lambda2 * pn_coeffs.XLALSimInspiralPNEnergy_10PNTidalCoeff(m2M)
    coeffs["Etidal12"] = lambda1 * pn_coeffs.XLALSimInspiralPNEnergy_12PNTidalCoeff(
        m1M
    ) + lambda2 * pn_coeffs.XLALSimInspiralPNEnergy_12PNTidalCoeff(m2M)

    return coeffs


# ============================================================================
# EVOLUTION DERIVATIVES
# ============================================================================
@jit
def XLALSimInspiralSpinTaylorT4wdot(omega: float, params: Dict[str, Any]) -> float:
    """
    Compute domega/dt with all PN corrections.

    Args:
        omega: Orbital frequency
        params: Evolution parameters

    Returns:
        domega/dt
    """
    v = jnp.cbrt(omega)
    v2 = v * v
    v11 = omega * omega * omega * v2

    # Get coefficients
    wdot_coeffs = params["wdot_coeffs"]

    # Compute spin contributions
    LNhdotS1 = params.get("LNhdotS1", 0.0)
    LNhdotS2 = params.get("LNhdotS2", 0.0)
    S1dotS2 = params.get("S1dotS2", 0.0)
    S1sq = params.get("S1sq", 0.0)
    S2sq = params.get("S2sq", 0.0)
    spinO = params.get("spinO", -1)

    wspin_dict = pn_coeffs.compute_wdotspin(
        params["spin_coeffs"], LNhdotS1, LNhdotS2, S1dotS2, S1sq, S2sq, spinO
    )

    # Build PN series
    domega = (
        params["wdotnewt"]
        * v11
        * (
            wdot_coeffs["0PN"]
            + v
            * (
                wdot_coeffs["1PN"]
                + v
                * (
                    wdot_coeffs["2PN"]
                    + v
                    * (
                        wdot_coeffs["3PN"]
                        + wspin_dict["wspin3"]
                        + v
                        * (
                            wdot_coeffs["4PN"]
                            + wspin_dict["wspin4Avg"]
                            + v
                            * (
                                wdot_coeffs["5PN"]
                                + wspin_dict["wspin5"]
                                + v
                                * (
                                    wdot_coeffs["6PN"]
                                    + wspin_dict["wspin6Avg"]
                                    + wdot_coeffs["6PN_log"] * jnp.log(v)
                                    + v
                                    * (
                                        wdot_coeffs["7PN"]
                                        + omega
                                        * (
                                            wdot_coeffs["10PN_tidal"]
                                            + v2 * wdot_coeffs["12PN_tidal"]
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return domega


@jit
def XLALSimInspiralSpinDerivativesAvg(
    v: float,
    LNhat: jax.Array,
    E1: jax.Array,
    S1: jax.Array,
    S2: jax.Array,
    LNhdotS1: float,
    LNhdotS2: float,
    params: Dict[str, Any],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute derivatives of spins and angular momentum vectors.

    Args:
        v: PN velocity parameter
        LNhat: Unit orbital angular momentum vector
        E1: Reference direction vector
        S1, S2: Spin vectors
        LNhdotS1, LNhdotS2: Dot products LNhat·S1, LNhat·S2
        params: Evolution parameters

    Returns:
        Tuple of (dLNhat, dE1, dS1, dS2)
    """
    # Get parameters
    eta = params["eta"]
    spinO = params.get("spinO", -1)
    lscorr = params.get("lscorr", 0)
    phenomtp = params.get("phenomtp", False)
    spin_coeffs = params["spin_coeffs"]

    # Basic derived quantities
    v2 = v * v
    omega = v * v2
    omega2 = omega * omega
    LN0mag = eta / v
    LNmag = LN0mag

    # Add 1PN correction to LNmag if spinO >= 5
    include_1PN = (spinO >= 5) | (spinO < 0)
    L1PN = pn_coeffs.XLALSimInspiralL_2PN(eta)
    LNmag = jnp.where(include_1PN, LNmag + LN0mag * v2 * L1PN, LNmag)

    # Boolean masks for different PN orders
    m3 = jnp.where((spinO >= 3) | (spinO < 0), 1.0, 0.0)
    m4 = jnp.where((spinO >= 4) | (spinO < 0), 1.0, 0.0)
    m5 = jnp.where((spinO >= 5) | (spinO < 0), 1.0, 0.0)
    m6 = jnp.where((spinO >= 6) | (spinO < 0), 1.0, 0.0)
    m7 = jnp.where((spinO >= 7) | (spinO < 0), 1.0, 0.0)

    # Precompute cross products
    LNh_x_S1 = cross_vec(LNhat, S1)
    LNh_x_S2 = cross_vec(LNhat, S2)
    S1_x_S2 = cross_vec(S1, S2)

    # Leading order (v^5) contributions
    v5 = omega * v2
    dS1_lo = spin_coeffs["S1dot3"] * v5 * LNh_x_S1 * m3
    dS2_lo = spin_coeffs["S2dot3"] * v5 * LNh_x_S2 * m3
    dLNhat_lo = -(dS1_lo + dS2_lo)

    # Next-to-leading order (v^6) contributions
    pref_v6 = omega2 * m4
    dS1_v6 = pref_v6 * (
        -spin_coeffs["S1dot4S2Avg"] * S1_x_S2
        + spin_coeffs["S1dot4S2OAvg"] * LNhdotS2 * LNh_x_S1
        + spin_coeffs["S1dot4QMS1OAvg"] * LNhdotS1 * LNh_x_S1
    )
    dS2_v6 = pref_v6 * (
        spin_coeffs["S1dot4S2Avg"] * S1_x_S2
        + spin_coeffs["S1dot4S2OAvg"] * LNhdotS1 * LNh_x_S2
        + spin_coeffs["S2dot4QMS2OAvg"] * LNhdotS2 * LNh_x_S2
    )
    dLNhat_v6 = -(dS1_v6 + dS2_v6)

    # v^7 contributions
    v7 = omega2 * v
    dS1_v7 = spin_coeffs["S1dot5"] * v7 * LNh_x_S1 * m5
    dS2_v7 = spin_coeffs["S2dot5"] * v7 * LNh_x_S2 * m5
    dLNhat_v7 = -(dS1_v7 + dS2_v7)

    # Higher-order contributions (branch based on phenomtp)
    def compute_non_phenom_v8():
        v8 = omega2 * v2
        dS1_v8 = v8 * (
            -spin_coeffs["S1dot6S2Avg"] * S1_x_S2
            + (
                spin_coeffs["S1dot6S1OAvg"] * LNhdotS1
                + spin_coeffs["S1dot6S2OAvg"] * LNhdotS2
            )
            * LNh_x_S1
            + spin_coeffs["S1dot6QMS1OAvg"] * LNhdotS1 * LNh_x_S1
        )
        dS2_v8 = v8 * (
            spin_coeffs["S2dot6S1Avg"] * S1_x_S2
            + (
                spin_coeffs["S2dot6S1OAvg"] * LNhdotS1
                + spin_coeffs["S2dot6S2OAvg"] * LNhdotS2
            )
            * LNh_x_S2
            + spin_coeffs["S2dot6QMS2OAvg"] * LNhdotS2 * LNh_x_S2
        )
        return dS1_v8 * m6, dS2_v8 * m6, jnp.zeros(3)

    def compute_phenom_branch():
        # Phenomtp branch - placeholder for higher-order terms
        omega3 = omega2 * omega
        S1dot7S2 = _get(spin_coeffs, "S1dot7S2", 0.0)
        S2dot7S1 = _get(spin_coeffs, "S2dot7S1", 0.0)
        dS1_p = S1dot7S2 * omega3 * LNh_x_S1 * m7
        dS2_p = S2dot7S1 * omega3 * LNh_x_S2 * m7
        return dS1_p, dS2_p, jnp.zeros(3)

    # Select branch
    dS1_v8, dS2_v8, dLNhat_v8 = lax.cond(
        jnp.logical_not(phenomtp),
        lambda _: compute_non_phenom_v8(),
        lambda _: compute_phenom_branch(),
        operand=None,
    )

    # LSCorr corrections
    lscorr_pref = eta * v2 * lscorr
    dLNhat_lscorr = -lscorr_pref * (
        (spin_coeffs["cS1"] * dS1_lo + spin_coeffs["cS2"] * dS2_lo) * m5
        + (spin_coeffs["cS1"] * dS1_v6 + spin_coeffs["cS2"] * dS2_v6) * m6
        + (spin_coeffs["cS1L"] * LNhdotS1 + spin_coeffs["cS2L"] * LNhdotS2)
        * dLNhat_lo
        / LN0mag
        * m6
    )

    # Combine all contributions
    dS1_total = dS1_lo + dS1_v6 + dS1_v7 + dS1_v8
    dS2_total = dS2_lo + dS2_v6 + dS2_v7 + dS2_v8
    dLNhat_total = dLNhat_lo + dLNhat_v6 + dLNhat_v7 + dLNhat_v8 + dLNhat_lscorr

    # Normalize dLNhat
    dLNhat = dLNhat_total / LNmag

    # Compute angular velocity Om = LNhat × dLNhat
    Om = cross_vec(LNhat, dLNhat)

    # Final derivatives
    dLNh_final = cross_vec(Om, LNhat)
    dE1_final = cross_vec(Om, E1)

    return dLNh_final, dE1_final, dS1_total, dS2_total


@jit
def XLALSimInspiralSpinTaylorT4DerivativesAvg(
    t: float, y: jax.Array, params: Dict[str, Any]
) -> jax.Array:
    """
    Main derivative function for SpinTaylorT4 evolution.

    Args:
        t: Current time
        y: Current state [phi, omega, LNx, LNy, LNz, S1x, S1y, S1z, S2x, S2y, S2z, E1x, E1y, E1z]
        params: Evolution parameters

    Returns:
        Array of derivatives dy/dt
    """
    # Extract state variables
    phi, omega = y[0], y[1]
    LNhx, LNhy, LNhz = y[2], y[3], y[4]
    S1x, S1y, S1z = y[5], y[6], y[7]
    S2x, S2y, S2z = y[8], y[9], y[10]
    E1x, E1y, E1z = y[11], y[12], y[13]

    # Guard against invalid omega
    omega = jnp.where(omega <= 0, 1e-20, omega)

    # Compute basic quantities
    v = jnp.cbrt(omega)

    # Compute dot products
    LNhdotS1 = cdot(LNhx, LNhy, LNhz, S1x, S1y, S1z)
    LNhdotS2 = cdot(LNhx, LNhy, LNhz, S2x, S2y, S2z)
    S1dotS2 = cdot(S1x, S1y, S1z, S2x, S2y, S2z)
    S1sq = normsq(S1x, S1y, S1z)
    S2sq = normsq(S2x, S2y, S2z)

    # Update params with current dot products for domega calculation
    params_with_dots = params.copy()
    params_with_dots.update(
        {
            "LNhdotS1": LNhdotS1,
            "LNhdotS2": LNhdotS2,
            "S1dotS2": S1dotS2,
            "S1sq": S1sq,
            "S2sq": S2sq,
        }
    )

    # Compute domega/dt
    domega = XLALSimInspiralSpinTaylorT4wdot(omega, params_with_dots)

    # Compute spin derivatives
    LNhat = jnp.array([LNhx, LNhy, LNhz])
    E1 = jnp.array([E1x, E1y, E1z])
    S1 = jnp.array([S1x, S1y, S1z])
    S2 = jnp.array([S2x, S2y, S2z])

    dLNh, dE1, dS1, dS2 = XLALSimInspiralSpinDerivativesAvg(
        v, LNhat, E1, S1, S2, LNhdotS1, LNhdotS2, params_with_dots
    )

    # Compute dphi/dt with spin corrections
    shift = omegashift(
        S1sq,
        S2sq,
        S1dotS2,
        LNhdotS1,
        LNhdotS2,
        params["omegashiftS1"],
        params["omegashiftS2"],
    )
    dphi = omega * (1 + omega * omega * shift)

    # Pack derivatives
    return jnp.array(
        [
            dphi,
            domega,
            dLNh[0],
            dLNh[1],
            dLNh[2],
            dS1[0],
            dS1[1],
            dS1[2],
            dS2[0],
            dS2[1],
            dS2[2],
            dE1[0],
            dE1[1],
            dE1[2],
        ]
    )


# ============================================================================
# TERMINATION CONDITIONS
# ============================================================================
@jit
def XLALSimInspiralGetEnergyPNTermsAvg(
    LNhdotS1: float,
    LNhdotS2: float,
    S1sq: float,
    S2sq: float,
    S1dotS2: float,
    params: Dict[str, Any],
) -> Tuple[float, float, float, float, float]:
    """
    Compute spin corrections to energy at various PN orders.

    Returns:
        Tuple of (Espin3, Espin4, Espin5, Espin6, Espin7)
    """
    spinO = params.get("spinO", -1)
    phenomtp = params.get("phenomtp", False)
    energy_coeffs = params["energy_coeffs"]

    # Initialize
    Espin3 = Espin4 = Espin5 = Espin6 = Espin7 = 0.0

    # Boolean masks for inclusion
    include_15PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_15PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_2PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_2PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_25PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_25PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_3PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_3PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_ALL = spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL

    # 1.5PN spin-orbit
    Espin3 = jnp.where(
        include_15PN,
        energy_coeffs["E3S1O"] * LNhdotS1 + energy_coeffs["E3S2O"] * LNhdotS2,
        Espin3,
    )

    # 2PN spin-spin and quadrupole-monopole
    Espin4_SS = (
        energy_coeffs["E4S1S2Avg"] * S1dotS2
        + energy_coeffs["E4S1OS2OAvg"] * LNhdotS1 * LNhdotS2
    )
    Espin4_QM = (
        energy_coeffs["E4QMS1S1Avg"] * S1sq
        + energy_coeffs["E4QMS2S2Avg"] * S2sq
        + energy_coeffs["E4QMS1OS1OAvg"] * LNhdotS1 * LNhdotS1
        + energy_coeffs["E4QMS2OS2OAvg"] * LNhdotS2 * LNhdotS2
    )

    Espin4 = jnp.where(include_2PN, Espin4_SS + Espin4_QM, Espin4)

    # 2.5PN spin-orbit
    Espin5 = jnp.where(
        include_25PN,
        energy_coeffs["E5S1O"] * LNhdotS1 + energy_coeffs["E5S2O"] * LNhdotS2,
        Espin5,
    )

    # 3PN spin-spin (only if NOT phenomtp)
    Espin6_val = (
        energy_coeffs["E6S1S2Avg"] * S1dotS2
        + energy_coeffs["E6S1OS2OAvg"] * LNhdotS1 * LNhdotS2
        + (energy_coeffs["E6S1S1Avg"] + energy_coeffs["E6QMS1S1Avg"]) * S1sq
        + (energy_coeffs["E6S2S2Avg"] + energy_coeffs["E6QMS2S2Avg"]) * S2sq
        + (energy_coeffs["E6S1OS1OAvg"] + energy_coeffs["E6QMS1OS1OAvg"]) * LNhdotS1**2
        + (energy_coeffs["E6S2OS2OAvg"] + energy_coeffs["E6QMS2OS2OAvg"]) * LNhdotS2**2
    )

    Espin6 = jnp.where(include_3PN & (~phenomtp), Espin6_val, Espin6)

    # 3.5PN (only if ALL and phenomtp)
    Espin7 = jnp.where(
        include_ALL & phenomtp,
        energy_coeffs["E7S1O"] * LNhdotS1 + energy_coeffs["E7S2O"] * LNhdotS2,
        Espin7,
    )

    return Espin3, Espin4, Espin5, Espin6, Espin7


@jit
def XLALSimInspiralSpinTaylorStoppingTest(
    t: float, y: jax.Array, dy: jax.Array, params: Dict[str, Any]
) -> float:
    """
    Determine whether to continue or stop integration.

    Args:
        t: Current time
        y: Current state
        dy: Current derivatives
        params: Evolution parameters

    Returns:
        Positive value to continue, negative to stop
    """
    # Extract state
    omega = y[1]
    v = jnp.cbrt(omega)
    LNhx, LNhy, LNhz = y[2], y[3], y[4]
    S1x, S1y, S1z = y[5], y[6], y[7]
    S2x, S2y, S2z = y[8], y[9], y[10]

    # Compute quantities needed for tests
    LNhdotS1 = cdot(LNhx, LNhy, LNhz, S1x, S1y, S1z)
    LNhdotS2 = cdot(LNhx, LNhy, LNhz, S2x, S2y, S2z)
    S1sq = normsq(S1x, S1y, S1z)
    S2sq = normsq(S2x, S2y, S2z)
    S1dotS2 = cdot(S1x, S1y, S1z, S2x, S2y, S2z)

    # Get parameters
    M = _get(params, "Msec", 0.0) / LAL_MTSUN_SI
    fStart = _get(params, "fStart", 0.0)
    fEnd = _get(params, "fEnd", 0.0)

    # Convert frequencies to omega
    omegaStart = jnp.pi * M * LAL_MTSUN_SI * fStart
    omegaEnd = jnp.pi * M * LAL_MTSUN_SI * fEnd

    # Get spin corrections to energy
    Espin3, Espin4, Espin5, Espin6, Espin7 = XLALSimInspiralGetEnergyPNTermsAvg(
        LNhdotS1, LNhdotS2, S1sq, S2sq, S1dotS2, params
    )

    # Energy coefficients
    Ecoeff = params["energy_coeffs"]["Ecoeff"]
    Etidal10 = params["energy_coeffs"]["Etidal10"]
    Etidal12 = params["energy_coeffs"]["Etidal12"]

    # Energy test: dE/domega
    v2 = v * v
    test = 2.0 + v2 * (
        4.0 * Ecoeff[2]
        + v
        * (
            5.0 * (Ecoeff[3] + Espin3)
            + v
            * (
                6.0 * (Ecoeff[4] + Espin4)
                + v
                * (
                    7.0 * (Ecoeff[5] + Espin5)
                    + v
                    * (
                        8.0 * (Ecoeff[6] + Espin6)
                        + v
                        * (
                            9.0 * (Ecoeff[7] + Espin7)
                            + v * v * v * (12.0 * Etidal10 + v2 * (14.0 * Etidal12))
                        )
                    )
                )
            )
        )
    )

    # Run multiple tests
    freq_above = (
        (jnp.abs(omegaEnd) > LAL_REAL4_EPS)
        & (omegaEnd > omegaStart)
        & (omega > omegaEnd)
    )
    freq_below = (
        (jnp.abs(omegaEnd) > 1e-6) & (omegaEnd < omegaStart) & (omega < omegaEnd)
    )
    energy_fail = test < 0.0
    omega_nan = jnp.isnan(omega)
    large_v = v >= 1.0

    # Return result (negative = stop, positive = continue)
    result = jnp.where(
        freq_above,
        -1.0,
        jnp.where(
            freq_below,
            -2.0,
            jnp.where(
                energy_fail,
                -3.0,
                jnp.where(omega_nan, -4.0, jnp.where(large_v, -5.0, 1.0)),
            ),
        ),
    )

    return result


def stopping_event(t: float, y: jax.Array, args: Dict[str, Any], **kwargs) -> float:
    """Event function for diffrax integration."""
    dy = XLALSimInspiralSpinTaylorT4DerivativesAvg(t, y, args)
    return XLALSimInspiralSpinTaylorStoppingTest(t, y, dy, args)


# ============================================================================
# MAIN EVOLUTION FUNCTION
# ============================================================================


def XLALSimInspiralSpinTaylorPNEvolveOrbit(
    deltaT: float,
    m1_SI: float,
    m2_SI: float,
    fStart: float,
    fEnd: float,
    s1x: float,
    s1y: float,
    s1z: float,
    s2x: float,
    s2y: float,
    s2z: float,
    lnhatx: float,
    lnhaty: float,
    lnhatz: float,
    e1x: float,
    e1y: float,
    e1z: float,
    lambda1: float,
    lambda2: float,
    quadparam1: float,
    quadparam2: float,
    spinO: int,
    tideO: int,
    phaseO: int,
    lscorr: int,
    phenomtp: bool = False,
    max_steps: int = 10000,
) -> Tuple[REAL8TimeSeries, ...]:
    """
    Evolve SpinTaylor inspiral orbit.

    This is the main interface function that sets up parameters, initial conditions,
    and runs the integration to produce inspiral waveform data.

    Args:
        deltaT: Time step (seconds)
        m1_SI, m2_SI: Component masses (kg)
        fStart, fEnd: Start and end GW frequencies (Hz)
        s1x, s1y, s1z: Dimensionless spin vector for mass 1
        s2x, s2y, s2z: Dimensionless spin vector for mass 2
        lnhatx, lnhaty, lnhatz: Initial orbital angular momentum direction
        e1x, e1y, e1z: Reference direction vector
        lambda1, lambda2: Tidal deformability parameters
        quadparam1, quadparam2: Quadrupole moment parameters
        spinO, tideO, phaseO: PN orders for spin, tidal, phase corrections
        lscorr: Leading singularity corrections flag
        phenomtp: Phenomenological transition flag
        max_steps: Maximum integration steps

    Returns:
        Tuple of REAL8TimeSeries containing evolved quantities:
        (V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, E1x, E1y, E1z)
    """
    # Setup evolution parameters
    params = XLALSimInspiralSpinTaylorT4Setup(
        m1_SI,
        m2_SI,
        fStart,
        fEnd,
        lambda1,
        lambda2,
        quadparam1,
        quadparam2,
        spinO,
        tideO,
        phaseO,
        lscorr,
        phenomtp,
    )

    # Convert to dictionary for JIT compatibility
    params_dict = {
        "wdotnewt": params.wdotnewt,
        "eta": params.eta,
        "Msec": params.Msec,
        "fStart": params.fStart,
        "fEnd": params.fEnd,
        "omegashiftS1": params.omegashiftS1,
        "omegashiftS2": params.omegashiftS2,
        "spinO": params.spinO,
        "lscorr": params.lscorr,
        "phenomtp": params.phenomtp,
        "wdot_coeffs": params.wdot_coeffs,
        "spin_coeffs": params.spin_coeffs,
        "energy_coeffs": params.energy_coeffs,
    }

    # Set up initial conditions
    yinit = jnp.array(
        [
            0.0,  # phi: initial phase = 0
            jnp.pi * params.Msec * fStart,  # omega: initial frequency
            lnhatx,
            lnhaty,
            lnhatz,  # LNhat: orbital angular momentum direction
            params.norm1 * s1x,  # S1 (normalized)
            params.norm1 * s1y,
            params.norm1 * s1z,
            params.norm2 * s2x,  # S2 (normalized)
            params.norm2 * s2y,
            params.norm2 * s2z,
            e1x,
            e1y,
            e1z,  # E1: reference direction
        ]
    )

    # Set up integration parameters
    sgn = jnp.where((fEnd < fStart) & (fEnd != 0.0), -1, 1)

    # Estimate integration time using Newtonian formula
    dtStart = (
        (5.0 / 256.0)
        * jnp.power(jnp.pi, -8.0 / 3.0)
        * jnp.power(params.Mcsec * fStart, -5.0 / 3.0)
        / fStart
    )
    dtEnd = jnp.where(
        fEnd == 0.0,
        0.0,
        (5.0 / 256.0)
        * jnp.power(jnp.pi, -8.0 / 3.0)
        * jnp.power(params.Mcsec * fEnd, -5.0 / 3.0)
        / fEnd,
    )

    lengths = dtStart - dtEnd
    t0 = 0.0
    t1 = lengths / params.Msec
    dt0 = sgn * deltaT / params.Msec

    # Create time points for solution
    sgnt1 = sgn * t1
    save_ts = jnp.arange(t0, sgnt1, dt0)

    # Run integration with adaptive stepping
    term = ODETerm(XLALSimInspiralSpinTaylorT4DerivativesAvg)
    solver = Tsit5()
    saveat = SaveAt(ts=save_ts)
    stepsize_controller = PIDController(
        rtol=LAL_ST4_RELATIVE_TOLERANCE, atol=LAL_ST4_ABSOLUTE_TOLERANCE
    )

    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=sgn * t1,
        dt0=dt0,
        y0=yinit,
        args=params_dict,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        event=Event(cond_fn=stopping_event),
    )

    yout = sol.ys

    # Filter out invalid solutions
    valid_mask = jnp.all(jnp.isfinite(yout), axis=1)
    yout = yout[valid_mask]
    len_result = yout.shape[0]

    # Handle frequency cutoff
    if fEnd != 0.0:
        wEnd = jnp.pi * params.Msec * fEnd
        omega_series = yout[:, 1]

        if fEnd < fStart:
            crosses = omega_series < wEnd
        else:
            crosses = omega_series > wEnd

        first_cross = jnp.argmax(crosses)
        has_crossing = jnp.any(crosses)
        cutlen = jnp.where(has_crossing, first_cross + 1, len_result)
        yout = yout[:cutlen]

    # Reverse if backward integration
    if (fEnd < fStart) and (fEnd != 0.0):
        yout = jnp.flip(yout, axis=0)

    # Extract time series
    phi_data = yout[:, 0]
    omega_data = yout[:, 1]
    v_data = jnp.cbrt(omega_data)

    lnhatx_data = yout[:, 2]
    lnhaty_data = yout[:, 3]
    lnhatz_data = yout[:, 4]

    # Denormalize spins
    s1x_data = yout[:, 5] / params.norm1
    s1y_data = yout[:, 6] / params.norm1
    s1z_data = yout[:, 7] / params.norm1

    s2x_data = yout[:, 8] / params.norm2
    s2y_data = yout[:, 9] / params.norm2
    s2z_data = yout[:, 10] / params.norm2

    e1x_data = yout[:, 11]
    e1y_data = yout[:, 12]
    e1z_data = yout[:, 13]

    # Create output time series
    return (
        REAL8TimeSeries(data=v_data, deltaT=deltaT),
        REAL8TimeSeries(data=phi_data, deltaT=deltaT),
        REAL8TimeSeries(data=s1x_data, deltaT=deltaT),
        REAL8TimeSeries(data=s1y_data, deltaT=deltaT),
        REAL8TimeSeries(data=s1z_data, deltaT=deltaT),
        REAL8TimeSeries(data=s2x_data, deltaT=deltaT),
        REAL8TimeSeries(data=s2y_data, deltaT=deltaT),
        REAL8TimeSeries(data=s2z_data, deltaT=deltaT),
        REAL8TimeSeries(data=lnhatx_data, deltaT=deltaT),
        REAL8TimeSeries(data=lnhaty_data, deltaT=deltaT),
        REAL8TimeSeries(data=lnhatz_data, deltaT=deltaT),
        REAL8TimeSeries(data=e1x_data, deltaT=deltaT),
        REAL8TimeSeries(data=e1y_data, deltaT=deltaT),
        REAL8TimeSeries(data=e1z_data, deltaT=deltaT),
    )


# ============================================================================
# CONVENIENCE FUNCTIONS AND EXAMPLES
# ============================================================================


def example_binary_evolution(
    m1_msun: float = 36.0,
    m2_msun: float = 29.0,
    chi1z: float = 0.3,
    chi2z: float = 0.2,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    delta_t: float = 0.01,
) -> Tuple[REAL8TimeSeries, ...]:
    """
    Create an example binary black hole evolution.

    Args:
        m1_msun, m2_msun: Component masses in solar masses
        chi1z, chi2z: Dimensionless z-component spins
        f_low, f_high: Start and end frequencies (Hz)
        delta_t: Time step (seconds)

    Returns:
        Tuple of evolved time series
    """
    # Convert masses
    m1_SI = m1_msun * LAL_MSUN_SI
    m2_SI = m2_msun * LAL_MSUN_SI

    # Set up initial conditions (aligned spins, face-on orbit)
    s1x = s1y = s2x = s2y = 0.0
    s1z, s2z = chi1z, chi2z

    lnhatx = lnhaty = 0.0
    lnhatz = 1.0
    e1x, e1y, e1z = 1.0, 0.0, 0.0

    # Physical parameters (black holes)
    lambda1 = lambda2 = 0.0
    quadparam1 = quadparam2 = 1.0

    # Use highest PN orders
    spinO = tideO = phaseO = -1
    lscorr = 0

    print(f"Evolving {m1_msun:.1f}+{m2_msun:.1f} M☉ binary")
    print(f"Spins: χ₁z={chi1z:.1f}, χ₂z={chi2z:.1f}")
    print(f"Frequency range: {f_low}-{f_high} Hz")

    return XLALSimInspiralSpinTaylorPNEvolveOrbit(
        deltaT=delta_t,
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        fStart=f_low,
        fEnd=f_high,
        s1x=s1x,
        s1y=s1y,
        s1z=s1z,
        s2x=s2x,
        s2y=s2y,
        s2z=s2z,
        lnhatx=lnhatx,
        lnhaty=lnhaty,
        lnhatz=lnhatz,
        e1x=e1x,
        e1y=e1y,
        e1z=e1z,
        lambda1=lambda1,
        lambda2=lambda2,
        quadparam1=quadparam1,
        quadparam2=quadparam2,
        spinO=spinO,
        tideO=tideO,
        phaseO=phaseO,
        lscorr=lscorr,
    )


def run_example():
    """Run example calculation and print summary."""
    result = example_binary_evolution()
    V, Phi = result[:2]

    print(f"✓ Generated {len(V.data)} time samples")
    print(f"✓ Evolution time: {len(V.data) * V.deltaT:.2f} seconds")
    print(f"✓ Final PN parameter: v = {V.data[-1]:.4f}")
    print(f"✓ Total orbital cycles: {Phi.data[-1] / (2 * jnp.pi):.1f}")


if __name__ == "__main__":
    run_example()
