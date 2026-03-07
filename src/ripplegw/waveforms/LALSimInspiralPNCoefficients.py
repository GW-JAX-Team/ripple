import jax.numpy as jnp
from jax import jit

# Constants (from LAL)
LAL_PI = jnp.pi
LAL_MSUN_SI = 1.98892e30
LAL_MTSUN_SI = 4.925491025543576e-06
LAL_G_SI = 6.67430e-11
LAL_C_SI = 299792458.0
LAL_GAMMA = 0.5772156649015329

LAL_ST4_ABSOLUTE_TOLERANCE = 1.0e-11
LAL_ST4_RELATIVE_TOLERANCE = 1.0e-9
LAL_NUM_ST4_VARIABLES = 14


XLALSimInspiralSpinDot_4PNS2CoeffAvg = 0.5
XLALSimInspiralSpinDot_4PNS2OCoeffAvg = -1.5


@jit
def XLALSimInspiralTaylorT4wdot_7PNCoeff(eta):
    return (LAL_PI / 12096.0) * (-13245.0 + 717350.0 * eta + 731960.0 * eta * eta)


@jit
def XLALSimInspiralTaylorT4wdot_6PNCoeff(eta):
    return (
        16447.322263 / 139.7088
        - 1712.0 / 105.0 * LAL_GAMMA
        - 561.98689 / 2.17728 * eta
        + LAL_PI * LAL_PI * (16.0 / 3.0 + 451.0 / 48.0 * eta)
        + 541.0 / 896.0 * eta * eta
        - 5605.0 / 2592.0 * eta * eta * eta
        - 856.0 / 105.0 * jnp.log(16.0)
    )


@jit
def XLALSimInspiralTaylorT4wdot_6PNLogCoeff(eta):
    return -(1712.0 / 105.0)


@jit
def XLALSimInspiralTaylorT4wdot_5PNCoeff(eta):
    return -(1.0 / 672.0) * LAL_PI * (4159.0 + 15876.0 * eta)


@jit
def XLALSimInspiralTaylorT4wdot_4PNCoeff(eta):
    return (34103.0 + 122949.0 * eta + 59472.0 * eta * eta) / 18144.0


@jit
def XLALSimInspiralTaylorT4wdot_3PNCoeff(eta):
    return 4.0 * LAL_PI


@jit
def XLALSimInspiralTaylorT4wdot_2PNCoeff(eta):
    return -(1.0 / 336.0) * (743.0 + 924.0 * eta)


@jit
def XLALSimInspiralTaylorT4wdot_0PNCoeff(eta):
    return 96.0 / 5.0 * eta


@jit
def XLALSimInspiralTaylorT4wdot_10PNTidalCoeff(mByM):

    return 6.0 * mByM * mByM * mByM * mByM * (12.0 - 11.0 * mByM)


@jit
def XLALSimInspiralTaylorT4wdot_12PNTidalCoeff(mByM):

    return (
        mByM
        * mByM
        * mByM
        * mByM
        * (
            4421.0 / 56.0
            - 12263.0 / 56.0 * mByM
            + 1893.0 / 4.0 * mByM * mByM
            - 661.0 / 2.0 * mByM * mByM * mByM
        )
    )


@jit
def XLALSimInspiralLDot_3PNSOCoeff(mByM):
    return 0.5 + 1.5 / mByM


@jit
def XLALSimInspiralSpinDot_6PNS2CoeffAvg(mByM):
    return XLALSimInspiralSpinDot_6PNS2Coeff(mByM) + 0.5 * (
        XLALSimInspiralSpinDot_6PNS2nCoeff(mByM)
        + XLALSimInspiralSpinDot_6PNS2vCoeff(mByM)
    )


@jit
def XLALSimInspiralSpinDot_6PNS2Coeff(mByM):

    return -1.5 - mByM


@jit
def XLALSimInspiralSpinDot_6PNS2nCoeff(mByM):
    return 1.5 + 2.0 * mByM + mByM * mByM


@jit
def XLALSimInspiralSpinDot_6PNS2vCoeff(mByM):
    return 1.5 + mByM


@jit
def XLALSimInspiralSpinDot_6PNS2OCoeffAvg(mByM):
    return -0.5 * (
        XLALSimInspiralSpinDot_6PNS2nCoeff(mByM)
        + XLALSimInspiralSpinDot_6PNS2vCoeff(mByM)
    )


@jit
def XLALSimInspiralSpinDot_6PNS1OCoeffAvg(mByM):
    return -0.5 * (
        XLALSimInspiralSpinDot_6PNS1nCoeff(mByM)
        + XLALSimInspiralSpinDot_6PNS1vCoeff(mByM)
    )


@jit
def XLALSimInspiralSpinDot_6PNS1nCoeff(mByM):
    return 3.5 - 3.0 / mByM - 0.5 * mByM * mByM


@jit
def XLALSimInspiralSpinDot_6PNS1vCoeff(mByM):
    return 3.0 - 1.5 * mByM - 1.5 / mByM


@jit
def XLALSimInspiralSpinDot_6PNQMSOCoeffAvg(mByM):
    return -0.5 * (
        XLALSimInspiralSpinDot_6PNQMSnCoeff(mByM)
        + XLALSimInspiralSpinDot_6PNQMSvCoeff(mByM)
    )


@jit
def XLALSimInspiralSpinDot_6PNQMSnCoeff(mByM):
    return 3.0 * (0.5 / mByM + 1.0 - mByM - 0.5 * mByM * mByM)


@jit
def XLALSimInspiralSpinDot_6PNQMSvCoeff(mByM):
    return 3.0 * (1.0 / mByM - 1.0)


@jit
def XLALSimInspiralSpinDot_4PNQMSOCoeffAvg(mByM):
    return 1.5 * (1.0 - 1.0 / mByM)


@jit
def XLALSimInspiralSpinDot_5PNCoeff(mByM):
    return (
        9.0 / 8.0
        - mByM / 2.0
        + 7.0 * mByM * mByM / 12.0
        - 7.0 * mByM * mByM * mByM / 6.0
        - mByM * mByM * mByM * mByM / 24.0
    )


@jit
def XLALSimInspiralL_3PNSicoeffAvg(mByM):
    return -0.75 - 0.25 / mByM


@jit
def XLALSimInspiralL_3PNSiLcoeffAvg(mByM):
    return -(1.0 / 3.0 + 9.0 / mByM) / 4.0


@jit
def XLALSimInspiralSpinDot_3PNCoeff(mByM):
    return 3.0 / 2.0 - mByM - mByM * mByM / 2.0


@jit
def XLALSimInspiralPNEnergy_3PNSOCoeff(mByM):
    return 2.0 / 3.0 + 2.0 / mByM


@jit
def XLALSimInspiralPNEnergy_4PNS1S2CoeffAvg(eta):
    return 1.0 / eta


@jit
def XLALSimInspiralPNEnergy_4PNS1OS2OCoeffAvg(eta):
    return -3.0 / eta


@jit
def XLALSimInspiralPNEnergy_4PNQMS1S1CoeffAvg(mByM):
    return 0.5 / mByM / mByM


@jit
def XLALSimInspiralPNEnergy_4PNQMS1OS1OCoeffAvg(mByM):
    return -1.5 / mByM / mByM


@jit
def XLALSimInspiralPNEnergy_5PNSOCoeff(mByM):
    return 5.0 / 3.0 + 3.0 / mByM + 29.0 * mByM / 9.0 + mByM * mByM / 9.0


@jit
def XLALSimInspiralPNEnergy_6PNS1S2CoeffAvg(eta):
    return 2.0 / eta - 11.0 / 6.0


@jit
def XLALSimInspiralPNEnergy_6PNS1OS2OCoeffAvg(eta):
    return -11.0 / 3.0 / eta + 2.3 / 1.8


@jit
def XLALSimInspiralPNEnergy_6PNS1S1CoeffAvg(mByM):
    return -1.0 / (mByM * mByM) - 1.0 / 6.0 / mByM - 0.5


@jit
def XLALSimInspiralPNEnergy_6PNQMS1S1CoeffAvg(mByM):
    return 1.25 / mByM / mByM + 1.25 / mByM + 5.0 / 12.0


@jit
def XLALSimInspiralPNEnergy_6PNS1OS1OCoeffAvg(mByM):
    return 6.0 / (mByM * mByM) - 1.5 / mByM - 1.1 / 1.8


@jit
def XLALSimInspiralPNEnergy_6PNQMS1OS1OCoeffAvg(mByM):
    return -3.75 / mByM / mByM - 3.75 / mByM - 1.25


@jit
def XLALSimInspiralPNEnergy_7PNSOCoeff(mByM):
    return (
        -75.0 / 4.0
        + 27.0 / (4.0 * mByM)
        + 53.0 * mByM / 2.0
        + 67 * mByM * mByM / 6.0
        + 17.0 * mByM * mByM * mByM / 12.0
        - mByM * mByM * mByM * mByM / 12.0
    )


@jit
def XLALSimInspiralPNEnergy_10PNTidalCoeff(mByM):
    return -9.0 * mByM * mByM * mByM * mByM * (1.0 - mByM)


@jit
def XLALSimInspiralPNEnergy_12PNTidalCoeff(mByM):
    return 6.0 / (mByM * mByM) - 1.5 / mByM - 1.1 / 1.8


@jit
def XLALSimInspiralPNEnergy_2PNCoeff(eta):
    return -(0.75 + eta / 12.0)


@jit
def XLALSimInspiralPNEnergy_4PNCoeff(eta):
    return -(27.0 / 8.0 - 19.0 / 8.0 * eta + 1.0 / 24.0 * eta * eta)


@jit
def XLALSimInspiralPNEnergy_6PNCoeff(eta):
    return -(
        67.5 / 6.4
        - (344.45 / 5.76 - 20.5 / 9.6 * LAL_PI * LAL_PI) * eta
        + 15.5 / 9.6 * eta * eta
        + 3.5 / 518.4 * eta * eta * eta
    )


@jit
def compute_wdotspin(params, LNhdotS1, LNhdotS2, S1dotS2, S1sq, S2sq, spinO):
    """
    Compute spin corrections to domega based on spin order.

    Args:
        params: dictionary containing wdot spin coefficients
        LNhdotS1, LNhdotS2: LNhat · S1, LNhat · S2
        S1dotS2: S1 · S2
        S1sq, S2sq: |S1|², |S2|²
        spinO: spin PN order

    Returns:
        dict with wspin3, wspin4Avg, wspin5, wspin6Avg
    """

    # Initialize all to zero
    wspin3 = 0.0
    wspin4Avg = 0.0
    wspin5 = 0.0
    wspin6Avg = 0.0

    # Define spin order constants (matching LAL)
    LAL_SIM_INSPIRAL_SPIN_ORDER_ALL = -1
    LAL_SIM_INSPIRAL_SPIN_ORDER_35PN = 7
    LAL_SIM_INSPIRAL_SPIN_ORDER_3PN = 6
    LAL_SIM_INSPIRAL_SPIN_ORDER_25PN = 5
    LAL_SIM_INSPIRAL_SPIN_ORDER_2PN = 4
    LAL_SIM_INSPIRAL_SPIN_ORDER_15PN = 3
    LAL_SIM_INSPIRAL_SPIN_ORDER_1PN = 2
    LAL_SIM_INSPIRAL_SPIN_ORDER_05PN = 1
    LAL_SIM_INSPIRAL_SPIN_ORDER_0PN = 0

    # Check which orders to include (mimics switch-case with fallthrough)
    include_3PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_3PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_25PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_25PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_2PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_2PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )
    include_15PN = (spinO >= LAL_SIM_INSPIRAL_SPIN_ORDER_15PN) | (
        spinO == LAL_SIM_INSPIRAL_SPIN_ORDER_ALL
    )

    # 1.5PN spin-orbit (wspin3)
    wspin3 = jnp.where(
        include_15PN,
        params["wdot3S1O"] * LNhdotS1 + params["wdot3S2O"] * LNhdotS2,
        wspin3,
    )

    # 2PN spin-spin and quadrupole-monopole (wspin4Avg)
    wspin4_SO = (
        params["wdot4S1S2Avg"] * S1dotS2
        + params["wdot4S1OS2OAvg"] * LNhdotS1 * LNhdotS2
    )
    wspin4_QM = (
        (params["wdot4S1S1Avg"] + params["wdot4QMS1S1Avg"]) * S1sq
        + (params["wdot4S2S2Avg"] + params["wdot4QMS2S2Avg"]) * S2sq
        + (params["wdot4S1OS1OAvg"] + params["wdot4QMS1OS1OAvg"]) * LNhdotS1 * LNhdotS1
        + (params["wdot4S2OS2OAvg"] + params["wdot4QMS2OS2OAvg"]) * LNhdotS2 * LNhdotS2
    )

    wspin4Avg = jnp.where(include_2PN, wspin4_SO + wspin4_QM, wspin4Avg)

    # 2.5PN spin-orbit (wspin5)
    wspin5 = jnp.where(
        include_25PN,
        params["wdot5S1O"] * LNhdotS1 + params["wdot5S2O"] * LNhdotS2,
        wspin5,
    )

    # 3PN spin-spin (wspin6Avg)
    wspin6_val = (
        params["wdot6S1O"] * LNhdotS1
        + params["wdot6S2O"] * LNhdotS2
        + params["wdot6S1S2Avg"] * S1dotS2
        + params["wdot6S1OS2OAvg"] * LNhdotS1 * LNhdotS2
        + (params["wdot6S1S1Avg"] + params["wdot6QMS1S1Avg"]) * S1sq
        + (params["wdot6S2S2Avg"] + params["wdot6QMS2S2Avg"]) * S2sq
        + (params["wdot6S1OS1OAvg"] + params["wdot6QMS1OS1OAvg"]) * LNhdotS1 * LNhdotS1
        + (params["wdot6S2OS2OAvg"] + params["wdot6QMS2OS2OAvg"]) * LNhdotS2 * LNhdotS2
    )

    wspin6Avg = jnp.where(include_3PN, wspin6_val, wspin6Avg)

    return {
        "wspin3": wspin3,
        "wspin4Avg": wspin4Avg,
        "wspin5": wspin5,
        "wspin6Avg": wspin6Avg,
    }


# Then in your setup function, add these coefficients:
@jit
def add_wdot_spin_coefficients_to_setup(m1M, m2M, eta, quadparam1, quadparam2):
    """
    Add wdot spin coefficients to params dictionary.
    These should be added in XLALSimInspiralSpinTaylorT4Setup.

    Args:
        params_dict: existing params dictionary
        m1M: m1/M
        m2M: m2/M
        eta: symmetric mass ratio
        quadparam1, quadparam2: quadrupole parameters
    """

    # 1.5PN spin-orbit coefficients

    params_dict = {}
    params_dict["wdot3S1O"] = XLALSimInspiralTaylorT4wdot_3PNSOCoeff(m1M)
    params_dict["wdot3S2O"] = XLALSimInspiralTaylorT4wdot_3PNSOCoeff(m2M)

    # 2PN spin-spin coefficients
    params_dict["wdot4S1S2Avg"] = XLALSimInspiralTaylorT4wdot_4PNS1S2CoeffAvg(eta)
    params_dict["wdot4S1OS2OAvg"] = XLALSimInspiralTaylorT4wdot_4PNS1OS2OCoeffAvg(eta)

    params_dict["wdot4S1S1Avg"] = XLALSimInspiralTaylorT4wdot_4PNS1S1CoeffAvg(m1M)
    params_dict["wdot4S2S2Avg"] = XLALSimInspiralTaylorT4wdot_4PNS1S1CoeffAvg(
        m2M
    )  # Note: uses S1S1 function

    params_dict["wdot4QMS1S1Avg"] = (
        quadparam1 * XLALSimInspiralTaylorT4wdot_4PNQMS1S1CoeffAvg(m1M)
    )
    params_dict["wdot4QMS2S2Avg"] = (
        quadparam2 * XLALSimInspiralTaylorT4wdot_4PNQMS1S1CoeffAvg(m2M)
    )

    params_dict["wdot4S1OS1OAvg"] = XLALSimInspiralTaylorT4wdot_4PNS1OS1OCoeffAvg(m1M)
    params_dict["wdot4S2OS2OAvg"] = XLALSimInspiralTaylorT4wdot_4PNS1OS1OCoeffAvg(m2M)

    params_dict["wdot4QMS1OS1OAvg"] = (
        quadparam1 * XLALSimInspiralTaylorT4wdot_4PNQMS1OS1OCoeffAvg(m1M)
    )
    params_dict["wdot4QMS2OS2OAvg"] = (
        quadparam2 * XLALSimInspiralTaylorT4wdot_4PNQMS1OS1OCoeffAvg(m2M)
    )

    # 2.5PN spin-orbit coefficients
    params_dict["wdot5S1O"] = XLALSimInspiralTaylorT4wdot_5PNSOCoeff(m1M)
    params_dict["wdot5S2O"] = XLALSimInspiralTaylorT4wdot_5PNSOCoeff(m2M)

    # 3PN spin-spin coefficients
    params_dict["wdot6S1O"] = XLALSimInspiralTaylorT4wdot_6PNSOCoeff(m1M)
    params_dict["wdot6S2O"] = XLALSimInspiralTaylorT4wdot_6PNSOCoeff(m2M)

    params_dict["wdot6S1S2Avg"] = XLALSimInspiralTaylorT4wdot_6PNS1S2CoeffAvg(eta)
    params_dict["wdot6S1OS2OAvg"] = XLALSimInspiralTaylorT4wdot_6PNS1OS2OCoeffAvg(eta)

    params_dict["wdot6S1S1Avg"] = XLALSimInspiralTaylorT4wdot_6PNS1S1CoeffAvg(m1M)
    params_dict["wdot6S2S2Avg"] = XLALSimInspiralTaylorT4wdot_6PNS1S1CoeffAvg(m2M)

    params_dict["wdot6QMS1S1Avg"] = (
        quadparam1 * XLALSimInspiralTaylorT4wdot_6PNQMS1S1CoeffAvg(m1M)
    )
    params_dict["wdot6QMS2S2Avg"] = (
        quadparam2 * XLALSimInspiralTaylorT4wdot_6PNQMS1S1CoeffAvg(m2M)
    )

    params_dict["wdot6S1OS1OAvg"] = XLALSimInspiralTaylorT4wdot_6PNS1OS1OCoeffAvg(m1M)
    params_dict["wdot6S2OS2OAvg"] = XLALSimInspiralTaylorT4wdot_6PNS1OS1OCoeffAvg(m2M)

    params_dict["wdot6QMS1OS1OAvg"] = (
        quadparam1 * XLALSimInspiralTaylorT4wdot_6PNQMS1OS1OCoeffAvg(m1M)
    )
    params_dict["wdot6QMS2OS2OAvg"] = (
        quadparam2 * XLALSimInspiralTaylorT4wdot_6PNQMS1OS1OCoeffAvg(m2M)
    )

    return params_dict


@jit
def XLALSimInspiralTaylorT4wdot_6PNS1OS2OCoeffAvg(eta):
    return 162.25 / (2.24 * eta) - 129.31 / 2.88


@jit
def XLALSimInspiralTaylorT4wdot_4PNS1OS2OCoeffAvg(eta):
    return 721.0 / 48.0 / eta


@jit
def XLALSimInspiralTaylorT4wdot_3PNSOCoeff(mByM):
    return -19.0 / 6.0 - 25.0 / 4.0 / mByM


@jit
def XLALSimInspiralTaylorT4wdot_4PNS1S2CoeffAvg(eta):
    return -247.0 / 48.0 / eta


@jit
def XLALSimInspiralTaylorT4wdot_4PNS1S1CoeffAvg(mByM):
    return 7.0 / 96.0 / mByM / mByM


@jit
def XLALSimInspiralTaylorT4wdot_4PNQMS1S1CoeffAvg(mByM):
    return -2.5 / mByM / mByM


@jit
def XLALSimInspiralTaylorT4wdot_4PNS1OS1OCoeffAvg(mByM):
    return -1.0 / 96.0 / mByM / mByM


@jit
def XLALSimInspiralTaylorT4wdot_4PNQMS1OS1OCoeffAvg(mByM):
    return 7.5 / mByM / mByM


@jit
def XLALSimInspiralTaylorT4wdot_5PNSOCoeff(mByM):
    return (
        -809.0 / (84.0 * mByM)
        + 13.795 / 1.008
        - 527.0 * mByM / 24.0
        - 79.0 * mByM * mByM / 6.0
    )


@jit
def XLALSimInspiralTaylorT4wdot_6PNSOCoeff(mByM):
    return jnp.pi * (-37.0 / 3.0 - 151.0 / 6.0 / mByM)


@jit
def XLALSimInspiralTaylorT4wdot_6PNS1S2CoeffAvg(eta):
    return 108.79 / (6.72 * eta) + 75.25 / 2.88


@jit
def XLALSimInspiralTaylorT4wdot_6PNS1S1CoeffAvg(mByM):
    return 101.9 / (6.4 * mByM * mByM) + 2.51 / (5.76 * mByM) + 13.33 / 5.76


@jit
def XLALSimInspiralTaylorT4wdot_6PNQMS1S1CoeffAvg(mByM):
    return -6.59 / (2.24 * mByM * mByM) + 7.3 / (4.8 * mByM) - 43.0 / 4.0


@jit
def XLALSimInspiralTaylorT4wdot_6PNS1OS1OCoeffAvg(mByM):
    return -49.3 / (6.4 * mByM * mByM) + 197.47 / (5.76 * mByM) + 56.45 / 5.76


@jit
def XLALSimInspiralTaylorT4wdot_6PNQMS1OS1OCoeffAvg(mByM):
    return 19.77 / (2.24 * mByM * mByM) - 7.3 / (1.6 * mByM) + 129.0 / 4.0


@jit
def XLALSimInspiralL_2PN(eta):

    return 1.5 + eta / 6.0
