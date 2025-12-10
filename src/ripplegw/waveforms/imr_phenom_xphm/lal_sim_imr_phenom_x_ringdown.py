"""LAL's IMRPhenomX_ringdown.c JAX implementation."""

from __future__ import annotations

from jax.experimental import checkify


@checkify.checkify
def imr_phenom_x_ringdown_phase_22_v4(eta: float, s: float, dchi: float, delta: float, rd_phase_flag: int) -> float:
    """IMRPhenomX_Ringdown_Phase_22_v4."""
    # /*
    #     Effective Spin Used: STotR.
    # */

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    checkify.check(
        rd_phase_flag == 105,
        "Error in IMRPhenomX_Ringdown_Phase_22_v4: "
        "IMRPhenomXRingdownPhaseVersion is not valid. Recommended flag is 105.",
    )

    # /* Canonical, 5 coefficients */
    no_spin = (
        -85.86062966719405
        - 4616.740713893726 * eta
        - 4925.756920247186 * eta2
        + 7732.064464348168 * eta3
        + 12828.269960300782 * eta4
        - 39783.51698102803 * eta5
    ) / (1.0 + 50.206318806624004 * eta)

    eq_spin = (
        s
        * (
            33.335857451144356
            - 36.49019206094966 * s
            + eta3 * (1497.3545918387515 - 101.72731770500685 * s) * s
            - 3.835967351280833 * s2
            + 2.302712009652155 * s3
            + eta2
            * (
                93.64156367505917
                - 18.184492163348665 * s
                + 423.48863373726243 * s2
                - 104.36120236420928 * s3
                - 719.8775484010988 * s4
            )
            + 1.6533417657003922 * s4
            + eta
            * (
                -69.19412903018717
                + 26.580344399838758 * s
                - 15.399770764623746 * s2
                + 31.231253209893488 * s3
                + 97.69027029734173 * s4
            )
            + eta4
            * (
                1075.8686153198323
                - 3443.0233614187396 * s
                - 4253.974688619423 * s2
                - 608.2901586790335 * s3
                + 5064.173605639933 * s4
            )
        )
    ) / (-1.3705601055555852 + 1.0 * s)

    uneq_spin = dchi * delta * eta * (22.363215261437862 + 156.08206945239374 * eta)

    return no_spin + eq_spin + uneq_spin
