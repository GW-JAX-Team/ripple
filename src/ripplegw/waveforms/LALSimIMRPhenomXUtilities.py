import jax.numpy as jnp


def IMRPhenomXPsi4ToStrain(eta: float, S: float, dchi: float) -> float:
    """
    Convert from Psi4 to strain for IMRPhenomX waveforms.

    This function computes a time/phase correction factor that accounts for
    the conversion from the Weyl scalar Psi4 to gravitational wave strain h.
    The correction depends on the symmetric mass ratio, aligned spin, and
    spin difference.

    Args:
        eta: Symmetric mass ratio (m1*m2)/(m1+m2)^2
        S: Total aligned spin STotR (dimensionless)
        dchi: Spin difference parameter (chi1L - chi2L)

    Returns:
        float: Psi4 to strain conversion factor
    """
    # Compute powers of eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2

    # Compute powers of S
    S2 = S * S
    S3 = S2 * S
    S4 = S2 * S2

    # Non-spinning contribution
    noSpin = (
        13.39320482758057
        - 175.42481512989315 * eta
        + 2097.425116152503 * eta2
        - 9862.84178637907 * eta3
        + 16026.897939722587 * eta4
    )

    # Equal-spin contribution (terms proportional to S)
    eqSpin = (
        (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta2) * S
        + (1.3934428041390161 - 97.51812681228478 * eta + 376.9200932531847 * eta2) * S2
        + (15.649521097877374 + 137.33317057388916 * eta - 755.9566456906406 * eta2)
        * S3
        + (13.097315867845788 + 149.30405703643288 * eta - 764.5242164872267 * eta2)
        * S4
    )

    # Unequal-spin contribution (antisymmetric in spins)
    uneqSpin = 105.37711654943146 * dchi * jnp.sqrt(1.0 - 4.0 * eta) * eta2

    return noSpin + eqSpin + uneqSpin


def IMRPhenomX_StepFuncBool(t: float, t1: float) -> bool:
    """
    Step function returning boolean value.

    This is a Heaviside step function that returns True when t >= t1,
    and False otherwise. Used for conditional logic in waveform generation.

    Translated from LALSimIMRPhenomXUtilities.c

    Args:
        t: Input value (typically time or frequency)
        t1: Threshold value

    Returns:
        bool: True if t >= t1, False otherwise

    Notes:
        In JAX, for differentiable step functions, consider using
        jax.numpy.heaviside or smooth approximations. This boolean
        version is for direct conditional evaluation.
    """
    return t >= t1
