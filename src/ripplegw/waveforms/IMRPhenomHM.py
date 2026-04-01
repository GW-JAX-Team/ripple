import jax
import jax.numpy as jnp
from ..constants import MSUN, MTSUN, MRSUN, MPC
from jaxtyping import Array
from .spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)


# Some pre-XPHM ripple code
from .IMRPhenomXPHM import (
    XLALSimIMRPhenomHMGethlmModes,
)


def gen_IMRPhenomHM(
    frequency_array,
    mass_1,
    mass_2,
    chi1,
    chi2,
    distance,  # in Mpc
    inclination,
    phi0,
    reference_frequency,
):
    """Generate IMRPhenomHM plus and cross polarizations."""

    m1_SI = mass_1 * MSUN
    m2_SI = mass_2 * MSUN
    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    extra_params = {
        "ModeArray": jnp.array(
            [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32
        )
    }

    hlm = XLALSimIMRPhenomHMGethlmModes(
        frequency_array,
        m1_SI,
        m2_SI,
        0,
        0,
        chi1,
        0,
        0,
        chi2,
        phi0,
        frequency_array[1] - frequency_array[0],
        reference_frequency,
        extra_params,
    )

    ells = extra_params["ModeArray"][:, 0]
    minus1l = jnp.where(ells % 2 != 0, -1, 1)
    mode_projections = jax.vmap(
        get_phenomHMFD_mode_projection,
        in_axes=(None, 0, 0, 0),
    )(
        inclination,
        minus1l,
        extra_params["ModeArray"][:, 0],
        extra_params["ModeArray"][:, 1],
    )

    # Reshape to (n_modes, 2, 1) and (n_modes, 1, f_sampling) so they broadcast to (n_modes, 2, f_sampling)
    projected = mode_projections[:, :, None] * hlm[:, None, :] * amp0
    hp, hc = jnp.sum(projected, axis=0)

    return hp, hc


def get_phenomHMFD_mode_projection(
    theta: float,
    minus1l: int | Array,
    ell: int | Array,
    m: int | Array,
) -> Array:
    """
    Helper function to compute mode-by-mode plus- and cross-polarisation prefactors
    """

    Y = jax.lax.switch(
        ell - 2,
        [
            lambda: compute_sminus2_l2(theta, m),
            lambda: compute_sminus2_l3(theta, m),
            lambda: compute_sminus2_l4(theta, m),
        ],
    )

    def sym_branch():
        # Equatorial symmetry: add in -m mode
        Ymstar = jax.lax.switch(
            ell - 2,
            [
                lambda: compute_sminus2_l2(theta, -m),
                lambda: compute_sminus2_l3(theta, -m),
                lambda: compute_sminus2_l4(theta, -m),
            ],
        )
        Ymstar = jnp.conj(Ymstar)
        factorp = 0.5 * (Y + minus1l * Ymstar)
        factorc = -1j * 0.5 * (Y - minus1l * Ymstar)
        return jnp.array([factorp, factorc])

    def asym_branch():  # NOTE This is for hypothetical m=0 modes, not currently implemented. Structure is there in case we ever want to use it
        # Not adding in the -m mode
        factorp = Y
        factorc = -1j * factorp
        return jnp.array([factorp, factorc])

    return jax.lax.select(
        m == 0,
        asym_branch(),
        sym_branch(),
    )
