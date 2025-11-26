"""Helper functions for IMRPhenomXPHM waveform model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.waveforms.imr_phenom_xphm.lal_constants import LAL_MSUN_SI
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_utilities import (
    imr_phenom_x_internal_nudge,
)
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral import xlal_sim_inspiral_set_quad_mon_params_from_lambdas
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_inspiral_waveform_flags import (
    xlal_sim_inspiral_mode_array_is_mode_active,
)
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


@checkify.checkify
def imr_phenom_x_initialize_powers(number: float | jnp.ndarray) -> IMRPhenomXUsefulPowersDataClass:
    """Initialize various powers of the input number.

    Args:
        number: Input number (float or JAX array).

    Returns:
        IMRPhenomXUsefulPowersDataClass containing computed power values.
    """
    # Ensure number is a JAX array for consistent operations
    number = jnp.asarray(number)

    # Sanity check
    checkify.check(jnp.all(number >= 0), "Error: number must be non-negative.")

    # Compute sixth root and its reciprocal
    sixth = jnp.power(number, 1.0 / 6.0)
    m_sixth = 1.0 / sixth

    # Build the powers dataclass
    return IMRPhenomXUsefulPowersDataClass(
        seven_sixths=sixth * number,
        one_sixth=sixth,
        ten_thirds=((((sixth * sixth) ** 2) ** 2) * number) * number,
        eight_thirds=((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth),
        seven_thirds=(((sixth * sixth) ** 2) ** 2) * number,
        five_thirds=(((sixth * sixth) ** 2) ** 2) * (sixth * sixth),
        four_thirds=((sixth * sixth) ** 2) ** 2,
        two_thirds=(sixth * sixth) ** 2,
        one_third=sixth * sixth,
        five=number * number * number * number * number,
        four=number * number * number * number,
        three=number * number * number,
        two=number * number,
        sqrt=sixth * sixth * sixth,  # Equivalent to sqrt(number)
        itself=number,
        m_sqrt=1.0 / (sixth * sixth * sixth),  # 1/sqrt(number)
        m_one=1.0 / number,
        m_two=1.0 / (number * number),
        m_three=1.0 / (number * number * number),
        m_four=1.0 / (number * number * number * number),
        m_five=1.0 / (number * number * number * number * number),
        m_six=m_sixth,
        m_one_third=1.0 / (sixth * sixth),
        m_two_thirds=1.0 / ((sixth * sixth) ** 2),
        m_four_thirds=1.0 / (((sixth * sixth) ** 2) ** 2),
        m_five_thirds=1.0 / ((((sixth * sixth) ** 2) ** 2) * (sixth * sixth)),
        m_seven_thirds=1.0 / ((((sixth * sixth) ** 2) ** 2) * number),
        m_eight_thirds=1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * (sixth * sixth)),
        m_ten_thirds=1.0 / (((((sixth * sixth) ** 2) ** 2) * number) * number),
        m_one_sixth=m_sixth,
        m_seven_sixths=m_sixth / number,
        log=jnp.log(number),
    )


@checkify.checkify
def _validate_inspiral_phase_version(ins_phase_version: jnp.ndarray) -> bool:
    """Validate that the inspiral phase version is allowed."""
    allowed = jnp.array([104, 105, 114, 115])
    is_valid = jnp.any(allowed == ins_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid inspiral phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_intermediate_phase_version(int_phase_version: jnp.ndarray) -> bool:
    """Validate that the intermediate phase version is allowed."""
    allowed = jnp.array([104, 105])
    is_valid = jnp.any(allowed == int_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid intermediate phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_ringdown_phase_version(rd_phase_version: jnp.ndarray) -> bool:
    """Validate that the ringdown phase version is allowed."""
    allowed = jnp.array([105])
    is_valid = jnp.any(allowed == rd_phase_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid ringdown phase version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_inspiral_amplitude_version(ins_amp_version: jnp.ndarray) -> bool:
    """Validate that the inspiral amplitude version is allowed."""
    allowed = jnp.array([103])
    is_valid = jnp.any(allowed == ins_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid inspiral amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_intermediate_amplitude_version(int_amp_version: jnp.ndarray) -> bool:
    """Validate that the intermediate amplitude version is allowed."""
    allowed = jnp.array([1043, 104, 105])
    is_valid = jnp.any(allowed == int_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid intermediate amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


@checkify.checkify
def _validate_ringdown_amplitude_version(rd_amp_version: jnp.ndarray) -> bool:
    """Validate that the ringdown amplitude version is allowed."""
    allowed = jnp.array([103])
    is_valid = jnp.any(allowed == rd_amp_version)

    # Use lax.cond to make the check traceable
    def invalid_branch(_):
        checkify.check(False, "Invalid ringdown amplitude version.")
        return False

    def valid_branch(_):
        return True

    result = jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)
    return result


# TODO: Not finished
def imr_phenom_x_set_waveform_variables(
    m1_si: float,
    m2_si: float,
    chi1l_in: float,
    chi2l_in: float,
    delta_f: float,
    f_ref: float,
    phi0: float,
    f_min: float,
    f_max: float,
    distance: float,
    inclination: float,
    lal_params: IMRPhenomXPHMParameterDataClass,
) -> IMRPhenomXWaveformDataClass:
    """Set up the IMRPhenomX waveform dataclass with given parameters.

    Args:
        m1_si: Mass 1 in SI units.
        m2_si: Mass 2 in SI units.
        chi1l_in: Aligned spin of mass 1.
        chi2l_in: Aligned spin of mass 2.
        delta_f: Frequency step size.
        f_ref: Reference frequency.
        phi0: Initial phase.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        distance: Distance to the source.
        inclination: Inclination angle.
        lal_params: Additional LAL parameters.

    Returns:
        IMRPhenomXWaveformDataClass with initialized waveform parameters.
    """
    # Validate the inspiral phase version
    _validate_inspiral_phase_version(lal_params.ins_phase_version)

    # Validate the intermediate phase version
    _validate_intermediate_phase_version(lal_params.int_phase_version)

    # Validate the ringdown phase version
    _validate_ringdown_phase_version(lal_params.rd_phase_version)

    # Validate the inspiral amplitude version
    _validate_inspiral_amplitude_version(lal_params.ins_amp_version)

    # Validate the intermediate amplitude version
    _validate_intermediate_amplitude_version(lal_params.int_amp_version)

    # Validate the ringdown amplitude version
    _validate_ringdown_amplitude_version(lal_params.rd_amp_version)

    # Rescale the mass in solar masses
    m1_in = m1_si / LAL_MSUN_SI
    m2_in = m2_si / LAL_MSUN_SI

    # Set matter parameters
    lambda1_in = 0.0
    lambda2_in = 0.0
    quad_param1_in = 1.0
    quad_param2_in = 1.0

    if lal_params.phen_x_tidal != 0:
        lambda1_in = lal_params.lambda1
        lambda2_in = lal_params.lambda2
        if lambda1_in < 0.0 or lambda2_in < 0.0:
            checkify.check(False, "Tidal deformabilities lambda1 and lambda2 must be non-negative.")
            # Set quadrupole-monopole parameters from tidal deformabilities
        lal_params = xlal_sim_inspiral_set_quad_mon_params_from_lambdas(lal_params)

        quad_param1_in = 1.0 + lal_params.d_quad_mon1
        quad_param2_in = 1.0 + lal_params.d_quad_mon2

    if m1_in >= m2_in:
        chi1l = chi1l_in
        chi2l = chi2l_in
        m1 = m1_in
        m2 = m2_in
        lambda1 = lambda1_in
        lambda2 = lambda2_in
        quad_param1 = quad_param1_in
        quad_param2 = quad_param2_in
    else:
        chi1l = chi2l_in
        chi2l = chi1l_in
        m1 = m2_in
        m2 = m1_in
        lambda1 = lambda2_in
        lambda2 = lambda1_in
        quad_param1 = quad_param2_in
        quad_param2 = quad_param1_in

    if chi1l > 1.0:
        chi1l = imr_phenom_x_internal_nudge(chi1l, 1.0, 1e-6)
    if chi2l > 1.0:
        chi2l = imr_phenom_x_internal_nudge(chi2l, 1.0, 1e-6)
    if chi1l < -1.0:
        chi1l = imr_phenom_x_internal_nudge(chi1l, -1.0, 1e-6)
    if chi2l < -1.0:
        chi2l = imr_phenom_x_internal_nudge(chi2l, -1.0, 1e-6)

    # If spins are still unphysical after checking for small round-off errors, fail.
    if chi1l > 1.0 or chi1l < -1.0 or chi2l > 1.0 or chi2l < -1.0:
        checkify.check(False, "Unphysical spins requested: must obey the Kerr bound [-1,1].")

    # Symmetric mass ratio
    delta = jnp.abs((m1 - m2) / (m1 + m2))
    eta = jnp.abs(0.25 * (1.0 - delta * delta))
    q = m1 / m2

    if eta > 0.25:
        eta = 0.25
    if eta > 0.25 or eta < 0.0:
        checkify.check(False, "Unphysical mass ratio %s requested.", eta)

    if eta == 0.25:
        q = 1.0

    # Check the mass ratio
    checkify.check(q > 1000.0, "The model is not supported for mass ratios > 1000.")

    m_tot = m1 + m2
    eta2 = eta * eta

    # waveform_dataclass = IMRPhenomXWaveformDataClass(
    #     imr_phenom_x_inspiral_phase_version=lal_params.ins_phase_version,
    #     imr_phenom_x_intermediate_phase_version=lal_params.int_phase_version,
    #     imr_phenom_x_ringdown_phase_version=lal_params.rd_phase_version,
    #     imr_phenom_x_inspiral_amp_version=lal_params.ins_amp_version,
    #     imr_phenom_x_intermediate_amp_version=lal_params.int_amp_version,
    #     imr_phenom_x_ringdown_amp_version=lal_params.rd_amp_version,
    #     imr_phenom_xpnr_use_tuned_coprec=lal_params.pnr_use_tuned_coprec,
    #     imr_phenom_xpnr_use_tuned_coprec33=lal_params.pnr_use_tuned_coprec33,
    #     phenom_x_only_return_phase=lal_params.phenom_x_only_return_phase,
    #     pnr_force_xhm_alignment=lal_params.pnr_force_xhm_alignment,
    #     m1_si=m1 * LAL_MSUN_SI,
    #     m2_si=m2 * LAL_MSUN_SI,
    #     q=q,
    #     eta=eta,
    #     m_tot_si=m1_si+m1_si,
    #     m_tot=m_tot,
    #     m1=m1/m_tot,
    #     m2=m2/m_tot,
    #     m_sec=m_tot*LAL_MTSUN_SI,
    #     delta=delta,
    #     eta2=eta2,
    #     eta3=eta*eta2,
    #     chi1l=chi1l,
    #     chi2l=chi2l,
    #     chi1l2l=chi1l*chi2l,
    #     chi1l2=chi1l*chi1l,
    #     chi1l3=chi1l*chi1l*chi1l,
    #     chi2l2=chi2l*chi2l,
    #     chi2l3=chi2l*chi2l*chi2l,
    #     chi_eff=xlal_sim_imr_phenom_x_chi_eff(eta, chi1l, chi2l),
    #     chi_pn_hat=xlal_sim_imr_phenom_x_chi_pn_hat(eta, chi1l, chi2l),
    # )


def _generate_valid_modes(max_l: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute valid (ell, emm) pairs statically.

    Args:
        max_l: Maximum ell value.

    Returns:
        Tuple of JAX arrays (ell_array, emm_array) containing valid modes.
    """
    ell_list = []
    emm_list = []
    for ell in range(2, max_l + 1):
        for emm in range(0, ell + 1):
            ell_list.append(ell)
            emm_list.append(emm)
    return jnp.array(ell_list, dtype=int), jnp.array(emm_list, dtype=int)


@checkify.checkify
def _check_input_mode_array_impl(
    mode_array: jnp.ndarray, ell_valid: jnp.ndarray, emm_valid: jnp.ndarray, max_l: int
) -> bool:
    """Internal implementation of mode array checking.

    Args:
        mode_array: JAX boolean array indicating active modes.
        ell_valid: JAX array of valid ell values to check.
        emm_valid: JAX array of valid emm values to check.
        max_l: Maximum ell value for bounds checking.

    Returns:
        True if all active modes are allowed, raises error otherwise.
    """
    # Allowed modes: (l, |m|) pairs
    allowed_l = jnp.array([2, 2, 3, 3, 4])
    allowed_m = jnp.array([2, 1, 3, 2, 4])

    # Check if each valid mode is active (positive or negative m)
    active_pos = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active_neg = jax.vmap(
        lambda ell, emm: xlal_sim_inspiral_mode_array_is_mode_active(modes=mode_array, ell=ell, m=-emm, max_l=max_l)
    )(ell_valid, emm_valid)
    active = active_pos | active_neg

    # For active modes, check if they are allowed
    def is_allowed(ell, emm):
        # Check if (ell, emm) is in allowed
        matches = (allowed_l == ell) & (allowed_m == emm)
        return jnp.any(matches)

    allowed_mask = jax.vmap(is_allowed)(ell_valid, emm_valid)

    # Invalid if active but not allowed
    invalid = active & ~allowed_mask

    # Check that no invalid modes are active
    checkify.check(
        ~jnp.any(invalid),
        "Invalid modes activated in mode_array. Only the modes (2,2), (2,1), (3,3), (3,2), and (4,4) are supported.",
    )

    # Always check a trivial condition to ensure err is properly initialized
    checkify.check(True, "")

    return True


def check_input_mode_array(lal_params: dict, max_l: int = 8) -> tuple[checkify.Error, bool]:
    """Check if mode_array in lal_params contains only allowed modes.

    Args:
        lal_params: Dictionary that may contain a 'mode_array' key.
        max_l: Maximum angular momentum number to check (must be concrete/static).

    Returns:
        Tuple of (error, result) from checkify.
    """
    # Precompute valid modes statically before any JAX operations
    ell_valid, emm_valid = _generate_valid_modes(max_l)

    # Check if mode_array is present
    has_mode_array = "mode_array" in lal_params

    def check_with_mode_array():
        """Check modes when mode_array is present."""
        mode_array = lal_params["mode_array"]
        return _check_input_mode_array_impl(mode_array, ell_valid, emm_valid, max_l)

    def check_without_mode_array():
        """Skip check when mode_array is not present."""
        return checkify.checkify(lambda: True)()

    # Use Python if since this is before JIT transforms the whole function
    # (check_input_mode_array itself is not transformed, only called by JIT)
    if has_mode_array:
        return check_with_mode_array()
    return check_without_mode_array()
