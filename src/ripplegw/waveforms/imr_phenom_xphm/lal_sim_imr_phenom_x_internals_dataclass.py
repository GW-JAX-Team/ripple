"""Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

from __future__ import annotations

import dataclasses

import jax
import jax.tree_util


def _register_dataclass(cls):
    """Register a dataclass with JAX tree utilities (version-agnostic)."""
    # Get all field names from the dataclass
    field_names = [f.name for f in dataclasses.fields(cls)]

    def flatten_fn(obj):
        values = tuple(getattr(obj, name) for name in field_names)
        return values, field_names

    def unflatten_fn(field_names, values):
        return cls(**dict(zip(field_names, values)))

    jax.tree_util.register_pytree_node(cls, flatten_fn, unflatten_fn)
    return cls


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXWaveformDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

    # Debug flag
    debug: int

    # Model version parameters
    imr_phenom_x_inspiral_phase_version: int
    imr_phenom_x_intermediate_phase_version: int
    imr_phenom_x_ringdown_phase_version: int

    # Implemented for PhenomPNR coprecessing model.
    # When used, this value will be determined by the corresponding value in prec_struct.
    imr_phenom_xpnr_use_tuned_coprec: int
    imr_phenom_xpnr_use_tuned_coprec_33: int

    # Toggle to return coprecessing model without any twisting up
    imr_phenom_x_return_co_prec: int

    # Toggle for only returning waveform (i.e. mode) phase
    phenom_x_only_return_phase: int

    # Parameters that define deviation of the tuned coprecessing mode PhenomXCP from PhenomX (500)
    mu1: float  # MR Amplitude
    mu2: float  # MR Amplitude: modifies gamma2
    mu3: float  # MR Amplitude: modifies gamma3
    mu4: float  # MR Amplitude: would modify appearance of fRing in MR amplitude
    nu0: float  # MR Phase
    nu4: float  # MR Phase
    nu5: float  # MR Phase
    nu6: float  # MR Phase
    zeta1: float  # INT Phase
    zeta2: float  # INT Phase
    pnr_dev_parameter: float  # Is zero when no precession, and non-zero otherwise
    pnr_window: float
    apply_pnr_deviations: int

    # Mass Parameters
    m1_si: float  # Mass in SI units
    m2_si: float  # Mass in SI units
    m_tot_si: float  # Total mass in SI units
    m1: float  # Mass in solar masses
    m2: float  # Mass in solar masses
    m_tot: float  # Total mass in solar masses
    mc: float  # Chirp mass in solar masses
    q: float  # Mass ratio >= 1
    eta: float  # Symmetric mass ratio
    delta: float  # PN symmetry parameter: sqrt(1-4*eta)

    # Spin Parameters
    chi1l: float
    chi2l: float
    chi_eff: float
    chi_pn_hat: float
    s_tot_r: float
    dchi: float
    dchi_half: float
    sl: float
    sigma_l: float
    chi_tot_perp: float
    chi_p: float
    theta_ls: float  # PNR Specific
    a1: float  # PNR Specific

    m: float
    m1_2: float
    m2_2: float

    # Matter parameters
    lambda1: float
    lambda2: float
    quad_param1: float
    quad_param2: float
    kappa2_t: float
    f_merger: float

    # Useful Powers (?)
    eta2: float
    eta3: float
    eta4: float
    chi1l2: float
    chi1l3: float
    chi2l2: float
    chi2l3: float
    chi1l2l: float  # chi1 * chi2

    # Amplitude and Phase Normalization
    dphase0: float
    amp0: float
    amp_norm: float

    # Frequencies
    f_meco: float
    f_isco: float

    # Ringdown value of precession angle beta
    beta_rd: float
    f_ring22_prec: float
    # Quantity needed to calculate effective RD frequencies (500)
    f_ring_eff_shift_divided_by_emm: float
    # Coprecessing frame ringdown frequencies for 22 Mode (500)
    f_ring_cp: float

    # Ringdown and Damping Frequencies for 22 Mode
    f_ring: float
    f_damp: float

    # Ringdown and Damping Frequencies for 21 Mode
    f_ring21: float
    f_damp21: float

    # Ringdown and Damping Frequencies for 33 Mode
    f_ring33: float
    f_damp33: float

    # Ringdown and Damping Frequencies for 32 Mode
    f_ring32: float
    f_damp32: float

    # Ringdown and Damping Frequencies for 44 Mode
    f_ring44: float
    f_damp44: float

    f_min: float
    f_max: float
    m_f_max: float
    f_max_prime: float
    delta_f: float
    delta_mf: float
    f_cut: float

    # Dimensionless frequency (Mf) defining the end of the waveform
    f_cut_def: float

    f_ref: float
    m_f_ref: float
    m_sec: float
    phi_ref_in: float
    phi0: float
    phi_f_ref: float
    pi_m: float
    v_ref: float

    # Final mass and spin
    e_rad: float
    a_final: float
    m_final: float

    # (500) Final mass and spin: It will at times be useful to use both separately.
    a_final_prec: float
    a_final_non_prec: float

    distance: float
    inclination: float
    beta: float

    lal_params: dict

    # PhenomXO4 variables
    pnr_single_spin: int

    # Frequency at which to force XAS/XHM phase and phase derivative value
    f_inspiral_align: float
    xas_dphase_at_f_inspiral_align: float
    xas_phase_at_f_inspiral_align: float
    # Strategy: values e.g. xhm_dphase_at_f_inspiral_align are updated within
    # loop over ell and emm within IMRPhenomXPHM_hplus_hcross. Within same loop,
    # e.g. IMRPhenomXHMGenerateFDOneMode is called to generate a coprecessing
    # moment. At that time, the values of e.g. xhm_dphase_at_f_inspiral_align
    # defined in IMRPhenomXHM_PNR_SetPhaseAlignmentParams.
    xhm_dphase_at_f_inspiral_align: float
    xhm_phase_at_f_inspiral_align: float

    imr_phenom_xpnr_force_xhm_alignment: int


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXUsefulPowersDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold useful powers for IMRPhenomX computations."""

    seven_sixths: float
    one_sixth: float
    ten_thirds: float
    eight_thirds: float
    seven_thirds: float
    five_thirds: float
    four_thirds: float
    two_thirds: float
    one_third: float
    five: float
    four: float
    three: float
    two: float
    sqrt: float
    itself: float
    m_sqrt: float
    m_one: float
    m_two: float
    m_three: float
    m_four: float
    m_five: float
    m_six: float
    m_one_third: float
    m_two_thirds: float
    m_four_thirds: float
    m_five_thirds: float
    m_seven_thirds: float
    m_eight_thirds: float
    m_ten_thirds: float
    m_one_sixth: float
    m_seven_sixths: float
    log: float
