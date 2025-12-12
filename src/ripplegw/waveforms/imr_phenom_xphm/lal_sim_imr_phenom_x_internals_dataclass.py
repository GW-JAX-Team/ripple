"""Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass

# /* Inherited from IMRPhenomD */
N_MAX_COLLOCATION_POINTS_PHASE_RD = 5
N_MAX_COLLOCATION_POINTS_PHASE_INT = 5
N_MAX_COLLOCATION_POINTS_PHASE_INS = 6


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXWaveformDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

    # Model version parameters
    imr_phenom_x_inspiral_phase_version: int
    imr_phenom_x_intermediate_phase_version: int
    imr_phenom_x_ringdown_phase_version: int

    imr_phenom_x_inspiral_amp_version: int
    imr_phenom_x_intermediate_amp_version: int
    imr_phenom_x_ringdown_amp_version: int

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

    lal_params: IMRPhenomXPHMParameterDataClass

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

    seven_sixths: float | Array
    one_sixth: float | Array
    ten_thirds: float | Array
    eight_thirds: float | Array
    seven_thirds: float | Array
    five_thirds: float | Array
    four_thirds: float | Array
    two_thirds: float | Array
    one_third: float | Array
    five: float | Array
    four: float | Array
    three: float | Array
    two: float | Array
    sqrt: float | Array
    itself: float | Array
    m_sqrt: float | Array
    m_one: float | Array
    m_two: float | Array
    m_three: float | Array
    m_four: float | Array
    m_five: float | Array
    m_six: float | Array
    m_one_third: float | Array
    m_two_thirds: float | Array
    m_four_thirds: float | Array
    m_five_thirds: float | Array
    m_seven_thirds: float | Array
    m_eight_thirds: float | Array
    m_ten_thirds: float | Array
    m_one_sixth: float | Array
    m_seven_sixths: float | Array
    log: float | Array

    # Debug flag
    debug: int = 0


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXPhaseCoefficientsDataClass:  # pylint: disable=too
    """Dataclass to hold phase coefficients for IMRPhenomX computations."""

    # /* PHASE */
    # /* Phase Transition Frequencies */
    f_phase_ins_min: float = 0.0
    f_phase_ins_max: float = 0.0
    f_phase_int_min: float = 0.0
    f_phase_int_max: float = 0.0
    f_phase_rd_min: float = 0.0
    f_phase_rd_max: float = 0.0

    f_phase_match_in: float = 0.0
    f_phase_match_im: float = 0.0

    c1_int: float = 0.0
    c2_int: float = 0.0
    c1_m_rd: float = 0.0
    c2_m_rd: float = 0.0

    # /* These are the RD phenomenological coefficients 					*/
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c_l: float = 0.0
    c_rd: float = 0.0
    c_lgr: float = 0.0

    # /* These are the intermediate phenomenological coefficients */
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    b4: float = 0.0

    # /* These are the inspiral phenomenological coefficients 		*/
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    # /* Coefficients enterting tidal phase */
    c2_pn_tidal: float = 0.0
    c3_pn_tidal: float = 0.0
    c3p5_pn_tidal: float = 0.0

    # /* Pre-cached variables */
    c4ov3: float = 0.0
    c_lovfda: float = 0.0
    non_gr_dcl: float = 0.0

    # /* TaylorF2 PN Coefficients */
    phi_minus2: float = 0.0
    phi_minus1: float = 0.0
    phi0: float = 0.0
    phi1: float = 0.0
    phi2: float = 0.0
    phi3: float = 0.0
    phi4: float = 0.0
    phi5: float = 0.0
    phi6: float = 0.0
    phi7: float = 0.0
    phi8: float = 0.0
    phi9: float = 0.0
    phi10: float = 0.0
    phi11: float = 0.0
    phi12: float = 0.0
    phi13: float = 0.0
    phi5l: float = 0.0
    phi6l: float = 0.0
    phi8l: float = 0.0
    phi9l: float = 0.0
    phi_initial: float = 0.0
    phi_norm: float = 0.0
    dphi_minus2: float = 0.0
    dphi_minus1: float = 0.0
    dphi0: float = 0.0
    dphi1: float = 0.0
    dphi2: float = 0.0
    dphi3: float = 0.0
    dphi4: float = 0.0
    dphi5: float = 0.0
    dphi6: float = 0.0
    dphi7: float = 0.0
    dphi8: float = 0.0
    dphi9: float = 0.0
    dphi10: float = 0.0
    dphi11: float = 0.0
    dphi12: float = 0.0
    dphi13: float = 0.0
    dphi5l: float = 0.0
    dphi6l: float = 0.0
    dphi8l: float = 0.0
    dphi9l: float = 0.0

    # /* Pseudo-PN Coefficients */
    sigma0: float = 0.0
    sigma1: float = 0.0
    sigma2: float = 0.0
    sigma3: float = 0.0
    sigma4: float = 0.0
    sigma5: float = 0.0

    # /* Flag to set how many collocation points the RD region uses 	*/
    n_collocation_points_rd: int = 0

    # /* Flag to set how many collocation points the INT region uses 	*/
    n_collocation_points_int: int = 0

    # /* Integer to tell us how many pseudo PN terms are used */
    n_pseudo_pn: int = 0
    n_collocation_points_phase_ins: int = 0

    # /* The canonical ringdown phase is constructed from 5 collocation points 			*/
    collocation_points_phase_rd: Array | None = None
    collocation_values_phase_rd: Array | None = None
    coefficients_phase_rd: Array | None = None

    # /* The canonical intermediate phase is constructed from 4/5 collocation points  */
    collocation_points_phase_int: Array | None = None
    collocation_values_phase_int: Array | None = None
    coefficients_phase_int: Array | None = None

    # /*
    #         For N pseudo-PN terms we need N+1 collocation points:
    #         We have set N_MAX_COLLOCATION_POINTS_INS = 5 to allow
    #         either 3 or 4 pseudo-PN coefficients to be used.
    # */
    collocation_points_phase_ins: Array | None = None
    collocation_values_phase_ins: Array | None = None
    coefficients_phase_ins: Array | None = None
