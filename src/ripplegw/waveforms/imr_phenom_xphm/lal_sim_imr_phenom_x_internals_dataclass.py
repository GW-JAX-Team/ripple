"""Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


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

    # Debug flag
    debug: int = 0


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXPrecessionDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold precession parameters for IMRPhenomX waveform generation."""

    # # Flag to define the version of IMRPhenomXP called
    IMRPhenomXPrecVersion: int = 0  # Flag to set version of Euler angles used.
    # Toggle to return coprecessing model without any twisting up
    IMRPhenomXReturnCoPrec: int = 0
    # /* Parameters that define deviation of the tuned coprecessing mode PhenomXCP from PhenomX */
    # MU2;   # MR Amplitude
    # MU3;   # MR Amplitude
    # MU4;   # MR Amplitude
    # NU4;   # MR Phase
    # NU5;   # MR Phase
    # NU6;   # MR Phase
    # ZETA2; # INT Phase

    # Debug flag
    debug_prec: int = 0

    # Mass and spin weightings
    A1: float = 0.0  # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
    A2: float = 0.0  # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
    ASp1: float = 0.0  # \f$A1 * S_{1 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
    ASp2: float = 0.0  # \f$A2 * S_{2 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
    chi1_perp: float = 0.0  # \f$ \chi_{1 \perp} \f$
    chi2_perp: float = 0.0  # \f$ \chi_{2 \perp} \f$
    S1_perp: float = 0.0  # \f$ S_{1 \perp} \f$
    S2_perp: float = 0.0  # \f$ S_{2 \perp} \f$
    SL: float = 0.0  # \f$ \chi_{1 L} m^1_2 + \chi_{2 L} m^2_2 \f$
    Sperp: float = 0.0  # \f$ \chi_{p} m^1_2 \f$
    STot_perp: float = 0.0  # \f$ S_{1 \perp} + S_{2 \perp} \f$
    chi_perp: float = 0.0
    chiTot_perp: float = 0.0  # \f$ S_{1 \perp} + S_{2 \perp} \f$

    # Effective precessing spin parameter: Schmidt et al, Phys. Rev. D 91, 024043 (2015), arXiv:1408.1810
    chi_p: float = 0.0  # \f$ \chi_{p} = S_p / (A_1 \chi^2_1) \f$, Eq. 3.4 of Schmidt et al, arXiv:1408.1810

    # Dimensionless aligned spin components on companion 1 and 2 respectively
    chi1L: float = 0.0  # \f$ \chi_{1L} = \chi_{1} \cdot \hat{L} \f$
    chi2L: float = 0.0  # \f$ \chi_{2L} = \chi_{2} \cdot \hat{L} \f$
    # Angle between J0 and line of sight (z-direction)
    thetaJN: float = 0.0  # Angle between J0 and line of sight (z-direction)

    # The initial phase that we pass to the underlying aligned spin IMR waveform model
    phi0_aligned: float = 0.0  # Initial phase to feed the underlying aligned-spin model

    # Angle to rotate the polarization
    zeta_polarization: float = 0.0  # Angle to rotate the polarizations

    # Post-Newtonian Euler angles
    alpha0: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha1: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha2: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha3: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha4L: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha5: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha6: float = 0.0  # Coefficient of \f$\alpha\f$
    alphaNNLO: float = 0.0  # Post Newtonian \f$\alpha\f$ at NNLO.
    alpha_offset: float = 0.0  # Offset for \f$\alpha\f$
    epsilon0: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon1: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon2: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon3: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon4L: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon5: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon6: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilonNNLO: float = 0.0  # Post Newtonian \f$\epsilon \f$ at NNLO.
    epsilon_offset: float = 0.0  # Offset for \f$\epsilon \f$
    # Alpha and epsilon offset for mprime !=2. alpha_offset corresponds to mprime=2
    alpha_offset_1: float = 0.0  # \f$\alpha\f$ offset passed to \f$m = 1\f$ modes.
    alpha_offset_3: float = 0.0  # \f$\alpha\f$ offset passed to \f$m = 3\f$ modes.
    alpha_offset_4: float = 0.0  # \f$\alpha\f$ offset passed to \f$m = 4\f$ modes.
    epsilon_offset_1: float = 0.0  # \f$\epsilon\f$ offset passed to \f$m = 1\f$ modes.
    epsilon_offset_3: float = 0.0  # \f$\epsilon\f$ offset passed to \f$m = 3\f$ modes.
    epsilon_offset_4: float = 0.0  # \f$\epsilon\f$ offset passed to \f$m = 4\f$ modes.

    # Complex exponential of the Euler angles
    cexp_i_alpha: complex = 0j  # \f$e^{i \alpha}\f$
    cexp_i_epsilon: complex = 0j  # \f$e^{i \epsilon}\f$
    cexp_i_betah: complex = 0j

    # Multibanding applied to Euler angles
    MBandPrecVersion: int = 0  # Flag to control multibanding for Euler angles.

    # Source Frame Variables
    J0x_Sf: float = 0.0  # \f$ J_{0,x}\f$ in L frame.
    J0y_Sf: float = 0.0  # \f$ J_{0,y}\f$ in L frame.
    J0z_Sf: float = 0.0  # \f$ J_{0,z}\f$ in L frame.
    J0: float = 0.0  # \f$ J_{0}\f$ in L frame.
    thetaJ_Sf: float = 0.0  # Angle between \f$J_0\f$ and \f$ L_{\rm{N}} \f$ (z-direction)
    phiJ_Sf: float = 0.0  # Azimuthal angle of \f$J_0\f$ in the L frame
    Nx_Sf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in L frame.
    Ny_Sf: float = 0.0  # Line-of-sight vector component \f$ N_{y}\f$ in L frame.
    Nz_Sf: float = 0.0  # Line-of-sight vector component \f$ N_{z}\f$ in L frame.
    Xx_Sf: float = 0.0  # Component of triad basis vector X in L frame.
    Xy_Sf: float = 0.0  # Component of triad basis vector X in L frame.
    Xz_Sf: float = 0.0  # Component of triad basis vector X in L frame.
    kappa: float = 0.0  # Eq. C12 of arXiv:XXXX.YYYY

    # J-frame variables
    Nx_Jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
    Ny_Jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
    Nz_Jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in LJ frame.
    PArunx_Jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    PAruny_Jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    PArunz_Jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    QArunx_Jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    QAruny_Jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    QArunz_Jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    XdotPArun: float = 0.0  # \f$ X \cdot P \f$
    XdotQArun: float = 0.0  # \f$ X \cdot Q \f$

    # Orbital angular momentum
    L0: float = 0.0
    L1: float = 0.0
    L2: float = 0.0
    L3: float = 0.0
    L4: float = 0.0
    L5: float = 0.0
    L6: float = 0.0
    L7: float = 0.0
    L8: float = 0.0
    L8L: float = 0.0
    LN: float = 0.0
    LOrb: float = 0.0
    LRef: float = 0.0
    LInit: float = 0.0

    # Reference frequencies
    omega_ref: float = 0.0
    logomega_ref: float = 0.0
    omega_ref_cbrt: float = 0.0
    omega_ref_cbrt2: float = 0.0

    # Spin weighted spherical harmonics
    # /* l = 2 */
    Y2m2: complex = 0j
    Y2m1: complex = 0j
    Y20: complex = 0j
    Y21: complex = 0j
    Y22: complex = 0j

    # /* l = 3 */
    Y3m3: complex = 0j
    Y3m2: complex = 0j
    Y3m1: complex = 0j
    Y30: complex = 0j
    Y31: complex = 0j
    Y32: complex = 0j
    Y3: complex = 0j

    # /* l = 4 */
    Y4m4: complex = 0j
    Y4m3: complex = 0j
    Y4m2: complex = 0j
    Y4m1: complex = 0j
    Y40: complex = 0j
    Y41: complex = 0j
    Y42: complex = 0j
    Y43: complex = 0j
    Y44: complex = 0j

    # Useful sqare roots
    sqrt2: float = 0.0
    sqrt5: float = 0.0
    sqrt6: float = 0.0
    sqrt7: float = 0.0
    sqrt10: float = 0.0
    sqrt14: float = 0.0
    sqrt15: float = 0.0
    sqrt70: float = 0.0
    sqrt30: float = 0.0
    sqrt2p5: float = 0.0

    # Variables for MSA precession angles of Chatziioannou et al, arXiv:1703.03967
    # Lhat = {0,0,1}
    Lhat_theta: float = 0.0
    Lhat_phi: float = 0.0
    Lhat_norm: float = 0.0
    Lhat_cos_theta: float = 0.0

    # Cartesian Dimensionful Spins
    S1x: float = 0.0  # \f$ S_{1,x} \f$ in L frame
    S1y: float = 0.0  # \f$ S_{1,y} \f$ in L frame
    S1z: float = 0.0  # \f$ S_{1,z} \f$ in L frame
    S2x: float = 0.0  # \f$ S_{2,x} \f$ in L frame
    S2y: float = 0.0  # \f$ S_{2,y} \f$ in L frame
    S2z: float = 0.0  # \f$ S_{2,z} \f$ in L frame

    # Spherical Polar Dimensionful Spins
    S1_norm: float = 0.0  # \f$ \left| S_{1} \right| \f$
    S1_theta: float = 0.0  # Spherical polar component \f$ S_{1,\theta} \f$ in L frame
    S1_phi: float = 0.0  # Spherical polar component \f$ S_{1,\phi} \f$ in L frame
    S1_cos_theta: float = 0.0  # Spherical polar component \f$ \cos S_{1,\theta} \f$ in L frame

    S2_norm: float = 0.0  # \f$ \left| S_{2} \right| \f$
    S2_theta: float = 0.0  # Spherical polar component \f$ S_{2,\theta} \f$ in L frame
    S2_phi: float = 0.0  # Spherical polar component \f$ S_{2,\phi} \f$ in L frame
    S2_cos_theta: float = 0.0  # Spherical polar component \f$ \cos S_{2,\theta} \f$ in L frame

    # Cartesian Dimensionless Spin Variables
    chi1x: float = 0.0  # \f$ \chi_{1,x} \f$ in L frame
    chi1y: float = 0.0  # \f$ \chi_{1,y} \f$ in L frame
    chi1z: float = 0.0  # \f$ \chi_{1,z} \f$ in L frame

    chi2x: float = 0.0  # \f$ \chi_{2,x} \f$ in L frame
    chi2y: float = 0.0  # \f$ \chi_{2,y} \f$ in L frame
    chi2z: float = 0.0  # \f$ \chi_{2,z} \f$ in L frame

    # Spherical Polar Dimensionless Spins
    chi1_theta: float = 0.0  # Spherical polar component \f$ \chi_{1,\theta} \f$ in L frame
    chi1_phi: float = 0.0  # Spherical polar component \f$ \chi_{1,\phi} \f$ in L frame
    chi1_norm: float = 0.0  # \f$ \left| \chi_{1} \right| \f$
    chi1_cos_theta: float = 0.0  # Spherical polar component \f$ \cos \chi_{1,\theta} \f$ in L frame

    chi2_theta: float = 0.0  # Spherical polar component \f$ \chi_{2,\theta} \f$ in L frame
    chi2_phi: float = 0.0  # Spherical polar component \f$ \chi_{2,\phi} \f$ in L frame
    chi2_norm: float = 0.0  # \f$ \left| \chi_{2} \right| \f$
    chi2_cos_theta: float = 0.0  # Spherical polar component \f$ \cos \chi_{2,\theta} \f$ in L frame

    ExpansionOrder: int = 0  # Flag to control expansion order of MSA system of equations.

    twopiGM: float = 0.0
    piGM: float = 0.0

    L_Norm_N: float = 0.0  # Norm of Newtonian orbital angular momentum \f$ \left| L_N \right| \f$
    L_Norm_3PN: float = 0.0  # Norm of orbital angular momentum at 3PN \f$ \left| L_{3 \rm{PN}} \right| \f$
    J_Norm_N: float = (
        0.0  # Norm of total angular momentum using Newtonian orbital angular momentum \f$ \left| J_{N} \right| \f$
    )
    J_Norm_3PN: float = 0.0  # Norm of total angular momentum at 3PN \f$ \left| J_{3 \rm{PN}} \right| \f$

    dotS1L: float = 0.0  # \f$ S_1 \cdot \hat{L} \f$
    dotS1Ln: float = 0.0  # \f$ \hat{S}_1 \cdot \hat{L} \f$
    dotS2L: float = 0.0  # \f$ S_2 \cdot \hat{L} \f$
    dotS2Ln: float = 0.0  # \f$ \hat{S}_1 \cdot \hat{L} \f$
    dotS1S2: float = 0.0  # \f$ S_1 \cdot S_2 \f$
    Seff: float = (
        0.0  # \f$ S_{\rm{eff}} = (1 + q^{-1}) S_1 \cdot \hat{L} + (1 + q) S_2 \cdot \hat{L} \f$, Eq. 7 of arXiv:1703.03967. Note convention for q. */
    )
    Seff2: float = 0.0  # \f$ S_{\rm{eff}}^2 \f$ */

    # vector S1_0; /**< Initial value for \f$ S_{1} \f$ */
    # vector S2_0; /**< Initial value for \f$ S_{2} \f$ */
    # vector L_0; /**< Initial value for \f$ L \f$ */
    # vector Lhat_0; /**< Initial value for \f$ \hat{L} \f$ */
    # vector S_0; /**< Initial value for \f$ S_1 + S_2 \f$ */
    # vector J_0; /**< Initial value for \f$ J \f$ */

    S_0_norm: float = 0.0
    S_0_norm_2: float = 0.0
    J_0_norm: float = 0.0
    J_0_norm_2: float = 0.0
    L_0_norm: float = 0.0
    L_0_norm_2: float = 0.0

    deltam_over_M: float = 0.0  # \f$ (m_1 - m_2) / (m_1 + m_2) \f$ */

    # //phiz_0, phiz_1, phiz_2, phiz_3, phiz_4, phiz_5
    phiz_0_coeff: float = 0.0
    phiz_1_coeff: float = 0.0
    phiz_2_coeff: float = 0.0
    phiz_3_coeff: float = 0.0
    phiz_4_coeff: float = 0.0
    phiz_5_coeff: float = 0.0

    # Omegaz_i coefficients in D10 - D15
    Omegaz0_coeff: float = 0.0
    Omegaz1_coeff: float = 0.0
    Omegaz2_coeff: float = 0.0
    Omegaz3_coeff: float = 0.0
    Omegaz4_coeff: float = 0.0
    Omegaz5_coeff: float = 0.0

    # Omegaz terms in D16 - D21
    Omegaz0: float = 0.0
    Omegaz1: float = 0.0
    Omegaz2: float = 0.0
    Omegaz3: float = 0.0
    Omegaz4: float = 0.0
    Omegaz5: float = 0.0

    Omegazeta0_coeff: float = 0.0
    Omegazeta1_coeff: float = 0.0
    Omegazeta2_coeff: float = 0.0
    Omegazeta3_coeff: float = 0.0
    Omegazeta4_coeff: float = 0.0
    Omegazeta5_coeff: float = 0.0

    Omegazeta0: float = 0.0
    Omegazeta1: float = 0.0
    Omegazeta2: float = 0.0
    Omegazeta3: float = 0.0
    Omegazeta4: float = 0.0
    Omegazeta5: float = 0.0

    # MSA-SUA Euler Angles
    phiz: float = 0.0  # Azimuthal angle of L around J
    zeta: float = 0.0  # Angle to describe L w.r.t. J
    cos_theta_L: float = 0.0  # Cosine of polar angle between L and J

    # First order MSA corrections
    zeta_0_MSA: float = 0.0  # First MSA correction \f$ \zeta_0 \f$, Eq. F19 of arXiv:1703.03967
    phiz_0_MSA: float = 0.0  # First MSA correction \f$ \phi_{z,0} \f$, Eq. 67 of arXiv:1703.03967

    # Initial values
    zeta_0: float = 0.0  # Initial value of \f$ \zeta \f$
    phiz_0: float = 0.0  # Initial value of \f$ \phi_{z,0} \f$

    # Orbital velocity, v and v^2
    v: float = 0.0  # Orbital velocity, \f$ v \f$
    v2: float = 0.0  # Orbital velocity squared, \f$ v^2 \f$
    # Reference orbital velocity, v and v^2
    v_0: float = 0.0  # Orbital velocity at reference frequency, \f$ v_{\rm{ref}} \f$
    v_0_2: float = 0.0  # Orbital velocity at reference frequency squared, \f$ v_{\rm{ref}}^2 \f$

    Delta: float = 0.0  # Eq. C3 of arXiv:1703.03967

    D2: float = 0.0
    D3: float = 0.0

    # Precession averaged total spin, Eq. 45
    SAv: float = 0.0  # \f$ S_{\rm{av}} \f$ as defined in Eq. 45 of arXiv:1703.03967
    SAv2: float = 0.0  # \f$ S_{\rm{av}}^2 \f$
    invSAv2: float = 0.0  # \f$ 1 / S_{\rm{av}}^2 \f$
    invSAv: float = 0.0  # \f$ 1 / S_{\rm{av}} \f$
    # Eq. C1, C2 for Eq. 51
    psi1: float = 0.0  # \f$ \psi_1 \f$ as defined in Eq. C1 of arXiv:1703.03967
    psi2: float = 0.0  # \f$ \psi_2 \f$ as defined in Eq. C2 of arXiv:1703.03967

    # Integration constant in Eq. 51
    psi0: float = 0.0  # \f$ \psi_0 \f$ as per Eq. 51 of arXiv:1703.03967

    # Eq. 51 and Eq. 24
    psi: float = 0.0  # \f$ \psi \f$ as defined by Eq. 51 of arXiv:1703.03967
    psi_dot: float = 0.0  # \f$ \dot{\psi} \f$ as per Eq. 50 of arXiv:1703.03967

    Cphi: float = 0.0  # \f$ C_{\phi} \f$ as defined by Eq. B14 of arXiv:1703.03967
    Dphi: float = 0.0  # \f$ C_{\phi} \f$ as defined by Eq. B15 of arXiv:1703.03967
    # phiz_0_MSA_Cphi_term, phiz_0_MSA_Dphi_term;

    # PN Coefficients in Appendix A of Chatziioannou et al, PRD, 88, 063011, (2013)
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    a6: float = 0.0
    a7: float = 0.0
    b6: float = 0.0
    a0_2: float = 0.0
    a0_3: float = 0.0
    a2_2: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c4: float = 0.0
    c12: float = 0.0
    c1_over_eta: float = 0.0

    # Eq. B9 - B11
    d0: float = 0.0
    d2: float = 0.0
    d4: float = 0.0

    # Eq. A1 - A8
    g0: float = 0.0
    g2: float = 0.0
    g3: float = 0.0
    g4: float = 0.0
    g5: float = 0.0
    g6: float = 0.0
    g7: float = 0.0
    g6L: float = 0.0

    # Spin-Orbit couplings
    beta3: float = 0.0
    beta5: float = 0.0
    beta6: float = 0.0
    beta7: float = 0.0
    sigma4: float = 0.0

    # Total normed spin of primary and secondary BH squared
    S1_norm_2: float = 0.0
    S2_norm_2: float = 0.0

    # Precession averaged spin couplings in A9 - A14
    S1L_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle S_1 \cdot \hat{L} \rangle_{\rm{pr}} \f$, Eq. A9 of arXiv:1703.03967
    )
    S2L_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle S_2 \cdot \hat{L} \rangle_{\rm{pr}} \f$, Eq. A10 of arXiv:1703.03967
    )
    S1S2_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle S_1 \cdot S_2 \rangle_{\rm{pr}} \f$, Eq. A11 of arXiv:1703.03967
    )
    S1Lsq_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$, Eq. A12 of arXiv:1703.03967
    )
    S2Lsq_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle (S_2 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$, Eq. A13 of arXiv:1703.03967
    )
    S1LS2L_pav: float = (
        0.0  # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L}) (S_2 \cdot \hat{L}) \rangle_{\rm{pr}} \f$, Eq. A14 of arXiv:1703.03967 |
    )
    # Total spin in Eq. 23 of Chatziioannou et al PRD, 95, 104004, (2017)
    S_norm: float = 0.0
    S_norm_2: float = 0.0

    Spl2: float = 0.0  # Largest root of polynomial \f$ S^2_+ \f$, Eq. 22 of arXiv:1703.03967
    Smi2: float = 0.0  # Smallest root of polynomial \f$ S^2_- \f$, Eq. 22 of arXiv:1703.03967
    S32: float = 0.0  # Third root of polynomial \f$ S^2_3 \f$, Eq. 22 of arXiv:1703.03967
    Spl: float = 0.0  # \f$ S_+ \f$
    Smi: float = 0.0  # \f$ S_- \f$
    S3: float = 0.0  # \f$ S_3 \f$
    Spl2mSmi2: float = 0.0  # \f$ S^2_+ - S^2_- \f$
    Spl2pSmi2: float = 0.0  # \f$ S^2_+ + S^2_- \f$

    A_coeff: float = 0.0
    B_coeff: float = 0.0
    C_coeff: float = 0.0
    D_coeff: float = 0.0
    qq: float = 0.0
    invqq: float = 0.0
    eta: float = 0.0
    eta2: float = 0.0
    eta3: float = 0.0
    eta4: float = 0.0

    delta_qq: float = 0.0
    delta2_qq: float = 0.0
    delta3_qq: float = 0.0
    delta4_qq: float = 0.0
    inveta: float = 0.0
    inveta2: float = 0.0
    inveta3: float = 0.0
    inveta4: float = 0.0
    sqrt_inveta: float = 0.0

    SConst: float = 0.0

    LPN_coefficients: Array | None = None

    constants_L: Array | None = None

    # // Variables to interpolate SpinTaylor angles, up to fmax_angles
    # gsl_spline *alpha_spline;
    # gsl_spline *cosbeta_spline;
    # gsl_spline *gamma_spline;

    # gsl_interp_accel *alpha_acc;
    # gsl_interp_accel *cosbeta_acc;
    # gsl_interp_accel *gamma_acc;

    Mfmax_angles: float = 0.0
    alpha_ref: float = 0.0  # Record value of alpha at f_ref
    gamma_ref: float = 0.0  # Record value of gamma at f_ref
    alpha_ftrans: float = (
        0.0  # Record value of alpha at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
    )
    cosbeta_ftrans: float = (
        0.0  # Record value of cosbeta at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
    )
    gamma_ftrans: float = (
        0.0  # Record value of gamma at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
    )
    gamma_in: float = 0.0  # Record last value of gamma, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
    # PhenomXPalphaMRD *alpha_params; # Parameters needed for analytical MRD continuation of alpha
    # PhenomXPbetaMRD *beta_params; # Parameters needed for analytical MRD continuation of cosbeta
    ftrans_MRD: float = 0.0
    fmax_inspiral: float = 0.0
    precessing_tag: int = 0
    deltaf_interpolation: float = 0.0
    M_MIN: float = 0.0
    M_MAX: float = 0.0
    L_MAX_PNR: float = 0.0
    PNarrays: Array | None = None
    integration_buffer: float = (
        0.0  # Buffer region for integration of SpinTaylor equations: added so that interpolated angles cover the frequency range requested by user
    )
    fmin_integration: float = (
        0.0  # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models
    )
    Mfmin_integration: float = (
        0.0  # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models
    )

    MSA_ERROR: float = 0.0  # Flag to track errors in initialization of MSA system.

    # /* PNR-specific additions for single-spin mapping */
    chi_singleSpin: float = (
        0.0  # Magnitude of effective single spin used for tapering two-spin angles, Eq. 18 of arXiv:2107.08876
    )
    costheta_singleSpin: float = 0.0  # Polar angle of effective single spin, Eq. 19 or arXiv:2107.08876
    costheta_final_singleSpin: float = (
        0.0  # Polar angle of approximate final spin, see technical document FIXME: add reference
    )

    chi_maxSpin: float = 0.0
    costheta_maxSpin: float = 0.0

    chi1x_evolved: float = 0.0  # x-component of spin on primary at end of SpinTaylor evolution
    chi1y_evolved: float = 0.0  # y-component of spin on primary at end of SpinTaylor evolution
    chi1z_evolved: float = 0.0  # z-component of spin on primary at end of SpinTaylor evolution
    chi2x_evolved: float = 0.0  # x-component of spin on secondary at end of SpinTaylor evolution
    chi2y_evolved: float = 0.0  # y-component of spin on secondary at end of SpinTaylor evolution
    chi2z_evolved: float = 0.0  # z-component of spin on secondary at end of SpinTaylor evolution

    chi_singleSpin_antisymmetric: float = (
        0.0  # magnitude of effective single spin of a two spin system for the antisymmetric waveform
    )
    theta_antisymmetric: float = 0.0  # Polar angle effective single spin for antisymmetric waveform

    PNR_HM_Mflow: float = 0.0  # Mf_alpha_lower stored from alphaParams struct, 2 A4 / 7 from arXiv:2107.08876
    PNR_HM_Mfhigh: float = 0.0  # Mf_beta_lower stored from betaParams struct, Eq. 58 from arXiv:2107.08876
    PNR_q_window_lower: float = 0.0  # Boundary values for PNR angle transition window
    PNR_q_window_upper: float = 0.0
    PNR_chi_window_lower: float = 0.0
    PNR_chi_window_upper: float = 0.0
    UPNRInspiralScaling: int = 0  # Enforce inpsiral scaling for HM angles outside of calibration window

    # /* Store PNR-specific waveform flags for turning on and off tuning */
    IMRPhenomXPNRUseTunedAngles: int = 0
    IMRPhenomXPNRUseTunedCoprec: int = 0
    IMRPhenomXPNRUseTunedCoprec33: int = 0
    IMRPhenomXPNRUseInputCoprecDeviations: int = 0
    IMRPhenomXPNRForceXHMAlignment: int = 0
    APPLY_PNR_DEVIATIONS: int = 0

    # /* A copy of the XAS 22 object */
    pWF22AS: IMRPhenomXWaveformDataClass | None = None
    IMRPhenomXPNRInterpTolerance: float = 0.0

    # /* Store anti-symmetric waveform flag for turning on and off */
    IMRPhenomXAntisymmetricWaveform: int = 0

    # /* polarization symmetry property, refer to XXXX.YYYYY for details */
    PolarizationSymmetry: float = 0.0

    # /* variables to store PNR angles for use in existing XP and XPHM twist-up functions */
    alphaPNR: float = 0.0
    betaPNR: float = 0.0
    gammaPNR: float = 0.0

    # /* flag to use MR beta or analytic continuation with PNR angles */
    UUseMRbeta: int = 0

    Mf_alpha_lower: float = 0.0

    # /* flag to toggle conditional precession multibanding */
    UconditionalPrecMBand: int = 0

    LALparams: dict | None = None
