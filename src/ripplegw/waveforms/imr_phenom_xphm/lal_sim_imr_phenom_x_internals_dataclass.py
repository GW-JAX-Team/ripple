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


# @_register_dataclass
# @dataclasses.dataclass(frozen=True)
# class IMRPhenomXPrecessionDataClass:  # pylint: disable=too-many-instance-attributes
#     """Dataclass to hold precession parameters for IMRPhenomX waveform generation."""

#     # # Flag to define the version of IMRPhenomXP called
#     IMRPhenomXPrecVersion: int # Flag to set version of Euler angles used.
#     # Toggle to return coprecessing model without any twisting up
#     IMRPhenomXReturnCoPrec: int
#     # /* Parameters that define deviation of the tuned coprecessing mode PhenomXCP from PhenomX */
#     # MU2;   # MR Amplitude
#     # MU3;   # MR Amplitude
#     # MU4;   # MR Amplitude
#     # NU4;   # MR Phase
#     # NU5;   # MR Phase
#     # NU6;   # MR Phase
#     # ZETA2; # INT Phase

#     # Debug flag
#     debug_prec: int

#     # Mass and spin weightings
#     A1: float        # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
#     A2: float        # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
#     ASp1: float      # \f$A1 * S_{1 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
#     ASp2: float      # \f$A2 * S_{2 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
#     chi1_perp: float # \f$ \chi_{1 \perp} \f$
#     chi2_perp: float # \f$ \chi_{2 \perp} \f$
#     S1_perp: float   # \f$ S_{1 \perp} \f$
#     S2_perp: float   # \f$ S_{2 \perp} \f$
#     SL: float        # \f$ \chi_{1 L} m^1_2 + \chi_{2 L} m^2_2 \f$
#     Sperp: float     # \f$ \chi_{p} m^1_2 \f$
#     STot_perp: float # \f$ S_{1 \perp} + S_{2 \perp} \f$
#     chi_perp: float
#     chiTot_perp: float # \f$ S_{1 \perp} + S_{2 \perp} \f$

#     # Effective precessing spin parameter: Schmidt et al, Phys. Rev. D 91, 024043 (2015), arXiv:1408.1810
#     chi_p: float   # \f$ \chi_{p} = S_p / (A_1 \chi^2_1) \f$, Eq. 3.4 of Schmidt et al, arXiv:1408.1810

#     # Dimensionless aligned spin components on companion 1 and 2 respectively
#     chi1L: float   # \f$ \chi_{1L} = \chi_{1} \cdot \hat{L} \f$
#     chi2L: float   # \f$ \chi_{2L} = \chi_{2} \cdot \hat{L} \f$
#     # Angle between J0 and line of sight (z-direction)
#     thetaJN: float # Angle between J0 and line of sight (z-direction)

#     # The initial phase that we pass to the underlying aligned spin IMR waveform model
#     phi0_aligned: float # Initial phase to feed the underlying aligned-spin model

#     # Angle to rotate the polarization
#     zeta_polarization: float # Angle to rotate the polarizations

#     # Post-Newtonian Euler angles
#     alpha0: float # Coefficient of \f$\alpha\f$
#     alpha1: float # Coefficient of \f$\alpha\f$
#     alpha2: float # Coefficient of \f$\alpha\f$
#     alpha3: float # Coefficient of \f$\alpha\f$
#     alpha4L: float # Coefficient of \f$\alpha\f$
#     alpha5: float # Coefficient of \f$\alpha\f$
#     alpha6: float # Coefficient of \f$\alpha\f$
#     alphaNNLO: float # Post Newtonian \f$\alpha\f$ at NNLO.
#     alpha_offset: float # Offset for \f$\alpha\f$
#     epsilon0: float # Coefficient of \f$\epsilon \f$
#     epsilon1: float # Coefficient of \f$\epsilon \f$
#     epsilon2: float # Coefficient of \f$\epsilon \f$
#     epsilon3: float # Coefficient of \f$\epsilon \f$
#     epsilon4L: float # Coefficient of \f$\epsilon \f$
#     epsilon5: float # Coefficient of \f$\epsilon \f$
#     epsilon6: float # Coefficient of \f$\epsilon \f$
#     epsilonNNLO: float # Post Newtonian \f$\epsilon \f$ at NNLO.
#     epsilon_offset: float # Offset for \f$\epsilon \f$
#     # Alpha and epsilon offset for mprime !=2. alpha_offset corresponds to mprime=2
#     alpha_offset_1: float # \f$\alpha\f$ offset passed to \f$m = 1\f$ modes.
#     alpha_offset_3: float # \f$\alpha\f$ offset passed to \f$m = 3\f$ modes.
#     alpha_offset_4: float # \f$\alpha\f$ offset passed to \f$m = 4\f$ modes.
#     epsilon_offset_1: float # \f$\epsilon\f$ offset passed to \f$m = 1\f$ modes.
#     epsilon_offset_3: float # \f$\epsilon\f$ offset passed to \f$m = 3\f$ modes.
#     epsilon_offset_4: float # \f$\epsilon\f$ offset passed to \f$m = 4\f$ modes.

#     # Complex exponential of the Euler angles
#     cexp_i_alpha: complex # \f$e^{i \alpha}\f$
#     cexp_i_epsilon: complex # \f$e^{i \epsilon}\f$
#     cexp_i_betah: complex

#     # Multibanding applied to Euler angles
#     MBandPrecVersion: int # Flag to control multibanding for Euler angles.

#     # Source Frame Variables
#     J0x_Sf: float # \f$ J_{0,x}\f$ in L frame.
#     J0y_Sf: float # \f$ J_{0,y}\f$ in L frame.
#     J0z_Sf: float # \f$ J_{0,z}\f$ in L frame.
#     J0: float # \f$ J_{0}\f$ in L frame.
#     thetaJ_Sf: float # Angle between \f$J_0\f$ and \f$ L_{\rm{N}} \f$ (z-direction)
#     phiJ_Sf: float # Azimuthal angle of \f$J_0\f$ in the L frame
#     Nx_Sf: float # Line-of-sight vector component \f$ N_{x}\f$ in L frame.
#     Ny_Sf: float # Line-of-sight vector component \f$ N_{y}\f$ in L frame.
#     Nz_Sf: float # Line-of-sight vector component \f$ N_{z}\f$ in L frame.
#     Xx_Sf: float # Component of triad basis vector X in L frame.
#     Xy_Sf: float # Component of triad basis vector X in L frame.
#     Xz_Sf: float # Component of triad basis vector X in L frame.
#     kappa: float # Eq. C12 of arXiv:XXXX.YYYY

#     # J-frame variables
#     Nx_Jf: float # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
#     Ny_Jf: float # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
#     Nz_Jf: float # Line-of-sight vector component \f$ N_{x}\f$ in LJ frame.
#     PArunx_Jf: float # Component of triad basis vector P in J frame, arXiv:0810.5336.
#     PAruny_Jf: float # Component of triad basis vector P in J frame, arXiv:0810.5336.
#     PArunz_Jf: float # Component of triad basis vector P in J frame, arXiv:0810.5336.
#     QArunx_Jf: float # Component of triad basis vector Q in J frame, arXiv:0810.5336.
#     QAruny_Jf: float # Component of triad basis vector Q in J frame, arXiv:0810.5336.
#     QArunz_Jf: float # Component of triad basis vector Q in J frame, arXiv:0810.5336.
#     XdotPArun: float # \f$ X \cdot P \f$
#     XdotQArun: float # \f$ X \cdot Q \f$

#     # Orbital angular momentum
#     L0: float
#     L1: float
#     L2: float
#     L3: float
#     L4: float
#     L5: float
#     L6: float
#     L7: float
#     L8: float
#     L8L: float
#     LN: float
#     LOrb: float
#     LRef: float
#     LInit: float

#     # Reference frequencies
#     omega_ref: float
#     logomega_ref: float
#     omega_ref_cbrt: float
#     omega_ref_cbrt2: float

#     # Spin weighted spherical harmonics
#     # /* l = 2 */
#     Y2m2: complex 
#     Y2m1: complex 
#     Y20: complex 
#     Y21: complex 
#     Y22: complex 

#     # /* l = 3 */
#     Y3m3: complex
#     Y3m2: complex
#     Y3m1: complex
#     Y30: complex
#     Y31: complex
#     Y32: complex
#     Y3: complex


#     # /* l = 4 */
#     Y4m4: complex
#     Y4m3: complex
#     Y4m2: complex
#     Y4m1: complex
#     Y40: complex
#     Y41: complex
#     Y42: complex
#     Y43: complex
#     Y44: complex

#     # Useful sqare roots
#     sqrt2: float
#     sqrt5: float
#     sqrt6: float
#     sqrt7: float
#     sqrt10: float
#     sqrt14: float
#     sqrt15: float
#     sqrt70: float
#     sqrt30: float
#     sqrt2p5: float

#     # Variables for MSA precession angles of Chatziioannou et al, arXiv:1703.03967
#     # Lhat = {0,0,1}
#     Lhat_theta: float
#     Lhat_phi: float
#     Lhat_norm: float
#     Lhat_cos_theta: float

#     # Cartesian Dimensionful Spins
#     S1x: float # \f$ S_{1,x} \f$ in L frame
#     S1y: float # \f$ S_{1,y} \f$ in L frame
#     S1z: float # \f$ S_{1,z} \f$ in L frame
#     S2x: float # \f$ S_{2,x} \f$ in L frame
#     S2y: float # \f$ S_{2,y} \f$ in L frame
#     S2z: float # \f$ S_{2,z} \f$ in L frame

#     # Spherical Polar Dimensionful Spins
#     S1_norm: float # \f$ \left| S_{1} \right| \f$
#     S1_theta: float # Spherical polar component \f$ S_{1,\theta} \f$ in L frame
#     S1_phi: float # Spherical polar component \f$ S_{1,\phi} \f$ in L frame
#     S1_cos_theta: float # Spherical polar component \f$ \cos S_{1,\theta} \f$ in L frame

#     S2_norm: float # \f$ \left| S_{2} \right| \f$
#     S2_theta: float # Spherical polar component \f$ S_{2,\theta} \f$ in L frame
#     S2_phi: float # Spherical polar component \f$ S_{2,\phi} \f$ in L frame
#     S2_cos_theta: float # Spherical polar component \f$ \cos S_{2,\theta} \f$ in L frame

#     # Cartesian Dimensionless Spin Variables
#     chi1x: float # \f$ \chi_{1,x} \f$ in L frame
#     chi1y: float # \f$ \chi_{1,y} \f$ in L frame
#     chi1z: float # \f$ \chi_{1,z} \f$ in L frame

#     chi2x: float # \f$ \chi_{2,x} \f$ in L frame
#     chi2y: float # \f$ \chi_{2,y} \f$ in L frame
#     chi2z: float # \f$ \chi_{2,z} \f$ in L frame

#     # Spherical Polar Dimensionless Spins
#     chi1_theta: float # Spherical polar component \f$ \chi_{1,\theta} \f$ in L frame
#     chi1_phi: float # Spherical polar component \f$ \chi_{1,\phi} \f$ in L frame
#     chi1_norm: float # \f$ \left| \chi_{1} \right| \f$
#     chi1_cos_theta: float # Spherical polar component \f$ \cos \chi_{1,\theta} \f$ in L frame

#     chi2_theta: float # Spherical polar component \f$ \chi_{2,\theta} \f$ in L frame
#     chi2_phi: float # Spherical polar component \f$ \chi_{2,\phi} \f$ in L frame
#     chi2_norm: float # \f$ \left| \chi_{2} \right| \f$
#     chi2_cos_theta: float # Spherical polar component \f$ \cos \chi_{2,\theta} \f$ in L frame

#     ExpansionOrder: int # Flag to control expansion order of MSA system of equations.

#     twopiGM: float
#     piGM: float

#     L_Norm_N: float # Norm of Newtonian orbital angular momentum \f$ \left| L_N \right| \f$
#     L_Norm_3PN: float # Norm of orbital angular momentum at 3PN \f$ \left| L_{3 \rm{PN}} \right| \f$
#     J_Norm_N: float # Norm of total angular momentum using Newtonian orbital angular momentum \f$ \left| J_{N} \right| \f$
#     J_Norm_3PN: float # Norm of total angular momentum at 3PN \f$ \left| J_{3 \rm{PN}} \right| \f$

#     dotS1L: float # \f$ S_1 \cdot \hat{L} \f$
#     dotS1Ln: float # \f$ \hat{S}_1 \cdot \hat{L} \f$
#     dotS2L: float # \f$ S_2 \cdot \hat{L} \f$
#     dotS2Ln: float # \f$ \hat{S}_1 \cdot \hat{L} \f$
#     dotS1S2: float # \f$ S_1 \cdot S_2 \f$
#     Seff: float # \f$ S_{\rm{eff}} = (1 + q^{-1}) S_1 \cdot \hat{L} + (1 + q) S_2 \cdot \hat{L} \f$, Eq. 7 of arXiv:1703.03967. Note convention for q. */
#     Seff2: float # \f$ S_{\rm{eff}}^2 \f$ */

#     # vector S1_0; /**< Initial value for \f$ S_{1} \f$ */
#     # vector S2_0; /**< Initial value for \f$ S_{2} \f$ */
#     # vector L_0; /**< Initial value for \f$ L \f$ */
#     # vector Lhat_0; /**< Initial value for \f$ \hat{L} \f$ */
#     # vector S_0; /**< Initial value for \f$ S_1 + S_2 \f$ */
#     # vector J_0; /**< Initial value for \f$ J \f$ */

#     S_0_norm: float
#     S_0_norm_2: float
#     J_0_norm: float
#     J_0_norm_2: float
#     L_0_norm: float
#     L_0_norm_2: float

#     deltam_over_M: float # \f$ (m_1 - m_2) / (m_1 + m_2) \f$ */

#     # //phiz_0, phiz_1, phiz_2, phiz_3, phiz_4, phiz_5
#     phiz_0_coeff: float
#     phiz_1_coeff: float
#     phiz_2_coeff: float
#     phiz_3_coeff: float
#     phiz_4_coeff: float
#     phiz_5_coeff: float

#     # Omegaz_i coefficients in D10 - D15
#     Omegaz0_coeff: float
#     Omegaz1_coeff: float
#     Omegaz2_coeff: float
#     Omegaz3_coeff: float
#     Omegaz4_coeff: float
#     Omegaz5_coeff: float

#     # Omegaz terms in D16 - D21
#     Omegaz0: float 
#     Omegaz1: float
#     Omegaz2: float
#     Omegaz3: float
#     Omegaz4: float
#     Omegaz5: float

#     Omegazeta0_coeff: float
#     Omegazeta1_coeff: float
#     Omegazeta2_coeff: float
#     Omegazeta3_coeff: float
#     Omegazeta4_coeff: float
#     Omegazeta5_coeff: float

#     Omegazeta0: float
#     Omegazeta1: float
#     Omegazeta2: float
#     Omegazeta3: float
#     Omegazeta4: float
#     Omegazeta5: float

#     # MSA-SUA Euler Angles
#     phiz: float # Azimuthal angle of L around J
#     zeta: float # Angle to describe L w.r.t. J
#     cos_theta_L: float # Cosine of polar angle between L and J

#     # First order MSA corrections
#     zeta_0_MSA: float # First MSA correction \f$ \zeta_0 \f$, Eq. F19 of arXiv:1703.03967
#     phiz_0_MSA: float # First MSA correction \f$ \phi_{z,0} \f$, Eq. 67 of arXiv:1703.03967

#     # Initial values
#     zeta_0: float # Initial value of \f$ \zeta \f$
#     phiz_0: float # Initial value of \f$ \phi_{z,0} \f$

#     # Orbital velocity, v and v^2
#     v: float # Orbital velocity, \f$ v \f$
#     v2: float # Orbital velocity squared, \f$ v^2 \f$
#     # Reference orbital velocity, v and v^2
#     v_0: float # Orbital velocity at reference frequency, \f$ v_{\rm{ref}} \f$
#     v_0_2: float # Orbital velocity at reference frequency squared, \f$ v_{\rm{ref}}^2 \f$

#     Delta: float # Eq. C3 of arXiv:1703.03967

#     D2: float
#     D3: float

#     # Precession averaged total spin, Eq. 45
#     SAv: float # \f$ S_{\rm{av}} \f$ as defined in Eq. 45 of arXiv:1703.03967
#     SAv2: float # \f$ S_{\rm{av}}^2 \f$
#     invSAv2: float # \f$ 1 / S_{\rm{av}}^2 \f$
#     invSAv: float # \f$ 1 / S_{\rm{av}} \f$
#     # Eq. C1, C2 for Eq. 51
#     psi1: float # \f$ \psi_1 \f$ as defined in Eq. C1 of arXiv:1703.03967
#     psi2: float # \f$ \psi_2 \f$ as defined in Eq. C2 of arXiv:1703.03967

#     # Integration constant in Eq. 51
#     psi0: float # \f$ \psi_0 \f$ as per Eq. 51 of arXiv:1703.03967

#     # Eq. 51 and Eq. 24
#     psi: float # \f$ \psi \f$ as defined by Eq. 51 of arXiv:1703.03967
#     psi_dot: float # \f$ \dot{\psi} \f$ as per Eq. 50 of arXiv:1703.03967


#     Cphi: float # \f$ C_{\phi} \f$ as defined by Eq. B14 of arXiv:1703.03967
#     Dphi: float # \f$ C_{\phi} \f$ as defined by Eq. B15 of arXiv:1703.03967
#     #phiz_0_MSA_Cphi_term, phiz_0_MSA_Dphi_term;

#     # PN Coefficients in Appendix A of Chatziioannou et al, PRD, 88, 063011, (2013)
#     a0: float
#     a1: float
#     a2: float
#     a3: float
#     a4: float
#     a5: float
#     a6: float
#     a7: float
#     b6: float
#     a0_2: float
#     a0_3: float
#     a2_2: float
#     c0: float
#     c1: float
#     c2: float
#     c4: float
#     c12: float
#     c1_over_eta: float

#     # Eq. B9 - B11
#     d0: float
#     d2: float
#     d4: float

#     # Eq. A1 - A8
#     g0: float
#     g2: float
#     g3: float
#     g4: float
#     g5: float
#     g6: float
#     g7: float
#     g6L: float

#     # Spin-Orbit couplings
#     beta3: float
#     beta5: float
#     beta6: float
#     beta7: float
#     sigma4: float

#     # Total normed spin of primary and secondary BH squared
#     S1_norm_2: float
#     S2_norm_2: float

#     # Precession averaged spin couplings in A9 - A14
#     S1L_pav: float # Precession averaged coupling \f$ \langle S_1 \cdot \hat{L} \rangle_{\rm{pr}} \f$, Eq. A9 of arXiv:1703.03967
#     S2L_pav: float # Precession averaged coupling \f$ \langle S_2 \cdot \hat{L} \rangle_{\rm{pr}} \f$, Eq. A10 of arXiv:1703.03967
#     S1S2_pav: float # Precession averaged coupling \f$ \langle S_1 \cdot S_2 \rangle_{\rm{pr}} \f$, Eq. A11 of arXiv:1703.03967
#     S1Lsq_pav: float # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$, Eq. A12 of arXiv:1703.03967
#     S2Lsq_pav: float # Precession averaged coupling \f$ \langle (S_2 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$, Eq. A13 of arXiv:1703.03967
#     S1LS2L_pav: float # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L}) (S_2 \cdot \hat{L}) \rangle_{\rm{pr}} \f$, Eq. A14 of arXiv:1703.03967 |
#     # Total spin in Eq. 23 of Chatziioannou et al PRD, 95, 104004, (2017)
#     S_norm: float
#     S_norm_2: float

#     Spl2: float # Largest root of polynomial \f$ S^2_+ \f$, Eq. 22 of arXiv:1703.03967
#     Smi2: float # Smallest root of polynomial \f$ S^2_- \f$, Eq. 22 of arXiv:1703.03967
#     S32: float # Third root of polynomial \f$ S^2_3 \f$, Eq. 22 of arXiv:1703.03967
#     Spl: float # \f$ S_+ \f$
#     Smi: float # \f$ S_- \f$
#     S3: float # \f$ S_3 \f$
#     Spl2mSmi2: float # \f$ S^2_+ - S^2_- \f$
#     Spl2pSmi2: float # \f$ S^2_+ + S^2_- \f$ 

#     A_coeff: float
#     B_coeff: float
#     C_coeff: float
#     D_coeff: float
#     qq: float
#     invqq: float 
#     eta: float 
#     eta2: float
#     eta3: float
#     eta4: float
    
#     delta_qq: float
#     delta2_qq: float
#     delta3_qq: float
#     delta4_qq: float
#     inveta: float
#     inveta2: float
#     inveta3: float
#     inveta4: float
#     sqrt_inveta: float

#     SConst: float 

#     LPN_coefficients: Array

#     constants_L: Array

#     # // Variables to interpolate SpinTaylor angles, up to fmax_angles
#     # gsl_spline *alpha_spline;
#     # gsl_spline *cosbeta_spline;
#     # gsl_spline *gamma_spline;

#     # gsl_interp_accel *alpha_acc;
#     # gsl_interp_accel *cosbeta_acc;
#     # gsl_interp_accel *gamma_acc;

#     Mfmax_angles: float
#     alpha_ref: float # Record value of alpha at f_ref
#     gamma_ref: float # Record value of gamma at f_ref 
#     alpha_ftrans: float # Record value of alpha at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
#     cosbeta_ftrans: float # Record value of cosbeta at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
#     gamma_ftrans: float # Record value of gamma at end of inspiral, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
#     gamma_in: float # Record last value of gamma, used in IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
#     # PhenomXPalphaMRD *alpha_params; # Parameters needed for analytical MRD continuation of alpha 
#     # PhenomXPbetaMRD *beta_params; # Parameters needed for analytical MRD continuation of cosbeta 
#     ftrans_MRD: float
#     fmax_inspiral: float
#     precessing_tag: int
#     deltaf_interpolation: float
#     M_MIN: float
#     M_MAX: float
#     L_MAX_PNR: float
#     PNarrays: Array
#     integration_buffer: float # Buffer region for integration of SpinTaylor equations: added so that interpolated angles cover the frequency range requested by user
#     fmin_integration: float # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models  
#     Mfmin_integration: float # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models  

#     MSA_ERROR: float # Flag to track errors in initialization of MSA system. 

#     # /* PNR-specific additions for single-spin mapping */
#     chi_singleSpin: float # Magnitude of effective single spin used for tapering two-spin angles, Eq. 18 of arXiv:2107.08876
#     costheta_singleSpin: float # Polar angle of effective single spin, Eq. 19 or arXiv:2107.08876
#     costheta_final_singleSpin: float # Polar angle of approximate final spin, see technical document FIXME: add reference

#     chi_maxSpin: float
#     costheta_maxSpin: float

#     chi1x_evolved: float # x-component of spin on primary at end of SpinTaylor evolution
#     chi1y_evolved: float # y-component of spin on primary at end of SpinTaylor evolution
#     chi1z_evolved: float # z-component of spin on primary at end of SpinTaylor evolution
#     chi2x_evolved: float # x-component of spin on secondary at end of SpinTaylor evolution
#     chi2y_evolved: float # y-component of spin on secondary at end of SpinTaylor evolution
#     chi2z_evolved: float # z-component of spin on secondary at end of SpinTaylor evolution

#     chi_singleSpin_antisymmetric: float # magnitude of effective single spin of a two spin system for the antisymmetric waveform
#     theta_antisymmetric: float # Polar angle effective single spin for antisymmetric waveform

#     PNR_HM_Mflow: float # Mf_alpha_lower stored from alphaParams struct, 2 A4 / 7 from arXiv:2107.08876
#     PNR_HM_Mfhigh: float # Mf_beta_lower stored from betaParams struct, Eq. 58 from arXiv:2107.08876
#     PNR_q_window_lower: float # Boundary values for PNR angle transition window
#     PNR_q_window_upper: float
#     PNR_chi_window_lower: float
#     PNR_chi_window_upper: float
#     UPNRInspiralScaling: int # Enforce inpsiral scaling for HM angles outside of calibration window

#     # /* Store PNR-specific waveform flags for turning on and off tuning */
#     IMRPhenomXPNRUseTunedAngles: int
#     IMRPhenomXPNRUseTunedCoprec: int
#     IMRPhenomXPNRUseTunedCoprec33: int
#     IMRPhenomXPNRUseInputCoprecDeviations: int
#     IMRPhenomXPNRForceXHMAlignment: int
#     APPLY_PNR_DEVIATIONS: int

#     # /* A copy of the XAS 22 object */
#     pWF22AS: IMRPhenomXWaveformDataClass
#     IMRPhenomXPNRInterpTolerance: float

#     # /* Store anti-symmetric waveform flag for turning on and off */
#     IMRPhenomXAntisymmetricWaveform: int

#     # /* polarization symmetry property, refer to XXXX.YYYYY for details */
#     PolarizationSymmetry: float

#     # /* variables to store PNR angles for use in existing XP and XPHM twist-up functions */
#     alphaPNR: float
#     betaPNR: float
#     gammaPNR: float

#     # /* flag to use MR beta or analytic continuation with PNR angles */
#     UUseMRbeta: int

#     Mf_alpha_lower: float

#     # /* flag to toggle conditional precession multibanding */
#     UconditionalPrecMBand: int

#     LALparams: dict