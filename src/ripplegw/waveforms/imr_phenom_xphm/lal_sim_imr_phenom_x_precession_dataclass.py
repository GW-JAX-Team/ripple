"""Dataclass to hold internal parameters for IMRPhenomX waveform generation."""

from __future__ import annotations

import dataclasses

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.dataclass_utils import _register_dataclass
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import IMRPhenomXWaveformDataClass
from ripplegw.waveforms.imr_phenom_xphm.parameter_dataclass import IMRPhenomXPHMParameterDataClass


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class PhenomXPInspiralArrays:
    """Dataclass to hold inspiral arrays for IMRPhenomX waveform generation."""

    v_pn: Array
    """Velocity parameter array."""
    s1x_pn: Array
    """Spin component 1 x array."""
    s1y_pn: Array
    """Spin component 1 y array."""
    s1z_pn: Array
    """Spin component 1 z array."""
    s2x_pn: Array
    """Spin component 2 x array."""
    s2y_pn: Array
    """Spin component 2 y array."""
    s2z_pn: Array
    """Spin component 2 z array."""
    ln_hat_x_pn: Array
    """Logarithm of unit vector x component array."""
    ln_hat_y_pn: Array
    """Logarithm of unit vector y component array."""
    ln_hat_z_pn: Array
    """Logarithm of unit vector z component array."""


@_register_dataclass
@dataclasses.dataclass(frozen=True)
class IMRPhenomXPrecessionDataClass:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold precession parameters for IMRPhenomX waveform generation."""

    # # Flag to define the version of IMRPhenomXP called
    imr_phenom_x_prec_version: int = 0  # Flag to set version of Euler angles used.
    # Toggle to return coprecessing model without any twisting up
    imr_phenom_x_return_co_prec: int = 0
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
    a1: float = 0.0  # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
    a2: float = 0.0  # Mass weighted pre-factor, see Eq. 3.2 of Schmidt et al, arXiv:1408.1810
    a_sp_1: float = 0.0  # \f$A1 * S_{1 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
    a_sp_2: float = 0.0  # \f$A2 * S_{2 \perp}\f$, see Eq. 3.3 of Schmidt et al, arXiv:1408.1810
    chi1_perp: float = 0.0  # \f$ \chi_{1 \perp} \f$
    chi2_perp: float = 0.0  # \f$ \chi_{2 \perp} \f$
    s1_perp: float = 0.0  # \f$ S_{1 \perp} \f$
    s2_perp: float = 0.0  # \f$ S_{2 \perp} \f$
    s_l: float = 0.0  # \f$ \chi_{1 L} m^1_2 + \chi_{2 L} m^2_2 \f$
    s_perp: float = 0.0  # \f$ \chi_{p} m^1_2 \f$
    s_tot_perp: float = 0.0  # \f$ S_{1 \perp} + S_{2 \perp} \f$
    chi_perp: float = 0.0
    chi_tot_perp: float = 0.0  # \f$ S_{1 \perp} + S_{2 \perp} \f$

    # Effective precessing spin parameter: Schmidt et al, Phys. Rev. D 91, 024043 (2015), arXiv:1408.1810
    chi_p: float = 0.0  # \f$ \chi_{p} = S_p / (A_1 \chi^2_1) \f$, Eq. 3.4 of Schmidt et al, arXiv:1408.1810

    # Dimensionless aligned spin components on companion 1 and 2 respectively
    chi1_l: float = 0.0  # \f$ \chi_{1L} = \chi_{1} \cdot \hat{L} \f$
    chi2_l: float = 0.0  # \f$ \chi_{2L} = \chi_{2} \cdot \hat{L} \f$
    # Angle between J0 and line of sight (z-direction)
    theta_jn: float = 0.0  # Angle between J0 and line of sight (z-direction)

    # The initial phase that we pass to the underlying aligned spin IMR waveform model
    phi0_aligned: float = 0.0  # Initial phase to feed the underlying aligned-spin model

    # Angle to rotate the polarization
    zeta_polarization: float = 0.0  # Angle to rotate the polarizations

    # Post-Newtonian Euler angles
    alpha0: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha1: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha2: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha3: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha4_l: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha5: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha6: float = 0.0  # Coefficient of \f$\alpha\f$
    alpha_nnlo: float = 0.0  # Post Newtonian \f$\alpha\f$ at NNLO.
    alpha_offset: float = 0.0  # Offset for \f$\alpha\f$
    epsilon0: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon1: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon2: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon3: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon4_l: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon5: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon6: float = 0.0  # Coefficient of \f$\epsilon \f$
    epsilon_nnlo: float = 0.0  # Post Newtonian \f$\epsilon \f$ at NNLO.
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
    m_band_prec_version: int = 0  # Flag to control multibanding for Euler angles.

    # Source Frame Variables
    j0x_sf: float = 0.0  # \f$ J_{0,x}\f$ in L frame.
    j0y_sf: float = 0.0  # \f$ J_{0,y}\f$ in L frame.
    j0z_sf: float = 0.0  # \f$ J_{0,z}\f$ in L frame.
    j0: float = 0.0  # \f$ J_{0}\f$ in L frame.
    theta_j_sf: float = 0.0  # Angle between \f$J_0\f$ and \f$ L_{\rm{N}} \f$ (z-direction)
    phi_j_sf: float = 0.0  # Azimuthal angle of \f$J_0\f$ in the L frame
    nx_sf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in L frame.
    ny_sf: float = 0.0  # Line-of-sight vector component \f$ N_{y}\f$ in L frame.
    nz_sf: float = 0.0  # Line-of-sight vector component \f$ N_{z}\f$ in L frame.
    xx_sf: float = 0.0  # Component of triad basis vector X in L frame.
    xy_sf: float = 0.0  # Component of triad basis vector X in L frame.
    xz_sf: float = 0.0  # Component of triad basis vector X in L frame.
    kappa: float = 0.0  # Eq. C12 of arXiv:XXXX.YYYY

    # J-frame variables
    nx_jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
    ny_jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in J frame.
    nz_jf: float = 0.0  # Line-of-sight vector component \f$ N_{x}\f$ in LJ frame.
    p_arun_x_jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    p_arun_y_jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    p_arun_z_jf: float = 0.0  # Component of triad basis vector P in J frame, arXiv:0810.5336.
    q_arun_x_jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    q_arun_y_jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    q_arun_z_jf: float = 0.0  # Component of triad basis vector Q in J frame, arXiv:0810.5336.
    x_dot_p_arun: float = 0.0  # \f$ X \cdot P \f$
    x_dot_q_arun: float = 0.0  # \f$ X \cdot Q \f$

    # Orbital angular momentum
    l0: float = 0.0
    l1: float = 0.0
    l2: float = 0.0
    l3: float = 0.0
    l4: float = 0.0
    l5: float = 0.0
    l6: float = 0.0
    l7: float = 0.0
    l8: float = 0.0
    l8_l: float = 0.0
    ln: float = 0.0
    l_orb: float = 0.0
    l_ref: float = 0.0
    l_init: float = 0.0

    # Reference frequencies
    omega_ref: float = 0.0
    log_omega_ref: float = 0.0
    omega_ref_cbrt: float = 0.0
    omega_ref_cbrt2: float = 0.0

    # Spin weighted spherical harmonics
    # /* l = 2 */
    y2m2: complex = 0j
    y2m1: complex = 0j
    y20: complex = 0j
    y21: complex = 0j
    y22: complex = 0j

    # /* l = 3 */
    y3m3: complex = 0j
    y3m2: complex = 0j
    y3m1: complex = 0j
    y30: complex = 0j
    y31: complex = 0j
    y32: complex = 0j
    y3: complex = 0j

    # /* l = 4 */
    y4m4: complex = 0j
    y4m3: complex = 0j
    y4m2: complex = 0j
    y4m1: complex = 0j
    y40: complex = 0j
    y41: complex = 0j
    y42: complex = 0j
    y43: complex = 0j
    y44: complex = 0j

    # Useful square roots
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
    # l_hat = {0,0,1}
    l_hat_theta: float = 0.0
    l_hat_phi: float = 0.0
    l_hat_norm: float = 0.0
    l_hat_cos_theta: float = 0.0

    # Cartesian Dimensionful Spins
    s1x: float = 0.0  # \f$ S_{1,x} \f$ in L frame
    s1y: float = 0.0  # \f$ S_{1,y} \f$ in L frame
    s1z: float = 0.0  # \f$ S_{1,z} \f$ in L frame
    s2x: float = 0.0  # \f$ S_{2,x} \f$ in L frame
    s2y: float = 0.0  # \f$ S_{2,y} \f$ in L frame
    s2z: float = 0.0  # \f$ S_{2,z} \f$ in L frame

    # Spherical Polar Dimensionful Spins
    s1_norm: float = 0.0  # \f$ \left| S_{1} \right| \f$
    s1_theta: float = 0.0  # Spherical polar component \f$ S_{1,\theta} \f$ in L frame
    s1_phi: float = 0.0  # Spherical polar component \f$ S_{1,\phi} \f$ in L frame
    s1_cos_theta: float = 0.0  # Spherical polar component \f$ \cos S_{1,\theta} \f$ in L frame

    s2_norm: float = 0.0  # \f$ \left| S_{2} \right| \f$
    s2_theta: float = 0.0  # Spherical polar component \f$ S_{2,\theta} \f$ in L frame
    s2_phi: float = 0.0  # Spherical polar component \f$ S_{2,\phi} \f$ in L frame
    s2_cos_theta: float = 0.0  # Spherical polar component \f$ \cos S_{2,\theta} \f$ in L frame

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

    expansion_order: int = 0  # Flag to control expansion order of MSA system of equations.

    two_pi_gm: float = 0.0
    pi_gm: float = 0.0

    l_norm_n: float = 0.0  # Norm of Newtonian orbital angular momentum \f$ \left| L_N \right| \f$
    l_norm_3pn: float = 0.0  # Norm of orbital angular momentum at 3PN \f$ \left| L_{3 \rm{PN}} \right| \f$
    j_norm_n: float = (
        0.0  # Norm of total angular momentum using Newtonian orbital angular momentum \f$ \left| J_{N} \right| \f$
    )
    j_norm_3pn: float = 0.0  # Norm of total angular momentum at 3PN \f$ \left| J_{3 \rm{PN}} \right| \f$

    dot_s1_l: float = 0.0  # \f$ S_1 \cdot \hat{L} \f$
    dot_s1_ln: float = 0.0  # \f$ \hat{S}_1 \cdot \hat{L} \f$
    dot_s2_l: float = 0.0  # \f$ S_2 \cdot \hat{L} \f$
    dot_s2_ln: float = 0.0  # \f$ \hat{S}_1 \cdot \hat{L} \f$
    dot_s1_s2: float = 0.0  # \f$ S_1 \cdot S_2 \f$
    # \f$ S_{\rm{eff}} = (1 + q^{-1}) S_1 \cdot \hat{L} + (1 + q) S_2 \cdot \hat{L} \f$,
    # Eq. 7 of arXiv:1703.03967. Note convention for q. */
    s_eff: float = 0.0
    s_eff2: float = 0.0  # \f$ S_{\rm{eff}}^2 \f$ */

    # vector S1_0; /**< Initial value for \f$ S_{1} \f$ */
    # vector S2_0; /**< Initial value for \f$ S_{2} \f$ */
    # vector L_0; /**< Initial value for \f$ L \f$ */
    # vector Lhat_0; /**< Initial value for \f$ \hat{L} \f$ */
    # vector S_0; /**< Initial value for \f$ S_1 + S_2 \f$ */
    # vector J_0; /**< Initial value for \f$ J \f$ */

    s_0_norm: float = 0.0
    s_0_norm_2: float = 0.0
    j_0_norm: float = 0.0
    j_0_norm_2: float = 0.0
    l_0_norm: float = 0.0
    l_0_norm_2: float = 0.0

    delta_m_over_m: float = 0.0  # \f$ (m_1 - m_2) / (m_1 + m_2) \f$ */

    # //phiz_0, phiz_1, phiz_2, phiz_3, phiz_4, phiz_5
    phiz_0_coeff: float = 0.0
    phiz_1_coeff: float = 0.0
    phiz_2_coeff: float = 0.0
    phiz_3_coeff: float = 0.0
    phiz_4_coeff: float = 0.0
    phiz_5_coeff: float = 0.0

    # Omegaz_i coefficients in D10 - D15
    omegaz0_coeff: float = 0.0
    omegaz1_coeff: float = 0.0
    omegaz2_coeff: float = 0.0
    omegaz3_coeff: float = 0.0
    omegaz4_coeff: float = 0.0
    omegaz5_coeff: float = 0.0

    # Omegaz terms in D16 - D21
    omegaz0: float = 0.0
    omegaz1: float = 0.0
    omegaz2: float = 0.0
    omegaz3: float = 0.0
    omegaz4: float = 0.0
    omegaz5: float = 0.0

    omega_zeta0_coeff: float = 0.0
    omega_zeta1_coeff: float = 0.0
    omega_zeta2_coeff: float = 0.0
    omega_zeta3_coeff: float = 0.0
    omega_zeta4_coeff: float = 0.0
    omega_zeta5_coeff: float = 0.0

    omega_zeta0: float = 0.0
    omega_zeta1: float = 0.0
    omega_zeta2: float = 0.0
    omega_zeta3: float = 0.0
    omega_zeta4: float = 0.0
    omega_zeta5: float = 0.0

    # MSA-SUA Euler Angles
    phiz: float = 0.0  # Azimuthal angle of L around J
    zeta: float = 0.0  # Angle to describe L w.r.t. J
    cos_theta_l: float = 0.0  # Cosine of polar angle between L and J

    # First order MSA corrections
    zeta_0_msa: float = 0.0  # First MSA correction \f$ \zeta_0 \f$, Eq. F19 of arXiv:1703.03967
    phiz_0_msa: float = 0.0  # First MSA correction \f$ \phi_{z,0} \f$, Eq. 67 of arXiv:1703.03967

    # Initial values
    zeta_0: float = 0.0  # Initial value of \f$ \zeta \f$
    phiz_0: float = 0.0  # Initial value of \f$ \phi_{z,0} \f$

    # Orbital velocity, v and v^2
    v: float = 0.0  # Orbital velocity, \f$ v \f$
    v2: float = 0.0  # Orbital velocity squared, \f$ v^2 \f$
    # Reference orbital velocity, v and v^2
    v_0: float = 0.0  # Orbital velocity at reference frequency, \f$ v_{\rm{ref}} \f$
    v_0_2: float = 0.0  # Orbital velocity at reference frequency squared, \f$ v_{\rm{ref}}^2 \f$

    delta: float = 0.0  # Eq. C3 of arXiv:1703.03967

    d2: float = 0.0
    d3: float = 0.0

    # Precession averaged total spin, Eq. 45
    s_av: float = 0.0  # \f$ S_{\rm{av}} \f$ as defined in Eq. 45 of arXiv:1703.03967
    s_av2: float = 0.0  # \f$ S_{\rm{av}}^2 \f$
    inv_s_av2: float = 0.0  # \f$ 1 / S_{\rm{av}}^2 \f$
    inv_s_av: float = 0.0  # \f$ 1 / S_{\rm{av}} \f$
    # Eq. C1, C2 for Eq. 51
    psi1: float = 0.0  # \f$ \psi_1 \f$ as defined in Eq. C1 of arXiv:1703.03967
    psi2: float = 0.0  # \f$ \psi_2 \f$ as defined in Eq. C2 of arXiv:1703.03967

    # Integration constant in Eq. 51
    psi0: float = 0.0  # \f$ \psi_0 \f$ as per Eq. 51 of arXiv:1703.03967

    # Eq. 51 and Eq. 24
    psi: float = 0.0  # \f$ \psi \f$ as defined by Eq. 51 of arXiv:1703.03967
    psi_dot: float = 0.0  # \f$ \dot{\psi} \f$ as per Eq. 50 of arXiv:1703.03967

    c_phi: float = 0.0  # \f$ C_{\phi} \f$ as defined by Eq. B14 of arXiv:1703.03967
    d_phi: float = 0.0  # \f$ C_{\phi} \f$ as defined by Eq. B15 of arXiv:1703.03967
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
    g6_l: float = 0.0

    # Spin-Orbit couplings
    beta3: float = 0.0
    beta5: float = 0.0
    beta6: float = 0.0
    beta7: float = 0.0
    sigma4: float = 0.0

    # Total normed spin of primary and secondary BH squared
    s1_norm_2: float = 0.0
    s2_norm_2: float = 0.0

    # Precession averaged spin couplings in A9 - A14
    # Precession averaged coupling \f$ \langle S_1 \cdot \hat{L} \rangle_{\rm{pr}} \f$,
    # Eq. A9 of arXiv:1703.03967
    s1_l_pav: float = 0.0
    # Precession averaged coupling \f$ \langle S_2 \cdot \hat{L} \rangle_{\rm{pr}} \f$,
    # Eq. A10 of arXiv:1703.03967
    s2_l_pav: float = 0.0
    # Precession averaged coupling \f$ \langle S_1 \cdot S_2 \rangle_{\rm{pr}} \f$,
    # Eq. A11 of arXiv:1703.03967
    s1_s2_pav: float = 0.0
    # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$,
    # Eq. A12 of arXiv:1703.03967
    s1_l_sq_pav: float = 0.0
    # Precession averaged coupling \f$ \langle (S_2 \cdot \hat{L})^2 \rangle_{\rm{pr}} \f$,
    # Eq. A13 of arXiv:1703.03967
    s2_l_sq_pav: float = 0.0
    # Precession averaged coupling \f$ \langle (S_1 \cdot \hat{L}) (S_2 \cdot \hat{L}) \rangle_{\rm{pr}} \f$,
    # Eq. A14 of arXiv:1703.03967 |
    s1_l_s2_l_pav: float = 0.0
    # Total spin in Eq. 23 of Chatziioannou et al PRD, 95, 104004, (2017)
    s_norm: float = 0.0
    s_norm_2: float = 0.0

    s_pl2: float = 0.0  # Largest root of polynomial \f$ S^2_+ \f$, Eq. 22 of arXiv:1703.03967
    s_mi2: float = 0.0  # Smallest root of polynomial \f$ S^2_- \f$, Eq. 22 of arXiv:1703.03967
    s32: float = 0.0  # Third root of polynomial \f$ S^2_3 \f$, Eq. 22 of arXiv:1703.03967
    s_pl: float = 0.0  # \f$ S_+ \f$
    s_mi: float = 0.0  # \f$ S_- \f$
    s3: float = 0.0  # \f$ S_3 \f$
    s_pl2_m_s_mi2: float = 0.0  # \f$ S^2_+ - S^2_- \f$
    s_pl2_p_s_mi2: float = 0.0  # \f$ S^2_+ + S^2_- \f$

    a_coeff: float = 0.0
    b_coeff: float = 0.0
    c_coeff: float = 0.0
    d_coeff: float = 0.0
    qq: float = 0.0
    inv_qq: float = 0.0
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

    s_const: float = 0.0

    l_pn_coefficients: Array | None = None

    constants_l: Array | None = None

    # // Variables to interpolate SpinTaylor angles, up to fmax_angles
    # gsl_spline *alpha_spline;
    # gsl_spline *cosbeta_spline;
    # gsl_spline *gamma_spline;

    # gsl_interp_accel *alpha_acc;
    # gsl_interp_accel *cosbeta_acc;
    # gsl_interp_accel *gamma_acc;

    m_fmax_angles: float = 0.0
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
    ftrans_mrd: float = 0.0
    fmax_inspiral: float = 0.0
    precessing_tag: int = 0
    deltaf_interpolation: float = 0.0
    m_min: float = 0.0
    m_max: float = 0.0
    l_max_pnr: float = 0.0
    pn_arrays: PhenomXPInspiralArrays | None = None
    # Buffer region for integration of SpinTaylor equations:
    # added so that interpolated angles cover the frequency range requested by user
    integration_buffer: float = 0.0
    # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models
    fmin_integration: float = 0.0
    # Minimum frequency covered by the integration of PN spin-precessing equations for SpinTaylor models
    m_fmin_integration: float = 0.0

    msa_error: float = 0.0  # Flag to track errors in initialization of MSA system.

    # /* PNR-specific additions for single-spin mapping */
    # Magnitude of effective single spin used for tapering two-spin angles, Eq. 18 of arXiv:2107.08876
    chi_single_spin: float = 0.0
    cos_theta_single_spin: float = 0.0  # Polar angle of effective single spin, Eq. 19 or arXiv:2107.08876
    # Polar angle of approximate final spin, see technical document FIXME: add reference
    cos_theta_final_single_spin: float = 0.0

    chi_max_spin: float = 0.0
    cos_theta_max_spin: float = 0.0

    chi1x_evolved: float = 0.0  # x-component of spin on primary at end of SpinTaylor evolution
    chi1y_evolved: float = 0.0  # y-component of spin on primary at end of SpinTaylor evolution
    chi1z_evolved: float = 0.0  # z-component of spin on primary at end of SpinTaylor evolution
    chi2x_evolved: float = 0.0  # x-component of spin on secondary at end of SpinTaylor evolution
    chi2y_evolved: float = 0.0  # y-component of spin on secondary at end of SpinTaylor evolution
    chi2z_evolved: float = 0.0  # z-component of spin on secondary at end of SpinTaylor evolution

    chi_single_spin_antisymmetric: float = (
        0.0  # magnitude of effective single spin of a two spin system for the antisymmetric waveform
    )
    theta_antisymmetric: float = 0.0  # Polar angle effective single spin for antisymmetric waveform

    pnr_hm_m_f_low: float = 0.0  # Mf_alpha_lower stored from alphaParams struct, 2 A4 / 7 from arXiv:2107.08876
    pnr_hm_m_f_high: float = 0.0  # Mf_beta_lower stored from betaParams struct, Eq. 58 from arXiv:2107.08876
    pnr_q_window_lower: float = 0.0  # Boundary values for PNR angle transition window
    pnr_q_window_upper: float = 0.0
    pnr_chi_window_lower: float = 0.0
    pnr_chi_window_upper: float = 0.0
    pnr_inspiral_scaling: int = 0  # Enforce inpsiral scaling for HM angles outside of calibration window

    # /* Store PNR-specific waveform flags for turning on and off tuning */
    imr_phenom_xpnr_use_tuned_angles: int = 0
    imr_phenom_xpnr_use_tuned_coprec: int = 0
    imr_phenom_xpnr_use_tuned_coprec33: int = 0
    imr_phenom_xpnr_use_input_coprec_deviations: int = 0
    imr_phenom_xpnr_force_xhm_alignment: int = 0
    apply_pnr_deviations: int = 0

    # /* A copy of the XAS 22 object */
    p_wf_22_as: IMRPhenomXWaveformDataClass | None = None
    imr_phenom_xpnr_interp_tolerance: float = 0.0

    # /* Store anti-symmetric waveform flag for turning on and off */
    imr_phenom_x_antisymmetric_waveform: int = 0

    # /* polarization symmetry property, refer to XXXX.YYYYY for details */
    polarization_symmetry: float = 0.0

    # /* variables to store PNR angles for use in existing XP and XPHM twist-up functions */
    alpha_pnr: float = 0.0
    beta_pnr: float = 0.0
    gamma_pnr: float = 0.0

    # /* flag to use MR beta or analytic continuation with PNR angles */
    use_mr_beta: int = 0

    mf_alpha_lower: float = 0.0

    # /* flag to toggle conditional precession multibanding */
    conditional_prec_mband: int = 0

    lal_params: IMRPhenomXPHMParameterDataClass | None = None
