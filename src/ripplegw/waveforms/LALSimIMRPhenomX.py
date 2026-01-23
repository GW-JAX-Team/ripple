import jax.numpy as jnp
import jax
from typing import Tuple, Optional, Dict, Any
from ..typing import Array
from ..constants import G, MSUN, C

from .LALSimIMRPhenomX_precession import (
    IMRPhenomX_SpinTaylorAnglesSplinesAll,
    IMRPhenomXGetAndSetPrecessionVariables,
    IMRPhenomX_Return_phi_zeta_costhetaL_MSA,
    XLALSimIMRPhenomXUtilsHztoMf,
    alphaMRD,
    betaMRD,
    gamma_from_alpha_cosbeta,
    set_epsilon0,
)
from .LALSimIMRPhenomX_internals import IMRPhenomXSetWaveformVariables


def XLALSimIMRPhenomXPMSAAngles(
    freqs: Array,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    inclination: float,
    fRef_In: float,
    mprime: int,
    lalParams: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, Array, Array]:
    """
    Compute MSA (Minimum rotation Sppinning Approximation) Euler angles for IMRPhenomXP waveforms.

    This function calculates the three Euler angles that describe the orientation of the orbital
    angular momentum L with respect to the total angular momentum J as a function of frequency.

    Parameters
    ----------
    freqs : Array
        Input frequency array [Hz] (gravitational-wave frequencies)
    m1_SI : float
        Mass of companion 1 in SI units (kg)
    m2_SI : float
        Mass of companion 2 in SI units (kg)
    chi1x : float
        x-component of dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
    chi1y : float
        y-component of dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
    chi1z : float
        z-component of dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
    chi2x : float
        x-component of dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
    chi2y : float
        y-component of dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
    chi2z : float
        z-component of dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
    inclination : float
        Inclination angle between LN and line of sight [radians]
    fRef_In : float
        Reference frequency [Hz]. If 0, defaults to first frequency in freqs
    mprime : int
        Spherical harmonic order m
    lalParams : dict, optional
        LAL parameters dictionary. If None, defaults will be used.

    Returns
    -------
    alpha_of_f : Array
        Azimuthal angle of L around J as function of frequency [radians]
    gamma_of_f : Array
        Third Euler angle describing L w.r.t. J (fixed by minimal rotation condition) [radians]
    cosbeta_of_f : Array
        Cosine of polar angle between L and J as function of frequency

    Notes
    -----
    - This implementation follows the LALSuite C implementation from LALSimIMRPhenomX_precession.c
    - The input frequencies are *gravitational-wave* frequencies, not orbital frequencies
    - Only supports IMRPhenomXPrecVersion 220, 221, 222, 223, or 224
    - The angles are computed using the MSA system which assumes minimal rotation

    Reference
    ---------
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c.html
    """

    # Extract aligned spin components
    chi1L = chi1z
    chi2L = chi2z

    # Get precession version flag
    pflag = 223

    # Create basic waveform parameters dictionary
    # This is a simplified version - in production you'd call IMRPhenomXSetWaveformVariables
    M_total = (m1_SI + m2_SI) / MSUN  # Total mass in solar masses
    eta = (m1_SI * m2_SI) / ((m1_SI + m2_SI) ** 2)  # Symmetric mass ratio

    # Compute piGM for velocity calculation
    piGM = jnp.pi * G * (m1_SI + m2_SI) / (C**3)

    lalParams_aux = lalParams.copy()

    # Create simplified waveform struct as dict
    pWF = IMRPhenomXSetWaveformVariables(
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        chi1L_In=chi1L,
        chi2L_In=chi2L,
        deltaF=0.0,
        fRef=fRef_In,
        phi0=0.0,
        f_min=freqs[0],
        f_max=freqs[-1],
        distance=1.0,
        inclination=inclination,
        lalParams=lalParams_aux,
        debug=False,
    )

    # Initialize precession struct
    # In full implementation, this calls IMRPhenomXGetAndSetPrecessionVariables
    pPrec = IMRPhenomXGetAndSetPrecessionVariables(
        pWF=pWF,
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        chi1x=chi1x,
        chi1y=chi1y,
        chi1z=chi1z,
        chi2x=chi2x,
        chi2y=chi2y,
        chi2z=chi2z,
        lalParams=lalParams_aux,
        debug_flag=False,
    )

    # Vectorized computation of angles over frequency array
    def compute_angles_at_freq(f):
        """Compute MSA angles at a single frequency."""
        # Convert GW frequency to velocity parameter
        # v = (pi * M * f_gw)^(1/3) where f_gw = m * f_orbital
        # For GW frequency: f_gw = mprime * f_orbital / 2
        v = jnp.cbrt(f * piGM * (2.0 / mprime))

        # Get MSA angles: returns [phi_z, zeta, cos(theta_L)]
        vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(pPrec, pWF, v)

        # Extract and apply offsets
        alpha = vangles[0] - pPrec.alpha_offset
        gamma = -(vangles[1] - pPrec.epsilon_offset)
        cosbeta = vangles[2]

        return alpha, gamma, cosbeta

    # Vectorize over frequency array

    alphas, gammas, cosbetas = jax.vmap(compute_angles_at_freq)(freqs)
    # print("JAX DEBUG pPrec.alpha_offset, pPrec.epsilon_offset", pPrec.alpha_offset, pPrec.epsilon_offset)
    return alphas, gammas, cosbetas


def XLALSimIMRPhenomXPSpinTaylorAngles(
    m1_SI: float,  # Mass of companion 1 (kg)
    m2_SI: float,  # Mass of companion 2 (kg)
    s1x: float,  # x component of primary spin
    s1y: float,  # y component of primary spin
    s1z: float,  # z component of primary spin
    s2x: float,  # x component of secondary spin
    s2y: float,  # y component of secondary spin
    s2z: float,  # z component of secondary spin
    fmin: float,  # starting GW frequency (Hz)
    fmax: float,  # maximum GW frequency (Hz)
    deltaF: float,  # starting GW frequency (Hz)
    fRef: float,  # reference GW frequency (Hz)
    phiRef: float,  # reference orbital phase (rad)
    LALparams: dict,  # LAL Dictionary struct
):
    pversion = 330  # set to this for SpinTaylor angles

    # Create basic waveform parameters dictionary
    # This is a simplified version - in production you'd call IMRPhenomXSetWaveformVariables
    M_total = (m1_SI + m2_SI) / MSUN  # Total mass in solar masses
    eta = (m1_SI * m2_SI) / ((m1_SI + m2_SI) ** 2)  # Symmetric mass ratio

    # Compute piGM for velocity calculation
    piGM = jnp.pi * G * (m1_SI + m2_SI) / (C**3)

    lalParams_aux = LALparams.copy()

    # Create simplified waveform struct as dict
    pWF = IMRPhenomXSetWaveformVariables(
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        chi1L_In=s1z,
        chi2L_In=s2z,
        deltaF=deltaF,
        fRef=fRef,
        phi0=phiRef,
        f_min=fmin,
        f_max=fmax,
        distance=1.0,
        inclination=0.0,
        lalParams=lalParams_aux,
        debug=False,
    )

    # Initialize precession struct
    # In full implementation, this calls IMRPhenomXGetAndSetPrecessionVariables
    pPrec = IMRPhenomXGetAndSetPrecessionVariables(
        pWF=pWF,
        m1_SI=m1_SI,
        m2_SI=m2_SI,
        chi1x=s1x,
        chi1y=s1y,
        chi1z=s1z,
        chi2x=s2x,
        chi2y=s2y,
        chi2z=s2z,
        lalParams=lalParams_aux,
        debug_flag=False,
    )
    # object.__setattr__(pPrec, 'M_MIN', 2) default is 2

    IMRPhenomX_SpinTaylorAnglesSplinesAll(fmin, fmax, pWF, pPrec, lalParams_aux)

    alpha0 = jnp.pi - pPrec.kappa  # For phenom_xp_convention = 1
    alphaOff = alpha0
    alpha_offset = -pPrec.alpha_ref + alpha0

    Mfmin = XLALSimIMRPhenomXUtilsHztoMf(fmin, pWF["Mtot"])
    alphamin = pPrec.alpha_spline(Mfmin)
    cosbetamin = pPrec.cosbeta_spline(Mfmin)

    # determine offset for gamma et the end
    gamma_at0 = 0
    alpha_at0 = alphamin - pPrec.alpha_ref + alphaOff
    cosbeta_at0 = cosbetamin

    deltaMF = pWF["deltaMF"]
    output_length = (fmax - fmin) / deltaF + 1
    frequencies = Mfmin + jnp.arange(output_length) * deltaMF
    Mf_array = frequencies[1:]

    # --- Vectorized computation of alpha and cosbeta ---

    # --- Split frequency array at ftrans_MRD ---
    in_inspiral_mask = Mf_array < pPrec.ftrans_MRD

    # Get indices for splitting
    # Find last inspiral index
    n_inspiral = jnp.sum(in_inspiral_mask)

    Mf_array_inspiral = Mf_array[:n_inspiral]
    Mf_array_mrd = Mf_array[n_inspiral:]

    # --- Inspiral region: use splines ---
    alpha_inspiral = pPrec.alpha_spline(Mf_array_inspiral) + alpha_offset
    cosbeta_inspiral = pPrec.cosbeta_spline(Mf_array_inspiral)

    # --- MRD region: depends on version ---
    use_analytic_mrd = jnp.logical_or(
        pPrec.IMRPhenomXPrecVersion == 320, pPrec.IMRPhenomXPrecVersion == 321
    )

    def compute_mrd_analytic(Mf):
        """MRD with PN (versions 320, 321)."""
        alpha = alphaMRD(Mf, pPrec.alpha_params) + alpha_offset
        beta = betaMRD(Mf, pPrec.beta_params)
        cosbeta = jnp.cos(beta)
        return alpha, cosbeta

    def compute_mrd_frozen(Mf_array_mrd):
        """MRD frozen at last inspiral value (other versions)."""
        # Get the last inspiral values
        alpha_last = alpha_inspiral[-1]
        cosbeta_last = cosbeta_inspiral[-1]
        n_mrd = Mf_array_mrd.shape[0]
        return jnp.full(n_mrd, alpha_last), jnp.full(n_mrd, cosbeta_last)

    # Compute MRD region based on version
    alpha_mrd, cosbeta_mrd = jax.lax.cond(
        use_analytic_mrd,
        jax.vmap(compute_mrd_analytic),
        compute_mrd_frozen,
        operand=Mf_array_mrd,
    )

    # --- Concatenate inspiral and MRD ---
    alphaFS = jnp.concatenate([jnp.array([alpha_at0]), alpha_inspiral, alpha_mrd])
    cosbetaFS = jnp.concatenate(
        [jnp.array([cosbeta_at0]), cosbeta_inspiral, cosbeta_mrd]
    )

    # --- Gamma computation using scan (cumulative) ---
    def compute_deltagamma_at_Mf(Mf):
        """Compute delta gamma at a single Mf."""
        in_inspiral = Mf < pPrec.ftrans_MRD
        use_analytic_mrd = jnp.logical_or(
            pPrec.IMRPhenomXPrecVersion == 320, pPrec.IMRPhenomXPrecVersion == 321
        )

        # Only compute deltagamma if in inspiral OR using analytic MRD
        should_compute = jnp.logical_or(in_inspiral, use_analytic_mrd)
        deltagamma = jnp.where(
            should_compute, gamma_from_alpha_cosbeta(Mf, deltaMF, pPrec), 0.0
        )
        return deltagamma

    # Compute all delta_gamma values (skip first frequency, gamma[0] = 0)
    delta_gamma_array = jax.vmap(compute_deltagamma_at_Mf)(Mf_array)

    # Cumulative sum to get gamma values
    gamma_cumsum = jnp.cumsum(delta_gamma_array)
    gammaFS = jnp.concatenate([jnp.array([gamma_at0]), gamma_cumsum])

    epsilon0 = set_epsilon0(1, pPrec.phiJ_Sf)
    gammaFS = -(gammaFS - pPrec.gamma_ref - epsilon0)

    return alphaFS, cosbetaFS, gammaFS
