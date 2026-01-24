import jax.numpy as jnp
import jax
from ..constants import MTSUN
from .initialise_MSA_system import IMRPhenomX_Initialize_MSA_System
jax.config.update("jax_enable_x64", True)


def XLALSimIMRPhenomXchiEff(eta: float, chi1L: float, chi2L: float) -> float:
    """
    Effective aligned spin parameter.

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Effective aligned spin
    """
    # Convention m1 >= m2 and chi1 is the spin on m1
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1 + delta)
    mm2 = 0.5 * (1 - delta)

    return mm1 * chi1L + mm2 * chi2L


def XLALSimIMRPhenomXSTotR(eta: float, chi1L: float, chi2L: float) -> float:
    """
    Total spin normalised to [-1,1].

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Normalized total spin
    """
    # Convention m1 >= m2 and chi1z is the spin projected along Lz on m1
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1 + delta)
    m2 = 0.5 * (1 - delta)
    m1Sq = m1 * m1
    m2Sq = m2 * m2

    return (m1Sq * chi1L + m2Sq * chi2L) / (m1Sq + m2Sq)



def XLALSimIMRPhenomXFinalSpin2017(eta: float, chi1L: float, chi2L: float) -> float:
    """
    Final Dimensionless Spin, X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332.

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Final dimensionless spin
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    m1Sq = m1 * m1
    m2Sq = m2 * m2

    eta2 = eta * eta
    eta3 = eta2 * eta

    S = XLALSimIMRPhenomXSTotR(eta, chi1L, chi2L)
    S2 = S * S
    S3 = S2 * S

    dchi = chi1L - chi2L
    dchi2 = dchi * dchi

    # No-spin contribution
    noSpin = (3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3) / (
        1 + 7.2388440419467335 * eta
    )

    # Equal-spin contribution
    eqSpin = (m1Sq + m2Sq) * S + (
        (-0.8561951310209386 * eta - 0.09939065676370885 * eta2 + 1.668810429851045 * eta3) * S +
        (0.5881660363307388 * eta - 2.149269067519131 * eta2 + 3.4768263932898678 * eta3) * S2 +
        (0.142443244743048 * eta - 0.9598353840147513 * eta2 + 1.9595643107593743 * eta3) * S3
    ) / (1 + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3) * S)

    # Unequal-spin contribution
    uneqSpin = (
        0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta) * eta2 -  # Linear in spin difference
        0.059808322561702126 * dchi2 * eta3 +  # Quadratic in spin difference
        2.3170397514509933 * dchi * delta * (1 - 3.2624649875884852 * eta) * eta3 * S  # Mixed term
    )

    return noSpin + eqSpin + uneqSpin


def XLALSimIMRPhenomXFinalMass2017(eta: float, chi1L: float, chi2L: float) -> float:
    """
    Final Mass in units of total initial mass, X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332.

    This function computes the final mass of the remnant black hole after merger,
    normalized by the total initial mass (Mfinal/Mtotal). The returned value represents
    1 - Erad, where Erad is the radiated energy.

    Args:
        eta: Symmetric mass ratio
        chi1L: Aligned spin of BH 1
        chi2L: Aligned spin of BH 2

    Returns:
        Final mass in units of total initial mass (1 - Erad)
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    S = XLALSimIMRPhenomXSTotR(eta, chi1L, chi2L)
    S2 = S * S
    S3 = S2 * S

    dchi = chi1L - chi2L
    dchi2 = dchi * dchi

    # No-spin contribution
    noSpin = (
        0.057190958417936644 * eta
        + 0.5609904135313374 * eta2
        - 0.84667563764404 * eta3
        + 3.145145224278187 * eta4
    )

    # Equal-spin contribution
    # Because of the way this is written, we need to subtract the noSpin term
    eqSpin = (
        (0.057190958417936644 * eta + 0.5609904135313374 * eta2 - 0.84667563764404 * eta3 + 3.145145224278187 * eta4)
        * (
            1
            + (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta2) * S
            + (-0.17762802148331427 + 2.176667900182948 * eta2) * S2
            + (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta2) * S3
        )
    ) / (1 + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2) * S)

    eqSpin = eqSpin - noSpin

    # Unequal-spin contribution
    uneqSpin = (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653 * dchi * delta * (1 - 4.91667749015812 * eta) * eta * S
    )

    # Mfinal = 1 - Erad, assuming that M = m1 + m2 = 1
    return 1.0 - (noSpin + eqSpin + uneqSpin)


def IMRPhenomX_PNR_GetAndSetPNRVariables(pPrec, pWF: dict):
    """
    Get and set PNR (Precessing Numerical Relativity) variables for single-spin mapping.

    Computes effective single-spin parameters from the two-spin system using the mapping
    described in arXiv:2107.08876. This function populates several fields in the pPrec
    structure that are used for PNR angle calculations.

    Args:
        pPrec: PhenomX precession struct (IMRPhenomXGetAndSetPrecessionVariables instance)
        pWF: PhenomX waveform struct (dict)

    Returns:
        int: Success status (1 for success)

    References:
        - Eq. 17-19 of arXiv:2107.08876
    """
    # Get needed quantities
    m1 = pWF['m1'] #* pWF['Mtot']
    m2 = pWF['m2'] #* pWF['Mtot']
    q = pWF['q']

    # Select spin values based on version
    is_version_330 = (pPrec.IMRPhenomXPrecVersion == 330)
    #FIXME
    '''
    chi1x = jnp.where(is_version_330, pPrec.chi1x_evolved, pPrec.chi1x)
    chi1y = jnp.where(is_version_330, pPrec.chi1y_evolved, pPrec.chi1y)
    chi2x = jnp.where(is_version_330, pPrec.chi2x_evolved, pPrec.chi2x)
    chi2y = jnp.where(is_version_330, pPrec.chi2y_evolved, pPrec.chi2y)

    chi1z_val = jnp.where(is_version_330, pPrec.chi1z_evolved, pPrec.chi1z)
    chi2z_val = jnp.where(is_version_330, pPrec.chi2z_evolved, pPrec.chi2z)
    '''
    chi1x, chi1y, chi1z_val = pPrec.chi1x, pPrec.chi1y, pPrec.chi1z
    chi2x, chi2y, chi2z_val = pPrec.chi2x, pPrec.chi2y, pPrec.chi2z
    
    chieff = XLALSimIMRPhenomXchiEff(pWF['eta'], chi1z_val, chi2z_val)

    # Compute parallel spin parameter
    chipar = pWF['Mtot'] * chieff / m1

    chiperp, chiperp_antisymmetric = evaluate_chiperp(m1, m2, chi1x, chi1y, chi2x, chi2y, pPrec.chi_p, pPrec.IMRPhenomXPrecVersion)

    # Get the total magnitude, Eq. 18 of arXiv:2107.08876
    chi_mag = jnp.sqrt(chipar * chipar + chiperp * chiperp)
    chi_mag_antisymmetric = jnp.sqrt(chipar * chipar + chiperp_antisymmetric * chiperp_antisymmetric)

    # Set attributes on pPrec
    object.__setattr__(pPrec, 'chi_singleSpin', chi_mag)
    object.__setattr__(pPrec, 'chi_singleSpin_antisymmetric', chi_mag_antisymmetric)

    # Get the opening angle of the single spin, Eq. 19 of arXiv:2107.08876
    costheta = jnp.where(chi_mag >= 1.0e-6, chipar / chi_mag, 0.0)
    theta_antisymmetric = jnp.where(
        chi_mag_antisymmetric >= 1.0e-6,
        jnp.arccos(chipar / chi_mag_antisymmetric),
        0.0
    )

    object.__setattr__(pPrec, 'costheta_singleSpin', costheta)
    object.__setattr__(pPrec, 'theta_antisymmetric', theta_antisymmetric)

    # Compute an approximate final spin using single-spin mapping
    chi1L = chi_mag * costheta
    chi2L = 0.0

    Xfparr = XLALSimIMRPhenomXFinalSpin2017(pWF['eta'], chi1L, chi2L)

    # Rescale Xfperp to use the final total mass of 1
    qfactor = q / (1.0 + q)
    Xfperp = qfactor * qfactor * chi_mag * jnp.sqrt(1.0 - costheta * costheta)
    xf = jnp.sqrt(Xfparr * Xfparr + Xfperp * Xfperp)

    costheta_final_singleSpin = jnp.where(xf > 1.0e-6, Xfparr / xf, 0.0)

    object.__setattr__(pPrec, 'costheta_final_singleSpin', costheta_final_singleSpin)

    # Initialize frequency values
    object.__setattr__(pPrec, 'PNR_HM_Mflow', 0.0)
    object.__setattr__(pPrec, 'PNR_HM_Mfhigh', 0.0)

    # Set angle window boundaries
    object.__setattr__(pPrec, 'PNR_q_window_lower', 8.5)
    object.__setattr__(pPrec, 'PNR_q_window_upper', 12.0)
    object.__setattr__(pPrec, 'PNR_chi_window_lower', 0.85)
    object.__setattr__(pPrec, 'PNR_chi_window_upper', 1.2)

    # Set inspiral scaling flag for HM frequency map
    PNRInspiralScaling = jnp.where(
        jnp.logical_or(q > 12.0, chi_mag > 1.2),
        1,
        0
    )
    object.__setattr__(pPrec, 'PNRInspiralScaling', PNRInspiralScaling)

    return pPrec  # XLAL_SUCCESS




def IMRPhenomX_PNR_HMInterpolationDeltaF(f_min: float, pWF: dict, pPrec: dict) -> float:
    """
    Compute deltaF required for PNR HM interpolation
    
    Args:
        f_min: minimum starting frequency (Hz) (float)
        pWF: PhenomX waveform struct (dict)
        pPrec: PhenomX precession struct (dict)
        
    Returns:
        float: Delta frequency for interpolation
    """
    
    error_tolerance = pPrec['IMRPhenomXPNRInterpTolerance']
    
    # aligned-spin limit means we don't need high resolution on the prec angles (which should be zero)
    aligned_spin_condition = jnp.logical_and(
        pPrec['chi1_perp'] == 0.0,
        pPrec['chi2_perp'] == 0.0
    )
    
    def aligned_spin_case():
        return jnp.where(pWF['deltaF'] != 0, pWF['deltaF'], 0.1)
    
    def precessing_case():
        # Compute approximate scaling from leading-order alpha contribution
        Mf = XLALSimPhenomUtilsHztoMf(f_min, pWF['Mtot'])
        eta_term = jnp.sqrt(1.0 - 4.0 * pWF['eta'])
        numerator = 3.0 * jnp.pi * jnp.power(Mf, 5) * error_tolerance * (1.0 + eta_term)
        denominator = 7.0 + 13.0 * eta_term
        constant = 4.0 * jnp.sqrt(2.0 / 5.0)
        
        deltaF_alpha = XLALSimPhenomUtilsMftoHz(
            constant * jnp.sqrt(jnp.sqrt(numerator / denominator)), 
            pWF['Mtot']
        )
        
        # Check for two-spin oscillations
        two_spin_condition = IMRPhenomX_PNR_CheckTwoSpin(pPrec)  # stub
        
        def handle_two_spin():
            precessing_tag = (pPrec['IMRPhenomXPrecVersion'] - 
                            (pPrec['IMRPhenomXPrecVersion'] % 100)) // 100
            
            # If SpinTaylor version (tag == 3), we need to set up L coefficients
            # This is just setup - the main logic happens regardless
            def setup_spintaylor():
                eta = pPrec.eta
                delta = pWF['delta']
                chi1L = pPrec.chi1L
                chi2L = pPrec.chi2L

                pPrec.L0   = 1.0
                pPrec.L1   = 0.0
                pPrec.L2   = 3.0/2. + eta/6.0
                pPrec.L3   = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
                pPrec.L4   = (81 + (-57 + eta)*eta)/24.
                pPrec.L5   = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
                pPrec.L6   = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*jnp.power(jnp.pi, 2)))/1296.
                pPrec.L7   = 0.0
                pPrec.L8   = 0.0
                #// This is the log(x) term
                pPrec.L8L  = 0.0

                user_version = pPrec.IMRPhenomXPrecVersion

                pPrec.IMRPhenomXPrecVersion = 223

                IMRPhenomX_Initialize_MSA_System(pWF, pPrec, pPrec.ExpansionOrder) #TODO 

                pPrec.IMRPhenomXPrecVersion = user_version
                return pPrec
                
            def no_setup():
                return pPrec
                
            # Do setup if needed (this just returns pPrec for now)
            pPrec_updated = jax.lax.cond(
                precessing_tag == 3,
                setup_spintaylor,
                no_setup
            )
            
            # NOW do the main two-spin logic (this happens regardless of precessing_tag)
            # Use precomputed MSA terms
            g0 = pPrec_updated['g0']
            deltam = pPrec_updated['delta_qq'] 
            psi1 = pPrec_updated['psi1']
            psi2 = pPrec_updated['psi2']
            
            v0 = jnp.power(jnp.pi * Mf, 1.0/3.0)
            v02 = v0 * v0
            v06 = v02 * v02 * v02
            
            # frequency derivative of psi, Eq. 51 of arXiv:1703.03967
            dpsi = g0 * deltam * jnp.pi / (4.0 * v06) * (3.0 + 2.0 * psi1 * v0 + psi2 * v02)
            dpsiInv = jnp.abs(1.0 / dpsi)
            
            # Compute L_fmin for beta calculations
            L_fmin = (pWF['Mtot'] * pWF['Mtot'] * 
                     XLALSimIMRPhenomXLPNAnsatz(v0, pWF['eta'] / v0, 
                                               pPrec_updated['L0'], pPrec_updated['L1'], pPrec_updated['L2'], 
                                               pPrec_updated['L3'], pPrec_updated['L4'], pPrec_updated['L5'], 
                                               pPrec_updated['L6'], pPrec_updated['L7'], pPrec_updated['L8'], 
                                               pPrec_updated['L8L']))  # stub
            
            betaMin = jnp.arctan2(
                jnp.abs(pPrec_updated['S1_perp'] - pPrec_updated['S2_perp']), 
                L_fmin + pPrec_updated['SL']
            )
            betaMax = jnp.arctan2(
                pPrec_updated['S1_perp'] + pPrec_updated['S2_perp'], 
                L_fmin + pPrec_updated['SL']
            )
            
            # Adjust dpsiInv if conditions are met
            large_oscillation_condition = jnp.logical_and(
                betaMin < 0.01,
                betaMin / betaMax < 0.55
            )
            dpsiInv = jnp.where(large_oscillation_condition, dpsiInv / 4, dpsiInv)
            
            # Sample 4 points per oscillation
            deltaF_twospin = XLALSimPhenomUtilsMftoHz(dpsiInv / 4.0, pWF['Mtot'])
            
            # Check if two-spin deltaF is smaller and valid
            use_twospin_condition = jnp.logical_and(
                deltaF_twospin < deltaF_alpha,
                jnp.logical_not(jnp.isnan(dpsi))
            )
            
            # If we should use two-spin deltaF, apply hard limit and return
            # Otherwise, fall through to use deltaF_alpha
            return jnp.where(
                use_twospin_condition,
                jnp.maximum(deltaF_twospin, 1e-2),  # hard limit
                jnp.maximum(deltaF_alpha, 1e-2)    # fall back to alpha with hard limit
            )
        
        def no_two_spin():
            # Apply hard limit of 0.01 to deltaF_alpha
            return jnp.maximum(deltaF_alpha, 1e-2)
        
        return jax.lax.cond(two_spin_condition, handle_two_spin, no_two_spin)
    
    return jax.lax.cond(aligned_spin_condition, aligned_spin_case, precessing_case)



def IMRPhenomX_PNR_CheckTwoSpin(pPrec: dict) -> bool:
    """
    Check for two-spin system conditions
    
    Args:
        pPrec: PhenomX precession struct (dict)
        
    Returns:
        bool: True if two-spin conditions are met, False otherwise
    """
    
    # Ensure we have:
    # - non-zero spin on the primary
    # - non-trivial spin on the secondary  
    # - activated the MSA angles
    condition = jnp.logical_and(
        jnp.logical_and(
            pPrec['chi1_norm'] != 0.0,
            pPrec['chi2_norm'] >= 1.0e-3
        ),
        pPrec['IMRPhenomXPrecVersion'] >= 200
    )
    
    return condition

def XLALSimPhenomUtilsHztoMf(fHz: float, Mtot: float) -> float:
    """Stub: Convert Hz to dimensionless frequency"""
    # Implementation needed - likely similar to XLALSimIMRPhenomXUtilsHztoMf
    return fHz * (MTSUN * Mtot)

def XLALSimPhenomUtilsMftoHz(Mf: float, Mtot: float) -> float:
    return Mf / (MTSUN * Mtot)



def XLALSimIMRPhenomXLPNAnsatz(v: float, LNorm: float, L0: float, L1: float, L2: float, 
                               L3: float, L4: float, L5: float, L6: float, L7: float, 
                               L8: float, L8L: float) -> float:
    """
    Compute orbital angular momentum using post-Newtonian expansion
    
    Args:
        v: Input velocity (float)
        LNorm: Orbital angular momentum normalization (float)
        L0: Newtonian orbital angular momentum (float)
        L1: 0.5PN Orbital angular momentum (float)
        L2: 1.0PN Orbital angular momentum (float)
        L3: 1.5PN Orbital angular momentum (float)
        L4: 2.0PN Orbital angular momentum (float)
        L5: 2.5PN Orbital angular momentum (float)
        L6: 3.0PN Orbital angular momentum (float)
        L7: 3.5PN Orbital angular momentum (float)
        L8: 4.0PN Orbital angular momentum (float)
        L8L: 4.0PN logarithmic orbital angular momentum term (float)
        
    Returns:
        float: Orbital angular momentum
    """
    
    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)
    
    # Here LN is the Newtonian pre-factor: LN = \eta / \sqrt{x} :
    # L = L_N \sum_a L_a x^{a/2}
    #   = L_N [ L0 + L1 x^{1/2} + L2 x^{2/2} + L3 x^{3/2} + ... ]
    
    return LNorm * (L0 + L1*sqx + L2*x + L3*(x*sqx) + L4*x2 + L5*(x2*sqx) +
                    L6*x3 + L7*(x3*sqx) + L8*x4 + L8L*x4*jnp.log(x))


def evaluate_chiperp(m1, m2, chi1x, chi1y, chi2x, chi2y, chi_p, IMRPhenomXPrecVersion):
    """
    Evaluate chiperp and chiperp_antisymmetric based on LALSimIMRPhenomX_PNR_internals.c logic.
    JAX-compatible implementation.

    Parameters:
    -----------
    m1 : float
        Mass of first body
    m2 : float
        Mass of second body
    chi1x : float
        x-component of first spin
    chi1y : float
        y-component of first spin
    chi2x : float
        x-component of second spin
    chi2y : float
        y-component of second spin
    chi_p : float
        Precession spin parameter
    IMRPhenomXPrecVersion : int
        Version number (e.g., 330)

    Returns:
    --------
    chiperp : float
        Perpendicular effective spin
    chiperp_antisymmetric : float
        Antisymmetric perpendicular effective spin
    """

    # Calculate mass ratio q = m1/m2 (assuming m1 >= m2)
    q = m1 / m2

    # Calculate symmetric chiperp (chis)
    chis = jnp.sqrt(
        (m1 * m1 * chi1x + m2 * m2 * chi2x) ** 2 +
        (m1 * m1 * chi1y + m2 * m2 * chi2y) ** 2
    ) / (m1 * m1)

    # For version 330, use chis directly
    is_version_330 = (IMRPhenomXPrecVersion == 330)

    # Calculate q-dependent weighting terms
    sin_term = jnp.sin((q - 1.0) * jnp.pi)
    cos_term = jnp.cos((q - 1.0) * jnp.pi)
    chiperp_q_weighted = sin_term * sin_term * chi_p + cos_term * cos_term * chis

    # Use chi_p for q > 1.5, otherwise use weighted version
    chiperp_low_q = jnp.where(q <= 1.5, chiperp_q_weighted, chi_p)

    # Final chiperp: version 330 uses chis, others use chiperp_low_q
    chiperp = jnp.where(is_version_330, chis, chiperp_low_q)

    # Calculate antisymmetric chiperp
    antisymmetric_chis = jnp.sqrt(
        (m1 * m1 * chi1x - m2 * m2 * chi2x) ** 2 +
        (m1 * m1 * chi1y - m2 * m2 * chi2y) ** 2
    ) / (m1 * m1)

    chiperp_antisymmetric_weighted = sin_term * sin_term * chi_p + cos_term * cos_term * antisymmetric_chis
    chiperp_antisymmetric = jnp.where(q <= 1.5, chiperp_antisymmetric_weighted, chi_p)

    return chiperp, chiperp_antisymmetric


def IMRPhenomX_PNR_CoprecWindow(pWF: dict) -> float:
    """
    Placeholder for coprecessing window function.

    Args:
        pWF: PhenomX waveform struct (dict)

    Returns:
        float: Window value for PNR coprecessing tuning
    """
    # This is a stub - implement the actual window function as needed
    return 1.0


def IMRPhenomX_PNR_GetAndSetCoPrecParams(pPrec, pWF: dict, lalParams: dict):
    """
    Get and set coprecessing parameters for PhenomXPNR model.

    This function configures the coprecessing frame tuning parameters used in the
    PhenomXPNR model. It handles toggles for outputting coprecessing models, applying
    PNR tunings, and setting deviation parameters.

    Args:
        pWF: PhenomX waveform struct (dict)
        pPrec: PhenomX precession struct (IMRPhenomXGetAndSetPrecessionVariables instance)
        lalParams: LAL parameter dictionary containing user-specified options (dict)

    Returns:
        int: Status (0 for success)

    Note:
        - Sets various deviation parameters (MU1-4, NU0-6, ZETA1-2) that modify
          coprecessing frame waveform properties
        - Handles different versions (e.g., version 330) with specific calibration limits
    """

    status = 0

    # Get toggle for outputting coprecessing model from LAL dictionary
    IMRPhenomXReturnCoPrec = False #lalParams.get('IMRPhenomXReturnCoPrec', False)
    object.__setattr__(pPrec, 'IMRPhenomXReturnCoPrec', IMRPhenomXReturnCoPrec)

    # Get toggle for PNR coprecessing tuning
    PNRUseTunedCoprec = False#lalParams.get('PNRUseTunedCoprec', False)
    pWF['IMRPhenomXPNRUseTunedCoprec'] = PNRUseTunedCoprec
    object.__setattr__(pPrec, 'IMRPhenomXPNRUseTunedCoprec', PNRUseTunedCoprec)

    # Same as above but for l=m=3
    PNRUseTunedCoprec33 = False #lalParams.get('PNRUseTunedCoprec33', False) and PNRUseTunedCoprec
    object.__setattr__(pPrec, 'IMRPhenomXPNRUseTunedCoprec33', PNRUseTunedCoprec33)
    pWF['IMRPhenomXPNRUseTunedCoprec33'] = PNRUseTunedCoprec33

    # Throw error if l=|m|=3 tuning is requested (not supported)
    if PNRUseTunedCoprec33:
        raise ValueError("Error: Coprecessing tuning for l=|m|=3 must be off.")

    # Get toggle for enforced use of non-precessing spin
    PNRUseInputCoprecDeviations = False #lalParams.get('PNRUseInputCoprecDeviations', False)
    object.__setattr__(pPrec, 'IMRPhenomXPNRUseInputCoprecDeviations', PNRUseInputCoprecDeviations)

    # Get toggle for forcing inspiral phase alignment
    PNRForceXHMAlignment = False
    pWF['IMRPhenomXPNRForceXHMAlignment'] = False

    # Validate PNR usage options
    if PNRUseInputCoprecDeviations and PNRUseTunedCoprec:
        raise ValueError(
            "Error: PNRUseTunedCoprec and PNRUseInputCoprecDeviations must not be enabled simultaneously."
        )

    # Define high-level toggle for whether to apply deviations
    APPLY_PNR_DEVIATIONS = PNRUseTunedCoprec or PNRUseInputCoprecDeviations
    pWF['APPLY_PNR_DEVIATIONS'] = APPLY_PNR_DEVIATIONS 

    # If applying PNR deviations with XHM alignment, set phase alignment params
    # (This part is commented out as it requires additional functions not yet implemented)
    if APPLY_PNR_DEVIATIONS and PNRForceXHMAlignment:
         IMRPhenomX_PNR_SetPhaseAlignmentParams(pWF, pPrec)

    # Define single spin parameters for fit evaluation
    a1 = pPrec.chi_singleSpin
    pWF['a1'] = a1
    cos_theta = pPrec.costheta_singleSpin

    # Compute opening angle
    theta_LS = jnp.arccos(cos_theta)
    pWF['theta_LS'] = theta_LS

    # Compute window of tuning deviations
    pnr_window = 0.0  # Default: tunings off
    if PNRUseTunedCoprec:
        pnr_window = IMRPhenomX_PNR_CoprecWindow(pWF)
    pWF['pnr_window'] = pnr_window

    # Store XCP deviation parameter
    PNR_DEV_PARAMETER = a1 * jnp.sin(theta_LS) * APPLY_PNR_DEVIATIONS
    if PNRUseTunedCoprec:
        PNR_DEV_PARAMETER = pnr_window * PNR_DEV_PARAMETER
    pWF['PNR_DEV_PARAMETER'] = PNR_DEV_PARAMETER

    # Get deviations from lalParams (used for PNRUseInputCoprecDeviations)
    # All default values are zero
    pWF['MU1'] = lalParams.get('CPMু1', 0.0)
    pWF['MU2'] = lalParams.get('CPMU2', 0.0)
    pWF['MU3'] = lalParams.get('CPMU3', 0.0)
    pWF['MU4'] = lalParams.get('CPMU4', 0.0)
    pWF['NU0'] = lalParams.get('CPNU0', 0.0)
    pWF['NU4'] = lalParams.get('CPNU4', 0.0)
    pWF['NU5'] = lalParams.get('CPNU5', 0.0)
    pWF['NU6'] = lalParams.get('CPNU6', 0.0)
    pWF['ZETA1'] = lalParams.get('CPZETA1', 0.0)
    pWF['ZETA2'] = lalParams.get('CPZETA2', 0.0)

    # If using tuned coprecessing parameters, get them from model fits
    if PNRUseTunedCoprec:
        # Apply flattening for extrapolation outside calibration region
        is_version_330 = (pPrec.IMRPhenomXPrecVersion == 330)

        # Flatten mass-ratio dependence
        coprec_eta = jnp.where(
            pWF['eta'] >= 0.09876,
            pWF['eta'],
            jnp.where(
                is_version_330,
                0.09876 - (0.09876 - pWF['eta']) * 0.1641,
                0.09876
            )
        )

        # Flatten spin dependence
        coprec_a1 = jnp.where(pWF['a1'] <= 0.8, pWF['a1'], 0.8 + (pWF['a1'] - 0.8) / 12.0)
        coprec_a1 = jnp.where(
            is_version_330,
            coprec_a1,
            jnp.where(pWF['a1'] <= 0.8, pWF['a1'], 0.8)
        )
        coprec_a1 = jnp.where(coprec_a1 >= 0.2, coprec_a1, 0.2)

        # These functions need to be implemented separately - using placeholders for now
        # In practice, these would be calls to fit functions from the coprecessing model

        # Placeholder: These should be actual fit functions
        # pWF['MU1'] = XLALSimIMRPhenomXCP_MU1_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['MU2'] = XLALSimIMRPhenomXCP_MU2_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['MU3'] = XLALSimIMRPhenomXCP_MU3_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['NU0'] = XLALSimIMRPhenomXCP_NU0_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['NU4'] = XLALSimIMRPhenomXCP_NU4_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['NU5'] = XLALSimIMRPhenomXCP_NU5_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['NU6'] = XLALSimIMRPhenomXCP_NU6_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['ZETA1'] = XLALSimIMRPhenomXCP_ZETA1_l2m2(theta_LS, coprec_eta, coprec_a1)
        # pWF['ZETA2'] = XLALSimIMRPhenomXCP_ZETA2_l2m2(theta_LS, coprec_eta, coprec_a1)

        # For now, set to zero (would be computed from fits in full implementation)
        pass

    # Override NU0 (as in original C code)
    pWF['NU0'] = 0.0

    return pPrec




def IMRPhenomX_PNR_SetPhaseAlignmentParams(x, y):
    #TODO

    return None