import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
from ..constants import EULERGAMMA, MSUN, G, C
from .elliptic_integrals import ellint_F
from .elliptic_integrals import gsl_sf_elljac_e


# /** This function initializes all the core variables required for the MSA system. This will be called first. */
def IMRPhenomX_Initialize_MSA_System(
    mass_1: Float,
    mass_2: Float,
    chi1x: Float,
    chi1y: Float,
    chi1z: Float,
    chi2x: Float,
    chi2y: Float,
    chi2z: Float,
    reference_frequency: float,
    pflag: int = 223,
    expansion_order: int = 5,
) -> Array:
    """
    First initialize the system of variables needed for Chatziioannou et al, PRD, 88, 063011, (2013), arXiv:1307.4418:

      - Racine et al, PRD, 80, 044010, (2009), arXiv:0812.4413
      - Favata, PRD, 80, 024002, (2009), arXiv:0812.0069
      - Blanchet et al, PRD, 84, 064041, (2011), arXiv:1104.5659
      - Bohe et al, CQG, 30, 135009, (2013), arXiv:1303.7412
    """
    # Sanity check on the precession version

    eta = mass_1 * mass_2 / jnp.power(mass_1 + mass_2, 2)

    eta2 = jnp.power(eta, 2)
    eta3 = jnp.power(eta, 3)
    eta4 = jnp.power(eta, 4)
    inveta = jnp.power(eta, -1)

    # PN Coefficients for d \omega / d t as per LALSimInspiralFDPrecAngles_internals.c
    LAL_LN2 = jnp.log(2.0)
    domegadt_constants_NS = [
        96.0 / 5.0,
        -1486.0 / 35.0,
        -264.0 / 5.0,
        384.0 * jnp.pi / 5.0,
        34103.0 / 945.0,
        13661.0 / 105.0,
        944.0 / 15.0,
        jnp.pi * (-4159.0 / 35.0),
        jnp.pi * (-2268.0 / 5.0),
        (
            16447322263.0 / 7276500.0
            + jnp.pi * jnp.pi * 512.0 / 5.0
            - LAL_LN2 * 109568.0 / 175.0
            - EULERGAMMA * 54784.0 / 175.0
        ),
        (-56198689.0 / 11340.0 + jnp.pi * jnp.pi * 902.0 / 5.0),
        1623.0 / 140.0,
        -1121.0 / 27.0,
        -54784.0 / 525.0,
        -jnp.pi * 883.0 / 42.0,
        jnp.pi * 71735.0 / 63.0,
        jnp.pi * 73196.0 / 63.0,
    ]

    domegadt_constants_SO = [
        -904.0 / 5.0,
        -120.0,
        -62638.0 / 105.0,
        4636.0 / 5.0,
        -6472.0 / 35.0,
        3372.0 / 5.0,
        -jnp.pi * 720.0,
        -jnp.pi * 2416.0 / 5.0,
        -208520.0 / 63.0,
        796069.0 / 105.0,
        -100019.0 / 45.0,
        -1195759.0 / 945.0,
        514046.0 / 105.0,
        -8709.0 / 5.0,
        -jnp.pi * 307708.0 / 105.0,
        jnp.pi * 44011.0 / 7.0,
        -jnp.pi * 7992.0 / 7.0,
        jnp.pi * 151449.0 / 35.0,
    ]

    domegadt_constants_SS = [-494.0 / 5.0, -1442.0 / 5.0, -233.0 / 5.0, -719.0 / 5.0]

    # Note that Chatziioannou et al use q = m2/m1, where m1 > m2 and therefore q < 1.
    # IMRPhenomX assumes m1 > m2 and q > 1. For the internal MSA code, flip q and
    # dump this to pPrec->qq, where qq explicitly denotes that this is 0 < q < 1.

    q = mass_2 / mass_1  # m2 / m1, q < 1, m1 > m2

    #    /* \delta and powers of \delta in terms of q < 1, should just be m1 - m2 */
    delta_qq = (1.0 - q) / (1.0 + q)

    # Initialize empty vectors (using dictionaries to represent vectors)

    # Define source frame such that \hat{L} = {0,0,1} with L_z pointing along \hat{z}
    Lhat = jnp.array([0.0, 0.0, 1.0])

    # Dimensionful spin vectors, note eta = m1 * m2 and q = m2/m1

    S1v = jnp.array([chi1x * eta / q, chi1y * eta / q, chi1z * eta / q])

    S2v = jnp.array([chi2x * eta * q, chi2y * eta * q, chi2z * eta * q])

    S1_0_norm = IMRPhenomX_vector_L2_norm(S1v)
    S2_0_norm = IMRPhenomX_vector_L2_norm(S2v)

    mass_1_SI = mass_1 * MSUN
    mass_2_SI = mass_2 * MSUN

    piGM = jnp.pi * (mass_1_SI + mass_2_SI) * (G / C) / (C * C)

    # Reference velocity v and v^2
    v_0 = jnp.power(piGM * reference_frequency, 1.0 / 3.0)
    v_0_2 = v_0 * v_0

    # Reference orbital angular momenta
    L_0 = IMRPhenomX_vector_scalar(Lhat, eta / v_0)

    # Inner products used in MSA system
    dotS1L = IMRPhenomX_vector_dot_product(S1v, Lhat)
    dotS2L = IMRPhenomX_vector_dot_product(S2v, Lhat)
    dotS1S2 = IMRPhenomX_vector_dot_product(S1v, S2v)
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm

    # Coefficients for PN orbital angular momentum at 3PN, as per LALSimInspiralFDPrecAngles_internals.c

    constants_L_0, constants_L_1, constants_L_2, constants_L_3, constants_L_4 = (
        compute_constants_L(eta, dotS1L, dotS2L, q)
    )

    # Effective total spin
    Seff = (1.0 + q) * dotS1L + (1 + (1.0 / q)) * dotS2L

    # Line 2347
    # Initial total spin, S = S1 + S2
    S0 = IMRPhenomX_vector_sum(S1v, S2v)

    #    /* Initial total angular momentum, J = L + S1 + S2 */
    J_0 = IMRPhenomX_vector_sum(L_0, S0)

    # Norm of total initial spin
    S_0_norm = IMRPhenomX_vector_L2_norm(S0)

    # Norm of orbital and total angular momenta
    L_0_norm = IMRPhenomX_vector_L2_norm(L_0)
    J_0_norm = IMRPhenomX_vector_L2_norm(J_0)

    L0norm = L_0_norm
    J0norm = J_0_norm

    S1_norm_2, S2_norm_2 = compute_spin_norm_squared(
        chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, mass_1, mass_2
    )

    vRoots = IMRPhenomX_Return_Roots_MSA(
        L_0_norm,
        J_0_norm,
        S1_norm_2,
        S2_norm_2,
        q,
        eta,
        delta_qq,
        Seff,
        dotS1Ln,
        dotS2Ln,
        S_0_norm,
    )

    # Line 2500
    Spl2 = vRoots[2]
    Smi2 = vRoots[1]
    S32 = vRoots[0]

    Spl2pSmi2 = Spl2 + Smi2
    Spl2mSmi2 = Spl2 - Smi2

    # S_+ and S_-
    Spl = jnp.sqrt(Spl2)

    # Eq. 45 of PRD 95, 104004, (2017), arXiv:1703.03967, set from initial conditions
    SAv2 = 0.5 * (Spl2pSmi2)
    SAv = jnp.sqrt(SAv2)
    invSAv2 = 1.0 / SAv2
    invSAv = 1.0 / SAv

    # c_1 is determined by Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967
    c_1 = 0.5 * (J0norm * J0norm - L0norm * L0norm - SAv2) / L_0_norm * eta
    c1_2 = c_1 * c_1

    # Useful powers and combinations of c_1
    c1 = c_1
    c1_over_eta = c_1 / eta
    c_1_over_eta = c_1 / eta

    # Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967
    # omqsq = (1.0 - q) * (1.0 - q) + 1e-16
    omq2 = 1.0 - q * q

    # Precession averaged spin couplings, Eq. A9 - A14 of arXiv:1703.03967, note that we only use the initial values
    # Guard against equal mass (omq2=0): XLA constant-folds 1-1=0 and drops 1e-16 epsilon under JIT,
    # so use jnp.where to safely return 0 at exact q=1.
    omq2_safe = jnp.where(omq2 == 0.0, jnp.ones_like(omq2), omq2)
    S1L_pav = jnp.where(
        omq2 == 0.0,
        jnp.zeros_like(omq2),
        (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2_safe),
    )
    S2L_pav = jnp.where(
        omq2 == 0.0,
        jnp.zeros_like(omq2),
        -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2_safe),
    )
    # S1S2_pav = 0.5 * SAv2 - 0.5 * (S1_norm_2 + S2_norm_2)
    # S1Lsq_pav = (S1L_pav*S1L_pav + ((Spl2mSmi2)*(Spl2mSmi2) * v_0_2) / (32.0 * eta2 * omqsq))
    # S2Lsq_pav = (S2L_pav*S2L_pav + (q*q*(Spl2mSmi2)*(Spl2mSmi2) * v_0_2) / (32.0 * eta2 * omqsq))
    # S1LS2L_pav = (S1L_pav*S2L_pav - q * (Spl2mSmi2)*(Spl2mSmi2)*v_0_2 / (32.0 * eta2 * omqsq))

    """
    # Spin couplings in arXiv:1703.03967
    beta3 = (((113./12.) + (25./4.)*(m2/m1)) * S1L_pav + ((113./12.) + (25./4.)*(m1/m2)) * S2L_pav)
    beta5 = (((31319./1008.) - (1159./24.)*eta) + (m2/m1)*((809./84) - (281./8.)*eta)) * S1L_pav + \
            (((31319./1008.) - (1159./24.)*eta) + (m1/m2)*((809./84) - (281./8.)*eta)) * S2L_pav
    beta6 = jnp.pi * (((75./2.) + (151./6.)*(m2/m1))*S1L_pav + ((75./2.) + (151./6.)*(m1/m2))*S2L_pav)
    beta7 = (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) + \
            (m2/m1)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * S1L_pav + \
            (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) + \
            (m1/m2)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * S2L_pav
    sigma4 = ((1.0/mu) * ((247./48.)*S1S2_pav - (721./48.)*S1L_pav*S2L_pav) + \
            (1.0/(m1*m1)) * ((233./96.)*S1_norm_2 - (719./96.)*S1Lsq_pav) + \
            (1.0/(m2*m2)) * ((233./96.)*S2_norm_2 - (719./96.)*S2Lsq_pav))
    """
    # Line 2597

    a0 = eta * domegadt_constants_NS[0]
    a2 = eta * (domegadt_constants_NS[1] + eta * (domegadt_constants_NS[2]))
    a3 = eta * (
        domegadt_constants_NS[3]
        + IMRPhenomX_Get_PN_beta(
            domegadt_constants_SO[0], domegadt_constants_SO[1], dotS1L, dotS2L, q
        )
    )
    a4 = eta * (
        domegadt_constants_NS[4]
        + eta * (domegadt_constants_NS[5] + eta * (domegadt_constants_NS[6]))
        + IMRPhenomX_Get_PN_sigma(
            domegadt_constants_SS[0],
            domegadt_constants_SS[1],
            inveta,
            dotS1S2,
            dotS1L,
            dotS2L,
        )
        + IMRPhenomX_Get_PN_tau(
            domegadt_constants_SS[2],
            domegadt_constants_SS[3],
            q,
            S1_norm_2,
            S2_norm_2,
            dotS1L,
            dotS2L,
            eta,
        )
    )
    a5 = eta * (
        domegadt_constants_NS[7]
        + eta * (domegadt_constants_NS[8])
        + IMRPhenomX_Get_PN_beta(
            (domegadt_constants_SO[2] + eta * (domegadt_constants_SO[3])),
            (domegadt_constants_SO[4] + eta * (domegadt_constants_SO[5])),
            dotS1L,
            dotS2L,
            q,
        )
    )

    # Useful powers of a_0
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a2_2 = a2 * a2

    # Calculate g coefficients as in Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    # These constants are used in TaylorT2 where domega/dt is expressed as an inverse polynomial
    g0 = 1 / a0
    g2 = -(a2 / a0_2)
    g3 = -(a3 / a0_2)
    g4 = -(a4 * a0 - a2_2) / a0_3
    g5 = -(a5 * a0 - 2.0 * a3 * a2) / a0_3

    # Useful powers of delta
    delta = delta_qq
    delta2 = delta * delta
    # delta3 = delta * delta2
    # delta4 = delta * delta3

    # \psi_1 is defined in Eq. C1 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967
    # Guard: for equal-mass delta2=0, psi1→±inf but delta_qq*psi1→0; set psi1=0 at limit.
    _delta2_safe = jnp.where(delta2 > 0.0, delta2, jnp.ones_like(delta2))
    psi1 = jnp.where(
        delta2 > 0.0, 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * _delta2_safe), 0.0
    )

    c_1_over_nu = c1_over_eta
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q) * (1.0 + q)
    Seff_2 = Seff * Seff
    q_2 = q * q
    one_m_q_sq = (1.0 - q) * (1.0 - q)
    one_m_q2_2 = (1.0 - q_2) * (1.0 - q_2)
    one_m_q_4 = one_m_q_sq * one_m_q_sq

    # This implements the Delta term as in LALSimInspiralFDPrecAngles.c
    # c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L145
    # if pflag == 222 or pflag == 223:
    Del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
    Del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * Seff
    Del3 = 4.0 * (one_m_q2_2 * S1_norm_2 - q_2 * Seff_2)
    Del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    Del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * Seff
    Del6 = 4.0 * (one_m_q2_2 * S2_norm_2 - q_2 * Seff_2)
    Delta = jnp.sqrt(jnp.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))

    """
    else:
        # Coefficients of \\Delta as defined in Eq. C3 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967.
        term1 = c1_2 * eta / (q * delta4)
        term2 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / (q * delta4)
        term3 = -eta2 * (delta2 * S1_norm_2 - eta2 * Seff_2) / delta4

        # Is this 1) (c1_2 * q * eta / delta4) or 2) c1_2*eta2/delta4?
        # - In paper.pdf, the expression 1) is used.
        # Using eta^2 leads to higher frequency oscillations, use q * eta
        term4 = c1_2 * eta * q / delta4
        term5 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / delta4
        term6 = -eta2 * (delta2 * S2_norm_2 - eta2 * Seff_2) / delta4
        Delta = jnp.sqrt(jnp.abs((term1 + term2 + term3) * (term4 + term5 + term6)))
    """
    # Line 2706

    # if pflag == 222 or pflag == 223:
    u1 = 3.0 * g2 / g0
    # Guard: for equal-mass one_m_q_4=0, u2→+inf but delta_qq*u2→0; set u2=0 at limit.
    _one_m_q_4_safe = jnp.where(one_m_q_4 > 0.0, one_m_q_4, jnp.ones_like(one_m_q_4))
    u2 = jnp.where(one_m_q_4 > 0.0, 0.75 * one_p_q_sq / _one_m_q_4_safe, 0.0)
    u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    u4 = (
        2.0
        * one_m_q2_2
        * (q * (2.0 + q) * S1_norm_2 + (1.0 + 2.0 * q) * S2_norm_2 - 2.0 * q * SAv2)
    )
    u5 = 2.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * 2.0 * c_1_over_nu * Seff
    u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
    u7 = q * Delta

    # Eq. C2 (1703.03967)
    psi2 = u1 + u2 * (u3 + u4 + u5 - u6 + u7)
    """
    else:
        # \\psi_2 is defined in Eq. C2 of Appendix C in PRD, 95, 104004, (2017). Here we implement system of equations as in paper.pdf
        term1 = 3.0 * g2 / g0

        # q^2 or no q^2 in term2? Consensus on retaining q^2 term: https://git.ligo.org/waveforms/reviews/phenompv3hm/issues/7
        term2 = 3.0 * q * q / (2.0 * eta3)
        term3 = 2.0 * Delta
        term4 = -2.0 * eta2 * SAv2 / delta2
        term5 = -10.0 * eta * c1_2 / delta4
        term6 = 2.0 * eta2 * (7.0 + 6.0 * q + 7.0 * q * q) * c_1 * Seff / (omqsq * delta2)
        term7 = -eta3 * (3.0 + 4.0 * q + 3.0 * q * q) * Seff_2 / (omqsq * delta2)
        term8 = eta * (q * (2.0 + q) * S1_norm_2 + (1.0 + 2.0 * q) * S2_norm_2) / omqsq

        # \\psi_2, C2 of Appendix C of PRD, 95, 104004, (2017)
        psi2 = term1 + term2 * (term3 + term4 + term5 + term6 + term7 + term8)
    """
    # Eq. D1 of PRD, 95, 104004, (2017), arXiv:1703.03967
    Rm = Spl2 - Smi2
    Rm_2 = Rm * Rm

    # Eq. D2 and D3 Appendix D of PRD, 95, 104004, (2017), arXiv:1703.03967
    cp = Spl2 * eta2 - c1_2
    cm = Smi2 * eta2 - c1_2

    # jnp.abs is here to help enforce positive definite cpcm
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)

    # Eq. D4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note difference to published version.
    a1dD = 0.5 + 0.75 / eta

    # Eq. D5 in PRD, 95, 104004, (2017), arXiv:1703.03967
    a2dD = -0.75 * Seff / eta

    # Eq. E3 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm * D2
    D2RmSq = (cp - sqrt_cpcm) / eta2

    # Eq. E4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm^2 * D4
    D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)

    S0m = S1_norm_2 - S2_norm_2

    # Difference of spin norms squared, as used in Eq. D6 of PRD, 95, 104004, (2017), arXiv:1703.03967
    aw = (
        -3.0
        * (1.0 + q)
        / q
        * (
            2.0 * (1.0 + q) * eta2 * Seff * c_1
            - (1.0 + q) * c1_2
            + (1.0 - q) * eta2 * S0m
        )
    )
    cw = 3.0 / 32.0 / eta * Rm_2
    dw = 4.0 * cp - 4.0 * D2RmSq * eta2
    hw = -2.0 * (2.0 * D2RmSq - Rm) * c_1
    fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2

    adD = aw / dw
    hdD = hw / dw
    cdD = cw / dw
    fdD = fw / dw

    gw = 3.0 / 16.0 / eta2 / eta * Rm_2 * (c_1 - eta2 * Seff)
    gdD = gw / dw

    # Useful powers of the coefficients
    hdD_2 = hdD * hdD
    adDfdD = adD * fdD
    adDfdDhdD = adDfdD * hdD
    adDhdD_2 = adD * hdD_2

    # Line 2800

    # Eq. D10 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz0 = a1dD + adD

    # Eq. D11 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz1 = a2dD - adD * Seff - adD * hdD

    # Eq. D12 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz2 = adD * hdD * Seff + cdD - adD * fdD + adD * hdD_2

    # Eq. D13 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz3 = (adDfdD - cdD - adDhdD_2) * (Seff + hdD) + adDfdDhdD

    # Eq. D14 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz4 = (cdD + adDhdD_2 - 2.0 * adDfdD) * (
        hdD * Seff + hdD_2 - fdD
    ) - adD * fdD * fdD

    # Eq. D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz5 = (
        (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0 * hdD)
        - (cdD + adDhdD_2 - 2.0 * adDfdD) * hdD_2 * (Seff + hdD)
        - adDfdD * fdD * hdD
    )

    # Coefficients of Eq. 65, as defined in Equations D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967
    Omegaz0_coeff = 3.0 * g0 * Omegaz0
    Omegaz1_coeff = 3.0 * g0 * Omegaz1
    Omegaz2_coeff = 3.0 * (g0 * Omegaz2 + g2 * Omegaz0)
    Omegaz3_coeff = 3.0 * (g0 * Omegaz3 + g2 * Omegaz1 + g3 * Omegaz0)
    Omegaz4_coeff = 3.0 * (g0 * Omegaz4 + g2 * Omegaz2 + g3 * Omegaz1 + g4 * Omegaz0)
    Omegaz5_coeff = 3.0 * (
        g0 * Omegaz5 + g2 * Omegaz3 + g3 * Omegaz2 + g4 * Omegaz1 + g5 * Omegaz0
    )

    # Coefficients of zeta: in Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967
    c1oveta2 = c_1 / eta2
    Omegazeta0 = Omegaz0
    Omegazeta1 = Omegaz1 + Omegaz0 * c1oveta2
    Omegazeta2 = Omegaz2 + Omegaz1 * c1oveta2
    Omegazeta3 = Omegaz3 + Omegaz2 * c1oveta2 + gdD
    Omegazeta4 = Omegaz4 + Omegaz3 * c1oveta2 - gdD * Seff - gdD * hdD
    Omegazeta5 = Omegaz5 + Omegaz4 * c1oveta2 + gdD * hdD * Seff + gdD * (hdD_2 - fdD)

    Omegazeta0_coeff = -g0 * Omegazeta0
    Omegazeta1_coeff = -1.5 * g0 * Omegazeta1
    Omegazeta2_coeff = -3.0 * (g0 * Omegazeta2 + g2 * Omegazeta0)
    Omegazeta3_coeff = 3.0 * (g0 * Omegazeta3 + g2 * Omegazeta1 + g3 * Omegazeta0)
    Omegazeta4_coeff = 3.0 * (
        g0 * Omegazeta4 + g2 * Omegazeta2 + g3 * Omegazeta1 + g4 * Omegazeta0
    )
    Omegazeta5_coeff = 1.5 * (
        g0 * Omegazeta5
        + g2 * Omegazeta3
        + g3 * Omegazeta2
        + g4 * Omegazeta1
        + g5 * Omegazeta0
    )

    # LAL's default PhenomXPExpansionOrder=5 truncates the 5th-order correction
    # coefficients before phiz_0/zeta_0 are initialized.
    if expansion_order == -1:
        pass
    elif expansion_order == 1:
        Omegaz1_coeff = 0.0
        Omegazeta1_coeff = 0.0
        Omegaz2_coeff = 0.0
        Omegazeta2_coeff = 0.0
        Omegaz3_coeff = 0.0
        Omegazeta3_coeff = 0.0
        Omegaz4_coeff = 0.0
        Omegazeta4_coeff = 0.0
        Omegaz5_coeff = 0.0
        Omegazeta5_coeff = 0.0
    elif expansion_order == 2:
        Omegaz2_coeff = 0.0
        Omegazeta2_coeff = 0.0
        Omegaz3_coeff = 0.0
        Omegazeta3_coeff = 0.0
        Omegaz4_coeff = 0.0
        Omegazeta4_coeff = 0.0
        Omegaz5_coeff = 0.0
        Omegazeta5_coeff = 0.0
    elif expansion_order == 3:
        Omegaz3_coeff = 0.0
        Omegazeta3_coeff = 0.0
        Omegaz4_coeff = 0.0
        Omegazeta4_coeff = 0.0
        Omegaz5_coeff = 0.0
        Omegazeta5_coeff = 0.0
    elif expansion_order == 4:
        Omegaz4_coeff = 0.0
        Omegazeta4_coeff = 0.0
        Omegaz5_coeff = 0.0
        Omegazeta5_coeff = 0.0
    elif expansion_order == 5:
        Omegaz5_coeff = 0.0
        Omegazeta5_coeff = 0.0
    else:
        raise ValueError(
            f"Expansion order for MSA corrections = {expansion_order} not recognized."
        )

    # Get psi0 term
    psi0 = compute_psi0(
        Smi2, Spl2, S32, S_0_norm, v_0, v_0_2, psi1, psi2, g0, delta_qq, L_0, S1v, S2v
    )

    # Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles
    condition = jnp.abs(Spl2 - Smi2) > 1e-5

    def compute_msa_corrections():
        return IMRPhenomX_Return_MSA_Corrections_MSA(
            v_0,
            L_0_norm,
            J_0_norm,
            Seff,
            eta,
            eta3,
            inveta,
            Spl,
            Spl2,
            Smi2,
            Spl2mSmi2,
            S1_norm_2,
            S2_norm_2,
            S32,
            delta_qq,
            g0,
            psi0,
            psi1,
            psi2,
        )

    def no_msa_corrections():
        return 0.0, 0.0

    vMSA = jax.lax.cond(condition, compute_msa_corrections, no_msa_corrections)

    # Initial \phi_z
    phiz_0_init = 0.0
    phiz_0 = IMRPhenomX_Return_phiz_MSA(
        v_0,
        J_0_norm,
        eta,
        inveta,
        eta2,
        eta4,
        c1,
        SAv,
        SAv2,
        invSAv,
        invSAv2,
        Omegaz0_coeff,
        Omegaz1_coeff,
        Omegaz2_coeff,
        Omegaz3_coeff,
        Omegaz4_coeff,
        Omegaz5_coeff,
        phiz_0_init,
    )

    # Initial \zeta
    zeta_0_init = 0.0
    zeta_0 = IMRPhenomX_Return_zeta_MSA(
        v_0,
        eta,
        Omegazeta0_coeff,
        Omegazeta1_coeff,
        Omegazeta2_coeff,
        Omegazeta3_coeff,
        Omegazeta4_coeff,
        Omegazeta5_coeff,
        zeta_0_init,
    )

    vMSA_phiz, vMSA_zeta = vMSA
    phiz_0 = -phiz_0 - vMSA_phiz
    zeta_0 = -zeta_0 - vMSA_zeta

    return jnp.array(
        [
            Omegaz0_coeff,
            Omegaz1_coeff,
            Omegaz2_coeff,
            Omegaz3_coeff,
            Omegaz4_coeff,
            Omegaz5_coeff,
            Omegazeta0_coeff,
            Omegazeta1_coeff,
            Omegazeta2_coeff,
            Omegazeta3_coeff,
            Omegazeta4_coeff,
            Omegazeta5_coeff,
            g0,
            c_1,
            c_1_over_eta,
            SAv2,
            Seff,
            dotS1Ln,
            dotS2Ln,
            S_0_norm,
            psi0,
            psi1,
            psi2,
            phiz_0,
            zeta_0,
            constants_L_0,
            constants_L_1,
            constants_L_2,
            constants_L_3,
            constants_L_4,
            S1_norm_2,
            S2_norm_2,
            S1L_pav,
            S2L_pav,
        ]
    )


def IMRPhenomX_Return_Roots_MSA(
    LNorm: Float,
    JNorm: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    qq: Float,
    eta: Float,
    delta_qq: Float,
    Seff: Float,
    dotS1Ln: Float,
    dotS2Ln: Float,
    S_0_norm: Float,
) -> Float[Array, "3"]:
    """
    Compute roots S32, Smi2, Spl2 for MSA approximation.

    Args:
        LNorm (Float): Normalized orbital angular momentum.
        JNorm (Float): Normalized total angular momentum.
        S1_norm_2 (Float): Spin 1 magnitude squared.
        S2_norm_2 (Float): Spin 2 magnitude squared.
        qq (Float): Mass ratio q = m2/m1.
        eta (Float): Symmetric mass ratio.
        delta_qq (Float): Mass difference parameter (m1-m2)/(m1+m2).
        Seff (Float): Effective spin parameter.
        dotS1Ln (Float): Dot product of S1 with L_hat.
        dotS2Ln (Float): Dot product of S2 with L_hat.
        S_0_norm (Float): Initial total spin magnitude.

    Returns:
        Float[Array, "3"]: Array of [S32, Smi2, Spl2] roots.
    """
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(
        LNorm,
        JNorm,
        S1_norm_2,
        S2_norm_2,
        qq,
        eta,
        delta_qq,
        Seff,
    )
    B, C, D = vBCD

    B2 = B * B
    B3 = B2 * B
    BC = B * C

    p = C - B2 / 3.0
    qc = (2.0 / 27.0) * B3 - BC / 3.0 + D

    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / (p * sqrtarg)
    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)

    vector_condition = jnp.logical_or(jnp.isnan(theta), (jnp.isnan(sqrtarg)))
    scalar_condition = jnp.any(
        jnp.array(
            [
                (dotS1Ln == 1.0),
                (dotS2Ln == 1.0),
                (dotS1Ln == -1.0),
                (dotS2Ln == -1.0),
                (S1_norm_2 == 0.0),
                (S2_norm_2 == 0.0),
            ]
        )
    )

    invalid_case = jnp.logical_or(vector_condition, scalar_condition)

    def roots_when_valid():
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 4.0 * jnp.pi / 3.0) - B / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * jnp.pi / 3.0) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        tmp6 = jnp.where(
            (tmp4 - tmp3 > 0.0) & (tmp5 - tmp3 < 0.0),
            tmp3,
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2),
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return jnp.array([S32, Smi2, Spl2])

    def roots_when_invalid():
        Smi2 = S_0_norm**2 * jnp.ones_like(LNorm)
        Spl2 = Smi2 + 1e-9
        S32 = jnp.zeros_like(LNorm)
        return jnp.array([S32, Smi2, Spl2])

    roots_array = jnp.where(
        jnp.atleast_1d(invalid_case),
        roots_when_invalid(),
        roots_when_valid(),
    )

    return roots_array


def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(
    LNorm: Float,
    JNorm: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    qq: Float,
    eta: Float,
    delta_qq: Float,
    Seff: Float,
):
    """
    Compute spin evolution coefficients B, C, D for MSA approximation.

    Args:
        LNorm (Float): Normalized orbital angular momentum.
        JNorm (Float): Normalized total angular momentum.
        S1_norm_2 (Float): Spin 1 magnitude squared.
        S2_norm_2 (Float): Spin 2 magnitude squared.
        qq (Float): Mass ratio q = m2/m1.
        eta (Float): Symmetric mass ratio.
        delta_qq (Float): Mass difference parameter (m1-m2)/(m1+m2).
        Seff (Float): Effective spin parameter.

    Returns:
        Tuple[float, float, float]: A tuple of (B_coeff, C_coeff, D_coeff) spin
            evolution coefficients for the MSA approximation.
    """
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = S1_norm_2
    S2Norm2 = S2_norm_2
    q = qq
    delta = delta_qq
    deltaSq = delta * delta

    J2mL2 = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = (
        (LNorm2 + S1Norm2) * q
        + 2.0 * LNorm * Seff
        - 2.0 * JNorm2
        - S1Norm2
        - S2Norm2
        + (LNorm2 + S2Norm2) / q
    )

    # C coefficient (Eq. B3)
    C_coeff = (
        J2mL2Sq
        - 2.0 * LNorm * Seff * J2mL2
        - 2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2)
        + 4.0 * eta * LNorm2 * Seff * Seff
        - 2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm
        + 2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2
    )

    # D coefficient (Eq. B4)
    D_coeff = (
        ((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq
        + deltaSq * (S1Norm2 - S2Norm2) ** 2 * LNorm2 / eta
        + 2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2
    )

    return B_coeff, C_coeff, D_coeff


def IMRPhenomX_Get_PN_sigma(
    a: Float,
    b: Float,
    inveta: Float,
    dotS1S2: Float,
    dotS1L: Float,
    dotS2L: Float,
) -> Float:
    """
    Calculate PN sigma coefficient

    Args:
        a: First coefficient (Float)
        b: Second coefficient (Float)
        inveta: Inverse of symmetric mass ratio (Float)
        dotS1S2: Dot product of S1 and S2 (Float)
        dotS1L: Dot product of S1 and L (Float)
        dotS2L: Dot product of S2 and L (Float)

    Returns: Float: PN sigma value
    """
    return inveta * (a * dotS1S2 - b * dotS1L * dotS2L)


def IMRPhenomX_Get_PN_tau(
    a: Float,
    b: Float,
    qq: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    dotS1L: Float,
    dotS2L: Float,
    eta: Float,
) -> Float:
    """
    Internal function to computes PN spin-spin couplings. As in LALSimInspiralFDPrecAngles.c

    Args:
        a: First coefficient (Float)
        b: Second coefficient (Float)
        qq: Mass ratio q = m1/m2 (Float)
        S1_norm_2: Squared norm of spin 1 (Float)
        S2_norm_2: Squared norm of spin 2 (Float)
        dotS1L: Dot product of S1 and L (Float)
        dotS2L: Dot product of S2 and L (Float)
        eta: Symmetric mass ratio (Float)

    Returns: Float: PN tau value
    """
    return (
        qq * ((S1_norm_2 * a) - b * dotS1L * dotS1L)
        + (a * S2_norm_2 - b * dotS2L * dotS2L) / qq
    ) / eta


def IMRPhenomX_Get_PN_beta(
    a: Float,
    b: Float,
    dotS1L: Float,
    dotS2L: Float,
    qq: Float,
) -> Float:
    """
    Calculate PN beta coefficient

    Args:
        a: First coefficient (Float)
        b: Second coefficient (Float)
        dotS1L: Dot product of S1 and L (Float)
        dotS2L: Dot product of S2 and L (Float)
        qq: Mass ratio q = m1/m2 (Float)

    Returns: Float: PN beta value
    """
    return dotS1L * (a + b * qq) + dotS2L * (a + b / qq)


def compute_constants_L(eta, dotS1L, dotS2L, q):
    """
    Compute coefficients for PN orbital angular momentum at 3PN.

    As per LALSimInspiralFDPrecAngles_internals.c

    Args:
        eta: Symmetric mass ratio
        dotS1L: Dot product of S1 and L
        dotS2L: Dot product of S2 and L
        q: Mass ratio m2/m1 (q < 1)

    Returns:
        tuple[Float, Float, Float, Float, Float]: 5 constants [constants_L_0, ..., constants_L_4]
    """
    L_csts_nonspin = [
        3.0 / 2.0,
        1.0 / 6.0,
        27.0 / 8.0,
        -19.0 / 8.0,
        1.0 / 24.0,
        135.0 / 16.0,
        -6889.0 / 144.0 + 41.0 / 24.0 * jnp.pi * jnp.pi,
        31.0 / 24.0,
        7.0 / 1296.0,
    ]

    L_csts_spinorbit = [
        -14.0 / 6.0,
        -3.0 / 2.0,
        -11.0 / 2.0,
        133.0 / 72.0,
        -33.0 / 8.0,
        7.0 / 4.0,
    ]

    constants_L_0 = L_csts_nonspin[0] + eta * L_csts_nonspin[1]
    constants_L_1 = IMRPhenomX_Get_PN_beta(
        L_csts_spinorbit[0], L_csts_spinorbit[1], dotS1L, dotS2L, q
    )
    constants_L_2 = (
        L_csts_nonspin[2] + eta * L_csts_nonspin[3] + eta * eta * L_csts_nonspin[4]
    )
    constants_L_3 = IMRPhenomX_Get_PN_beta(
        (L_csts_spinorbit[2] + L_csts_spinorbit[3] * eta),
        (L_csts_spinorbit[4] + L_csts_spinorbit[5] * eta),
        dotS1L,
        dotS2L,
        q,
    )
    constants_L_4 = (
        L_csts_nonspin[5]
        + L_csts_nonspin[6] * eta
        + L_csts_nonspin[7] * eta * eta
        + L_csts_nonspin[8] * eta * eta * eta
    )

    return constants_L_0, constants_L_1, constants_L_2, constants_L_3, constants_L_4


def compute_spin_norm_squared(chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, mass_1, mass_2):
    """
    Compute the squared norms of the dimensionless spin vectors S1 and S2.

    Args:
        chi1x, chi1y, chi1z: Components of dimensionless spin vector for mass 1
        chi2x, chi2y, chi2z: Components of dimensionless spin vector for mass 2
        mass_1: Mass of the primary (m1 > m2)
        mass_2: Mass of the secondary

    Returns:
        tuple: (S1_norm_2, S2_norm_2) - squared norms of the spin vectors
    """
    chi1_norm = jnp.sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z)
    chi2_norm = jnp.sqrt(chi2x * chi2x + chi2y * chi2y + chi2z * chi2z)

    total_mass = mass_1 + mass_2
    mass_1_fraction = mass_1 / total_mass
    mass_2_fraction = mass_2 / total_mass

    S1_norm = jnp.abs(chi1_norm) * jnp.power(mass_1_fraction, 2)
    S2_norm = jnp.abs(chi2_norm) * jnp.power(mass_2_fraction, 2)

    S1_norm_2 = jnp.power(S1_norm, 2)
    S2_norm_2 = jnp.power(S2_norm, 2)

    return S1_norm_2, S2_norm_2


def compute_psi0(
    Smi2: Float,
    Spl2: Float,
    S32: Float,
    S_0_norm: Float,
    v_0: Float,
    v_0_2: Float,
    psi1: Float,
    psi2: Float,
    g0: Float,
    delta_qq: Float,
    L_0: Float[Array, "3"],
    S1v: Float[Array, "3"],
    S2v: Float[Array, "3"],
) -> Float:
    """
    Compute initial psi0 value for MSA approximation.

    Args:
        Smi2: S_minus squared root (Float)
        Spl2: S_plus squared root (Float)
        S32: S_3 squared root (Float)
        S_0_norm: Initial total spin norm (Float)
        v_0: Initial velocity parameter (Float)
        v_0_2: v_0 squared (Float)
        psi1: Psi coefficient 1 (Float)
        psi2: Psi coefficient 2 (Float)
        g0: g0 coefficient (Float)
        delta_qq: Delta mass ratio term (Float)
        L_0: Initial orbital angular momentum vector (array)
        S1v: Spin 1 vector (array)
        S2v: Spin 2 vector (array)

    Returns: Float: Initial psi0 value
    """
    condition = jnp.abs(Smi2 - Spl2) < 1.0e-5

    def psi0_zero():
        return 0.0

    def psi0_nonzero():
        mm = jnp.sqrt((Smi2 - Spl2) / (S32 - Spl2))
        tmpB = (S_0_norm * S_0_norm - Spl2) / (Smi2 - Spl2)

        volume_element = IMRPhenomX_vector_dot_product(
            IMRPhenomX_vector_cross_product(L_0, S1v), S2v
        )
        vol_sign = jnp.sign(
            volume_element
        )  # equivalent to (volume_element > 0) - (volume_element < 0)

        psi_of_v0 = IMRPhenomX_psiofv(v_0, v_0_2, 0.0, psi1, psi2, g0, delta_qq)

        # Handle boundary cases for tmpB
        def handle_boundary_cases():
            # If tmpB > 1.0 and close to 1
            case1_condition = jnp.logical_and(tmpB > 1.0, (tmpB - 1.0) < 0.00001)
            case1_result = (
                ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(1.0)), mm) - psi_of_v0
            )  # stub

            # If tmpB < 0.0 and close to 0
            case2_condition = jnp.logical_and(tmpB < 0.0, tmpB > -0.00001)
            case2_result = (
                ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(0.0)), mm) - psi_of_v0
            )  # stub

            # Normal case
            normal_result = (
                ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0
            )  # stub

            return jnp.where(
                case1_condition,
                case1_result,
                jnp.where(case2_condition, case2_result, normal_result),
            )

        def normal_case():
            return (
                ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0
            )  # stub

        # Check if we're in boundary case
        boundary_condition = jnp.logical_or(tmpB < 0.0, tmpB > 1.0)

        return jax.lax.cond(boundary_condition, handle_boundary_cases, normal_case)

    return jax.lax.cond(condition, psi0_zero, psi0_nonzero)


def IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, g0, delta_qq):
    """
    Compute psi(v) for the MSA approximation.

    Based on Equation 51 in arXiv:1703.03967.

    Args:
        v (Float): Orbital velocity parameter.
        v2 (Float): v squared.
        psi0, psi1, psi2 (Float): Psi expansion coefficients.
        g0 (Float): Precession coefficient g0.
        delta_qq (Float): Mass difference parameter delta_qq.

    Returns:
        psi (Float): The psi angle at velocity v.
    """
    return psi0 - 0.75 * g0 * delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)


def IMRPhenomX_vector_cross_product(
    v1: Float[Array, "3"], v2: Float[Array, "3"]
) -> Float[Array, "3"]:
    """
    Calculate cross product of two 3D vectors

    Args:
        v1: First 3D vector as JAX array [x, y, z] (Float[Array, "3"])
        v2: Second 3D vector as JAX array [x, y, z] (Float[Array, "3"])

    Returns:
        Float[Array, "3"]: Cross product vector
    """
    return jnp.cross(v1, v2)


def IMRPhenomX_Return_Constants_c_MSA(
    v: Float,
    JNorm: Float,
    Seff: Float,
    eta: Float,
    eta3: Float,
    inveta: Float,
    Spl2: Float,
    Smi2: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    delta_qq: Float,
):
    """
    Compute c constants for MSA approximation.

    Args:
        v (Float): Orbital velocity parameter.
        JNorm (Float): Normalized total angular momentum.
        Seff (Float): Effective spin parameter.
        eta (Float): Symmetric mass ratio.
        eta3 (Float): eta cubed.
        inveta (Float): Inverse of eta (1/eta).
        Spl2 (Float): S_plus squared.
        Smi2 (Float): S_minus squared.
        S1_norm_2 (Float): Spin 1 magnitude squared.
        S2_norm_2 (Float): Spin 2 magnitude squared.
        delta_qq (Float): MSA coefficient delta_qq.

    Returns:
        Tuple[float, float, float]: A tuple of (c0, c2, c4) MSA constants.
    """
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    JNorm2 = JNorm * JNorm

    x = JNorm * (
        0.75
        * (1.0 - Seff * v)
        * v2
        * (
            eta3
            + 4.0 * eta3 * Seff * v
            - 2.0
            * eta
            * (JNorm2 - Spl2 + 2.0 * (S1_norm_2 - S2_norm_2) * delta_qq)
            * v2
            - 4.0 * eta * Seff * (JNorm2 - Spl2) * v3
            + (JNorm2 - Spl2) ** 2 * v4 * inveta
        )
    )

    y = JNorm * (
        -1.5
        * eta
        * (Spl2 - Smi2)
        * (1.0 + 2.0 * Seff * v - (JNorm2 - Spl2) * v2 * inveta**2)
        * (1.0 - Seff * v)
        * v4
    )

    z = JNorm * (0.75 * inveta * (Spl2 - Smi2) ** 2 * (1.0 - Seff * v) * v6)

    return x, y, z


def IMRPhenomX_Return_Constants_d_MSA(
    LNorm: Float,
    JNorm: Float,
    Spl: Float,
    Spl2: Float,
    Smi2: Float,
):
    """
    Compute d constants for MSA approximation.

    Args:
        LNorm (Float): Normalized orbital angular momentum.
        JNorm (Float): Normalized total angular momentum.
        Spl (Float): S_plus.
        Spl2 (Float): S_plus squared.
        Smi2 (Float): S_minus squared.

    Returns:
        Tuple[float, float, float]: A tuple of (d0, d2, d4) MSA constants.
    """
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = -(JNorm2 - (LNorm + Spl) * (LNorm + Spl)) * (
        JNorm2 - (LNorm - Spl) * (LNorm - Spl)
    )

    y = -2.0 * (Spl2 - Smi2) * (JNorm2 + LNorm2 - Spl2)

    z = -((Spl2 - Smi2) ** 2)

    return x, y, z


def IMRPhenomX_Return_Psi_MSA(
    v: Float,
    v2: Float,
    g0: Float,
    delta_qq: Float,
    psi1: Float,
    psi2: Float,
) -> Float:
    """
    Compute psi for MSA approximation.

    Args:
        v (Float): Orbital velocity parameter.
        v2 (Float): v squared.
        g0 (Float): MSA coefficient g0.
        delta_qq (Float): MSA coefficient delta_qq.
        psi1 (Float): MSA coefficient psi1.
        psi2 (Float): MSA coefficient psi2.

    Returns:
        Float: Psi value.
    """
    return -0.75 * g0 * delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)


def IMRPhenomX_Return_Psi_dot_MSA(
    v: Float,
    Seff: Float,
    inveta: Float,
    Spl2: Float,
    S32: Float,
) -> Float:
    """
    Compute the time derivative of psi for MSA approximation.

    Args:
        v (Float): Orbital velocity parameter.
        Seff (Float): Effective spin parameter.
        inveta (Float): Inverse of symmetric mass ratio (1/eta).
        Spl2 (Float): S_plus squared.
        S32 (Float): S_3 squared.

    Returns:
        Float: Time derivative of psi.
    """
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * Seff) * jnp.sqrt(inveta)
    psi_dot = 0.5 * A_coeff * jnp.sqrt(Spl2 - S32)

    return psi_dot


def IMRPhenomX_Return_MSA_Corrections_MSA(
    v: Float,
    LNorm: Float,
    JNorm: Float,
    Seff: Float,
    eta: Float,
    eta3: Float,
    inveta: Float,
    Spl: Float,
    Spl2: Float,
    Smi2: Float,
    Spl2mSmi2: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    S32: Float,
    delta_qq: Float,
    g0: Float,
    psi0: Float,
    psi1: Float,
    psi2: Float,
):
    """
    Compute MSA corrections for precession angles.

    Args:
        v (Float): Orbital velocity parameter.
        LNorm (Float): Normalized orbital angular momentum.
        JNorm (Float): Normalized total angular momentum.
        Seff (Float): Effective spin parameter.
        eta (Float): Symmetric mass ratio.
        eta3 (Float): eta cubed.
        inveta (Float): Inverse of eta (1/eta).
        Spl (Float): S_plus.
        Spl2 (Float): S_plus squared.
        Smi2 (Float): S_minus squared.
        Spl2mSmi2 (Float): Spl2 - Smi2.
        S1_norm_2 (Float): Spin 1 magnitude squared.
        S2_norm_2 (Float): Spin 2 magnitude squared.
        S32 (Float): S_3 squared.
        delta_qq (Float): MSA coefficient delta_qq.
        g0 (Float): MSA coefficient g0.
        psi0 (Float): Initial psi value.
        psi1 (Float): MSA coefficient psi1.
        psi2 (Float): MSA coefficient psi2.

    Returns:
        Tuple[float, float]: A tuple of (vMSA_x, vMSA_y) MSA corrections.
    """
    v2 = v * v

    # Sets c0, c2 and c4 as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(
        v,
        JNorm,
        Seff,
        eta,
        eta3,
        inveta,
        Spl2,
        Smi2,
        S1_norm_2,
        S2_norm_2,
        delta_qq,
    )
    # Sets d0, d2 and d4 as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, Spl, Spl2, Smi2)

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0

    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (Spl2 / (JNorm * LNorm)))
    B_theta_L = 0.5 * Spl2mSmi2 / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(v, v2, g0, delta_qq, psi1, psi2) + psi0
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(v, Seff, inveta, Spl2, S32)

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(
        tan_psi
    )  # wraps psi to (-π/2, π/2) — required for the incomplete elliptic integral formula
    """
    # C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)

    C2num = (
        c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4)
        - c2 * d0 * (d2 + 2.0 * d4)
        + c4 * d0 * (two_d0 + d2)
    )
    """
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    # C2 = C2num / C2den

    # Cphi = C1 + C2
    # Dphi = C1 - C2

    def compute_Cphi_term():
        return (
            jnp.abs(
                (
                    c4 * d0 * ((2 * d0 + d2) + sd)
                    - c2 * d0 * ((d2 + 2.0 * d4) - sd)
                    - c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - sd))
                )
                / C2den
            )
            * (sqrt_nc / (nc - 1.0))
            * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))
            / psi_dot
        )

    def compute_Dphi_term():
        return (
            jnp.abs(
                (
                    -c4 * d0 * ((2 * d0 + d2) - sd)
                    + c2 * d0 * ((d2 + 2.0 * d4) + sd)
                    - c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))
                )
            )
            / C2den
            * (sqrt_nd / (nd - 1.0))
            * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))
            / psi_dot
        )

    phiz_0_MSA_Cphi_term: Float = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term: Float = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x: Float = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    vMSA_y: Float = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
        phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2)
    )

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return vMSA_x, vMSA_y


def IMRPhenomX_Return_phiz_MSA(
    v: Float,
    JNorm: Float,
    eta: Float,
    inveta: Float,
    eta2: Float,
    eta4: Float,
    c1: Float,
    SAv: Float,
    SAv2: Float,
    invSAv: Float,
    invSAv2: Float,
    Omegaz0_coeff: Float,
    Omegaz1_coeff: Float,
    Omegaz2_coeff: Float,
    Omegaz3_coeff: Float,
    Omegaz4_coeff: Float,
    Omegaz5_coeff: Float,
    phiz_0: Float,
) -> Float:
    """
    Compute the azimuthal precession angle phi_z using the MSA approximation.

    Based on Eq. 66 and D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017),
    arXiv:1703.03967.

    Args:
        v (Float): Orbital velocity parameter.
        JNorm (Float): Magnitude of the total angular momentum.
        eta (Float): Symmetric mass ratio.
        inveta (Float): Inverse of symmetric mass ratio (1/eta).
        eta2 (Float): eta squared.
        eta4 (Float): eta to the fourth power.
        c1 (Float): Precession constant c1.
        SAv (Float): Spin parameter SAv.
        SAv2 (Float): SAv squared.
        invSAv (Float): Inverse of SAv (1/SAv).
        invSAv2 (Float): Inverse of SAv squared (1/SAv^2).
        Omegaz0_coeff, ..., Omegaz5_coeff (Float): Omega_z expansion coefficients from Eqs. D15-D20.
        phiz_0 (Float): Initial phi_z value.

    Returns:
        phiz_out (Float): The azimuthal precession angle phi_z.
    """
    invv = 1.0 / v
    invv2 = invv * invv
    LNewt = eta / v

    c12 = c1 * c1

    inveta2 = inveta * inveta
    inveta3 = inveta2 * inveta
    inveta4 = inveta2 * inveta2

    invSAv3 = invSAv2 * invSAv
    invSAv4 = invSAv2 * invSAv2
    invSAv5 = invSAv4 * invSAv

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + JNorm * eta + eta * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))

    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (JNorm * inveta4) * (
        0.5 * c12
        - (c1 * eta2 * invv) / 6.0
        - (SAv2 * eta2) / 3.0
        - (eta4 * invv2) / 3.0
    ) - (0.5 * c1 * inveta) * (c12 * inveta4 - SAv2 * inveta2) * log1

    phiz_1_coeff = (
        -0.5 * JNorm * inveta2 * (c1 + eta * LNewt)
        + 0.5 * inveta3 * (c12 - eta2 * SAv2) * log1
    )

    phiz_2_coeff = -JNorm + SAv * log2 - c1 * log1 * inveta

    phiz_3_coeff = JNorm * v - eta * log1 + c1 * log2 * invSAv

    phiz_4_coeff = (
        0.5 * JNorm * invSAv2 * v * (c1 + v * SAv2)
        - 0.5 * invSAv3 * (c12 - eta2 * SAv2) * log2
    )

    phiz_5_coeff = (
        -JNorm
        * v
        * (
            0.5 * c12 * invSAv4
            - c1 * v * invSAv2 / 6.0
            - v * v / 3.0
            - eta2 * invSAv2 / 3.0
        )
        + 0.5 * c1 * invSAv5 * (c12 - eta2 * SAv2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0
    # Note that the <\Omega_z>^(n) are given by Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * Omegaz0_coeff
        + phiz_1_coeff * Omegaz1_coeff
        + phiz_2_coeff * Omegaz2_coeff
        + phiz_3_coeff * Omegaz3_coeff
        + phiz_4_coeff * Omegaz4_coeff
        + phiz_5_coeff * Omegaz5_coeff
        + phiz_0
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out


def IMRPhenomX_Return_zeta_MSA(
    v: Float,
    eta: Float,
    Omegazeta0_coeff: Float,
    Omegazeta1_coeff: Float,
    Omegazeta2_coeff: Float,
    Omegazeta3_coeff: Float,
    Omegazeta4_coeff: Float,
    Omegazeta5_coeff: Float,
    zeta_0: Float,
) -> Float:
    """
    Compute zeta angle for MSA approximation.

    Args:
        v (Float): Orbital velocity parameter.
        eta (Float): Symmetric mass ratio.
        Omegazeta0_coeff (Float): Zeta precession coefficient (order 0).
        Omegazeta1_coeff (Float): Zeta precession coefficient (order 1).
        Omegazeta2_coeff (Float): Zeta precession coefficient (order 2).
        Omegazeta3_coeff (Float): Zeta precession coefficient (order 3).
        Omegazeta4_coeff (Float): Zeta precession coefficient (order 4).
        Omegazeta5_coeff (Float): Zeta precession coefficient (order 5).
        zeta_0 (Float): Initial zeta value.

    Returns:
        Float: Zeta angle.
    """
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = (
        eta
        * (
            Omegazeta0_coeff * invv3
            + Omegazeta1_coeff * invv2
            + Omegazeta2_coeff * invv
            + Omegazeta3_coeff * logv
            + Omegazeta4_coeff * v
            + Omegazeta5_coeff * v2
        )
        + zeta_0
    )

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def IMRPhenomX_vector_sum(
    v1: Float[Array, "3"], v2: Float[Array, "3"]
) -> Float[Array, "3"]:
    """
    Calculate sum of two 3D vectors

    Args:
        v1: First 3D vector as JAX array (Float[Array, "3"])
        v2: Second 3D vector as JAX array (Float[Array, "3"])

    Returns:
        Float[Array, "3"]: Sum of the vectors
    """
    return v1 + v2


def IMRPhenomX_vector_L2_norm(v1: Float[Array, "3"]) -> Float:
    """
    Calculate L2 norm of a 3D vector

    Args:
        v1: 3D vector as JAX array [x, y, z] (Float[Array, "3"])

    Returns: Float: L2 norm of the vector
    """
    return jnp.linalg.norm(v1)


def IMRPhenomX_vector_scalar(v1: Float[Array, "3"], a: Float) -> Float[Array, "3"]:
    """
    Multiply a vector by a scalar

    Args:
        v1: 3D vector as JAX array [x, y, z] (Float[Array, "3"])
        a: Scalar multiplier (Float)

    Returns:
        Float[Array, "3"]: Scaled vector
    """
    v2 = jnp.array([a * v1[0], a * v1[1], a * v1[2]])
    return v2


def IMRPhenomX_JNorm_MSA(LNorm: Float, c1_over_eta: Float, SAv2: Float) -> Float:
    JNorm2 = LNorm * LNorm + 2.0 * LNorm * c1_over_eta + SAv2
    return jnp.sqrt(JNorm2)


def IMRPhenomX_L_norm_3PN_of_v(
    v: jax.Array,
    L_norm: Float,
    constants_L_0: Float,
    constants_L_1: Float,
    constants_L_2: Float,
    constants_L_3: Float,
    constants_L_4: Float,
) -> Float:
    """
    Compute L_norm at 3PN order.

    Args:
        v (jax.Array): Orbital velocity parameter.
        L_norm (Float): Normalized orbital angular momentum.
        constants_L_0 (Float): L polynomial coefficient (index 0).
        constants_L_1 (Float): L polynomial coefficient (index 1).
        constants_L_2 (Float): L polynomial coefficient (index 2).
        constants_L_3 (Float): L polynomial coefficient (index 3).
        constants_L_4 (Float): L polynomial coefficient (index 4).

    Returns:
        Float: L_norm at 3PN order.
    """
    v2 = v * v
    L_norm3PN = L_norm * (
        1.0
        + v2
        * (
            constants_L_0
            + v * constants_L_1
            + v2 * (constants_L_2 + v * constants_L_3 + v2 * constants_L_4)
        )
    )
    return L_norm3PN


def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(
    v: Float,
    eta: Float,
    eta2: Float,
    eta3: Float,
    eta4: Float,
    inveta: Float,
    c1: Float,
    c1_over_eta: Float,
    SAv: Float,
    SAv2: Float,
    invSAv: Float,
    invSAv2: Float,
    constants_L_0: Float,
    constants_L_1: Float,
    constants_L_2: Float,
    constants_L_3: Float,
    constants_L_4: Float,
    S1_norm_2: Float,
    S2_norm_2: Float,
    qq: Float,
    delta_qq: Float,
    Seff: Float,
    dotS1Ln: Float,
    dotS2Ln: Float,
    S_0_norm: Float,
    psi0: Float,
    psi1: Float,
    psi2: Float,
    g0: Float,
    Omegaz0_coeff: Float,
    Omegaz1_coeff: Float,
    Omegaz2_coeff: Float,
    Omegaz3_coeff: Float,
    Omegaz4_coeff: Float,
    Omegaz5_coeff: Float,
    phiz_0: Float,
    Omegazeta0_coeff: Float,
    Omegazeta1_coeff: Float,
    Omegazeta2_coeff: Float,
    Omegazeta3_coeff: Float,
    Omegazeta4_coeff: Float,
    Omegazeta5_coeff: Float,
    zeta_0: Float,
):
    """
    Wrapper to generate phi_z, zeta and cos(theta_L) at a given frequency.

    Args:
        v: Velocity parameter (Float)
        eta: Symmetric mass ratio (Float)
        eta2: eta squared (Float)
        eta3: eta cubed (Float)
        eta4: eta to the fourth (Float)
        inveta: Inverse of eta (Float)
        c1: c1 coefficient (Float)
        c1_over_eta: c1 divided by eta (Float)
        SAv: Spin average (Float)
        SAv2: Spin average squared (Float)
        invSAv: Inverse of SAv (Float)
        invSAv2: Inverse of SAv squared (Float)
        constants_L: Array of L constants [L0, L1, L2, L3, L4] (array)
        S1_norm_2: Squared norm of spin 1 (Float)
        S2_norm_2: Squared norm of spin 2 (Float)
        qq: Mass ratio q = m1/m2 (Float)
        delta_qq: Delta mass ratio term (Float)
        Seff: Effective spin (Float)
        dotS1Ln: Dot product of S1 and Lhat (Float)
        dotS2Ln: Dot product of S2 and Lhat (Float)
        S_0_norm: Initial total spin norm (Float)
        psi0: Psi coefficient 0 (Float)
        psi1: Psi coefficient 1 (Float)
        psi2: Psi coefficient 2 (Float)
        g0: g0 coefficient (Float)
        Omegaz0_coeff through Omegaz5_coeff: Omega_z coefficients (floats)
        phiz_0: Initial phi_z value (Float)
        Omegazeta0_coeff through Omegazeta5_coeff: Omega_zeta coefficients (floats)
        zeta_0: Initial zeta value (Float)

    Returns:
        Tuple[float, float, float]: A tuple of (phi_z + phi_z_MSA, zeta + zeta_MSA, cos(theta_L))
    """
    L_norm = eta / v

    J_norm = IMRPhenomX_JNorm_MSA(L_norm, c1_over_eta, SAv2)

    # Compressing line 2212 - 2220
    L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(
        v,
        L_norm,
        constants_L_0,
        constants_L_1,
        constants_L_2,
        constants_L_3,
        constants_L_4,
    )

    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN, c1_over_eta, SAv2)
    vRoots = IMRPhenomX_Return_Roots_MSA(
        L_norm,
        J_norm,
        S1_norm_2,
        S2_norm_2,
        qq,
        eta,
        delta_qq,
        Seff,
        dotS1Ln,
        dotS2Ln,
        S_0_norm,
    )

    S32 = vRoots[0]
    Smi2 = vRoots[1]
    Spl2 = vRoots[2]

    Spl2mSmi2 = Spl2 - Smi2
    Spl = jnp.sqrt(Spl2)

    SNorm = IMRPhenomX_Return_SNorm_MSA(
        v, Smi2, Spl2, S32, psi0, psi1, psi2, g0, delta_qq
    )

    # Compressing line 2245-2249
    vMSA_correction = IMRPhenomX_Return_MSA_Corrections_MSA(
        v,
        L_norm,
        J_norm,
        Seff,
        eta,
        eta3,
        inveta,
        Spl,
        Spl2,
        Smi2,
        Spl2mSmi2,
        S1_norm_2,
        S2_norm_2,
        S32,
        delta_qq,
        g0,
        psi0,
        psi1,
        psi2,
    )
    cond = jnp.abs(Smi2 - Spl2) > 1.0e-5

    phiz_MSA_corr, zeta_MSA_corr = vMSA_correction
    phiz_MSA = jnp.where(cond, phiz_MSA_corr, 0.0)
    zeta_MSA = jnp.where(cond, zeta_MSA_corr, 0.0)

    phiz = IMRPhenomX_Return_phiz_MSA(
        v,
        J_norm,
        eta,
        inveta,
        eta2,
        eta4,
        c1,
        SAv,
        SAv2,
        invSAv,
        invSAv2,
        Omegaz0_coeff,
        Omegaz1_coeff,
        Omegaz2_coeff,
        Omegaz3_coeff,
        Omegaz4_coeff,
        Omegaz5_coeff,
        phiz_0,
    )

    zeta = IMRPhenomX_Return_zeta_MSA(
        v,
        eta,
        Omegazeta0_coeff,
        Omegazeta1_coeff,
        Omegazeta2_coeff,
        Omegazeta3_coeff,
        Omegazeta4_coeff,
        Omegazeta5_coeff,
        zeta_0,
    )
    cos_theta_L = IMRPhenomX_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)

    return phiz + phiz_MSA, zeta + zeta_MSA, cos_theta_L


def IMRPhenomX_costhetaLJ(L_norm: Float, J_norm: Float, S_norm: Float) -> Float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / (L_norm * J_norm)

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ


def IMRPhenomX_Return_SNorm_MSA(
    v: Float,
    Smi2: Float,
    Spl2: Float,
    S32: Float,
    psi0: Float,
    psi1: Float,
    psi2: Float,
    g0: Float,
    delta_qq: Float,
) -> Float:
    """
    Compute the spin magnitude SNorm using the MSA approximation.

    Based on Equations 23 and 25 of Chatziioannou et al, PRD 95, 104004, (2017),
    arXiv:1703.03967.

    Args:
        v (Float): Orbital velocity parameter.
        Smi2 (Float): S_minus squared.
        Spl2 (Float): S_plus squared.
        S32 (Float): S_3 squared.
        psi0, psi1, psi2 (Float): Psi expansion coefficients.
        g0 (Float): Precession coefficient g0.
        delta_qq (Float): Mass difference parameter delta_qq.

    Returns:
        SNorm (Float): The spin magnitude.
    """
    v2 = v * v

    cancel_condition = jnp.abs(Smi2 - Spl2) < 1e-5

    def sn_jacobi(_):
        # Equation 25 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
        m = (Smi2 - Spl2) / (S32 - Spl2)

        psi = IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, g0, delta_qq)

        # Jacobi elliptic functions
        sn, _cn, _dn = gsl_sf_elljac_e(
            psi, m, max_iter=6
        )  # 6 Landen iterations suffice for float64
        return sn

    sn = jnp.where(cancel_condition, 0.0, sn_jacobi(None))

    # Equation 23 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    SNorm2 = Spl2 + (Smi2 - Spl2) * sn * sn

    return jnp.sqrt(SNorm2)


def IMRPhenomX_vector_dot_product(
    v1: Float[Array, "3"], v2: Float[Array, "3"]
) -> Float:
    """
    Calculate dot product of two 3D vectors

    Args:
        v1: First 3D vector as JAX array (Float[Array, "3"])
        v2: Second 3D vector as JAX array (Float[Array, "3"])

    Returns: Float: Dot product
    """
    return jnp.dot(v1, v2)
