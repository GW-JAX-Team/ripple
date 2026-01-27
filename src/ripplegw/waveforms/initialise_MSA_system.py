
import jax.numpy as jnp
import jax
from ..constants import MTSUN, GAMMA
from .elliptic_integrals import ellint_F
from .elliptic_integrals import gsl_sf_elljac_e


#/** This function initializes all the core variables required for the MSA system. This will be called first. */
def IMRPhenomX_Initialize_MSA_System(pPrec, pWF: dict, ExpansionOrder: int):

    #Sanity check on the precession version
    pflag = pPrec.IMRPhenomXPrecVersion
    if pflag not in [220, 221, 222, 223, 224]:
        raise ValueError("Error: MSA system requires IMRPhenomXPrecVersion 220, 221, 222, 223 or 224.")
    
    '''
      First initialize the system of variables needed for Chatziioannou et al, PRD, 88, 063011, (2013), arXiv:1307.4418:
        - Racine et al, PRD, 80, 044010, (2009), arXiv:0812.4413
        - Favata, PRD, 80, 024002, (2009), arXiv:0812.0069
        - Blanchet et al, PRD, 84, 064041, (2011), arXiv:1104.5659
        - Bohe et al, CQG, 30, 135009, (2013), arXiv:1303.7412
    '''

    eta = pPrec.eta
    eta2 = pPrec.eta2
    eta3 = pPrec.eta3
    eta4 = pPrec.eta4

    m1 = pWF['m1']
    m2 = pWF['m2']

    # PN Coefficients for d \omega / d t as per LALSimInspiralFDPrecAngles_internals.c 
    LAL_LN2 = jnp.log(2.0)
    domegadt_constants_NS = [
        96./5., -1486./35., -264./5., 384.*jnp.pi/5., 34103./945., 13661./105., 944./15., 
        jnp.pi*(-4159./35.), jnp.pi*(-2268./5.), 
        (16447322263./7276500. + jnp.pi*jnp.pi*512./5. - LAL_LN2*109568./175. - GAMMA*54784./175.),
        (-56198689./11340. + jnp.pi*jnp.pi*902./5.), 1623./140., -1121./27., -54784./525., 
        -jnp.pi*883./42., jnp.pi*71735./63., jnp.pi*73196./63.
    ]

    domegadt_constants_SO = [
        -904./5., -120., -62638./105., 4636./5., -6472./35., 3372./5., -jnp.pi*720., 
        -jnp.pi*2416./5., -208520./63., 796069./105., -100019./45., -1195759./945., 
        514046./105., -8709./5., -jnp.pi*307708./105., jnp.pi*44011./7., 
        -jnp.pi*7992./7., jnp.pi*151449./35.
    ]

    domegadt_constants_SS = [-494./5., -1442./5., -233./5., -719./5.]

    L_csts_nonspin = [
        3./2., 1./6., 27./8., -19./8., 1./24., 135./16., 
        -6889./144. + 41./24.*jnp.pi*jnp.pi, 31./24., 7./1296.
    ]

    L_csts_spinorbit = [-14./6., -3./2., -11./2., 133./72., -33./8., 7./4.]

    '''
        Note that Chatziioannou et al use q = m2/m1, where m1 > m2 and therefore q < 1
        IMRPhenomX assumes m1 > m2 and q > 1. For the internal MSA code, flip q and
        dump this to pPrec->qq, where qq explicitly dentoes that this is 0 < q < 1.
    '''

    q = m2 / m1  # m2 / m1, q < 1, m1 > m2
    invq = 1.0 / q  # m1 / m2, invq > 1, m1 > m2
    object.__setattr__(pPrec, 'qq', q)
    object.__setattr__(pPrec, 'invqq', invq)

    mu = (m1 * m2) / (m1 + m2)

    #    /* \delta and powers of \delta in terms of q < 1, should just be m1 - m2 */
    object.__setattr__(pPrec, 'delta_qq', (1.0 - pPrec.qq) / (1.0 + pPrec.qq))
    object.__setattr__(pPrec, 'delta2_qq', pPrec.delta_qq * pPrec.delta_qq)
    object.__setattr__(pPrec, 'delta3_qq', pPrec.delta_qq * pPrec.delta2_qq)
    object.__setattr__(pPrec, 'delta4_qq', pPrec.delta_qq * pPrec.delta3_qq)

    # Initialize empty vectors (using dictionaries to represent vectors)
    S1v = jnp.array([0.0, 0.0, 0.0])
    S2v =jnp.array([0.0, 0.0, 0.0])

    # Define source frame such that \hat{L} = {0,0,1} with L_z pointing along \hat{z}
    Lhat =jnp.array([0.0, 0.0, 1.0])

    # Set LHat variables - these are fixed
    object.__setattr__(pPrec, 'Lhat_cos_theta', 1.0)  # Cosine of Polar angle of orbital angular momentum
    object.__setattr__(pPrec, 'Lhat_phi', 0.0)        # Azimuthal angle of orbital angular momentum
    object.__setattr__(pPrec, 'Lhat_theta', 0.0)      # Polar angle of orbital angular momentum

    # Dimensionful spin vectors, note eta = m1 * m2 and q = m2/m1
    S1v = S1v.at[0].set(pPrec.chi1x * eta/q)  # eta / q = m1^2
    S1v = S1v.at[1].set(pPrec.chi1y * eta/q)
    S1v = S1v.at[2].set(pPrec.chi1z * eta/q)

    S2v = S2v.at[0].set(pPrec.chi2x * eta*q)  # eta * q = m2^2
    S2v = S2v.at[1].set(pPrec.chi2y * eta*q)
    S2v = S2v.at[2].set(pPrec.chi2z * eta*q)

    S1_0_norm = IMRPhenomX_vector_L2_norm(S1v)  # stub
    S2_0_norm = IMRPhenomX_vector_L2_norm(S2v)  # stub


    # Initial dimensionful spin vectors at reference frequency
    # S1 = {S1x,S1y,S1z}
    object.__setattr__(pPrec, 'S1_0', jnp.array([S1v[0], S1v[1], S1v[2]]))

    # S2 = {S2x,S2y,S2z}
    object.__setattr__(pPrec, 'S2_0', jnp.array([S2v[0], S2v[1], S2v[2]]))

    # Reference velocity v and v^2
    object.__setattr__(pPrec, 'v_0', jnp.power(pPrec.piGM * pWF['fRef'], 1.0/3.0))
    object.__setattr__(pPrec, 'v_0_2', pPrec.v_0 * pPrec.v_0)

    # Reference orbital angular momenta
    L_0 = jnp.array([0.0, 0.0, 0.0])
    L_0 = IMRPhenomX_vector_scalar(Lhat, pPrec.eta / pPrec.v_0)  # stub
    object.__setattr__(pPrec, 'L_0', L_0)    

    # Inner products used in MSA system
    dotS1L = IMRPhenomX_vector_dot_product(S1v, Lhat)  # stub
    dotS2L = IMRPhenomX_vector_dot_product(S2v, Lhat)  # stub
    dotS1S2 = IMRPhenomX_vector_dot_product(S1v, S2v)  # stub
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm

    # Add dot products to struct
    object.__setattr__(pPrec, 'dotS1L', dotS1L)
    object.__setattr__(pPrec, 'dotS2L', dotS2L)
    object.__setattr__(pPrec, 'dotS1S2', dotS1S2)
    object.__setattr__(pPrec, 'dotS1Ln', dotS1Ln)
    object.__setattr__(pPrec, 'dotS2Ln', dotS2Ln)


    # Coefficients for PN orbital angular momentum at 3PN, as per LALSimInspiralFDPrecAngles_internals.c
    constants_L = jnp.zeros(5)  # Initialize array for 5 coefficients

    constants_L = constants_L.at[0].set(
        L_csts_nonspin[0] + eta * L_csts_nonspin[1]
    )
    constants_L = constants_L.at[1].set(
        IMRPhenomX_Get_PN_beta(L_csts_spinorbit[0], L_csts_spinorbit[1], pPrec)  # stub
    )
    constants_L = constants_L.at[2].set(
        L_csts_nonspin[2] + eta * L_csts_nonspin[3] + eta * eta * L_csts_nonspin[4]
    )
    constants_L = constants_L.at[3].set(
        IMRPhenomX_Get_PN_beta(
            (L_csts_spinorbit[2] + L_csts_spinorbit[3] * eta),
            (L_csts_spinorbit[4] + L_csts_spinorbit[5] * eta),
            pPrec
        )  # stub
    )
    constants_L = constants_L.at[4].set(
        L_csts_nonspin[5] + L_csts_nonspin[6] * eta + L_csts_nonspin[7] * eta * eta + L_csts_nonspin[8] * eta * eta * eta
    )
    object.__setattr__(pPrec, 'constants_L', constants_L)

    # Effective total spin
    Seff = (1.0 + q) * pPrec.dotS1L + (1 + (1.0/q)) * pPrec.dotS2L
    Seff2 = Seff * Seff

    object.__setattr__(pPrec, 'Seff', Seff)
    object.__setattr__(pPrec, 'Seff2', Seff2)

    #Line 2347
    # Initial total spin, S = S1 + S2
    S0 = jnp.array([0.0, 0.0, 0.0])
    S0 = IMRPhenomX_vector_sum(S1v, S2v)  # stub

    # Cache total spin in the precession struct
    object.__setattr__(pPrec, 'S_0', S0)

    #    /* Initial total angular momentum, J = L + S1 + S2 */
    object.__setattr__(pPrec, 'J_0', IMRPhenomX_vector_sum(pPrec.L_0, pPrec.S_0))

    # Norm of total initial spin
    object.__setattr__(pPrec, 'S_0_norm', IMRPhenomX_vector_L2_norm(S0))
    object.__setattr__(pPrec, 'S_0_norm_2', pPrec.S_0_norm * pPrec.S_0_norm)

    # Norm of orbital and total angular momenta
    object.__setattr__(pPrec, 'L_0_norm', IMRPhenomX_vector_L2_norm(pPrec.L_0))
    object.__setattr__(pPrec, 'J_0_norm', IMRPhenomX_vector_L2_norm(pPrec.J_0))

    L0norm = pPrec.L_0_norm
    J0norm = pPrec.J_0_norm

    # Useful powers
    object.__setattr__(pPrec, 'S_0_norm_2', pPrec.S_0_norm * pPrec.S_0_norm)
    object.__setattr__(pPrec, 'J_0_norm_2', pPrec.J_0_norm * pPrec.J_0_norm)
    object.__setattr__(pPrec, 'L_0_norm_2', pPrec.L_0_norm * pPrec.L_0_norm)

    # Vector for obtaining B, C, D coefficients
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(
        pPrec.L_0_norm, pPrec.J_0_norm, pPrec)


    vRoots = jnp.array([0.0, 0.0, 0.0])

    vRoots = IMRPhenomX_Return_Roots_MSA(pPrec.L_0_norm, pPrec.J_0_norm, pPrec)

    #Line 2500

    object.__setattr__(pPrec, 'Spl2', vRoots[2])
    object.__setattr__(pPrec, 'Smi2', vRoots[1])
    object.__setattr__(pPrec, 'S32', vRoots[0])

    # S^2_+ + S^2_-
    object.__setattr__(pPrec, 'Spl2pSmi2', pPrec.Spl2 + pPrec.Smi2)

    # S^2_+ - S^2_-
    object.__setattr__(pPrec, 'Spl2mSmi2', pPrec.Spl2 - pPrec.Smi2)

    # S_+ and S_-
    object.__setattr__(pPrec, 'Spl', jnp.sqrt(pPrec.Spl2))
    object.__setattr__(pPrec, 'Smi', jnp.sqrt(pPrec.Smi2))

    # Eq. 45 of PRD 95, 104004, (2017), arXiv:1703.03967, set from initial conditions
    object.__setattr__(pPrec, 'SAv2', 0.5 * (pPrec.Spl2pSmi2))
    object.__setattr__(pPrec, 'SAv', jnp.sqrt(pPrec.SAv2))
    object.__setattr__(pPrec, 'invSAv2', 1.0 / pPrec.SAv2)
    object.__setattr__(pPrec, 'invSAv', 1.0 / pPrec.SAv)


    # c_1 is determined by Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967
    c_1 = 0.5 * (J0norm*J0norm - L0norm*L0norm - pPrec.SAv2) / pPrec.L_0_norm * eta
    c1_2 = c_1 * c_1

    # Useful powers and combinations of c_1
    object.__setattr__(pPrec, 'c1', c_1)
    object.__setattr__(pPrec, 'c12', c_1 * c_1)
    object.__setattr__(pPrec, 'c1_over_eta', c_1 / eta)

    # Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967
    omqsq = (1.0 - q) * (1.0 - q) + 1e-16
    omq2 = (1.0 - q * q) + 1e-16

    # Precession averaged spin couplings, Eq. A9 - A14 of arXiv:1703.03967, note that we only use the initial values
    object.__setattr__(pPrec, 'S1L_pav', (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2))
    object.__setattr__(pPrec, 'S2L_pav', -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2))
    object.__setattr__(pPrec, 'S1S2_pav', 0.5 * pPrec.SAv2 - 0.5 * (pPrec.S1_norm_2 + pPrec.S2_norm_2))
    object.__setattr__(pPrec, 'S1Lsq_pav', (pPrec.S1L_pav*pPrec.S1L_pav +
                        ((pPrec.Spl2mSmi2)*(pPrec.Spl2mSmi2) * pPrec.v_0_2) / (32.0 * eta2 * omqsq)))
    object.__setattr__(pPrec, 'S2Lsq_pav', (pPrec.S2L_pav*pPrec.S2L_pav +
                        (q*q*(pPrec.Spl2mSmi2)*(pPrec.Spl2mSmi2) * pPrec.v_0_2) / (32.0 * eta2 * omqsq)))
    object.__setattr__(pPrec, 'S1LS2L_pav', (pPrec.S1L_pav*pPrec.S2L_pav -
                        q * (pPrec.Spl2mSmi2)*(pPrec.Spl2mSmi2)*pPrec.v_0_2 / (32.0 * eta2 * omqsq)))
    
    # Spin couplings in arXiv:1703.03967
    object.__setattr__(pPrec, 'beta3', (((113./12.) + (25./4.)*(m2/m1)) * pPrec.S1L_pav +
                    ((113./12.) + (25./4.)*(m1/m2)) * pPrec.S2L_pav))

    object.__setattr__(pPrec, 'beta5', (((31319./1008.) - (1159./24.)*eta) + (m2/m1)*((809./84) - (281./8.)*eta)) * pPrec.S1L_pav +
                    (((31319./1008.) - (1159./24.)*eta) + (m1/m2)*((809./84) - (281./8.)*eta)) * pPrec.S2L_pav)

    object.__setattr__(pPrec, 'beta6', jnp.pi * (((75./2.) + (151./6.)*(m2/m1))*pPrec.S1L_pav +
                            ((75./2.) + (151./6.)*(m1/m2))*pPrec.S2L_pav))

    object.__setattr__(pPrec, 'beta7', (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) +
                    (m2/m1)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * pPrec.S1L_pav +
                    (((130325./756) - (796069./2016)*eta + (100019./864.)*eta2) +
                    (m1/m2)*((1195759./18144) - (257023./1008.)*eta + (2903/32.)*eta2)) * pPrec.S2L_pav)

    object.__setattr__(pPrec, 'sigma4', ((1.0/mu) * ((247./48.)*pPrec.S1S2_pav - (721./48.)*pPrec.S1L_pav*pPrec.S2L_pav) +
                    (1.0/(m1*m1)) * ((233./96.)*pPrec.S1_norm_2 - (719./96.)*pPrec.S1Lsq_pav) +
                    (1.0/(m2*m2)) * ((233./96.)*pPrec.S2_norm_2 - (719./96.)*pPrec.S2Lsq_pav)))
    

    # Compute PN coefficients using precession-averaged spin couplings
    object.__setattr__(pPrec, 'a0', 96.0 * eta / 5.0)

    # These are all normalized by a factor of a0
    object.__setattr__(pPrec, 'a2', -(743./336.) - (11.0/4.)*eta)
    object.__setattr__(pPrec, 'a3', 4.0 * jnp.pi - pPrec.beta3)
    object.__setattr__(pPrec, 'a4', (34103./18144.) + (13661./2016.)*eta + (59./18.)*eta2 - pPrec.sigma4)
    object.__setattr__(pPrec, 'a5', -(4159./672.)*jnp.pi - (189./8.)*jnp.pi*eta - pPrec.beta5)
    object.__setattr__(pPrec, 'a6', ((16447322263./139708800.) + (16./3.)*jnp.pi*jnp.pi - (856./105)*jnp.log(16.) -
                (1712./105.)*GAMMA - pPrec.beta6 +
                eta*((451./48)*jnp.pi*jnp.pi - (56198689./217728.)) +
                eta2*(541./896.) - eta3*(5605./2592.)))
    object.__setattr__(pPrec, 'a7', (-(4415./4032.)*jnp.pi + (358675./6048.)*jnp.pi*eta +
                (91495./1512.)*jnp.pi*eta2 - pPrec.beta7))

    # Coefficients are weighted by an additional factor of a_0
    object.__setattr__(pPrec, 'a2', pPrec.a2 * pPrec.a0)
    object.__setattr__(pPrec, 'a3', pPrec.a3 * pPrec.a0)
    object.__setattr__(pPrec, 'a4', pPrec.a4 * pPrec.a0)
    object.__setattr__(pPrec, 'a5', pPrec.a5 * pPrec.a0)
    object.__setattr__(pPrec, 'a6', pPrec.a6 * pPrec.a0)
    object.__setattr__(pPrec, 'a7', pPrec.a7 * pPrec.a0)


    #Line 2597
    if pflag == 222 or pflag == 223:
        object.__setattr__(pPrec, 'a0', eta * domegadt_constants_NS[0])
        object.__setattr__(pPrec, 'a2', eta * (domegadt_constants_NS[1] + eta * (domegadt_constants_NS[2])))
        object.__setattr__(pPrec, 'a3', eta * (domegadt_constants_NS[3] +
                            IMRPhenomX_Get_PN_beta(domegadt_constants_SO[0], domegadt_constants_SO[1], pPrec)))
        object.__setattr__(pPrec, 'a4', eta * (domegadt_constants_NS[4] + eta * (domegadt_constants_NS[5] + eta * (domegadt_constants_NS[6])) +
                            IMRPhenomX_Get_PN_sigma(domegadt_constants_SS[0], domegadt_constants_SS[1], pPrec) +  # stub
                            IMRPhenomX_Get_PN_tau(domegadt_constants_SS[2], domegadt_constants_SS[3], pPrec)))    # stub
        object.__setattr__(pPrec, 'a5', eta * (domegadt_constants_NS[7] + eta * (domegadt_constants_NS[8]) +
                            IMRPhenomX_Get_PN_beta((domegadt_constants_SO[2] + eta * (domegadt_constants_SO[3])),
                                                    (domegadt_constants_SO[4] + eta * (domegadt_constants_SO[5])),
                                                    pPrec)))

    

    # Useful powers of a_0
    object.__setattr__(pPrec, 'a0_2', pPrec.a0 * pPrec.a0)
    object.__setattr__(pPrec, 'a0_3', pPrec.a0_2 * pPrec.a0)
    object.__setattr__(pPrec, 'a2_2', pPrec.a2 * pPrec.a2)

    # Calculate g coefficients as in Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    # These constants are used in TaylorT2 where domega/dt is expressed as an inverse polynomial
    object.__setattr__(pPrec, 'g0', 1.0 / pPrec.a0)

    # Eq. A2 (1703.03967)
    object.__setattr__(pPrec, 'g2', -(pPrec.a2 / pPrec.a0_2))

    # Eq. A3 (1703.03967)
    object.__setattr__(pPrec, 'g3', -(pPrec.a3 / pPrec.a0_2))

    # Eq.A4 (1703.03967)
    object.__setattr__(pPrec, 'g4', -(pPrec.a4 * pPrec.a0 - pPrec.a2_2) / pPrec.a0_3)

    # Eq. A5 (1703.03967)
    object.__setattr__(pPrec, 'g5', -(pPrec.a5 * pPrec.a0 - 2.0 * pPrec.a3 * pPrec.a2) / pPrec.a0_3)

    # Useful powers of delta
    delta = pPrec.delta_qq
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta * delta3

    # These are the phase coefficients of Eq. 51 of PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'psi0', 0.0)
    object.__setattr__(pPrec, 'psi1', 0.0)
    object.__setattr__(pPrec, 'psi2', 0.0)

    # \psi_1 is defined in Eq. C1 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'psi1', 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2))

    c_1_over_nu = pPrec.c1_over_eta
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q) * (1.0 + q)
    Seff_2 = Seff * Seff
    q_2 = q * q
    one_m_q_sq = (1.0 - q) * (1.0 - q)
    one_m_q2_2 = (1.0 - q_2) * (1.0 - q_2)
    one_m_q_4 = one_m_q_sq * one_m_q_sq

    # This implements the Delta term as in LALSimInspiralFDPrecAngles.c
    # c.f. https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralFDPrecAngles_internals.c#L145
    if pflag == 222 or pflag == 223:
        Del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
        Del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * Seff
        Del3 = 4.0 * (one_m_q2_2 * pPrec.S1_norm_2 - q_2 * Seff_2)
        Del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        Del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * Seff
        Del6 = 4.0 * (one_m_q2_2 * pPrec.S2_norm_2 - q_2 * Seff_2)
        object.__setattr__(pPrec, 'Delta', jnp.sqrt(jnp.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6))))
    else:
        # Coefficients of \Delta as defined in Eq. C3 of Appendix C in PRD, 95, 104004, (2017), arXiv:1703.03967.
        term1 = c1_2 * eta / (q * delta4)
        term2 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / (q * delta4)
        term3 = -eta2 * (delta2 * pPrec.S1_norm_2 - eta2 * Seff_2) / delta4
        
        # Is this 1) (c1_2 * q * eta / delta4) or 2) c1_2*eta2/delta4?
        # - In paper.pdf, the expression 1) is used.
        # Using eta^2 leads to higher frequency oscillations, use q * eta
        term4 = c1_2 * eta * q / delta4
        term5 = -2.0 * c_1 * eta3 * (1.0 + q) * Seff / delta4
        term6 = -eta2 * (delta2 * pPrec.S2_norm_2 - eta2 * Seff_2) / delta4
        object.__setattr__(pPrec, 'Delta', jnp.sqrt( jnp.abs( (term1 + term2 + term3) * (term4 + term5 + term6) ) ))

    # Line 2706

    if pflag == 222 or pflag == 223:
        u1 = 3.0 * pPrec.g2 / pPrec.g0
        u2 = 0.75 * one_p_q_sq / one_m_q_4
        u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
        u4 = 2.0 * one_m_q2_2 * (q * (2.0 + q) * pPrec.S1_norm_2 + (1.0 + 2.0 * q) * pPrec.S2_norm_2 - 2.0 * q * pPrec.SAv2)
        u5 = 2.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * 2.0 * c_1_over_nu * Seff
        u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
        u7 = q * pPrec.Delta
        
        # Eq. C2 (1703.03967)
        object.__setattr__(pPrec, 'psi2', u1 + u2 * (u3 + u4 + u5 - u6 + u7))
    else:
        # \psi_2 is defined in Eq. C2 of Appendix C in PRD, 95, 104004, (2017). Here we implement system of equations as in paper.pdf
        term1 = 3.0 * pPrec.g2 / pPrec.g0
        
        # q^2 or no q^2 in term2? Consensus on retaining q^2 term: https://git.ligo.org/waveforms/reviews/phenompv3hm/issues/7
        term2 = 3.0 * q * q / (2.0 * eta3)
        term3 = 2.0 * pPrec.Delta
        term4 = -2.0 * eta2 * pPrec.SAv2 / delta2
        term5 = -10.0 * eta * c1_2 / delta4
        term6 = 2.0 * eta2 * (7.0 + 6.0 * q + 7.0 * q * q) * c_1 * Seff / (omqsq * delta2)
        term7 = -eta3 * (3.0 + 4.0 * q + 3.0 * q * q) * Seff_2 / (omqsq * delta2)
        term8 = eta * (q * (2.0 + q) * pPrec.S1_norm_2 + (1.0 + 2.0 * q) * pPrec.S2_norm_2) / omqsq
        
        # \psi_2, C2 of Appendix C of PRD, 95, 104004, (2017)
        object.__setattr__(pPrec, 'psi2', term1 + term2 * (term3 + term4 + term5 + term6 + term7 + term8))


    # Eq. D1 of PRD, 95, 104004, (2017), arXiv:1703.03967
    Rm = pPrec.Spl2 - pPrec.Smi2
    Rm_2 = Rm * Rm

    # Eq. D2 and D3 Appendix D of PRD, 95, 104004, (2017), arXiv:1703.03967
    cp = pPrec.Spl2 * eta2 - c1_2
    cm = pPrec.Smi2 * eta2 - c1_2

    # Check if cm goes negative, this is likely pathological. If so, set MSA_ERROR to 1, so that waveform generator can handle
    # the error appropriately
    # if cm < 0.0:
    #     pPrec.MSA_ERROR = 1
    #     print(f"Error, coefficient cm = {cm:.16f}, which is negative and likely to be pathological. Triggering MSA failure.")

    # jnp.abs is here to help enforce positive definite cpcm
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)

    # Eq. D4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note difference to published version.
    a1dD = 0.5 + 0.75/eta

    # Eq. D5 in PRD, 95, 104004, (2017), arXiv:1703.03967
    a2dD = -0.75*Seff/eta

    # Eq. E3 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm * D2
    D2RmSq = (cp - sqrt_cpcm) / eta2

    # Eq. E4 in PRD, 95, 104004, (2017), arXiv:1703.03967 ; Note that this is Rm^2 * D4
    D4RmSq = -0.5*Rm*sqrt_cpcm/eta2 - cp/eta4*(sqrt_cpcm - cp)

    S0m = pPrec.S1_norm_2 - pPrec.S2_norm_2

    # Difference of spin norms squared, as used in Eq. D6 of PRD, 95, 104004, (2017), arXiv:1703.03967
    aw = (-3.0*(1.0 + q)/q*(2.0*(1.0 + q)*eta2*Seff*c_1 - (1.0 + q)*c1_2 + (1.0 - q)*eta2*S0m))
    cw = 3.0/32.0/eta*Rm_2
    dw = 4.0*cp - 4.0*D2RmSq*eta2
    hw = -2.0*(2.0*D2RmSq - Rm)*c_1
    fw = Rm*D2RmSq - D4RmSq - 0.25*Rm_2

    adD = aw / dw
    hdD = hw / dw
    cdD = cw / dw
    fdD = fw / dw

    gw = 3.0/16.0/eta2/eta*Rm_2*(c_1 - eta2*Seff)
    gdD = gw / dw

    # Useful powers of the coefficients
    hdD_2 = hdD * hdD
    adDfdD = adD * fdD
    adDfdDhdD = adDfdD * hdD
    adDhdD_2 = adD * hdD_2
    
    #Line 2800


    # Eq. D10 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz0', a1dD + adD)

    # Eq. D11 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz1', a2dD - adD*Seff - adD*hdD)

    # Eq. D12 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz2', adD*hdD*Seff + cdD - adD*fdD + adD*hdD_2)

    # Eq. D13 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz3', (adDfdD - cdD - adDhdD_2)*(Seff + hdD) + adDfdDhdD)

    # Eq. D14 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz4', (cdD + adDhdD_2 - 2.0*adDfdD)*(hdD*Seff + hdD_2 - fdD) - adD*fdD*fdD)

    # Eq. D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz5', ((cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0*hdD) -
                        (cdD + adDhdD_2 - 2.0*adDfdD) * hdD_2 * (Seff + hdD) -
                        adDfdD*fdD*hdD))
        
    # If Omegaz5 > 1000, this is larger than we expect and the system may be pathological.
    # Set MSA_ERROR = 1 to trigger an error
    if jnp.abs(pPrec.Omegaz5) > 1000.0:
        object.__setattr__(pPrec, 'MSA_ERROR', 1)
        print(f"Warning, |Omegaz5| = {pPrec.Omegaz5:.16f}, which is larger than expected and may be pathological. Triggering MSA failure.")

    g0 = pPrec.g0
    # Coefficients of Eq. 65, as defined in Equations D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967
    object.__setattr__(pPrec, 'Omegaz0_coeff', 3.0 * g0 * pPrec.Omegaz0)
    object.__setattr__(pPrec, 'Omegaz1_coeff', 3.0 * g0 * pPrec.Omegaz1)
    object.__setattr__(pPrec, 'Omegaz2_coeff', 3.0 * (g0 * pPrec.Omegaz2 + pPrec.g2*pPrec.Omegaz0))
    object.__setattr__(pPrec, 'Omegaz3_coeff', 3.0 * (g0 * pPrec.Omegaz3 + pPrec.g2*pPrec.Omegaz1 + pPrec.g3*pPrec.Omegaz0))
    object.__setattr__(pPrec, 'Omegaz4_coeff', 3.0 * (g0 * pPrec.Omegaz4 + pPrec.g2*pPrec.Omegaz2 + pPrec.g3*pPrec.Omegaz1 + pPrec.g4*pPrec.Omegaz0))
    #object.__setattr__(pPrec, 'Omegaz5_coeff', 3.0 * (g0 * pPrec.Omegaz5 + pPrec.g2*pPrec.Omegaz3 + pPrec.g3*pPrec.Omegaz2 + pPrec.g4*pPrec.Omegaz1 + pPrec.g5*pPrec.Omegaz0))
    object.__setattr__(pPrec, 'Omegaz5_coeff', 0.0) #FIXME
    
    # Coefficients of zeta: in Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967
    c1oveta2 = c_1 / eta2
    object.__setattr__(pPrec, 'Omegazeta0', pPrec.Omegaz0)
    object.__setattr__(pPrec, 'Omegazeta1', pPrec.Omegaz1 + pPrec.Omegaz0 * c1oveta2)
    object.__setattr__(pPrec, 'Omegazeta2', pPrec.Omegaz2 + pPrec.Omegaz1 * c1oveta2)
    object.__setattr__(pPrec, 'Omegazeta3', pPrec.Omegaz3 + pPrec.Omegaz2 * c1oveta2 + gdD)
    object.__setattr__(pPrec, 'Omegazeta4', pPrec.Omegaz4 + pPrec.Omegaz3 * c1oveta2 - gdD*Seff - gdD*hdD)
    object.__setattr__(pPrec, 'Omegazeta5', pPrec.Omegaz5 + pPrec.Omegaz4 * c1oveta2 + gdD*hdD*Seff + gdD*(hdD_2 - fdD))

    object.__setattr__(pPrec, 'Omegazeta0_coeff', -pPrec.g0 * pPrec.Omegazeta0)
    object.__setattr__(pPrec, 'Omegazeta1_coeff', -1.5 * pPrec.g0 * pPrec.Omegazeta1)
    object.__setattr__(pPrec, 'Omegazeta2_coeff', -3.0*(pPrec.g0 * pPrec.Omegazeta2 + pPrec.g2*pPrec.Omegazeta0))
    object.__setattr__(pPrec, 'Omegazeta3_coeff', 3.0*(pPrec.g0 * pPrec.Omegazeta3 + pPrec.g2*pPrec.Omegazeta1 + pPrec.g3*pPrec.Omegazeta0))
    object.__setattr__(pPrec, 'Omegazeta4_coeff', 3.0*(pPrec.g0 * pPrec.Omegazeta4 + pPrec.g2*pPrec.Omegazeta2 + pPrec.g3*pPrec.Omegazeta1 + pPrec.g4*pPrec.Omegazeta0))
    #object.__setattr__(pPrec, 'Omegazeta5_coeff', 1.5*(pPrec.g0*pPrec.Omegazeta5 + pPrec.g2*pPrec.Omegazeta3 + pPrec.g3*pPrec.Omegazeta2 + pPrec.g4*pPrec.Omegazeta1 + pPrec.g5*pPrec.Omegazeta0))
    object.__setattr__(pPrec, 'Omegazeta5_coeff', 0.0) #FIXME
        
    #Line 2887 - 2943 compressed
    pPrec = apply_expansion_order(pPrec, ExpansionOrder)

    #Line 2960 - 3004 compressed
    # Get psi0 term
    #psi_of_v0 = 0.0
    #mm = 0.0
    #tmpB = 0.0
    #volume_element = 0.0
    #vol_sign = 0.0

    object.__setattr__(pPrec, 'psi0', compute_psi0(pPrec, L_0, S1v, S2v) )

    #vMSA = jnp.array([0.0, 0.0, 0.0])

    phiz_0 = 0.0
    #phiz_0_MSA = 0.0  # UNUSED in original

    zeta_0 = 0.0
    #zeta_0_MSA = 0.0  # UNUSED in original

    # Tolerance chosen to be consistent with implementation in LALSimInspiralFDPrecAngles
    condition = jnp.abs(pPrec.Spl2 - pPrec.Smi2) > 1e-5

    def compute_msa_corrections():
        return IMRPhenomX_Return_MSA_Corrections_MSA(pPrec.v_0, pPrec.L_0_norm, pPrec.J_0_norm, pPrec)  # stub

    def no_msa_corrections():
        return jnp.array([0.0, 0.0, 0.0])

    vMSA = jax.lax.cond(condition, compute_msa_corrections, no_msa_corrections)

    #phiz_0_MSA = vMSA[0]
    #zeta_0_MSA = vMSA[1]

    # Initial \phi_z
    object.__setattr__(pPrec, 'phiz_0', 0.0)
    #phiz_0 = IMRPhenomX_Return_phiz_MSA(pPrec.v_0, pPrec.J_0_norm, pPrec)  # stub


    phiz_0 =IMRPhenomX_Return_phiz_MSA(pPrec.v_0, pPrec.J_0_norm,
                                        pPrec.eta, pPrec.inveta, pPrec.eta2, pPrec.eta4,
                                        pPrec.c1, pPrec.SAv, pPrec.SAv2, pPrec.invSAv, pPrec.invSAv2,
                                        pPrec.Omegaz0_coeff, pPrec.Omegaz1_coeff, pPrec.Omegaz2_coeff,
                                        pPrec.Omegaz3_coeff, pPrec.Omegaz4_coeff, pPrec.Omegaz5_coeff,
                                        pPrec.phiz_0
                                    )

    # Initial \zeta
    object.__setattr__(pPrec, 'zeta_0', 0.0)
    zeta_0 = IMRPhenomX_Return_zeta_MSA(
        pPrec.v_0,
        pPrec.eta,
        pPrec.Omegazeta0_coeff,
        pPrec.Omegazeta1_coeff,
        pPrec.Omegazeta2_coeff,
        pPrec.Omegazeta3_coeff,
        pPrec.Omegazeta4_coeff,
        pPrec.Omegazeta5_coeff,
        pPrec.zeta_0,
    )  # stub

    object.__setattr__(pPrec, 'phiz_0', -phiz_0 - vMSA[0])
    object.__setattr__(pPrec, 'zeta_0', -zeta_0 - vMSA[1])

    return pPrec





def IMRPhenomX_Return_Roots_MSA(LNorm, JNorm, pPrec):
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)  
    B, C, D = vBCD[0], vBCD[1], vBCD[2]

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
    
    #print(f'{p=}, {sqrtarg=}, {theta=}, {B=}, {B2=}, {C=}')
    
    vector_condition = jnp.logical_or(jnp.isnan(theta),
                                                   (jnp.isnan(sqrtarg)))
    scalar_condition = jnp.any(jnp.array([(pPrec.dotS1Ln == 1.0),
                                       (pPrec.dotS2Ln == 1.0),
                                       (pPrec.dotS1Ln == -1.0),
                                       (pPrec.dotS2Ln == -1.0),
                                       (pPrec.S1_norm_2 == 0.0),
                                       (pPrec.S2_norm_2 == 0.0)]))

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
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2)
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return jnp.array([S32, Smi2, Spl2])

    def roots_when_invalid():
        Smi2 = pPrec.S_0_norm**2 * jnp.ones_like(LNorm)
        Spl2 = Smi2 + 1e-9
        S32 = jnp.zeros_like(LNorm)
        return jnp.array([S32, Smi2, Spl2])

    roots_array = jnp.where(
        jnp.atleast_1d(invalid_case),
        roots_when_invalid(),
        roots_when_valid()
    )
    

    return roots_array




def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec):
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = pPrec.S1_norm_2
    S2Norm2 = pPrec.S2_norm_2
    q       = pPrec.qq
    eta     = pPrec.eta
    delta   = pPrec.delta_qq
    deltaSq = delta * delta
    Seff    = pPrec.Seff

    J2mL2   = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = ((LNorm2 + S1Norm2) * q +
               2.0 * LNorm * Seff -
               2.0 * JNorm2 -
               S1Norm2 - S2Norm2 +
               (LNorm2 + S2Norm2) / q)

    # C coefficient (Eq. B3)
    C_coeff = (J2mL2Sq -
               2.0 * LNorm * Seff * J2mL2 -
               2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) +
               4.0 * eta * LNorm2 * Seff * Seff -
               2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm +
               2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2)

    # D coefficient (Eq. B4)
    D_coeff = (((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq +
               deltaSq * (S1Norm2 - S2Norm2)**2 * LNorm2 / eta +
               2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2)

    return jnp.array([B_coeff, C_coeff, D_coeff])




def IMRPhenomX_Get_PN_sigma(a: float, b: float, pPrec: dict) -> float:
    """
    Calculate PN sigma coefficient
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN sigma value
    """
    return pPrec.inveta * (a * pPrec.dotS1S2 - b * pPrec.dotS1L * pPrec.dotS2L)

def IMRPhenomX_Get_PN_tau(a: float, b: float, pPrec: dict) -> float:
    """
    Internal function to computes PN spin-spin couplings. As in LALSimInspiralFDPrecAngles.c
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN tau value
    """
    return ((pPrec.qq * ((pPrec.S1_norm_2 * a) - b * pPrec.dotS1L * pPrec.dotS1L) + 
             (a * pPrec.S2_norm_2 - b * pPrec.dotS2L * pPrec.dotS2L) / pPrec.qq) / 
            pPrec.eta)




def IMRPhenomX_Get_PN_beta(a: float, b: float, pPrec: dict) -> float:
    """
    Calculate PN beta coefficient
    
    Args:
        a: First coefficient (float)
        b: Second coefficient (float)
        pPrec: Precession structure dictionary (dict)
        
    Returns:
        float: PN beta value
    """
    return (pPrec.dotS1L * (a + b * pPrec.qq) + 
            pPrec.dotS2L * (a + b / pPrec.qq))




def apply_expansion_order(pPrec: dict, ExpansionOrder: int) -> dict:
    """
    Apply expansion order corrections in a JAX-friendly way
    
    Args:
        pPrec: Precession structure dictionary (dict)
        ExpansionOrder: Order of expansion (-1 for all orders, 1-5 for specific cutoffs) (int)
        
    Returns:
        dict: Updated pPrec dictionary
    """
    
    # Create masks for which coefficients to zero out based on expansion order
    zero_1_and_higher = ExpansionOrder <= 1
    zero_2_and_higher = ExpansionOrder <= 2
    zero_3_and_higher = ExpansionOrder <= 3
    zero_4_and_higher = ExpansionOrder <= 4
    zero_5_and_higher = ExpansionOrder <= 5
    
    # Apply corrections based on expansion order
    # For expansion order -1, keep all coefficients (no changes)
    # For higher orders, zero out coefficients beyond the specified order
    '''
    object.__setattr__(pPrec, 'Omegaz1_coeff', jnp.where(zero_1_and_higher, 0.0, pPrec.Omegaz1_coeff))
    object.__setattr__(pPrec, 'Omegazeta1_coeff', jnp.where(zero_1_and_higher, 0.0, pPrec.Omegazeta1_coeff))
    
    object.__setattr__(pPrec, 'Omegaz2_coeff', jnp.where(zero_2_and_higher, 0.0, pPrec.Omegaz2_coeff))
    object.__setattr__(pPrec, 'Omegazeta2_coeff', jnp.where(zero_2_and_higher, 0.0, pPrec.Omegazeta2_coeff))
    
    object.__setattr__(pPrec, 'Omegaz3_coeff', jnp.where(zero_3_and_higher, 0.0, pPrec.Omegaz3_coeff))
    object.__setattr__(pPrec, 'Omegazeta3_coeff', jnp.where(zero_3_and_higher, 0.0, pPrec.Omegazeta3_coeff))
    
    object.__setattr__(pPrec, 'Omegaz4_coeff', jnp.where(zero_4_and_higher, 0.0, pPrec.Omegaz4_coeff))
    object.__setattr__(pPrec, 'Omegazeta4_coeff', jnp.where(zero_4_and_higher, 0.0, pPrec.Omegazeta4_coeff))
    
    object.__setattr__(pPrec, 'Omegaz5_coeff', jnp.where(zero_5_and_higher, 0.0, pPrec.Omegaz5_coeff))
    object.__setattr__(pPrec, 'Omegazeta5_coeff', jnp.where(zero_5_and_higher, 0.0, pPrec.Omegazeta5_coeff))
    '''
    return pPrec




def compute_psi0(pPrec, L_0, S1v, S2v):
    condition = jnp.abs(pPrec.Smi2 - pPrec.Spl2) < 1.0e-5
    
    def psi0_zero():
        return 0.0
    
    def psi0_nonzero():
        mm = jnp.sqrt((pPrec.Smi2 - pPrec.Spl2) / (pPrec.S32 - pPrec.Spl2))
        tmpB = (pPrec.S_0_norm*pPrec.S_0_norm - pPrec.Spl2) / (pPrec.Smi2 - pPrec.Spl2)
        
        volume_element = IMRPhenomX_vector_dot_product(
            IMRPhenomX_vector_cross_product(L_0, S1v), S2v  
        )
        vol_sign = jnp.sign(volume_element)  # equivalent to (volume_element > 0) - (volume_element < 0)
        
        psi_of_v0 = IMRPhenomX_psiofv(pPrec.v_0, pPrec.v_0_2, 0.0, pPrec.psi1, pPrec.psi2, pPrec.g0, pPrec.delta_qq)  
        
        # Handle boundary cases for tmpB
        def handle_boundary_cases():
            # If tmpB > 1.0 and close to 1
            case1_condition = jnp.logical_and(tmpB > 1.0, (tmpB - 1.0) < 0.00001)
            case1_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(1.0)), mm) - psi_of_v0  # stub
            
            # If tmpB < 0.0 and close to 0
            case2_condition = jnp.logical_and(tmpB < 0.0, tmpB > -0.00001)
            case2_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(0.0)), mm) - psi_of_v0  # stub
            
            # Normal case
            normal_result = ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0  # stub
            
            return jnp.where(
                case1_condition, case1_result,
                jnp.where(case2_condition, case2_result, normal_result)
            )
        
        def normal_case():
            return ellint_F(jnp.arcsin(vol_sign * jnp.sqrt(tmpB)), mm) - psi_of_v0  # stub
        
        # Check if we're in boundary case
        boundary_condition = jnp.logical_or(tmpB < 0.0, tmpB > 1.0)
        #jax.debug.print('JAX debug {} {} {}', handle_boundary_cases(), normal_case(), boundary_condition)

        return jax.lax.cond(boundary_condition, handle_boundary_cases, normal_case)
    
    return jax.lax.cond(condition, psi0_zero, psi0_nonzero)



def IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, g0, delta_qq):
    """
    Compute psi(v) for the MSA approximation.

    Based on Equation 51 in arXiv:1703.03967.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    v2 : float
        v squared.
    psi0, psi1, psi2 : float
        Psi expansion coefficients.
    g0 : float
        Precession coefficient g0.
    delta_qq : float
        Mass difference parameter delta_qq.

    Returns
    -------
    psi : float
        The psi angle at velocity v.
    """
    return psi0 - 0.75 * g0 * delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)


def IMRPhenomX_vector_cross_product(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate cross product of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array [x, y, z] (jnp.ndarray)
        v2: Second 3D vector as JAX array [x, y, z] (jnp.ndarray)
        
    Returns:
        jnp.ndarray: Cross product vector
    """
    return jnp.cross(v1, v2)



def IMRPhenomX_Return_Constants_c_MSA(
    v: float,
    JNorm: float,
    Seff: float,
    eta: float,
    eta3: float,
    inveta: float,
    Spl2: float,
    Smi2: float,
    S1_norm_2: float,
    S2_norm_2: float,
    delta_qq: float,
) -> jnp.ndarray:
    """
    Compute c constants for MSA approximation.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    JNorm : float
        Normalized total angular momentum.
    Seff : float
        Effective spin parameter.
    eta : float
        Symmetric mass ratio.
    eta3 : float
        eta cubed.
    inveta : float
        Inverse of eta (1/eta).
    Spl2 : float
        S_plus squared.
    Smi2 : float
        S_minus squared.
    S1_norm_2 : float
        Spin 1 magnitude squared.
    S2_norm_2 : float
        Spin 2 magnitude squared.
    delta_qq : float
        MSA coefficient delta_qq.

    Returns
    -------
    jnp.ndarray
        Array of [c0, c2, c4] constants.
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
            - 2.0 * eta * (JNorm2 - Spl2 + 2.0 * (S1_norm_2 - S2_norm_2) * delta_qq) * v2
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

    return jnp.array([x, y, z])


def IMRPhenomX_Return_Constants_d_MSA(
    LNorm: float,
    JNorm: float,
    Spl: float,
    Spl2: float,
    Smi2: float,
) -> jnp.ndarray:
    """
    Compute d constants for MSA approximation.

    Parameters
    ----------
    LNorm : float
        Normalized orbital angular momentum.
    JNorm : float
        Normalized total angular momentum.
    Spl : float
        S_plus.
    Spl2 : float
        S_plus squared.
    Smi2 : float
        S_minus squared.

    Returns
    -------
    jnp.ndarray
        Array of [d0, d2, d4] constants.
    """
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = -(JNorm2 - (LNorm + Spl) * (LNorm + Spl)) * (
        JNorm2 - (LNorm - Spl) * (LNorm - Spl)
    )

    y = -2.0 * (Spl2 - Smi2) * (JNorm2 + LNorm2 - Spl2)

    z = -(Spl2 - Smi2) ** 2

    return jnp.array([x, y, z])


def IMRPhenomX_Return_Psi_MSA(
    v: float,
    v2: float,
    g0: float,
    delta_qq: float,
    psi1: float,
    psi2: float,
) -> float:
    """
    Compute psi for MSA approximation.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    v2 : float
        v squared.
    g0 : float
        MSA coefficient g0.
    delta_qq : float
        MSA coefficient delta_qq.
    psi1 : float
        MSA coefficient psi1.
    psi2 : float
        MSA coefficient psi2.

    Returns
    -------
    float
        Psi value.
    """
    return -0.75 * g0 * delta_qq * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)





def IMRPhenomX_Return_Psi_dot_MSA(
    v: float,
    Seff: float,
    inveta: float,
    Spl2: float,
    S32: float,
) -> float:
    """
    Compute the time derivative of psi for MSA approximation.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    Seff : float
        Effective spin parameter.
    inveta : float
        Inverse of symmetric mass ratio (1/eta).
    Spl2 : float
        S_plus squared.
    S32 : float
        S_3 squared.

    Returns
    -------
    float
        Time derivative of psi.
    """
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * Seff) * jnp.sqrt(inveta)
    psi_dot = 0.5 * A_coeff * jnp.sqrt(Spl2 - S32)

    return psi_dot




def IMRPhenomX_Return_MSA_Corrections_MSA(
    v, 
    LNorm, 
    JNorm, 
    pPrec
    ):
    
    v2 = v * v

    # Sets c0, c2 and c4 in pPrec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(
        v,
        JNorm,
        pPrec.Seff,
        pPrec.eta,
        pPrec.eta3,
        pPrec.inveta,
        pPrec.Spl2,
        pPrec.Smi2,
        pPrec.S1_norm_2,
        pPrec.S2_norm_2,
        pPrec.delta_qq,
    )
    # Sets d0, d2 and d4 in pPrec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(
        LNorm, JNorm, pPrec.Spl, pPrec.Spl2, pPrec.Smi2
    )  

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    #jax.debug.print("jax D vector {} {} {}", d0, d2, d4)

    two_d0 = 2.0 * d0
    
    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (pPrec.Spl2 / (JNorm * LNorm)))
    B_theta_L = 0.5 * pPrec.Spl2mSmi2 / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(
        v, v2, pPrec.g0, pPrec.delta_qq, pPrec.psi1, pPrec.psi2
    ) + pPrec.psi0
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(
        v, pPrec.Seff, pPrec.inveta, pPrec.Spl2, pPrec.S32
    ) 

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    C2num = (c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) -
             c2 * d0 * (d2 + 2.0 * d4) +
             c4 * d0 * (two_d0 + d2))
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    C2 = C2num / C2den

    Cphi = C1 + C2
    Dphi = C1 - C2

    def compute_Cphi_term():
        
        return jnp.abs((
            (c4 * d0 * ((2 * d0 + d2) + sd) -
                c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - 
                sd))) / C2den) * (sqrt_nc / (nc - 1.0)) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))) / psi_dot
        
    def compute_Dphi_term():
            return jnp.abs((
                (-c4 * d0 * ((2 * d0 + d2) - sd) +
                 c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                 c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den
            ) * (sqrt_nd / (nd - 1.0)) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))) / psi_dot

    phiz_0_MSA_Cphi_term = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    #####  restart from here
    vMSA_y = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
                phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2))

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return jnp.stack([vMSA_x, vMSA_y, jnp.zeros_like(vMSA_x)], axis=0)




def IMRPhenomX_Return_phiz_MSA(
    v: float,
    JNorm: float,
    eta: float,
    inveta: float,
    eta2: float,
    eta4: float,
    c1: float,
    SAv: float,
    SAv2: float,
    invSAv: float,
    invSAv2: float,
    Omegaz0_coeff: float,
    Omegaz1_coeff: float,
    Omegaz2_coeff: float,
    Omegaz3_coeff: float,
    Omegaz4_coeff: float,
    Omegaz5_coeff: float,
    phiz_0: float,
) -> float:
    """
    Compute the azimuthal precession angle phi_z using the MSA approximation.

    Based on Eq. 66 and D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017),
    arXiv:1703.03967.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    JNorm : float
        Magnitude of the total angular momentum.
    eta : float
        Symmetric mass ratio.
    inveta : float
        Inverse of symmetric mass ratio (1/eta).
    eta2 : float
        eta squared.
    eta4 : float
        eta to the fourth power.
    c1 : float
        Precession constant c1.
    SAv : float
        Spin parameter SAv.
    SAv2 : float
        SAv squared.
    invSAv : float
        Inverse of SAv (1/SAv).
    invSAv2 : float
        Inverse of SAv squared (1/SAv^2).
    Omegaz0_coeff, ..., Omegaz5_coeff : float
        Omega_z expansion coefficients from Eqs. D15-D20.
    phiz_0 : float
        Initial phi_z value.

    Returns
    -------
    phiz_out : float
        The azimuthal precession angle phi_z.
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
        0.5 * c12 - (c1 * eta2 * invv) / 6.0 - (SAv2 * eta2) / 3.0 - (eta4 * invv2) / 3.0
    ) - (0.5 * c1 * inveta) * (
        c12 * inveta4 - SAv2 * inveta2
    ) * log1

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
        -JNorm * v * (
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
    v: float,
    eta: float,
    Omegazeta0_coeff: float,
    Omegazeta1_coeff: float,
    Omegazeta2_coeff: float,
    Omegazeta3_coeff: float,
    Omegazeta4_coeff: float,
    Omegazeta5_coeff: float,
    zeta_0: float,
) -> float:
    """
    Compute zeta angle for MSA approximation.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    eta : float
        Symmetric mass ratio.
    Omegazeta0_coeff : float
        Zeta precession coefficient (order 0).
    Omegazeta1_coeff : float
        Zeta precession coefficient (order 1).
    Omegazeta2_coeff : float
        Zeta precession coefficient (order 2).
    Omegazeta3_coeff : float
        Zeta precession coefficient (order 3).
    Omegazeta4_coeff : float
        Zeta precession coefficient (order 4).
    Omegazeta5_coeff : float
        Zeta precession coefficient (order 5).
    zeta_0 : float
        Initial zeta value.

    Returns
    -------
    float
        Zeta angle.
    """
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = eta * (
        Omegazeta0_coeff * invv3
        + Omegazeta1_coeff * invv2
        + Omegazeta2_coeff * invv
        + Omegazeta3_coeff * logv
        + Omegazeta4_coeff * v
        + Omegazeta5_coeff * v2
    ) + zeta_0

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def IMRPhenomX_vector_sum(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate sum of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array (jnp.ndarray)
        v2: Second 3D vector as JAX array (jnp.ndarray)
        
    Returns:
        jnp.ndarray: Sum of the vectors
    """
    return v1 + v2



def IMRPhenomX_vector_L2_norm(v1: jnp.ndarray) -> float:
    """
    Calculate L2 norm of a 3D vector
    
    Args:
        v1: 3D vector as JAX array [x, y, z] (jnp.ndarray)
        
    Returns:
        float: L2 norm of the vector
    """
    return jnp.linalg.norm(v1)



def IMRPhenomX_vector_scalar(v1: jnp.ndarray, a: float) -> jnp.ndarray:
    """
    Multiply a vector by a scalar
    
    Args:
        v1: 3D vector as JAX array [x, y, z] (jnp.ndarray)
        a: Scalar multiplier (float)
        
    Returns:
        jnp.ndarray: Scaled vector
    """
    v2 = jnp.array([a * v1[0], a * v1[1], a * v1[2]])
    return v2




def IMRPhenomX_JNorm_MSA(LNorm:float, c1_over_eta:float, SAv2:float)->float:
    JNorm2 = (LNorm * LNorm + 2.0 * LNorm * c1_over_eta + SAv2)
    return jnp.sqrt(JNorm2)




def IMRPhenomX_L_norm_3PN_of_v(
    v: jax.Array,
    L_norm: float,
    constants_L_0: float,
    constants_L_1: float,
    constants_L_2: float,
    constants_L_3: float,
    constants_L_4: float,
) -> float:
    """
    Compute L_norm at 3PN order.

    Parameters
    ----------
    v : jax.Array
        Orbital velocity parameter.
    L_norm : float
        Normalized orbital angular momentum.
    constants_L_0 : float
        L polynomial coefficient (index 0).
    constants_L_1 : float
        L polynomial coefficient (index 1).
    constants_L_2 : float
        L polynomial coefficient (index 2).
    constants_L_3 : float
        L polynomial coefficient (index 3).
    constants_L_4 : float
        L polynomial coefficient (index 4).

    Returns
    -------
    float
        L_norm at 3PN order.
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




def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(pPrec, pWF, v):
    # Wrapper to generate \f$\phi_z\f$, \f$\zeta\f$ and \f$\cos \theta_L\f$ at a given frequency

    

    L_norm = pWF['eta']/v

    J_norm = IMRPhenomX_JNorm_MSA(L_norm, pPrec.c1_over_eta, pPrec.SAv2)

    # Compressing line 2212 - 2220
    L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(
        v,
        L_norm,
        pPrec.constants_L[0],
        pPrec.constants_L[1],
        pPrec.constants_L[2],
        pPrec.constants_L[3],
        pPrec.constants_L[4],
    )

    '''
    if (pPrec.IMRPhenomXPrecVersion == 222) | (pPrec.IMRPhenomXPrecVersion == 223):
        L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)

    else:
        L_norm3PN = XLALSimIMRPhenomXLPNAnsatz(v, L_norm, pPrec.L0, pPrec.L1, pPrec.L2, pPrec.L3, pPrec.L4, pPrec.L5, pPrec.L6, pPrec.L7, pPrec.L8, pPrec.L8L)
    '''
    

    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN, pPrec.c1_over_eta, pPrec.SAv2)
    vRoots    = IMRPhenomX_Return_Roots_MSA(L_norm, J_norm, pPrec)

    object.__setattr__(pPrec, 'S32', vRoots[0])
    object.__setattr__(pPrec, 'Smi2', vRoots[1])
    object.__setattr__(pPrec, 'Spl2', vRoots[2])

    object.__setattr__(pPrec, 'Spl2mSmi2', pPrec.Spl2 - pPrec.Smi2)
    object.__setattr__(pPrec, 'Spl2pSmi2', pPrec.Spl2 + pPrec.Smi2)
    object.__setattr__(pPrec, 'Spl', jnp.sqrt(pPrec.Spl2))
    object.__setattr__(pPrec, 'Smi', jnp.sqrt(pPrec.Smi2))

    SNorm = IMRPhenomX_Return_SNorm_MSA(
        v,
        pPrec.Smi2, pPrec.Spl2, pPrec.S32,
        pPrec.psi0, pPrec.psi1, pPrec.psi2,
        pPrec.g0, pPrec.delta_qq
    )
    object.__setattr__(pPrec, 'S_norm', SNorm)
    object.__setattr__(pPrec, 'S_norm_2', SNorm * SNorm)

    # Compressing line 2245-2249
    vMSA_correction = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    cond = (jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5)

    # Create vMSA with zeros matching the shape of vMSA_correction
    vMSA_zeros = jnp.zeros_like(vMSA_correction)
    vMSA = jnp.where(cond, vMSA_correction, vMSA_zeros)
    
    '''
    if(jnp.abs(pPrec.Smi2 - pPrec.Spl2) > 1.e-5):

        #Get phiz_0_MSA and zeta_0_MSA
        vMSA = IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec)
    '''

    phiz_MSA     = vMSA[0]
    zeta_MSA     = vMSA[1]

    phiz         = IMRPhenomX_Return_phiz_MSA(
        v, J_norm,
        pPrec.eta, pPrec.inveta, pPrec.eta2, pPrec.eta4,
        pPrec.c1, pPrec.SAv, pPrec.SAv2, pPrec.invSAv, pPrec.invSAv2,
        pPrec.Omegaz0_coeff, pPrec.Omegaz1_coeff, pPrec.Omegaz2_coeff,
        pPrec.Omegaz3_coeff, pPrec.Omegaz4_coeff, pPrec.Omegaz5_coeff,
        pPrec.phiz_0
    )
    zeta         = IMRPhenomX_Return_zeta_MSA(
        v,
        pPrec.eta,
        pPrec.Omegazeta0_coeff,
        pPrec.Omegazeta1_coeff,
        pPrec.Omegazeta2_coeff,
        pPrec.Omegazeta3_coeff,
        pPrec.Omegazeta4_coeff,
        pPrec.Omegazeta5_coeff,
        pPrec.zeta_0,
    )
    cos_theta_L        = IMRPhenomX_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)

    vout1 = phiz + phiz_MSA
    vout2 = zeta + zeta_MSA
    vout3 = cos_theta_L

    #jax.debug.print("JAX debug v {} cos_theta_L {} ", v, cos_theta_L)


    return jnp.array([vout1, vout2, vout3])




def IMRPhenomX_costhetaLJ(
    L_norm: float, 
    J_norm: float, 
    S_norm: float
    ) -> float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / (L_norm * J_norm)

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ

def IMRPhenomX_Return_SNorm_MSA(
    v: float,
    Smi2: float,
    Spl2: float,
    S32: float,
    psi0: float,
    psi1: float,
    psi2: float,
    g0: float,
    delta_qq: float,
) -> float:
    """
    Compute the spin magnitude SNorm using the MSA approximation.

    Based on Equations 23 and 25 of Chatziioannou et al, PRD 95, 104004, (2017),
    arXiv:1703.03967.

    Parameters
    ----------
    v : float
        Orbital velocity parameter.
    Smi2 : float
        S_minus squared.
    Spl2 : float
        S_plus squared.
    S32 : float
        S_3 squared.
    psi0, psi1, psi2 : float
        Psi expansion coefficients.
    g0 : float
        Precession coefficient g0.
    delta_qq : float
        Mass difference parameter delta_qq.

    Returns
    -------
    SNorm : float
        The spin magnitude.
    """
    v2 = v * v

    cancel_condition = jnp.abs(Smi2 - Spl2) < 1e-5

    def sn_jacobi(_):
        # Equation 25 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
        m = (Smi2 - Spl2) / (S32 - Spl2)

        psi = IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, g0, delta_qq)

        # Jacobi elliptic functions
        sn, cn, dn = gsl_sf_elljac_e(psi, m)  # FIXME
        return sn

    sn = jnp.where(cancel_condition, 0.0, sn_jacobi(None))

    # Equation 23 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    SNorm2 = Spl2 + (Smi2 - Spl2) * sn * sn

    return jnp.sqrt(SNorm2)




def IMRPhenomX_vector_dot_product(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """
    Calculate dot product of two 3D vectors
    
    Args:
        v1: First 3D vector as JAX array (jnp.ndarray)
        v2: Second 3D vector as JAX array (jnp.ndarray)
        
    Returns:
        float: Dot product
    """
    return jnp.dot(v1, v2)