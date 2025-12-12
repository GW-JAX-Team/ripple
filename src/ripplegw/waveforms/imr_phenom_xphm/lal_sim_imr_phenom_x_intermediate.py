"""LAL's IMRPhenomX_intermediate.c JAX implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXUsefulPowersDataClass,
    IMRPhenomXWaveformDataClass,
)


def imr_phenom_x_intermediate_phase_22_v2m_rd_v4(
    eta: float, s: float, dchi: float, delta: float, int_phase_flag: int
) -> float:
    """
    v2mRDv4 = vIM2 - vRD4. See Section VII.B of arXiv:2001.11412.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    s2 = s * s

    is_valid = (int_phase_flag == 104) | (int_phase_flag == 105)
    checkify.check(
        is_valid,
        "Error in imr_phenom_x_intermediate_phase_22_v2m_rd_v4: IMRPhenomXIntermediatePhaseVersion is not valid. Recommended flag is 104.",
    )

    def case_104():  # Canonical, 4 coefficients
        no_spin = (
            eta * (-8.244230124407343 - 182.80239160435949 * eta + 638.2046409916306 * eta2 - 578.878727101827 * eta3)
        ) / (-0.004863669418916522 - 0.5095088831841608 * eta + 1.0 * eta2)

        eq_spin = (
            s
            * (
                0.1344136125169328
                + 0.0751872427147183 * s
                + eta2 * (7.158823192173721 + 25.489598292111104 * s - 7.982299108059922 * s2)
                + eta * (-5.792368563647471 + 1.0190164430971749 * s + 0.29150399620268874 * s2)
                + 0.033627267594199865 * s2
                + eta3 * (17.426503081351534 - 90.69790378048448 * s + 20.080325133405847 * s2)
            )
        ) / (
            0.03449545664201546
            - 0.027941977370442107 * s
            + (0.005274757661661763 + 0.0455523144123269 * eta - 0.3880379911692037 * eta2 + 1.0 * eta3) * s2
        )

        uneq_spin = 160.2975913661124 * dchi * delta * eta2

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 5 coefficients

        no_spin = (eta * (0.9951733419499662 + 101.21991715215253 * eta + 632.4731389009143 * eta2)) / (
            0.00016803066316882238 + 0.11412314719189287 * eta + 1.8413983770369362 * eta2 + 1.0 * eta3
        )

        eq_spin = (
            s
            * (
                18.694178521101332
                + 16.89845522539974 * s
                + 4941.31613710257 * eta2 * s
                + eta * (-697.6773920613674 - 147.53381808989846 * s2)
                + 0.3612417066833153 * s2
                + eta3 * (3531.552143264721 - 14302.70838220423 * s + 178.85850322465944 * s2)
            )
        ) / (2.965640445745779 - 2.7706595614504725 * s + 1.0 * s2)

        uneq_spin = dchi * delta * eta2 * (356.74395864902294 + 1693.326644293169 * eta2 * s)

        return no_spin + eq_spin + uneq_spin

    return jax.lax.cond(int_phase_flag == 104, lambda _: case_104(), lambda _: case_105(), operand=None)


def imr_phenom_x_intermediate_phase_22_v3m_rd_v4(
    eta: float, s: float, dchi: float, delta: float, int_phase_flag: int
) -> float:
    """
    v3mRDv4 = vIM3 - vRD4. See Section VII.B of arXiv:2001.11412.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta6 = eta3 * eta3

    s2 = s * s

    is_valid = (int_phase_flag == 104) | (int_phase_flag == 105)
    checkify.check(
        is_valid,
        "Error in imr_phenom_x_intermediate_phase_22_v3m_rd_v4: IMRPhenomXIntermediatePhaseVersion is not valid. Recommended flag is 104.",
    )

    def case_104():  # Canonical, 4 coefficients

        no_spin = (
            0.3145740304678042 + 299.79825045000655 * eta - 605.8886581267144 * eta2 - 1302.0954503758007 * eta3
        ) / (1.0 + 2.3903703875353255 * eta - 19.03836730923657 * eta2)

        eq_spin = (
            s
            * (
                1.150925084789774
                - 0.3072381261655531 * s
                + eta4 * (12160.22339193134 - 1459.725263347619 * s - 9581.694749116636 * s2)
                + eta2 * (1240.3900459406875 - 289.48392062629966 * s - 1218.1320231846412 * s2)
                - 1.6191217310064605 * s2
                + eta * (-41.38762957457647 + 60.47477582076457 * s2)
                + eta3 * (-7540.259952257055 + 1379.3429194626635 * s + 6372.99271204178 * s2)
            )
        ) / (-1.4325421225106187 + 1.0 * s)

        uneq_spin = dchi * delta * eta3 * (-444.797248674011 + 1448.47758082697 * eta + 152.49290092435044 * s)

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 5 coefficients
        no_spin = (
            eta * (-5.126358906504587 - 227.46830225846668 * eta + 688.3609087244353 * eta2 - 751.4184178636324 * eta3)
        ) / (-0.004551938711031158 - 0.7811680872741462 * eta + 1.0 * eta2)

        eq_spin = (
            s
            * (
                0.1549280856660919
                - 0.9539250460041732 * s
                - 539.4071941841604 * eta2 * s
                + eta * (73.79645135116367 - 8.13494176717772 * s2)
                - 2.84311102369862 * s2
                + eta3 * (-936.3740515136005 + 1862.9097047992134 * s + 224.77581754671272 * s2)
            )
        ) / (-1.5308507364054487 + 1.0 * s)

        uneq_spin = 2993.3598520496153 * dchi * delta * eta6

        return no_spin + eq_spin + uneq_spin

    return jax.lax.cond(int_phase_flag == 104, lambda _: case_104(), lambda _: case_105(), operand=None)


def imr_phenom_x_intermediate_phase_22_v2(
    eta: float, s: float, dchi: float, delta: float, int_phase_flag: int
) -> float:
    """
    Intermediate phase collocation point v2. See Section VII.B of arXiv:2001.11412.
    """

    eta2 = eta * eta
    eta3 = eta2 * eta

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    is_valid = (int_phase_flag == 104) | (int_phase_flag == 105)
    checkify.check(
        is_valid,
        "Error in imr_phenom_x_intermediate_phase_22_v2: IMRPhenomXIntermediatePhaseVersion is not valid. Recommended flag is 104.",
    )

    def case_104():  # Canonical, 4 coefficients

        no_spin = (-84.09400000000004 - 1782.8025405571802 * eta + 5384.38721936653 * eta2) / (
            1.0 + 28.515617312596103 * eta + 12.404097877099353 * eta2
        )

        eq_spin = (
            s
            * (
                22.5665046165141
                - 39.94907120140026 * s
                + 4.668251961072 * s2
                + 12.648584361431245 * s3
                + eta2
                * (
                    -298.7528127869681
                    + 14.228745354543983 * s
                    + 398.1736953382448 * s2
                    + 506.94924905801673 * s3
                    - 626.3693674479357 * s4
                )
                - 5.360554789680035 * s4
                + eta
                * (152.07900889608595 - 121.70617732909962 * s2 - 169.36311036967322 * s3 + 182.40064586992762 * s4)
            )
        ) / (-1.1571201220629341 + 1.0 * s)

        uneq_spin = dchi * delta * eta3 * (5357.695021063607 - 15642.019339339662 * eta + 674.8698102366333 * s)

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 5 coefficients

        no_spin = (
            -82.54500000000004 - 5.58197349185435e6 * eta - 3.5225742421184325e8 * eta2 + 1.4667258334378073e9 * eta3
        ) / (1.0 + 66757.12830903867 * eta + 5.385164380400193e6 * eta2 + 2.5176585751772933e6 * eta3)

        eq_spin = (
            s
            * (
                19.416719811164853
                - 36.066611959079935 * s
                - 0.8612656616290079 * s2
                + eta2 * (170.97203068800542 - 107.41099349364234 * s - 647.8103976942541 * s3)
                + 5.95010003393006 * s3
                + eta3
                * (
                    -1365.1499998427248
                    + 1152.425940764218 * s
                    + 415.7134909564443 * s2
                    + 1897.5444343138167 * s3
                    - 866.283566780576 * s4
                )
                + 4.984750041013893 * s4
                + eta
                * (
                    207.69898051583655
                    - 132.88417400679026 * s
                    - 17.671713040498304 * s2
                    + 29.071788188638315 * s3
                    + 37.462217031512786 * s4
                )
            )
        ) / (-1.1492259468169692 + 1.0 * s)

        uneq_spin = dchi * delta * eta3 * (7343.130973149263 - 20486.813161100774 * eta + 515.9898508588834 * s)

        return no_spin + eq_spin + uneq_spin

    return jax.lax.cond(int_phase_flag == 104, lambda _: case_104(), lambda _: case_105(), operand=None)


def imr_phenom_x_intermediate_phase_22_d43(
    eta: float, s: float, dchi: float, delta: float, int_phase_flag: int
) -> float:
    """
    d43 = v4 - v3. See Section VII.B of arXiv:2001.11412.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    s2 = s * s

    is_valid = int_phase_flag == 105
    checkify.check(
        is_valid,
        "Error in imr_phenom_x_intermediate_phase_22_d43: Point d43 should not be called when using IMRPhenomXIntermediatePhaseVersion == 104.",
    )

    # Canonical, 5 coefficients */
    no_spin = (
        0.4248820426833804
        - 906.746595921514 * eta
        - 282820.39946006844 * eta2
        - 967049.2793750163 * eta3
        + 670077.5414916876 * eta4
    ) / (1.0 + 1670.9440812294847 * eta + 19783.077247023448 * eta2)

    eq_spin = (
        s
        * (
            0.22814271667259703
            + 1.1366593671801855 * s
            + eta3 * (3499.432393555856 - 877.8811492839261 * s - 4974.189172654984 * s2)
            + eta * (12.840649528989287 - 61.17248283184154 * s2)
            + 0.4818323187946999 * s2
            + eta2 * (-711.8532052499075 + 269.9234918621958 * s + 941.6974723887743 * s2)
            + eta4 * (-4939.642457025497 - 227.7672020783411 * s + 8745.201037897836 * s2)
        )
    ) / (-1.2442293719740283 + 1.0 * s)

    uneq_spin = dchi * delta * (-514.8494071830514 + 1493.3851099678195 * eta) * eta3

    return no_spin + eq_spin + uneq_spin


def imr_phenom_x_intermediate_phase_22_ansatz(
        ff: float,
        powers_of_f: IMRPhenomXUsefulPowersDataClass,
        p_wf: IMRPhenomXWaveformDataClass,
        p_phase: IMRPhenomXPhaseCoefficientsDataClass,
) -> float:
    """
    	See section VII. B of arXiv:2001.11412.
        Phase derivative ansatz for intermediate region, Eq. 7.6 of arXiv:2001.11412.
        This is the canonical intermediate ansatz:
        a_0 + a_1 ft^(-1) + a_2 ft^(-2) + a_3 ft^(-3) + a4 ft^(-4) + (4 * a_RD) / ( (2 * f_fdamp)^2 + (f - f_ring)^2 )
        ft = (f / f_ring)
    """
    invff1 = powers_of_f.m_one
    invff2 = powers_of_f.m_two
    invff3 = powers_of_f.m_three
    invff4 = powers_of_f.m_four

    fda    = p_wf.f_damp
    frd    = p_wf.f_ring
    # /* We pass the GR Lorentzian term to make sure that variations to the 
    # GR coefficients in the ringdown decouple from the intermediate regime */
    cL     = p_phase.c_lgr


    int_phase_version = p_wf.imr_phenom_x_intermediate_phase_version

    checkify.check(
        jnp.logical_or(int_phase_version == 104, int_phase_version == 105),
        "Error in IMRPhenomX_Intermediate_Phase_22_Ansatz: IMRPhenomXIntermediatePhaseVersion is not valid. Recommended flag is 104.",
    )

    def case_104(): # Canonical, 4 coefficients
        # /* This is the Lorentzian term where cL = - a_{RD} dphase0 */
        lorentzian_term = (4.0 * cL) / ( (4.0*fda*fda) + (ff - frd)*(ff - frd) )

        # /* Return a polynomial embedded in the background from the merger */
        phase_out = p_phase.b0 + p_phase.b1*invff1 + p_phase.b2*invff2 + p_phase.b4*invff4 + lorentzian_term

        return phase_out
    
    def case_105(): # Canonical, 5 coefficients
        # /* This is the Lorentzian term where cL = - a_{RD} dphase0 */
        lorentzian_term = (4.0 * cL) / ( (4.0*fda*fda) + (ff - frd)*(ff - frd) )

        # /* Return a polynomial embedded in the background from the merger */
        phase_out = (p_phase.b0) + (p_phase.b1)*invff1 + (p_phase.b2)*invff2 + (p_phase.b3)*invff3 + (p_phase.b4)*invff4 + lorentzian_term

        return phase_out
        

    return jax.lax.cond(
        int_phase_version == 104,
        lambda _: case_104(),
        lambda _: case_105(),
        operand=None,
    )


def imr_phenom_x_intermediate_phase_22_ansatz_int(
        f: float,
        powers_of_f: IMRPhenomXUsefulPowersDataClass,
        p_wf: IMRPhenomXWaveformDataClass,
        p_phase: IMRPhenomXPhaseCoefficientsDataClass,
) -> float:
    """
    	See section VII. B of arXiv:2001.11412.
        Integrated phase ansatz for intermediate region, Eq. 7.6 of arXiv:2001.11412.
        Effective spin parameterization used = StotR
    """
  
    invff1 = powers_of_f.m_one
    invff2 = powers_of_f.m_two
    invff3 = powers_of_f.m_three
    logfv   = powers_of_f.log

    frd = p_wf.f_ring
    fda = p_wf.f_damp

    b0  = p_phase.b0
    b1  = p_phase.b1
    b2  = p_phase.b2
    b3  = p_phase.b3
    b4  = p_phase.b4

    # /* We pass the GR Lorentzian term to make sure that variations to the GR coefficients in the ringdown decouple from the intermediate regime */
    cL  = p_phase.c_lgr

    int_phase_version = p_wf.imr_phenom_x_intermediate_phase_version

    checkify.check(
        jnp.logical_or(int_phase_version == 104, int_phase_version == 105),
        "Error in IMRPhenomX_Intermediate_Phase_22_AnsatzInt: IMRPhenomXIntermediatePhaseVersion is not valid. Recommended flag is 104.",
    )

    def case_104(): # Canonical, 4 coefficients
        phase_out = b0*f + b1*logfv - b2*invff1 - (b4*invff3/3.0) + ( 2.0*cL*jax.numpy.arctan( (f - frd) / (2.0 * fda) ) ) / fda 

        return phase_out
    
    def case_105(): # Canonical, 5 coefficients
        phase_out = b0*f + b1*logfv - b2*invff1 - b3*invff2/2.0 - (b4*invff3/3.0) + ( 2.0 * cL * jax.numpy.arctan( (f - frd) / (2.0 * fda) ) ) / fda 

        return phase_out

    return jax.lax.cond(
        int_phase_version == 104,
        lambda _: case_104(),
        lambda _: case_105(),
        operand=None,
    )
