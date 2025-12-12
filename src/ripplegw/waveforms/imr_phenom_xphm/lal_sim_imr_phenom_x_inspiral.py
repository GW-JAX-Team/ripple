"""LAL's IMRPhenomX_inspiral.c JAX implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ripplegw.typing import Array
from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_internals_dataclass import (
    IMRPhenomXPhaseCoefficientsDataClass,
    IMRPhenomXUsefulPowersDataClass,
)


def imr_phenom_x_inspiral_phase_22_ansatz(
    mf: float, powers_of_mf: IMRPhenomXUsefulPowersDataClass, p_phase: IMRPhenomXPhaseCoefficientsDataClass
) -> float | Array:
    """Docstring because I want pre commit to shut it"""

    # Assemble PN phase derivative series
    phase_in = p_phase.dphi_minus2 / powers_of_mf.two_thirds  # f^{-2/3}, v =-2  -1PN
    phase_in += p_phase.dphi_minus1 / powers_of_mf.one_third  # f^{-1/3}, v =-1  -0.5PN
    phase_in += p_phase.dphi0  # f^{0/3}
    phase_in += p_phase.dphi1 * powers_of_mf.one_third  # f^{1/3}
    phase_in += p_phase.dphi2 * powers_of_mf.two_thirds  # f^{2/3}
    phase_in += p_phase.dphi3 * mf  # f^{3/3}
    phase_in += p_phase.dphi4 * powers_of_mf.four_thirds  # f^{4/3}
    phase_in += p_phase.dphi5 * powers_of_mf.five_thirds  # f^{5/3}
    phase_in += p_phase.dphi6 * powers_of_mf.two  # f^{6/3}
    phase_in += p_phase.dphi6l * powers_of_mf.two * powers_of_mf.log  # f^{6/3}, Log[f]
    phase_in += p_phase.dphi7 * powers_of_mf.seven_thirds  # f^{7/3}
    phase_in += p_phase.dphi8 * powers_of_mf.eight_thirds  # f^{8/3}
    phase_in += p_phase.dphi8l * powers_of_mf.eight_thirds * powers_of_mf.log  # f^{8/3}
    phase_in += p_phase.dphi9 * powers_of_mf.three  # f^{9/3}
    phase_in += p_phase.dphi9l * powers_of_mf.three * powers_of_mf.log  # f^{9/3}

    # Add pseudo-PN Coefficient
    phase_in += (
        p_phase.a0 * powers_of_mf.eight_thirds
        + p_phase.a1 * powers_of_mf.three
        + p_phase.a2 * powers_of_mf.eight_thirds * powers_of_mf.two_thirds
        + p_phase.a3 * powers_of_mf.eight_thirds * powers_of_mf.itself
        + p_phase.a4 * powers_of_mf.eight_thirds * powers_of_mf.four_thirds
    )

    phase_in *= powers_of_mf.m_eight_thirds * (5.0 / (128.0 * powers_of_mf.five_thirds))

    return phase_in


def imr_phenom_x_inspiral_phase_22_v3(eta: float, s: float, dchi: float, delta: float, insp_phase_flag: int) -> float:
    """
    Value of phase collocation point at v_3. See section VII.A of arXiv:2001.11412

    Effective spin parameterization used = chiPNHat
    """

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    s2 = s * s
    s3 = s2 * s

    dchi2 = dchi * dchi

    is_valid = (insp_phase_flag == 104) | (insp_phase_flag == 105) | (insp_phase_flag == 114) | (insp_phase_flag == 115)
    checkify.check(is_valid, "Error in IMRPhenomX_Inspiral_Phase_22_v3: NPseudoPN requested is not valid.")

    def case_104():  # Canonical, 3 pseudo PN terms
        no_spin = (
            15415.000000000007
            + 873401.6255736464 * eta
            + 376665.64637025696 * eta2
            - 3.9719980569125614e6 * eta3
            + 8.913612508054944e6 * eta4
        ) / (1.0 + 46.83697749859996 * eta)

        eq_spin = (
            s
            * (
                397951.95299014193
                - 207180.42746987 * s
                + eta3 * (4.662143741417853e6 - 584728.050612325 * s - 1.6894189124921719e6 * s2)
                + eta * (-1.0053073129700898e6 + 1.235279439281927e6 * s - 174952.69161683554 * s2)
                - 130668.37221912303 * s2
                + eta2 * (-1.9826323844247842e6 + 208349.45742548333 * s + 895372.155565861 * s2)
            )
        ) / (-9.675704197652225 + 3.5804521763363075 * s + 2.5298346636273306 * s2 + 1.0 * s3)

        uneq_spin = -1296.9289110696955 * dchi2 * eta + dchi * delta * eta * (
            -24708.109411857182 + 24703.28267342699 * eta + 47752.17032707405 * s
        )

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 4 pseudo PN terms

        no_spin = (
            11717.332402222377
            + 4.972361134612872e6 * eta
            + 2.137585030930089e7 * eta2
            - 8.882868155876668e7 * eta3
            + 2.4104945956043008e8 * eta4
            - 2.3143426271719798e8 * eta5
        ) / (1.0 + 363.5524719849582 * eta)

        eq_spin = (
            s
            * (
                52.42001436116159
                - 50.547943589389966 * s
                + eta3 * s * (-15355.56020802297 + 20159.588079899433 * s)
                + eta2 * (-286.9576245212502 + 2795.982637986682 * s - 2633.1870842242447 * s2)
                - 1.0824224105690476 * s2
                + eta * (-123.78531181532225 + 136.1961976556154 * s - 7.534890781638552 * s3)
                + 5.973206330664007 * s3
                + eta4 * (1777.2176433016125 + 24069.288079063674 * s - 44794.9522164669 * s2 + 1584.1515277998406 * s3)
            )
        ) / (-0.0015307616935628491 + (0.0010676159178395538 - 0.25 * eta3 + 1.0 * eta4) * s)

        uneq_spin = -1357.9794908614106 * dchi2 * eta + dchi * delta * eta * (
            -23093.829989687543 + 21908.057881789653 * eta + 49493.91485992256 * s
        )

        return no_spin + eq_spin + uneq_spin

    def case_114():  # Extended, 3 pseudo PN terms
        no_spin = (
            68014.00000000003 + 1.1513072539654972e6 * eta - 2.725589921577228e6 * eta2 + 312571.92531733884 * eta3
        ) / (1.0 + 17.48539665509149 * eta)

        eq_spin = (
            s
            * (
                -34467.00643820664
                + 99693.81839115614 * eta
                + 144345.24343461913 * eta4
                + (23618.044919850676 - 89494.69555164348 * eta + 725554.5749749158 * eta4 - 103449.15865381068 * eta2)
                * s
                + (
                    10350.863429774612
                    - 73238.45609787296 * eta
                    + 3.559251543095961e6 * eta4
                    + 888228.5439003729 * eta2
                    - 3.4602940487291473e6 * eta3
                )
                * s2
            )
        ) / (1.0 - 0.056846656084188936 * s - 0.32681474740130184 * s2 - 0.30562055811022015 * s3)

        uneq_spin = -1182.4036752941936 * dchi2 * eta + dchi * delta * eta * (
            -0.39185419821851025 - 99764.21095663306 * eta + 41826.177356107364 * s
        )

        return no_spin + eq_spin + uneq_spin

    def case_115():  # Extended, 4 pseudo PN terms
        no_spin = (
            60484.00000000003
            + 4.370611564781374e6 * eta
            - 5.785128542827255e6 * eta2
            - 8.82141241633613e6 * eta3
            + 1.3406996319926713e7 * eta4
        ) / (1.0 + 70.89393713617065 * eta)

        eq_spin = (
            s
            * (
                21.91241092620993
                - 32.57779678272056 * s
                + eta2 * (-102.4982890239095 + 2570.7566494633033 * s - 2915.1250015652076 * s2)
                + 8.130585173873232 * s2
                + eta * (-28.158436727309365 + 47.42665852824481 * s2)
                + eta3 * (-1635.6270690726785 - 13745.336370568011 * s + 19539.310912464192 * s2)
                + 1.2792283911312285 * s3
                + eta4 * (5558.382039622131 + 21398.7730201213 * s - 37773.40511355719 * s2 + 768.6183344184254 * s3)
            )
        ) / (-0.0007758753818017038 + (0.0005304023864415552 - 0.25000000000000006 * eta3 + 1.0 * eta4) * s)

        uneq_spin = -1223.769262298142 * dchi2 * eta + dchi * delta * eta * (
            -16.705471562129436 - 93771.93750060834 * eta + 43675.70151058481 * s
        )

        return no_spin + eq_spin + uneq_spin

    flag_choices = jnp.array([104, 105, 114, 115])
    index = jnp.argmax(flag_choices == insp_phase_flag)

    cases = (
        lambda _: case_104(),
        lambda _: case_105(),
        lambda _: case_114(),
        lambda _: case_115(),
    )

    return jax.lax.switch(index, cases, operand=None)


def imr_phenom_x_inspiral_phase_22_d13(eta: float, s: float, dchi: float, delta: float, insp_phase_flag: int) -> float:
    """
    Value of phase collocation point for d13 = v1 - v3. See section VII.A of arXiv:2001.11412

    Effective spin parameterization used = chiPNHat
    """

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    is_valid = (insp_phase_flag == 104) | (insp_phase_flag == 105) | (insp_phase_flag == 114) | (insp_phase_flag == 115)
    checkify.check(is_valid, "Error in IMRPhenomX_Inspiral_Phase_22_d13: NPseudoPN requested is not valid.")

    def case_104():  # Canonical, 3 pseudo PN terms
        no_spin = (-17294.000000000007 - 19943.076428555978 * eta + 483033.0998073767 * eta2) / (
            1.0 + 4.460294035404433 * eta
        )

        eq_spin = (
            s
            * (
                68384.62786426462
                + 67663.42759836042 * s
                - 2179.3505885609297 * s2
                + eta * (-58475.33302037833 + 62190.404951852535 * s + 18298.307770807573 * s2 - 303141.1945565486 * s3)
                + 19703.894135534803 * s3
                + eta2
                * (-148368.4954044637 - 758386.5685734496 * s - 137991.37032619823 * s2 + 1.0765877367729193e6 * s3)
                + 32614.091002011017 * s4
            )
        ) / (2.0412979553629143 + 1.0 * s)

        uneq_spin = 12017.062595934838 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 4 pseudo PN terms
        no_spin = (-14234.000000000007 + 16956.107542097994 * eta + 176345.7518697656 * eta2) / (
            1.0 + 1.294432443903631 * eta
        )

        eq_spin = (
            s
            * (
                814.4249470391651
                + 539.3944162216742 * s
                + 1985.8695471257474 * s2
                + eta * (-918.541450687484 + 2531.457116826593 * s - 14325.55532310136 * s2 - 19213.48002675173 * s3)
                + 1517.4561843692861 * s3
                + eta2
                * (-517.7142591907573 - 14328.567448748548 * s + 21305.033147575057 * s2 + 50143.99945676916 * s3)
            )
        ) / (
            0.03332712934306297
            + 0.0025905919215826172 * s
            + (0.07388087063636305 - 0.6397891808905215 * eta + 1.0 * eta2) * s2
        )

        uneq_spin = dchi * delta * eta * (0.09704682517844336 + 69335.84692284222 * eta)

        return no_spin + eq_spin + uneq_spin

    def case_114():  # Extended, 3 pseudo PN terms

        no_spin = (
            -36664.000000000015
            + 277640.10051158903 * eta
            - 581120.4916255298 * eta2
            + 1.415628418251648e6 * eta3
            - 7.640937162029471e6 * eta4
            + 1.1572710625359124e7 * eta5
        ) / (1.0 - 4.011038704323779 * eta)

        eq_spin = (
            s
            * (
                -38790.01253014577
                - 50295.77273512981 * s
                + 15182.324439704937 * s2
                + eta2 * (57814.07222969789 + 344650.11918139807 * s + 17020.46497164955 * s2 - 574047.1384792664 * s3)
                + 24626.598127509922 * s3
                + eta
                * (23058.264859112394 - 16563.935447608965 * s - 36698.430436426395 * s2 + 105713.91549712936 * s3)
            )
        ) / (-1.5445637219268247 - 0.24997068896075847 * s + 1.0 * s2)

        uneq_spin = 74115.77361380383 * dchi * delta * eta2

        return no_spin + eq_spin + uneq_spin

    def case_115():  # Extended, 4 pseudo PN terms
        no_spin = (
            -29240.00000000001 - 12488.41035199958 * eta + 1.3911845288427814e6 * eta2 - 3.492477584609041e6 * eta3
        ) / (1.0 + 2.6711462529779824 * eta - 26.80876660227278 * eta2)

        eq_spin = (
            s
            * (
                -29536.155624432842
                - 40024.5988680615 * s
                + 11596.401177843705 * s2
                + eta2 * (122185.06346551726 + 351091.59147835104 * s - 37366.6143666202 * s2 - 505834.54206320125 * s3)
                + 20386.815769841945 * s3
                + eta * (-9638.828456576934 - 30683.453790630676 * s - 15694.962417099561 * s2 + 91690.51338194775 * s3)
            )
        ) / (-1.5343852108869265 - 0.2215651087041266 * s + 1.0 * s2)

        uneq_spin = 68951.75344813892 * dchi * delta * eta2

        return no_spin + eq_spin + uneq_spin

    flag_choices = jnp.array([104, 105, 114, 115])
    index = jnp.argmax(flag_choices == insp_phase_flag)

    cases = (
        lambda _: case_104(),
        lambda _: case_105(),
        lambda _: case_114(),
        lambda _: case_115(),
    )

    return jax.lax.switch(index, cases, operand=None)


def imr_phenom_x_inspiral_phase_22_d23(eta: float, s: float, dchi: float, delta: float, insp_phase_flag: int) -> float:
    """
    Value of phase collocation point for d23 = v2 - v3. See section VII.A of arXiv:2001.11412

    Effective spin parameterization used = chiPNHat
    """

    eta2 = eta * eta
    eta3 = eta2 * eta

    dchi2 = dchi * dchi

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    is_valid = (insp_phase_flag == 104) | (insp_phase_flag == 105) | (insp_phase_flag == 114) | (insp_phase_flag == 115)
    checkify.check(is_valid, "Error in IMRPhenomX_Inspiral_Phase_22_d23: NPseudoPN requested is not valid.")

    def case_104():  # Canonical, 4 pseudo PN terms
        no_spin = (
            -7579.300000000004 - 120297.86185566607 * eta + 1.1694356931282217e6 * eta2 - 557253.0066989232 * eta3
        ) / (1.0 + 18.53018618227582 * eta)

        eq_spin = (
            s
            * (
                -27089.36915061857
                - 66228.9369155027 * s
                + eta2 * (150022.21343386435 - 50166.382087278434 * s - 399712.22891153296 * s2)
                - 44331.41741405198 * s2
                + eta * (50644.13475990821 + 157036.45676788126 * s + 126736.43159783827 * s2)
                + eta3 * (-593633.5370110178 - 325423.99477314285 * s + 847483.2999508682 * s2)
            )
        ) / (-1.5232497464826662 - 3.062957826830017 * s - 1.130185486082531 * s2 + 1.0 * s3)

        uneq_spin = 3843.083992827935 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 5 pseudo PN terms
        no_spin = (-7520.900000000003 - 49463.18828584058 * eta + 437634.8057596484 * eta2) / (
            1.0 + 9.10538019868398 * eta
        )

        eq_spin = (
            s
            * (
                25380.485895523005
                + 30617.553968012628 * s
                + 5296.659585425608 * s2
                + eta * (-49447.74841021405 - 94312.78229903466 * s - 5731.614612941746 * s3)
                + 2609.50444822972 * s3
                + 5206.717656940992 * s4
                + eta2
                * (
                    54627.758819129864
                    + 157460.98527210607 * s
                    - 69726.85196686552 * s2
                    + 4674.992397927943 * s3
                    + 20704.368650323784 * s4
                )
            )
        ) / (1.5668927528319367 + 1.0 * s)

        uneq_spin = -95.38600275845481 * dchi2 * eta + dchi * delta * eta * (
            3271.8128884730654 + 12399.826237672185 * eta + 9343.380589951552 * s
        )

        return no_spin + eq_spin + uneq_spin

    def case_114():  # Extended, 4 pseudo PN terms
        no_spin = (-17762.000000000007 - 1.6929191194109183e6 * eta + 8.420903644926643e6 * eta2) / (
            1.0 + 98.061533474615 * eta
        )

        eq_spin = (
            s
            * (
                -46901.6486082098
                - 83648.57463631754 * s
                + eta2 * (1.2502334322912344e6 + 1.4500798116821344e6 * s - 1.4822181506831646e6 * s2)
                - 41234.966418619966 * s2
                + eta * (-24017.33452114588 - 15241.079745314566 * s + 136554.48806839858 * s2)
                + eta3 * (-3.584298922116994e6 - 3.9566921791790277e6 * s + 4.357599992831832e6 * s2)
            )
        ) / (-3.190220646817508 - 3.4308485421201387 * s - 0.6347932583034377 * s2 + 1.0 * s3)

        uneq_spin = 24906.33337911219 * dchi * delta * eta2

        return no_spin + eq_spin + uneq_spin

    def case_115():  # Extended, 5 pseudo PN terms
        no_spin = (
            -18482.000000000007 - 1.2846128476247871e6 * eta + 4.894853535651343e6 * eta2 + 3.1555931338015324e6 * eta3
        ) / (1.0 + 82.79386070797756 * eta)

        eq_spin = (
            s
            * (
                -19977.10130179636
                - 24729.114403562427 * s
                + 10325.295899053815 * s2
                + eta * (30934.123894659646 + 58636.235226102894 * s - 32465.70372990005 * s2 - 38032.16219587224 * s3)
                + 15485.725898689267 * s3
                + eta2 * (-38410.1127729419 - 87088.84559983511 * s + 61286.73536122325 * s2 + 42503.913487705235 * s3)
            )
        ) / (-1.5148031011828222 - 0.24267195338525768 * s + 1.0 * s2)

        uneq_spin = 5661.027157084334 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    flag_choices = jnp.array([104, 105, 114, 115])
    index = jnp.argmax(flag_choices == insp_phase_flag)

    cases = (
        lambda _: case_104(),
        lambda _: case_105(),
        lambda _: case_114(),
        lambda _: case_115(),
    )
    return jax.lax.switch(index, cases, operand=None)


def imr_phenom_x_inspiral_phase_22_d43(eta: float, s: float, dchi: float, delta: float, insp_phase_flag: int) -> float:
    """
    Value of phase collocation point for d43 = v4 - v3. See section VII.A of arXiv:2001.11412

    Effective spin parameterization used = chiPNHat
    """

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    s2 = s * s
    s3 = s2 * s

    is_valid = (insp_phase_flag == 104) | (insp_phase_flag == 105) | (insp_phase_flag == 114) | (insp_phase_flag == 115)
    checkify.check(is_valid, "Error in IMRPhenomX_Inspiral_Phase_22_d43: NPseudoPN requested is not valid.")

    def case_104():  # Canonical, 3 pseudo PN coefficients
        no_spin = (2439.000000000001 - 31133.52170083207 * eta + 28867.73328134167 * eta2) / (
            1.0 + 0.41143032589262585 * eta
        )

        eq_spin = (
            s
            * (
                16116.057657391262
                + eta3 * (-375818.0132734753 - 386247.80765802023 * s)
                + eta * (-82355.86732027541 - 25843.06175439942 * s)
                + 9861.635308837876 * s
                + eta2 * (229284.04542668918 + 117410.37432997991 * s)
            )
        ) / (-3.7385208695213668 + 0.25294420589064653 * s + 1.0 * s2)

        uneq_spin = 194.5554531509207 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    def case_105():  # Canonical, 4 pseudo PN coefficients
        no_spin = (
            4085.300000000002
            + 62935.7755506329 * eta
            - 1.3712743918777364e6 * eta2
            + 5.024685134555112e6 * eta3
            - 3.242443755025284e6 * eta4
        ) / (1.0 + 20.889132970603523 * eta - 99.39498823723363 * eta2)

        eq_spin = (
            s
            * (
                -299.6987332025542
                - 106.2596940493108 * s
                + eta3 * (2383.4907865977148 - 13637.11364447208 * s - 14808.138346145908 * s2)
                + eta * (1205.2993091547498 - 508.05838536573464 * s - 1453.1997617403304 * s2)
                + 132.22338129554674 * s2
                + eta2 * (-2438.4917103042208 + 5032.341879949591 * s + 7026.9206794027405 * s2)
            )
        ) / (0.03089183275944264 + 1.0 * eta3 * s - 0.010670764224621489 * s2)

        uneq_spin = -1392.6847421907178 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    def case_114():  # Extended, 3 pseudo PN coefficients
        no_spin = (5749.000000000003 - 37877.95816426952 * eta) / (1.0 + 1.1883386102990128 * eta)

        eq_spin = (
            (-4285.982163759047 + 24558.689969419473 * eta - 49270.2296311733 * eta2) * s
            + eta * (-24205.71407420114 + 70777.38402634041 * eta) * s2
            + (2250.661418551257 + 187.95136178643946 * eta - 11976.624134935797 * eta2) * s3
        ) / (1.0 - 0.7220334077284601 * s)

        uneq_spin = dchi * delta * eta * (339.69292150803585 - 3459.894150148715 * s)

        return no_spin + eq_spin + uneq_spin

    def case_115():  # Extended, 4 pseudo PN coefficients
        no_spin = (
            9760.400000000005 + 9839.852773121198 * eta - 398521.0434645335 * eta2 + 267575.4709475981 * eta3
        ) / (1.0 + 6.1355249449135005 * eta)

        eq_spin = (
            s
            * (
                -1271.406488219572
                + eta2 * (-9641.611385554736 - 9620.333878140807 * s)
                - 1144.516395171019 * s
                + eta * (5155.337817255137 + 4450.755534615418 * s)
            )
        ) / (
            0.1491519640750958
            + (-0.0008208549820159909 - 0.15468508831447628 * eta + 0.7266887643762937 * eta2) * s
            + (0.02282542856845755 - 0.445924460572114 * eta + 1.0 * eta2) * s2
        )

        uneq_spin = -1366.7949288045616 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    flag_choices = jnp.array([104, 105, 114, 115])
    index = jnp.argmax(flag_choices == insp_phase_flag)

    cases = (
        lambda _: case_104(),
        lambda _: case_105(),
        lambda _: case_114(),
        lambda _: case_115(),
    )
    return jax.lax.switch(index, cases, operand=None)


def imr_phenom_x_inspiral_phase_22_d53(eta: float, s: float, dchi: float, delta: float, insp_phase_flag: int) -> float:
    """
    Value of phase collocation point for d53 = v5 - v3. See section VII.A of arXiv:2001.11412

    Effective spin parameterization used = chiPNHat
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    s2 = s * s

    # 104 and 114 should not be called for 4 pseudo-PN coefficients
    check_4pn = jnp.logical_or(insp_phase_flag == 114, insp_phase_flag == 104)
    checkify.check(
        jnp.logical_not(check_4pn),
        "Calling IMRPhenomX_Inspiral_Phase_22_d53 but trying to pass "
        "InspPhaseFlag for 4 pseudo-PN coefficients. Check this.",
    )

    is_valid = jnp.logical_or(insp_phase_flag == 105, insp_phase_flag == 115)
    checkify.check(is_valid, "Error in IMRPhenomX_Inspiral_Phase_22_d53: NPseudoPN requested is not valid.")

    def case_105():  # Canonical, 5 pseudo PN coefficients
        no_spin = (
            5474.400000000003 + 131008.0112992443 * eta - 1.9692364337640922e6 * eta2 + 1.8732325307375633e6 * eta3
        ) / (1.0 + 32.90929274981482 * eta)

        eq_spin = (
            s
            * (
                18609.016486281424
                - 1337.4947536109685 * s
                + eta2 * (219014.98908698096 - 307162.33823247004 * s - 124111.02067626518 * s2)
                - 7394.9595046977365 * s2
                + eta * (-87820.37490863055 + 53564.4178831741 * s + 34070.909093771494 * s2)
                + eta3 * (-248096.84257893753 + 536024.5354098587 * s + 243877.65824670633 * s2)
            )
        ) / (-1.5282904337787517 + 1.0 * s)

        uneq_spin = -429.1148607925461 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    def case_115():  # Extended, 5 pseudo PN terms

        no_spin = (12971.000000000005 - 93606.05144508784 * eta + 102472.4473167639 * eta2) / (
            1.0 - 0.8909484992212859 * eta
        )

        eq_spin = (
            s
            * (
                16182.268123259992
                + 3513.8535400032874 * s
                + eta2 * (343388.99445324624 - 240407.0282222587 * s - 312202.59917289804 * s2)
                - 10814.056847109632 * s2
                + eta * (-94090.9232151429 + 35305.66247590705 * s + 65450.36389642103 * s2)
                + eta3 * (-484443.15601144277 + 449511.3965208116 * s + 552355.592066788 * s2)
            )
        ) / (-1.4720837917195788 + 1.0 * s)

        uneq_spin = -494.2754225110706 * dchi * delta * eta

        return no_spin + eq_spin + uneq_spin

    return jax.lax.cond(insp_phase_flag == 105, lambda _: case_105(), lambda _: case_115(), operand=None)
