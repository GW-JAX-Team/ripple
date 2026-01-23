"""
IMRPhenomXHM inspiral amplitude functions converted to JAX.

This module contains the inspiral amplitude coefficient functions for the
IMRPhenomXHM waveform model, converted from the LALSimulation C code to JAX.
"""

import jax.numpy as jnp


def IMRPhenomXHM_Insp_Amp_21_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.037868557189995156 + 0.10740090317702103 * eta + 1.963812986867654 * eta2
            - 16.706455229589558 * eta3 + 69.75910808095745 * eta4 - 98.3062466823662 * eta5
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.007963757232702219 + 0.10627108779259965 * eta - 0.008044970210401218 * S
            + eta2 * (-0.4735861262934258 - 0.5985436493302649 * S - 0.08217216660522082 * S2)
        )

        uneqSpin = (
            -0.257787704938017 * pWF['dchi'] * eta2 * (1. + 8.75928187268504 * eta2)
            - 0.2597503605427412 * pWF['dchi'] * eta2 * S
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = jnp.abs(
            chidiff1 * eta5 * (-3962.5020052272976 + 987.635855365408 * S1 - 134.98527058315528 * S2)
            + delta * (19.30531354642419 + 16.6640319856064 * eta1 - 120.58166037019478 * eta2
                      + 220.77233521626252 * eta3) * sqroot
            + chidiff1 * delta * (31.364509907424765 * eta1 - 843.6414532232126 * eta2
                                 + 2638.3077554662905 * eta3) * sqroot
            + chidiff1 * delta * (32.374226994179054 * eta1 - 202.86279451816662 * eta2
                                 + 347.1621871204769 * eta3) * S1 * sqroot
            + delta * S1 * (
                -16.75726972301224 * (1.1787350890261943 - 7.812073811917883 * eta1
                                     + 99.47071002831267 * eta2 - 500.4821414428368 * eta3
                                     + 876.4704270866478 * eta4)
                + 2.3439955698372663 * (0.9373952326655807 + 7.176140122833879 * eta1
                                       - 279.6409723479635 * eta2 + 2178.375177755584 * eta3
                                       - 4768.212511142035 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_21_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.05511628628738656 - 0.12579599745414977 * eta + 2.831411618302815 * eta2
            - 14.27268643447161 * eta3 + 28.3307320191161 * eta4
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.008692738851491525 + eta * (0.09512553997347649 + 0.116470975986383 * S)
            - 0.009520793625590234 * S + eta2 * (-0.3409769288480959 - 0.8321002363767336 * S
                                                   - 0.13099477081654226 * S2)
            - 0.006383232900211555 * S2
        )

        uneqSpin = (
            -0.2962753588645467 * pWF['dchi'] * eta2 * (1. + 1.3993978458830476 * eta2)
            - 0.17100612756133535 * pWF['dchi'] * eta2 * S * (1. + 18.974303741922743 * eta2 * delta)
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = jnp.abs(
            chidiff1 * eta5 * (-2898.9172078672705 + 580.9465034962822 * S1 + 22.251142639924076 * S2)
            + delta * (
                chidiff2 * (-18.541685007214625 * eta1 + 166.7427445020744 * eta2 - 417.5186332459383 * eta3)
                + chidiff1 * (41.61457952037761 * eta1 - 779.9151607638761 * eta2 + 2308.6520892707795 * eta3)
            ) * sqroot
            + delta * (11.414934585404561 + 30.883118528233638 * eta1 - 260.9979123967537 * eta2
                      + 1046.3187137392433 * eta3 - 1556.9475493549746 * eta4) * sqroot
            + delta * S1 * (
                -10.809007068469844 * (1.1408749895922659 - 18.140470190766937 * eta1
                                      + 368.25127088896744 * eta2 - 3064.7291458207815 * eta3
                                      + 11501.848278358668 * eta4 - 16075.676528787526 * eta5)
                + 1.0088254664333147 * (1.2322739396680107 - 192.2461213084741 * eta1
                                       + 4257.760834055382 * eta2 - 35561.24587952242 * eta3
                                       + 130764.22485304279 * eta4 - 177907.92440833704 * eta5) * S1
            ) * sqroot
            + delta * (
                chidiff1 * (36.88578491943111 * eta1 - 321.2569602623214 * eta2 + 748.6659668096737 * eta3) * S1
                + chidiff1 * (-95.42418611585117 * eta1 + 1217.338674959742 * eta2
                             - 3656.192371615541 * eta3) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_21_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 21 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 21 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.059110044024271766 - 0.0024538774422098405 * eta + 0.2428578654261086 * eta2
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            -0.007044339356171243 - 0.006952154764487417 * S
            + eta2 * (-0.016643018304732624 - 0.12702579620537421 * S + 0.004623467175906347 * S2)
            - 0.007685497720848461 * S2
        )

        uneqSpin = (
            -0.3172310538516028 * pWF['dchi'] * (1. - 2.9155919835488024 * eta2) * eta2
            - 0.11975485688200693 * pWF['dchi'] * eta2 * S * (1. + 17.27626751837825 * eta2 * delta)
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = jnp.abs(
            chidiff1 * eta5 * (-2282.9983216879655 + 157.94791186394787 * S1 + 16.379731479465033 * S2)
            + chidiff1 * delta * (21.935833431534224 * eta1 - 460.7130131927895 * eta2
                                 + 1350.476411541137 * eta3) * sqroot
            + delta * (5.390240326328237 + 69.01761987509603 * eta1 - 568.0027716789259 * eta2
                      + 2435.4098320959706 * eta3 - 3914.3390484239667 * eta4) * sqroot
            + chidiff1 * delta * (29.731007410186827 * eta1 - 372.09609843131386 * eta2
                                 + 1034.4897198648962 * eta3) * S1 * sqroot
            + delta * S1 * (
                -7.1976397556450715 * (0.7603360145475428 - 6.587249958654174 * eta1
                                      + 120.87934060776237 * eta2 - 635.1835857158857 * eta3
                                      + 1109.0598539312573 * eta4)
                - 0.0811847192323969 * (7.951454648295709 + 517.4039644814231 * eta1
                                       - 9548.970156895082 * eta2 + 52586.63520999897 * eta3
                                       - 93272.17990295641 * eta4) * S1
                - 0.28384547935698246 * (-0.8870770459576875 + 180.0378964169756 * eta1
                                        - 2707.9572896559484 * eta2 + 14158.178124971111 * eta3
                                        - 24507.800226675925 * eta4) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_21_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        S2 = S ** 2

        noSpin = (
            jnp.sqrt(1. - 4. * eta) * (-0.056586690934283326 - 0.14374841547279146 * eta
                                       + 0.5584776628959615 * eta2)
        ) / (-0.3996185676368123 + eta)

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            (0.056042044149691175 + 0.12482426029674777 * S) * S
            + eta * (2.1108074577110343 - 1.7827773156978863 * S2)
            + eta2 * (-7.657635515668849 - 0.07646730296478217 * S + 5.343277927456605 * S2)
        )

        uneqSpin = 0.45866449225302536 * pWF['dchi'] * (1. - 9.603750707244906 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = (
            chidiff1 * eta5 * (155.1434307076563 + 26.852777193715088 * S1 + 1.4157230717300835 * S2)
            + chidiff1 * delta * (6.296698171560171 * eta1 + 15.81328761563562 * eta2
                                 - 141.85538063933927 * eta3) * sqroot
            + delta * (20.94372147101354 + 68.14577638017842 * eta1 - 898.470298591732 * eta2
                      + 4598.64854748635 * eta3 - 8113.199260593833 * eta4) * sqroot
            + chidiff1 * delta * (29.221863857271703 * eta1 - 348.1658322276406 * eta2
                                 + 965.4670353331536 * eta3) * S1 * sqroot
            + delta * S1 * (
                -9.753610761811967 * (1.7819678168496158 - 44.07982999150369 * eta1
                                     + 750.8933447725581 * eta2 - 5652.44754829634 * eta3
                                     + 19794.855873435758 * eta4 - 26407.40988450443 * eta5)
                + 0.014210376114848208 * (-196.97328616330392 + 7264.159472864562 * eta1
                                         - 125763.47850622259 * eta2 + 1.1458022059130718e6 * eta3
                                         - 4.948175330328345e6 * eta4 + 7.911048294733888e6 * eta5) * S1
                - 0.26859293613553986 * (-8.029069605349488 + 888.7768796633982 * eta1
                                        - 16664.276483466252 * eta2 + 128973.72291098491 * eta3
                                        - 462437.2690007375 * eta4 + 639989.1197424605 * eta5) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.2137734510411439 - 0.7692194209223682 * eta + 26.10570221351058 * eta2
            - 316.0643979123107 * eta3 + 2090.9063511488234 * eta4 - 6897.3285171507105 * eta5
            + 8968.893362362503 * eta6
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            0.018546836505210842 + 0.05924304311104228 * S
            + eta * (1.6484440612224325 - 0.4683932646001618 * S - 2.110311135456494 * S2)
            + 0.10701786057882816 * S2
            + eta2 * (-6.51575737684721 + 1.6692205620001157 * S + 8.351789152096782 * S2)
        )

        uneqSpin = 0.3929315188124088 * pWF['dchi'] * (1. - 11.289452844364227 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff

        total = (
            chidiff1 * eta5 * (161.62678370819597 + 37.141092711336846 * S1 - 0.16889712161410445 * S2)
            + chidiff1 * delta * (3.4895829486899825 * eta1 + 51.07954458810889 * eta2
                                 - 249.71072528701757 * eta3) * sqroot
            + delta * (12.501397517602173 + 35.75290806646574 * eta1 - 357.6437296928763 * eta2
                      + 1773.8883882162215 * eta3 - 3100.2396041211605 * eta4) * sqroot
            + chidiff1 * delta * (13.854211287141906 * eta1 - 135.54916401086845 * eta2
                                 + 327.2467193417936 * eta3) * S1 * sqroot
            + delta * S1 * (
                -5.2580116732827085 * (1.7794900975289085 - 48.20753331991333 * eta1
                                      + 861.1650630146937 * eta2 - 6879.681319382729 * eta3
                                      + 25678.53964955809 * eta4 - 36383.824902258915 * eta5)
                + 0.028627002336747746 * (-50.57295946557892 + 734.7581857539398 * eta1
                                         - 2287.0465658878725 * eta2 + 15062.821881048358 * eta3
                                         - 168311.2370167227 * eta4 + 454655.37836367317 * eta5) * S1
                - 0.15528289788512326 * (-12.738184090548508 + 1129.44485109116 * eta1
                                        - 25091.14888164863 * eta2 + 231384.03447562453 * eta3
                                        - 953010.5908118751 * eta4 + 1.4516597366230418e6 * eta5) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_33_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 33 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 33 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 4. * eta) * (
            0.2363760327127446 + 0.2855410252403732 * eta - 10.159877125359897 * eta2
            + 162.65372389693505 * eta3 - 1154.7315106095564 * eta4 + 3952.61320206691 * eta5
            - 5207.67472857814 * eta6
        )

        eqSpin = jnp.sqrt(1. - 4. * eta) * S * (
            0.04573095188775319 + 0.048249943132325494 * S
            + eta * (0.15922377052827502 - 0.1837289613228469 * S - 0.2834348500565196 * S2)
            + 0.052963737236081304 * S2
        )

        uneqSpin = 0.25187274502769835 * pWF['dchi'] * (1. - 12.172961866410864 * eta2) * eta2

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        eta7 = eta1 * eta6
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            chidiff1 * delta * (-0.5869777957488564 * eta1 + 32.65536124256588 * eta2
                               - 110.10276573567405 * eta3)
            + chidiff1 * delta * (3.524800489907584 * eta1 - 40.26479860265549 * eta2
                                 + 113.77466499598913 * eta3) * S1
            + delta * S1 * (
                -1.2846335585108297 * (0.09991079016763821 + 1.37856806162599 * eta1
                                      + 23.26434219690476 * eta2 - 34.842921754693386 * eta3
                                      - 70.83896459998664 * eta4)
                - 0.03496714763391888 * (-0.230558571912664 + 188.38585449575902 * eta1
                                        - 3736.1574640444287 * eta2 + 22714.70643022915 * eta3
                                        - 43221.0453556626 * eta4) * S1
            )
            + chidiff1 * eta7 * (2667.3441342894776 + 47.94869769580204 * chidiff2
                                + 793.5988192446642 * S1 + 293.89657731755483 * S2)
            + delta * (5.148353856800232 + 148.98231189649468 * eta1 - 2774.5868652930294 * eta2
                      + 29052.156454239772 * eta3 - 162498.31493332976 * eta4 + 460912.76402476896 * eta5
                      - 521279.50781871413 * eta6) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_33_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        eta8 = eta6 * eta * eta
        S2 = S ** 2
        S3 = S2 * S

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.019069933430190773 - 0.19396651989685837 * eta + 11.95224600241255 * eta2
            - 158.90113442757382 * eta3 + 1046.65239329071 * eta4 - 3476.940285294999 * eta5
            + 4707.249209858949 * eta6
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.0046910348789512895 + 0.40231360805609434 * eta - 0.0038263656140933152 * S
            + 0.018963579407636953 * S2
            + eta2 * (-1.955352354930108 + 2.3753413452420133 * S - 0.9085620866763245 * S3)
            + 0.02738043801805805 * S3
            + eta3 * (7.977057990568723 - 7.9259853291789515 * S + 0.49784942656123987 * S2
                     + 5.2255665027119145 * S3)
        )

        uneqSpin = (
            0.058560321425018165 * pWF['dchi'] ** 2 * (1. - 19.936477485971217 * eta2) * eta2
            + 1635.4240644598524 * pWF['dchi'] * eta8 * delta
            + 0.2735219358839411 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (-0.739317114582042 * eta1 - 47.473246070362634 * eta2
                                + 278.9717709112207 * eta3 - 566.6420939162068 * eta4)
             + chidiff2 * (-0.5873680378268906 * eta1 + 6.692187014925888 * eta2
                          - 24.37776782232888 * eta3 + 23.783684827838247 * eta4)) * sqroot
            + (3.2940434453819694 + 4.94285331708559 * eta1 - 343.3143244815765 * eta2
              + 3585.9269057886418 * eta3 - 19279.186145681153 * eta4 + 51904.91007211022 * eta5
              - 55436.68857586653 * eta6) * sqroot
            + chidiff1 * delta * (12.488240781993923 * eta1 - 209.32038774208385 * eta2
                                 + 1160.9833883184604 * eta3 - 2069.5349737049073 * eta4) * S1 * sqroot
            + S1 * (
                0.6343034651912586 * (-2.5844888818001737 + 78.98200041834092 * eta1
                                     - 1087.6241783616488 * eta2 + 7616.234910399297 * eta3
                                     - 24776.529123239357 * eta4 + 30602.210950069973 * eta5)
                - 0.062088720220899465 * (6.5586380356588565 + 36.01386705325694 * eta1
                                         - 3124.4712274775407 * eta2 + 33822.437731298516 * eta3
                                         - 138572.93700180828 * eta4 + 198366.10615196894 * eta5) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta8 = eta4 * eta4
        S2 = S ** 2
        S3 = S2 * S

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.024621376891809633 - 0.09692699636236377 * eta + 2.7200998230836158 * eta2
            - 16.160563094841066 * eta3 + 32.930430889650836 * eta4
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.008522695567479373 - 1.1104639098529456 * eta2 - 0.00362963820787208 * S
            + 0.016978054142418417 * S2
            + eta * (0.24280554040831698 + 0.15878436411950506 * S - 0.1470288177047577 * S3)
            + 0.029465887557447824 * S3
            + eta3 * (4.649438233164449 - 0.7550771176087877 * S + 0.3381436950547799 * S2
                     + 2.5663386135613093 * S3)
        )

        uneqSpin = (
            -0.007061187955941243 * pWF['dchi'] ** 2 * (1. - 2.024701925508361 * eta2) * eta2
            + 215.06940561269835 * pWF['dchi'] * eta8 * delta
            + 0.1465612311350642 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.03940151060321499 * eta1 + 1.9034209537174116 * eta2 - 8.78587250202154 * eta3)
             + chidiff1 * delta * (-1.704299788495861 * eta1 - 4.923510922214181 * eta2
                                  + 0.36790005839460627 * eta3)) * sqroot
            + (2.2911849711339123 - 5.1846950040514335 * eta1 + 60.10368251688146 * eta2
              - 1139.110227749627 * eta3 + 7970.929280907627 * eta4 - 25472.73682092519 * eta5
              + 30950.67053883646 * eta6) * sqroot
            + S1 * (
                0.7718201508695763 * (-1.3012906461000349 + 26.432880113146012 * eta1
                                     - 186.5001124789369 * eta2 + 712.9101229418721 * eta3
                                     - 970.2126139442341 * eta4)
                + 0.04832734931068797 * (-5.9999628512498315 + 78.98681284391004 * eta1
                                        + 1.8360177574514709 * eta2 - 2537.636347529708 * eta3
                                        + 6858.003573909322 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_32_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 32 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 32 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta8 = eta2 * eta2 * eta2 * eta2
        S2 = S ** 2
        S3 = S2 * S

        noSpin = (
            jnp.sqrt(1. - 3. * eta) * (-0.006150151041614737 + 0.017454430190035 * eta
                                       + 0.02620962593739105 * eta2 - 0.019043090896351363 * eta3)
        ) / (-0.2655505633361449 + eta)

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.011073381681404716 + 0.00347699923233349 * S
            + eta * S * (0.05592992411391443 - 0.15666140197050316 * S2)
            + 0.012079324401547036 * S2
            + eta2 * (0.5440307361144313 - 0.008730335213434078 * S + 0.04615964369925028 * S2
                     + 0.6703688097531089 * S3)
            + 0.016323101357296865 * S3
        )

        uneqSpin = (
            -0.020140175824954427 * pWF['dchi'] ** 2 * (1. - 12.675522774051249 * eta2) * eta2
            - 417.3604094454253 * pWF['dchi'] * eta8 * delta
            + 0.10464021067936538 * pWF['dchi'] * eta2 * S * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.6358511175987503 * eta1 + 5.555088747533164 * eta2 - 14.078156877577733 * eta3)
             + chidiff1 * delta * (0.23205448591711159 * eta1 - 19.46049432345157 * eta2
                                  + 36.20685853857613 * eta3)) * sqroot
            + (1.1525594672495008 + 7.380126197972549 * eta1 - 17.51265776660515 * eta2
              - 976.9940395257111 * eta3 + 8880.536804741967 * eta4 - 30849.228936891763 * eta5
              + 38785.53683146884 * eta6) * sqroot
            + chidiff1 * delta * (1.904350804857431 * eta1 - 25.565242391371093 * eta2
                                 + 80.67120303906654 * eta3) * S1 * sqroot
            + S1 * (
                0.785171689871352 * (-0.4634745514643032 + 18.70856733065619 * eta1
                                    - 167.9231114864569 * eta2 + 744.7699462372949 * eta3
                                    - 1115.008825153004 * eta4)
                + 0.13469300326662165 * (-2.7311391326835133 + 72.17373498208947 * eta1
                                        - 483.7040402103785 * eta2 + 1136.8367114738041 * eta3
                                        - 472.02962341590774 * eta4) * S1
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_32_iv3: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv1(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv1 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv1 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S ** 2

        noSpin = jnp.sqrt(1. - 3. * eta) * (
            0.06190013067931406 + 0.1928897813606222 * eta + 1.9024723168424225 * eta2
            - 15.988716302668415 * eta3 + 35.21461767354364 * eta4
        )

        eqSpin = jnp.sqrt(1. - 3. * eta) * S * (
            0.011454874900772544 + 0.044702230915643903 * S
            + eta * (0.6600413908621988 + 0.12149520289658673 * S - 0.4482406547006759 * S2)
            + 0.07327810908370004 * S2
            + eta2 * (-2.1705970511116486 - 0.6512813450832168 * S + 1.1237234702682313 * S2)
        )

        uneqSpin = (
            0.4766851579723911 * pWF['dchi'] * (1. - 15.950025762198988 * eta2) * eta2
            + 0.127900699645338 * pWF['dchi'] ** 2 * (1. - 15.79329306044842 * eta2) * eta2
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (0.5697308729057493 * eta1 + 8.895576813118867 * eta2
                                - 34.98399465240273 * eta3)
             + chidiff2 * (1.6370346538130884 * eta1 - 14.597095790380884 * eta2
                          + 33.182723737396294 * eta3)) * sqroot
            + (5.2601381002242595 - 3.557926105832778 * eta1 - 138.9749850448088 * eta2
              + 603.7453704122706 * eta3 - 923.5495700703648 * eta4) * sqroot
            + S1 * (
                -0.41839636169678796 * (5.143510231379954 + 104.62892421207803 * eta1
                                       - 4232.508174045782 * eta2 + 50694.024801783446 * eta3
                                       - 283097.33358214336 * eta4 + 758333.2655404843 * eta5
                                       - 788783.0559069642 * eta6)
                - 0.05653522061311774 * (5.605483124564013 + 694.00652410087 * eta1
                                        - 17551.398321516353 * eta2 + 165236.6480734229 * eta3
                                        - 761661.9645651339 * eta4 + 1.7440315410044065e6 * eta5
                                        - 1.6010489769238676e6 * eta6) * S1
                - 0.023693246676754775 * (16.437107575918503 - 2911.2154288136217 * eta1
                                         + 89338.32554683842 * eta2 - 1.0803340811860575e6 * eta3
                                         + 6.255666490084672e6 * eta4 - 1.7434160932177313e7 * eta5
                                         + 1.883460394974573e7 * eta6) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv1: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv2(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv2 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv2 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        S2 = S ** 2
        S3 = S2 * S

        noSpin = (
            0.08406011695496626 - 0.1469952725049322 * eta + 0.2997223283799925 * eta2
            - 1.2910560244510723 * eta3
        )

        eqSpin = (
            (0.023924074703897662 + 0.26110236039648027 * eta - 1.1536009170220438 * eta2) * S
            + (0.04479727299752669 - 0.1439868858871802 * eta + 0.05736387085230215 * eta2) * S2
            + (0.06028104440131858 - 0.4759412992529712 * eta + 1.1090751649419717 * eta2) * S3
        )

        uneqSpin = (
            0.10346324686812074 * pWF['dchi'] ** 2 * (1. - 16.135903382018213 * eta2) * eta2
            + 0.2648241309154185 * pWF['dchi'] * eta2 * delta
        )

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        S3 = S1 * S2
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff2 * (-0.8318312659717388 * eta1 + 7.6541168007977864 * eta2 - 16.648660653220123 * eta3)
             + chidiff1 * delta * (2.214478316304753 * eta1 - 7.028104574328955 * eta2
                                  + 5.56587823143958 * eta3)) * sqroot
            + (3.173191054680422 + 6.707695566702527 * eta1 - 155.22519772642607 * eta2
              + 604.0067075996933 * eta3 - 876.5048298377644 * eta4) * sqroot
            + chidiff1 * delta * (4.749663394334708 * eta1 - 42.62996105525792 * eta2
                                 + 97.01712147349483 * eta3) * S1 * sqroot
            + S1 * (
                -0.2627203100303006 * (6.460396349297595 - 52.82425783851536 * eta1
                                      - 552.1725902144143 * eta2 + 12546.255587592654 * eta3
                                      - 81525.50289542897 * eta4 + 227254.37897941095 * eta5
                                      - 234487.3875219032 * eta6)
                - 0.008424003742397579 * (-109.26773035716548 + 15514.571912666677 * eta1
                                         - 408022.6805482195 * eta2 + 4.620165968920881e6 * eta3
                                         - 2.6446950627957724e7 * eta4 + 7.539643948937692e7 * eta5
                                         - 8.510662871580401e7 * eta6) * S1
                - 0.008830881730801855 * (-37.49992494976597 + 1359.7883958101172 * eta1
                                         - 23328.560285901796 * eta2 + 260027.4121353132 * eta3
                                         - 1.723865744472182e6 * eta4 + 5.858455766230802e6 * eta5
                                         - 7.756341721552802e6 * eta6) * S2
                - 0.027167813927224657 * (34.281932237450256 - 3312.7658728016568 * eta1
                                         + 84126.14531363266 * eta2 - 956052.0170024392 * eta3
                                         + 5.570748509263883e6 * eta4 - 1.6270212243584689e7 * eta5
                                         + 1.8855858173287075e7 * eta6) * S3
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv2: version {InspAmpFlag} is not valid.")

    return total


def IMRPhenomXHM_Insp_Amp_44_iv3(pWF: dict, InspAmpFlag: int) -> float:
    """
    Compute the iv3 coefficient for the 44 mode inspiral amplitude.

    Args:
        pWF: Waveform structure dictionary containing eta, chiPNHat, dchi, delta, etc.
        InspAmpFlag: Flag to select the fitting formula (122018 or 122022)

    Returns:
        The iv3 amplitude coefficient for the 44 mode

    Raises:
        ValueError: If InspAmpFlag is not one of the valid values
    """
    if InspAmpFlag == 122018:
        eta = pWF['eta']
        S = pWF['chiPNHat']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = (
            0.08212436946985402 - 0.025332770704783136 * eta - 3.2466088293309885 * eta2
            + 28.404235115663706 * eta3 - 111.36325359782991 * eta4 + 157.05954559045156 * eta5
        )

        eqSpin = S * (
            0.03488890057062679 + 0.039491331923244756 * S
            + eta * (-0.08968833480313292 - 0.12754920943544915 * S - 0.11199012099701576 * S2)
            + 0.034468577523793176 * S2
        )

        uneqSpin = 0.2062291124580944 * pWF['dchi'] * eta2 * delta

        total = noSpin + eqSpin + uneqSpin

    elif InspAmpFlag == 122022:
        eta = pWF['eta']
        delta = pWF['delta']
        sqroot = jnp.sqrt(eta)
        S = pWF['chiPNHat']
        chidiff = pWF['dchi_half']

        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1

        total = (
            (chidiff1 * delta * (1.4739380748149558 * eta1 + 0.06541707987699942 * eta2
                                - 9.473290540936633 * eta3)
             + chidiff2 * (-0.3640838331639651 * eta1 + 3.7369795937033756 * eta2
                          - 8.709159662885131 * eta3)) * sqroot
            + (1.7335503724888923 + 12.656614578053683 * eta1 - 139.6610487470118 * eta2
              + 456.78649322753824 * eta3 - 599.2709938848282 * eta4) * sqroot
            + chidiff1 * delta * (2.3532739003216254 * eta1 - 21.37216554136868 * eta2
                                 + 53.35003268489743 * eta3) * S1 * sqroot
            + S1 * (
                -0.15782329022461472 * (6.0309399412954345 - 229.16361598098678 * eta1
                                       + 3777.477006415653 * eta2 - 31109.307191210424 * eta3
                                       + 139319.8239886073 * eta4 - 324891.4001578353 * eta5
                                       + 307714.3954026392 * eta6)
                - 0.03050157254864058 * (4.232861441291087 + 1609.4251694451375 * eta1
                                        - 51213.27604422822 * eta2 + 612317.1751155312 * eta3
                                        - 3.5589766538499263e6 * eta4 + 1.0147654212772278e7 * eta5
                                        - 1.138861230369246e7 * eta6) * S1
                - 0.026407497690308382 * (-17.184685557542196 + 744.4743953122965 * eta1
                                         - 10494.512487701073 * eta2 + 66150.52694069289 * eta3
                                         - 184787.79377504133 * eta4 + 148102.4257785174 * eta5
                                         + 128167.89151782403 * eta6) * S2
            ) * sqroot
        )

    else:
        raise ValueError(f"Error in IMRPhenomXHM_Insp_Amp_44_iv3: version {InspAmpFlag} is not valid.")

    return total


# ==================== Inspiral Phase Functions ====================

def IMRPhenomXHM_Insp_Phase_21_lambda(pWF: dict, InspPhaseFlag: int) -> float:
    """
    Compute the lambda coefficient for the 21 mode inspiral phase.

    Args:
        pWF: Waveform structure dictionary containing eta, STotR, dchi, etc.
        InspPhaseFlag: Flag to select the fitting formula (122019)

    Returns:
        The lambda phase coefficient for the 21 mode

    Raises:
        ValueError: If InspPhaseFlag is not one of the valid values
    """
    if InspPhaseFlag == 122019:
        eta = pWF['eta']
        S = pWF['STotR']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = (
            13.664473636545068 - 170.08866400251395 * eta + 3535.657736681598 * eta2
            - 26847.690494515424 * eta3 + 96463.68163125668 * eta4 - 133820.89317471132 * eta5
        )

        eqSpin = (
            S * (
                18.52571430563905 - 41.55066592130464 * S
                + eta3 * (83493.24265292779 + 16501.749243703132 * S - 149700.4915210766 * S2)
                + eta * (3642.5891077598003 + 1198.4163078715173 * S - 6961.484805326852 * S2)
                + 33.8697137964237 * S2
                + eta2 * (-35031.361998480075 - 7233.191207000735 * S + 62149.00902591944 * S2)
            )
        ) / (6.880288191574696 + 1. * S)

        uneqSpin = -134.27742343186577 * pWF['dchi'] * jnp.sqrt(1. - 4. * eta) * eta2

        total = noSpin + eqSpin + uneqSpin

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Insp_Phase_21_lambda: version {InspPhaseFlag} is not valid. "
            f"Recommended version is 122019."
        )

    return total


def IMRPhenomXHM_Insp_Phase_33_lambda(pWF: dict, InspPhaseFlag: int) -> float:
    """
    Compute the lambda coefficient for the 33 mode inspiral phase.

    Args:
        pWF: Waveform structure dictionary containing eta, STotR, dchi, etc.
        InspPhaseFlag: Flag to select the fitting formula (122019)

    Returns:
        The lambda phase coefficient for the 33 mode

    Raises:
        ValueError: If InspPhaseFlag is not one of the valid values
    """
    if InspPhaseFlag == 122019:
        eta = pWF['eta']
        S = pWF['STotR']
        delta = jnp.sqrt(1. - 4. * eta)
        eta2 = eta ** 2
        eta3 = eta2 * eta

        noSpin = (
            4.1138398568400705 + 9.772510519809892 * eta - 103.92956504520747 * eta2
            + 242.3428625556764 * eta3
        )

        eqSpin = (
            (-0.13253553909611435 + 26.644159828590055 * eta - 105.09339163109497 * eta2) * S
        ) / (1. + 0.11322426762297967 * S)

        uneqSpin = -19.705359163581168 * pWF['dchi'] * eta2 * delta

        total = noSpin + eqSpin + uneqSpin

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Insp_Phase_33_lambda: version {InspPhaseFlag} is not valid. "
            f"Recommended version is 122019."
        )

    return total


def IMRPhenomXHM_Insp_Phase_32_lambda(pWF: dict, InspPhaseFlag: int) -> float:
    """
    Compute the lambda coefficient for the 32 mode inspiral phase.

    Args:
        pWF: Waveform structure dictionary containing eta, STotR, dchi, chi1L, chi2L, etc.
        InspPhaseFlag: Flag to select the fitting formula (122019)

    Returns:
        The lambda phase coefficient for the 32 mode

    Raises:
        ValueError: If InspPhaseFlag is not one of the valid values
    """
    if InspPhaseFlag == 122019:
        eta = pWF['eta']
        S = pWF['STotR']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S ** 2
        S3 = S2 * S
        S4 = S3 * S

        noSpin = (
            9.913819875501506 + 18.424900617803107 * eta - 574.8672384388947 * eta2
            + 2671.7813055097877 * eta3 - 6244.001932443913 * eta4
        ) / (1. - 0.9103118343073325 * eta)

        eqSpin = (
            (-4.367632806613781 + 245.06757304950986 * eta - 2233.9319708029775 * eta2
             + 5894.355429022858 * eta3) * S
            + (-1.375112297530783 - 1876.760129419146 * eta + 17608.172965575013 * eta2
               - 40928.07304790013 * eta3) * S2
            + (-1.28324755577382 - 138.36970336658558 * eta + 708.1455154504333 * eta2
               - 273.23750933544176 * eta3) * S3
            + (1.8403161863444328 + 2009.7361967331492 * eta - 18636.271414571278 * eta2
               + 42379.205045791656 * eta3) * S4
        )

        uneqSpin = (
            pWF['dchi'] * jnp.sqrt(1. - 4. * eta) * eta2 * (
                -105.34550407768225 - 1566.1242344157668 * pWF['chi1L'] * eta
                + 1566.1242344157668 * pWF['chi2L'] * eta + 2155.472229664981 * eta * S
            )
        )

        total = noSpin + eqSpin + uneqSpin

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Insp_Phase_32_lambda: version {InspPhaseFlag} is not valid. "
            f"Recommended version is 122019."
        )

    return total


def IMRPhenomXHM_Insp_Phase_44_lambda(pWF: dict, InspPhaseFlag: int) -> float:
    """
    Compute the lambda coefficient for the 44 mode inspiral phase.

    Args:
        pWF: Waveform structure dictionary containing eta, STotR, dchi, etc.
        InspPhaseFlag: Flag to select the fitting formula (122019)

    Returns:
        The lambda phase coefficient for the 44 mode

    Raises:
        ValueError: If InspPhaseFlag is not one of the valid values
    """
    if InspPhaseFlag == 122019:
        eta = pWF['eta']
        S = pWF['STotR']
        eta2 = eta ** 2
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        S2 = S ** 2

        noSpin = (
            5.254484747463392 - 21.277760168559862 * eta + 160.43721442910618 * eta2
            - 1162.954360723399 * eta3 + 1685.5912722190276 * eta4 - 1538.6661348106031 * eta5
        )

        eqSpin = (
            (0.007067861615983771 - 10.945895160727437 * eta + 246.8787141453734 * eta2
             - 810.7773268493444 * eta3) * S
            + (0.17447830920234977 + 4.530539154777984 * eta - 176.4987316167203 * eta2
               + 621.6920322846844 * eta3) * S2
        )

        uneqSpin = -8.384066369867833 * pWF['dchi'] * jnp.sqrt(1. - 4. * eta) * eta2

        total = noSpin + eqSpin + uneqSpin

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Insp_Phase_44_lambda: version {InspPhaseFlag} is not valid. "
            f"Recommended version is 122019."
        )

    return total


def IMRPhenomXHM_Insp_Phase_LambdaPN(eta: float, modeInt: int) -> float:
    """
    Returns linear-in-f term from the complex part of each mode's amplitude.

    This is added to the orbital phase and comes from an expansion of Eq. (4.9)
    truncated at linear order.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio
    modeInt : int
        Mode identifier (21, 33, 32, 44)

    Returns
    -------
    float
        The lambda_PN coefficient for the phase
    """
    PI = jnp.pi

    if modeInt == 21:
        # 2 f π (-(1/2) - Log[16]/2)
        output = 2.0 * PI * (-0.5 - 2.0 * jnp.log(2.0))
    elif modeInt == 33:
        # 2/3 f π (-(21/5) + 6 Log[3/2])
        output = 2.0 / 3.0 * PI * (-21.0 / 5.0 + 6.0 * jnp.log(1.5))
    elif modeInt == 32:
        # -((2376 f π (-5 + 22 η))/(-3960 + 11880 η))
        output = -((2376.0 * PI * (-5.0 + 22.0 * eta)) / (-3960.0 + 11880.0 * eta))
    elif modeInt == 44:
        # (45045 f π (336 - 1193 η + 320 (-1 + 3 η) Log[2]))/(2 (-1801800 + 5405400 η))
        output = (45045.0 * PI * (336.0 - 1193.0 * eta + 320.0 * (-1.0 + 3.0 * eta) * jnp.log(2.0)) /
                  (2.0 * (-1801800.0 + 5405400.0 * eta)))
    else:
        output = 0.0

    return -1.0 * output


def IMRPhenomXHM_Inspiral_Phase_AnsatzInt(Mf: float, powers_of_Mf: dict, pPhase: dict) -> float:
    """
    Compute the integral of the inspiral phase ansatz.

    This function loads the rescaled PhenomX coefficients for the phase and corrects
    the result with a linear-in-f term coming from the complex part of the PN amplitude.
    The rescaling of the 22-coefficients is explained in App. D.

    Parameters
    ----------
    Mf : float
        Geometric frequency (frequency * total mass)
    powers_of_Mf : dict
        Dictionary containing various powers of Mf for efficient computation
        Expected keys: m_seven_thirds, m_two, m_five_thirds, m_four_thirds, m_one,
                      m_two_thirds, m_one_third, one_third, two_thirds, four_thirds,
                      five_thirds, two, seven_thirds, log
    pPhase : dict
        Phase coefficients structure containing 'phi' and 'phiL' arrays
        phi: array of 15 phase coefficients
        phiL: array of 15 log coefficients

    Returns
    -------
    float
        The inspiral phase value at frequency Mf
    """
    # Number of coefficients
    N_MAX_COEFFICIENTS_PHASE_INS = 15

    # Frequency powers array corresponding to the phase expansion
    freqs = jnp.array([
        powers_of_Mf['m_seven_thirds'],
        powers_of_Mf['m_two'],
        powers_of_Mf['m_five_thirds'],
        powers_of_Mf['m_four_thirds'],
        powers_of_Mf['m_one'],
        powers_of_Mf['m_two_thirds'],
        powers_of_Mf['m_one_third'],
        1.0,
        powers_of_Mf['one_third'],
        powers_of_Mf['two_thirds'],
        Mf,
        powers_of_Mf['four_thirds'],
        powers_of_Mf['five_thirds'],
        powers_of_Mf['two'],
        powers_of_Mf['seven_thirds']
    ])

    logMf = powers_of_Mf['log']

    # Compute the orbital phase by loading the rescaled PhenomX coefficients
    philm = 0.0
    for i in range(N_MAX_COEFFICIENTS_PHASE_INS):
        philm += (pPhase['phi'][i] + pPhase['phiL'][i] * logMf) * freqs[i]

    return philm


def IMRPhenomXHM_Inspiral_Phase_Ansatz(Mf: float, powers_of_Mf: dict, pPhase: dict) -> float:
    """
    Compute the derivative of the inspiral phase ansatz.

    This function computes the frequency derivative of the inspiral phase by loading
    the rescaled PhenomX coefficients.

    Parameters
    ----------
    Mf : float
        Geometric frequency (frequency * total mass)
    powers_of_Mf : dict
        Dictionary containing various powers of Mf for efficient computation
        Expected keys: m_ten_thirds, m_three, m_eight_thirds, m_seven_thirds, m_two,
                      m_five_thirds, m_four_thirds, m_one, m_two_thirds, m_one_third,
                      one_third, two_thirds, four_thirds, log
    pPhase : dict
        Phase coefficients structure containing 'phi' and 'phiL' arrays
        phi: array of 15 phase coefficients
        phiL: array of 15 log coefficients

    Returns
    -------
    float
        The derivative of the inspiral phase at frequency Mf
    """
    # Number of coefficients
    N_MAX_COEFFICIENTS_PHASE_INS = 15

    # Power law coefficients for the derivative
    coeffs = jnp.array([-7.0/3.0, -2.0, -5.0/3.0, -4.0/3.0, -1.0, -2.0/3.0, -1.0/3.0,
                        0.0, 1.0/3.0, 2.0/3.0, 1.0, 4.0/3.0, 5.0/3.0, 2.0, 7.0/3.0])

    # Frequency powers array for the derivative
    freqs = jnp.array([
        powers_of_Mf['m_ten_thirds'],
        powers_of_Mf['m_three'],
        powers_of_Mf['m_eight_thirds'],
        powers_of_Mf['m_seven_thirds'],
        powers_of_Mf['m_two'],
        powers_of_Mf['m_five_thirds'],
        powers_of_Mf['m_four_thirds'],
        powers_of_Mf['m_one'],
        powers_of_Mf['m_two_thirds'],
        powers_of_Mf['m_one_third'],
        1.0,
        powers_of_Mf['one_third'],
        powers_of_Mf['two_thirds'],
        Mf,
        powers_of_Mf['four_thirds']
    ])

    logMf = powers_of_Mf['log']

    # Compute the derivative of the orbital phase
    dphilm = 0.0
    for i in range(N_MAX_COEFFICIENTS_PHASE_INS):
        dphilm += ((pPhase['phi'][i] + pPhase['phiL'][i] * logMf) * coeffs[i] +
                   pPhase['phiL'][i]) * freqs[i]

    return dphilm


# ==================== Pseudo-PN Amplitude Coefficients ====================

def IMRPhenomXHM_Inspiral_Amp_rho1(
    v1: float, v2: float, v3: float,
    powers_of_fcutInsp: dict,
    powers_of_f1: dict,
    powers_of_f2: dict,
    powers_of_f3: dict,
    pWFHM: dict
) -> float:
    """
    Get Pseudo PN Amp coefficient rho1 at f^(7/3).

    This function computes the rho1 coefficient for the inspiral amplitude reconstruction
    using collocation points. The coefficient is computed differently depending on the
    number of collocation points used (0, 1, 2, or 3).

    Parameters
    ----------
    v1, v2, v3 : float
        Values at the three collocation points
    powers_of_fcutInsp : dict
        Dictionary of powers of the inspiral cutoff frequency
    powers_of_f1, powers_of_f2, powers_of_f3 : dict
        Dictionaries of powers of the collocation point frequencies
    pWFHM : dict
        Waveform structure containing IMRPhenomXHMInspiralAmpVersion

    Returns
    -------
    float
        The rho1 coefficient

    Raises
    ------
    ValueError
        If IMRPhenomXHMInspiralAmpVersion is not 0, 1, 2, or 3
    """
    IMRPhenomXHMInspiralAmpVersion = pWFHM['IMRPhenomXHMInspiralAmpVersion']

    if IMRPhenomXHMInspiralAmpVersion == 3:  # 3 inspiral collocation points (default)
        retVal = (
            powers_of_fcutInsp['seven_thirds'] *
            (-(powers_of_f1['three'] * powers_of_f3['eight_thirds'] * v2) +
             powers_of_f1['eight_thirds'] * powers_of_f3['three'] * v2 +
             powers_of_f2['three'] * (powers_of_f3['eight_thirds'] * v1 -
                                      powers_of_f1['eight_thirds'] * v3) +
             powers_of_f2['eight_thirds'] * (-(powers_of_f3['three'] * v1) +
                                             powers_of_f1['three'] * v3))
        ) / (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f3['one_third']) *
             (powers_of_f2['one_third'] - powers_of_f3['one_third']) * powers_of_f3['seven_thirds'])

    elif IMRPhenomXHMInspiralAmpVersion == 2:  # 2 inspiral collocation points
        retVal = (
            powers_of_fcutInsp['seven_thirds'] *
            (-(powers_of_f2['eight_thirds'] * v1) + powers_of_f1['eight_thirds'] * v2)
        ) / (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'])

    elif IMRPhenomXHMInspiralAmpVersion == 1:  # 1 inspiral collocation point
        retVal = (powers_of_fcutInsp['seven_thirds'] * v1) / powers_of_f1['seven_thirds']

    elif IMRPhenomXHMInspiralAmpVersion == 0:  # No collocation points, reconstruct with PN only
        retVal = 0.0

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Inspiral_Amp_rho1: version {IMRPhenomXHMInspiralAmpVersion} "
            f"is not valid. Versions available are 0, 1, 2, 3."
        )

    return retVal


def IMRPhenomXHM_Inspiral_Amp_rho2(
    v1: float, v2: float, v3: float,
    powers_of_fcutInsp: dict,
    powers_of_f1: dict,
    powers_of_f2: dict,
    powers_of_f3: dict,
    pWFHM: dict
) -> float:
    """
    Get Pseudo PN Amp coefficient rho2 at f^(8/3).

    This function computes the rho2 coefficient for the inspiral amplitude reconstruction
    using collocation points. The coefficient is computed differently depending on the
    number of collocation points used (0, 1, 2, or 3).

    Parameters
    ----------
    v1, v2, v3 : float
        Values at the three collocation points
    powers_of_fcutInsp : dict
        Dictionary of powers of the inspiral cutoff frequency
    powers_of_f1, powers_of_f2, powers_of_f3 : dict
        Dictionaries of powers of the collocation point frequencies
    pWFHM : dict
        Waveform structure containing IMRPhenomXHMInspiralAmpVersion

    Returns
    -------
    float
        The rho2 coefficient

    Raises
    ------
    ValueError
        If IMRPhenomXHMInspiralAmpVersion is not 0, 1, 2, or 3
    """
    IMRPhenomXHMInspiralAmpVersion = pWFHM['IMRPhenomXHMInspiralAmpVersion']

    if IMRPhenomXHMInspiralAmpVersion == 3:  # 3 inspiral collocation points (default)
        retVal = (
            -(powers_of_fcutInsp['eight_thirds'] *
              (-(powers_of_f1['three'] * powers_of_f3['seven_thirds'] * v2) +
               powers_of_f1['seven_thirds'] * powers_of_f3['three'] * v2 +
               powers_of_f2['three'] * (powers_of_f3['seven_thirds'] * v1 -
                                        powers_of_f1['seven_thirds'] * v3) +
               powers_of_f2['seven_thirds'] * (-(powers_of_f3['three'] * v1) +
                                               powers_of_f1['three'] * v3)))
        ) / (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f3['one_third']) *
             (powers_of_f2['one_third'] - powers_of_f3['one_third']) * powers_of_f3['seven_thirds'])

    elif IMRPhenomXHMInspiralAmpVersion == 2:  # 2 inspiral collocation points
        retVal = (
            -(powers_of_fcutInsp['eight_thirds'] *
              (-(powers_of_f2['seven_thirds'] * v1) + powers_of_f1['seven_thirds'] * v2))
        ) / (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'])

    elif IMRPhenomXHMInspiralAmpVersion == 1:  # 1 inspiral collocation point
        retVal = 0.0

    elif IMRPhenomXHMInspiralAmpVersion == 0:  # No collocation points, reconstruct with PN only
        retVal = 0.0

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Inspiral_Amp_rho2: version {IMRPhenomXHMInspiralAmpVersion} "
            f"is not valid. Versions available are 0, 1, 2, 3."
        )

    return retVal


def IMRPhenomXHM_Inspiral_Amp_rho3(
    v1: float, v2: float, v3: float,
    powers_of_fcutInsp: dict,
    powers_of_f1: dict,
    powers_of_f2: dict,
    powers_of_f3: dict,
    pWFHM: dict
) -> float:
    """
    Get Pseudo PN Amp coefficient rho3 at f^(9/3).

    This function computes the rho3 coefficient for the inspiral amplitude reconstruction
    using collocation points. The coefficient is computed differently depending on the
    number of collocation points used (0, 1, 2, or 3).

    Parameters
    ----------
    v1, v2, v3 : float
        Values at the three collocation points
    powers_of_fcutInsp : dict
        Dictionary of powers of the inspiral cutoff frequency
    powers_of_f1, powers_of_f2, powers_of_f3 : dict
        Dictionaries of powers of the collocation point frequencies
    pWFHM : dict
        Waveform structure containing IMRPhenomXHMInspiralAmpVersion

    Returns
    -------
    float
        The rho3 coefficient

    Raises
    ------
    ValueError
        If IMRPhenomXHMInspiralAmpVersion is not 0, 1, 2, or 3
    """
    IMRPhenomXHMInspiralAmpVersion = pWFHM['IMRPhenomXHMInspiralAmpVersion']

    if IMRPhenomXHMInspiralAmpVersion == 3:  # 3 inspiral collocation points (default)
        retVal = (
            powers_of_fcutInsp['three'] *
            (powers_of_f1['seven_thirds'] * (-powers_of_f1['one_third'] + powers_of_f3['one_third']) *
             powers_of_f3['seven_thirds'] * v2 +
             powers_of_f2['seven_thirds'] * (-(powers_of_f3['eight_thirds'] * v1) +
                                             powers_of_f1['eight_thirds'] * v3) +
             powers_of_f2['eight_thirds'] * (powers_of_f3['seven_thirds'] * v1 -
                                             powers_of_f1['seven_thirds'] * v3))
        ) / (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f3['one_third']) *
             (powers_of_f2['one_third'] - powers_of_f3['one_third']) * powers_of_f3['seven_thirds'])

    elif IMRPhenomXHMInspiralAmpVersion == 2:  # 2 inspiral collocation points
        retVal = 0.0

    elif IMRPhenomXHMInspiralAmpVersion == 1:  # 1 inspiral collocation point
        retVal = 0.0

    elif IMRPhenomXHMInspiralAmpVersion == 0:  # No collocation points, reconstruct with PN only
        retVal = 0.0

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Inspiral_Amp_rho3: version {IMRPhenomXHMInspiralAmpVersion} "
            f"is not valid. Versions available are 0, 1, 2, 3."
        )

    return retVal


# ==================== Inspiral Amplitude Ansatz Functions ====================

def IMRPhenomXHM_Inspiral_PNAmp_Ansatz(
    powers_of_Mf: dict,
    pWFHM: dict,
    pAmp: dict
) -> float:
    """
    Return the Fourier Domain Post-Newtonian ansatz up to 3PN without the pseudoPN terms
    for a particular frequency.

    The ansatz is built by performing the Stationary Phase Approximation of the
    Time-Domain Amplitude up to 3PN. The result is re-expanded in power series up to 3PN,
    except for the 21 mode, which is better behaved without the re-expansion.

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pWFHM : dict
        Waveform structure containing useFAmpPN flag
    pAmp : dict
        Amplitude coefficients structure containing PN coefficients

    Returns
    -------
    float
        The PN amplitude ansatz (rescaled with 22 mode prefactor)
    """
    # The 21 mode is special, is not a power series
    if pWFHM['useFAmpPN'] == 1:
        return IMRPhenomXHM_Inspiral_PNAmp_21Ansatz(powers_of_Mf, pWFHM, pAmp)

    # This returns the amplitude strain rescaled with the prefactor of the 22 mode:
    # divided by sqrt(2*eta/3.)/pi^(1/6)*Mf^(-7/6)
    pnAmp = jnp.abs(
        pAmp['pnInitial'] +
        powers_of_Mf['one_third'] * pAmp['pnOneThird'] +
        powers_of_Mf['two_thirds'] * pAmp['pnTwoThirds'] +
        powers_of_Mf['itself'] * pAmp['pnThreeThirds'] +
        powers_of_Mf['four_thirds'] * pAmp['pnFourThirds'] +
        powers_of_Mf['five_thirds'] * pAmp['pnFiveThirds'] +
        powers_of_Mf['two'] * pAmp['pnSixThirds']
    )

    if pAmp['InspRescaleFactor'] == 0:
        pnAmp *= pAmp['PNglobalfactor'] * powers_of_Mf['m_seven_sixths'] * pWFHM['ampNorm']
    elif pAmp['InspRescaleFactor'] == 1:
        pnAmp *= pAmp['PNglobalfactor']

    return pnAmp


def IMRPhenomXHM_Inspiral_PNAmp_21Ansatz(
    powers_of_Mf: dict,
    pWFHM: dict,
    pAmp: dict
) -> float:
    """
    Return the 21 mode Fourier Domain Post-Newtonian ansatz up to 3PN without the
    pseudoPN terms for a particular frequency.

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pWFHM : dict
        Waveform structure containing IMRPhenomXHMInspiralAmpFreqsVersion
    pAmp : dict
        Amplitude coefficients structure

    Returns
    -------
    float
        The 21 mode PN amplitude ansatz

    Raises
    ------
    ValueError
        If IMRPhenomXHMInspiralAmpFreqsVersion is not valid
    """
    InsAmpFlag = pWFHM['IMRPhenomXHMInspiralAmpFreqsVersion']

    if InsAmpFlag == 122018:
        two_to_m_one_sixths = 0.8908987181403393
        three_to_m_one_second = 0.5773502691896257
        lalpi = jnp.pi  # Assuming lalpiHM is just pi

        x_to_m_one_four = two_to_m_one_sixths * (lalpi ** (-1.0/6.0)) * powers_of_Mf['m_one_sixth']

        # Complex time-domain Post-Newtonian amplitude, power series
        CpnAmpTD = (
            powers_of_Mf['one_third'] * pAmp['x05'] +
            powers_of_Mf['two_thirds'] * pAmp['x1'] +
            powers_of_Mf['itself'] * pAmp['x15'] +
            powers_of_Mf['four_thirds'] * pAmp['x2'] +
            powers_of_Mf['five_thirds'] * pAmp['x25'] +
            powers_of_Mf['two'] * pAmp['x3']
        )
        CpnAmpTD = CpnAmpTD * powers_of_Mf['two_thirds'] * pAmp['PNTDfactor']

        # Value of the derivative of the PN expansion parameter X given by TaylorT4
        XdotT4 = (
            powers_of_Mf['five_thirds'] * powers_of_Mf['five_thirds'] * pAmp['xdot5'] +
            powers_of_Mf['four'] * pAmp['xdot6'] +
            powers_of_Mf['eight_thirds'] * powers_of_Mf['five_thirds'] * pAmp['xdot65'] +
            powers_of_Mf['seven_thirds'] * powers_of_Mf['seven_thirds'] * pAmp['xdot7'] +
            powers_of_Mf['eight_thirds'] * powers_of_Mf['seven_thirds'] * pAmp['xdot75'] +
            powers_of_Mf['eight_thirds'] * powers_of_Mf['eight_thirds'] * pAmp['xdot8'] +
            (powers_of_Mf['log'] * 2.0/3.0 + pAmp['log2pi_two_thirds']) *
            powers_of_Mf['eight_thirds'] * powers_of_Mf['eight_thirds'] * pAmp['xdot8Log'] +
            powers_of_Mf['eight_thirds'] * powers_of_Mf['eight_thirds'] * powers_of_Mf['one_third'] * pAmp['xdot85']
        )

        # Perform the SPA, multiply time-domain by the phasing factor
        pnAmp = (2.0 * jnp.sqrt(lalpi) * three_to_m_one_second * jnp.abs(CpnAmpTD) *
                 x_to_m_one_four / jnp.sqrt(XdotT4))

        if pAmp['InspRescaleFactor'] != 0:
            pnAmp /= RescaleFactor(powers_of_Mf, pAmp, pAmp['InspRescaleFactor'])

    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Inspiral_PNAmp_Ansatz: "
            f"IMRPhenomXInspiralAmpVersion {InsAmpFlag} is not valid. "
            f"Recommended version is 122018."
        )

    return pnAmp


def IMRPhenomXHM_Inspiral_Amp_Ansatz(
    powers_of_Mf: dict,
    pWFHM: dict,
    pAmp: dict
) -> float:
    """
    This is the complete Inspiral Amplitude Ansatz: PN ansatz + pseudoPN terms.

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pWFHM : dict
        Waveform structure containing IMRPhenomXHMInspiralAmpFreqsVersion
    pAmp : dict
        Amplitude coefficients structure

    Returns
    -------
    float
        The complete inspiral amplitude ansatz
    """
    if pWFHM['IMRPhenomXHMInspiralAmpFreqsVersion'] == 122018:
        tmp = pAmp['InspRescaleFactor']
        pAmp['InspRescaleFactor'] = 1

        InspAmp = (
            IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_Mf, pWFHM, pAmp) +
            powers_of_Mf['seven_thirds'] / pAmp['fcutInsp_seven_thirds'] * pAmp['rho1'] +
            powers_of_Mf['eight_thirds'] / pAmp['fcutInsp_eight_thirds'] * pAmp['rho2'] +
            powers_of_Mf['three'] / pAmp['fcutInsp_three'] * pAmp['rho3']
        )

        pAmp['InspRescaleFactor'] = tmp

        if pAmp['InspRescaleFactor'] == 0:
            InspAmp *= powers_of_Mf['m_seven_sixths'] * pWFHM['ampNorm']

    else:
        # New release. FIXME: improve how the ansatz is built
        InspAmp = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(powers_of_Mf, pWFHM, pAmp)

        pseudoterms = (
            powers_of_Mf['seven_thirds'] / pAmp['fcutInsp_seven_thirds'] * pAmp['InspiralCoefficient'][0] +
            powers_of_Mf['eight_thirds'] / pAmp['fcutInsp_eight_thirds'] * pAmp['InspiralCoefficient'][1] +
            powers_of_Mf['three'] / pAmp['fcutInsp_three'] * pAmp['InspiralCoefficient'][2]
        )
        pseudoterms *= powers_of_Mf['m_seven_sixths'] * pAmp['PNdominant']

        InspAmp += pseudoterms

        if pAmp['InspRescaleFactor'] != 0:
            InspAmp /= RescaleFactor(powers_of_Mf, pAmp, pAmp['InspRescaleFactor'])

    return InspAmp


def IMRPhenomXHM_Inspiral_Amp_NDAnsatz(
    powers_of_Mf: dict,
    pWFHM: dict,
    pAmp: dict
) -> float:
    """
    Numerical derivative of the inspiral amplitude (4th order finite differences).
    It is used for reconstructing the intermediate region.

    Parameters
    ----------
    powers_of_Mf : dict
        Dictionary of powers of the frequency (Mf)
    pWFHM : dict
        Waveform structure
    pAmp : dict
        Amplitude coefficients structure

    Returns
    -------
    float
        Numerical derivative of the inspiral amplitude
    """
    df = 10e-10
    centralfreq = powers_of_Mf['itself']

    # Initialize powers for finite difference points
    # Note: This assumes you have a compute_powers_of_f function
    powers_of_Mf2R = compute_powers_of_f(centralfreq + 2*df)
    powers_of_MfR = compute_powers_of_f(centralfreq + df)
    powers_of_MfL = compute_powers_of_f(centralfreq - df)
    powers_of_Mf2L = compute_powers_of_f(centralfreq - 2*df)

    fun2R = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_Mf2R, pWFHM, pAmp)
    funR = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_MfR, pWFHM, pAmp)
    funL = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_MfL, pWFHM, pAmp)
    fun2L = IMRPhenomXHM_Inspiral_Amp_Ansatz(powers_of_Mf2L, pWFHM, pAmp)

    Nder = (-fun2R + 8*funR - 8*funL + fun2L) / (12*df)

    return Nder


# ==================== Veto and Helper Functions ====================

def IMRPhenomXHM_Inspiral_Amplitude_Veto(
    iv1: float,
    iv2: float,
    iv3: float,
    powers_of_f1: dict,
    powers_of_f2: dict,
    powers_of_f3: dict,
    pAmp: dict,
    pWFHM: dict
) -> tuple:
    """
    VETO function: when extrapolating the model outside the calibration region,
    the collocation points can be bad behaved. With this function we select those
    that will be used in the reconstruction.

    Parameters
    ----------
    iv1, iv2, iv3 : float
        Initial values at collocation points
    powers_of_f1, powers_of_f2, powers_of_f3 : dict
        Dictionaries of powers of the collocation point frequencies
    pAmp : dict
        Amplitude coefficients structure
    pWFHM : dict
        Waveform structure (will be modified)

    Returns
    -------
    tuple
        (modified_iv1, modified_iv2, modified_iv3, modified_pWFHM)
    """
    threshold = 0.2 / pWFHM['ampNorm']

    # Make copies to avoid modifying inputs directly
    new_iv1, new_iv2, new_iv3 = iv1, iv2, iv3
    new_version = pWFHM['IMRPhenomXHMInspiralAmpVersion']

    # Remove too low collocation points (heuristic)
    if pAmp['CollocationPointsValuesAmplitudeInsp'][0] < threshold * powers_of_f1['seven_sixths']:
        new_iv1 = 0.0
        new_version = 2

    if pAmp['CollocationPointsValuesAmplitudeInsp'][1] < threshold * powers_of_f2['seven_sixths']:
        new_iv2 = 0.0
        new_version = new_version - 1

    if pAmp['CollocationPointsValuesAmplitudeInsp'][2] < threshold * powers_of_f3['seven_sixths']:
        new_iv3 = 0.0
        new_version = new_version - 1

    # Update the waveform structure
    pWFHM_updated = pWFHM.copy()
    pWFHM_updated['IMRPhenomXHMInspiralAmpVersion'] = new_version

    return new_iv1, new_iv2, new_iv3, pWFHM_updated


def WavyPoints(p1: float, p2: float, p3: float) -> int:
    """
    Check if the three collocation points are wavy (non-monotonic).

    Parameters
    ----------
    p1, p2, p3 : float
        Three point values to check

    Returns
    -------
    int
        1 if points are wavy, 0 otherwise
    """
    if (p1 > p2 and p2 < p3) or (p1 < p2 and p2 > p3):
        return 1
    else:
        return 0


# Note: Helper function that may be referenced by the functions above
def compute_powers_of_f(f: float) -> dict:
    """
    Compute various powers of frequency f.

    Parameters
    ----------
    f : float
        Frequency value

    Returns
    -------
    dict
        Dictionary containing various powers of f
    """
    return {
        'itself': f,
        'one_third': f ** (1.0/3.0),
        'two_thirds': f ** (2.0/3.0),
        'four_thirds': f ** (4.0/3.0),
        'five_thirds': f ** (5.0/3.0),
        'seven_thirds': f ** (7.0/3.0),
        'eight_thirds': f ** (8.0/3.0),
        'three': f ** 3.0,
        'two': f ** 2.0,
        'four': f ** 4.0,
        'seven_sixths': f ** (7.0/6.0),
        'm_one_sixth': f ** (-1.0/6.0),
        'm_seven_sixths': f ** (-7.0/6.0),
        'log': jnp.log(f)
    }


# ==================== Coefficient Computation Functions ====================

def IMRPhenomXHM_Inspiral_Amp_CollocationPoints(pAmp: dict, pWFHM: dict, pWF22: dict) -> dict:
    """
    Compute collocation point frequencies and values for inspiral amplitude reconstruction.

    This function sets the frequencies where the amplitude will be evaluated for
    collocation, and evaluates the fitted amplitude values at those frequencies.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients structure containing:
        - fAmpMatchIN: Matching frequency for inspiral-intermediate transition
        - InspiralAmpFits: Array of fit functions for inspiral amplitude
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - IMRPhenomXHMInspiralAmpFreqsVersion: Version flag for frequency selection
        - nCollocPtsInspAmp: Number of collocation points
        - modeInt: Mode integer index
        - IMRPhenomXHMInspiralAmpFitsVersion: Version flag for fit formulas
        - InspiralAmpVeto: Flag for applying veto logic
    pWF22 : dict
        Waveform structure for the 22 mode

    Returns
    -------
    dict
        Updated pAmp dictionary with:
        - CollocationPointsFreqsAmplitudeInsp: Frequencies at collocation points
        - CollocationPointsValuesAmplitudeInsp: Amplitude values at collocation points
        Also updates pWFHM['IMRPhenomXHMInspiralAmpVersion'] if veto is applied

    Raises
    ------
    ValueError
        If IMRPhenomXHMInspiralAmpFreqsVersion is not valid
    """
    pAmp_out = pAmp.copy()
    pWFHM_out = pWFHM.copy()

    # Initialize arrays
    pAmp_out['CollocationPointsFreqsAmplitudeInsp'] = [0.0] * pWFHM['nCollocPtsInspAmp']
    pAmp_out['CollocationPointsValuesAmplitudeInsp'] = [0.0] * pWFHM['nCollocPtsInspAmp']

    AmpFreqsVersion = pWFHM['IMRPhenomXHMInspiralAmpFreqsVersion']

    if AmpFreqsVersion == 122022 or AmpFreqsVersion == 102021:
        # Standard collocation point frequencies
        pAmp_out['CollocationPointsFreqsAmplitudeInsp'][0] = 0.5 * pAmp['fAmpMatchIN']
        pAmp_out['CollocationPointsFreqsAmplitudeInsp'][1] = 0.75 * pAmp['fAmpMatchIN']
        pAmp_out['CollocationPointsFreqsAmplitudeInsp'][2] = pAmp['fAmpMatchIN']
    else:
        raise ValueError(
            f"Error in IMRPhenomXHM_Inspiral_Amp_CollocationPoints: "
            f"IMRPhenomXHMInspiralAmpFreqsVersion = {AmpFreqsVersion} is not valid. "
            f"Recommended version is 102021."
        )

    # Evaluate fits at collocation points
    for i in range(pWFHM['nCollocPtsInspAmp']):
        pAmp_out['CollocationPointsValuesAmplitudeInsp'][i] = jnp.abs(
            pAmp['InspiralAmpFits'][pWFHM['modeInt'] * pWFHM['nCollocPtsInspAmp'] + i](
                pWF22, pWFHM['IMRPhenomXHMInspiralAmpFitsVersion']
            )
        )

    # Apply veto if requested
    if pWFHM.get('InspiralAmpVeto', 0) == 1:
        pWFHM_out['IMRPhenomXHMInspiralAmpVersion'] = 13

    return pAmp_out, pWFHM_out


def IMRPhenomXHM_Inspiral_Amp_Coefficients(
    pAmp: dict,
    powers_of_Mf_inspcollpoints: list,
    pWFHM: dict
) -> dict:
    """
    Compute inspiral amplitude reconstruction coefficients.

    This function computes polynomial coefficients for the inspiral amplitude
    reconstruction based on the collocation point values and the PN amplitude.
    Different reconstruction schemes are used depending on which collocation
    points pass the veto criteria.

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients structure containing:
        - CollocationPointsValuesAmplitudeInsp: Amplitude values at collocation points
        - PNAmplitudeInsp: PN amplitude values at collocation points
        - PNdominant: Dominant PN term normalization
    powers_of_Mf_inspcollpoints : list
        List of power dictionaries for each collocation point plus matching frequency
    pWFHM : dict
        Waveform structure containing:
        - IMRPhenomXHMInspiralAmpVersion: Version flag determining reconstruction scheme

    Returns
    -------
    dict
        Updated pAmp dictionary with InspiralCoefficient array computed
    """
    N_MAX_COEFFICIENTS_AMPLITUDE_INS = 10  # Maximum number of coefficients
    pAmp_out = pAmp.copy()

    # Initialize coefficients
    pAmp_out['InspiralCoefficient'] = [0.0] * N_MAX_COEFFICIENTS_AMPLITUDE_INS

    # Get power structures
    powers_of_f1 = powers_of_Mf_inspcollpoints[0]
    powers_of_f2 = powers_of_Mf_inspcollpoints[1]
    powers_of_f3 = powers_of_Mf_inspcollpoints[2]
    powers_of_finsp = powers_of_Mf_inspcollpoints[3]

    # Compute normalized residuals at collocation points
    v1 = (pAmp['CollocationPointsValuesAmplitudeInsp'][0] - pAmp['PNAmplitudeInsp'][0]) / \
         pAmp['PNdominant'] / powers_of_f1['m_seven_sixths']
    v2 = (pAmp['CollocationPointsValuesAmplitudeInsp'][1] - pAmp['PNAmplitudeInsp'][1]) / \
         pAmp['PNdominant'] / powers_of_f2['m_seven_sixths']
    v3 = (pAmp['CollocationPointsValuesAmplitudeInsp'][2] - pAmp['PNAmplitudeInsp'][2]) / \
         pAmp['PNdominant'] / powers_of_f3['m_seven_sixths']

    InspAmpVersion = pWFHM['IMRPhenomXHMInspiralAmpVersion']

    if InspAmpVersion == 1:
        # Use only first collocation point (constant reconstruction)
        pAmp_out['InspiralCoefficient'][0] = (
            (powers_of_finsp['seven_thirds'] * v1) / powers_of_f1['seven_thirds']
        )

    elif InspAmpVersion == 2:
        # Use only second collocation point
        pAmp_out['InspiralCoefficient'][0] = (
            (powers_of_finsp['seven_thirds'] * v2) / powers_of_f2['seven_thirds']
        )

    elif InspAmpVersion == 3:
        # Use only third collocation point
        pAmp_out['InspiralCoefficient'][0] = (
            (powers_of_finsp['seven_thirds'] * v3) / powers_of_f3['seven_thirds']
        )

    elif InspAmpVersion == 12:
        # Use first and second collocation points (linear reconstruction)
        pAmp_out['InspiralCoefficient'][0] = (
            powers_of_finsp['seven_thirds'] *
            (-(powers_of_f2['eight_thirds'] * v1) + powers_of_f1['eight_thirds'] * v2) /
            (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
             powers_of_f2['seven_thirds'])
        )
        pAmp_out['InspiralCoefficient'][1] = (
            powers_of_finsp['eight_thirds'] *
            (powers_of_f1['m_seven_thirds'] * v1 - powers_of_f2['m_seven_thirds'] * v2) /
            (powers_of_f1['one_third'] - powers_of_f2['one_third'])
        )

    elif InspAmpVersion == 13:
        # Use first and third collocation points
        pAmp_out['InspiralCoefficient'][0] = (
            powers_of_finsp['seven_thirds'] *
            (-(powers_of_f3['eight_thirds'] * v1) + powers_of_f1['eight_thirds'] * v3) /
            (powers_of_f1['seven_thirds'] * (powers_of_f1['one_third'] - powers_of_f3['one_third']) *
             powers_of_f3['seven_thirds'])
        )
        pAmp_out['InspiralCoefficient'][1] = (
            powers_of_finsp['eight_thirds'] *
            (powers_of_f1['m_seven_thirds'] * v1 - powers_of_f3['m_seven_thirds'] * v3) /
            (powers_of_f1['one_third'] - powers_of_f3['one_third'])
        )

    elif InspAmpVersion == 23:
        # Use second and third collocation points
        pAmp_out['InspiralCoefficient'][0] = (
            powers_of_finsp['seven_thirds'] *
            (-(powers_of_f3['eight_thirds'] * v2) + powers_of_f2['eight_thirds'] * v3) /
            (powers_of_f2['seven_thirds'] * (powers_of_f2['one_third'] - powers_of_f3['one_third']) *
             powers_of_f3['seven_thirds'])
        )
        pAmp_out['InspiralCoefficient'][1] = (
            powers_of_finsp['eight_thirds'] *
            (powers_of_f1['m_seven_thirds'] * v1 - powers_of_f3['m_seven_thirds'] * v3) /
            (powers_of_f1['one_third'] - powers_of_f3['one_third'])
        )

    elif InspAmpVersion == 123:
        # Use all three collocation points (quadratic reconstruction)
        denom = (powers_of_f1['seven_thirds'] *
                 (powers_of_f1['one_third'] - powers_of_f2['one_third']) *
                 powers_of_f2['seven_thirds'] *
                 (powers_of_f1['one_third'] - powers_of_f3['one_third']) *
                 (powers_of_f2['one_third'] - powers_of_f3['one_third']) *
                 powers_of_f3['seven_thirds'])

        pAmp_out['InspiralCoefficient'][0] = (
            powers_of_finsp['seven_thirds'] *
            (-(powers_of_f1['three'] * powers_of_f3['eight_thirds'] * v2) +
             powers_of_f1['eight_thirds'] * powers_of_f3['three'] * v2 +
             powers_of_f2['three'] * (powers_of_f3['eight_thirds'] * v1 - powers_of_f1['eight_thirds'] * v3) +
             powers_of_f2['eight_thirds'] * (-(powers_of_f3['three'] * v1) + powers_of_f1['three'] * v3)) /
            denom
        )

        pAmp_out['InspiralCoefficient'][1] = (
            powers_of_finsp['eight_thirds'] *
            (powers_of_f1['three'] * powers_of_f3['seven_thirds'] * v2 -
             powers_of_f1['seven_thirds'] * powers_of_f3['three'] * v2 +
             powers_of_f2['three'] * (-(powers_of_f3['seven_thirds'] * v1) + powers_of_f1['seven_thirds'] * v3) +
             powers_of_f2['seven_thirds'] * (powers_of_f3['three'] * v1 - powers_of_f1['three'] * v3)) /
            denom
        )

        pAmp_out['InspiralCoefficient'][2] = (
            powers_of_finsp['three'] *
            (powers_of_f1['seven_thirds'] * (-powers_of_f1['one_third'] + powers_of_f3['one_third']) *
             powers_of_f3['seven_thirds'] * v2 +
             powers_of_f2['seven_thirds'] * (-(powers_of_f3['eight_thirds'] * v1) + powers_of_f1['eight_thirds'] * v3) +
             powers_of_f2['eight_thirds'] * (powers_of_f3['seven_thirds'] * v1 - powers_of_f1['seven_thirds'] * v3)) /
            denom
        )

    return pAmp_out


def IMRPhenomXHM_Get_Inspiral_Amp_Coefficients(pAmp: dict, pWFHM: dict, pWF22: dict) -> dict:
    """
    Main function to compute all inspiral amplitude coefficients.

    This is the top-level function that orchestrates the computation of inspiral
    amplitude coefficients. It:
    1. Computes collocation point frequencies and values
    2. Initializes power structures for all collocation points
    3. Evaluates PN amplitude at collocation points
    4. Computes reconstruction coefficients

    Parameters
    ----------
    pAmp : dict
        Amplitude coefficients structure containing fit function arrays
    pWFHM : dict
        Waveform structure for the higher mode containing:
        - nCollocPtsInspAmp: Number of collocation points (typically 3)
        - Various version flags and parameters
    pWF22 : dict
        Waveform structure for the 22 mode

    Returns
    -------
    dict
        Updated pAmp dictionary with all inspiral amplitude coefficients computed:
        - CollocationPointsFreqsAmplitudeInsp: Collocation point frequencies
        - CollocationPointsValuesAmplitudeInsp: Amplitude values at collocation points
        - PNAmplitudeInsp: PN amplitude at collocation points
        - InspiralCoefficient: Reconstruction polynomial coefficients
        - fcutInsp_seven_thirds, fcutInsp_eight_thirds, fcutInsp_three: Power terms
    """
    # Get collocation points
    pAmp_updated, pWFHM_updated = IMRPhenomXHM_Inspiral_Amp_CollocationPoints(pAmp, pWFHM, pWF22)

    # Initialize power structures for collocation points + matching frequency
    powers_of_Mf_inspcollpoints = []
    for i in range(pWFHM['nCollocPtsInspAmp']):
        powers_of_Mf_inspcollpoints.append(
            compute_powers_of_f(pAmp_updated['CollocationPointsFreqsAmplitudeInsp'][i])
        )

    # Add power structure for matching frequency
    powers_of_Mf_inspcollpoints.append(compute_powers_of_f(pAmp_updated['fAmpMatchIN']))

    # Store specific powers for matching frequency
    pAmp_updated['fcutInsp_seven_thirds'] = powers_of_Mf_inspcollpoints[pWFHM['nCollocPtsInspAmp']]['seven_thirds']
    pAmp_updated['fcutInsp_eight_thirds'] = powers_of_Mf_inspcollpoints[pWFHM['nCollocPtsInspAmp']]['eight_thirds']
    pAmp_updated['fcutInsp_three'] = powers_of_Mf_inspcollpoints[pWFHM['nCollocPtsInspAmp']]['three']

    # Initialize PN amplitude array
    pAmp_updated['PNAmplitudeInsp'] = [0.0] * pWFHM['nCollocPtsInspAmp']

    # Evaluate PN amplitude at collocation points
    for i in range(pWFHM['nCollocPtsInspAmp']):
        pAmp_updated['PNAmplitudeInsp'][i] = IMRPhenomXHM_Inspiral_PNAmp_Ansatz(
            powers_of_Mf_inspcollpoints[i], pWFHM_updated, pAmp_updated
        )

    # Compute reconstruction coefficients
    pAmp_final = IMRPhenomXHM_Inspiral_Amp_Coefficients(
        pAmp_updated, powers_of_Mf_inspcollpoints, pWFHM_updated
    )

    return pAmp_final

