"""
IMRPhenomXHM.py — JAX port of LAL's PhenomXHM higher-mode infrastructure.

Ports the 122019-release code from:
  LALSimIMRPhenomXHM_qnm.c          — QNM fits (fRING, fDAMP per mode)
  LALSimIMRPhenomXHM_internals.c    — waveform struct, coefficient computation
  LALSimIMRPhenomXHM_inspiral.c     — inspiral parameter-space fits
  LALSimIMRPhenomXHM_intermediate.c — intermediate parameter-space fits
  LALSimIMRPhenomXHM_ringdown.c     — ringdown parameter-space fits
  LALSimIMRPhenomX_internals.c      — IMRPhenomX_TimeShift_22, XLALSimIMRPhenomXLinb

The 22-mode reuses ripple's existing IMRPhenomXAS.py (Phase + Amp functions).
"""

from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array

from .IMRPhenomXAS import (
    get_inspiral_phase,
    get_mergerringdown_Amp,
    Phase as IMRPhenomXAS_Phase,
    Amp as IMRPhenomXAS_Amp,
)
from . import IMRPhenomX_utils
from ..constants import PI, MTSUN, MPC, C, MRSUN
from .spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)

# ---------------------------------------------------------------------------
# Section 1: QNM fits
# ---------------------------------------------------------------------------
# All fits are rational polynomials in final spin `a`.
# Source: LALSimIMRPhenomXHM_qnm.c


def evaluate_QNMfit_fring21(a: float) -> float:
    """Ringdown frequency for (2,1) mode as a function of final spin."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    num = (
        0.059471695665734674
        - 0.07585416297991414 * a
        + 0.021967909664591865 * x2
        - 0.0018964744613388146 * x3
        + 0.001164879406179587 * x4
        - 0.0003387374454044957 * x5
    )
    den = 1.0 - 1.4437415542456158 * a + 0.49246920313191234 * x2
    return num / den


def evaluate_QNMfit_fdamp21(a: float) -> float:
    """Damping frequency for (2,1) mode as a function of final spin."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    num = (
        2.0696914454467294
        - 3.1358071947583093 * a
        + 0.14456081596393977 * x2
        + 1.2194717985037946 * x3
        - 0.2947372598589144 * x4
        + 0.002943057145913646 * x5
    )
    den = (
        146.1779212636481
        - 219.81790388304876 * a
        + 17.7141194900164 * x2
        + 75.90115083917898 * x3
        - 18.975287709794745 * x4
    )
    return num / den


def evaluate_QNMfit_fring33(a: float) -> float:
    """Ringdown frequency for (3,3) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    num = (
        0.09540436245212061
        - 0.22799517865876945 * a
        + 0.13402916709362475 * x2
        + 0.03343753057911253 * x3
        - 0.030848060170259615 * x4
        - 0.006756504382964637 * x5
        + 0.0027301732074159835 * x6
    )
    den = (
        1.0
        - 2.7265947806178334 * a
        + 2.144070539525238 * x2
        - 0.4706873667569393 * x4
        + 0.05321818246993958 * x6
    )
    return num / den


def evaluate_QNMfit_fdamp33(a: float) -> float:
    """Damping frequency for (3,3) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    num = (
        0.014754148319335946
        - 0.03124423610028678 * a
        + 0.017192623913708124 * x2
        + 0.001034954865629645 * x3
        - 0.0015925124814622795 * x4
        - 0.0001414350555699256 * x5
    )
    den = (
        1.0 - 2.0963684630756894 * a + 1.196809702382645 * x2 - 0.09874113387889819 * x4
    )
    return num / den


def evaluate_QNMfit_fring32(a: float) -> float:
    """Ringdown frequency for (3,2) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    num = (
        0.09540436245212061
        - 0.13628306966373951 * a
        + 0.030099881830507727 * x2
        - 0.000673589757007597 * x3
        + 0.0118277880067919 * x4
        + 0.0020533816327907334 * x5
        - 0.0015206141948469621 * x6
    )
    den = (
        1.0
        - 1.6531854335715193 * a
        + 0.5634705514193629 * x2
        + 0.12256204148002939 * x4
        - 0.027297817699401976 * x6
    )
    return num / den


def evaluate_QNMfit_fdamp32(a: float) -> float:
    """Damping frequency for (3,2) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    num = (
        0.014754148319335946
        - 0.03445752346074498 * a
        + 0.02168855041940869 * x2
        + 0.0014945908223317514 * x3
        - 0.0034761714223258693 * x4
    )
    den = (
        1.0 - 2.320722660848874 * a + 1.5096146036915865 * x2 - 0.18791187563554512 * x4
    )
    return num / den


def evaluate_QNMfit_fring44(a: float) -> float:
    """Ringdown frequency for (4,4) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    num = (
        0.1287821193485683
        - 0.21224284094693793 * a
        + 0.0710926778043916 * x2
        + 0.015487322972031054 * x3
        - 0.002795401084713644 * x4
        + 0.000045483523029172406 * x5
        + 0.00034775290179000503 * x6
    )
    den = (
        1.0
        - 1.9931645124693607 * a
        + 1.0593147376898773 * x2
        - 0.06378640753152783 * x4
    )
    return num / den


def evaluate_QNMfit_fdamp44(a: float) -> float:
    """Damping frequency for (4,4) mode."""
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    num = (
        0.014986847152355699
        - 0.01722587715950451 * a
        - 0.0016734788189065538 * x2
        + 0.0002837322846047305 * x3
        + 0.002510528746148588 * x4
        + 0.00031983835498725354 * x5
        + 0.000812185411753066 * x6
    )
    den = (
        1.0
        - 1.1350205970682399 * a
        - 0.0500827971270845 * x2
        + 0.13983808071522857 * x4
        + 0.051876225199833995 * x6
    )
    return num / den


# ---------------------------------------------------------------------------
# Section 2: Waveform struct
# ---------------------------------------------------------------------------


@dataclass
class XHMWaveformStruct:
    """
    Per-mode waveform variables. Mirrors IMRPhenomXHMWaveformStruct from
    LALSimIMRPhenomXHM_structs.h. Populated by xhm_set_waveform_variables.
    """

    ell: int
    emm: int
    modeTag: int  # ell*10 + emm
    modeInt: int  # 0=21, 1=33, 2=32, 3=44

    fRING: float  # lm QNM ringdown frequency (geometric units, M=1)
    fDAMP: float  # lm QNM damping frequency (geometric units, M=1)
    fMECOlm: float  # fMECO * emm/2

    chi_s: float  # (chi1L + chi2L)/2
    chi_a: float  # (chi1L - chi2L)/2

    MixingOn: bool  # True only for (3,2)
    Ampzero: bool  # True if amplitude is zero (e.g. equal mass for odd modes)

    # Version/collocation counts
    nCollocPtsInspAmp: int  # = 3
    nCollocPtsInterAmp: int  # = 2
    nCollocPtsRDPhase: int  # = 3 for 32, else 4
    nCollocPtsInterPhase: int  # = 6 for 32, else 5
    etaEMR: float  # = 0.05

    # Model version parameters
    XHMRingdownAmpVersion: int


def build_pWF22(
    m1: float | Array,
    m2: float | Array,
    chi1z: float | Array,
    chi2z: float | Array,
    f_ref: float,
    chip: float | Array = 0.0,
    msa_SAv2: float | Array | None = None,
    msa_S1L_pav: float | Array | None = None,
    msa_S2L_pav: float | Array | None = None,
) -> dict:
    """
    Build the 22-mode waveform parameter dict needed by XHM functions.

    Contains all spin/mass combinations and 22-mode QNM frequencies.
    All frequencies are in geometric units (dimensionless: M_total * f in Hz).

    chip: in-plane spin parameter for afinal_prec = sign(a)*sqrt((chip*mm1^2)^2+a^2).
    When called from XP/XPHM pass chiTot_perp (not chip_p), because LAL defaults to
    PhenomXPFinalSpinMod=4 which uses |S1_perp+S2_perp| / mm1^2 as the in-plane spin,
    giving Sperp = chiTot_perp*mm1^2 = |S1_perp+S2_perp|. This sets fRING22/fDAMP22
    from the precessing final spin (pWF->afinal = afinal_prec).

    pWF22 keys:
      eta, delta, S, STotR, dchi, chi1L, chi2L
      afinal, finmass (= 1 - Erad)
      fMECO, fRING22, fDAMP22
      theta (tuple: m1_Msun, m2_Msun, chi1z, chi2z)
      phase_coeffs (PhenomD fitting coefficients for XAS phase)
      Mf_ref (geometric reference frequency)
    """
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / M_s**2
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    chi_eff = mm1 * chi1z + mm2 * chi2z
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1z + chi2z)) / (1.0 - 76.0 * eta / 113.0)
    STotR = (mm1**2 * chi1z + mm2**2 * chi2z) / (mm1**2 + mm2**2)
    dchi = chi1z - chi2z

    # Recompute afinal and Erad (same formula as IMRPhenomX_utils.get_cutoff_fMs)
    # but also exposed here for XHM use.
    STotR2 = STotR * STotR
    STotR3 = STotR2 * STotR
    dchi2 = dchi * dchi
    eta4 = eta3 * eta

    afinal = (
        (3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3)
        / (1.0 + 7.2388440419467335 * eta)
        + (
            (mm1**2 + mm2**2) * STotR
            + (
                (
                    -0.8561951310209386 * eta
                    - 0.09939065676370885 * eta2
                    + 1.668810429851045 * eta3
                )
                * STotR
                + (
                    0.5881660363307388 * eta
                    - 2.149269067519131 * eta2
                    + 3.4768263932898678 * eta3
                )
                * STotR2
                + (
                    0.142443244743048 * eta
                    - 0.9598353840147513 * eta2
                    + 1.9595643107593743 * eta3
                )
                * STotR3
            )
            / (
                1.0
                + (
                    -0.9142232693081653
                    + 2.3191363426522633 * eta
                    - 9.710576749140989 * eta3
                )
                * STotR
            )
        )
        + (
            0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta) * eta2
            - 0.059808322561702126 * dchi2 * eta3
            + 2.3170397514509933
            * dchi
            * delta
            * (1 - 3.2624649875884852 * eta)
            * eta3
            * STotR
        )
    )

    Erad = (
        (
            0.057190958417936644 * eta
            + 0.5609904135313374 * eta2
            - 0.84667563764404 * eta3
            + 3.145145224278187 * eta4
        )
        * (
            1.0
            + (
                -0.13084389181783257
                - 1.1387311580238488 * eta
                + 5.49074464410971 * eta2
            )
            * STotR
            + (-0.17762802148331427 + 2.176667900182948 * eta2) * STotR2
            + (
                -0.6320191645391563
                + 4.952698546796005 * eta
                - 10.023747993978121 * eta2
            )
            * STotR3
        )
    ) / (
        1.0
        + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2)
        * STotR
    ) + (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653
        * dchi
        * delta
        * (1 - 4.91667749015812 * eta)
        * eta
        * STotR
    )
    finmass = 1.0 - Erad

    # Precessing final spin: LAL sets pWF->afinal = afinal_prec before computing
    # fRING/fDAMP when chip > 0. When chip=0 this reduces to afinal identically.
    #
    # For PrecVersion=222 (precessing_tag=2), LAL redirects fsflag=4→3 and uses the
    # MSA orbit-averaged formula: afinal_prec = sqrt(SAv2 + Lfinal^2 + 2*Lfinal*(S1L_pav+S2L_pav))
    # where Lfinal = afinal_aln - mm1^2*chi1z - mm2^2*chi2z. Pass msa_* quantities
    # from IMRPhenomX_Initialize_MSA_System to enable this path; otherwise uses chip.
    if msa_SAv2 is not None:
        a_aln = afinal  # aligned-spin final spin (before precessing correction)
        Lfinal = a_aln - mm1**2 * chi1z - mm2**2 * chi2z
        afinal_prec = jnp.sqrt(
            msa_SAv2 + Lfinal**2 + 2.0 * Lfinal * (msa_S1L_pav + msa_S2L_pav)
        )
        chip = jnp.sqrt(jnp.maximum(afinal_prec**2 - a_aln**2, 0.0)) / (mm1**2)
        afinal = jnp.copysign(1.0, a_aln) * afinal_prec
    else:
        Sperp_prec = chip * mm1 * mm1  # chip * (m1/M)^2
        afinal = jnp.copysign(1.0, afinal) * jnp.sqrt(Sperp_prec**2 + afinal**2)

    fRING22, fDAMP22, fMECO, fISCO22 = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1z, chi2z, chip=chip
    )

    chi_eff = mm1 * chi1z + mm2 * chi2z
    # chiPN = LAL's chiPNHat: (chi_eff - (38/113)*eta*(chi1+chi2)) / (1 - 76*eta/113)
    # Source: XLALSimIMRPhenomXchiPNHat in LALSimIMRPhenomXUtilities.c
    chiPN = (chi_eff - (38.0 / 113.0) * eta * (chi1z + chi2z)) / (
        1.0 - 76.0 * eta / 113.0
    )
    # Amplitude normalization: sqrt(2*eta/3) * pi^(-1/6)
    # Source: LALSimIMRPhenomX_internals.c line 610:
    #   wf->ampNorm = sqrt(2.0/3.0) * sqrt(wf->eta) * powers_of_lalpi.m_one_sixth;
    ampNorm = jnp.sqrt(2.0 * eta / 3.0) * PI ** (-1.0 / 6.0)

    theta = jnp.array([m1, m2, chi1z, chi2z])
    # IMRPhenomXAS_Phase expects the precomputed 2D coefficient table (shape 13×49),
    # not the PhenomD 1D coefficients from get_XAS_coeffs.
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    Mf_ref = f_ref * M_s

    return dict(
        eta=eta,
        delta=delta,
        S=S,
        STotR=STotR,
        dchi=dchi,
        chi1L=chi1z,
        chi2L=chi2z,
        afinal=afinal,
        finmass=finmass,
        fMECO=fMECO,
        fRING22=fRING22,
        fDAMP22=fDAMP22,
        fISCO22=fISCO22,
        chi_eff=chi_eff,
        chiPN=chiPN,
        ampNorm=ampNorm,
        theta=theta,
        phase_coeffs=phase_coeffs,
        Mf_ref=Mf_ref,
        M_s=M_s,
        chip=chip,
    )


# modeInt mapping: 0=21, 1=33, 2=32, 3=44
_MODE_INT = {(2, 1): 0, (3, 3): 1, (3, 2): 2, (4, 4): 3}
_QNM_FRING = [
    evaluate_QNMfit_fring21,
    evaluate_QNMfit_fring33,
    evaluate_QNMfit_fring32,
    evaluate_QNMfit_fring44,
]
_QNM_FDAMP = [
    evaluate_QNMfit_fdamp21,
    evaluate_QNMfit_fdamp33,
    evaluate_QNMfit_fdamp32,
    evaluate_QNMfit_fdamp44,
]


def xhm_set_waveform_variables(ell: int, emm: int, pWF22: dict) -> XHMWaveformStruct:
    """
    Compute per-mode struct from the 22-mode waveform parameters.

    Mirrors IMRPhenomXHM_SetHMWaveformVariables in LALSimIMRPhenomXHM_internals.c.

    pWF22 must be built by build_pWF22.
    """
    modeInt = _MODE_INT[(ell, emm)]
    afinal = pWF22["afinal"]
    finmass = pWF22["finmass"]

    fRING = _QNM_FRING[modeInt](afinal) / finmass
    fDAMP = _QNM_FDAMP[modeInt](afinal) / finmass
    fMECOlm = pWF22["fMECO"] * emm / 2.0

    chi_s = (pWF22["chi1L"] + pWF22["chi2L"]) / 2.0
    chi_a = (pWF22["chi1L"] - pWF22["chi2L"]) / 2.0

    MixingOn = ell == 3 and emm == 2
    # Ampzero: odd modes vanish at equal mass (delta=0); check conservatively
    Ampzero = False  # evaluated per-frequency; not statically zero for GW150914
    XHMRingdownAmpVersion = 1 if modeInt == 2 else 0

    return XHMWaveformStruct(
        ell=ell,
        emm=emm,
        modeTag=ell * 10 + emm,
        modeInt=modeInt,
        fRING=fRING,
        fDAMP=fDAMP,
        fMECOlm=fMECOlm,
        chi_s=chi_s,
        chi_a=chi_a,
        MixingOn=MixingOn,
        Ampzero=Ampzero,
        nCollocPtsInspAmp=3,
        nCollocPtsInterAmp=2,
        nCollocPtsRDPhase=4,
        nCollocPtsInterPhase=6 if MixingOn else 5,
        etaEMR=0.05,
        XHMRingdownAmpVersion=XHMRingdownAmpVersion,
    )


# ---------------------------------------------------------------------------
# Section 3: IMRPhenomX_TimeShift_22 and helpers
# ---------------------------------------------------------------------------


def XLALSimIMRPhenomXLinb(eta: float, STotR: float, dchi: float, delta: float) -> float:
    """
    Linear-in-frequency coefficient of the 22 phase (group delay offset).

    Port of XLALSimIMRPhenomXLinb in LALSimIMRPhenomXUtilities.c.
    Uses STotR (not the chi_eff-derived S) as the spin parameter.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    S = STotR
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    noSpin = (
        3155.1635543201924
        + 1257.9949740608242 * eta
        - 32243.28428870599 * eta2
        + 347213.65466875216 * eta3
        - 1.9223851649491738e6 * eta4
        + 5.3035911346921865e6 * eta5
        - 5.789128656876938e6 * eta6
    )
    eqSpin = (
        (-24.181508118588667 + 115.49264174560281 * eta - 380.19778216022763 * eta2) * S
        + (24.72585609641552 - 328.3762360751952 * eta + 725.6024119989094 * eta2) * S2
        + (23.404604124552 - 646.3410199799737 * eta + 1941.8836639529036 * eta2) * S3
        + (-12.814828278938885 - 325.92980012408367 * eta + 1320.102640190539 * eta2)
        * S4
    )
    uneqSpin = -148.17317525117338 * dchi * delta * eta2
    return noSpin + eqSpin + uneqSpin


def XLALSimIMRPhenomXPsi4ToStrain(eta: float, STotR: float, dchi: float) -> Array:
    """
    Psi4-to-strain conversion factor for the time-shift.

    Port of XLALSimIMRPhenomXPsi4ToStrain in LALSimIMRPhenomXUtilities.c.
    Uses STotR as the spin parameter.
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S = STotR
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    noSpin = (
        13.39320482758057
        - 175.42481512989315 * eta
        + 2097.425116152503 * eta2
        - 9862.84178637907 * eta3
        + 16026.897939722587 * eta4
    )
    eqSpin = (
        (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta2) * S
        + (1.3934428041390161 - 97.51812681228478 * eta + 376.9200932531847 * eta2) * S2
        + (15.649521097877374 + 137.33317057388916 * eta - 755.9566456906406 * eta2)
        * S3
        + (13.097315867845788 + 149.30405703643288 * eta - 764.5242164872267 * eta2)
        * S4
    )
    uneqSpin = (
        105.37711654943146 * dchi * jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0)) * eta2
    )
    return noSpin + eqSpin + uneqSpin


def IMRPhenomX_TimeShift_22(pWF22: dict) -> float:
    """
    Global time shift for the 22 mode that aligns the waveform peak near t=0.

    tshift = linb - dphi22Ref - 2*pi*(500 + psi4tostrain)

    where:
      linb         = XLALSimIMRPhenomXLinb(eta, STotR, dchi, delta)
      dphi22Ref    = d/d(Mf) [IMRPhenomXAS_Phase] at fRING22 - fDAMP22
                   = jax.grad(IMRPhenomXAS_Phase)(frefFit_Hz) / M_s
      psi4tostrain = XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi)

    PNR_DEV_PARAMETER is zero for default settings, so the NU0 correction is omitted.
    Port of IMRPhenomX_TimeShift_22 in LALSimIMRPhenomX_internals.c:2624.
    This is the critical fix for the t_c shift in PE.
    """
    eta = pWF22["eta"]
    STotR = pWF22["STotR"]
    dchi = pWF22["dchi"]
    delta = pWF22["delta"]
    M_s = pWF22["M_s"]
    fRING22 = pWF22["fRING22"]
    fDAMP22 = pWF22["fDAMP22"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]

    chip = pWF22.get("chip", 0.0)
    linb = XLALSimIMRPhenomXLinb(eta, STotR, dchi, delta)
    psi4tostrain = XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi)

    # frefFit = fRING22 - fDAMP22 in geometric units; convert to Hz for XAS.
    # dphi22Ref must use chip-corrected fRING inside IMRPhenomXAS_Phase so that
    # the ringdown Lorentzian and transition frequencies are consistent with frefFit.
    frefFit_Hz = (fRING22 - fDAMP22) / M_s
    dphi22Ref = jax.grad(
        lambda f_: IMRPhenomXAS_Phase(f_, theta, phase_coeffs, chip)
    )(frefFit_Hz) / M_s

    tshift = linb - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
    return tshift


# ---------------------------------------------------------------------------
# Section 4: Inspiral phase fits
# ---------------------------------------------------------------------------


def _xhm_insp_phase_LambdaPN(modeTag: int, eta: float) -> float | Array:
    """
    Leading-order PN phase correction for each mode's inspiral ansatz.

    LambdaPN is the linear-in-Mf term coming from the complex PN amplitude,
    added to the XAS inspiral phase at the rescaled frequency.

    Source: IMRPhenomXHM_Insp_Phase_LambdaPN in LALSimIMRPhenomXHM_inspiral.c:1051.
    Returns -output (note sign convention in LAL).
    modeTag: 21, 33, 32, 44. For eta <= 0.01 (EMR), use parameter-space fits instead.
    """
    if modeTag == 21:
        # 2*pi*(-0.5 - log(16)/2) = 2*pi*(-0.5 - 2*log(2))
        output = 2.0 * PI * (-0.5 - 2.0 * jnp.log(2.0))
    elif modeTag == 33:
        # 2/3*pi*(-21/5 + 6*log(3/2))
        output = (2.0 / 3.0) * PI * (-21.0 / 5.0 + 6.0 * jnp.log(1.5))
    elif modeTag == 32:
        # -((2376*pi*(-5 + 22*eta))/(-3960 + 11880*eta))
        output = -((2376.0 * PI * (-5.0 + 22.0 * eta)) / (-3960.0 + 11880.0 * eta))
    else:  # 44
        # 45045*pi*(336 - 1193*eta + 320*(-1+3*eta)*log(2)) / (2*(-1801800 + 5405400*eta))
        output = (
            45045.0
            * PI
            * (336.0 - 1193.0 * eta + 320.0 * (-1.0 + 3.0 * eta) * jnp.log(2.0))
            / (2.0 * (-1801800.0 + 5405400.0 * eta))
        )
    return -output


# ---------------------------------------------------------------------------
# Section 5: Intermediate phase fits
# ---------------------------------------------------------------------------


def _xhm_inter_phase_colloc_pts(
    modeInt: int,
    eta: float,
    STotR: float,
    dchi: float,
    delta: float,
    chi1L: float,
    chi2L: float,
) -> Array:
    """
    6 intermediate phase collocation point derivative values (p1..p6).

    These are derivative values dphi/dMf at the 6 collocation frequencies.
    DeltaT is added at the higher level (xhm_get_phase_coefficients).

    Source: IMRPhenomXHM_Inter_Phase_* in LALSimIMRPhenomXHM_intermediate.c (122019).
    Returns shape (6,) array.
    modeInt: 0=21, 1=33, 2=32, 3=44
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S

    if modeInt == 0:  # 21
        p1_noSpin = (
            4045.84
            + 7.63226 / eta
            - 1956.93 * eta
            - 23428.1 * eta2
            + 369153.0 * eta3
            - 2.28832e6 * eta4
            + 6.82533e6 * eta5
            - 7.86254e6 * eta6
        )
        p1_eqSpin = (
            -347.273 * S
            + 83.5428 * S2
            - 355.67 * S3
            + (4.44457 * S + 16.5548 * S2 + 13.6971 * S3) / eta
            + eta * (-79.761 * S - 355.299 * S2 + 1114.51 * S3 - 1077.75 * S4)
            + 92.6654 * S4
            + eta2 * (-619.837 * S - 722.787 * S2 + 2392.73 * S3 + 2689.18 * S4)
        )
        p1_uneqSpin = (918.976 * (chi1L - chi2L) * delta) * eta + (
            91.7679 * (chi1L - chi2L) * delta
        ) * eta2
        p1 = p1_noSpin + p1_eqSpin + p1_uneqSpin

        p2_noSpin = (
            3509.09
            + 0.91868 / eta
            + 194.72 * eta
            - 27556.2 * eta2
            + 369153.0 * eta3
            - 2.28832e6 * eta4
            + 6.82533e6 * eta5
            - 7.86254e6 * eta6
        )
        p2_eqSpin = (
            (0.7083999999999999 - 60.1611 * eta + 131.815 * eta2 - 619.837 * eta3) * S
            + (6.104720000000001 - 59.2068 * eta + 278.588 * eta2 - 722.787 * eta3) * S2
            + (5.7791 + 117.913 * eta - 1180.4 * eta2 + 2392.73 * eta3) * S3
            + eta * (92.6654 - 1077.75 * eta + 2689.18 * eta2) * S4
        ) / eta
        p2_uneqSpin = (
            -91.7679
            * delta
            * eta
            * (chi1L * (-1.6012352903357276 - eta) + chi2L * (1.6012352903357276 + eta))
        )
        p2 = p2_noSpin + p2_eqSpin + p2_uneqSpin

        p3_noSpin = (
            3241.68
            + 890.016 * eta
            - 28651.9 * eta2
            + 369153.0 * eta3
            - 2.28832e6 * eta4
            + 6.82533e6 * eta5
            - 7.86254e6 * eta6
        )
        p3_eqSpin = (
            (-2.2484 + 187.641 * eta - 619.837 * eta2) * S
            + (3.22603 + 166.323 * eta - 722.787 * eta2) * S2
            + (117.913 - 1094.59 * eta + 2392.73 * eta2) * S3
            + (92.6654 - 1077.75 * eta + 2689.18 * eta2) * S4
        )
        p3_uneqSpin = 91.7679 * dchi * delta * eta2
        p3 = p3_noSpin + p3_eqSpin + p3_uneqSpin

        p4_noSpin = (
            3160.88
            + 974.355 * eta
            - 28932.5 * eta2
            + 369780.0 * eta3
            - 2.28832e6 * eta4
            + 6.82533e6 * eta5
            - 7.86254e6 * eta6
        )
        p4_eqSpin = (
            (26.3355 - 196.851 * eta + 438.401 * eta2) * S
            + (45.9957 - 256.248 * eta + 117.563 * eta2) * S2
            + (-20.0261 + 467.057 * eta - 1613.0 * eta2) * S3
            + (-61.7446 + 577.057 * eta - 1096.81 * eta2) * S4
        )
        p4_uneqSpin = 65.3326 * dchi * delta * eta2
        p4 = p4_noSpin + p4_eqSpin + p4_uneqSpin

        p5_noSpin = 3102.36 + 315.911 * eta - 1688.26 * eta2 + 3635.76 * eta3
        p5_eqSpin = (
            (-23.0959 + 320.93 * eta - 1029.76 * eta2) * S
            + (-49.5435 + 826.816 * eta - 3079.39 * eta2) * S2
            + (40.7054 - 365.842 * eta + 1094.11 * eta2) * S3
            + (81.8379 - 1243.26 * eta + 4689.22 * eta2) * S4
        )
        p5_uneqSpin = 119.014 * dchi * delta * eta2
        p5 = p5_noSpin + p5_eqSpin + p5_uneqSpin

        p6_noSpin = 3089.18 + 4.89194 * eta + 190.008 * eta2 - 255.245 * eta3
        p6_eqSpin = (
            (2.96997 + 57.1612 * eta - 432.223 * eta2) * S
            + (-18.8929 + 630.516 * eta - 2804.66 * eta2) * S2
            + (-24.6193 + 549.085 * eta2) * S3
            + (-12.8798 - 722.674 * eta + 3967.43 * eta2) * S4
        )
        p6_uneqSpin = 74.0984 * dchi * delta * eta2
        p6 = p6_noSpin + p6_eqSpin + p6_uneqSpin

    elif modeInt == 1:  # 33
        p1_noSpin = (
            4360.19
            + 4.27128 / eta
            - 8727.4 * eta
            + 18485.9 * eta2
            + 371303.0 * eta3
            - 3.22792e6 * eta4
            + 1.01799e7 * eta5
            - 1.15659e7 * eta6
        )
        p1_eqSpin = (
            (
                11.6635
                - 251.579 * eta
                - 3255.6400000000003 * eta2
                + 19614.6 * eta3
                - 34860.2 * eta4
            )
            * S
            + (
                14.8017
                + 204.025 * eta
                - 5421.92 * eta2
                + 36587.3 * eta3
                - 74299.5 * eta4
            )
            * S2
        ) / eta
        p1_uneqSpin = eta * (
            223.651 * (chi1L - chi2L) * delta * (3.9201300240106223 + eta)
        )
        p1 = p1_noSpin + p1_eqSpin + p1_uneqSpin

        p2_noSpin = (
            3797.06
            + 0.786684 / eta
            - 2397.09 * eta
            - 25514.0 * eta2
            + 518315.0 * eta3
            - 3.41708e6 * eta4
            + 1.01799e7 * eta5
            - 1.15659e7 * eta6
        )
        p2_eqSpin = (
            (
                6.7812399999999995
                + 39.4668 * eta
                - 3520.37 * eta2
                + 19614.6 * eta3
                - 34860.2 * eta4
            )
            * S
            + (
                4.80384
                + 293.215 * eta
                - 5914.61 * eta2
                + 36587.3 * eta3
                - 74299.5 * eta4
            )
            * S2
        ) / eta
        p2_uneqSpin = (
            -223.651
            * delta
            * eta
            * (chi1L * (-1.3095134830606614 - eta) + chi2L * (1.3095134830606614 + eta))
        )
        p2 = p2_noSpin + p2_eqSpin + p2_uneqSpin

        p3_noSpin = (
            3321.83
            + 1796.03 * eta
            - 52406.1 * eta2
            + 605028.0 * eta3
            - 3.52532e6 * eta4
            + 1.01799e7 * eta5
            - 1.15659e7 * eta6
        )
        p3_eqSpin = (223.601 - 3714.77 * eta + 19614.6 * eta2 - 34860.2 * eta3) * S + (
            314.317 - 5906.46 * eta + 36587.3 * eta2 - 74299.5 * eta3
        ) * S2
        p3_uneqSpin = 223.651 * dchi * delta * eta2
        p3 = p3_noSpin + p3_eqSpin + p3_uneqSpin

        p4_noSpin = (
            3239.44
            - 661.15 * eta
            + 5139.79 * eta2
            + 3456.2 * eta3
            - 248477.0 * eta4
            + 1.17255e6 * eta5
            - 1.70363e6 * eta6
        )
        p4_eqSpin = (
            (225.859 - 4150.09 * eta + 24364.0 * eta2 - 46537.3 * eta3) * S
            + (35.2439 - 994.971 * eta + 8953.98 * eta2 - 23603.5 * eta3) * S2
            + (-310.489 + 5946.15 * eta - 35337.1 * eta2 + 67102.4 * eta3) * S3
        )
        p4_uneqSpin = 30.484 * dchi * delta * eta2
        p4 = p4_noSpin + p4_eqSpin + p4_uneqSpin

        eta7 = eta6 * eta
        p5_noSpin = (
            3114.3
            + 2143.06 * eta
            - 49428.3 * eta2
            + 563997.0 * eta3
            - 3.35991e6 * eta4
            + 9.99745e6 * eta5
            - 1.17123e7 * eta6
        )
        p5_eqSpin = (
            (190.051 - 3705.08 * eta + 23046.2 * eta2 - 46537.3 * eta3) * S
            + (63.6615 - 1414.2 * eta + 10166.1 * eta2 - 23603.5 * eta3) * S2
            + (-257.524 + 5179.97 * eta - 33001.4 * eta2 + 67102.4 * eta3) * S3
        )
        p5_uneqSpin = 54.9833 * dchi * delta * eta2
        p5 = p5_noSpin + p5_eqSpin + p5_uneqSpin

        p6_noSpin = (
            3111.46
            + 384.121 * eta
            - 13003.6 * eta2
            + 179537.0 * eta3
            - 1.19313e6 * eta4
            + 3.79886e6 * eta5
            - 4.64858e6 * eta6
        )
        p6_eqSpin = (
            (182.864 - 3834.22 * eta + 24532.9 * eta2 - 50165.9 * eta3) * S
            + (21.0158 - 746.957 * eta + 6701.33 * eta2 - 17842.3 * eta3) * S2
            + (-292.855 + 5886.62 * eta - 37382.4 * eta2 + 75501.8 * eta3) * S3
        )
        p6_uneqSpin = 75.5162 * dchi * delta * eta2
        p6 = p6_noSpin + p6_eqSpin + p6_uneqSpin

    elif modeInt == 2:  # 32
        p1_noSpin = (
            4414.11
            + 4.21564 / eta
            - 10687.8 * eta
            + 58234.6 * eta2
            - 64068.4 * eta3
            - 704442.0 * eta4
            + 2.86393e6 * eta5
            - 3.26362e6 * eta6
        )
        p1_eqSpin = (
            (6.39833 - 610.267 * eta + 2095.72 * eta2 - 3970.89 * eta3) * S
            + (22.9567 - 99.1551 * eta + 331.593 * eta2 - 794.79 * eta3) * S2
            + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * S3
            + eta * (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        ) / eta
        p1_uneqSpin = (
            132.244
            * delta
            * eta
            * (chi1L * (6.227738120444028 - eta) + chi2L * (-6.227738120444028 + eta))
        )
        p1 = p1_noSpin + p1_eqSpin + p1_uneqSpin

        p2_noSpin = (
            3980.7
            + 0.956703 / eta
            - 6202.38 * eta
            + 29218.1 * eta2
            + 24484.2 * eta3
            - 807629.0 * eta4
            + 2.86393e6 * eta5
            - 3.26362e6 * eta6
        )
        p2_eqSpin = (
            (1.92692 - 226.825 * eta + 75.246 * eta2 + 1291.56 * eta3) * S
            + (15.3287 - 99.1551 * eta + 608.328 * eta2 - 2402.94 * eta3) * S2
            + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * S3
            + eta * (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        ) / eta
        p2_uneqSpin = (
            132.244
            * delta
            * eta
            * (chi1L * (2.5769789177580837 - eta) + chi2L * (-2.5769789177580837 + eta))
        )
        p2 = p2_noSpin + p2_eqSpin + p2_uneqSpin

        p3_noSpin = (
            3416.57
            + 2308.63 * eta
            - 84042.9 * eta2
            + 1.01936e6 * eta3
            - 6.0644e6 * eta4
            + 1.76399e7 * eta5
            - 2.0065e7 * eta6
        )
        p3_eqSpin = (
            (24.6295 - 282.354 * eta - 2582.55 * eta2 + 12750.0 * eta3) * S
            + (433.675 - 8775.86 * eta + 56407.8 * eta2 - 114798.0 * eta3) * S2
            + (559.705 - 10627.4 * eta + 61581.0 * eta2 - 114029.0 * eta3) * S3
            + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        )
        p3_uneqSpin = 63.9466 * dchi * delta * eta2
        p3 = p3_noSpin + p3_eqSpin + p3_uneqSpin

        p4_noSpin = (
            3307.49
            - 476.909 * eta
            - 5980.37 * eta2
            + 127610.0 * eta3
            - 919108.0 * eta4
            + 2.86393e6 * eta5
            - 3.26362e6 * eta6
        )
        p4_eqSpin = (
            (-5.02553 - 282.354 * eta + 1291.56 * eta2) * S
            + (-43.8823 + 740.123 * eta - 2402.94 * eta2) * S2
            + (43.8812 - 370.362 * eta + 294.289 * eta2) * S3
            + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        )
        p4_uneqSpin = -132.244 * dchi * delta * eta2
        p4 = p4_noSpin + p4_eqSpin + p4_uneqSpin

        eta7 = eta6 * eta
        p5_noSpin = (
            3259.03
            - 3967.58 * eta
            + 111203.0 * eta2
            - 1.81883e6 * eta3
            + 1.73811e7 * eta4
            - 9.56988e7 * eta5
            + 2.75056e8 * eta6
            - 3.15866e8 * eta7
        )
        p5_eqSpin = (
            (19.7509 - 1104.53 * eta + 3810.18 * eta2) * S
            + (-230.07 + 2314.51 * eta - 5944.49 * eta2) * S2
            + (-201.633 + 2183.43 * eta - 6233.99 * eta2) * S3
            + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        )
        p5_uneqSpin = 112.714 * dchi * delta * eta2
        p5 = p5_noSpin + p5_eqSpin + p5_uneqSpin

        p6_noSpin = p5_noSpin  # p6 = p5 for 32 mode (same fit)
        p6_eqSpin = p5_eqSpin
        p6_uneqSpin = p5_uneqSpin
        p6 = p5  # identical fit

    else:  # modeInt == 3, 44
        p1_noSpin = (
            4349.66
            + 4.34125 / eta
            - 8202.33 * eta
            + 5534.1 * eta2
            + 536500.0 * eta3
            - 4.33197e6 * eta4
            + 1.37792e7 * eta5
            - 1.60802e7 * eta6
        )
        p1_eqSpin = (
            (12.0704 - 528.098 * eta + 1822.91 * eta2 - 9349.73 * eta3 + 17900.9 * eta4)
            * S
            + (
                10.4092
                + 253.334 * eta
                - 5452.04 * eta2
                + 35416.6 * eta3
                - 71523.0 * eta4
            )
            * S2
            + eta * (492.603 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3
        ) / eta
        p1_uneqSpin = (
            -262.143
            * delta
            * eta
            * (chi1L * (-3.0782778864970646 - eta) + chi2L * (3.0782778864970646 + eta))
        )
        p1 = p1_noSpin + p1_eqSpin + p1_uneqSpin

        p2_noSpin = (
            3804.19
            + 0.66144 / eta
            - 2421.77 * eta
            - 33475.8 * eta2
            + 665951.0 * eta3
            - 4.50145e6 * eta4
            + 1.37792e7 * eta5
            - 1.60802e7 * eta6
        )
        p2_eqSpin = (
            (5.83038 - 172.047 * eta + 926.576 * eta2 - 7676.87 * eta3 + 17900.9 * eta4)
            * S
            + (
                6.17601
                + 253.334 * eta
                - 5672.02 * eta2
                + 35722.1 * eta3
                - 71523.0 * eta4
            )
            * S2
            + eta * (492.603 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3
        ) / eta
        p2_uneqSpin = (
            -262.143
            * delta
            * eta
            * (chi1L * (-1.0543062374352932 - eta) + chi2L * (1.0543062374352932 + eta))
        )
        p2 = p2_noSpin + p2_eqSpin + p2_uneqSpin

        p3_noSpin = (
            3308.97
            + 2353.58 * eta
            - 66340.1 * eta2
            + 777272.0 * eta3
            - 4.64438e6 * eta4
            + 1.37792e7 * eta5
            - 1.60802e7 * eta6
        )
        p3_eqSpin = (
            (-21.5697 + 926.576 * eta - 7989.26 * eta2 + 17900.9 * eta3) * S
            + (353.539 - 6403.24 * eta + 37599.5 * eta2 - 71523.0 * eta3) * S2
            + (492.603 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3
        )
        p3_uneqSpin = 262.143 * dchi * delta * eta2
        p3 = p3_noSpin + p3_eqSpin + p3_uneqSpin

        p4_noSpin = (
            3245.63
            - 928.56 * eta
            + 8463.89 * eta2
            - 17422.6 * eta3
            - 165169.0 * eta4
            + 908279.0 * eta5
            - 1.31138e6 * eta6
        )
        p4_eqSpin = (
            (32.506 - 590.293 * eta + 3536.61 * eta2 - 6758.52 * eta3) * S
            + (-25.7716 + 738.141 * eta - 4867.87 * eta2 + 9129.45 * eta3) * S2
            + (-15.7439 + 620.695 * eta - 4679.24 * eta2 + 9582.58 * eta3) * S3
        )
        p4_uneqSpin = 87.0832 * dchi * delta * eta2
        p4 = p4_noSpin + p4_eqSpin + p4_uneqSpin

        eta7 = eta6 * eta
        p5_noSpin = (
            3108.38
            + 3722.46 * eta
            - 119588.0 * eta2
            + 1.92148e6 * eta3
            - 1.69796e7 * eta4
            + 8.39194e7 * eta5
            - 2.17143e8 * eta6
            + 2.28297e8 * eta7
        )
        p5_eqSpin = (
            (118.319 - 529.854 * eta) * eta * S
            + (21.0314 - 240.648 * eta + 516.333 * eta2) * S2
            + (20.3384 - 356.241 * eta + 999.417 * eta2) * S3
        )
        p5_uneqSpin = 97.1364 * dchi * delta * eta2
        p5 = p5_noSpin + p5_eqSpin + p5_uneqSpin

        p6_noSpin = (
            3096.03
            + 986.752 * eta
            - 20371.1 * eta2
            + 220332.0 * eta3
            - 1.31523e6 * eta4
            + 4.29193e6 * eta5
            - 6.01179e6 * eta6
        )
        p6_eqSpin = (
            (-9.96292 - 118.526 * eta + 2255.76 * eta2 - 6758.52 * eta3) * S
            + (-14.4869 + 370.039 * eta - 3605.8 * eta2 + 9129.45 * eta3) * S2
            + (17.0209 + 70.1931 * eta - 3070.08 * eta2 + 9582.58 * eta3) * S3
        )
        p6_uneqSpin = 23.0759 * dchi * delta * eta2
        p6 = p6_noSpin + p6_eqSpin + p6_uneqSpin

    return jnp.array([p1, p2, p3, p4, p5, p6])


def _xhm_inter_phase_ansatz_int(
    Mf: float | Array,
    c0: float | Array,
    c1: float | Array,
    c2: float | Array,
    c4: float | Array,
    cL: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> Array:
    """
    Integral of intermediate phase ansatz (non-32 modes):
      phi = c0*f + c1*log(f) - c2/f - c4/(3*f^3) + cL*atan((f-fRD)/fDA)

    Source: IMRPhenomXHM_Inter_Phase_AnsatzInt in LALSimIMRPhenomXHM_intermediate.c.
    """
    invf = 1.0 / Mf
    return (
        c0 * Mf
        + c1 * jnp.log(Mf)
        - c2 * invf
        - (c4 / 3.0) * invf**3
        + cL * jnp.arctan((Mf - fRD) / fDA)
    )


def _xhm_inter_phase_ansatz_deriv(
    Mf: float | Array,
    c0: float | Array,
    c1: float | Array,
    c2: float | Array,
    c4: float | Array,
    cL: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> float | Array:
    """
    Derivative of intermediate phase ansatz (non-32 modes):
      dphi = c0 + c1/f + c2/f^2 + c4/f^4 + cL*fDA/(fDA^2+(f-fRD)^2)

    Source: IMRPhenomXHM_Inter_Phase_Ansatz in LALSimIMRPhenomXHM_intermediate.c.
    """
    invf = 1.0 / Mf
    return (
        c0
        + c1 * invf
        + c2 * invf**2
        + c4 * invf**4
        + cL * fDA / (fDA**2 + (Mf - fRD) ** 2)
    )


# ---------------------------------------------------------------------------
# Section 6: Ringdown phase fits
# ---------------------------------------------------------------------------


def _xhm_rd_phase_alpha2_22fit(
    eta: float, STotR: float, dchi: float, delta: float
) -> float:
    """
    22-mode ringdown phase alpha2 fit (rescaled to lm inside coefficient solver).

    Source: IMRPhenomXHM_RD_Phase_22_alpha2 in LALSimIMRPhenomXHM_ringdown.c (122019).
    Uses STotR as spin parameter.
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    noSpin = (
        0.2088669311744758
        - 0.37138987533788487 * eta
        + 6.510807976353186 * eta2
        - 31.330215053905395 * eta3
        + 55.45508989446867 * eta4
    )
    eqSpin = (
        (
            0.2393965714370633
            + 1.6966740823756759 * eta
            - 16.874355161681766 * eta2
            + 38.61300158832203 * eta3
        )
        * S
    ) / (1.0 - 0.633218538432246 * S)
    uneqSpin = (
        dchi
        * (0.9088578269496244 * eta**2.5 + 15.619592332008951 * dchi * eta**3.5)
        * delta
    )
    return noSpin + eqSpin + uneqSpin


def _xhm_rd_phase_alphaL_22fit(
    eta: float, STotR: float, dchi: float, delta: float, chi1L: float, chi2L: float
) -> float:
    """
    22-mode ringdown phase alphaL fit (rescaled to lm inside coefficient solver).

    Source: IMRPhenomXHM_RD_Phase_22_alphaL in LALSimIMRPhenomXHM_ringdown.c (122019).
    Uses STotR as spin parameter.
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S2 = S * S
    noSpin = eta * (
        -1.1926122248825484
        + 2.5400257699690143 * eta
        - 16.504334734464244 * eta2
        + 27.623649807617376 * eta3
    )
    eqSpin = (
        eta3 * S * (35.803988443700824 + 9.700178927988006 * S - 77.2346297158916 * S2)
        + eta
        * S
        * (0.1034526554654983 - 0.21477847929548569 * S - 0.06417449517826644 * S2)
        + eta2
        * S
        * (-4.7282481007397825 + 0.8743576195364632 * S + 8.170616575493503 * S2)
        + eta4
        * S
        * (-72.50310678862684 - 39.83460092417137 * S + 180.8345521274853 * S2)
    )
    uneqSpin = (
        -0.7428134042821221 * chi1L * eta**3.5
        + 0.7428134042821221 * chi2L * eta**3.5
        + 17.588573345324154 * chi1L**2 * eta**4.5
        - 35.17714669064831 * chi1L * chi2L * eta**4.5
        + 17.588573345324154 * chi2L**2 * eta**4.5
    ) * delta
    return noSpin + eqSpin + uneqSpin


def _xhm_rd_phase_ansatz_int(
    Mf: float | Array,
    alpha0: float | Array,
    alpha2: float | Array,
    alphaL: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> Array:
    """
    Integral of ringdown phase ansatz (alpha0 is effectively C1RD from continuity):
      phi = alpha0*f - fRD^2*alpha2/f + alphaL*atan((f-fRD)/fDA)

    Source: IMRPhenomXHM_RD_Phase_AnsatzInt in LALSimIMRPhenomXHM_ringdown.c.
    """
    return alpha0 * Mf - fRD**2 * alpha2 / Mf + alphaL * jnp.arctan((Mf - fRD) / fDA)


def _xhm_rd_phase_ansatz_deriv(
    Mf: float | Array,
    alpha0: float | Array,
    alpha2: float | Array,
    alphaL: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> float | Array:
    """
    Derivative of ringdown phase ansatz:
      dphi = alpha0 + fRD^2*alpha2/f^2 + alphaL*fDA/(fDA^2+(f-fRD)^2)

    Source: IMRPhenomXHM_RD_Phase_Ansatz in LALSimIMRPhenomXHM_ringdown.c.
    """
    return alpha0 + fRD**2 * alpha2 / Mf**2 + alphaL * fDA / (fDA**2 + (Mf - fRD) ** 2)


# ---------------------------------------------------------------------------
# Section 6b: Spheroidal mixing helpers (for (3,2) mode)
# ---------------------------------------------------------------------------


def _xhm_inter_phase_ansatz_int_6(
    Mf: float | Array,
    c0: float | Array,
    cL: float | Array,
    c1: float | Array,
    c2: float | Array,
    c4: float | Array,
    c3: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> Array:
    """6-coefficient intermediate phase integral (used for 32 mode with mixing).
    Ansatz: c0 + cL*L + c1/f + c2/f^2 + c4/f^4 + c3/f^3  (derivative)
    Integral: c0*f + cL*atan(...) + c1*log(f) - c2/f - c4/(3f^3) - c3/(2f^2)
    """
    return (
        c0 * Mf
        + cL * jnp.arctan((Mf - fRD) / fDA)
        + c1 * jnp.log(Mf)
        - c2 / Mf
        - (1.0 / 3.0) * c4 / Mf**3
        - 0.5 * c3 / Mf**2
    )


def _xhm_inter_phase_ansatz_deriv_6(
    Mf: float | Array,
    c0: float | Array,
    cL: float | Array,
    c1: float | Array,
    c2: float | Array,
    c4: float | Array,
    c3: float | Array,
    fRD: float | Array,
    fDA: float | Array,
) -> float | Array:
    """6-coefficient intermediate phase derivative (used for 32 mode with mixing)."""
    return (
        c0
        + cL * fDA / (fDA**2 + (Mf - fRD) ** 2)
        + c1 / Mf
        + c2 / Mf**2
        + c4 / Mf**4
        + c3 / Mf**3
    )


def _xhm_rd_phase_spheroidal_int(
    Mf: float | Array,
    alpha0_S: float | Array,
    alphaL_S: float | Array,
    alpha2_S: float | Array,
    alpha4_S: float | Array,
    phi0_S: float | Array,
    fRING: float | Array,
    fDAMP: float | Array,
) -> Array:
    """Spheroidal RD phase integral (version 122019).
    phi(f) = phi0_S + alpha0_S*f - alpha2_S/f - alpha4_S/(3*f^3) + alphaL_S*atan(...)
    Source: IMRPhenomXHM_RD_Phase_AnsatzInt MixingOn=1, LALSimIMRPhenomXHM_ringdown.c.
    """
    return (
        phi0_S
        + alpha0_S * Mf
        - alpha2_S / Mf
        - (1.0 / 3.0) * alpha4_S / Mf**3
        + alphaL_S * jnp.arctan((Mf - fRING) / fDAMP)
    )


def _xhm_rd_phase_spheroidal_deriv(
    Mf: float,
    alpha0_S: float,
    alphaL_S: float,
    alpha2_S: float,
    alpha4_S: float,
    fRING: float,
    fDAMP: float,
) -> float:
    """Spheroidal RD phase derivative (version 122019).
    dphi = alpha0_S + alpha2_S/f^2 + alpha4_S/f^4 + alphaL_S*Lorentzian
    Source: IMRPhenomXHM_RD_Phase_Ansatz MixingOn=1, LALSimIMRPhenomXHM_ringdown.c.
    """
    return (
        alpha0_S
        + alpha2_S / Mf**2
        + alpha4_S / Mf**4
        + alphaL_S * fDAMP / (fDAMP**2 + (Mf - fRING) ** 2)
    )


def _xhm_rd_phase_32_collocpts(
    eta: float, STotR: float, dchi: float, delta: float, chi1L: float, chi2L: float
) -> Array:
    """4 ringdown phase collocpt derivative values for the 32 mode (version 122019).
    Source: IMRPhenomXHM_RD_Phase_32_p1..p4 in LALSimIMRPhenomXHM_ringdown.c.
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    S5 = S4 * S

    p1_noSpin = (
        3169.372056189274
        + 426.8372805022653 * eta
        - 12569.748101922158 * eta2
        + 149846.7281073725 * eta3
        - 817182.2896823225 * eta4
        + 1.5674053633767858e6 * eta5
    )
    p1_eqSpin = (
        (
            19.23408352151287
            - 1762.6573670619173 * eta
            + 7855.316419853637 * eta2
            - 3785.49764771212 * eta3
        )
        * S
        + (
            -42.88446003698396
            + 336.8340966473415 * eta
            - 5615.908682338113 * eta2
            + 20497.5021807654 * eta3
        )
        * S2
        + (
            13.918237996338371
            + 10145.53174542332 * eta
            - 91664.12621864353 * eta2
            + 201204.5096556517 * eta3
        )
        * S3
        + (
            -24.72321125342808
            - 4901.068176970293 * eta
            + 53893.9479532688 * eta2
            - 139322.02687945773 * eta3
        )
        * S4
        + (
            -61.01931672442576
            - 16556.65370439302 * eta
            + 162941.8009556697 * eta2
            - 384336.57477596396 * eta3
        )
        * S5
    )
    p1_uneqSpin = (
        dchi
        * jnp.sqrt(1.0 - 4.0 * eta)
        * eta2
        * (
            641.2473192044652
            - 1600.240100295189 * chi1L * eta
            + 1600.240100295189 * chi2L * eta
            + 13275.623692212472 * eta * S
        )
    )
    p1 = p1_noSpin + p1_eqSpin + p1_uneqSpin

    p2_noSpin = (
        3131.0260952676376
        + 206.09687819102305 * eta
        - 2636.4344627081873 * eta2
        + 7475.062269742079 * eta3
    )
    p2_eqSpin = (
        (
            49.90874152040307
            - 691.9815135740145 * eta
            - 434.60154548208334 * eta2
            + 10514.68111669422 * eta3
        )
        * S
        + (
            97.3078084654917
            - 3458.2579971189534 * eta
            + 26748.805404989867 * eta2
            - 56142.13736008524 * eta3
        )
        * S2
        + (
            -132.49105074500454
            + 429.0787542102207 * eta
            + 7269.262546204149 * eta2
            - 27654.067482558712 * eta3
        )
        * S3
        + (
            -227.8023564332453
            + 5119.138772157134 * eta
            - 34444.2579678986 * eta2
            + 69666.01833764123 * eta3
        )
        * S4
    )
    p2_uneqSpin = 477.51566939885424 * dchi * jnp.sqrt(1.0 - 4.0 * eta) * eta2
    p2 = p2_noSpin + p2_eqSpin + p2_uneqSpin

    p3_noSpin = (
        3082.803556599222
        + 76.94679795837645 * eta
        - 586.2469821978381 * eta2
        + 977.6115755788503 * eta3
    )
    p3_eqSpin = (
        (
            45.08944710349874
            - 807.7353772747749 * eta
            + 1775.4343704616288 * eta2
            + 2472.6476419567534 * eta3
        )
        * S
        + (
            95.57355060136699
            - 2224.9613131172046 * eta
            + 13821.251641893134 * eta2
            - 25583.314298758105 * eta3
        )
        * S2
        + (
            -144.96370424517866
            + 2268.4693587493093 * eta
            - 10971.864789147161 * eta2
            + 16259.911572457446 * eta3
        )
        * S3
        + (
            -227.8023564332453
            + 5119.138772157134 * eta
            - 34444.2579678986 * eta2
            + 69666.01833764123 * eta3
        )
        * S4
    )
    p3_uneqSpin = 378.2359918274837 * dchi * jnp.sqrt(1.0 - 4.0 * eta) * eta2
    p3 = p3_noSpin + p3_eqSpin + p3_uneqSpin

    p4_noSpin = 3077.0657367004565 + 64.99844502520415 * eta - 357.38692756785395 * eta2
    p4_eqSpin = (
        (
            34.793450080444714
            - 986.7751755509875 * eta
            + 5700.682624203565 * eta2
            - 9490.641676924794 * eta3
        )
        * S
        + (
            57.38106384558743
            - 1644.6690499868596 * eta
            + 11008.881935880598 * eta2
            - 19906.416384606226 * eta3
        )
        * S2
        + (
            -126.02362949830213
            + 3169.3397351803583 * eta
            - 26766.730897942085 * eta2
            + 62863.79877094988 * eta3
        )
        * S3
        + (
            -169.30909412804587
            + 4900.706039920717 * eta
            - 41414.05689348732 * eta2
            + 95314.99988114933 * eta3
        )
        * S4
    )
    p4_uneqSpin = 390.5443469721231 * dchi * jnp.sqrt(1.0 - 4.0 * eta) * eta2
    p4 = p4_noSpin + p4_eqSpin + p4_uneqSpin

    return jnp.array([p1, p2, p3, p4])


def _xhm_rd_phase_32_spheroidal_time_shift(
    eta: float, STotR: float, dchi: float, delta: float
) -> Array:
    """Time shift fit for 32-mode spheroidal phase (version 122019).
    Source: IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift, LALSimIMRPhenomXHM_ringdown.c.
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    noSpin = (
        11.851438981981772
        + 167.95086712701223 * eta
        - 4565.033758777737 * eta2
        + 61559.132976189896 * eta3
        - 364129.24735853914 * eta4
        + 739270.8814129328 * eta5
    )
    eqSpin = (
        (
            9.506768471271634
            + 434.31707030999445 * eta
            - 8046.364492927503 * eta2
            + 26929.677144312944 * eta3
        )
        * S
        + (
            -5.949655484033632
            - 307.67253970367034 * eta
            + 1334.1062451631644 * eta2
            + 3575.347142399199 * eta3
        )
        * S2
        + (
            3.4881615575084797
            - 2244.4613237912527 * eta
            + 24145.932943269272 * eta2
            - 60929.87465551446 * eta3
        )
        * S3
        + (
            15.585154698977842
            - 2292.778112523392 * eta
            + 24793.809334683185 * eta2
            - 65993.84497923202 * eta3
        )
        * S4
    )
    uneqSpin = 465.7904934097202 * dchi * jnp.sqrt(1.0 - 4.0 * eta) * eta2
    return noSpin + eqSpin + uneqSpin


def _xhm_rd_phase_32_spheroidal_phase_shift(
    eta: float, STotR: float, dchi: float, delta: float, chi1L: float, chi2L: float
) -> Array:
    """Phase shift fit for 32-mode spheroidal phase (version 122019).
    Source: IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift, LALSimIMRPhenomXHM_ringdown.c.
    """
    S = STotR
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    noSpin = (
        -1.3328895897490733
        - 22.209549522908667 * eta
        + 1056.2426481245027 * eta2
        - 21256.376324666326 * eta3
        + 246313.12887984765 * eta4
        - 1.6312968467540336e6 * eta5
        + 5.614617173188322e6 * eta6
        - 7.612233821752137e6 * eta7
    )
    eqSpin = (
        S
        * (
            -1.622727240110213
            + 0.9960210841611344 * S
            - 1.1239505323267036 * S2
            - 1.9586085340429995 * S3
            + eta2
            * (
                196.7055281997748
                + 135.25216875394943 * S
                + 1086.7504825459278 * S2
                + 546.6246807461155 * S3
                - 312.1010566468068 * S4
            )
            + 0.7638287749489343 * S4
            + eta
            * (
                -47.475568056234245
                - 35.074072557604445 * S
                - 97.16014978329918 * S2
                - 34.498125910065156 * S3
                + 24.02858084544326 * S4
            )
            + eta3
            * (
                62.632493533037625
                - 22.59781899512552 * S
                - 2683.947280170815 * S2
                - 1493.177074873678 * S3
                + 805.0266029288334 * S4
            )
        )
        / (-2.950271397057221 + S)
    )
    uneqSpin = (
        jnp.sqrt(1.0 - 4.0 * eta)
        * (
            chi2L * eta**2.5 * (88.56162028006072 - 30.01812659282717 * S)
            + chi2L * eta2 * (43.126266433486435 - 14.617728550838805 * S)
            + chi1L * eta2 * (-43.126266433486435 + 14.617728550838805 * S)
            + chi1L * eta**2.5 * (-88.56162028006072 + 30.01812659282717 * S)
        )
        / (-2.950271397057221 + S)
    )
    return noSpin + eqSpin + uneqSpin


def _xhm_qnm_mu_l3m2(afinal: float) -> tuple:
    """QNM mixing coefficients mu322 and mu323 for the 32 spheroidal mode.
    Returns (mu322, mu323) as complex numbers (with the minus sign from LAL).
    Source: evaluate_QNMfit_{re,im}_l3m2lp{2,3} in LALSimIMRPhenomXHM_qnm.c;
    negated per lines 70-71 of LALSimIMRPhenomXHM_internals.c.
    """
    x = afinal
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x3 * x2

    re322 = (
        x
        * (
            0.47513455283841244
            - 0.9016636384605536 * x
            + 0.3844811236426182 * x2
            + 0.0855565148647794 * x3
            - 0.03620067426672167 * x4
            - 0.006557249133752502 * x5
        )
        / (-6.76894063440646 + 15.170831931186493 * x - 9.406169787571082 * x2 + x4)
    )
    x6 = x3 * x3
    im322 = (
        x
        * (
            -2.8704762147145533
            + 4.436434016918535 * x
            - 1.0115343326360486 * x2
            - 0.08965314412106505 * x3
            - 0.4236810894599512 * x4
            - 0.041787576033810676 * x5
        )
        / (
            -171.80908957903395
            + 272.362882450877 * x
            - 76.68544453077854 * x2
            - 25.14197656531123 * x4
            + x6
        )
    )
    mu322 = -(re322 + 1j * im322)

    re323 = (
        1.0
        - 2.107852425643677 * x
        + 1.1906393634562715 * x2
        + 0.02244848864087732 * x3
        - 0.09593447799423722 * x4
        - 0.0021343381708933025 * x5
        - 0.005319515989331159 * x6
    ) / (
        1.0
        - 2.1078515887706324 * x
        + 1.2043484690080966 * x2
        - 0.08910191596778137 * x4
        - 0.005471749827809503 * x6
    )
    im323 = (
        x
        * (
            12.45701482868677
            - 29.398484595717147 * x
            + 18.26221675782779 * x2
            + 1.9308599142669403 * x3
            - 3.159763242921214 * x4
            - 0.0910871567367674 * x5
        )
        / (
            345.52914639836257
            - 815.4349339779621 * x
            + 538.3888932415709 * x2
            - 69.3840921447381 * x4
            + x6
        )
    )
    mu323 = -(re323 + 1j * im323)

    return mu322, mu323


def _xhm_get_spheroidal_coeffs(
    pWFHM: "XHMWaveformStruct",
    pWF22: dict,
    alambda32: float,
    lambda32: float,
    sigma32: float,
    t0: float,
    phifRef: float,
) -> tuple:
    """Compute spheroidal RD phase coefficients for the 32 mode (version 122019).
    Port of GetSpheroidalCoefficients in LALSimIMRPhenomXHM_internals.c.
    Returns (alpha0_S, alphaL_S, alpha2_S, alpha4_S, phi0_S).
    """
    eta = pWF22["eta"]
    STotR = pWF22["STotR"]
    dchi = pWF22["dchi"]
    delta = pWF22["delta"]
    chi1L = pWF22["chi1L"]
    chi2L = pWF22["chi2L"]
    M_s = pWF22["M_s"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    chip = pWF22.get("chip", 0.0)
    fRING32 = pWFHM.fRING
    fDAMP32 = pWFHM.fDAMP
    fRING22 = pWF22["fRING22"]
    fDAMP22 = pWF22["fDAMP22"]

    # Collocation freqs: [fRING22, fRING32-1.5*fDAMP32, fRING32-0.5*fDAMP32, fRING32+0.5*fDAMP32]
    f_collocpts = jnp.array(
        [
            fRING22,
            fRING32 - 1.5 * fDAMP32,
            fRING32 - 0.5 * fDAMP32,
            fRING32 + 0.5 * fDAMP32,
        ]
    )
    vals = _xhm_rd_phase_32_collocpts(eta, STotR, dchi, delta, chi1L, chi2L)

    # 4x4 matrix: derivative ansatz basis {1, L, 1/f^2, 1/f^4}
    def _mrow(f):
        ffm2 = 1.0 / (f * f)
        L = fDAMP32 / (fDAMP32**2 + (f - fRING32) ** 2)
        return jnp.array([1.0, L, ffm2, ffm2 * ffm2])

    A = jnp.stack([_mrow(f_collocpts[i]) for i in range(4)])
    sol = jnp.linalg.solve(A, vals)
    alpha0_S = sol[0]
    alphaL_S = sol[1]
    alpha2_S = sol[2]
    alpha4_S = sol[3]

    # Time-shift: frefTS = fRING22 + fDAMP22
    frefTS = fRING22 + fDAMP22
    tshift = _xhm_rd_phase_32_spheroidal_time_shift(eta, STotR, dchi, delta)
    dphi22_frefTS = (
        jax.grad(lambda Mf: IMRPhenomXAS_Phase(Mf / M_s, theta, phase_coeffs, chip))(frefTS)
        + t0
    )
    dphi32_frefTS = _xhm_rd_phase_spheroidal_deriv(
        frefTS, alpha0_S, alphaL_S, alpha2_S, alpha4_S, fRING32, fDAMP32
    )
    alpha0_S = alpha0_S + dphi22_frefTS + tshift - dphi32_frefTS

    # Phase-shift: frefPS = fRING22
    frefPS = fRING22
    phi22_frefPS = (
        IMRPhenomXAS_Phase(frefPS / M_s, theta, phase_coeffs, chip) + t0 * frefPS + phifRef
    )
    phishift = _xhm_rd_phase_32_spheroidal_phase_shift(
        eta, STotR, dchi, delta, chi1L, chi2L
    )
    phi32_frefPS = _xhm_rd_phase_spheroidal_int(
        frefPS, alpha0_S, alphaL_S, alpha2_S, alpha4_S, 0.0, fRING32, fDAMP32
    )
    phi0_S = phi22_frefPS - phi32_frefPS + phishift

    return alpha0_S, alphaL_S, alpha2_S, alpha4_S, phi0_S


def _xhm_s2s_complex(
    Mf: float,
    alambda32: float,
    lambda32: float,
    sigma32: float,
    fRING32: float,
    fDAMP32: float,
    alpha0_S: float,
    alphaL_S: float,
    alpha2_S: float,
    alpha4_S: float,
    phi0_S: float,
    mu322: complex,
    mu323: complex,
    theta: Array,
    phase_coeffs: Array,
    M_s: float,
    t0: float,
    phifRef: float,
    amp_coeffs_22: Array,
    ampNorm: float,
    fRDAux32: float,
    fAmpRDfalloff32: float,
    rdaux_poly_c: Array,
    rdaux_falloff_amp: float,
    rdaux_falloff_slope: float | Array,
    chip: float = 0.0,
) -> complex:
    """SpheroidalToSpherical for version 122022, mode 32 (RingdownAmpVersion=1).

    amplm uses 3 regions (LAL case 1 with nCoefficientsRDAux=4):
      f < fRDAux:           cubic polynomial (rdaux_poly_c)
      fRDAux <= f < ffall:  Lorentzian (no sigma in numerator, no f^(-1/12))
      f >= ffall:           exponential falloff
    wf22R = amp22 * ampNorm * Mf^(-7/6) * exp(i*phi22)  (v1: scaled)
    Returns |S2S| in full-strain units.
    """
    amp22, _ = get_mergerringdown_Amp(Mf, theta, amp_coeffs_22, chip=chip)
    phi22 = IMRPhenomXAS_Phase(Mf / M_s, theta, phase_coeffs, chip) + t0 * Mf + phifRef
    wf22R = amp22 * ampNorm * Mf ** (-7.0 / 6.0) * jnp.exp(1j * phi22)

    # amplm: 3-region spheroidal amplitude
    amplm_lor = _xhm_rd_rescaled_v1(Mf, alambda32, lambda32, sigma32, fRING32, fDAMP32)
    amplm_poly = (
        rdaux_poly_c[0]
        + rdaux_poly_c[1] * Mf
        + rdaux_poly_c[2] * Mf**2
        + rdaux_poly_c[3] * Mf**3
    )
    amplm_fall = rdaux_falloff_amp * jnp.exp(
        -rdaux_falloff_slope * (Mf - fAmpRDfalloff32)
    )
    amplm = jnp.where(
        Mf < fRDAux32,
        amplm_poly,
        jnp.where(Mf < fAmpRDfalloff32, amplm_lor, amplm_fall),
    )

    philm = _xhm_rd_phase_spheroidal_int(
        Mf, alpha0_S, alphaL_S, alpha2_S, alpha4_S, phi0_S, fRING32, fDAMP32
    )

    return jnp.conj(mu322) * wf22R + jnp.conj(mu323) * amplm * jnp.exp(1j * philm)


def _compute_32_hlm(
    freqs_geom: Array,
    pWFHM: "XHMWaveformStruct",
    pWF22: dict,
    t0: float,
    phifRef: float,
    phi0: float | Array,
) -> Array:
    """Evaluate complex h_{3,2}(f) with spheroidal mode mixing (version 122019).

    Three-region piecewise:
      Inspiral (Mf < fMatchIN):   standard XHM inspiral amplitude + phase
      Intermediate (fMatchIN <= Mf < fMatchIM): polynomial amplitude + 6-coeff phase
      Ringdown (Mf >= fMatchIM):  |S2S|*f^(-7/6)*ampNorm + arg(S2S) + C1RD*f + CRD

    Returns hlm without the external exp(i*(emm/2*phifRef + emm*phi0)) factor
    (that is applied by the caller in XLALSimIMRPhenomXHMGethlmModes).
    """
    eta = pWF22["eta"]
    STotR = pWF22["STotR"]
    dchi = pWF22["dchi"]
    delta = pWF22["delta"]
    chi1L = pWF22["chi1L"]
    chi2L = pWF22["chi2L"]
    fMECO = pWF22["fMECO"]
    M_s = pWF22["M_s"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    chip = pWF22.get("chip", 0.0)
    afinal = pWF22["afinal"]
    fRING22 = pWF22["fRING22"]
    fDAMP22 = pWF22["fDAMP22"]

    emm = pWFHM.emm
    fRING32 = pWFHM.fRING
    fDAMP32 = pWFHM.fDAMP
    fMECOlm = pWFHM.fMECOlm
    etaEMR = pWFHM.etaEMR
    m_over_2 = emm * 0.5  # = 1.0 for (3,2)

    amp_coeffs_22 = IMRPhenomX_utils.PhenomX_amp_coeff_table

    # -----------------------------------------------------------------------
    # Step 1: RD amplitude fits (alambda, lambda_, sigma)
    # -----------------------------------------------------------------------
    chiPN = pWF22["chiPN"]
    dchi_half = dchi * 0.5
    alambda32, lambda32, sigma32 = _xhm_rd_amp_fit_coeffs(
        32, eta, STotR, dchi_half, delta, chiPN
    )

    # -----------------------------------------------------------------------
    # Step 2: Spheroidal phase coefficients
    # -----------------------------------------------------------------------
    alpha0_S, alphaL_S, alpha2_S, alpha4_S, phi0_S = _xhm_get_spheroidal_coeffs(
        pWFHM, pWF22, alambda32, lambda32, sigma32, t0, phifRef
    )

    # -----------------------------------------------------------------------
    # Step 3: QNM mixing coefficients
    # -----------------------------------------------------------------------
    mu322, mu323 = _xhm_qnm_mu_l3m2(afinal)

    # Convenience: S2S at a given frequency.
    # Note: fRDAux32, fAmpRDfalloff32, rdaux_poly_c_32, rdaux_falloff_amp32,
    # rdaux_falloff_slope32 are computed in Step 4 below before s2s is first called.
    def s2s(Mf):
        return _xhm_s2s_complex(
            Mf,
            alambda32,
            lambda32,
            sigma32,
            fRING32,
            fDAMP32,
            alpha0_S,
            alphaL_S,
            alpha2_S,
            alpha4_S,
            phi0_S,
            mu322,
            mu323,
            theta,
            phase_coeffs,
            M_s,
            t0,
            phifRef,
            amp_coeffs_22,
            ampNorm,
            fRDAux32,
            fAmpRDfalloff32,
            rdaux_poly_c_32,
            rdaux_falloff_amp32,
            rdaux_falloff_slope32,
            chip,
        )

    # -----------------------------------------------------------------------
    # Step 4: Amplitude match frequencies and intermediate amplitude polynomial
    # -----------------------------------------------------------------------
    fIN = _xhm_fAmpMatchIN(pWFHM, pWF22)
    fIM = _xhm_fAmpMatchIM(pWFHM, pWF22)

    # RDAux polynomial for amplm (spheroidal component of S2S, version 1)
    # LAL: nCoefficientsRDAux=4, fRDAux=fRING32-fDAMP32, fAmpRDfalloff=fRING32+2*fDAMP32
    fRDAux32 = fRING32 - fDAMP32
    fAmpRDfalloff32 = fRING32 + 2.0 * fDAMP32

    def _amplm_lor(f):
        dfr = f - fRING32
        dfd = fDAMP32 * sigma32
        return (
            jnp.abs(alambda32)
            * fDAMP32
            / (jnp.exp(lambda32 * dfr / dfd) * (dfr**2 + dfd**2))
        )

    rdaux1_32, rdaux2_32 = _xhm_rd_amp_rdaux_pts(eta, STotR, dchi_half, delta, chiPN)
    faux_mid32 = 0.5 * (fIM + fRDAux32)
    lor_fRDAux32 = _amplm_lor(fRDAux32)
    dlor_fRDAux32 = jax.grad(_amplm_lor)(fRDAux32)

    # 4x4 system: c0 + c1*f + c2*f^2 + c3*f^3 matches rdaux1 at fIM,
    # rdaux2 at midpoint, Lorentzian value+derivative at fRDAux32
    _A_rdaux = jnp.array(
        [
            [1.0, fIM, fIM**2, fIM**3],
            [1.0, faux_mid32, faux_mid32**2, faux_mid32**3],
            [1.0, fRDAux32, fRDAux32**2, fRDAux32**3],
            [0.0, 1.0, 2.0 * fRDAux32, 3.0 * fRDAux32**2],
        ]
    )
    _b_rdaux = jnp.array(
        [jnp.abs(rdaux1_32), jnp.abs(rdaux2_32), lor_fRDAux32, dlor_fRDAux32]
    )
    rdaux_poly_c_32 = jnp.linalg.solve(_A_rdaux, _b_rdaux)

    # Exponential falloff for f >= fAmpRDfalloff32
    rdaux_falloff_amp32 = _amplm_lor(fAmpRDfalloff32)
    rdaux_falloff_slope32 = jnp.where(
        rdaux_falloff_amp32 > 0,
        -jax.grad(_amplm_lor)(fAmpRDfalloff32) / rdaux_falloff_amp32,
        0.0,
    )

    # Inspiral amplitude fits
    modeTag = pWFHM.modeTag
    chiPN = pWF22["chiPN"]
    ampNorm = pWF22["ampNorm"]
    PNgf = (2.0 / pWFHM.emm) ** (-7.0 / 6.0) * _AMP_PREFACTORS[pWFHM.modeInt]
    pn_coeffs = _xhm_pn_amp_coeffs(modeTag, eta, delta, chi1L, chi2L)
    iv1_raw, iv2_raw, iv3_raw = _xhm_insp_amp_colloc_pts(
        modeTag, eta, chiPN, dchi_half, delta
    )
    # 122022 colloc freq ordering: f1=0.5*fIN, f2=0.75*fIN, f3=fIN
    # Version 123 (3 points), no veto (InspiralAmpVeto=0)
    f1 = 0.5 * fIN
    f2 = 0.75 * fIN
    f3 = fIN
    iv1_raw_abs = jnp.abs(iv1_raw)
    iv2_raw_abs = jnp.abs(iv2_raw)
    iv3_raw_abs = jnp.abs(iv3_raw)

    # Full-strain PN at each colloc freq (InspRescaleFactor=0)
    PNf1_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f1, pn_coeffs)) * ampNorm * f1 ** (-7.0 / 6.0)
    )
    PNf2_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f2, pn_coeffs)) * ampNorm * f2 ** (-7.0 / 6.0)
    )
    PNf3_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f3, pn_coeffs)) * ampNorm * f3 ** (-7.0 / 6.0)
    )

    # PNdominant = ampNorm * (2/emm)^{-7/6}
    PNdominant_32 = ampNorm * (2.0 / pWFHM.emm) ** (-7.0 / 6.0)

    v1_32 = (iv1_raw_abs - PNf1_full) * f1 ** (7.0 / 6.0) / PNdominant_32
    v2_32 = (iv2_raw_abs - PNf2_full) * f2 ** (7.0 / 6.0) / PNdominant_32
    v3_32 = (iv3_raw_abs - PNf3_full) * f3 ** (7.0 / 6.0) / PNdominant_32

    f1_73 = f1 ** (7.0 / 3.0)
    f2_73 = f2 ** (7.0 / 3.0)
    f3_73 = f3 ** (7.0 / 3.0)
    f1_83 = f1 ** (8.0 / 3.0)
    f2_83 = f2 ** (8.0 / 3.0)
    f3_83 = f3 ** (8.0 / 3.0)
    f1_3 = f1**3.0
    f2_3 = f2**3.0
    f3_3 = f3**3.0
    f1_13 = f1 ** (1.0 / 3.0)
    f2_13 = f2 ** (1.0 / 3.0)
    f3_13 = f3 ** (1.0 / 3.0)
    fc_73 = fIN ** (7.0 / 3.0)
    fc_83 = fIN ** (8.0 / 3.0)
    fc_3 = fIN**3.0
    denom3 = f1_73 * (f1_13 - f2_13) * f2_73 * (f1_13 - f3_13) * (f2_13 - f3_13) * f3_73

    c0_LAL_32 = (
        fc_73
        * (
            -(f1_3 * f3_83 * v2_32)
            + f1_83 * f3_3 * v2_32
            + f2_3 * (f3_83 * v1_32 - f1_83 * v3_32)
            + f2_83 * (-(f3_3 * v1_32) + f1_3 * v3_32)
        )
    ) / denom3
    c1_LAL_32 = (
        fc_83
        * (
            f1_3 * f3_73 * v2_32
            - f1_73 * f3_3 * v2_32
            + f2_3 * (-(f3_73 * v1_32) + f1_73 * v3_32)
            + f2_73 * (f3_3 * v1_32 - f1_3 * v3_32)
        )
    ) / denom3
    c2_LAL_32 = (
        fc_3
        * (
            f1_73 * (-f1_13 + f3_13) * f3_73 * v2_32
            + f2_73 * (-(f3_83 * v1_32) + f1_83 * v3_32)
            + f2_83 * (f3_73 * v1_32 - f1_73 * v3_32)
        )
    ) / denom3

    emm_factor_32 = (2.0 / pWFHM.emm) ** (-7.0 / 6.0)
    rho1 = c0_LAL_32 * emm_factor_32
    rho2 = c1_LAL_32 * emm_factor_32
    rho3 = c2_LAL_32 * emm_factor_32

    # Intermediate: 122022 direct polynomial, pattern 211112 (8 constraints)
    # Equispaced: deltaf = (fIM - fIN) / 5, 6 slots
    # Values: insp(fIN)+d, int1..int4, |S2S(fIM)|+d (with mixing RD boundary)
    int1_32, int2_32, int3_32, int4_32 = _xhm_inter_amp_colloc_pts(
        modeTag, eta, STotR, dchi_half, delta, chiPN
    )
    int1_32 = jnp.abs(int1_32)
    int2_32 = jnp.abs(int2_32)
    int3_32 = jnp.abs(int3_32)
    int4_32 = jnp.abs(int4_32)

    df_inter = (fIM - fIN) / 5.0
    f_int1_32 = fIN + df_inter
    f_int2_32 = fIN + 2.0 * df_inter
    f_int3_32 = fIN + 3.0 * df_inter
    f_int4_32 = fIN + 4.0 * df_inter

    # Full-strain inspiral at fIN (value + derivative)
    def insp_amp_32(f):
        v = _xhm_insp_rescaled(f, PNgf, pn_coeffs, rho1, rho2, rho3, fIN)
        return ampNorm * f ** (-7.0 / 6.0) * v

    inspF_IN_32 = insp_amp_32(fIN)
    d_inspF_IN_32 = jax.grad(insp_amp_32)(fIN)

    # Full-strain RD at fIM: |S2S_v1(fIM)| for version 1 (122022)
    # wf22R in S2S_v1 already includes ampNorm*f^(-7/6), so no extra scaling needed.
    def rd_amp_32(f):
        return jnp.abs(s2s(f))

    rdF_IM_32 = rd_amp_32(fIM)
    d_rdF_IM_32 = jax.grad(rd_amp_32)(fIM)

    _AMP_EPS = 1e-15
    inspF_IN_32_s = jnp.where(jnp.abs(inspF_IN_32) < _AMP_EPS, _AMP_EPS, inspF_IN_32)
    rdF_IM_32_s = jnp.where(jnp.abs(rdF_IM_32) < _AMP_EPS, _AMP_EPS, rdF_IM_32)

    # Build 8x8 linear system for direct polynomial A(f)=f^(-7/6)*sum_j c_j*f^j
    nC_32 = 8
    freqs_b_32 = jnp.array(
        [fIN, fIN, f_int1_32, f_int2_32, f_int3_32, f_int4_32, fIM, fIM]
    )
    vals_b_32 = jnp.array(
        [
            inspF_IN_32_s,
            d_inspF_IN_32,
            int1_32,
            int2_32,
            int3_32,
            int4_32,
            rdF_IM_32_s,
            d_rdF_IM_32,
        ]
    )
    use_deriv_32 = [False, True, False, False, False, False, False, True]
    rows_32 = []
    for i, (f_col, is_deriv) in enumerate(zip(freqs_b_32, use_deriv_32)):
        if is_deriv:
            row = jnp.array(
                [(j - 7.0 / 6.0) * f_col ** (j - 13.0 / 6.0) for j in range(nC_32)]
            )
        else:
            row = jnp.array([f_col ** (j - 7.0 / 6.0) for j in range(nC_32)])
        rows_32.append(row)
    A_mat_32 = jnp.stack(rows_32)
    inter_c_32 = jnp.linalg.solve(A_mat_32, vals_b_32)

    # -----------------------------------------------------------------------
    # Step 5: Intermediate phase 6-point fit using finite-diff S2S at fcutRD
    # -----------------------------------------------------------------------
    fMatchIN = fMECOlm
    # For (3,2) mode, phase-matching frequency uses the 22-mode RD (LAL: fEnd = fRING22 - 0.5*fDAMP22)
    fEnd = fRING22 - 0.5 * fDAMP22
    fMatchIM = fEnd

    # 6 collocation frequencies (32 mode uses fEnd as the reference, not fRING32)
    fcut = (1.0 + 0.001 * (0.25 / eta - 1.0)) * m_over_2 * fMECO
    f0 = fcut
    f1_ph = (jnp.sqrt(3.0) * (fcut - fEnd) + 2.0 * (fcut + fEnd)) / 4.0
    f2_ph = (3.0 * fcut + fEnd) / 4.0
    f3_ph = (fcut + fEnd) / 2.0
    # f4 and f5 are both fEnd for the 32 mode (set below via override logic)
    all_freqs = jnp.array([f0, f1_ph, f2_ph, f3_ph, fEnd, fEnd])

    psi4tostrain = XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi)
    DeltaT = -2.0 * PI * (500.0 + psi4tostrain)
    all_vals = (
        _xhm_inter_phase_colloc_pts(2, eta, STotR, dchi, delta, chi1L, chi2L) + DeltaT
    )

    # Finite differences at fcutRD = fMatchIM for the spherical-harmonic phase
    fcutRD = fMatchIM
    fstep = 1e-7
    phases_fd = jnp.array([jnp.angle(s2s(fcutRD + (i - 1) * fstep)) for i in range(3)])
    # Branch to (-2pi, 0]
    phases_fd = jnp.where(phases_fd > 0.0, phases_fd - 2.0 * PI, phases_fd)
    phi0RD = phases_fd[1]
    dphi0RD = 0.5 / fstep * (phases_fd[2] - phases_fd[0])
    d2phi0RD = (phases_fd[2] - 2.0 * phases_fd[1] + phases_fd[0]) / fstep**2

    # Override collocpt 4 with dphi0RD (first deriv at fcutRD) when eta > etaEMR
    freq4 = jnp.where(eta > etaEMR, fcutRD, all_freqs[4])
    val4 = jnp.where(eta > etaEMR, dphi0RD, all_vals[4])
    # Collocpt 5 is always the second deriv at fcutRD
    freq5 = fcutRD
    val5 = d2phi0RD

    freqs_6 = jnp.array(
        [all_freqs[0], all_freqs[1], all_freqs[2], all_freqs[3], freq4, freq5]
    )
    vals_6 = jnp.array([all_vals[0], all_vals[1], all_vals[2], all_vals[3], val4, val5])

    # Build 6x6 matrix: rows 0..4 = first-derivative basis, row 5 = second-derivative basis
    def _ph_row1(f):
        ffm1 = 1.0 / f
        ffm2 = ffm1 * ffm1
        return jnp.array(
            [
                1.0,
                fDAMP32 / (fDAMP32**2 + (f - fRING32) ** 2),
                ffm1,
                ffm2,
                ffm2 * ffm2,
                ffm1 * ffm2,
            ]
        )

    def _ph_row2(f):
        ffm1 = 1.0 / f
        ffm2 = ffm1 * ffm1
        ffm3 = ffm2 * ffm1
        ffm4 = ffm2 * ffm2
        ffm5 = ffm3 * ffm2
        dL = -2.0 * fDAMP32 * (f - fRING32) / (fDAMP32**2 + (f - fRING32) ** 2) ** 2
        return jnp.array([0.0, dL, -ffm2, -2.0 * ffm3, -4.0 * ffm5, -3.0 * ffm4])

    A_ph = jnp.stack([_ph_row1(freqs_6[i]) for i in range(5)] + [_ph_row2(freq5)])
    coeffs_ph = jnp.linalg.solve(A_ph, vals_6)
    c0_ph = coeffs_ph[0]
    cL_ph = coeffs_ph[1]
    c1_ph = coeffs_ph[2]
    c2_ph = coeffs_ph[3]
    c4_ph = coeffs_ph[4]
    c3_ph = coeffs_ph[5]

    # EMR correction: glue intermediate to RD at fcutRD
    inter_dphi_fcutRD = _xhm_inter_phase_ansatz_deriv_6(
        fcutRD, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
    )
    c0_ph = jnp.where(eta <= etaEMR, c0_ph + dphi0RD - inter_dphi_fcutRD, c0_ph)

    # -----------------------------------------------------------------------
    # Step 6: Inspiral continuity (C1INSP, CINSP) and deltaphiLM
    # -----------------------------------------------------------------------
    def ins_phase(Mf_):
        LambdaPN = _xhm_insp_phase_LambdaPN(32, eta)
        return (
            m_over_2 / eta * get_inspiral_phase(2.0 * Mf_ / emm, theta, phase_coeffs)
            + LambdaPN * Mf_
        )

    ins_phi_fIN = ins_phase(fMatchIN)
    ins_dphi_fIN = jax.grad(ins_phase)(fMatchIN)
    inter_phi_fIN = _xhm_inter_phase_ansatz_int_6(
        fMatchIN, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
    )
    inter_dphi_fIN = _xhm_inter_phase_ansatz_deriv_6(
        fMatchIN, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
    )
    C1INSP = inter_dphi_fIN - ins_dphi_fIN
    CINSP = inter_phi_fIN - ins_phi_fIN - C1INSP * fMatchIN

    # deltaphiLM (without phifRef — external phifRef applied by caller)
    falign = jnp.where(eta > etaEMR, 0.6 * m_over_2 * fMECO, m_over_2 * fMECO)
    two_over_m = 1.0 / m_over_2
    phi_22_at_falign = (1.0 / eta) * get_inspiral_phase(
        two_over_m * falign, theta, phase_coeffs
    )
    phi_ins_falign_with_C1 = ins_phase(falign) + C1INSP * falign + CINSP
    deltaphiLM = (
        m_over_2 * phi_22_at_falign
        + t0 * falign
        - 3.0 * PI / 4.0 * (1.0 - m_over_2)
        - phi_ins_falign_with_C1
    ) % (2.0 * PI)

    # -----------------------------------------------------------------------
    # Step 7: Ringdown continuity (C1RD, CRD)
    # -----------------------------------------------------------------------
    inter_phi_fIM = _xhm_inter_phase_ansatz_int_6(
        fMatchIM, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
    )
    inter_dphi_fIM = _xhm_inter_phase_ansatz_deriv_6(
        fMatchIM, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
    )
    C1RD = inter_dphi_fIM - dphi0RD
    CRD = inter_phi_fIM - phi0RD - C1RD * fMatchIM

    # -----------------------------------------------------------------------
    # Step 8: Evaluate hlm at all frequencies
    # -----------------------------------------------------------------------
    def phase_ins(Mf):
        return ins_phase(Mf) + C1INSP * Mf + CINSP + deltaphiLM

    def phase_inter(Mf):
        return (
            _xhm_inter_phase_ansatz_int_6(
                Mf, c0_ph, cL_ph, c1_ph, c2_ph, c4_ph, c3_ph, fRING32, fDAMP32
            )
            + deltaphiLM
        )

    def phase_rd(Mf):
        return jnp.angle(s2s(Mf)) + C1RD * Mf + CRD + deltaphiLM

    def amp_ins(Mf):
        v = _xhm_insp_rescaled(Mf, PNgf, pn_coeffs, rho1, rho2, rho3, fIN)
        return ampNorm * Mf ** (-7.0 / 6.0) * v

    def amp_inter(Mf):
        c = inter_c_32
        poly = sum(c[j] * Mf**j for j in range(nC_32))
        return Mf ** (-7.0 / 6.0) * poly

    def amp_rd(Mf):
        return jnp.abs(s2s(Mf))

    def hlm_at(Mf):
        amp = jnp.where(
            Mf < fIN, amp_ins(Mf), jnp.where(Mf < fIM, amp_inter(Mf), amp_rd(Mf))
        )
        ph = jnp.where(
            Mf < fMatchIN,
            phase_ins(Mf),
            jnp.where(Mf < fMatchIM, phase_inter(Mf), phase_rd(Mf)),
        )
        return amp * jnp.exp(1j * (ph - emm * phi0))

    return jax.vmap(hlm_at)(freqs_geom)


# ---------------------------------------------------------------------------
# Section 7: Phase coefficient computation
# ---------------------------------------------------------------------------


@dataclass
class XHMPhaseCoefficients:
    """
    All phase coefficients for one higher mode.
    Populated by xhm_get_phase_coefficients.
    """

    # Inspiral: deviation from 22-mode rescaling
    ins_c0: float | Array
    ins_c1: float | Array
    ins_c2: float | Array
    ins_c4: float | Array
    ins_C1INSP: float | Array  # continuity shift for dphi at fMatchIN
    ins_CINSP: float | Array  # continuity shift for phi at fMatchIN

    # Intermediate: 5 polynomial coefficients [c0, c1, c2, c4, cL]
    inter_c0: float | Array
    inter_c1: float | Array
    inter_c2: float | Array
    inter_c4: float | Array
    inter_cL: float | Array

    # Ringdown: alpha2, alphaL (alpha0 = C1RD), CRD
    rd_alpha2: float | Array
    rd_alphaL: float | Array
    rd_C1RD: float | Array
    rd_CRD: float | Array

    # Phase normalization
    deltaphiLM: float | Array

    # Boundary frequencies
    fMatchIN: float
    fMatchIM: float
    fRING: float
    fDAMP: float


def xhm_get_phase_coefficients(
    pWFHM: XHMWaveformStruct, pWF22: dict, t0: float
) -> XHMPhaseCoefficients:
    """
    Solve for all phase coefficients of one higher mode (non-32).

    Algorithm (mirrors IMRPhenomXHM_GetPhaseCoefficients in LALSimIMRPhenomXHM_internals.c):
      1. Compute 6 collocation frequencies for intermediate region.
      2. Evaluate p1..p6 fits + DeltaT (= t0 for XHM) to get derivative values.
      3. Compute alpha2, alphaL for ringdown.
      4. Compute phi0RD, dphi0RD (RD ansatz at fMatchIM with alpha0=0).
      5. Select 5 collocation points based on eta/STotR (typical: [0,1,2,3,5]).
      6. Build 5x5 matrix and solve for [c0, cL, c1, c2, c4].
      7. Compute C1INSP/CINSP continuity at fMatchIN.
      8. Compute C1RD/CRD continuity at fMatchIM.
      9. Compute deltaphiLM normalization at falign.

    t0: IMRPhenomX_TimeShift_22 value (DeltaT added to each collocation point derivative).
    Note: 32-mode mixing not implemented (deferred to second pass).
    """
    ell = pWFHM.ell
    emm = pWFHM.emm
    modeInt = pWFHM.modeInt
    eta = pWF22["eta"]
    STotR = pWF22["STotR"]
    dchi = pWF22["dchi"]
    delta = pWF22["delta"]
    chi1L = pWF22["chi1L"]
    chi2L = pWF22["chi2L"]
    fMECO = pWF22["fMECO"]
    M_s = pWF22["M_s"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    chip = pWF22.get("chip", 0.0)

    fRING = pWFHM.fRING
    fDAMP = pWFHM.fDAMP
    fMECOlm = pWFHM.fMECOlm

    # Boundary frequencies
    fMatchIN = fMECOlm  # = fMECO * emm/2
    fMatchIM = fRING - fDAMP

    # -----------------------------------------------------------------------
    # Step 1: Collocation frequencies (6 points in intermediate region)
    # Source: IMRPhenomXHM_Intermediate_CollocPtsFreqs
    # -----------------------------------------------------------------------
    m_over_2 = emm * 0.5
    fcut = (1.0 + 0.001 * (0.25 / eta - 1.0)) * m_over_2 * fMECO
    fring = fRING  # shorthand
    f0 = fcut
    f1 = (jnp.sqrt(3.0) * (fcut - fring) + 2.0 * (fcut + fring)) / 4.0
    f2 = (3.0 * fcut + fring) / 4.0
    f3 = (fcut + fring) / 2.0
    f4 = (fcut + 3.0 * fring) / 4.0
    f5 = (fcut + 7.0 * fring) / 8.0
    all_freqs = jnp.array([f0, f1, f2, f3, f4, f5])

    # -----------------------------------------------------------------------
    # Step 2: Intermediate phase derivative values at all 6 collocation points
    #         + DeltaT (psi4-based time shift, distinct from IMRPhenomX_TimeShift_22).
    # LAL: pWFHM->DeltaT = -2*pi*(500 + XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi))
    # This is NOT the same as IMRPhenomX_TimeShift_22 (which also adds linb - dphi22Ref).
    # -----------------------------------------------------------------------
    psi4tostrain = XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi)
    DeltaT = -2.0 * PI * (500.0 + psi4tostrain)
    all_vals = (
        _xhm_inter_phase_colloc_pts(modeInt, eta, STotR, dchi, delta, chi1L, chi2L)
        + DeltaT
    )

    # LAL applies an extra high-spin 21 safeguard before choosing the 5/6
    # intermediate collocation points: it rewrites the first two derivative
    # values using finite differences of the full 22-mode phase derivative.
    if pWFHM.modeTag == 21:
        two_over_m = 2.0 / emm

        def dphi22_at(Mf_):
            return jax.grad(IMRPhenomXAS_Phase)(Mf_ / M_s, theta, phase_coeffs, chip) / M_s

        insp_vals = jnp.array([dphi22_at(two_over_m * all_freqs[i]) for i in range(3)])
        diff12 = insp_vals[0] - insp_vals[1]
        diff23 = insp_vals[1] - insp_vals[2]
        all_vals_21hs = all_vals.at[1].set(all_vals[2] + diff23)
        all_vals_21hs = all_vals_21hs.at[0].set(all_vals_21hs[1] + diff12)
        all_vals = jnp.where(STotR >= 0.8, all_vals_21hs, all_vals)

    # -----------------------------------------------------------------------
    # Step 3: Ringdown: alpha2, alphaL (with wlm rescaling)
    # -----------------------------------------------------------------------
    alpha2_22 = _xhm_rd_phase_alpha2_22fit(eta, STotR, dchi, delta)
    alphaL_22 = _xhm_rd_phase_alphaL_22fit(eta, STotR, dchi, delta, chi1L, chi2L)

    wlm = 2.0 if (ell == emm) else emm / 3.0
    alpha2 = (wlm / fRING**2) * alpha2_22
    alphaL = (1.0 / eta) * alphaL_22

    # Step 4: phi0RD, dphi0RD at fMatchIM (with alpha0=0)
    phi0RD = _xhm_rd_phase_ansatz_int(fMatchIM, 0.0, alpha2, alphaL, fRING, fDAMP)
    dphi0RD = _xhm_rd_phase_ansatz_deriv(fMatchIM, 0.0, alpha2, alphaL, fRING, fDAMP)

    # -----------------------------------------------------------------------
    # Step 5: Collocation point selection (5 out of 6)
    # Rule: cpoints = [0, 1, i2, i3, 5] where i2, i3 depend on parameters
    # -----------------------------------------------------------------------
    etaEMR = pWFHM.etaEMR
    modeTag = pWFHM.modeTag

    # Determine i2, i3 with JAX-safe branching
    cond_emr = eta < etaEMR
    cond_hispin_ll = (ell == emm) and (STotR >= 0.8)  # static ell==emm check OK
    cond_33_negspin = (modeTag == 33) and (STotR < 0)  # static modeTag check OK
    cond_21_hispin = (modeTag == 21) and (STotR >= 0.8)

    cond_emr_or_hispin = cond_emr | jnp.array(cond_hispin_ll | cond_33_negspin)
    cond_21_hs = jnp.array(cond_21_hispin) & ~cond_emr_or_hispin

    i2 = jnp.where(cond_emr_or_hispin, 3, 2)
    i3 = jnp.where(cond_emr_or_hispin | cond_21_hs, 4, 3)

    # Build selected 5 frequencies and values
    freqs_5 = jnp.array(
        [all_freqs[0], all_freqs[1], all_freqs[i2], all_freqs[i3], all_freqs[5]]
    )
    vals_5 = jnp.array(
        [all_vals[0], all_vals[1], all_vals[i2], all_vals[i3], all_vals[5]]
    )

    # -----------------------------------------------------------------------
    # Step 6: Build 5x5 matrix and solve for [c0, cL, c1, c2, c4]
    # Column ordering: [c0, cL, c1, c2, c4]
    # Row: [1, fDA/(fDA^2+(f-fRD)^2), 1/f, 1/f^2, 1/f^4]
    # -----------------------------------------------------------------------
    def matrix_row(f):
        ffm1 = 1.0 / f
        ffm2 = ffm1 * ffm1
        return jnp.array(
            [1.0, fDAMP / (fDAMP**2 + (f - fRING) ** 2), ffm1, ffm2, ffm2 * ffm2]
        )

    A = jnp.stack([matrix_row(freqs_5[i]) for i in range(5)])
    coeffs = jnp.linalg.solve(A, vals_5)
    c0, cL, c1, c2, c4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

    # -----------------------------------------------------------------------
    # Step 7: Inspiral continuity at fMatchIN
    # C1INSP = inter_dphi(fMatchIN) - ins_dphi(fMatchIN)
    # CINSP  = inter_phi(fMatchIN) - ins_phi(fMatchIN) - C1INSP * fMatchIN
    # -----------------------------------------------------------------------
    def ins_phase(Mf_):
        # (emm/2) * (1/eta) * raw_inspiral(2*Mf/emm) + LambdaPN * Mf
        # Use get_inspiral_phase (raw polynomial, no C1 corrections) to match LAL's
        # IMRPhenomXHM_Inspiral_Phase_AnsatzInt which uses rescaled pPhase22->phi* coefficients.
        LambdaPN = _xhm_insp_phase_LambdaPN(modeTag, eta)
        return (
            m_over_2 / eta * get_inspiral_phase(2.0 * Mf_ / emm, theta, phase_coeffs)
            + LambdaPN * Mf_
        )

    ins_phi_fIN = ins_phase(fMatchIN)
    ins_dphi_fIN = jax.grad(ins_phase)(fMatchIN)

    inter_phi_fIN = _xhm_inter_phase_ansatz_int(
        fMatchIN, c0, c1, c2, c4, cL, fRING, fDAMP
    )
    inter_dphi_fIN = _xhm_inter_phase_ansatz_deriv(
        fMatchIN, c0, c1, c2, c4, cL, fRING, fDAMP
    )

    C1INSP = inter_dphi_fIN - ins_dphi_fIN
    CINSP = inter_phi_fIN - ins_phi_fIN - C1INSP * fMatchIN

    # -----------------------------------------------------------------------
    # Step 8: Ringdown continuity at fMatchIM
    # C1RD = inter_dphi(fMatchIM) - dphi0RD
    # CRD  = inter_phi(fMatchIM) - phi0RD - C1RD * fMatchIM
    # -----------------------------------------------------------------------
    inter_phi_fIM = _xhm_inter_phase_ansatz_int(
        fMatchIM, c0, c1, c2, c4, cL, fRING, fDAMP
    )
    inter_dphi_fIM = _xhm_inter_phase_ansatz_deriv(
        fMatchIM, c0, c1, c2, c4, cL, fRING, fDAMP
    )

    C1RD = inter_dphi_fIM - dphi0RD
    CRD = inter_phi_fIM - phi0RD - C1RD * fMatchIM

    # -----------------------------------------------------------------------
    # Step 9: deltaphiLM normalization
    # falign = 0.6 * m_over_2 * fMECO (for non-EMR)
    # deltaphiLM = m_over_2*(phi_22(two_over_m*falign)/eta + phiref22)
    #              + t0*falign - 3*pi/4*(1-m_over_2)
    #              - (phi_ins(falign) + C1INSP*falign + CINSP)
    # -----------------------------------------------------------------------
    falign = jnp.where(eta > etaEMR, 0.6 * m_over_2 * fMECO, m_over_2 * fMECO)
    two_over_m = 1.0 / m_over_2

    # Phase of 22 mode at two_over_m * falign (raw inspiral polynomial, no C1 corrections).
    # falign = 0.6*m/2*fMECO, so two_over_m*falign = 0.6*fMECO which is in XAS inspiral region.
    phi_22_at_falign = (1.0 / eta) * get_inspiral_phase(
        two_over_m * falign, theta, phase_coeffs
    )
    # t0 * falign is the time-shift contribution

    # deltaphiLM formula from LAL line 2368 (omitting phiref22 and phaseshift — those are
    # handled at the waveform level via phifRef in XLALSimIMRPhenomXHMGethlmModes):
    phi_ins_falign = ins_phase(falign)
    phi_ins_falign_with_C1 = phi_ins_falign + C1INSP * falign + CINSP

    deltaphiLM = (
        m_over_2 * phi_22_at_falign
        + t0 * falign
        - 3.0 * PI / 4.0 * (1.0 - m_over_2)
        - phi_ins_falign_with_C1
    )

    # (2,1) sign correction: LAL adds PI to deltaphiLM when the PN amplitude
    # sign at ff=0.008 is positive (IMRPhenomXHM_PN21AmpSign, line 2400-2404).
    # Source: LALSimIMRPhenomXHM_internals.c:1937-1949.
    if modeTag == 21:
        ff = 0.008
        delta_mass = pWF22["delta"]
        pn_sign = (
            (-16.0 * delta_mass * eta * ff * PI**1.5) / (3.0 * jnp.sqrt(5.0))
            + (
                4.0
                * 2.0 ** (1.0 / 3.0)
                * (chi1L - chi2L + delta_mass * (chi1L + chi2L))
                * eta
                * ff ** (4.0 / 3.0)
                * PI ** (11.0 / 6.0)
            )
            / jnp.sqrt(5.0)
            + (
                2.0
                * 2.0 ** (2.0 / 3.0)
                * eta
                * (306.0 * delta_mass - 360.0 * delta_mass * eta)
                * ff ** (5.0 / 3.0)
                * PI ** (13.0 / 6.0)
            )
            / (189.0 * jnp.sqrt(5.0))
        )
        deltaphiLM = jnp.where(pn_sign >= 0.0, deltaphiLM + PI, deltaphiLM)

    deltaphiLM = deltaphiLM % (2.0 * PI)

    return XHMPhaseCoefficients(
        ins_c0=0.0,
        ins_c1=0.0,
        ins_c2=0.0,
        ins_c4=0.0,  # not used (XAS remapping)
        ins_C1INSP=C1INSP,
        ins_CINSP=CINSP,
        inter_c0=c0,
        inter_c1=c1,
        inter_c2=c2,
        inter_c4=c4,
        inter_cL=cL,
        rd_alpha2=alpha2,
        rd_alphaL=alphaL,
        rd_C1RD=C1RD,
        rd_CRD=CRD,
        deltaphiLM=deltaphiLM,
        fMatchIN=fMatchIN,
        fMatchIM=fMatchIM,
        fRING=fRING,
        fDAMP=fDAMP,
    )


# ---------------------------------------------------------------------------
# Section 8: Phase evaluation
# ---------------------------------------------------------------------------


def xhm_phase_noModeMixing(
    Mf: Array,
    pPhase: XHMPhaseCoefficients,
    pWFHM: XHMWaveformStruct,
    pWF22: dict,
    t0: float,
) -> Array:
    """
    Evaluate the (l,m) mode phase at frequencies Mf (no mode mixing: 21, 33, 44).

    Three-region piecewise (all regions evaluated everywhere, switched by jnp.where):
      Mf < fMatchIN  -> inspiral: (emm/2)*phi_22(2Mf/emm/M_s) + LambdaPN*Mf
                        + C1INSP*Mf + CINSP
      fMatchIN <= Mf < fMatchIM -> intermediate: c0*Mf+c1*log(Mf)-c2/Mf-c4/(3Mf^3)+cL*atan(...)
      Mf >= fMatchIM -> ringdown: C1RD*Mf + CRD - fRD^2*alpha2/Mf + alphaL*atan(...)

    deltaphiLM is added as a global phase offset at evaluation time.
    Uses jnp.where for JAX-compatible branching.
    Source: IMRPhenomXHM_Phase_noModeMixing in LALSimIMRPhenomXHM.c.
    """
    emm = pWFHM.emm
    modeTag = pWFHM.modeTag
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    eta = pWF22["eta"]
    m_over_2 = emm * 0.5

    fMatchIN = pPhase.fMatchIN
    fMatchIM = pPhase.fMatchIM
    fRING = pPhase.fRING
    fDAMP = pPhase.fDAMP

    # Inspiral phase: (emm/2) * (1/eta) * raw_XAS_inspiral(2*Mf/emm) + LambdaPN*Mf + C1INSP*Mf + CINSP
    # Use get_inspiral_phase (raw polynomial, no C1 continuity corrections) to match
    # LAL's IMRPhenomXHM_Inspiral_Phase_AnsatzInt which uses pPhase22->phi* (raw XAS coefficients).
    # Using IMRPhenomXAS_Phase(f_remapped) would incorrectly include the XAS C1 continuity terms
    # (alpha1*fM_s + alpha0) because at Mf=fMatchIN, f_remapped=fMECO falls in the XAS
    # intermediate region.
    LambdaPN = _xhm_insp_phase_LambdaPN(modeTag, eta)
    phi_ins = (
        m_over_2 / eta * get_inspiral_phase(2.0 * Mf / emm, theta, phase_coeffs)
        + LambdaPN * Mf
        + pPhase.ins_C1INSP * Mf
        + pPhase.ins_CINSP
    )

    # Intermediate phase
    phi_inter = _xhm_inter_phase_ansatz_int(
        Mf,
        pPhase.inter_c0,
        pPhase.inter_c1,
        pPhase.inter_c2,
        pPhase.inter_c4,
        pPhase.inter_cL,
        fRING,
        fDAMP,
    )

    # Ringdown phase: (alpha0=C1RD)*Mf - fRD^2*alpha2/Mf + alphaL*atan(...) + CRD
    phi_ring = (
        _xhm_rd_phase_ansatz_int(
            Mf, pPhase.rd_C1RD, pPhase.rd_alpha2, pPhase.rd_alphaL, fRING, fDAMP
        )
        + pPhase.rd_CRD
    )

    # Three-region switch
    phi = jnp.where(
        Mf < fMatchIN, phi_ins, jnp.where(Mf < fMatchIM, phi_inter, phi_ring)
    )

    # Global phase normalization
    return phi + pPhase.deltaphiLM


# ---------------------------------------------------------------------------
# Section 9: Amplitude fits and evaluation
# ---------------------------------------------------------------------------

# PNglobalfactor prefactors per mode (modeInt 0=21, 1=33, 2=32, 3=44)
_AMP_PREFACTORS = [
    jnp.sqrt(2.0) / 3.0,  # 21
    0.75 * jnp.sqrt(5.0 / 7.0),  # 33
    jnp.sqrt(5.0 / 7.0) / 3.0,  # 32
    4.0 * jnp.sqrt(2.0) / 9.0 * jnp.sqrt(5.0 / 7.0),  # 44
]


def _xhm_pn_amp_coeffs(
    modeTag: int, eta: float, delta: float, chi1L: float, chi2L: float
) -> tuple:
    """
    Complex PN polynomial coefficients (pnInit..pnSixTh) for inspiral amplitude.

    For mode 21: uses useFAmpPN=0 polynomial approximation (not TaylorT4 SPA).
    Source: IMRPhenomXHM_GetPNAmplitudeCoefficients in LALSimIMRPhenomXHM_internals.c.
    Returns tuple of 7 complex scalars.
    """
    chiA = 0.5 * (chi1L - chi2L)
    chiS = 0.5 * (chi1L + chi2L)
    eta2 = eta * eta
    eta3 = eta2 * eta

    if modeTag == 21:
        # useFAmpPN=0: polynomial PN ansatz (avoids TaylorT4 SPA complexity)
        p13 = (2.0) ** (1.0 / 3.0)
        p23 = (2.0) ** (2.0 / 3.0)
        p43 = (2.0) ** (4.0 / 3.0)
        p53 = (2.0) ** (5.0 / 3.0)
        p2 = 4.0
        pnInit = 0.0 + 0.0j
        pnOneTh = delta * PI ** (1.0 / 3.0) * p13 + 0.0j
        pnTwoTh = (-3.0 * (chiA + chiS * delta) / 2.0) * PI ** (2.0 / 3.0) * p23
        pnThreeTh = (335.0 * delta + 1404.0 * delta * eta) / 672.0 * PI * 2.0 + 0.0j
        pnFourTh = (
            (
                3427.0 * chiA
                - 672.0j * delta
                + 3427.0 * chiS * delta
                - 8404.0 * chiA * eta
                - 3860.0 * chiS * delta * eta
                - 1344.0 * delta * PI
                - 672.0j * delta * jnp.log(16.0)
            )
            / 1344.0
            * PI ** (4.0 / 3.0)
            * p43
        )
        pnFiveTh = (
            (
                -155965824.0 * chiA * chiS
                - 964357.0 * delta
                + 432843264.0 * chiA * chiS * eta
                - 23670792.0 * delta * eta
                + 24385536.0 * chiA * PI
                + 24385536.0 * chiS * delta * PI
                - 77982912.0 * delta * chiA**2
                + 81285120.0 * delta * eta * chiA**2
                - 77982912.0 * delta * chiS**2
                + 39626496.0 * delta * eta * chiS**2
                + 21535920.0 * delta * eta2
            )
            / 8128512.0
            * PI ** (5.0 / 3.0)
            * p53
        )
        pnSixTh = (
            (
                143063173.0 * chiA
                - 1350720.0j * delta
                + 143063173.0 * chiS * delta
                - 546199608.0 * chiA * eta
                - 72043776.0j * delta * eta
                - 169191096.0 * chiS * delta * eta
                - 9898560.0 * delta * PI
                + 20176128.0 * delta * eta * PI
                - 5402880.0j * delta * jnp.log(2.0)
                - 17224704.0j * delta * eta * jnp.log(2.0)
                + 61725888.0 * chiS * delta * chiA**2
                - 81285120.0 * chiS * delta * eta * chiA**2
                + 20575296.0 * chiA**3
                - 81285120.0 * eta * chiA**3
                + 61725888.0 * chiA * chiS**2
                - 165618432.0 * chiA * eta * chiS**2
                + 20575296.0 * delta * chiS**3
                - 1016064.0 * delta * eta * chiS**3
                + 128873808.0 * chiA * eta2
                - 3859632.0 * chiS * delta * eta2
            )
            / 5419008.0
            * PI**2.0
            * p2
        )

    elif modeTag == 33:
        r23 = (2.0 / 3.0) ** (1.0 / 3.0)
        r43 = (2.0 / 3.0) ** (4.0 / 3.0)
        r53 = (2.0 / 3.0) ** (5.0 / 3.0)
        r2 = (2.0 / 3.0) ** 2.0
        pnInit = 0.0 + 0.0j
        pnOneTh = delta * PI ** (1.0 / 3.0) * r23 + 0.0j
        pnTwoTh = 0.0 + 0.0j
        pnThreeTh = (-1945.0 * delta + 2268.0 * delta * eta) / 672.0 * PI * (
            2.0 / 3.0
        ) + 0.0j
        pnFourTh = (
            (
                325.0 * chiA
                - 504.0j * delta
                + 325.0 * chiS * delta
                - 1120.0 * chiA * eta
                - 80.0 * chiS * delta * eta
                + 120.0 * delta * PI
                + 720.0j * delta * jnp.log(1.5)
            )
            / 120.0
            * PI ** (4.0 / 3.0)
            * r43
        )
        pnFiveTh = (
            (
                -2263282560.0 * chiA * chiS
                - 1077664867.0 * delta
                + 9053130240.0 * chiA * chiS * eta
                - 5926068792.0 * delta * eta
                - 1131641280.0 * delta * chiA**2
                + 4470681600.0 * delta * eta * chiA**2
                - 1131641280.0 * delta * chiS**2
                + 55883520.0 * delta * eta * chiS**2
                + 2966264784.0 * delta * eta2
            )
            / 447068160.0
            * PI ** (5.0 / 3.0)
            * r53
        )
        pnSixTh = (
            (
                22007835.0 * chiA
                + 26467560.0j * delta
                + 22007835.0 * chiS * delta
                - 80190540.0 * chiA * eta
                - 98774368.0j * delta * eta
                - 31722300.0 * chiS * delta * eta
                - 9193500.0 * delta * PI
                + 17826480.0 * delta * eta * PI
                - 37810800.0j * delta * jnp.log(1.5)
                + 37558080.0j * delta * eta * jnp.log(1.5)
                - 12428640.0 * chiA * eta2
                - 6078240.0 * chiS * delta * eta2
            )
            / 2177280.0
            * PI**2.0
            * r2
        )

    elif modeTag == 32:
        pnInit = 0.0 + 0.0j
        pnOneTh = 0.0 + 0.0j
        pnTwoTh = (-1.0 + 3.0 * eta) * PI ** (2.0 / 3.0) + 0.0j
        pnThreeTh = -4.0 * chiS * eta * PI + 0.0j
        pnFourTh = (10471.0 - 61625.0 * eta + 82460.0 * eta2) / 10080.0 * PI ** (
            4.0 / 3.0
        ) + 0.0j
        pnFiveTh = (
            (
                2520.0j
                - 3955.0 * chiS
                - 3955.0 * chiA * delta
                - 11088.0j * eta
                + 10810.0 * chiS * eta
                + 11865.0 * chiA * delta * eta
                - 12600.0 * chiS * eta2
            )
            / 840.0
            * PI ** (5.0 / 3.0)
        )
        pnSixTh = (
            (
                824173699.0
                + 2263282560.0 * chiA * chiS * delta
                - 26069649.0 * eta
                - 15209631360.0 * chiA * chiS * delta * eta
                + 3576545280.0 * chiS * eta * PI
                + 1131641280.0 * chiA**2
                - 7865605440.0 * eta * chiA**2
                + 1131641280.0 * chiS**2
                - 11870591040.0 * eta * chiS**2
                - 13202119896.0 * eta2
                + 13412044800.0 * chiA**2 * eta2
                + 5830513920.0 * chiS**2 * eta2
                + 5907445488.0 * eta3
            )
            / 447068160.0
            * PI**2.0
        )

    else:  # modeTag == 44
        h13 = (0.5) ** (1.0 / 3.0)
        h43 = (0.5) ** (4.0 / 3.0)
        h53 = (0.5) ** (5.0 / 3.0)
        h2 = 0.25
        pnInit = 0.0 + 0.0j
        pnOneTh = 0.0 + 0.0j
        pnTwoTh = (1.0 - 3.0 * eta) * PI ** (2.0 / 3.0) * h13**2 + 0.0j
        pnThreeTh = 0.0 + 0.0j
        pnFourTh = (-158383.0 + 641105.0 * eta - 446460.0 * eta2) / 36960.0 * PI ** (
            4.0 / 3.0
        ) * h43 + 0.0j
        pnFiveTh = (
            (
                -1008.0j
                + 565.0 * chiS
                + 565.0 * chiA * delta
                + 3579.0j * eta
                - 2075.0 * chiS * eta
                - 1695.0 * chiA * delta * eta
                + 240.0 * PI
                - 720.0 * eta * PI
                + 960.0j * jnp.log(2.0)
                - 2880.0j * eta * jnp.log(2.0)
                + 1140.0 * chiS * eta2
            )
            / 120.0
            * PI ** (5.0 / 3.0)
            * h53
        )
        pnSixTh = (
            (
                7888301437.0
                - 147113366400.0 * chiA * chiS * delta
                - 745140957231.0 * eta
                + 441340099200.0 * chiA * chiS * delta * eta
                - 73556683200.0 * chiA**2
                + 511264353600.0 * eta * chiA**2
                - 73556683200.0 * chiS**2
                + 224302478400.0 * eta * chiS**2
                + 2271682065240.0 * eta2
                - 871782912000.0 * chiA**2 * eta2
                - 10897286400.0 * chiS**2 * eta2
                - 805075876080.0 * eta3
            )
            / 29059430400.0
            * PI**2.0
            * h2
        )

    return (pnInit, pnOneTh, pnTwoTh, pnThreeTh, pnFourTh, pnFiveTh, pnSixTh)


def _xhm_pn_poly(f: float | Array, pn_coeffs: tuple) -> Array:
    """Evaluate complex PN polynomial at frequency f."""
    c0, c1, c2, c3, c4, c5, c6 = pn_coeffs
    return (
        c0
        + f ** (1.0 / 3.0) * c1
        + f ** (2.0 / 3.0) * c2
        + f * c3
        + f ** (4.0 / 3.0) * c4
        + f ** (5.0 / 3.0) * c5
        + f**2.0 * c6
    )


def _xhm_insp_rescaled(
    f: float | Array,
    PNgf: float | Array,
    pn_coeffs: tuple,
    rho1: float | Array,
    rho2: float | Array,
    rho3: float | Array,
    fc: float | Array,
) -> Array:
    """
    Inspiral amplitude ansatz in 'rescaled' units (InspRescaleFactor=1).
    Returns: PNglobalfactor * |pn_poly(f)| + rho1*(f/fc)^(7/3) + rho2*(f/fc)^(8/3) + rho3*(f/fc)^3
    """
    return (
        PNgf * jnp.abs(_xhm_pn_poly(f, pn_coeffs))
        + rho1 * (f / fc) ** (7.0 / 3.0)
        + rho2 * (f / fc) ** (8.0 / 3.0)
        + rho3 * (f / fc) ** 3.0
    )


def _xhm_rd_rescaled_v1(
    f: float | Array,
    alambda: float | Array,
    lambda_: float | Array,
    sigma: float | Array,
    fRING: float | Array,
    fDAMP: float | Array,
) -> Array:
    """
    RD amplitude ansatz case 1 (122022 for mode 32).
    Returns: fDAMP*|alambda| * exp(-lambda*(f-fRING)/(fDAMP*sigma)) / ((f-fRING)^2+(fDAMP*sigma)^2)
    No sigma in numerator, no f^(-1/12). Used when wf22R is already scaled by ampNorm*f^(-7/6).
    """
    dfr = f - fRING
    dfd = fDAMP * sigma
    return fDAMP * jnp.abs(alambda) * jnp.exp(-dfr * lambda_ / dfd) / (dfr**2 + dfd**2)


def _xhm_insp_amp_colloc_pts(
    modeTag: int, eta: float, chiPN: float, dchi_half: float, delta: float
) -> tuple:
    """122022 inspiral amplitude collocation-point fits (iv1, iv2, iv3)."""
    S = chiPN
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    eta7 = eta1 * eta6
    S1 = S
    S2 = S1 * S1
    S3 = S1 * S2
    chidiff1 = dchi_half
    chidiff2 = chidiff1 * chidiff1
    sqroot = jnp.sqrt(eta)

    if modeTag == 21:
        iv1 = jnp.abs(
            chidiff1
            * eta5
            * (-3962.5020052272976 + 987.635855365408 * S1 - 134.98527058315528 * S2)
            + delta
            * (
                19.30531354642419
                + 16.6640319856064 * eta1
                - 120.58166037019478 * eta2
                + 220.77233521626252 * eta3
            )
            * sqroot
            + chidiff1
            * delta
            * (
                31.364509907424765 * eta1
                - 843.6414532232126 * eta2
                + 2638.3077554662905 * eta3
            )
            * sqroot
            + chidiff1
            * delta
            * (
                32.374226994179054 * eta1
                - 202.86279451816662 * eta2
                + 347.1621871204769 * eta3
            )
            * S1
            * sqroot
            + delta
            * S1
            * (
                -16.75726972301224
                * (
                    1.1787350890261943
                    - 7.812073811917883 * eta1
                    + 99.47071002831267 * eta2
                    - 500.4821414428368 * eta3
                    + 876.4704270866478 * eta4
                )
                + 2.3439955698372663
                * (
                    0.9373952326655807
                    + 7.176140122833879 * eta1
                    - 279.6409723479635 * eta2
                    + 2178.375177755584 * eta3
                    - 4768.212511142035 * eta4
                )
                * S1
            )
            * sqroot
        )
        iv2 = jnp.abs(
            chidiff1
            * eta5
            * (-2898.9172078672705 + 580.9465034962822 * S1 + 22.251142639924076 * S2)
            + delta
            * (
                chidiff2
                * (
                    -18.541685007214625 * eta1
                    + 166.7427445020744 * eta2
                    - 417.5186332459383 * eta3
                )
                + chidiff1
                * (
                    41.61457952037761 * eta1
                    - 779.9151607638761 * eta2
                    + 2308.6520892707795 * eta3
                )
            )
            * sqroot
            + delta
            * (
                11.414934585404561
                + 30.883118528233638 * eta1
                - 260.9979123967537 * eta2
                + 1046.3187137392433 * eta3
                - 1556.9475493549746 * eta4
            )
            * sqroot
            + delta
            * S1
            * (
                -10.809007068469844
                * (
                    1.1408749895922659
                    - 18.140470190766937 * eta1
                    + 368.25127088896744 * eta2
                    - 3064.7291458207815 * eta3
                    + 11501.848278358668 * eta4
                    - 16075.676528787526 * eta5
                )
                + 1.0088254664333147
                * (
                    1.2322739396680107
                    - 192.2461213084741 * eta1
                    + 4257.760834055382 * eta2
                    - 35561.24587952242 * eta3
                    + 130764.22485304279 * eta4
                    - 177907.92440833704 * eta5
                )
                * S1
            )
            * sqroot
            + delta
            * (
                chidiff1
                * (
                    36.88578491943111 * eta1
                    - 321.2569602623214 * eta2
                    + 748.6659668096737 * eta3
                )
                * S1
                + chidiff1
                * (
                    -95.42418611585117 * eta1
                    + 1217.338674959742 * eta2
                    - 3656.192371615541 * eta3
                )
                * S2
            )
            * sqroot
        )
        iv3 = jnp.abs(
            chidiff1
            * eta5
            * (-2282.9983216879655 + 157.94791186394787 * S1 + 16.379731479465033 * S2)
            + chidiff1
            * delta
            * (
                21.935833431534224 * eta1
                - 460.7130131927895 * eta2
                + 1350.476411541137 * eta3
            )
            * sqroot
            + delta
            * (
                5.390240326328237
                + 69.01761987509603 * eta1
                - 568.0027716789259 * eta2
                + 2435.4098320959706 * eta3
                - 3914.3390484239667 * eta4
            )
            * sqroot
            + chidiff1
            * delta
            * (
                29.731007410186827 * eta1
                - 372.09609843131386 * eta2
                + 1034.4897198648962 * eta3
            )
            * S1
            * sqroot
            + delta
            * S1
            * (
                -7.1976397556450715
                * (
                    0.7603360145475428
                    - 6.587249958654174 * eta1
                    + 120.87934060776237 * eta2
                    - 635.1835857158857 * eta3
                    + 1109.0598539312573 * eta4
                )
                - 0.0811847192323969
                * (
                    7.951454648295709
                    + 517.4039644814231 * eta1
                    - 9548.970156895082 * eta2
                    + 52586.63520999897 * eta3
                    - 93272.17990295641 * eta4
                )
                * S1
                - 0.28384547935698246
                * (
                    -0.8870770459576875
                    + 180.0378964169756 * eta1
                    - 2707.9572896559484 * eta2
                    + 14158.178124971111 * eta3
                    - 24507.800226675925 * eta4
                )
                * S2
            )
            * sqroot
        )
    elif modeTag == 33:
        iv1 = (
            chidiff1
            * eta5
            * (155.1434307076563 + 26.852777193715088 * S1 + 1.4157230717300835 * S2)
            + chidiff1
            * delta
            * (
                6.296698171560171 * eta1
                + 15.81328761563562 * eta2
                - 141.85538063933927 * eta3
            )
            * sqroot
            + delta
            * (
                20.94372147101354
                + 68.14577638017842 * eta1
                - 898.470298591732 * eta2
                + 4598.64854748635 * eta3
                - 8113.199260593833 * eta4
            )
            * sqroot
            + chidiff1
            * delta
            * (
                29.221863857271703 * eta1
                - 348.1658322276406 * eta2
                + 965.4670353331536 * eta3
            )
            * S1
            * sqroot
            + delta
            * S1
            * (
                -9.753610761811967
                * (
                    1.7819678168496158
                    - 44.07982999150369 * eta1
                    + 750.8933447725581 * eta2
                    - 5652.44754829634 * eta3
                    + 19794.855873435758 * eta4
                    - 26407.40988450443 * eta5
                )
                + 0.014210376114848208
                * (
                    -196.97328616330392
                    + 7264.159472864562 * eta1
                    - 125763.47850622259 * eta2
                    + 1.1458022059130718e6 * eta3
                    - 4.948175330328345e6 * eta4
                    + 7.911048294733888e6 * eta5
                )
                * S1
                - 0.26859293613553986
                * (
                    -8.029069605349488
                    + 888.7768796633982 * eta1
                    - 16664.276483466252 * eta2
                    + 128973.72291098491 * eta3
                    - 462437.2690007375 * eta4
                    + 639989.1197424605 * eta5
                )
                * S2
            )
            * sqroot
        )
        iv2 = (
            chidiff1
            * eta5
            * (161.62678370819597 + 37.141092711336846 * S1 - 0.16889712161410445 * S2)
            + chidiff1
            * delta
            * (
                3.4895829486899825 * eta1
                + 51.07954458810889 * eta2
                - 249.71072528701757 * eta3
            )
            * sqroot
            + delta
            * (
                12.501397517602173
                + 35.75290806646574 * eta1
                - 357.6437296928763 * eta2
                + 1773.8883882162215 * eta3
                - 3100.2396041211605 * eta4
            )
            * sqroot
            + chidiff1
            * delta
            * (
                13.854211287141906 * eta1
                - 135.54916401086845 * eta2
                + 327.2467193417936 * eta3
            )
            * S1
            * sqroot
            + delta
            * S1
            * (
                -5.2580116732827085
                * (
                    1.7794900975289085
                    - 48.20753331991333 * eta1
                    + 861.1650630146937 * eta2
                    - 6879.681319382729 * eta3
                    + 25678.53964955809 * eta4
                    - 36383.824902258915 * eta5
                )
                + 0.028627002336747746
                * (
                    -50.57295946557892
                    + 734.7581857539398 * eta1
                    - 2287.0465658878725 * eta2
                    + 15062.821881048358 * eta3
                    - 168311.2370167227 * eta4
                    + 454655.37836367317 * eta5
                )
                * S1
                - 0.15528289788512326
                * (
                    -12.738184090548508
                    + 1129.44485109116 * eta1
                    - 25091.14888164863 * eta2
                    + 231384.03447562453 * eta3
                    - 953010.5908118751 * eta4
                    + 1.4516597366230418e6 * eta5
                )
                * S2
            )
            * sqroot
        )
        iv3 = (
            chidiff1
            * delta
            * (
                -0.5869777957488564 * eta1
                + 32.65536124256588 * eta2
                - 110.10276573567405 * eta3
            )
            + chidiff1
            * delta
            * (
                3.524800489907584 * eta1
                - 40.26479860265549 * eta2
                + 113.77466499598913 * eta3
            )
            * S1
            + delta
            * S1
            * (
                -1.2846335585108297
                * (
                    0.09991079016763821
                    + 1.37856806162599 * eta1
                    + 23.26434219690476 * eta2
                    - 34.842921754693386 * eta3
                    - 70.83896459998664 * eta4
                )
                - 0.03496714763391888
                * (
                    -0.230558571912664
                    + 188.38585449575902 * eta1
                    - 3736.1574640444287 * eta2
                    + 22714.70643022915 * eta3
                    - 43221.0453556626 * eta4
                )
                * S1
            )
            + chidiff1
            * eta7
            * (
                2667.3441342894776
                + 47.94869769580204 * chidiff2
                + 793.5988192446642 * S1
                + 293.89657731755483 * S2
            )
            + delta
            * (
                5.148353856800232
                + 148.98231189649468 * eta1
                - 2774.5868652930294 * eta2
                + 29052.156454239772 * eta3
                - 162498.31493332976 * eta4
                + 460912.76402476896 * eta5
                - 521279.50781871413 * eta6
            )
            * sqroot
        )
    elif modeTag == 32:
        iv1 = (
            (
                chidiff1
                * delta
                * (
                    -0.739317114582042 * eta1
                    - 47.473246070362634 * eta2
                    + 278.9717709112207 * eta3
                    - 566.6420939162068 * eta4
                )
                + chidiff2
                * (
                    -0.5873680378268906 * eta1
                    + 6.692187014925888 * eta2
                    - 24.37776782232888 * eta3
                    + 23.783684827838247 * eta4
                )
            )
            * sqroot
            + (
                3.2940434453819694
                + 4.94285331708559 * eta1
                - 343.3143244815765 * eta2
                + 3585.9269057886418 * eta3
                - 19279.186145681153 * eta4
                + 51904.91007211022 * eta5
                - 55436.68857586653 * eta6
            )
            * sqroot
            + chidiff1
            * delta
            * (
                12.488240781993923 * eta1
                - 209.32038774208385 * eta2
                + 1160.9833883184604 * eta3
                - 2069.5349737049073 * eta4
            )
            * S1
            * sqroot
            + S1
            * (
                0.6343034651912586
                * (
                    -2.5844888818001737
                    + 78.98200041834092 * eta1
                    - 1087.6241783616488 * eta2
                    + 7616.234910399297 * eta3
                    - 24776.529123239357 * eta4
                    + 30602.210950069973 * eta5
                )
                - 0.062088720220899465
                * (
                    6.5586380356588565
                    + 36.01386705325694 * eta1
                    - 3124.4712274775407 * eta2
                    + 33822.437731298516 * eta3
                    - 138572.93700180828 * eta4
                    + 198366.10615196894 * eta5
                )
                * S1
            )
            * sqroot
        )
        iv2 = (
            (
                chidiff2
                * (
                    -0.03940151060321499 * eta1
                    + 1.9034209537174116 * eta2
                    - 8.78587250202154 * eta3
                )
                + chidiff1
                * delta
                * (
                    -1.704299788495861 * eta1
                    - 4.923510922214181 * eta2
                    + 0.36790005839460627 * eta3
                )
            )
            * sqroot
            + (
                2.2911849711339123
                - 5.1846950040514335 * eta1
                + 60.10368251688146 * eta2
                - 1139.110227749627 * eta3
                + 7970.929280907627 * eta4
                - 25472.73682092519 * eta5
                + 30950.67053883646 * eta6
            )
            * sqroot
            + S1
            * (
                0.7718201508695763
                * (
                    -1.3012906461000349
                    + 26.432880113146012 * eta1
                    - 186.5001124789369 * eta2
                    + 712.9101229418721 * eta3
                    - 970.2126139442341 * eta4
                )
                + 0.04832734931068797
                * (
                    -5.9999628512498315
                    + 78.98681284391004 * eta1
                    + 1.8360177574514709 * eta2
                    - 2537.636347529708 * eta3
                    + 6858.003573909322 * eta4
                )
                * S1
            )
            * sqroot
        )
        iv3 = (
            (
                chidiff2
                * (
                    -0.6358511175987503 * eta1
                    + 5.555088747533164 * eta2
                    - 14.078156877577733 * eta3
                )
                + chidiff1
                * delta
                * (
                    0.23205448591711159 * eta1
                    - 19.46049432345157 * eta2
                    + 36.20685853857613 * eta3
                )
            )
            * sqroot
            + (
                1.1525594672495008
                + 7.380126197972549 * eta1
                - 17.51265776660515 * eta2
                - 976.9940395257111 * eta3
                + 8880.536804741967 * eta4
                - 30849.228936891763 * eta5
                + 38785.53683146884 * eta6
            )
            * sqroot
            + chidiff1
            * delta
            * (
                1.904350804857431 * eta1
                - 25.565242391371093 * eta2
                + 80.67120303906654 * eta3
            )
            * S1
            * sqroot
            + S1
            * (
                0.785171689871352
                * (
                    -0.4634745514643032
                    + 18.70856733065619 * eta1
                    - 167.9231114864569 * eta2
                    + 744.7699462372949 * eta3
                    - 1115.008825153004 * eta4
                )
                + 0.13469300326662165
                * (
                    -2.7311391326835133
                    + 72.17373498208947 * eta1
                    - 483.7040402103785 * eta2
                    + 1136.8367114738041 * eta3
                    - 472.02962341590774 * eta4
                )
                * S1
            )
            * sqroot
        )
    else:
        iv1 = (
            (
                chidiff1
                * delta
                * (
                    0.5697308729057493 * eta1
                    + 8.895576813118867 * eta2
                    - 34.98399465240273 * eta3
                )
                + chidiff2
                * (
                    1.6370346538130884 * eta1
                    - 14.597095790380884 * eta2
                    + 33.182723737396294 * eta3
                )
            )
            * sqroot
            + (
                5.2601381002242595
                - 3.557926105832778 * eta1
                - 138.9749850448088 * eta2
                + 603.7453704122706 * eta3
                - 923.5495700703648 * eta4
            )
            * sqroot
            + S1
            * (
                -0.41839636169678796
                * (
                    5.143510231379954
                    + 104.62892421207803 * eta1
                    - 4232.508174045782 * eta2
                    + 50694.024801783446 * eta3
                    - 283097.33358214336 * eta4
                    + 758333.2655404843 * eta5
                    - 788783.0559069642 * eta6
                )
                - 0.05653522061311774
                * (
                    5.605483124564013
                    + 694.00652410087 * eta1
                    - 17551.398321516353 * eta2
                    + 165236.6480734229 * eta3
                    - 761661.9645651339 * eta4
                    + 1.7440315410044065e6 * eta5
                    - 1.6010489769238676e6 * eta6
                )
                * S1
                - 0.023693246676754775
                * (
                    16.437107575918503
                    - 2911.2154288136217 * eta1
                    + 89338.32554683842 * eta2
                    - 1.0803340811860575e6 * eta3
                    + 6.255666490084672e6 * eta4
                    - 1.7434160932177313e7 * eta5
                    + 1.883460394974573e7 * eta6
                )
                * S2
            )
            * sqroot
        )
        iv2 = (
            (
                chidiff2
                * (
                    -0.8318312659717388 * eta1
                    + 7.6541168007977864 * eta2
                    - 16.648660653220123 * eta3
                )
                + chidiff1
                * delta
                * (
                    2.214478316304753 * eta1
                    - 7.028104574328955 * eta2
                    + 5.56587823143958 * eta3
                )
            )
            * sqroot
            + (
                3.173191054680422
                + 6.707695566702527 * eta1
                - 155.22519772642607 * eta2
                + 604.0067075996933 * eta3
                - 876.5048298377644 * eta4
            )
            * sqroot
            + chidiff1
            * delta
            * (
                4.749663394334708 * eta1
                - 42.62996105525792 * eta2
                + 97.01712147349483 * eta3
            )
            * S1
            * sqroot
            + S1
            * (
                -0.2627203100303006
                * (
                    6.460396349297595
                    - 52.82425783851536 * eta1
                    - 552.1725902144143 * eta2
                    + 12546.255587592654 * eta3
                    - 81525.50289542897 * eta4
                    + 227254.37897941095 * eta5
                    - 234487.3875219032 * eta6
                )
                - 0.008424003742397579
                * (
                    -109.26773035716548
                    + 15514.571912666677 * eta1
                    - 408022.6805482195 * eta2
                    + 4.620165968920881e6 * eta3
                    - 2.6446950627957724e7 * eta4
                    + 7.539643948937692e7 * eta5
                    - 8.510662871580401e7 * eta6
                )
                * S1
                - 0.008830881730801855
                * (
                    -37.49992494976597
                    + 1359.7883958101172 * eta1
                    - 23328.560285901796 * eta2
                    + 260027.4121353132 * eta3
                    - 1.723865744472182e6 * eta4
                    + 5.858455766230802e6 * eta5
                    - 7.756341721552802e6 * eta6
                )
                * S2
                - 0.027167813927224657
                * (
                    34.281932237450256
                    - 3312.7658728016568 * eta1
                    + 84126.14531363266 * eta2
                    - 956052.0170024392 * eta3
                    + 5.570748509263883e6 * eta4
                    - 1.6270212243584689e7 * eta5
                    + 1.8855858173287075e7 * eta6
                )
                * S3
            )
            * sqroot
        )
        iv3 = (
            (
                chidiff1
                * delta
                * (
                    1.4739380748149558 * eta1
                    + 0.06541707987699942 * eta2
                    - 9.473290540936633 * eta3
                )
                + chidiff2
                * (
                    -0.3640838331639651 * eta1
                    + 3.7369795937033756 * eta2
                    - 8.709159662885131 * eta3
                )
            )
            * sqroot
            + (
                1.7335503724888923
                + 12.656614578053683 * eta1
                - 139.6610487470118 * eta2
                + 456.78649322753824 * eta3
                - 599.2709938848282 * eta4
            )
            * sqroot
            + chidiff1
            * delta
            * (
                2.3532739003216254 * eta1
                - 21.37216554136868 * eta2
                + 53.35003268489743 * eta3
            )
            * S1
            * sqroot
            + S1
            * (
                -0.15782329022461472
                * (
                    6.0309399412954345
                    - 229.16361598098678 * eta1
                    + 3777.477006415653 * eta2
                    - 31109.307191210424 * eta3
                    + 139319.8239886073 * eta4
                    - 324891.4001578353 * eta5
                    + 307714.3954026392 * eta6
                )
                - 0.03050157254864058
                * (
                    4.232861441291087
                    + 1609.4251694451375 * eta1
                    - 51213.27604422822 * eta2
                    + 612317.1751155312 * eta3
                    - 3.5589766538499263e6 * eta4
                    + 1.0147654212772278e7 * eta5
                    - 1.138861230369246e7 * eta6
                )
                * S1
                - 0.026407497690308382
                * (
                    -17.184685557542196
                    + 744.4743953122965 * eta1
                    - 10494.512487701073 * eta2
                    + 66150.52694069289 * eta3
                    - 184787.79377504133 * eta4
                    + 148102.4257785174 * eta5
                    + 128167.89151782403 * eta6
                )
                * S2
            )
            * sqroot
        )

    return iv1, iv2, iv3


def _xhm_inter_amp_colloc_pts(
    modeTag: int, eta: float, STotR: float, dchi_half: float, delta: float, chiPN: float
) -> tuple:
    """122022 intermediate amplitude collocation-point fits (int1..int4)."""
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = STotR
    S2 = S1 * S1
    chidiff1 = dchi_half
    chidiff2 = chidiff1 * chidiff1
    sqroot = jnp.sqrt(eta)

    if modeTag == 21:
        S1 = STotR
        S2 = S1 * S1
        int1 = jnp.abs(
            delta
            * eta1
            * (
                chidiff2
                * (
                    5.159755997682368 * eta1
                    - 30.293198248154948 * eta2
                    + 63.70715919820867 * eta3
                )
                + chidiff1
                * (
                    8.262642080222694 * eta1
                    - 415.88826990259116 * eta2
                    + 1427.5951158851076 * eta3
                )
            )
            + delta
            * eta1
            * (
                18.55363583212328
                - 66.46950491124205 * eta1
                + 447.2214642597892 * eta2
                - 1614.178472020212 * eta3
                + 2199.614895727586 * eta4
            )
            + chidiff1
            * eta5
            * (-1698.841763891122 - 195.27885562092342 * S1 - 1.3098861736238572 * S2)
            + delta
            * eta1
            * (
                chidiff1
                * (
                    34.17829404207186 * eta1
                    - 386.34587928670015 * eta2
                    + 1022.8553774274128 * eta3
                )
                * S1
                + chidiff1
                * (
                    56.76554600963724 * eta1
                    - 491.4593694689354 * eta2
                    + 1016.6019654342113 * eta3
                )
                * S2
            )
            + delta
            * eta1
            * S1
            * (
                -8.276366844994188
                * (
                    1.0677538075697492
                    - 24.12941323757896 * eta1
                    + 516.7886322104276 * eta2
                    - 4389.799658723288 * eta3
                    + 16770.447637953577 * eta4
                    - 23896.392706809565 * eta5
                )
                - 1.6908277400304084
                * (
                    3.4799140066657928
                    - 29.00026389706585 * eta1
                    + 114.8330693231833 * eta2
                    - 184.13091281984674 * eta3
                    + 592.300353344717 * eta4
                    - 2085.0821513466053 * eta5
                )
                * S1
                - 0.46006975902558517
                * (
                    -2.1663474937625975
                    + 826.026625945615 * eta1
                    - 17333.549622759732 * eta2
                    + 142904.08962903373 * eta3
                    - 528521.6231015554 * eta4
                    + 731179.456702448 * eta5
                )
                * S2
            )
        )
        int2 = jnp.abs(
            delta
            * eta1
            * (
                13.757856231617446
                - 12.783698329428516 * eta1
                + 12.048194546899204 * eta2
            )
            + chidiff1
            * delta
            * eta1
            * (
                15.107530092096438 * eta1
                - 416.811753638553 * eta2
                + 1333.6181181686939 * eta3
            )
            + chidiff1
            * eta5
            * (-1549.6199518612063 - 102.34716990474509 * S1 - 3.3637011939285015 * S2)
            + delta
            * eta1
            * (
                chidiff1
                * (
                    36.358142200869295 * eta1
                    - 384.2123173145321 * eta2
                    + 984.6826660818275 * eta3
                )
                * S1
                + chidiff1
                * (
                    4.159271594881928 * eta1
                    + 105.10911749116399 * eta2
                    - 639.190132707115 * eta3
                )
                * S2
            )
            + delta
            * eta1
            * S1
            * (
                -8.097876227116853
                * (
                    0.6569459700232806
                    + 9.861355377849485 * eta1
                    - 116.88834714736281 * eta2
                    + 593.8035334117192 * eta3
                    - 1063.0692862578455 * eta4
                )
                - 1.0546375154878165
                * (
                    0.745557030602097
                    + 65.25215540635162 * eta1
                    - 902.5751736558435 * eta2
                    + 4350.442990924205 * eta3
                    - 7141.611333893155 * eta4
                )
                * S1
                - 0.5006664599166409
                * (
                    10.289020582277626
                    - 212.00728173197498 * eta1
                    + 2334.0029399672358 * eta2
                    - 11939.621138801092 * eta3
                    + 21974.8201355744 * eta4
                )
                * S2
            )
        )
        int3 = jnp.abs(
            delta
            * eta1
            * (
                13.318990196097973
                - 21.755549987331054 * eta1
                + 76.14884211156267 * eta2
                - 127.62161159798488 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                17.704321326939414 * eta1
                - 434.4390350012534 * eta2
                + 1366.2408490833282 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                11.877985158418596 * eta1
                - 131.04937626836355 * eta2
                + 343.79587860999874 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (-1522.8543551416456 - 16.639896279650678 * S1 + 3.0053086651515843 * S2)
            + delta
            * eta1
            * S1
            * (
                -8.665646058245033
                * (
                    0.7862132291286934
                    + 8.293609541933655 * eta1
                    - 111.70764910503321 * eta2
                    + 576.7172598056907 * eta3
                    - 1001.2370065269745 * eta4
                )
                - 0.9459820574514348
                * (
                    1.309016452198605
                    + 48.94077040282239 * eta1
                    - 817.7854010574645 * eta2
                    + 4331.56002883546 * eta3
                    - 7518.309520232795 * eta4
                )
                * S1
                - 0.4308267743835775
                * (
                    9.970654092010587
                    - 302.9708323417439 * eta1
                    + 3662.099161055873 * eta2
                    - 17712.883990278668 * eta3
                    + 29480.158198408903 * eta4
                )
                * S2
            )
        )
        int4 = jnp.abs(
            delta
            * eta1
            * (
                13.094382343446163
                - 22.831152256559523 * eta1
                + 83.20619262213437 * eta2
                - 139.25546924151664 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                20.120192352555357 * eta1
                - 458.2592421214168 * eta2
                + 1430.3698681181 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                12.925363020014743 * eta1
                - 126.87194512915104 * eta2
                + 280.6003655502327 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (-1528.956015503355 + 74.44462583487345 * S1 - 2.2456928156392197 * S2)
            + delta
            * eta1
            * S1
            * (
                -9.499741513411829
                * (
                    0.912120958549489
                    + 2.400945118514037 * eta1
                    - 33.651192908287236 * eta2
                    + 166.04254881175257 * eta3
                    - 248.5050377498615 * eta4
                )
                - 0.7850652143322492
                * (
                    1.534131218043425
                    + 60.81773903539479 * eta1
                    - 1032.1319480683567 * eta2
                    + 5381.481380750608 * eta3
                    - 9077.037917192794 * eta4
                )
                * S1
                - 0.21540359093306097
                * (
                    9.42805409480658
                    - 109.06544597367301 * eta1
                    + 385.8345793110262 * eta2
                    + 1889.9613367802453 * eta3
                    - 9835.416414460055 * eta4
                )
                * S2
            )
        )
    elif modeTag == 33:
        S1 = STotR
        S2 = S1 * S1
        int1 = (
            chidiff1
            * delta
            * eta1
            * (
                -0.3516244197696068 * eta1
                + 40.425151307421416 * eta2
                - 148.3162618111991 * eta3
            )
            + delta
            * eta1
            * (
                26.998512565991778
                - 146.29035440932105 * eta1
                + 914.5350366065115 * eta2
                - 3047.513201789169 * eta3
                + 3996.417635728702 * eta4
            )
            + chidiff1
            * delta
            * eta1
            * (
                5.575274516197629 * eta1
                - 44.592719238427094 * eta2
                + 99.91399033058927 * eta3
            )
            * S1
            + delta
            * eta1
            * S1
            * (
                -0.5383304368673182
                * (
                    -7.456619067234563
                    + 129.36947401891433 * eta1
                    - 843.7897535238325 * eta2
                    + 3507.3655567272644 * eta3
                    - 9675.194644814854 * eta4
                    + 11959.83533107835 * eta5
                )
                - 0.28042799223829407
                * (
                    -6.212827413930676
                    + 266.69059813274475 * eta1
                    - 4241.537539226717 * eta2
                    + 32634.43965039936 * eta3
                    - 119209.70783201039 * eta4
                    + 166056.27237509796 * eta5
                )
                * S1
            )
            + chidiff1
            * eta5
            * (199.6863414922219 + 53.36849263931051 * S1 + 7.650565415855383 * S2)
        )
        int2 = (
            delta
            * eta1
            * (
                17.42562079069636
                - 28.970875603981295 * eta1
                + 50.726220750178435 * eta2
            )
            + chidiff1
            * delta
            * eta1
            * (
                -7.861956897615623 * eta1
                + 93.45476935080045 * eta2
                - 273.1170921735085 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                -0.3265505633310564 * eta1
                - 9.861644053348053 * eta2
                + 60.38649425562178 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (234.13476431269862 + 51.2153901931183 * S1 - 10.05114600643587 * S2)
            + delta
            * eta1
            * S1
            * (
                0.3104472390387834
                * (
                    6.073591341439855
                    + 169.85423386969634 * eta1
                    - 4964.199967099143 * eta2
                    + 42566.59565666228 * eta3
                    - 154255.3408672655 * eta4
                    + 205525.13910847943 * eta5
                )
                + 0.2295327944679772
                * (
                    19.236275867648594
                    - 354.7914372697625 * eta1
                    + 1876.408148917458 * eta2
                    + 2404.4151687877525 * eta3
                    - 41567.07396803811 * eta4
                    + 79210.33893514868 * eta5
                )
                * S1
                + 0.30983324991828787
                * (
                    11.302200127272357
                    - 719.9854052004307 * eta1
                    + 13278.047199998868 * eta2
                    - 104863.50453518033 * eta3
                    + 376409.2335857397 * eta4
                    - 504089.07690692553 * eta5
                )
                * S2
            )
        )
        int3 = (
            delta
            * eta1
            * (
                14.555522136327964
                - 12.799844096694798 * eta1
                + 16.79500349318081 * eta2
            )
            + chidiff1
            * delta
            * eta1
            * (
                -16.292654447108134 * eta1
                + 190.3516012682791 * eta2
                - 562.0936797781519 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                -7.048898856045782 * eta1
                + 49.941617405768135 * eta2
                - 73.62033985436068 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (263.5151703818307 + 44.408527093031566 * S1 + 10.457035444964653 * S2)
            + delta
            * eta1
            * S1
            * (
                0.4590550434774332
                * (
                    3.0594364612798635
                    + 207.74562213604057 * eta1
                    - 5545.0086137386525 * eta2
                    + 50003.94075934942 * eta3
                    - 195187.55422847517 * eta4
                    + 282064.174913521 * eta5
                )
                + 0.657748992123043
                * (
                    5.57939137343977
                    - 124.06189543062042 * eta1
                    + 1276.6209573025596 * eta2
                    - 6999.7659193505915 * eta3
                    + 19714.675715229736 * eta4
                    - 20879.999628681435 * eta5
                )
                * S1
                + 0.3695850566805098
                * (
                    6.077183107132255
                    - 498.95526910874986 * eta1
                    + 10426.348944657859 * eta2
                    - 91096.64982858274 * eta3
                    + 360950.6686625352 * eta4
                    - 534437.8832860565 * eta5
                )
                * S2
            )
        )
        int4 = (
            delta
            * eta1
            * (
                13.312095699772305
                - 7.449975618083432 * eta1
                + 17.098576301150125 * eta2
            )
            + delta
            * eta1
            * (
                chidiff1
                * (
                    -31.171150896110156 * eta1
                    + 371.1389274783572 * eta2
                    - 1103.1917047361735 * eta3
                )
                + chidiff2
                * (
                    32.78644599730888 * eta1
                    - 395.15713118955387 * eta2
                    + 1164.9282236341376 * eta3
                )
            )
            + chidiff1
            * delta
            * eta1
            * (
                -46.85669289852532 * eta1
                + 522.3965959942979 * eta2
                - 1485.5134187612182 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (
                287.90444670305715
                - 21.102665129433042 * chidiff2
                + 7.635582066682054 * S1
                - 29.471275170013012 * S2
            )
            + delta
            * eta1
            * S1
            * (
                0.6893003654021495
                * (
                    3.1014226377197027
                    - 44.83989278653052 * eta1
                    + 565.3767256471909 * eta2
                    - 4797.429130246123 * eta3
                    + 19514.812242035154 * eta4
                    - 27679.226582207506 * eta5
                )
                + 0.7068016563068026
                * (
                    4.071212304920691
                    - 118.51094098279343 * eta1
                    + 1788.1730303291356 * eta2
                    - 13485.270489656365 * eta3
                    + 48603.96661003743 * eta4
                    - 65658.74746265226 * eta5
                )
                * S1
                + 0.2181399561677432
                * (
                    -1.6754158383043574
                    + 303.9394443302189 * eta1
                    - 6857.936471898544 * eta2
                    + 59288.71069769708 * eta3
                    - 216137.90827404748 * eta4
                    + 277256.38289831823 * eta5
                )
                * S2
            )
        )
    elif modeTag == 32:
        S1 = chiPN
        int1 = (
            (
                chidiff2
                * (
                    -0.2341404256829785 * eta1
                    + 2.606326837996192 * eta2
                    - 8.68296921440857 * eta3
                )
                + chidiff1
                * delta
                * (
                    0.5454562486736877 * eta1
                    - 25.19759222940851 * eta2
                    + 73.40268975811729 * eta3
                )
            )
            * sqroot
            + chidiff1
            * delta
            * (
                0.4422257616009941 * eta1
                - 8.490112284851655 * eta2
                + 32.22238925527844 * eta3
            )
            * S1
            * sqroot
            + S1
            * (
                0.7067243321652764
                * (
                    0.12885110296881636
                    + 9.608999847549535 * eta1
                    - 85.46581740280585 * eta2
                    + 325.71940024255775 * eta3
                    + 175.4194342269804 * eta4
                    - 1929.9084724384807 * eta5
                )
                + 0.1540566313813899
                * (
                    -0.3261041495083288
                    + 45.55785402900492 * eta1
                    - 827.591235943271 * eta2
                    + 7184.647314370326 * eta3
                    - 28804.241518798244 * eta4
                    + 43309.69769878964 * eta5
                )
                * S1
            )
            * sqroot
            + (
                480.0434256230109 * eta1
                + 25346.341240810478 * eta2
                - 99873.4707358776 * eta3
                + 106683.98302194536 * eta4
            )
            * sqroot
            * ((1 + 1082.6574834474493 * eta1 + 10083.297670051445 * eta2) ** (-1))
        )
        S1 = STotR
        int2 = (
            eta1
            * (
                chidiff2
                * (
                    -4.175680729484314 * eta1
                    + 47.54281549129226 * eta2
                    - 128.88334273588077 * eta3
                )
                + chidiff1
                * delta
                * (
                    -0.18274358639599947 * eta1
                    - 71.01128541687838 * eta2
                    + 208.07105580635888 * eta3
                )
            )
            + eta1
            * (
                4.760999387359598
                - 38.57900689641654 * eta1
                + 456.2188780552874 * eta2
                - 4544.076411013166 * eta3
                + 24956.9592553473 * eta4
                - 69430.10468748478 * eta5
                + 77839.74180254337 * eta6
            )
            + chidiff1
            * delta
            * eta1
            * (
                1.2198776533959694 * eta1
                - 26.816651899746475 * eta2
                + 68.72798751937934 * eta3
            )
            * S1
            + eta1
            * S1
            * (
                1.5098291294292217
                * (
                    0.4844667556328104
                    + 9.848766999273414 * eta1
                    - 143.66427232396376 * eta2
                    + 856.9917885742416 * eta3
                    - 1633.3295758142904 * eta4
                )
                + 0.32413108737204144
                * (
                    2.835358206961064
                    - 62.37317183581803 * eta1
                    + 761.6103793011912 * eta2
                    - 3811.5047139343505 * eta3
                    + 6660.304740652403 * eta4
                )
                * S1
            )
        )
        S1 = chiPN
        S2 = S1 * S1
        int3 = (
            3.881450518842405 * eta1
            - 12.580316392558837 * eta2
            + 1.7262466525848588 * eta3
            + chidiff2
            * (
                -7.065118823041031 * eta2
                + 77.97950589523865 * eta3
                - 203.65975422378446 * eta4
            )
            - 58.408542930248046 * eta4
            + chidiff1
            * delta
            * (
                1.924723094787216 * eta2
                - 90.92716917757797 * eta3
                + 387.00162600306226 * eta4
            )
            + 403.5748987560612 * eta5
            + chidiff1
            * delta
            * (
                -0.2566958540737833 * eta2
                + 14.488550203412675 * eta3
                - 26.46699529970884 * eta4
            )
            * S1
            + S1
            * (
                0.3650871458400108
                * (
                    71.57390929624825 * eta2
                    - 994.5272351916166 * eta3
                    + 6734.058809060536 * eta4
                    - 18580.859291282686 * eta5
                    + 16001.318492586077 * eta6
                )
                + 0.0960146077440495
                * (
                    451.74917589707513 * eta2
                    - 9719.470997418284 * eta3
                    + 83403.5743434538 * eta4
                    - 318877.43061174755 * eta5
                    + 451546.88775684836 * eta6
                )
                * S1
                - 0.03985156529181297
                * (
                    -304.92981902871617 * eta2
                    + 3614.518459296278 * eta3
                    - 7859.4784979916085 * eta4
                    - 46454.57664737511 * eta5
                    + 162398.81483375572 * eta6
                )
                * S2
            )
        )
        S1 = STotR
        int4 = (
            eta1
            * (
                chidiff2
                * (
                    -8.572797326909152 * eta1
                    + 92.95723645687826 * eta2
                    - 236.2438921965621 * eta3
                )
                + chidiff1
                * delta
                * (
                    6.674358856924571 * eta1
                    - 171.4826985994883 * eta2
                    + 645.2760206304703 * eta3
                )
            )
            + eta1
            * (
                3.921660532875504
                - 16.57299637423352 * eta1
                + 25.254017911686333 * eta2
                - 143.41033155133266 * eta3
                + 692.926425981414 * eta4
            )
            + chidiff1
            * delta
            * eta1
            * (
                -3.582040878719185 * eta1
                + 57.75888914133383 * eta2
                - 144.21651114700492 * eta3
            )
            * S1
            + eta1
            * S1
            * (
                1.242750265695504
                * (
                    -0.522172424518215
                    + 25.168480118950065 * eta1
                    - 303.5223688400309 * eta2
                    + 1858.1518762309654 * eta3
                    - 3797.3561904195085 * eta4
                )
                + 0.2927045241764365
                * (
                    0.5056957789079993
                    - 15.488754837330958 * eta1
                    + 471.64047356915603 * eta2
                    - 3131.5783196211587 * eta3
                    + 6097.887891566872 * eta4
                )
                * S1
            )
        )
    else:
        S1 = chiPN
        S2 = S1 * S1
        int1 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    1.5378890240544967 * eta1
                    - 3.4499418893734903 * eta2
                    + 16.879953490422782 * eta3
                )
                + chidiff2
                * (
                    1.720226708214248 * eta1
                    - 11.87925165364241 * eta2
                    + 23.259283336239545 * eta3
                )
            )
            + eta1
            * (
                8.790173464969538
                - 64.95499142822892 * eta1
                + 324.1998823562892 * eta2
                - 1111.9864921907126 * eta3
                + 1575.602443847111 * eta4
            )
            + eta1
            * S1
            * (
                -0.062333275821238224
                * (
                    -21.630297087123807
                    + 137.4395894877131 * eta1
                    + 64.92115530780129 * eta2
                    - 1013.1110639471394 * eta3
                )
                - 0.11014697070998722
                * (
                    4.149721483857751
                    - 108.6912882442823 * eta1
                    + 831.6073263887092 * eta2
                    - 1828.2527520190122 * eta3
                )
                * S1
                - 0.07704777584463054
                * (
                    4.581767671445529
                    - 50.35070009227704 * eta1
                    + 344.9177692251726 * eta2
                    - 858.9168637051405 * eta3
                )
                * S2
            )
        )
        int2 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    2.3123974306694057 * eta1
                    - 12.237594841284904 * eta2
                    + 44.78225529547671 * eta3
                )
                + chidiff2
                * (
                    2.9282931698944292 * eta1
                    - 25.624210264341933 * eta2
                    + 61.05270871360041 * eta3
                )
            )
            + eta1
            * (
                6.98072197826729
                - 46.81443520117986 * eta1
                + 236.76146303619544 * eta2
                - 920.358408667518 * eta3
                + 1478.050456337336 * eta4
            )
            + eta1
            * S1
            * (
                -0.07801583359561987
                * (
                    -28.29972282146242
                    + 752.1603553640072 * eta1
                    - 10671.072606753183 * eta2
                    + 83447.0461509547 * eta3
                    - 350025.2112501252 * eta4
                    + 760889.6919776166 * eta5
                    - 702172.2934567826 * eta6
                )
                + 0.013159545629626014
                * (
                    91.1469833190294
                    - 3557.5003799977294 * eta1
                    + 52391.684517955284 * eta2
                    - 344254.9973814295 * eta3
                    + 1.0141877915334814e6 * eta4
                    - 1.1505186449682908e6 * eta5
                    + 268756.85659532435 * eta6
                )
                * S1
            )
        )
        int3 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    -0.8765502142143329 * eta1
                    + 22.806632458441996 * eta2
                    - 43.675503209991184 * eta3
                )
                + chidiff2
                * (
                    0.48698617426180074 * eta1
                    - 4.302527065360426 * eta2
                    + 16.18571810759235 * eta3
                )
            )
            + eta1
            * (
                6.379772583015967
                - 44.10631039734796 * eta1
                + 269.44092930942793 * eta2
                - 1285.7635006711453 * eta3
                + 2379.538739132234 * eta4
            )
            + eta1
            * S1
            * (
                -0.23316184683282615
                * (
                    -1.7279023138971559
                    - 23.606399143993716 * eta1
                    + 409.3387618483284 * eta2
                    - 1115.4147472977265 * eta3
                )
                - 0.09653777612560172
                * (
                    -5.310643306559746
                    - 2.1852511802701264 * eta1
                    + 541.1248219096527 * eta2
                    - 1815.7529908827103 * eta3
                )
                * S1
                - 0.060477799540741804
                * (
                    -14.578189130145661
                    + 175.6116682068523 * eta1
                    - 569.4799973930861 * eta2
                    + 426.0861915646515 * eta3
                )
                * S2
            )
        )
        int4 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    -2.461738962276138 * eta1
                    + 45.3240543970684 * eta2
                    - 112.2714974622516 * eta3
                )
                + chidiff2
                * (
                    0.9158352037567031 * eta1
                    - 8.724582331021695 * eta2
                    + 28.44633544874233 * eta3
                )
            )
            + eta1
            * (
                6.098676337298138
                - 45.42463610529546 * eta1
                + 350.97192927929433 * eta2
                - 2002.2013283876834 * eta3
                + 4067.1685640401033 * eta4
            )
            + eta1
            * S1
            * (
                -0.36068516166901304
                * (
                    -2.120354236840677
                    - 47.56175350408845 * eta1
                    + 1618.4222330016048 * eta2
                    - 14925.514654896673 * eta3
                    + 60287.45399959349 * eta4
                    - 91269.3745059139 * eta5
                )
                - 0.09635801207669747
                * (
                    -11.824692837267394
                    + 371.7551657959369 * eta1
                    - 4176.398139238679 * eta2
                    + 16655.87939259747 * eta3
                    - 4102.218189945819 * eta4
                    - 67024.98285179552 * eta5
                )
                * S1
                - 0.06565232123453196
                * (
                    -26.15227471380236
                    + 1869.0168486099005 * eta1
                    - 33951.35186039629 * eta2
                    + 253694.6032002248 * eta3
                    - 845341.6001856657 * eta4
                    + 1.0442282862506858e6 * eta5
                )
                * S2
            )
        )

    return int1, int2, int3, int4


def _xhm_rd_amp_fit_coeffs(
    modeTag: int, eta: float, STotR: float, dchi_half: float, delta: float, chiPN: float
) -> tuple:
    """122022 ringdown fit coefficients (alambda, lambda, sigma)."""
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = STotR
    S2 = S1 * S1
    chidiff1 = dchi_half
    chidiff2 = chidiff1 * chidiff1

    if modeTag == 21:
        S1 = chiPN
        S2 = S1 * S1
        alambda = jnp.abs(
            delta * (0.24548180919287976 - 0.25565119457386487 * eta1) * eta1
            + chidiff1
            * delta
            * eta1
            * (
                0.5670798742968471 * eta1
                - 14.276514548218454 * eta2
                + 45.014547333879136 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                0.4580805242442763 * eta1
                - 4.859294663135058 * eta2
                + 14.995447609839573 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (-27.031582936285528 + 6.468164760468401 * S1 + 0.34222101136488015 * S2)
            + delta
            * eta1
            * S1
            * (
                -0.2204878224611389
                * (
                    1.0730799832007898
                    - 3.44643820338605 * eta1
                    + 32.520429274459836 * eta2
                    - 83.21097158567372 * eta3
                )
                + 0.008901444811471891
                * (
                    -5.876973170072921
                    + 120.70115519895002 * eta1
                    - 916.5281661566283 * eta2
                    + 2306.8425350489847 * eta3
                )
                * S1
                + 0.015541783867953005
                * (
                    2.4780170686140455
                    + 17.377013149762398 * eta1
                    - 380.91157168170236 * eta2
                    + 1227.5332509075172 * eta3
                )
                * S2
            )
        )
        lambda_ = jnp.abs(
            1.0092933052569775
            - 0.2791855444800297 * eta1
            + 1.7110615047319937 * eta2
            + chidiff1
            * (
                -0.1054835719277311 * eta1
                + 7.506083919925026 * eta2
                - 30.1595680078279 * eta3
            )
            + chidiff1
            * (
                2.078267611384239 * eta1
                - 10.166026002515457 * eta2
                - 1.2091616330625208 * eta3
            )
            * S1
            + S1
            * (
                0.17250873250247642
                * (
                    1.0170226856985174
                    + 1.0395650952176598 * eta1
                    - 35.73623734051525 * eta2
                    + 403.68074286921444 * eta3
                    - 1194.6152711219886 * eta4
                )
                + 0.06850746964805364
                * (
                    1.507796537056924
                    + 37.81075363806507 * eta1
                    - 863.117144661059 * eta2
                    + 6429.543634627373 * eta3
                    - 15108.557419182316 * eta4
                )
                * S1
            )
        )
        S1 = STotR
        sigma = jnp.abs(
            1.374451177213076
            - 0.1147381625630186 * eta1
            + chidiff1
            * (
                0.6646459256372743 * eta1
                - 5.020585319906719 * eta2
                + 9.817281653770431 * eta3
            )
            + chidiff1
            * (
                3.8734254747587973 * eta1
                - 39.880716190740465 * eta2
                + 99.05511583518896 * eta3
            )
            * S1
            + S1
            * (
                0.013272603498067647
                * (
                    1.809972721953344
                    - 12.560287006325837 * eta1
                    - 134.597005438578 * eta2
                    + 786.2235720637008 * eta3
                )
                + 0.006850483944311038
                * (
                    -6.478737679813189
                    - 200.29813775611166 * eta1
                    + 2744.3629484255357 * eta2
                    - 7612.096007280672 * eta3
                )
                * S1
            )
        )
    elif modeTag == 32:
        S1 = STotR
        S2 = S1 * S1
        alambda = (
            chidiff2
            * (
                -3.4614418482110163 * eta3
                + 35.464117772624164 * eta4
                - 85.19723511005235 * eta5
            )
            + chidiff1
            * delta
            * (
                2.0328561081997463 * eta3
                - 46.18751757691501 * eta4
                + 170.9266105597438 * eta5
            )
            + chidiff2
            * (
                -0.4600401291210382 * eta3
                + 12.23450117663151 * eta4
                - 42.74689906831975 * eta5
            )
            * S1
            + chidiff1
            * delta
            * (
                5.786292428422767 * eta3
                - 53.60467819078566 * eta4
                + 117.66195692191727 * eta5
            )
            * S1
            + S1
            * (
                -0.0013330716557843666
                * (
                    56.35538385647113 * eta1
                    - 1218.1550992423377 * eta2
                    + 16509.69605686402 * eta3
                    - 102969.88022112886 * eta4
                    + 252228.94931931415 * eta5
                    - 150504.2927996263 * eta6
                )
                + 0.0010126460331462495
                * (
                    -33.87083889060834 * eta1
                    + 502.6221651850776 * eta2
                    - 1304.9210590188136 * eta3
                    - 36980.079328277505 * eta4
                    + 295469.28617550555 * eta5
                    - 597155.7619486618 * eta6
                )
                * S1
                - 0.00043088431510840695
                * (
                    -30.014415072587354 * eta1
                    - 1900.5495690280086 * eta2
                    + 76517.21042363928 * eta3
                    - 870035.1394696251 * eta4
                    + 3.9072674134789007e6 * eta5
                    - 6.094089675611567e6 * eta6
                )
                * S2
            )
            + (
                0.08408469319155859 * eta1
                - 1.223794846617597 * eta2
                + 6.5972460654253515 * eta3
                - 15.707327897569396 * eta4
                + 14.163264397061505 * eta5
            )
            * ((1 - 8.612447115134758 * eta1 + 18.93655612952139 * eta2) ** (-1))
        )
        lambda_ = (
            0.978510781593996
            + 0.36457571743142897 * eta1
            - 12.259851752618998 * eta2
            + 49.19719473681921 * eta3
            + chidiff1
            * delta
            * (
                -188.37119473865533 * eta3
                + 2151.8731700399308 * eta4
                - 6328.182823770599 * eta5
            )
            + chidiff2
            * (
                115.3689949926392 * eta3
                - 1159.8596972989067 * eta4
                + 2657.6998831179444 * eta5
            )
            + S1
            * (
                0.22358643406992756
                * (
                    0.48943645614341924
                    - 32.06682257944444 * eta1
                    + 365.2485484044132 * eta2
                    - 915.2489655397206 * eta3
                )
                + 0.0792473022309144
                * (
                    1.877251717679991
                    - 103.65639889587327 * eta1
                    + 1202.174780792418 * eta2
                    - 3206.340850767219 * eta3
                )
                * S1
            )
        )
        sigma = (
            1.3353917551819414
            + 0.13401718687342024 * eta1
            + chidiff1
            * delta
            * (
                144.37065005786636 * eta3
                - 754.4085447486738 * eta4
                + 123.86194078913776 * eta5
            )
            + chidiff2
            * (
                209.09202210427972 * eta3
                - 1769.4658099037918 * eta4
                + 3592.287297392387 * eta5
            )
            + S1
            * (
                -0.012086025709597246
                * (
                    -6.230497473791485
                    + 600.5968613752918 * eta1
                    - 6606.1009717965735 * eta2
                    + 17277.60594350428 * eta3
                )
                - 0.06066548829900489
                * (
                    -0.9208054306316676
                    + 142.0346574366267 * eta1
                    - 1567.249168668069 * eta2
                    + 4119.373703246675 * eta3
                )
                * S1
            )
        )
    else:
        alambda = 0.0
        lambda_ = 0.0
        sigma = 1.33

    return alambda, lambda_, sigma


def _xhm_rd_amp_colloc_pts(
    modeTag: int, eta: float, STotR: float, dchi_half: float, delta: float, chiPN: float
) -> tuple:
    """122022 ringdown amplitude collocation-point fits (rdcp1, rdcp2, rdcp3)."""
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = STotR
    S2 = S1 * S1
    chidiff1 = dchi_half
    chidiff2 = chidiff1 * chidiff1

    if modeTag == 21:
        S1 = chiPN
        S2 = S1 * S1
        rdcp1 = jnp.abs(
            delta
            * eta1
            * (
                12.880905080761432
                - 23.5291063016996 * eta1
                + 92.6090002736012 * eta2
                - 175.16681482428694 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                26.89427230731867 * eta1
                - 710.8871223808559 * eta2
                + 2255.040486907459 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                21.402708785047853 * eta1
                - 232.07306353130417 * eta2
                + 591.1097623278739 * eta3
            )
            * S1
            + delta
            * eta1
            * S1
            * (
                -10.090867481062709
                * (
                    0.9580746052260011
                    + 5.388149112485179 * eta1
                    - 107.22993216128548 * eta2
                    + 801.3948756800821 * eta3
                    - 2688.211889175019 * eta4
                    + 3950.7894052628735 * eta5
                    - 1992.9074348833092 * eta6
                )
                - 0.42972412296628143
                * (
                    1.9193131231064235
                    + 139.73149069609775 * eta1
                    - 1616.9974609915555 * eta2
                    - 3176.4950303461164 * eta3
                    + 107980.65459735804 * eta4
                    - 479649.75188253267 * eta5
                    + 658866.0983367155 * eta6
                )
                * S1
            )
            + chidiff1
            * eta5
            * (-1512.439342647443 + 175.59081294852444 * S1 + 10.13490934572329 * S2)
        )
        rdcp2 = jnp.abs(
            delta * (9.112452928978168 - 7.5304766811877455 * eta1) * eta1
            + chidiff1
            * delta
            * eta1
            * (
                16.236533863306132 * eta1
                - 500.11964987628926 * eta2
                + 1618.0818430353293 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                2.7866868976718226 * eta1
                - 0.4210629980868266 * eta2
                - 20.274691328125606 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (-1116.4039232324135 + 245.73200219767514 * S1 + 21.159179960295855 * S2)
            + delta
            * eta1
            * S1
            * (
                -8.236485576091717
                * (
                    0.8917610178208336
                    + 5.1501231412520285 * eta1
                    - 87.05136337926156 * eta2
                    + 519.0146702141192 * eta3
                    - 997.6961311502365 * eta4
                )
                + 0.2836840678615208
                * (
                    -0.19281297100324718
                    - 57.65586769647737 * eta1
                    + 586.7942442434971 * eta2
                    - 1882.2040277496196 * eta3
                    + 2330.3534917059906 * eta4
                )
                * S1
                + 0.40226131643223145
                * (
                    -3.834742668014861
                    + 190.42214703482531 * eta1
                    - 2885.5110686004946 * eta2
                    + 16087.433824017446 * eta3
                    - 29331.524552164105 * eta4
                )
                * S2
            )
        )
        rdcp3 = jnp.abs(
            delta * (2.920930733198033 - 3.038523690239521 * eta1) * eta1
            + chidiff1
            * delta
            * eta1
            * (
                6.3472251472354975 * eta1
                - 171.23657247338042 * eta2
                + 544.1978232314333 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                1.9701247529688362 * eta1
                - 2.8616711550845575 * eta2
                - 0.7347258030219584 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (-334.0969956136684 + 92.91301644484749 * S1 - 5.353399481074393 * S2)
            + delta
            * eta1
            * S1
            * (
                -2.7294297839371824
                * (
                    1.148166706456899
                    - 4.384077347340523 * eta1
                    + 36.120093043420326 * eta2
                    - 87.26454353763077 * eta3
                )
                + 0.23949142867803436
                * (
                    -0.6931516433988293
                    + 33.33372867559165 * eta1
                    - 307.3404155231787 * eta2
                    + 862.3123076782916 * eta3
                )
                * S1
                + 0.1930861073906724
                * (
                    3.7735099269174106
                    - 19.11543562444476 * eta1
                    - 78.07256429516346 * eta2
                    + 485.67801863289293 * eta3
                )
                * S2
            )
        )
    elif modeTag == 33:
        rdcp1 = (
            delta
            * eta1
            * (
                12.439702602599235
                - 4.436329538596615 * eta1
                + 22.780673360839497 * eta2
            )
            + delta
            * eta1
            * (
                chidiff1
                * (
                    -41.04442169938298 * eta1
                    + 502.9246970179746 * eta2
                    - 1524.2981907688634 * eta3
                )
                + chidiff2
                * (
                    32.23960072974939 * eta1
                    - 365.1526474476759 * eta2
                    + 1020.6734178547847 * eta3
                )
            )
            + chidiff1
            * delta
            * eta1
            * (
                -52.85961155799673 * eta1
                + 577.6347407795782 * eta2
                - 1653.496174539196 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (
                257.33227387984863
                - 34.5074027042393 * chidiff2
                - 21.836905132600755 * S1
                - 15.81624534976308 * S2
            )
            + 13.499999999999998
            * delta
            * eta1
            * S1
            * (
                -0.13654149379906394
                * (
                    2.719687834084113
                    + 29.023992126142304 * eta1
                    - 742.1357702210267 * eta2
                    + 4142.974510926698 * eta3
                    - 6167.08766058184 * eta4
                    - 3591.1757995710486 * eta5
                )
                - 0.06248535354306988
                * (
                    6.697567446351289
                    - 78.23231700361792 * eta1
                    + 444.79350113344543 * eta2
                    - 1907.008984765889 * eta3
                    + 6601.918552659412 * eta4
                    - 10056.98422430965 * eta5
                )
                * S1
            )
            * ((-3.9329308614837704 + S1) ** (-1))
        )
        rdcp2 = (
            delta * eta1 * (8.425057692276933 + 4.543696144846763 * eta1)
            + chidiff1
            * delta
            * eta1
            * (
                -32.18860840414171 * eta1
                + 412.07321398189293 * eta2
                - 1293.422289802462 * eta3
            )
            + chidiff1
            * delta
            * eta1
            * (
                -17.18006888428382 * eta1
                + 190.73514518113845 * eta2
                - 636.4802385540647 * eta3
            )
            * S1
            + delta
            * eta1
            * S1
            * (
                0.1206817303851239
                * (
                    8.667503604073314
                    - 144.08062755162752 * eta1
                    + 3188.189172446398 * eta2
                    - 35378.156133055556 * eta3
                    + 163644.2192178668 * eta4
                    - 265581.70142471837 * eta5
                )
                + 0.08028332044013944
                * (
                    12.632478544060636
                    - 322.95832000179297 * eta1
                    + 4777.45310151897 * eta2
                    - 35625.58409457366 * eta3
                    + 121293.97832549023 * eta4
                    - 148782.33687815256 * eta5
                )
                * S1
            )
            + chidiff1
            * eta5
            * (
                159.72371180117415
                - 29.10412708633528 * chidiff2
                - 1.873799747678187 * S1
                + 41.321480132899524 * S2
            )
        )
        rdcp3 = (
            delta * eta1 * (2.485784720088995 + 2.321696430921996 * eta1)
            + delta
            * eta1
            * (
                chidiff1
                * (
                    -10.454376404653859 * eta1
                    + 147.10344302665484 * eta2
                    - 496.1564538739011 * eta3
                )
                + chidiff2
                * (
                    -5.9236399792925996 * eta1
                    + 65.86115501723127 * eta2
                    - 197.51205149250532 * eta3
                )
            )
            + chidiff1
            * delta
            * eta1
            * (
                -10.27418232676514 * eta1
                + 136.5150165348149 * eta2
                - 473.30988537734174 * eta3
            )
            * S1
            + chidiff1
            * eta5
            * (
                32.07819766300362
                - 3.071422453072518 * chidiff2
                + 35.09131921815571 * S1
                + 67.23189816732847 * S2
            )
            + 13.499999999999998
            * delta
            * eta1
            * S1
            * (
                0.0011484326782460882
                * (
                    4.1815722950796035
                    - 172.58816646768219 * eta1
                    + 5709.239330076732 * eta2
                    - 67368.27397765424 * eta3
                    + 316864.0589150127 * eta4
                    - 517034.11171277676 * eta5
                )
                - 0.009496797093329243
                * (
                    0.9233282181397624
                    - 118.35865186626413 * eta1
                    + 2628.6024206791726 * eta2
                    - 23464.64953722729 * eta3
                    + 94309.57566199072 * eta4
                    - 140089.40725211444 * eta5
                )
                * S1
            )
            * ((0.09549360183532198 - 0.41099904730526465 * S1 + S2) ** (-1))
        )
    elif modeTag == 32:
        rdcp1 = (
            chidiff2
            * (
                -261.63903838092017 * eta3
                + 2482.4929818200458 * eta4
                - 5662.765952006266 * eta5
            )
            + chidiff1
            * delta
            * (
                200.3023530582654 * eta3
                - 3383.07742098347 * eta4
                + 11417.842708417566 * eta5
            )
            + chidiff2
            * (
                -177.2481070662751 * eta3
                + 1820.8637746828358 * eta4
                - 4448.151940319403 * eta5
            )
            * S1
            + chidiff1
            * delta
            * (
                412.749304734278 * eta3
                - 4156.641392955615 * eta4
                + 10116.974216563232 * eta5
            )
            * S1
            + S1
            * (
                -0.07383539239633188
                * (
                    40.59996146686051 * eta1
                    - 527.5322650311067 * eta2
                    + 4167.108061823492 * eta3
                    - 13288.883172763119 * eta4
                    - 23800.671572828596 * eta5
                    + 146181.8016013141 * eta6
                )
                + 0.03576631753501686
                * (
                    -13.96758180764024 * eta1
                    - 797.1235306450683 * eta2
                    + 18007.56663810595 * eta3
                    - 151803.40642097822 * eta4
                    + 593811.4596071478 * eta5
                    - 878123.747877138 * eta6
                )
                * S1
                + 0.01007493097350273
                * (
                    -27.77590078264459 * eta1
                    + 4011.1960424049857 * eta2
                    - 152384.01804465035 * eta3
                    + 1.7595145936445233e6 * eta4
                    - 7.889230647117076e6 * eta5
                    + 1.2172078072446395e7 * eta6
                )
                * S2
            )
            + (
                4.146029818148087 * eta1
                - 61.060972560568054 * eta2
                + 336.3725848841942 * eta3
                - 832.785332776221 * eta4
                + 802.5027431944313 * eta5
            )
            * ((1 - 8.662174796705683 * eta1 + 19.288918757536685 * eta2) ** (-1))
        )
        rdcp2 = (
            chidiff2
            * (
                -220.42133216774002 * eta3
                + 2082.031407555522 * eta4
                - 4739.292554291661 * eta5
            )
            + chidiff1
            * delta
            * (
                179.07548162694007 * eta3
                - 2878.2078963030094 * eta4
                + 9497.998559135678 * eta5
            )
            + chidiff2
            * (
                -128.07917402087625 * eta3
                + 1392.4598433465628 * eta4
                - 3546.2644951338134 * eta5
            )
            * S1
            + chidiff1
            * delta
            * (
                384.31792882093424 * eta3
                - 3816.5687272960417 * eta4
                + 9235.479593415908 * eta5
            )
            * S1
            + S1
            * (
                -0.06144774696295017
                * (
                    35.72693522898656 * eta1
                    - 168.08433700852038 * eta2
                    - 3010.678442066521 * eta3
                    + 45110.034521934074 * eta4
                    - 231569.4154711447 * eta5
                    + 414234.84895584086 * eta6
                )
                + 0.03663881822701642
                * (
                    -22.057692852225696 * eta1
                    + 223.9912685075838 * eta2
                    - 1028.5261783449762 * eta3
                    - 12761.957255385 * eta4
                    + 141784.13567610556 * eta5
                    - 328718.5349981628 * eta6
                )
                * S1
                + 0.004849853669413881
                * (
                    -90.35491669965123 * eta1
                    + 19286.158446325957 * eta2
                    - 528138.5557827373 * eta3
                    + 5.175061086459432e6 * eta4
                    - 2.1142182400264673e7 * eta5
                    + 3.0737963347449116e7 * eta6
                )
                * S2
            )
            + (
                3.133378729082171 * eta1
                - 45.83572706555282 * eta2
                + 250.23275606463622 * eta3
                - 612.0498767005383 * eta4
                + 580.3574091493459 * eta5
            )
            * ((1 - 8.698032720488515 * eta1 + 19.38621948411302 * eta2) ** (-1))
        )
        rdcp3 = (
            chidiff2
            * (
                -79.14146757219045 * eta3
                + 748.8207876524461 * eta4
                - 1712.3401586150026 * eta5
            )
            + chidiff1
            * delta
            * (
                65.1786095079065 * eta3
                - 996.4553252426255 * eta4
                + 3206.5675278160684 * eta5
            )
            + chidiff2
            * (
                -36.474455088940225 * eta3
                + 421.8842792746865 * eta4
                - 1117.0227933265749 * eta5
            )
            * S1
            + chidiff1
            * delta
            * (
                169.07368933925878 * eta3
                - 1675.2562326502878 * eta4
                + 4040.0077967763787 * eta5
            )
            * S1
            + S1
            * (
                -0.01992370601225598
                * (
                    36.307098892574196 * eta1
                    - 846.997262853445 * eta2
                    + 16033.60939445582 * eta3
                    - 138800.53021166887 * eta4
                    + 507922.88946543116 * eta5
                    - 647376.1499824544 * eta6
                )
                + 0.014207919520826501
                * (
                    -33.80287899746716 * eta1
                    + 1662.2913368534057 * eta2
                    - 31688.885017467597 * eta3
                    + 242813.43893659746 * eta4
                    - 793178.4767168422 * eta5
                    + 929016.897093022 * eta6
                )
                * S1
            )
            + (
                0.9641853854287679 * eta1
                - 13.801372413989519 * eta2
                + 72.80610853168994 * eta3
                - 168.65551450831953 * eta4
                + 147.2372582604103 * eta5
            )
            * ((1 - 8.65963828355163 * eta1 + 19.112920222001367 * eta2) ** (-1))
        )
    else:
        rdcp1 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    -8.51952446214978 * eta1
                    + 117.76530248141987 * eta2
                    - 297.2592736781142 * eta3
                )
                + chidiff2
                * (
                    -0.2750098647982238 * eta1
                    + 4.456900599347149 * eta2
                    - 8.017569928870929 * eta3
                )
            )
            + eta1
            * (
                5.635069974807398
                - 33.67252878543393 * eta1
                + 287.9418482197136 * eta2
                - 3514.3385364216438 * eta3
                + 25108.811524802128 * eta4
                - 98374.18361532023 * eta5
                + 158292.58792484726 * eta6
            )
            + eta1
            * S1
            * (
                -0.4360849737360132
                * (
                    -0.9543114627170375
                    - 58.70494649755802 * eta1
                    + 1729.1839588870455 * eta2
                    - 16718.425586396803 * eta3
                    + 71236.86532610047 * eta4
                    - 111910.71267453219 * eta5
                )
                - 0.024861802943501172
                * (
                    -52.25045490410733
                    + 1585.462602954658 * eta1
                    - 15866.093368857853 * eta2
                    + 35332.328181283 * eta3
                    + 168937.32229060197 * eta4
                    - 581776.5303770923 * eta5
                )
                * S1
                + 0.005856387555754387
                * (
                    186.39698091707513
                    - 9560.410655118145 * eta1
                    + 156431.3764198244 * eta2
                    - 1.0461268207440731e6 * eta3
                    + 3.054333578686424e6 * eta4
                    - 3.2369858387064277e6 * eta5
                )
                * S2
            )
        )
        rdcp2 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    -2.861653255976984 * eta1
                    + 50.50227103211222 * eta2
                    - 123.94152825700999 * eta3
                )
                + chidiff2
                * (
                    2.9415751419018865 * eta1
                    - 28.79779545444817 * eta2
                    + 72.40230240887851 * eta3
                )
            )
            + eta1
            * (
                3.2461722686239307
                + 25.15310593958783 * eta1
                - 792.0167314124681 * eta2
                + 7168.843978909433 * eta3
                - 30595.4993786313 * eta4
                + 49148.57065911245 * eta5
            )
            + eta1
            * S1
            * (
                -0.23311779185707152
                * (
                    -1.0795711755430002
                    - 20.12558747513885 * eta1
                    + 1163.9107546486134 * eta2
                    - 14672.23221502075 * eta3
                    + 73397.72190288734 * eta4
                    - 127148.27131388368 * eta5
                )
                + 0.025805905356653
                * (
                    11.929946153728276
                    + 350.93274421955806 * eta1
                    - 14580.02701600596 * eta2
                    + 174164.91607515427 * eta3
                    - 819148.9390278616 * eta4
                    + 1.3238624538095295e6 * eta5
                )
                * S1
                + 0.019740635678180102
                * (
                    -7.046295936301379
                    + 1535.781942095697 * eta1
                    - 27212.67022616794 * eta2
                    + 201981.0743810629 * eta3
                    - 696891.1349708183 * eta4
                    + 910729.0219043035 * eta5
                )
                * S2
            )
        )
        rdcp3 = (
            eta1
            * (
                chidiff1
                * delta
                * (
                    2.4286414692113816 * eta1
                    - 23.213332913737403 * eta2
                    + 66.58241012629095 * eta3
                )
                + chidiff2
                * (
                    3.085167288859442 * eta1
                    - 31.60440418701438 * eta2
                    + 78.49621016381445 * eta3
                )
            )
            + eta1
            * (
                0.861883217178703
                + 13.695204704208976 * eta1
                - 337.70598252897696 * eta2
                + 2932.3415281149432 * eta3
                - 12028.786386004691 * eta4
                + 18536.937955014455 * eta5
            )
            + eta1
            * S1
            * (
                -0.048465588779596405
                * (
                    -0.34041762314288154
                    - 81.33156665674845 * eta1
                    + 1744.329802302927 * eta2
                    - 16522.343895064576 * eta3
                    + 76620.18243090731 * eta4
                    - 133340.93723954144 * eta5
                )
                + 0.024804027856323612
                * (
                    -8.666095805675418
                    + 711.8727878341302 * eta1
                    - 13644.988225595187 * eta2
                    + 112832.04975245205 * eta3
                    - 422282.0368440555 * eta4
                    + 584744.0406581408 * eta5
                )
                * S1
            )
        )

    return rdcp1, rdcp2, rdcp3


def _xhm_rd_amp_rdaux_pts(
    eta: float, STotR: float, dchi_half: float, delta: float, chiPN: float
) -> tuple:
    """122022 32-mode RDAux collocation-point fits (rdaux1, rdaux2)."""
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    eta7 = eta1 * eta6
    chidiff1 = dchi_half
    chidiff2 = chidiff1 * chidiff1

    S1 = STotR
    rdaux1 = (
        chidiff2
        * (
            -4.188795724777721 * eta2
            + 53.39200466700963 * eta3
            - 131.19660856923554 * eta4
        )
        + chidiff1
        * delta
        * (
            14.284921364132623 * eta2
            - 321.26423637658746 * eta3
            + 1242.865584938088 * eta4
        )
        + S1
        * (
            -0.022968727462555794
            * (
                83.66854837403105 * eta1
                - 3330.6261333413177 * eta2
                + 77424.12614733395 * eta3
                - 710313.3016672594 * eta4
                + 2.6934917075009225e6 * eta5
                - 3.572465179268999e6 * eta6
            )
            + 0.0014795114305436387
            * (
                -1672.7273629876313 * eta1
                + 90877.38260964208 * eta2
                - 1.6690169155105734e6 * eta3
                + 1.3705532554135624e7 * eta4
                - 5.116110998398143e7 * eta5
                + 7.06066766311127e7 * eta6
            )
            * S1
        )
        + (
            4.45156488896258 * eta1
            - 77.39303992494544 * eta2
            + 522.5070635563092 * eta3
            - 1642.3057499049708 * eta4
            + 2048.333892310575 * eta5
        )
        * ((1 - 9.611489164758915 * eta1 + 24.249594730050312 * eta2) ** (-1))
    )
    S1 = chiPN
    rdaux2 = (
        chidiff2
        * (
            -18.550171209458394 * eta2
            + 188.99161055445936 * eta3
            - 440.26516625611 * eta4
        )
        + chidiff1
        * delta
        * (
            13.132625215315063 * eta2
            - 340.5204040505528 * eta3
            + 1327.1224176812448 * eta4
        )
        + S1
        * (
            -0.16707403272774676
            * (
                6.678916447469937 * eta1
                + 1331.480396625797 * eta2
                - 41908.45179140144 * eta3
                + 520786.0225074669 * eta4
                - 3.1894624909922685e6 * eta5
                + 9.51553823212259e6 * eta6
                - 1.1006903622406831e7 * eta7
            )
            + 0.015205286051218441
            * (
                108.10032279461095 * eta1
                - 16084.215590200103 * eta2
                + 462957.5593513407 * eta3
                - 5.635028227588545e6 * eta4
                + 3.379925277713386e7 * eta5
                - 9.865815275452062e7 * eta6
                + 1.1201307979786257e8 * eta7
            )
            * S1
        )
        + (
            3.902154247490771 * eta1
            - 55.77521071924907 * eta2
            + 294.9496843041973 * eta3
            - 693.6803787318279 * eta4
            + 636.0141528226893 * eta5
        )
        * ((1 - 8.56699762573719 * eta1 + 19.119341007236955 * eta2) ** (-1))
    )
    return rdaux1, rdaux2


def _xhm_fAmpMatchIN(pWFHM: "XHMWaveformStruct", pWF22: dict) -> Array:
    """Inspiral-to-intermediate cutoff frequency (version 122022)."""
    eta = pWF22["eta"]
    chi1 = pWF22["chi1L"]
    fMECO = pWFHM.fMECOlm
    emm = float(pWFHM.emm)

    fcutEMR = (
        1.25
        * emm
        * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * chi1
                + 0.000316707064291237 * chi1**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * eta)
        )
        / (1.2857082764038923 - 0.9977728883419751 * chi1)
    )

    # 122022: same formula for ALL modes.
    # q < 20  (eta > 20/441 ≈ 0.04535) → fMECO
    # q >= 20 (eta < 20/441)            → smooth tanh transition
    eta_q20 = 20.0 / 441.0  # eta at q = 20
    transition_eta = 0.0192234  # q ≈ 50
    sharpness = 0.004
    funcs = 0.5 + 0.5 * jnp.tanh((eta - transition_eta) / sharpness)
    fcut_emr = funcs * fMECO + (1.0 - funcs) * fcutEMR
    return jnp.where(eta > eta_q20, fMECO, fcut_emr)


def _xhm_fAmpMatchIM(pWFHM: "XHMWaveformStruct", pWF22: dict) -> float:
    """Intermediate-to-ringdown cutoff frequency (version 122022)."""
    fRING = pWFHM.fRING
    fDAMP = pWFHM.fDAMP
    # 122022: mode 32 (MixingOn) → fRING22 - 0.5*fDAMP22; else → fRING - fDAMP
    if pWFHM.MixingOn:
        return pWF22["fRING22"] - 0.5 * pWF22["fDAMP22"]
    else:
        return fRING - fDAMP


@dataclass
class XHMAmpCoefficients:
    """
    All amplitude coefficients for one higher mode.
    Populated by xhm_get_amp_coefficients.
    """

    # PN global factor: (2/emm)^(-7/6) * prefactor[modeInt]
    PNglobalfactor: float | Array
    # Complex PN polynomial coefficients
    pn_coeffs: (
        tuple  # (pnInit, pnOneTh, pnTwoTh, pnThreeTh, pnFourTh, pnFiveTh, pnSixTh)
    )
    # Inspiral pseudo-PN coefficients (rho1, rho2, rho3)
    ins_rho1: float | Array
    ins_rho2: float | Array
    ins_rho3: float | Array
    # Inspiral cutoff (= fAmpMatchIN after veto, for pseudo-PN normalization)
    fAmpMatchIN: float | Array
    # Intermediate polynomial coefficients (122022 direct: c0..c7, shape (8,))
    inter_c: Array
    # Ringdown parameters
    ring_alambda: float | Array
    ring_lambda: float | Array
    ring_sigma: float | Array
    # Boundary frequencies
    fAmpMatchIM: float | Array
    fRING: float | Array
    fDAMP: float | Array
    # Amplitude normalization (sqrt(2*eta/3)*pi^(-1/6), from pWF22->ampNorm)
    ampNorm: float | Array


def xhm_get_amp_coefficients(
    pWFHM: "XHMWaveformStruct", pWF22: dict
) -> XHMAmpCoefficients:
    """
    Compute all amplitude coefficients for one higher mode.

    Algorithm mirrors IMRPhenomXHM_GetAmplitudeCoefficients, 122022 release path:
      1. Boundary frequencies (same for all modes in 122022).
      2. PN coefficients.
      3. Inspiral colloc pts (0.5/0.75/1.0)*fIN, always version 13 (f1+f3 only).
      4. RD colloc pts → vetos → compute (alambda, lambda_, sigma).
      5. RD falloff: value + slope at fRING+2*fDAMP.
      6. Intermediate: direct polynomial f^(-7/6)*poly via linear solve.
    """
    modeTag = pWFHM.modeTag
    modeInt = pWFHM.modeInt
    eta = pWF22["eta"]
    delta = pWF22["delta"]
    chiPN = pWF22["chiPN"]
    STotR = pWF22["STotR"]
    dchi = pWF22["dchi"]
    chi1L = pWF22["chi1L"]
    chi2L = pWF22["chi2L"]
    ampNorm = pWF22["ampNorm"]
    dchi_half = dchi * 0.5

    # PNglobalfactor = (2/emm)^(-7/6) * prefactor[modeInt]
    PNgf = (2.0 / pWFHM.emm) ** (-7.0 / 6.0) * _AMP_PREFACTORS[modeInt]

    # PN polynomial coefficients
    pn_coeffs = _xhm_pn_amp_coeffs(modeTag, eta, delta, chi1L, chi2L)
    # Boundary frequencies
    fIN = _xhm_fAmpMatchIN(pWFHM, pWF22)
    fIM = _xhm_fAmpMatchIM(pWFHM, pWF22)
    fRING = pWFHM.fRING
    fDAMP = pWFHM.fDAMP

    # ------------------------------------------------------------------
    # Inspiral: 122022 uses version 123 (3 points), no veto (InspiralAmpVeto=0)
    # LAL ordering: f[0]=0.5*fIN, f[1]=0.75*fIN, f[2]=fIN
    # PNAmplitudeInsp computed with InspRescaleFactor=0 → full strain
    # Source: LALSimIMRPhenomXHM_inspiral.c lines 893-940, case 123
    # ------------------------------------------------------------------
    f1, f2, f3 = 0.5 * fIN, 0.75 * fIN, fIN

    iv1_raw, iv2_raw, iv3_raw = _xhm_insp_amp_colloc_pts(
        modeTag, eta, chiPN, dchi_half, delta
    )
    iv1_abs = jnp.abs(iv1_raw)
    iv2_abs = jnp.abs(iv2_raw)
    iv3_abs = jnp.abs(iv3_raw)

    # Full-strain PN at each colloc freq (InspRescaleFactor=0 in LAL for 122022)
    # PNAmplitudeInsp[i] = PNgf * |pn_poly(fi)| * ampNorm * fi^{-7/6}
    PNf1_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f1, pn_coeffs)) * ampNorm * f1 ** (-7.0 / 6.0)
    )
    PNf2_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f2, pn_coeffs)) * ampNorm * f2 ** (-7.0 / 6.0)
    )
    PNf3_full = (
        PNgf * jnp.abs(_xhm_pn_poly(f3, pn_coeffs)) * ampNorm * f3 ** (-7.0 / 6.0)
    )

    # PNdominant = ampNorm * (2/emm)^{-7/6}  (LAL internals.c line 1151)
    PNdominant = ampNorm * (2.0 / pWFHM.emm) ** (-7.0 / 6.0)

    # v_i = (CollPt_i - PNAmp_full_i) * fi^{7/6} / PNdominant  (LAL inspiral.c 904-906)
    v1 = (iv1_abs - PNf1_full) * f1 ** (7.0 / 6.0) / PNdominant
    v2 = (iv2_abs - PNf2_full) * f2 ** (7.0 / 6.0) / PNdominant
    v3 = (iv3_abs - PNf3_full) * f3 ** (7.0 / 6.0) / PNdominant

    # Case 123: all 3 colloc points  (LAL inspiral.c case 123)
    f1_73 = f1 ** (7.0 / 3.0)
    f2_73 = f2 ** (7.0 / 3.0)
    f3_73 = f3 ** (7.0 / 3.0)
    f1_83 = f1 ** (8.0 / 3.0)
    f2_83 = f2 ** (8.0 / 3.0)
    f3_83 = f3 ** (8.0 / 3.0)
    f1_3 = f1**3.0
    f2_3 = f2**3.0
    f3_3 = f3**3.0
    f1_13 = f1 ** (1.0 / 3.0)
    f2_13 = f2 ** (1.0 / 3.0)
    f3_13 = f3 ** (1.0 / 3.0)
    finsp = fIN
    fc_73 = finsp ** (7.0 / 3.0)
    fc_83 = finsp ** (8.0 / 3.0)
    fc_3 = finsp**3.0
    denom = f1_73 * (f1_13 - f2_13) * f2_73 * (f1_13 - f3_13) * (f2_13 - f3_13) * f3_73

    c0_LAL = (
        fc_73
        * (
            -(f1_3 * f3_83 * v2)
            + f1_83 * f3_3 * v2
            + f2_3 * (f3_83 * v1 - f1_83 * v3)
            + f2_83 * (-(f3_3 * v1) + f1_3 * v3)
        )
    ) / denom
    c1_LAL = (
        fc_83
        * (
            f1_3 * f3_73 * v2
            - f1_73 * f3_3 * v2
            + f2_3 * (-(f3_73 * v1) + f1_73 * v3)
            + f2_73 * (f3_3 * v1 - f1_3 * v3)
        )
    ) / denom
    c2_LAL = (
        fc_3
        * (
            f1_73 * (-f1_13 + f3_13) * f3_73 * v2
            + f2_73 * (-(f3_83 * v1) + f1_83 * v3)
            + f2_83 * (f3_73 * v1 - f1_73 * v3)
        )
    ) / denom

    # ripple rho = LAL InspiralCoefficient * PNdominant/ampNorm = c_LAL * (2/emm)^{-7/6}
    emm_factor = (2.0 / pWFHM.emm) ** (-7.0 / 6.0)
    rho1 = c0_LAL * emm_factor
    rho2 = c1_LAL * emm_factor
    rho3 = c2_LAL * emm_factor

    F1 = fIN

    # ------------------------------------------------------------------
    # Ringdown: collocation points → (alambda, lambda_, sigma)
    # 122022 version 2: colloc pts at fRING-fDAMP, fRING, fRING+fDAMP
    # ------------------------------------------------------------------
    rdcp1_raw, rdcp2_raw, rdcp3_raw = _xhm_rd_amp_colloc_pts(
        modeTag, eta, STotR, dchi_half, delta, chiPN
    )
    rdcp1 = jnp.abs(rdcp1_raw)
    rdcp2 = jnp.abs(rdcp2_raw)
    rdcp3 = jnp.abs(rdcp3_raw)

    # Veto conditions (in-order, JAX-compatible)
    rdcp3 = jnp.where(
        rdcp3 >= rdcp2**2 / jnp.where(rdcp1 > 0, rdcp1, 1e-30),
        0.5 * rdcp2**2 / jnp.where(rdcp1 > 0, rdcp1, 1e-30),
        rdcp3,
    )
    rdcp3 = jnp.where(rdcp3 > rdcp2, 0.5 * rdcp2, rdcp3)
    rdcp3 = jnp.where((rdcp1 < rdcp2) & (rdcp3 > rdcp1), rdcp1, rdcp3)

    safe1 = jnp.where(rdcp1 > 0, rdcp1, 1e-30)
    safe2 = jnp.where(rdcp2 > 0, rdcp2, 1e-30)
    safe3 = jnp.where(rdcp3 > 0, rdcp3, 1e-30)
    deno = jnp.sqrt(safe1 / safe3) - safe1 / safe2
    deno = jnp.where(deno <= 0, 1e-16, deno)
    alambda = safe1 * fDAMP / deno
    sigma = jnp.sqrt(alambda / (safe2 * fDAMP))
    lambda_ = 0.5 * sigma * jnp.log(safe1 / safe3)

    # RD ansatz (122022, version 2): Lorentzian, no f^{-1/12} factor
    def _rd_amp22(f):
        dfr = f - fRING
        dfd = fDAMP * sigma
        return alambda * fDAMP / (jnp.exp(lambda_ * dfr / dfd) * (dfr**2 + dfd**2))

    # Falloff region: fAmpRDfalloff = fRING + 2*fDAMP
    ffall = fRING + 2.0 * fDAMP
    A_fall = _rd_amp22(ffall)
    dA_fall = jax.grad(_rd_amp22)(ffall)
    rd_falloff_amp = A_fall
    rd_falloff_slope = jnp.where(A_fall > 0, -dA_fall / A_fall, 0.0)

    def _rd_amp22_full(f):
        """RD amplitude including falloff region."""
        lorentz = _rd_amp22(f)
        falloff = rd_falloff_amp * jnp.exp(-rd_falloff_slope * (f - ffall))
        return jnp.where(f < ffall, lorentz, falloff)

    # ------------------------------------------------------------------
    # Intermediate: direct polynomial A(f) = f^(-7/6) * (c0 + c1*f + ...)
    # 122022 VersionCollocPtsInter:
    #   21:       [1,1,0,1,0,2] → nCoeffs = 5  (no left derivative, skip int2/int4)
    #   32/33/44: [2,1,1,1,1,2] → nCoeffs = 8
    # Equispaced: deltaf = (fIM - fIN) / 5, 6 slots
    # ------------------------------------------------------------------
    deltaf = (fIM - fIN) / 5.0
    f_int1 = fIN + deltaf
    f_int2 = fIN + 2.0 * deltaf
    f_int3 = fIN + 3.0 * deltaf
    f_int4 = fIN + 4.0 * deltaf

    int1, int2, int3, int4 = _xhm_inter_amp_colloc_pts(
        modeTag, eta, STotR, dchi_half, delta, chiPN
    )
    int1 = jnp.abs(int1)
    int2 = jnp.abs(int2)
    int3 = jnp.abs(int3)
    int4 = jnp.abs(int4)

    # Inspiral ansatz at fIN (boundary value + derivative)
    def insp_strain(f):
        v = _xhm_insp_rescaled(f, PNgf, pn_coeffs, rho1, rho2, rho3, fIN)
        return ampNorm * f ** (-7.0 / 6.0) * v

    inspF_IN = insp_strain(fIN)
    d_inspF_IN = jax.grad(insp_strain)(fIN)

    # RD ansatz at fIM (boundary value + derivative)
    # For 122022 version 2, RDRescaleFactor=0: the Lorentzian already represents
    # the full strain amplitude (same units as inspiral with InspRescaleFactor=0).
    # Do NOT multiply by ampNorm.
    def rd_strain(f):
        return _rd_amp22_full(f)

    rdF_IM = rd_strain(fIM)
    d_rdF_IM = jax.grad(rd_strain)(fIM)

    _EPS = 1e-30
    inspF_IN_s = jnp.where(jnp.abs(inspF_IN) < 1e-15, 1e-15, inspF_IN)
    rdF_IM_s = jnp.where(jnp.abs(rdF_IM) < 1e-15, 1e-15, rdF_IM)

    # Build and solve linear system for intermediate polynomial coefficients c_j
    # such that A_inter(f) = f^(-7/6) * sum_j c_j * f^j
    # Value constraint at f:     A[row,j] = f^(j-7/6)
    # Deriv constraint at f:     A[row,j] = (j - 7/6) * f^(j-13/6)

    if modeTag == 21:
        # [1,1,0,1,0,2] → 5 constraints: fIN(v), f_int1(v), f_int3(v), fIM(v+d)
        nC = 5
        freqs_b = jnp.array([fIN, f_int1, f_int3, fIM, fIM])
        vals_b = jnp.array([inspF_IN_s, int1, int3, rdF_IM_s, d_rdF_IM])
        use_deriv = [False, False, False, False, True]
    else:
        # [2,1,1,1,1,2] → 8 constraints: fIN(v+d), f_int1..int4(v), fIM(v+d)
        nC = 8
        freqs_b = jnp.array([fIN, fIN, f_int1, f_int2, f_int3, f_int4, fIM, fIM])
        vals_b = jnp.array(
            [inspF_IN_s, d_inspF_IN, int1, int2, int3, int4, rdF_IM_s, d_rdF_IM]
        )
        use_deriv = [False, True, False, False, False, False, False, True]

    rows = []
    for i, (f_col, is_deriv) in enumerate(zip(freqs_b, use_deriv)):
        if is_deriv:
            # row = [(j - 7/6) * f^(j - 13/6) for j in 0..nC-1]
            row = jnp.array(
                [(j - 7.0 / 6.0) * f_col ** (j - 13.0 / 6.0) for j in range(nC)]
            )
        else:
            row = jnp.array([f_col ** (j - 7.0 / 6.0) for j in range(nC)])
        rows.append(row)
    A_mat = jnp.stack(rows)
    inter_c = jnp.linalg.solve(A_mat, vals_b)

    return XHMAmpCoefficients(
        PNglobalfactor=PNgf,
        pn_coeffs=pn_coeffs,
        ins_rho1=rho1,
        ins_rho2=rho2,
        ins_rho3=rho3,
        fAmpMatchIN=F1,
        inter_c=inter_c,
        ring_alambda=alambda,
        ring_lambda=lambda_,
        ring_sigma=sigma,
        fAmpMatchIM=fIM,
        fRING=fRING,
        fDAMP=fDAMP,
        ampNorm=ampNorm,
    )


def xhm_amp_noModeMixing(
    Mf: Array, pAmp: XHMAmpCoefficients, pWFHM: "XHMWaveformStruct"
) -> Array:
    """
    Evaluate the (l,m) mode amplitude at frequencies Mf (no mode mixing).

    Full strain amplitude = ampNorm * V(Mf) where:
      Inspiral (Mf < fAmpMatchIN):
        A(Mf) = ampNorm * Mf^(-7/6) * [PNgf*|pn(Mf)| + rho1*(Mf/fc)^(7/3) + ...]
      Intermediate (fAmpMatchIN <= Mf < fAmpMatchIM):
        A(Mf) = ampNorm / Q(Mf)  where Q is degree-5 polynomial (inter_d)
      Ringdown (Mf >= fAmpMatchIM):
        A(Mf) = ampNorm * Mf^(-7/6) * fDAMP*|alambda|*sigma*exp(...)/(dfr^2+dfd^2)*Mf^(-1/12)

    Uses jnp.where for JAX-compatible branching.
    Source: IMRPhenomXHM_Amplitude_noModeMixing in LALSimIMRPhenomXHM.c.
    """
    fIN = pAmp.fAmpMatchIN
    fIM = pAmp.fAmpMatchIM
    PNgf = pAmp.PNglobalfactor
    an = pAmp.ampNorm

    # Inspiral amplitude: 122022 uses standard PN polynomial for all modes
    def insp_amp(f):
        v = _xhm_insp_rescaled(
            f, PNgf, pAmp.pn_coeffs, pAmp.ins_rho1, pAmp.ins_rho2, pAmp.ins_rho3, fIN
        )
        return an * f ** (-7.0 / 6.0) * v

    # Intermediate amplitude: 122022 direct polynomial A(f) = f^(-7/6) * poly(c, f)
    nC = 5 if pWFHM.modeTag == 21 else 8

    def inter_amp(f):
        c = pAmp.inter_c
        poly = sum(c[j] * f**j for j in range(nC))
        return f ** (-7.0 / 6.0) * poly

    # Ringdown amplitude: 122022 Lorentzian with exponential falloff
    ffall = pAmp.fRING + 2.0 * pAmp.fDAMP

    def _lorentz(f):
        dfr = f - pAmp.fRING
        dfd = pAmp.fDAMP * pAmp.ring_sigma
        return (
            pAmp.ring_alambda
            * pAmp.fDAMP
            / (jnp.exp(pAmp.ring_lambda * dfr / dfd) * (dfr**2 + dfd**2))
        )

    # Compute falloff anchor (avoid jax.grad inside vmap; use analytic derivative)
    A_fall = _lorentz(ffall)
    # Analytic derivative of Lorentzian:
    # dA/df = -A*(lambda_/dfd + 2*dfr/(dfr^2+dfd^2))
    dfr_f = ffall - pAmp.fRING
    dfd_f = pAmp.fDAMP * pAmp.ring_sigma
    rd_slope = pAmp.ring_lambda / dfd_f + 2.0 * dfr_f / (dfr_f**2 + dfd_f**2)

    def rd_amp(f):
        lorentz = _lorentz(f)
        falloff = A_fall * jnp.exp(-rd_slope * (f - ffall))
        # 122022 version 2 (RDRescaleFactor=0): the Lorentzian is already in
        # full-strain units; do NOT multiply by ampNorm.
        return jnp.where(f < ffall, lorentz, falloff)

    # Vectorise over Mf with jnp.where
    amp_i = jax.vmap(insp_amp)(Mf)
    amp_m = jax.vmap(inter_amp)(Mf)
    amp_r = jax.vmap(rd_amp)(Mf)

    result = jnp.where(Mf < fIN, amp_i, jnp.where(Mf < fIM, amp_m, amp_r))
    return result


# ---------------------------------------------------------------------------
# Section 10: Main mode generation function
# ---------------------------------------------------------------------------


def XLALSimIMRPhenomXHMEvaluateOnehlmMode(
    freqs_geom: Array,
    pWFHM: XHMWaveformStruct,
    pPhase: XHMPhaseCoefficients,
    pAmp: XHMAmpCoefficients,
    pWF22: dict,
    t0: float,
    phi0: float | Array,
) -> Array:
    """
    Evaluate complex hlm for one mode at all frequencies.

    Returns complex array of shape (len(freqs_geom),):
      hlm(Mf) = Amp(Mf) * exp(-i * (t0*(Mf - Mf_ref) + phase_lm(Mf) - emm*phi0))

    The t0 is IMRPhenomX_TimeShift_22 (not the DPhiMRD t0 from old HM).
    Source: IMRPhenomXHMEvaluateOnehlmMode in LALSimIMRPhenomXHM.c.
    """
    emm = pWFHM.emm
    # Phase
    # if pWFHM.MixingOn:
    #     phase_lm = xhm_phase_ModeMixing(freqs_geom, pPhase, pWFHM, pWF22, t0)
    phase_lm = xhm_phase_noModeMixing(freqs_geom, pPhase, pWFHM, pWF22, t0)

    # Amplitude
    if pAmp is not None:
        amp_lm = xhm_amp_noModeMixing(freqs_geom, pAmp, pWFHM)
    else:
        amp_lm = jnp.ones_like(freqs_geom)

    # Phase convention: use same positive-phase sign as ripple XAS (exp(+i*Psi)),
    # matching the convention used by the old IMRPhenomHMEvaluateOnehlmMode.
    # xhm_phase_noModeMixing already encodes t0 through the intermediate collocation
    # derivatives (all_vals = fits + t0) and C1INSP.
    phase_total = phase_lm - emm * phi0
    return amp_lm * jnp.exp(1j * phase_total)


def XLALSimIMRPhenomXHMGethlmModes(
    freqs_geom: Array,
    pWF22: dict,
    phi0: float | Array,
    ell_mm_pairs: list,
) -> dict:
    """
    Generate all requested higher modes in geometric units.

    Entry point called from IMRPhenomXPHM.py to replace XLALSimIMRPhenomHMGethlmModes.

    Args:
      freqs_geom:   frequency array in geometric units (M_total * f)
      pWF22:        22-mode waveform parameter dict (from build_pWF22)
      phi0:         reference phase (coalescence phase)
      ell_mm_pairs: list of (ell, mm) pairs, e.g. [(2,2),(2,1),(3,3),(3,2),(4,4)]

    Returns:
      dict mapping (ell, mm) -> complex array of shape (len(freqs_geom),)

    Algorithm (mirrors LALSimIMRPhenomXHM.c main loop):
      1. Compute t0 = IMRPhenomX_TimeShift_22.
      2. Compute phifRef (22-mode reference phase).
      3. Generate 22 mode via XAS Phase + t0 + phifRef.
      4. For each higher mode (ell, mm) != (2,2):
         a. xhm_set_waveform_variables -> pWFHM
         b. xhm_get_phase_coefficients -> pPhase
         c. xhm_get_amp_coefficients -> pAmp
         d. XLALSimIMRPhenomXHMEvaluateOnehlmMode -> hlm
    """
    M_s = pWF22["M_s"]
    theta = pWF22["theta"]
    phase_coeffs = pWF22["phase_coeffs"]
    Mf_ref = pWF22["Mf_ref"]
    chip = pWF22.get("chip", 0.0)
    # Step 1: time shift for 22 mode
    t0 = IMRPhenomX_TimeShift_22(pWF22)

    # Step 2: Reference phase of the 22-mode at Mf_ref.
    # phifRef (for the (2,2) mode): sets phase(Mf_ref) = 2*phi0.
    # phiref22 (for higher modes): LAL's pWFHM->phiref22 convention.
    phi_22_at_ref = IMRPhenomXAS_Phase(Mf_ref / M_s, theta, phase_coeffs, chip)
    # phifRef sets the reference phase for the (2,2) mode (and equals LAL's phiref22).
    # LAL: phiref22 = -phi_22(Mf_ref)/eta - timeshift*Mf_ref - phaseshift + 2*phi0 + PI/4
    # (phaseshift=0 in LAL; phi_22_at_ref already includes 1/eta normalization in ripple)
    phifRef = -phi_22_at_ref - t0 * Mf_ref + 2.0 * phi0 + PI / 4.0

    hlm_dict = {}

    for ell, mm in ell_mm_pairs:
        if ell == 2 and mm == 2:
            # 22-mode: XAS Phase + t0 + phifRef
            phase_22 = IMRPhenomXAS_Phase(freqs_geom / M_s, theta, phase_coeffs, chip)
            # 22-mode amplitude: XAS amplitude without distance factor.
            # IMRPhenomXAS_Amp returns Overallamp * Amp_internal * fMs^(-7/6) where
            # Overallamp = amp0_dist * ampNorm.  Dividing out amp0_1mpc leaves
            # ampNorm * Amp_internal * fMs^(-7/6) — the same convention used by
            # the higher-mode amplitudes — so generate_xphm can apply amp0 uniformly.
            amp_coeffs_22 = IMRPhenomX_utils.PhenomX_amp_coeff_table
            amp0_1mpc = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M_s**2 / (MPC / C)
            amp_22 = (
                IMRPhenomXAS_Amp(freqs_geom / M_s, theta, amp_coeffs_22, D=1.0, chip=chip)
                / amp0_1mpc
            )
            # 22 mode: phifRef already encodes 2*phi0, so no extra subtraction needed.
            # LAL: Psi_22 = t0*(f-Mf_ref) + phase_22 - phi_22_at_ref + 2*phi0 + PI/4
            phase_total = t0 * freqs_geom + phase_22 + phifRef
            hlm_dict[(ell, mm)] = amp_22 * jnp.exp(1j * phase_total)
        else:
            pWFHM = xhm_set_waveform_variables(ell, mm, pWF22)
            if pWFHM.MixingOn:
                hlm = _compute_32_hlm(freqs_geom, pWFHM, pWF22, t0, phifRef, phi0)
                hlm_dict[(ell, mm)] = hlm * jnp.exp(
                    1j * ((mm / 2.0) * phifRef + mm * phi0)
                )
                continue
            pPhase = xhm_get_phase_coefficients(pWFHM, pWF22, t0)
            pAmp = xhm_get_amp_coefficients(pWFHM, pWF22)
            hlm = XLALSimIMRPhenomXHMEvaluateOnehlmMode(
                freqs_geom, pWFHM, pPhase, pAmp, pWF22, t0, phi0
            )
            # LAL includes (mm/2)*phiref22 in deltaphiLM (phiref22 = phifRef here),
            # which expands to (mm/2)*phifRef_const + mm*phi0. Combined with the
            # -mm*phi0 already in hlm from evaluateOnehlmMode, the net is:
            #   -mm*phi0 + (mm/2)*phifRef + mm*phi0 = (mm/2)*phifRef_const + mm*phi0
            # This matches LAL's phi0-dependent mode phases (eq. 2364/2368 LALSimIMRPhenomXHM.c).
            hlm_dict[(ell, mm)] = hlm * jnp.exp(1j * ((mm / 2.0) * phifRef + mm * phi0))

    return hlm_dict


def gen_IMRPhenomXHM_hphc(
    freqs: Array,
    theta: Array,
    f_ref: float,
) -> Tuple[Array, Array]:
    """
    Generate IMRPhenomXHM hp, hc polarizations for an aligned-spin binary.

    Matches LAL's SimInspiralChooseFDWaveform("IMRPhenomXHM").

    theta = [m1, m2, chi1z, chi2z, dist_mpc, tc, phi_ref, iota]
      m1, m2   : component masses in solar masses
      chi1z/2z : aligned spins [-1, 1]
      dist_mpc : luminosity distance in Mpc
      tc       : coalescence time (not used; included for interface consistency)
      phi_ref  : orbital phase at reference frequency [rad]
      iota     : inclination angle [rad]

    Assembly mirrors LAL's IMRPhenomXHMFDAddMode (sym=1, phi=pi/2):
      For each mode (l, m>0):
        Ym      = Y_{l,-m}^{-2}(iota, pi/2) = F_{l,-m}(iota) * (-i)^m
        Ystar   = conj(Y_{l,m}^{-2}(iota, pi/2)) = F_{lm}(iota) * (-i)^m
        minus1l = (-1)^l
        factorp = 0.5 * (Ym + minus1l * Ystar)
        factorc = 0.5j * (Ym - minus1l * Ystar)
        hp += factorp * hlm
        hc += factorc * hlm
    where hlm is from XLALSimIMRPhenomXHMGethlmModes (positive-phase convention,
    same as LAL's h_{l,-m} FD mode at positive frequencies).
    """
    m1, m2 = theta[0], theta[1]  # solar masses
    chi1z, chi2z = theta[2], theta[3]
    dist_mpc = theta[4]
    # theta[5] = tc (unused)
    phi_ref = theta[6]
    iota = theta[7]

    Mtot = m1 + m2  # solar masses
    M_s = Mtot * MTSUN  # total mass in seconds
    dist_m = dist_mpc * MPC  # distance in metres
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    freqs_geom = freqs * M_s
    pWF22 = build_pWF22(m1, m2, chi1z, chi2z, f_ref)

    ell_mm_pairs = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
    hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
        freqs_geom, pWF22, phi0=phi_ref, ell_mm_pairs=ell_mm_pairs
    )

    _sminus2 = {2: compute_sminus2_l2, 3: compute_sminus2_l3, 4: compute_sminus2_l4}

    hp = jnp.zeros(len(freqs), dtype=jnp.complex128)
    hc = jnp.zeros(len(freqs), dtype=jnp.complex128)

    for ell, emm in ell_mm_pairs:
        # LAL applies (-1)^l to Amp0 for odd-l modes (LALSimIMRPhenomXHM.c line 601-604).
        minus1l = 1 if ell % 2 == 0 else -1
        hlm = hlm_dict[(ell, emm)] * amp0 * minus1l
        Ylm_fn = _sminus2[ell]
        F_neg = Ylm_fn(iota, -emm)  # F_{l,-m}(iota) — real
        F_pos = Ylm_fn(iota, emm)  # F_{l,+m}(iota) — real
        neg_im = (-1j) ** emm  # (-i)^m = exp(-i*m*pi/2)
        factorp = 0.5 * neg_im * (F_neg + minus1l * F_pos)
        factorc = 0.5j * neg_im * (F_neg - minus1l * F_pos)
        hp = hp + factorp * hlm
        hc = hc + factorc * hlm

    return hp, hc
