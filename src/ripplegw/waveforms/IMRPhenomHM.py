import jax
import jax.numpy as jnp
from ..constants import PI, MSUN, MTSUN, MRSUN, MPC
from jaxtyping import Array, Float, Integer
from .spherical_harmonics import (
    compute_sminus2_l2,
    compute_sminus2_l3,
    compute_sminus2_l4,
)
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp
from .IMRPhenomD_utils import (
    EradRational0815,
    get_coeffs,
    get_transition_frequencies_from_fRD_fdamp,
)
from .IMRPhenomD import Phase as IMRPhenomD_Phase
from .IMRPhenomD import IMRPhenDAmplitude_NoCut
from .IMRPhenomD import get_IIb_raw_phase
from .IMRPhenomPv2_utils import FinalSpin0815
from .IMRPhenomXPHM import XLALSimIMRPhenomXUtilsHztoMf


# Phase shift due to leading order complex amplitude
# [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
# "Spherical hrmonic modes for numerical relativity"
# List of phase shifts: the index is the azimuthal number m
CSHIFT = jnp.array([0.0, PI / 2.0, 0.0, -PI / 2.0, PI, PI / 2.0, 0.0])


def gen_IMRPhenomHM(
    frequency_array,
    mass_1,
    mass_2,
    chi1,
    chi2,
    distance,  # in Mpc
    inclination,
    phi0,
    reference_frequency,
):
    """Generate IMRPhenomHM plus and cross polarizations."""

    m1_SI = mass_1 * MSUN
    m2_SI = mass_2 * MSUN
    Mtot = mass_1 + mass_2

    # Overall amplitude prefactor from LAL's XLALSimPhenomUtilsFDamp0:
    # amp0 = Mtot * MRSUN * Mtot * MTSUN / distance
    # where Mtot is in solar masses and distance is in meters
    dist_m = distance * MPC  # distance in meters
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m

    extra_params = {
        "ModeArray": jnp.array(
            [[2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 4]], dtype=jnp.int32
        )
    }

    hlm = XLALSimIMRPhenomHMGethlmModes(
        frequency_array,
        m1_SI,
        m2_SI,
        0,
        0,
        chi1,
        0,
        0,
        chi2,
        phi0,
        frequency_array[1] - frequency_array[0],
        reference_frequency,
        extra_params,
    )

    ells = extra_params["ModeArray"][:, 0]
    minus1l = jnp.where(ells % 2 != 0, -1, 1)
    mode_projections = jax.vmap(
        get_phenomHMFD_mode_projection,
        in_axes=(None, 0, 0, 0),
    )(
        inclination,
        minus1l,
        ells,
        extra_params["ModeArray"][:, 1],
    )

    # Reshape to (n_modes, 2, 1) and (n_modes, 1, f_sampling) so they broadcast to (n_modes, 2, f_sampling)
    projected = mode_projections[:, :, None] * hlm[:, None, :] * amp0
    hp, hc = jnp.sum(projected, axis=0)

    return hp, hc


def get_phenomHMFD_mode_projection(
    theta: float,
    minus1l: int | Array,
    ell: int | Array,
    m: int | Array,
) -> Array:
    """
    Helper function to compute mode-by-mode plus- and cross-polarisation prefactors
    """

    Y = jax.lax.switch(
        ell - 2,
        [
            lambda: compute_sminus2_l2(theta, m),
            lambda: compute_sminus2_l3(theta, m),
            lambda: compute_sminus2_l4(theta, m),
        ],
    )

    def sym_branch():
        # Equatorial symmetry: add in -m mode
        Ymstar = jax.lax.switch(
            ell - 2,
            [
                lambda: compute_sminus2_l2(theta, -m),
                lambda: compute_sminus2_l3(theta, -m),
                lambda: compute_sminus2_l4(theta, -m),
            ],
        )
        Ymstar = jnp.conj(Ymstar)
        factorp = 0.5 * (Y + minus1l * Ymstar)
        factorc = -1j * 0.5 * (Y - minus1l * Ymstar)
        return jnp.array([factorp, factorc])

    def asym_branch():  # NOTE This is for hypothetical m=0 modes, not currently implemented. Structure is there in case we ever want to use it
        # Not adding in the -m mode
        factorp = Y
        factorc = -1j * factorp
        return jnp.array([factorp, factorc])

    return jax.lax.select(
        m == 0,
        asym_branch(),
        sym_branch(),
    )


def XLALSimIMRPhenomHMGethlmModes(
    freqs: Array,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    phiRef: float,
    deltaF: float,
    f_ref: float,
    extraParams: dict,
):
    """Compute all hlm modes for IMRPhenomHM. JAX translation of XLALSimIMRPhenomHMGethlmModes."""
    ModeArray = extraParams["ModeArray"]

    pHM = {}
    pHM = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        freqs,
        deltaF,
        f_ref,
        phiRef,
        ModeArray,
    )

    # FIXME? LAL does some frequency spacing here, I'm not sure yet whether we need to do this

    # line 1288
    # Might be unused since we use ripple IMRPhenomD, which uses f[Hz]
    freqs_geom = XLALSimIMRPhenomXUtilsHztoMf(freqs, pHM["Mtot"])

    # Pre-compute mode-independent PhenomD quantities once (used by phase, amplitude, t0)
    theta = jnp.array([pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]])
    PhenomD_coeffs = get_coeffs(theta)
    M_s = pHM["Mtot"] * MTSUN
    f_RD = pHM["Mf_RD_22_PhenomD"] / M_s
    f_damp = pHM["Mf_DM_22_PhenomD"] / M_s
    PhenomD_transition_freqs = get_transition_frequencies_from_fRD_fdamp(
        theta, PhenomD_coeffs[5], PhenomD_coeffs[6], f_RD, f_damp
    )

    # Compute f4 and t0 once (mode-independent)
    gamma2, gamma3 = PhenomD_coeffs[5], PhenomD_coeffs[6]
    f4 = jnp.where(
        gamma2 >= 1,
        jnp.abs(f_RD + (-f_damp * gamma3) / gamma2),
        jnp.abs(f_RD + (f_damp * (-1 + jnp.sqrt(1 - gamma2**2)) * gamma3) / gamma2),
    )
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta, PhenomD_coeffs, f_RD, f_damp)

    # Store pre-computed quantities in pHM for per-mode functions
    pHM["_theta"] = theta
    pHM["_PhenomD_coeffs"] = PhenomD_coeffs
    pHM["_M_s"] = M_s
    pHM["_f_RD"] = f_RD
    pHM["_f_damp"] = f_damp
    pHM["_PhenomD_transition_freqs"] = PhenomD_transition_freqs
    pHM["_t0"] = t0

    # line 1316
    # compute the reference phase shift need to align the waveform so that
    # the phase is equal to phiRef at the reference frequency f_ref.
    phi_22_at_f_ref = IMRPhenomD_Phase(
        f_ref,
        jnp.array([pHM["m1"], pHM["m2"], chi1z, chi2z]),
        PhenomD_coeffs,
        PhenomD_transition_freqs,
    )
    phi0 = 0.5 * phi_22_at_f_ref + phiRef

    vmapped_IMRPhenomHMEvaluateOnehlmMode = jax.vmap(
        IMRPhenomHMEvaluateOnehlmMode,
        in_axes=(
            None,  # freqs_geom
            None,  # pHM
            0,  # ell
            0,  # mm
            None,  # phi0
        ),
    )

    hlms = vmapped_IMRPhenomHMEvaluateOnehlmMode(
        freqs_geom, pHM, pHM["ell_mm_pairs"][:, 0], pHM["ell_mm_pairs"][:, 1], phi0
    )

    return hlms


def IMRPhenomHMEvaluateOnehlmMode(
    freqs_geom: Float, pHM: dict, ell: int, mm: int, phi0: Float
):
    """
    Implementation of IMRPhenomHMEvaluateOnehlmMode in LALSimIMRPhenomHM.c
    """

    # generate phase and amplitude for single l,m mode
    phase_lm = IMRPhenomHMPhase(freqs_geom, pHM, ell, mm)
    amp_lm = IMRPhenomHMAmplitude(freqs_geom, pHM, ell, mm)

    # Use pre-computed t0 from pHM (mode-independent)
    t0 = pHM["_t0"]

    Mf = freqs_geom
    phase_term1 = -t0 * (Mf - pHM["Mf_ref"])
    phase_term2 = phase_lm - (mm * phi0)
    return amp_lm * jnp.exp(-1j * (phase_term1 + phase_term2))


def XLALSimPhenomUtilsPhenomPv2FinalSpin(
    m1: Float, m2: Float, chi1_l: Float, chi2_l: Float, chip: Float
):
    """
    Implementation of XLALSimPhenomUtilsPhenomPv2FinalSpin in LALSimPhenomUtils.c
    Assuming m1 >= m2
    """

    M = m1 + m2
    eta = m1 * m2 / (M * M)

    q_factor = m1 / M

    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l)

    Sperp = chip * q_factor * q_factor

    return jnp.copysign(1.0, af_parallel) * jnp.sqrt(Sperp**2 + af_parallel**2)


def init_PhenomHM_Storage(
    p: dict,
    m1_SI: float,
    m2_SI: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    freqs: Array,
    deltaF: float,
    f_ref: float,
    phiRef: float,
    ModeArray: Array,
):
    """
    Precompute a bunch of PhenomHM related quantities and store them
    Implementation of init_PhenomHM_Storage in LALSimIMRPhenomHM.c
    """

    p["m1"] = m1_SI / MSUN
    p["m2"] = m2_SI / MSUN
    p["Mtot"] = p["m1"] + p["m2"]
    p["eta"] = p["m1"] * p["m2"] / (p["Mtot"] * p["Mtot"])
    p["chi1x"] = chi1x
    p["chi1y"] = chi1y
    p["chi1z"] = chi1z
    p["chi2x"] = chi2x
    p["chi2y"] = chi2y
    p["chi2z"] = chi2z
    p["phiRef"] = phiRef
    p["deltaF"] = deltaF
    p["freqs"] = freqs
    p["f_ref"] = f_ref
    p["Mf_ref"] = XLALSimIMRPhenomXUtilsHztoMf(f_ref, p["Mtot"])

    p["chip"] = XLALSimPhenomUtilsChiP(
        p["m1"], p["m2"], p["chi1x"], p["chi1y"], p["chi2x"], p["chi2y"]
    )

    p["finmass"] = 1.0 - EradRational0815(p["eta"], p["chi1z"], p["chi2z"])
    p["finspin"] = XLALSimPhenomUtilsPhenomPv2FinalSpin(
        p["m1"], p["m2"], p["chi1z"], p["chi2z"], p["chip"]
    )

    # Define the supported modes and their indices
    ell_mm_pairs = ModeArray
    p["ell_mm_pairs"] = ell_mm_pairs

    # Create a mapping from (ell, mm) to array index for JAX-compatible lookup
    # We'll use a 2D array where mode_index_map[ell, mm] gives the index
    # Maximum ell=4, mm=4, so we need a 5x5 array (indices 0-4)
    # IMPORTANT: Build the map dynamically based on the actual ModeArray order
    mode_index_map = jnp.full((5, 5), -1, dtype=jnp.int32)
    ell_vals = ModeArray[:, 0].astype(jnp.int32)
    mm_vals = ModeArray[:, 1].astype(jnp.int32)
    mode_index_map = mode_index_map.at[ell_vals, mm_vals].set(
        jnp.arange(len(ModeArray), dtype=jnp.int32)
    )
    p["mode_index_map"] = mode_index_map

    vmapped_IMRPhenomHMGetRingdownFrequency = jax.vmap(
        IMRPhenomHMGetRingdownFrequency, in_axes=(0, 0, None, None)
    )
    f_rd_array, f_damp_array = vmapped_IMRPhenomHMGetRingdownFrequency(
        ell_mm_pairs[:, 0], ell_mm_pairs[:, 1], p["finmass"], p["finspin"]
    )

    # Store as 1D arrays indexed by mode order
    p["PhenomHMfring"] = f_rd_array  # shape: (5,)
    p["PhenomHMfdamp"] = f_damp_array  # shape: (5,)
    p["Mf_RD_22"] = f_rd_array[1]
    p["Mf_DM_22"] = f_damp_array[1]

    # Rholm and Taulm as 1D arrays (one per mode)
    p["Rholm"] = p["Mf_RD_22"] / f_rd_array  # shape: (5,)
    p["Taulm"] = f_damp_array / p["Mf_DM_22"]  # shape: (5,)

    # IMPORTANT: For the PhenomD amplitude calculation, LAL's IMRPhenomDAmpFrequencySequence
    # uses the PhenomD QNM data (QNMData_fRD, QNMData_fdamp) rather than SimRingdownCW_CW07102016.
    # We need to store these separately for use in IMRPhenomHMAmplitude.

    p["Mf_RD_22_PhenomD"] = (
        jnp.interp(p["finspin"], QNMData_a, QNMData_fRD) / p["finmass"]
    )
    p["Mf_DM_22_PhenomD"] = (
        jnp.interp(p["finspin"], QNMData_a, QNMData_fdamp) / p["finmass"]
    )

    return p


def IMRPhenomHMGetRingdownFrequency(
    ell: Integer, mm: Integer, finalmass: Float, finalspin: Float
):
    """
    Implementation of IMRPhenomHMGetRingdownFrequency in LALSimIMRPhenomHM.c
    """

    inv2Pi = 0.5 / PI
    ZZ = SimRingdownCW_CW07102016(SimRingdownCW_KAPPA(finalspin, ell, mm), ell, mm, 0)
    Mf_RD_tmp = inv2Pi * jnp.real(
        ZZ
    )  # GW ringdown frequency, converted from angular frequency
    fringdown = Mf_RD_tmp / finalmass  # scale by predicted final mass
    # lm mode ringdown damping time (imaginary part of ringdown), geometric units
    f_DAMP_tmp = inv2Pi * jnp.imag(ZZ)  # this is the 1./tau in the complex QNM
    fdamp = f_DAMP_tmp / finalmass  # scale by predicted final mass

    return fringdown, fdamp


def SimRingdownCW_KAPPA(jf: Float, ell: Integer, emm: Integer):
    """
    Domain mapping for dimnesionless BH spin
    """
    alpha = jnp.log(2.0 - jf) / jnp.log(3)
    beta = 1.0 / (2.0 + ell - jnp.abs(emm))
    return alpha**beta


def SimRingdownCW_CW07102016(kappa: Float, ell: Integer, input_m: Integer, n: int):
    """
    Dimensionless QNM Frequencies: Note that name encodes date of writing
    """

    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa
    kappa4 = kappa3 * kappa

    m = jnp.abs(input_m)

    def branch_220():
        # Fit for (l,m,n) == (2,2,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.0 + kappa * (
            1.557847 * jnp.exp(2.903124 * 1j)
            + 1.95097051 * jnp.exp(5.920970 * 1j) * kappa
            + 2.09971716 * jnp.exp(2.760585 * 1j) * kappa2
            + 1.41094660 * jnp.exp(5.914340 * 1j) * kappa3
            + 0.41063923 * jnp.exp(2.795235 * 1j) * kappa4
        )

    # def branch_221(): # Unused in XPHM
    #     return

    def branch_320():
        kappa5 = kappa4 * kappa
        kappa6 = kappa5 * kappa
        # Fit for (l,m,n) == (3,2,0). This is NOT a zero-damped mode in the extremal Kerr limit.
        return (
            1.022464 * jnp.exp(0.004870 * 1j)
            + 0.24731213 * jnp.exp(0.665292 * 1j) * kappa
            + 1.70468239 * jnp.exp(3.138283 * 1j) * kappa2
            + 0.94604882 * jnp.exp(0.163247 * 1j) * kappa3
            + 1.53189884 * jnp.exp(5.703573 * 1j) * kappa4
            + 2.28052668 * jnp.exp(2.685231 * 1j) * kappa5
            + 0.92150314 * jnp.exp(5.841704 * 1j) * kappa6
        )

    def branch_440():
        # Fit for (l,m,n) == (4,4,0). This is a zero-damped mode in the extremal Kerr limit.
        return 2.0 + kappa * (
            2.658908 * jnp.exp(3.002787 * 1j)
            + 2.97825567 * jnp.exp(6.050955 * 1j) * kappa
            + 3.21842350 * jnp.exp(2.877514 * 1j) * kappa2
            + 2.12764967 * jnp.exp(5.989669 * 1j) * kappa3
            + 0.60338186 * jnp.exp(2.830031 * 1j) * kappa4
        )

    def branch_210():
        kappa5 = kappa4 * kappa
        kappa6 = kappa5 * kappa
        # Fit for (l,m,n) == (2,1,0). This is NOT a zero-damped mode in the extremal Kerr limit.
        return (
            0.589113 * jnp.exp(0.043525 * 1j)
            + 0.18896353 * jnp.exp(2.289868 * 1j) * kappa
            + 1.15012965 * jnp.exp(5.810057 * 1j) * kappa2
            + 6.04585476 * jnp.exp(2.741967 * 1j) * kappa3
            + 11.12627777 * jnp.exp(5.844130 * 1j) * kappa4
            + 9.34711461 * jnp.exp(2.669372 * 1j) * kappa5
            + 3.03838318 * jnp.exp(5.791518 * 1j) * kappa6
        )

    def branch_330():
        # Fit for (l,m,n) == (3,3,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.5 + kappa * (
            2.095657 * jnp.exp(2.964973 * 1j)
            + 2.46964352 * jnp.exp(5.996734 * 1j) * kappa
            + 2.66552551 * jnp.exp(2.817591 * 1j) * kappa2
            + 1.75836443 * jnp.exp(5.932693 * 1j) * kappa3
            + 0.49905688 * jnp.exp(2.781658 * 1j) * kappa4
        )

    # def branch_331(): # Unused in XPHM
    #     return

    def branch_430():
        # Fit for (l,m,n) == (4,3,0). This is a zero-damped mode in the extremal Kerr limit.
        return 1.5 + kappa * (
            0.205046 * jnp.exp(0.595328 * 1j)
            + 3.10333396 * jnp.exp(3.016200 * 1j) * kappa
            + 4.23612166 * jnp.exp(6.038842 * 1j) * kappa2
            + 3.02890198 * jnp.exp(2.826239 * 1j) * kappa3
            + 0.90843949 * jnp.exp(5.915164 * 1j) * kappa4
        )

    # def branch_550(): # Unused in XPHM
    #     return

    def branch_not_implemented():
        return 0.0 * 1j  # Return complex nr. so pytree structure is preserved

    # Determine index of branch to use. If other modes are added, this will need to be expanded for new modes
    # Create a unique key from l, m, n: key = l * 100 + m * 10 + n
    key = ell * 100 + jnp.abs(m) * 10 + n

    # Map keys to indices
    # 210 → 0, 220 → 1, 320 → 2, 330 → 3, 430 → 4, 440 → 5
    index = jnp.where(
        key == 210,
        0,
        jnp.where(
            key == 220,
            1,
            jnp.where(
                key == 320,
                2,
                jnp.where(
                    key == 330, 3, jnp.where(key == 430, 4, jnp.where(key == 440, 5, 6))
                ),
            ),
        ),
    )

    ans = jax.lax.switch(
        index,
        [
            branch_210,
            branch_220,
            branch_320,
            branch_330,
            branch_430,
            branch_440,
            branch_not_implemented,
        ],
    )

    return jax.lax.select(  # If m<0, then take the *Negative* conjugate
        input_m < 0, -jnp.conj(ans), ans
    )


def IMRPhenomHMFreqDomainMap(Mflm, ell, mm, pHM, AmpFlag):
    """Map input frequency Mflm to the effective 22-mode frequency Mf22 for the (ell, mm) mode."""
    # Mflm here has the same meaning as Mf_wf in XLALSimIMRPhenomHMFreqDomainMapHM (old deleted function).
    # Following variables not used in this funciton but are returned in IMRPhenomHMFreqDomainMapParams
    a, b = IMRPhenomHMFreqDomainMapParams(Mflm, ell, mm, pHM, AmpFlag)
    Mf22 = a * Mflm + b
    return Mf22


def IMRPhenomHMAmplitude(freqs_geom: Array, pHM: dict, ell: int, mm: int):
    """
    Returns IMRPhenomHM amplitude evaluated at a set of input frequencies for the l,m mode
    Implementation of IMRPhenomHMAmplitude in LALSimIMRPhenomHM.c
    """

    # scale input frequencies according to PhenomHM model
    # LL: Map the input domain (frequencies) for this ell mm multipole
    # to those appropirate for the ell=|mm| multipole
    freqs_amp = IMRPhenomHMFreqDomainMap(freqs_geom, ell, mm, pHM, AmpFlag=True)

    # LL: Compute the PhenomD Amplitude at the mapped l=m=2 fequencies
    # NOTE: Use IMRPhenDAmplitude_NoCut instead of IMRPhenomD_Amp because
    # the mapped frequencies can exceed fM_CUT for higher modes

    # Use pre-computed quantities from pHM
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    amps_normalized = IMRPhenDAmplitude_NoCut(
        freqs_amp / pHM["_M_s"],
        theta,
        PhenomD_coeffs,
        PhenomD_transition_freqs,
    )

    # Apply the Amp0 prefactor: amp0 = sqrt(2/3 * eta) * pi^(-1/6) * f^(-7/6)
    # This matches LAL's IMRPhenDAmplitude which multiplies AmpInsAnsatz by AmpPreFac
    eta = pHM["eta"]
    amp0 = jnp.sqrt(2.0 / 3.0 * eta) * (PI ** (-1.0 / 6.0))
    # The prefactor is applied at the mapped frequencies (freqs_amp)
    amps = amp0 * (freqs_amp ** (-7.0 / 6.0)) * amps_normalized

    # LL: Here we map the ampliude's range using two steps:
    # (1) We divide by the leading order l=m=2 behavior, and then
    # scale in the expected PN behavior for the multipole of interest.
    # NOTE that this step is done at the mapped frequencies,
    # which results in smooth behavior despite the sharp featured of the domain map.
    # There are other (perhaps more intuitive) options for mapping the amplitudes,
    # but these do not have the desired smooth features.
    # (2) An additional scaling is needed to recover the desired PN ampitude.
    # This is needed becuase only frequencies appropriate for the dominant
    # quadrupole have been used thusly, so the current answer does not
    # conform to PN expectations for inspiral.
    # This is trikier than described here, so please give it a deeper think.

    # LL: Calculate the corrective factor for step #2
    beta_term1 = IMRPhenomHMOnePointFiveSpinPN(
        freqs_geom, ell, mm, pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]
    )

    # COMMENT FROM LAL CODE:
    # HACK to fix equal black hole case producing NaNs.
    # More elegant solution needed.
    def beta_term1_nozero():
        beta_term2 = IMRPhenomHMOnePointFiveSpinPN(
            2 * freqs_geom / mm,
            ell,
            mm,
            pHM["m1"],
            pHM["m2"],
            pHM["chi1z"],
            pHM["chi2z"],
        )

        beta = beta_term1 / beta_term2

        # LL: Apply steps #1 and #2
        HMamp_term1 = IMRPhenomHMOnePointFiveSpinPN(
            freqs_amp, ell, mm, pHM["m1"], pHM["m2"], pHM["chi1z"], pHM["chi2z"]
        )
        HMamp_term2 = IMRPhenomHMOnePointFiveSpinPN(
            freqs_amp, 2, 2, pHM["m1"], pHM["m2"], 0.0, 0.0
        )

        return beta * HMamp_term1 / HMamp_term2

    rescaling = jnp.where(
        beta_term1 == 0.0,
        0.0,
        beta_term1_nozero(),
    )
    return amps * rescaling


def IMRPhenomHMOnePointFiveSpinPN(fM, ell, m, M1, M2, X1z, X2z):
    """
    Implementation of IMRPhenomHMOnePointFiveSpinPN from LALSimIMRPhenomHM.c
    Currently supported modes: (2,1), (2,2), (3,2), (3,3), (4,3), (4,4)
    """

    # LLondon 2017

    # Define effective intinsic parameters
    M_INPUT = M1 + M2
    M1 = M1 / (M_INPUT)
    M2 = M2 / (M_INPUT)
    M = M1 + M2
    eta = M1 * M2 / (M * M)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4 * eta, 0.0))
    Xs = 0.5 * (X1z + X2z)
    Xa = 0.5 * (X1z - X2z)

    # Define PN parameter and realed powers
    v = jnp.power(M * 2.0 * PI * fM / m, 1.0 / 3.0)
    v2 = v * v
    v3 = v * v2

    # Define Leading Order Ampitude for each supported multipole

    # (l,m) = (2,2)
    # THIS IS LEADING ORDER
    def lm_22():
        return jnp.full_like(fM, 1.0, dtype=complex)

    def lm_21():
        # (l,m) = (2,1)
        # SPIN TERMS ADDED

        # UP TO 4PN
        v4 = v * v3
        return (jnp.sqrt(2.0) / 3.0) * (
            v * delta
            - v2 * 1.5 * (Xa + delta * Xs)
            + v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0))
            + v4
            * (
                Xa * (3427.0 / 1344 - eta * 2101.0 / 336)
                + delta * Xs * (3427.0 / 1344 - eta * 965 / 336)
                + delta * (-1j * 0.5 - PI - 2 * 1j * 0.69314718056)
            )
        )

    def lm_33():
        # (l,m) = (3,3)
        # THIS IS LEADING ORDER
        return 0.75 * jnp.sqrt(5.0 / 7.0) * (v * delta) + 0 * 1j

    def lm_32():
        # (l,m) = (3,2)
        # NO SPIN TERMS to avoid roots
        return (1.0 / 3.0) * jnp.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta)) + 0 * 1j

    def lm_43():
        # (l,m) = (4,3)
        # NO SPIN TERMS TO ADD AT DESIRED ORDER
        return 0.75 * jnp.sqrt(3.0 / 35.0) * v3 * delta * (1.0 - 2.0 * eta) + 0 * 1j

    def lm_44():
        # (l,m) = (4,4)
        # THIS IS LEADING ORDER
        return (4.0 / 9.0) * jnp.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta) + 0 * 1j

    key = ell * 10 + jnp.abs(m)

    # Map keys to indices
    index = jnp.where(
        key == 21,
        0,
        jnp.where(
            key == 22,
            1,
            jnp.where(
                key == 32, 2, jnp.where(key == 33, 3, jnp.where(key == 43, 4, 5))
            ),
        ),
    )

    Hlm = jax.lax.switch(index, [lm_21, lm_22, lm_32, lm_33, lm_43, lm_44])

    # Compute the final PN Amplitude at Leading Order in fM
    return M * M * PI * jnp.sqrt(eta * 2.0 / 3) * v ** (-3.5) * jnp.abs(Hlm)


def IMRPhenomHMPhase(freqs_geom: Array, pHM: dict, ell: int, mm: int):
    """
    Returns IMRPhenomHM phase evaluated at a set of input frequencies for the l,m mode
    Implementation of IMRPhenomHMPhase in LALSimIMRPhenomHM.c
    """

    q = {}
    q = IMRPhenomHMPhasePreComp(q, ell, mm, pHM)

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]
    Rholm = pHM["Rholm"][mode_idx]
    Taulm = pHM["Taulm"][mode_idx]

    # Use pre-computed quantities from pHM
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    M_s = pHM["_M_s"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    # Fused piecewise phase: single IMRPhenomD_Phase call with per-region frequency mapping
    is_A = freqs_geom <= q["fi"]
    is_B = (freqs_geom > q["fi"]) & (freqs_geom <= q["fr"])
    # is_C = ~is_A & ~is_B

    # Map frequencies: each region uses its own (a, b) pair
    Mf_fused = jnp.where(
        is_A,
        (q["ai"] * freqs_geom + q["bi"]) / M_s,
        jnp.where(
            is_B,
            (q["am"] * freqs_geom + q["bm"]) / M_s,
            (q["ar"] * freqs_geom + q["br"]) / M_s,
        ),
    )

    # Single vectorized IMRPhenomD_Phase call
    phi_raw = IMRPhenomD_Phase(
        Mf_fused, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
    )

    # Apply per-region scaling and offsets
    inv_a = jnp.where(
        is_A, 1.0 / q["ai"], jnp.where(is_B, 1.0 / q["am"], 1.0 / q["ar"])
    )
    offset = jnp.where(
        is_A,
        0.0,
        jnp.where(
            is_B,
            -q["PhDBconst"] + q["PhDBAterm"],
            -q["PhDCconst"] + q["tmpphaseC"],
        ),
    )

    phase = CSHIFT[jnp.abs(mm)] + phi_raw * inv_a + offset

    return phase


def IMRPhenomHMPhasePreComp(q: dict, ell: int, emm: int, pHM: dict):
    """
    Implementation of IMRPhenomHMPhasePreComp in LALSimIMRPhenomHM.c
    """

    # NOTE: As long as Mfshift isn't >= fr then the value of the shift is arbitrary.
    Mfshift = 0.0001

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, emm]

    # I have moved the computation of f1, fi and fr outside of IMRPhenomHMFreqDomainMapParams
    f1 = 0.018  # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
    fi = f1 / pHM["Rholm"][mode_idx]
    fr = pHM["PhenomHMfring"][mode_idx]

    flm_0 = Mfshift
    flm_i = fi + Mfshift
    flm_r = fr + Mfshift

    ai, bi = IMRPhenomHMFreqDomainMapParams(flm_0, ell, emm, pHM, ampFlag=False)
    am, bm = IMRPhenomHMFreqDomainMapParams(flm_i, ell, emm, pHM, ampFlag=False)
    ar, br = IMRPhenomHMFreqDomainMapParams(flm_r, ell, emm, pHM, ampFlag=False)

    q["ai"] = ai
    q["bi"] = bi
    q["am"] = am
    q["bm"] = bm
    q["ar"] = ar
    q["br"] = br

    q["fi"] = fi
    q["fr"] = fr

    Rholm = pHM["Rholm"][mode_idx]
    Taulm = pHM["Taulm"][mode_idx]

    # Use pre-computed quantities from pHM
    M_s = pHM["_M_s"]
    theta = pHM["_theta"]
    PhenomD_coeffs = pHM["_PhenomD_coeffs"]
    PhenomD_transition_freqs = pHM["_PhenomD_transition_freqs"]

    # Batch all 4 scalar Phase evaluations into a single vectorized call
    PhDBMf = q["am"] * fi + q["bm"]
    PhDCMf = q["ar"] * fr + q["br"]
    PhDBAMf = q["ai"] * fi + q["bi"]
    Mfr_mapped_Mf = q["am"] * fr + q["bm"]

    boundary_freqs = jnp.array([PhDBMf, PhDCMf, PhDBAMf, Mfr_mapped_Mf]) / M_s
    boundary_phases = IMRPhenomD_Phase(
        boundary_freqs, theta, PhenomD_coeffs, PhenomD_transition_freqs, Rholm, Taulm
    )

    q["PhDBconst"] = boundary_phases[0] / q["am"]
    q["PhDCconst"] = boundary_phases[1] / q["ar"]
    q["PhDBAterm"] = boundary_phases[2] / q["ai"]
    q["tmpphaseC"] = boundary_phases[3] / q["am"] - q["PhDBconst"] + q["PhDBAterm"]
    return q


def IMRPhenomHMFreqDomainMapParams(
    flm: float,  # input waveform frequency
    ell: int,  # spherical harmonics ell mode
    mm: int,  # spherical harmonics m mode
    pHM: dict,
    ampFlag: bool,  # is ==1 then computes for amplitude, if ==0 then computes for phase
):
    """
    Implementation of the phase computation of IMRPhenomHMFreqDomainMapParams in LALSimIMRPhenomHM.c
    """

    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]

    Mf_1_22 = jax.lax.select(
        ampFlag,
        0.014,  # Dimensionless frequency (Mf) at which the inspiral amplitude switches to the intermediate amplitude
        0.018,  # Dimensionless frequency (Mf) at which the inspiral phase switches to the intermediate phase
    )
    Mf_RD_22 = pHM["Mf_RD_22"]
    Mf_RD_lm = pHM["PhenomHMfring"][mode_idx]

    # Define a ratio of QNM frequencies to be used for scaling various quantities
    Rholm = pHM["Rholm"][mode_idx]

    # Given experiments with the l!=m modes, it appears that the QNM scaling rather than the PN scaling may be optimal for mapping f1
    Mf_1_lm = Mf_1_22 / Rholm

    # Define transition frequencies
    fi = Mf_1_lm
    fr = Mf_RD_lm

    # Define the slope and intercepts of the linear transformation used
    Ai = 2.0 / mm
    Bi = 0.0

    Am, Bm = IMRPhenomHMSlopeAmAndBm(mm, fi, fr, Mf_RD_22, Mf_RD_lm, ampFlag, ell, pHM)

    Ar = jax.lax.select(
        ampFlag,
        1.0,  # For amplitude
        Rholm,  # For phase
    )
    Br = jax.lax.select(
        ampFlag,
        -Mf_RD_lm + Mf_RD_22,  # For amplitude
        0.0,  # For phase
    )

    a, b = IMRPhenomHMMapParams(flm, fi, fr, Ai, Bi, Am, Bm, Ar, Br)

    return a, b


def IMRPhenomHMSlopeAmAndBm(
    mm: int,
    fi: float,
    fr: float,
    Mf_RD_22: float,
    Mf_RD_lm: float,
    AmpFlag: bool,
    ell: int,
    pHM: dict,
):
    """
    Implementation of IMRPhenomHMSlopeAmAndBm in LALSimIMRPhenomHM.c
    """
    # Get mode index for array lookup
    mode_idx = pHM["mode_index_map"][ell, mm]

    Trd = IMRPhenomHMTrd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, mode_idx, pHM)
    Ti = 2.0 * fi / mm  # = IMRPhenomHMTi(fi, mm), line 543

    Am = (Trd - Ti) / (fr - fi)
    Bm = Ti - fi * Am

    return Am, Bm


def IMRPhenomHMTrd(
    Mf: float, Mf_RD_22: float, Mf_RD_lm: float, AmpFlag: bool, mode_idx: int, pHM: dict
):
    """
    Implementation of IMRPhenomHMTrd in LALSimIMRPhenomHM.c
    domain mapping function - ringdown
    """

    return jax.lax.select(
        AmpFlag,
        Mf
        - Mf_RD_lm
        + Mf_RD_22,  # Used for the Amplitude as an approx fix for post merger powerlaw slope
        pHM["Rholm"][mode_idx] * Mf,  # Used for the Phase
    )


def IMRPhenomHMMapParams(
    flm: Float,
    fi: Float,
    fr: Float,
    Ai: Float,
    Bi: Float,
    Am: Float,
    Bm: Float,
    Ar: Float,
    Br: Float,
):
    """
    Implementation of IMRPhenomHMMapParams in LALSimIMRPhenomHM.c, line 557
    """
    # Define function to output map params used depending on
    a = jnp.where(flm > fi, jnp.where(flm > fr, Ar, Am), Ai)
    b = jnp.where(flm > fi, jnp.where(flm > fr, Br, Bm), Bi)
    return a, b


def XLALSimPhenomUtilsChiP(m1, m2, s1x, s1y, s2x, s2y):
    """
    Compute the effective precession parameter chip.

    This is a JAX translation of LALSimIMRPhenomUtils.c XLALSimPhenomUtilsChiP.

    Args:
        m1 (float or array): Mass of companion 1 (solar masses).
        m2 (float or array): Mass of companion 2 (solar masses).
        s1x (float or array): x-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1).
        s1y (float or array): y-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1).
        s2x (float or array): x-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1).
        s2y (float or array): y-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1).

    Returns:
        float or array: Effective precession parameter chip.
    """
    m1_2 = m1 * m1
    m2_2 = m2 * m2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * jnp.sqrt(s1x * s1x + s1y * s1y)
    S2_perp = m2_2 * jnp.sqrt(s2x * s2x + s2y * s2y)

    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp

    num = jnp.where(ASp2 > ASp1, ASp2, ASp1)
    den = jnp.where(m2 > m1, A2 * m2_2, A1 * m1_2)
    chip = num / den

    return chip
