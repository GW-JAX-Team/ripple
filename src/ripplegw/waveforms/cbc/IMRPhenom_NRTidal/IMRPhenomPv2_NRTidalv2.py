"""
IMRPhenomPv2_NRTidalv2: precessing BNS waveform, see LAL's ``LALSimIMRPhenomP.c``
(``IMRPhenomPv2NRTidal_V`` with ``NRTidalv2_V``) and http://arxiv.org/abs/1905.06011.

As in LAL, the NRTidalv2 corrections are applied to the co-precessing-frame
(aligned-spin IMRPhenomD) waveform *before* the precession twist-up:

    h_cop(f) = window(f) * (A_D(f) + A_T(f)) * exp(-i (Psi_D(f) + Psi_T(f) + Psi_SS(f)))

and ``h_cop`` is then rotated to the inertial frame by ``PhenomPCoreTwistUp``.
The tidal building blocks are shared with ``IMRPhenomD_NRTidalv2``, including the
quadrupole-monopole terms LAL feeds into the IMRPhenomD inspiral phase.
"""

from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from ripplegw.interfaces import DistanceScaledWaveform, FrequencyDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.utils.tidal import get_kappa
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    Phase_with_qm_correction,
    _get_merger_frequency,
    get_planck_taper,
    get_spin_phase_correction,
    get_tidal_amplitude,
    get_tidal_phase,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import Amp as PhDAmp
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import get_coeffs
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2 import (
    PhenomPCoreTwistUp,
    get_precession_setup,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2_utils import (
    phP_get_transition_frequencies,
)

# Number of points of the phase grid LAL uses to estimate the time shift, and the
# relative half-width of that grid around the merger frequency.
_N_FIXED = 10
_FIXED_GRID_HALF_WIDTH = 0.2

# Y_{2,2} normalisation: ripple's IMRPhenomD amplitudes include it, the twist-up
# (like LAL's PhenomP core) expects it divided out.
_NORM = 2.0 * jnp.sqrt(5.0 / (64.0 * PI))


def _natural_cubic_spline_derivative(
    x: Float[Array, " n"], y: Float[Array, " n"], x_eval: FloatLike
) -> FloatLike:
    """First derivative at ``x_eval`` of the natural cubic spline through ``(x, y)``.

    Equivalent to ``gsl_spline_eval_deriv`` on a ``gsl_interp_cspline`` spline,
    which is what LAL uses for its time-shift correction.
    """
    n = x.shape[0]
    h = jnp.diff(x)
    rhs = 6.0 * jnp.diff(jnp.diff(y) / h)
    # Tridiagonal system for the interior second derivatives; M_0 = M_{n-1} = 0.
    A = (
        jnp.diag(2.0 * (h[:-1] + h[1:]))
        + jnp.diag(h[1:-1], k=1)
        + jnp.diag(h[1:-1], k=-1)
    )
    M = jnp.concatenate([jnp.zeros(1), jnp.linalg.solve(A, rhs), jnp.zeros(1)])

    i = jnp.clip(jnp.searchsorted(x, x_eval, side="right") - 1, 0, n - 2)
    t = x_eval - x[i]
    b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0
    return b + M[i] * t + (M[i + 1] - M[i]) * t**2 / (2.0 * h[i])


def _tidal_phase(
    f: Float[Array, " n_freq"] | FloatLike,
    theta_intrinsic: Float[Array, "6"],
    kappa: FloatLike,
) -> Float[Array, " n_freq"] | FloatLike:
    """NRTidalv2 tidal phase plus the 3.5PN spin-tidal terms, as added by LAL
    to the IMRPhenomD phase in the co-precessing frame."""
    m1, m2, _, _, _, _ = theta_intrinsic
    x = (PI * (m1 + m2) * MTSUN * f) ** (2.0 / 3.0)
    return get_tidal_phase(x, theta_intrinsic, kappa) + get_spin_phase_correction(
        x, theta_intrinsic
    )


def gen_IMRPhenomPv2_NRTidalv2(
    f: Float[Array, " n_freq"],
    params: Float[Array, "14"],
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """
    Generate the IMRPhenomPv2_NRTidalv2 plus and cross polarizations.

    params = [Mchirp, eta, s1x, s1y, s1z, s2x, s2y, s2z, lambda_tilde/lambda1,
    delta_lambda_tilde/lambda2, D, tc, phic, iota]; body 1 is the heavier object,
    spins are in the LAL source frame at ``f_ref``.

    Args:
        f (Array): Frequencies in Hz.
        params (Array): Parameters, ordered as above (D in Mpc).
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): Whether the tidal entries are
            ``(lambda_tilde, delta_lambda_tilde)`` rather than ``(lambda_1, lambda_2)``.
        no_taper (bool): Disable the Planck taper (e.g. for relative binning).

    Returns:
        hp (array): Plus polarization.
        hc (array): Cross polarization.
    """
    Mc, eta_in = params[0], params[1]
    s1x, s1y, s1z, s2x, s2y, s2z = params[2:8]
    dist_mpc, tc, phiRef, incl = params[10:]

    m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta_in]))
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(
            jnp.array([params[8], params[9], m1, m2])
        )
    else:
        lambda1, lambda2 = params[8], params[9]

    # LAL's PhenomP convention is m1 < m2: flip masses, spins and lambdas.
    m1, m2 = m2, m1
    s1x, s2x = s2x, s1x
    s1y, s2y = s2y, s1y
    s1z, s2z = s2z, s1z
    lambda1, lambda2 = lambda2, lambda1

    setup = get_precession_setup(
        m1, m2, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z
    )
    chi1_l, chi2_l, chip = setup.chi1_l, setup.chi2_l, setup.chip
    M_s = setup.M * MTSUN

    # Co-precessing frame: IMRPhenomD takes the heavier body first, which is also
    # the ordering of the NRTidalv2 building blocks (merger frequency uses m1/m2 >= 1).
    theta_bbh = jnp.array([m2, m1, chi2_l, chi1_l])
    theta_intrinsic = jnp.array([m2, m1, chi2_l, chi1_l, lambda2, lambda1])
    coeffs = get_coeffs(theta_bbh)
    transition_freqs = phP_get_transition_frequencies(
        theta_bbh, coeffs[5], coeffs[6], chip
    )

    kappa = get_kappa(theta_intrinsic)
    f_merger = _get_merger_frequency(theta_intrinsic, kappa)

    def coprecessing_phase(freq):
        """Phase of the co-precessing waveform, without the ``phic`` offset."""
        return Phase_with_qm_correction(
            freq, theta_bbh, theta_intrinsic, coeffs, transition_freqs
        ) + _tidal_phase(freq, theta_intrinsic, kappa)

    phase = coprecessing_phase(f) - setup.phic
    amp = PhDAmp(f, theta_bbh, coeffs, transition_freqs, D=dist_mpc) / _NORM
    amp += (
        get_tidal_amplitude(
            (PI * M_s * f) ** (2.0 / 3.0), theta_intrinsic, kappa, distance=dist_mpc
        )
        / _NORM
    )
    if not no_taper:
        amp *= get_planck_taper(f, f_merger)
    hPhenom = amp * jnp.exp(-1j * phase)

    hp, hc = PhenomPCoreTwistUp(
        f,
        hPhenom,
        setup.eta,
        chi1_l,
        chi2_l,
        chip,
        setup.M,
        setup.angcoeffs,
        setup.Y2m,
        setup.alphaoffset,
        setup.epsilonoffset,
    )

    # Time shift so that the peak sits at t=0: dPhi/df at the merger frequency,
    # from the natural cubic spline LAL builds on a small grid around it.
    f_stop = jnp.minimum((1.0 + _FIXED_GRID_HALF_WIDTH) * f_merger, fM_CUT / M_s)
    f_fixed = jnp.linspace((1.0 - _FIXED_GRID_HALF_WIDTH) * f_merger, f_stop, _N_FIXED)
    t0 = _natural_cubic_spline_derivative(
        f_fixed, -coprecessing_phase(f_fixed), f_merger
    ) / (2 * PI)
    phase_corr = jnp.exp(-1j * f * (2 * PI * t0 + M_s * tc))
    hp *= phase_corr
    hc *= phase_corr

    c2z = jnp.cos(2 * setup.zeta_polariz)
    s2z = jnp.sin(2 * setup.zeta_polariz)
    return c2z * hp + s2z * hc, c2z * hc - s2z * hp


@register("IMRPhenomPv2_NRTidalv2", is_tidal=True, is_precessing=True, highest_mode=2)
class IMRPhenomPv2_NRTidalv2(FrequencyDomainWaveform, DistanceScaledWaveform):
    """IMRPhenomPv2_NRTidalv2 frequency-domain waveform (precessing spins, NRTidalv2 tides).

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` /
            ``delta_lambda_tilde``; otherwise ``lambda_1`` / ``lambda_2``.
        no_taper (bool): If True, the Planck taper in the amplitude is disabled.
    """

    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` (Eq. 5-6 of
                arXiv:1402.5156) instead of ``lambda_1`` / ``lambda_2``.
                Defaults to False.
            no_taper (bool): Whether to remove the Planck taper in the amplitude
                (useful for relative binning runs). Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            *(
                ("lambda_tilde", "delta_lambda_tilde")
                if self.use_lambda_tildes
                else ("lambda_1", "lambda_2")
            ),
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: Mapping[str, FloatLike]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomPv2_NRTidalv2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params: Source parameters with keys ``M_c``, ``eta``, ``s1_x``,
                ``s1_y``, ``s1_z``, ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``, plus tidal keys depending on
                ``use_lambda_tildes``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        first_lambda, second_lambda = (
            (params["lambda_tilde"], params["delta_lambda_tilde"])
            if self.use_lambda_tildes
            else (params["lambda_1"], params["lambda_2"])
        )
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                first_lambda,
                second_lambda,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_NRTidalv2(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return (
            f"IMRPhenomPv2_NRTidalv2(f_ref={self.f_ref}, "
            f"use_lambda_tildes={self.use_lambda_tildes}, no_taper={self.no_taper})"
        )
