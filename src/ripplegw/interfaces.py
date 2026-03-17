from abc import ABC

import jax.numpy as jnp
from jaxtyping import Array, Float

from .waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from .waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from .waveforms.TaylorF2 import gen_TaylorF2_hphc
from .waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
from .waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from .waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from .waveforms.SineGaussian import gen_SineGaussian_hphc
from .waveforms.IMRPhenomXPHM import generate_xphm
from .conversions import Mc_eta_to_ms


class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        return NotImplemented


class IMRPhenomD(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomD(f_ref={self.f_ref})"


class IMRPhenomPv2(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}
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
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomPv2(f_ref={self.f_ref})"


class TaylorF2(Waveform):
    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_TaylorF2_hphc(
            frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"TaylorF2(f_ref={self.f_ref})"


class IMRPhenomD_NRTidalv2(Waveform):
    f_ref: float
    use_lambda_tildes: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ):
        """
        Initialize the waveform.

        Args:
            f_ref (float, optional): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool, optional): Whether we sample over lambda_tilde and delta_lambda_tilde, as defined for instance in Equation (5) and Equation (6) of arXiv:1402.5156, rather than lambda_1 and lambda_2. Defaults to False.
            no_taper (bool, optional): Whether to remove the Planck taper in the amplitude of the waveform, which we use for relative binning runs. Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )

        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomD_NRTidalv2(f_ref={self.f_ref})"


class IMRPhenomXAS(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomXAS(f_ref={self.f_ref})"


class IMRPhenomXAS_NRTidalv3(Waveform):
    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ):
        """
        Initialize the waveform.

        Args:
            f_ref (float, optional): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool, optional): Whether we sample over lambda_tilde and delta_lambda_tilde rather than lambda_1 and lambda_2. Defaults to False.
            no_taper (bool, optional): Whether to disable tapering. Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_NRTidalv3_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomXAS_NRTidalv3(f_ref={self.f_ref})"


class IMRPhenomXPHM(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = generate_xphm(
            m1,
            m2,
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            frequency,
            self.f_ref,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomXPHM(f_ref={self.f_ref})"


class SineGaussian(Waveform):
    def __init__(self):
        pass

    def __call__(
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """
        Args:
            t: Time grid centered at t=0. Create using
               ``jnp.arange(-duration/2, duration/2, 1/fs)``.
            params: Dictionary with keys ``Q`` (quality factor), ``f_0``
                (central frequency in Hz), ``hrss``, ``phi`` (phase),
                ``e`` (eccentricity).
        """
        output = {}
        theta = jnp.array(
            [
                params["Q"],
                params["f_0"],
                params["hrss"],
                params["phase"],
                params["e"],
            ]
        )
        hp, hc = gen_SineGaussian_hphc(t, theta)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return "SineGaussian()"


waveform_preset = {
    "IMRPhenomD": IMRPhenomD,
    "IMRPhenomPv2": IMRPhenomPv2,
    "TaylorF2": TaylorF2,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2,
    "IMRPhenomXAS": IMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": IMRPhenomXAS_NRTidalv3,
    "IMRPhenomXPHM": IMRPhenomXPHM,
    "SineGaussian": SineGaussian,
}
