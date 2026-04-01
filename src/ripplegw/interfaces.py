from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Array, Float

from .waveforms.TaylorF2 import gen_TaylorF2_hphc
from .waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from .waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
from .waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from .waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from .waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from .waveforms.IMRPhenomXHM import gen_IMRPhenomXHM_hphc
from .waveforms.IMRPhenomXP import gen_IMRPhenomXP_hphc
from .waveforms.IMRPhenomXPHM import generate_xphm
from .waveforms.IMRPhenomHM import gen_IMRPhenomHM
from .waveforms.SineGaussian import gen_SineGaussian_hphc
from .conversions import Mc_eta_to_ms


class Waveform(ABC):
    """Abstract base class for gravitational waveform models.

    Subclasses implement the frequency- (or time-) domain waveform and expose it
    via ``__call__``, returning a dictionary with polarization keys ``"p"`` (plus)
    and ``"c"`` (cross).
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered tuple of parameter names required by this waveform model.

        Returns:
            tuple[str, ...]: Parameter names in the order they are consumed,
                matching the keys expected in the ``params`` dict passed to
                ``__call__``.
        """
        raise NotImplementedError(
            "Waveform.parameter_names must be implemented by subclasses"
        )

    @abstractmethod
    def __call__(
        self, axis: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the waveform.

        Args:
            axis (Float[Array, " n_freq"]): Frequency (or time) grid.
            params (dict[str, Float]): Source parameter dictionary.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Dictionary with keys ``"p"``
                (plus polarization) and ``"c"`` (cross polarization).
        """
        raise NotImplementedError("Waveform.__call__ must be implemented by subclasses")


class TaylorF2(Waveform):
    """TaylorF2 post-Newtonian frequency-domain waveform including tidal effects.

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` and
            ``delta_lambda_tilde``; otherwise expects ``lambda_1`` and ``lambda_2``.
    """

    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` (as in Eq. 5-6 of
                arXiv:1402.5156) instead of ``lambda_1`` / ``lambda_2``.
                Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
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
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the TaylorF2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_TaylorF2_hphc(
            frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"TaylorF2(f_ref={self.f_ref})"


class IMRPhenomD(Waveform):
    """IMRPhenomD frequency-domain waveform (non-precessing, aligned spins).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomD waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomD(f_ref={self.f_ref})"


class IMRPhenomD_NRTidalv2(Waveform):
    """IMRPhenomD_NRTidalv2 frequency-domain waveform (non-precessing, NRTidalv2 tides).

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
            "s1_z",
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
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomD_NRTidalv2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
                0.0,
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
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomD_NRTidalv2(f_ref={self.f_ref})"


class IMRPhenomPv2(Waveform):
    """IMRPhenomPv2 frequency-domain waveform (precessing spins).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

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
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomPv2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomPv2(f_ref={self.f_ref})"


class IMRPhenomXAS(Waveform):
    """IMRPhenomXAS frequency-domain waveform (non-precessing, aligned spins, X family).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXAS waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXAS(f_ref={self.f_ref})"


class IMRPhenomXAS_NRTidalv3(Waveform):
    """IMRPhenomXAS_NRTidalv3 frequency-domain waveform (non-precessing, NRTidalv3 tides).

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
                via ``lambda_tilde`` / ``delta_lambda_tilde`` rather than
                ``lambda_1`` / ``lambda_2``. Defaults to False.
            no_taper (bool): Whether to disable tapering (useful for relative
                binning runs). Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
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
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXAS_NRTidalv3 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
                0.0,
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
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXAS_NRTidalv3(f_ref={self.f_ref})"


class IMRPhenomXHM(Waveform):
    """IMRPhenomXHM frequency-domain waveform (aligned spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        theta = jnp.array(
            [
                m1,
                m2,
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXHM_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXHM(f_ref={self.f_ref})"


class IMRPhenomXP(Waveform):
    """IMRPhenomXP frequency-domain waveform (precessing spins, 22-mode only).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

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
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXP waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXP_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXP(f_ref={self.f_ref})"


class IMRPhenomXPHM(Waveform):
    """IMRPhenomXPHM frequency-domain waveform (precessing spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

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
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXPHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Float[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomHM(f_ref={self.f_ref})"


class IMRPhenomHM(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_freq"]]:
        output = {}
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = gen_IMRPhenomHM(
            frequency,
            m1,
            m2,
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            self.f_ref,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomHM(f_ref={self.f_ref})"


class SineGaussian(Waveform):
    """Sine-Gaussian time-domain burst waveform."""

    def __init__(self) -> None:
        pass

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("Q", "f_0", "hrss", "phase", "e")

    def __call__(
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the SineGaussian waveform.

        Args:
            t (Float[Array, " n_time"]): Time grid centered at t=0. Create using
                ``jnp.arange(-duration/2, duration/2, 1/fs)``.
            params (dict[str, Float]): Source parameters with keys ``Q``
                (quality factor), ``f_0`` (central frequency in Hz), ``hrss``,
                ``phase``, ``e`` (eccentricity).

        Returns:
            dict[str, Float[Array, " n_time"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
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
        return {"p": hp, "c": hc}

    def __repr__(self):
        return "SineGaussian()"


#: Mapping from model name strings to ``Waveform`` subclasses.
#: Useful for selecting waveform models by name at runtime, e.g. from a
#: configuration file.
waveform_preset: dict[str, type[Waveform]] = {
    "TaylorF2": TaylorF2,
    "IMRPhenomD": IMRPhenomD,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2,
    "IMRPhenomPv2": IMRPhenomPv2,
    "IMRPhenomXAS": IMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": IMRPhenomXAS_NRTidalv3,
    "IMRPhenomXHM": IMRPhenomXHM,
    "IMRPhenomXP": IMRPhenomXP,
    "IMRPhenomXPHM": IMRPhenomXPHM,
    "IMRPhenomHM": IMRPhenomHM,
    "SineGaussian": SineGaussian,
}
