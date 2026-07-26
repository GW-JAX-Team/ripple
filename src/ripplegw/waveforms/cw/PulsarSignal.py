"""ripple ``Waveform``-compatible interface for the full isolated pulsar signal."""

from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.interfaces import TimeDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cw.detectors import Detector, get_detector
from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import generate_pulsar_polarizations


@register("PulsarSignal", is_binary=False)
class PulsarSignal(TimeDomainWaveform):
    """Full ``XLALGeneratePulsarSignal`` polarizations for an isolated pulsar.

    Like :class:`~ripplegw.waveforms.cw.ExactPulsarSignal.ExactPulsarSignal`
    but using the *full* barycentering delay (Roemer + Earth-rotation with
    precession/nutation + Einstein - Shapiro) and optional heterodyning.
    Requires both Earth and Sun ephemerides.

    Attributes:
        detector (Detector): Detector defining the arrival-time grid.
        start_gps (int): Integer GPS second of ``t == 0`` / heterodyne epoch.
        n_spindowns (int): Number of spindown parameters.
        ref_time_ssb (float | None): SSB spin reference epoch (default: full
            SSB time of the first sample).
        f_heterodyne (float): Heterodyne frequency (Hz).
    """

    detector: Detector
    start_gps: int
    n_spindowns: int
    ref_time_ssb: float | None
    f_heterodyne: float

    def __init__(
        self,
        detector: str | Detector,
        earth_ephemeris_file: str,
        sun_ephemeris_file: str,
        start_gps: int,
        n_spindowns: int = 0,
        ref_time_ssb: float | None = None,
        f_heterodyne: float = 0.0,
    ) -> None:
        """
        Args:
            detector: Detector name or :class:`Detector`.
            earth_ephemeris_file (str): Earth ephemeris path, or a standard
                LALPulsar name (e.g. ``"earth00-40-DE405.dat.gz"``) to
                download and cache automatically -- see
                :func:`~ripplegw.waveforms.cw.ephemeris.resolve_ephemeris_path`.
            sun_ephemeris_file (str): Sun ephemeris path or standard name
                (for the Shapiro term), same rules as ``earth_ephemeris_file``.
            start_gps (int): GPS second of ``t == 0`` (also the heterodyne epoch).
            n_spindowns (int): Number of spindown parameters.
            ref_time_ssb (float | None): SSB spin reference epoch.
            f_heterodyne (float): Heterodyne frequency (Hz).
        """
        self.detector = (
            detector if isinstance(detector, Detector) else get_detector(detector)
        )
        self.start_gps = int(start_gps)
        self.n_spindowns = int(n_spindowns)
        self.ref_time_ssb = ref_time_ssb
        self.f_heterodyne = float(f_heterodyne)
        e = read_ephemeris_file(earth_ephemeris_file)
        s = read_ephemeris_file(sun_ephemeris_file)
        self._e = (
            e.gps0,
            e.dt,
            jnp.asarray(e.pos),
            jnp.asarray(e.vel),
            jnp.asarray(e.acc),
        )
        self._s = (
            s.gps0,
            s.dt,
            jnp.asarray(s.pos),
            jnp.asarray(s.vel),
            jnp.asarray(s.acc),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "alpha",
            "delta",
            "f0",
            "phi0",
            "aplus",
            "across",
            *(f"f{i}" for i in range(1, self.n_spindowns + 1)),
        )

    def __call__(
        self, t: Float[Array, " n_time"], params: Mapping[str, FloatLike]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the full isolated-pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Sample times (s) relative to ``start_gps``.
            params: ``alpha``, ``delta``, ``f0``, ``phi0``,
                ``aplus``, ``across``, and ``f1`` … ``f{n_spindowns}``.

        Returns:
            dict[str, Float[Array, " n_time"]]: ``{"p", "c"}`` polarizations.
        """
        fkdot = tuple(params[f"f{i}"] for i in range(1, self.n_spindowns + 1))
        hp, hc = generate_pulsar_polarizations(
            t,
            self.start_gps,
            params["alpha"],
            params["delta"],
            params["f0"],
            params["phi0"],
            params["aplus"],
            params["across"],
            self.detector.location,
            *self._e,
            *self._s,
            fkdot=fkdot,
            ref_time_ssb=self.ref_time_ssb,
            f_heterodyne=self.f_heterodyne,
        )
        return {"p": hp, "c": hc}

    def __repr__(self) -> str:
        return (
            f"PulsarSignal(detector={self.detector.name!r}, "
            f"start_gps={self.start_gps}, n_spindowns={self.n_spindowns}, "
            f"f_heterodyne={self.f_heterodyne})"
        )
