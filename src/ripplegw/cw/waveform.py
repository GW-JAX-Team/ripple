"""ripple ``Waveform``-compatible interface for the exact pulsar signal.

Wraps :func:`ripplegw.cw.pulsar_signal.exact_pulsar_polarizations` in a class
matching ripple's :class:`~ripplegw.interfaces.Waveform` API: it is called with
a time grid and a parameter dict and returns ``{"p": h_plus, "c": h_cross}``.

Unlike compact-binary waveforms, a continuous-wave signal's polarizations
depend on the *absolute* GPS epoch (through Earth's ephemeris and rotation), so
the observation start time, detector and ephemeris are fixed at construction;
the per-call time axis is measured in seconds relative to that start.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interfaces import Waveform
from .detectors import Detector, get_detector
from .ephemeris import read_ephemeris_file
from .pulsar_signal import (
    exact_pulsar_polarizations,
    generate_binary_pulsar_polarizations,
    generate_pulsar_polarizations,
)

__all__ = ["ExactPulsarSignal", "PulsarSignal", "BinaryPulsarSignal"]


class ExactPulsarSignal(Waveform):
    """Exact (geometric) continuous-wave pulsar signal polarizations.

    JAX port of the source model of ``XLALSimulateExactPulsarSignal``: the
    barycentered plus/cross polarizations of an isolated pulsar, evaluated at a
    ground-based detector. Relativistic timing terms (Einstein/Shapiro) are
    neglected exactly as in the LAL reference; the barycentering uses the JPL
    ephemeris and matches ``XLALBarycenter`` to ~1e-13 s.

    Attributes:
        detector (Detector): Detector whose arrival times define the grid.
        start_gps (int): Integer GPS second of ``t == 0`` on the call axis.
        n_spindowns (int): Number of spindown terms expected in ``params``
            (``f1`` … ``f{n}``).
        ref_time_ssb (float | None): SSB reference epoch for the spin
            parameters (seconds). If ``None``, the SSB time of the first sample
            is used (LAL's default).
    """

    detector: Detector
    start_gps: int
    n_spindowns: int
    ref_time_ssb: float | None

    def __init__(
        self,
        detector: str | Detector,
        earth_ephemeris_file: str,
        start_gps: int,
        sun_ephemeris_file: str | None = None,
        n_spindowns: int = 0,
        ref_time_ssb: float | None = None,
    ) -> None:
        """
        Args:
            detector (str | Detector): Detector name (``"H1"``, ``"L1"``,
                ``"V1"``) or a :class:`Detector` instance.
            earth_ephemeris_file (str): Path to a LALPulsar Earth ephemeris
                file (e.g. ``earth00-40-DE405.dat.gz``).
            start_gps (int): Integer GPS second corresponding to ``t == 0`` on
                the time axis passed to :meth:`__call__`.
            sun_ephemeris_file (str | None): Unused by the exact model (the
                Shapiro term is neglected); accepted for interface symmetry.
            n_spindowns (int): Number of spindown parameters (``f1`` …).
            ref_time_ssb (float | None): SSB reference epoch for the spins.
        """
        self.detector = (
            detector if isinstance(detector, Detector) else get_detector(detector)
        )
        self.start_gps = int(start_gps)
        self.n_spindowns = int(n_spindowns)
        self.ref_time_ssb = ref_time_ssb

        eph = read_ephemeris_file(earth_ephemeris_file)
        self._eph_gps0 = eph.gps0
        self._eph_dt = eph.dt
        self._eph_pos = jnp.asarray(eph.pos)
        self._eph_vel = jnp.asarray(eph.vel)
        self._eph_acc = jnp.asarray(eph.acc)

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
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Detector-frame sample times in seconds
                relative to ``start_gps`` (e.g. ``jnp.arange(N) / fs``).
            params (dict[str, Float]): Source parameters with keys ``alpha``,
                ``delta`` (sky position, rad), ``f0`` (Hz), ``phi0`` (rad),
                ``aplus``, ``across`` (polarization amplitudes), and ``f1`` …
                ``f{n_spindowns}`` (spindowns, Hz/s, …).

        Returns:
            dict[str, Float[Array, " n_time"]]: Plus (``"p"``) and cross
                (``"c"``) polarizations.
        """
        fkdot = tuple(params[f"f{i}"] for i in range(1, self.n_spindowns + 1))
        hp, hc = exact_pulsar_polarizations(
            t,
            self.start_gps,
            params["alpha"],
            params["delta"],
            params["f0"],
            params["phi0"],
            params["aplus"],
            params["across"],
            self.detector.location,
            self._eph_gps0,
            self._eph_dt,
            self._eph_pos,
            self._eph_vel,
            self._eph_acc,
            fkdot=fkdot,
            ref_time_ssb=self.ref_time_ssb,
        )
        return {"p": hp, "c": hc}

    def __repr__(self) -> str:
        return (
            f"ExactPulsarSignal(detector={self.detector.name!r}, "
            f"start_gps={self.start_gps}, n_spindowns={self.n_spindowns})"
        )


class PulsarSignal(Waveform):
    """Full ``XLALGeneratePulsarSignal`` polarizations for an isolated pulsar.

    Like :class:`ExactPulsarSignal` but using the *full* barycentering delay
    (Roemer + Earth-rotation with precession/nutation + Einstein - Shapiro) and
    optional heterodyning. Requires both Earth and Sun ephemerides.

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
            earth_ephemeris_file (str): Earth ephemeris path.
            sun_ephemeris_file (str): Sun ephemeris path (for the Shapiro term).
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
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the full isolated-pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Sample times (s) relative to ``start_gps``.
            params (dict[str, Float]): ``alpha``, ``delta``, ``f0``, ``phi0``,
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


class BinaryPulsarSignal(PulsarSignal):
    """Full ``XLALGeneratePulsarSignal`` polarizations for a pulsar in a binary.

    Adds the elliptic-orbit modulation of ``XLALGenerateSpinOrbitCW`` on top of
    :class:`PulsarSignal`. The orbital elements are supplied per call as
    parameters (so they too are differentiable).
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "alpha",
            "delta",
            "f0",
            "phi0",
            "aplus",
            "across",
            "asini",
            "ecc",
            "period",
            "argp",
            "tp_ssb",
            *(f"f{i}" for i in range(1, self.n_spindowns + 1)),
        )

    def __call__(
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the binary-pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Sample times (s) relative to ``start_gps``.
            params (dict[str, Float]): The isolated keys plus orbital elements
                ``asini`` (light-s), ``ecc``, ``period`` (s), ``argp`` (rad),
                ``tp_ssb`` (SSB periapsis time, GPS seconds).

        Returns:
            dict[str, Float[Array, " n_time"]]: ``{"p", "c"}`` polarizations.
        """
        fkdot = tuple(params[f"f{i}"] for i in range(1, self.n_spindowns + 1))
        hp, hc = generate_binary_pulsar_polarizations(
            t,
            self.start_gps,
            params["alpha"],
            params["delta"],
            params["f0"],
            params["phi0"],
            params["aplus"],
            params["across"],
            params["asini"],
            params["ecc"],
            params["period"],
            params["argp"],
            params["tp_ssb"],
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
            f"BinaryPulsarSignal(detector={self.detector.name!r}, "
            f"start_gps={self.start_gps}, n_spindowns={self.n_spindowns})"
        )
