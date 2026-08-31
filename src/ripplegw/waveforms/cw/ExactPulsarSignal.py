"""ripple ``Waveform``-compatible interface for the exact pulsar signal.

Wraps :func:`ripplegw.waveforms.cw.pulsar_signal.exact_pulsar_polarizations` in
a class matching ripple's :class:`~ripplegw.interfaces.Waveform` API: it is
called with a time grid and a parameter dict and returns
``{"p": h_plus, "c": h_cross}``.

Unlike compact-binary waveforms, a continuous-wave signal's polarizations
depend on the *absolute* GPS epoch (through Earth's ephemeris and rotation),
so the observation start time and ephemeris are fixed at construction. The
observing site's location is supplied per call instead (``site_x``,
``site_y``, ``site_z`` in ``params``) precisely so a single instance can be
``jax.vmap``ed over multiple sites. The per-call time axis is measured in
seconds relative to that start.
"""

from collections.abc import Mapping
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ripplegw.interfaces import TimeDomainWaveform
from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import exact_pulsar_polarizations


@register("ExactPulsarSignal", is_binary=False)
class ExactPulsarSignal(TimeDomainWaveform):
    """Exact (geometric) continuous-wave pulsar signal polarizations.

    JAX port of the source model of ``XLALSimulateExactPulsarSignal``: the
    barycentered plus/cross polarizations of an isolated pulsar, evaluated at a
    ground-based site. Relativistic timing terms (Einstein/Shapiro) are
    neglected exactly as in the LAL reference; the barycentering uses the JPL
    ephemeris and matches ``XLALBarycenter`` to ~1e-13 s.

    Attributes:
        start_gps (int): Integer GPS second of ``t == 0`` on the call axis.
        n_spindowns (int): Number of spindown terms expected in ``params``
            (``f1`` … ``f{n}``).
        ref_time_ssb (Optional[float]): SSB reference epoch for the spin
            parameters (seconds). If ``None``, the SSB time of the first sample
            is used (LAL's default).
    """

    start_gps: int
    n_spindowns: int
    ref_time_ssb: Optional[float]

    def __init__(
        self,
        earth_ephemeris_file: str,
        start_gps: int,
        sun_ephemeris_file: Optional[str] = None,
        n_spindowns: int = 0,
        ref_time_ssb: Optional[float] = None,
    ) -> None:
        """
        Args:
            earth_ephemeris_file (str): Path to a LALPulsar Earth ephemeris
                file, or a standard name (e.g. ``"earth00-40-DE405.dat.gz"``)
                to download and cache automatically -- see
                :func:`~ripplegw.waveforms.cw.ephemeris.resolve_ephemeris_path`.
            start_gps (int): Integer GPS second corresponding to ``t == 0`` on
                the time axis passed to :meth:`__call__`.
            sun_ephemeris_file (Optional[str]): Unused by the exact model (the
                Shapiro term is neglected); accepted for interface symmetry.
            n_spindowns (int): Number of spindown parameters (``f1`` …).
            ref_time_ssb (Optional[float]): SSB reference epoch for the spins.

        Note:
            The observing site's location is not a constructor argument --
            pass it per call as ``site_x``, ``site_y``, ``site_z`` (metres,
            geocentric Cartesian) in ``params``; see :attr:`parameter_names`.
        """
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
            "site_x",
            "site_y",
            "site_z",
            *(f"f{i}" for i in range(1, self.n_spindowns + 1)),
        )

    def __call__(
        self, t: Float[Array, " n_time"], params: Mapping[str, FloatLike]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Detector-frame sample times in seconds
                relative to ``start_gps`` (e.g. ``jnp.arange(N) / fs``).
            params: Source parameters with keys ``alpha``,
                ``delta`` (sky position, rad), ``f0`` (Hz), ``phi0`` (rad),
                ``aplus``, ``across`` (polarization amplitudes), ``site_x``,
                ``site_y``, ``site_z`` (observing site's geocentric Cartesian
                location, metres -- fixed site metadata, not a quantity to
                sample/infer), and ``f1`` … ``f{n_spindowns}`` (spindowns, Hz/s,
                …).

        Returns:
            dict[str, Float[Array, " n_time"]]: Plus (``"p"``) and cross
                (``"c"``) polarizations.
        """
        fkdot = tuple(params[f"f{i}"] for i in range(1, self.n_spindowns + 1))
        det_location_m = (params["site_x"], params["site_y"], params["site_z"])
        hp, hc = exact_pulsar_polarizations(
            t,
            self.start_gps,
            params["alpha"],
            params["delta"],
            params["f0"],
            params["phi0"],
            params["aplus"],
            params["across"],
            det_location_m,
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
            f"ExactPulsarSignal(start_gps={self.start_gps}, "
            f"n_spindowns={self.n_spindowns})"
        )
