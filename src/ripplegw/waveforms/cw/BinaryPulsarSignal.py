"""ripple ``Waveform``-compatible interface for the binary pulsar signal."""

from collections.abc import Mapping

from jaxtyping import Array, Float

from ripplegw.registry import register
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cw.pulsar_signal import generate_binary_pulsar_polarizations
from ripplegw.waveforms.cw.PulsarSignal import PulsarSignal


@register("BinaryPulsarSignal", is_binary=True)
class BinaryPulsarSignal(PulsarSignal):
    """Full ``XLALGeneratePulsarSignal`` polarizations for a pulsar in a binary.

    Adds the elliptic-orbit modulation of ``XLALGenerateSpinOrbitCW`` on top of
    :class:`~ripplegw.waveforms.cw.PulsarSignal.PulsarSignal`. The orbital
    elements are supplied per call as parameters (so they too are
    differentiable).
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
            "site_x",
            "site_y",
            "site_z",
            *(f"f{i}" for i in range(1, self.n_spindowns + 1)),
        )

    def __call__(
        self, t: Float[Array, " n_time"], params: Mapping[str, FloatLike]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the binary-pulsar polarizations.

        Args:
            t (Float[Array, " n_time"]): Sample times (s) relative to ``start_gps``.
            params: The isolated keys plus orbital elements
                ``asini`` (light-s), ``ecc``, ``period`` (s), ``argp`` (rad),
                ``tp_ssb`` (SSB periapsis time, GPS seconds), and ``site_x``,
                ``site_y``, ``site_z`` (observing site's geocentric Cartesian
                location, metres -- fixed site metadata, not a quantity to
                sample/infer).

        Returns:
            dict[str, Float[Array, " n_time"]]: ``{"p", "c"}`` polarizations.
        """
        fkdot = tuple(params[f"f{i}"] for i in range(1, self.n_spindowns + 1))
        det_location_m = (params["site_x"], params["site_y"], params["site_z"])
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
            det_location_m,
            *self._e,
            *self._s,
            fkdot=fkdot,
            ref_time_ssb=self.ref_time_ssb,
            f_heterodyne=self.f_heterodyne,
        )
        return {"p": hp, "c": hc}

    def __repr__(self) -> str:
        return (
            f"BinaryPulsarSignal(start_gps={self.start_gps}, "
            f"n_spindowns={self.n_spindowns})"
        )
