"""Reader for LALPulsar JPL ephemeris files.

LALPulsar distributes Earth and Sun ephemerides as (optionally gzip-compressed)
ASCII tables derived from the JPL DE ephemerides (DE200/DE405/DE421/DE430/...).
This module parses those files into plain numpy arrays so the rest of the
continuous-wave machinery can run in JAX without any LAL runtime dependency.

File format (see ``XLALReadEphemerisFile`` in LALInitBarycenter.c):

* arbitrary leading comment lines beginning with ``#``;
* one header line ``gpsYr  dt  nEntries``;
* then ``nEntries`` rows. Each row holds 10 floats::

      gps  pos_x pos_y pos_z  vel_x vel_y vel_z  acc_x acc_y acc_z

  Older files spread each entry over 4 lines; both layouts are supported.

All quantities are in *natural* units used by LAL barycentering: positions in
light-seconds, velocities dimensionless (``v/c``) and accelerations in
``1/s``. The first column is the GPS time of the entry in seconds.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np

__all__ = ["Ephemeris", "read_ephemeris_file", "load_ephemeris"]


@dataclass(frozen=True)
class Ephemeris:
    """A tabulated position/velocity/acceleration ephemeris.

    Attributes:
        gps0 (float): GPS time of the first table entry (seconds).
        dt (float): Spacing between consecutive entries (seconds).
        pos (np.ndarray): ``(N, 3)`` positions in light-seconds.
        vel (np.ndarray): ``(N, 3)`` velocities (dimensionless, ``v/c``).
        acc (np.ndarray): ``(N, 3)`` accelerations in ``1/s``.
        etype (str): Ephemeris type string parsed from the file name
            (e.g. ``"DE405"``), or ``"DE405"`` if it cannot be determined.
    """

    gps0: float
    dt: float
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    etype: str = "DE405"

    @property
    def n_entries(self) -> int:
        """Number of table entries."""
        return self.pos.shape[0]

    @property
    def gps_end(self) -> float:
        """GPS time of the last table entry (seconds)."""
        return self.gps0 + (self.n_entries - 1) * self.dt


def _open_maybe_gzip(path: str):
    """Open ``path`` transparently handling gzip compression.

    Tries ``path`` then ``path + ".gz"`` (mirroring LAL's behaviour), and
    detects gzip content by magic bytes rather than relying on the extension.
    """
    candidates = [path]
    if not path.endswith(".gz"):
        candidates.append(path + ".gz")
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "rb") as fh:
                magic = fh.read(2)
            if magic == b"\x1f\x8b":
                return gzip.open(cand, "rt")
            return open(cand, "rt")
    raise FileNotFoundError(f"Could not find ephemeris file '{path}[.gz]'")


def _etype_from_name(name: str) -> str:
    for tag in ("DE200", "DE405", "DE414", "DE421", "DE430"):
        if tag in name:
            return tag
    return "DE405"


def read_ephemeris_file(path: str) -> Ephemeris:
    """Parse a single LALPulsar ephemeris file into an :class:`Ephemeris`.

    Args:
        path (str): Path to an ``earth*`` or ``sun*`` ephemeris file. A
            ``.gz`` suffix is added automatically if the bare name is absent.

    Returns:
        Ephemeris: Parsed table.

    Raises:
        FileNotFoundError: If neither ``path`` nor ``path + ".gz"`` exists.
        ValueError: If the header or row count is inconsistent with the data.
    """
    with _open_maybe_gzip(path) as fh:
        tokens: list[str] = []
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens.extend(stripped.split())

    if len(tokens) < 3:
        raise ValueError(f"Ephemeris file '{path}' has no header line")

    # Header: gpsYr dt nEntries (gpsYr is not needed for interpolation)
    dt = float(tokens[1])
    n_entries = int(float(tokens[2]))
    body = tokens[3:]

    if len(body) != 10 * n_entries:
        raise ValueError(
            f"Ephemeris file '{path}': header declares {n_entries} entries "
            f"(={10 * n_entries} values) but found {len(body)} data values"
        )

    data = np.asarray(body, dtype=np.float64).reshape(n_entries, 10)
    gps = data[:, 0]
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    acc = data[:, 7:10]

    gps0 = float(gps[0])
    # Sanity check on uniform spacing (LAL enforces this strictly).
    if n_entries > 1:
        measured_dt = float(gps[1] - gps[0])
        if not np.isclose(measured_dt, dt, rtol=0, atol=1e-6):
            raise ValueError(
                f"Ephemeris file '{path}': header dt={dt} disagrees with "
                f"measured spacing {measured_dt}"
            )

    return Ephemeris(
        gps0=gps0,
        dt=dt,
        pos=np.ascontiguousarray(pos),
        vel=np.ascontiguousarray(vel),
        acc=np.ascontiguousarray(acc),
        etype=_etype_from_name(os.path.basename(path)),
    )


def load_ephemeris(
    earth_file: str, sun_file: str | None = None
) -> tuple[Ephemeris, Ephemeris | None]:
    """Load an Earth ephemeris and (optionally) a Sun ephemeris.

    The Sun ephemeris is only required for the Shapiro-delay term, which the
    exact CW signal generator neglects; it is therefore optional here.

    Args:
        earth_file (str): Path to the Earth ephemeris file.
        sun_file (str | None): Path to the Sun ephemeris file, or ``None``.

    Returns:
        tuple[Ephemeris, Ephemeris | None]: ``(earth, sun)`` where ``sun`` is
            ``None`` if ``sun_file`` was not provided.
    """
    earth = read_ephemeris_file(earth_file)
    sun = read_ephemeris_file(sun_file) if sun_file is not None else None
    return earth, sun
