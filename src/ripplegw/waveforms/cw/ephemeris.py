"""Reader for LALPulsar JPL ephemeris files.

LALPulsar distributes Earth and Sun ephemerides as (optionally gzip-compressed)
ASCII tables derived from the JPL DE ephemerides (DE200/DE405/DE421/DE430/...).
This module parses those files into jax arrays so the rest of the
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

A standard ``earth*``/``sun*`` file isn't bundled with ripple (see
``docs/installation.md``); if the requested name isn't found on disk,
:func:`read_ephemeris_file` downloads it from the LALSuite repository and
caches it locally (see :func:`resolve_ephemeris_path`) rather than requiring
users to fetch it manually.
"""

import gzip
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

# Raw-file base for LALSuite's tracked copies of the standard ephemerides.
_LALSUITE_RAW_BASE = "https://git.ligo.org/lscsoft/lalsuite/-/raw/master/lalpulsar/lib"
# LALPulsar's standard ephemeris naming convention, e.g. "earth00-40-DE405.dat.gz".
# Auto-download is only ever attempted for a bare name matching this pattern --
# ripple never guesses at an arbitrary URL.
_EPHEMERIS_NAME_RE = re.compile(r"\A(earth|sun)\d{2}-\d{2}-DE\d{3}\.dat(\.gz)?\Z")


@dataclass(frozen=True)
class Ephemeris:
    """A tabulated position/velocity/acceleration ephemeris.

    Attributes:
        gps0 (float): GPS time of the first table entry (seconds).
        dt (float): Spacing between consecutive entries (seconds).
        pos (Float[Array, "n 3"]): Positions in light-seconds.
        vel (Float[Array, "n 3"]): Velocities (dimensionless, ``v/c``).
        acc (Float[Array, "n 3"]): Accelerations in ``1/s``.
        etype (str): Ephemeris type string parsed from the file name
            (e.g. ``"DE405"``), or ``"DE405"`` if it cannot be determined.
    """

    gps0: float
    dt: float
    pos: Float[Array, "n 3"]
    vel: Float[Array, "n 3"]
    acc: Float[Array, "n 3"]
    etype: str = "DE405"

    @property
    def n_entries(self) -> int:
        """Number of table entries."""
        return self.pos.shape[0]

    @property
    def gps_end(self) -> float:
        """GPS time of the last table entry (seconds)."""
        return self.gps0 + (self.n_entries - 1) * self.dt


def _cache_dir() -> str:
    """Local directory auto-downloaded ephemeris files are cached under.

    Override with the ``RIPPLEGW_CACHE_DIR`` environment variable; otherwise
    ``$XDG_CACHE_HOME/ripplegw/ephemeris`` (``~/.cache`` if unset), or
    ``~/Library/Caches/ripplegw/ephemeris`` on macOS.
    """
    override = os.environ.get("RIPPLEGW_CACHE_DIR")
    if override:
        base = override
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "ripplegw", "ephemeris")


def _download_to_cache(name: str) -> str:
    """Download a standard LALPulsar ephemeris file to the local cache.

    Downloads to a temporary file in the same directory and atomically
    renames it into place, so a failed or interrupted download never leaves
    behind a cache entry that looks valid but isn't. Returns the cached path.
    """
    cache_dir = _cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, name)
    if os.path.exists(dest):
        return dest
    url = f"{_LALSUITE_RAW_BASE}/{name}"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, prefix=f".{name}.", suffix=".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, os.fdopen(tmp_fd, "wb") as tmp_file:
            shutil.copyfileobj(response, tmp_file)
        os.replace(tmp_path, dest)
    except BaseException:
        os.remove(tmp_path)
        raise
    return dest


def resolve_ephemeris_path(name_or_path: str) -> str:
    """Resolve an ephemeris argument to a local path, downloading if needed.

    If ``name_or_path`` (or ``name_or_path + ".gz"``) already exists on disk,
    it is returned as-is -- no network access. Otherwise, if it is a bare
    filename matching LALPulsar's standard ``earth*``/``sun*`` naming
    convention (e.g. ``"earth00-40-DE405.dat.gz"``), it is downloaded from
    the LALSuite repository and cached locally (see :func:`_cache_dir`) so
    later calls reuse it without re-downloading. Anything else -- a path with
    a directory component, or a name that doesn't match the convention --
    raises ``FileNotFoundError`` without attempting a download: ripple never
    guesses at an arbitrary URL.

    Args:
        name_or_path (str): A local path, or a standard ephemeris file name.

    Returns:
        str: A local path to the (possibly newly cached) ephemeris file.

    Raises:
        FileNotFoundError: The file isn't local and isn't a recognised
            standard name, or downloading it failed.
    """
    if os.path.exists(name_or_path) or os.path.exists(name_or_path + ".gz"):
        return name_or_path
    basename = os.path.basename(name_or_path)
    if os.path.dirname(name_or_path) or not _EPHEMERIS_NAME_RE.match(basename):
        raise FileNotFoundError(
            f"Ephemeris file '{name_or_path}' not found, and it isn't a standard "
            "LALPulsar 'earth*'/'sun*' name ripple can fetch automatically (e.g. "
            "'earth00-40-DE405.dat.gz'). Pass an existing file path, or one of "
            "the standard names to have it downloaded and cached automatically."
        )
    try:
        return _download_to_cache(basename)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise FileNotFoundError(
            f"Ephemeris file '{name_or_path}' isn't cached locally, and "
            f"downloading it from {_LALSUITE_RAW_BASE}/{basename} failed: {exc}. "
            "See docs/installation.md to obtain it manually and pass the local "
            "path instead."
        ) from exc


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
            If ``path`` doesn't exist locally but is a standard LALPulsar
            name (e.g. ``"earth00-40-DE405.dat.gz"``), it is downloaded and
            cached automatically -- see :func:`resolve_ephemeris_path`.

    Returns:
        Ephemeris: Parsed table.

    Raises:
        FileNotFoundError: ``path`` doesn't exist, isn't a recognised
            standard name to fetch automatically, or downloading it failed.
        ValueError: If the header or row count is inconsistent with the data.
    """
    path = resolve_ephemeris_path(path)
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
    if n_entries <= 0:
        raise ValueError(
            f"Ephemeris file '{path}': header declares non-positive entry count "
            f"n_entries={n_entries}"
        )
    body = tokens[3:]

    if len(body) != 10 * n_entries:
        raise ValueError(
            f"Ephemeris file '{path}': header declares {n_entries} entries "
            f"(={10 * n_entries} values) but found {len(body)} data values"
        )

    data = jnp.asarray(body, dtype=jnp.float64).reshape(n_entries, 10)
    gps = data[:, 0]
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    acc = data[:, 7:10]

    gps0 = float(gps[0])
    # Sanity check on uniform spacing across ALL entries (LAL enforces this
    # strictly); the barycentering index lookup assumes a constant dt.
    if n_entries > 1:
        diffs = jnp.diff(gps)
        if not jnp.allclose(diffs, dt, rtol=0, atol=1e-6):
            bad = int(jnp.argmax(jnp.abs(diffs - dt)))
            raise ValueError(
                f"Ephemeris file '{path}': non-uniform GPS spacing; header dt={dt} "
                f"but entry {bad}->{bad + 1} has spacing {float(diffs[bad])}"
            )

    return Ephemeris(
        gps0=gps0,
        dt=dt,
        pos=pos,
        vel=vel,
        acc=acc,
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
