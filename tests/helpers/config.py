"""Constructor kwargs for waveforms that cannot be built with zero arguments.

Every built-in waveform except the continuous-wave (CW) family accepts
``ripplegw.waveform(name)`` with no extra kwargs -- that zero-arg-construction
assumption is baked into the generic test infrastructure (``conftest.py``,
``test_edge_cases.py``, ``test_output_format.py``, ``test_transforms.py``,
``docs/gen_api.py``). CW's constructors need a detector, an ephemeris file, and
a GPS epoch; there is no safe default epoch to guess (it would silently
barycenter at the wrong time with no error), so each CW name gets a minimal
viable config here instead, built against a small synthetic ephemeris so this
runs in CI with no LAL/ephemeris-data dependency.
"""

import tempfile
from pathlib import Path
from typing import Optional

_CW_NAMES = {"ExactPulsarSignal", "PulsarSignal", "BinaryPulsarSignal"}
_START_GPS = 1_000_000_000
_CACHE: Optional[tuple[str, str]] = None

# H1 geocentric vertex location (metres), from LALDetectors.h. ripple's CW
# waveforms take a raw location rather than looking up a detector by name.
H1_LOCATION = (-2.16141492636e06, -3.83469517889e06, 4.60035022664e06)


def _write_synthetic_ephemeris(path, pos, vel, acc, *, gps0, dt=7200.0, n=4):
    """Write a minimal, valid LALPulsar-format ephemeris table to ``path``."""
    lines = [f"2000 {dt} {n}"]
    for i in range(n):
        gps = gps0 + i * dt
        lines.append(
            f"{gps} {pos[0]} {pos[1]} {pos[2]} "
            f"{vel[0]} {vel[1]} {vel[2]} {acc[0]} {acc[1]} {acc[2]}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def synthetic_ephemeris_files() -> tuple[str, str]:
    """``(earth, sun)`` synthetic ephemeris file paths spanning ``_START_GPS``.

    Cached (module-level) and shared with
    ``tests/integration/test_cw_waveforms.py`` so every CW-touching test in the
    suite reads from the same two files.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    tmpdir = Path(tempfile.mkdtemp(prefix="ripple-cw-ephemeris-"))
    gps0 = _START_GPS - 7200.0
    earth = tmpdir / "earth-synthetic.dat"
    sun = tmpdir / "sun-synthetic.dat"
    _write_synthetic_ephemeris(
        earth,
        pos=(500.0, 30.0, -10.0),
        vel=(1e-4, -2e-5, 3e-5),
        acc=(0.0, 0.0, 0.0),
        gps0=gps0,
    )
    _write_synthetic_ephemeris(
        sun,
        pos=(2.0, 1.0, 0.5),
        vel=(1e-5, 2e-5, -1e-5),
        acc=(0.0, 0.0, 0.0),
        gps0=gps0,
    )
    _CACHE = (str(earth), str(sun))
    return _CACHE


def default_config(name: str) -> dict:
    """Minimal viable constructor kwargs for ``name``, or ``{}`` if none needed."""
    if name not in _CW_NAMES:
        return {}
    earth, sun = synthetic_ephemeris_files()
    config = {
        "detector_location": H1_LOCATION,
        "earth_ephemeris_file": earth,
        "start_gps": _START_GPS,
    }
    if name != "ExactPulsarSignal":
        config["sun_ephemeris_file"] = sun
    return config
