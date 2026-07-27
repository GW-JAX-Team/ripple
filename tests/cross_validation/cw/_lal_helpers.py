"""Shared helpers for CW-vs-LALPulsar cross-validation (tests/cross_validation/cw/).

LAL does not SWIG-wrap ``PulsarSignalParams``, so ``XLALSimulateExactPulsarSignal``/
``XLALGeneratePulsarSignal`` cannot be called directly from Python. Every test in this
directory instead reproduces the relevant LAL reference computation in Python from its
SWIG-exposed building blocks (``XLALGetDetectorStates``, ``XLALComputeAMCoeffs``,
``XLALBarycenter``, ``XLALGenerateSpinOrbitCW``) -- this is why CW has its own directory
here instead of a ``ReferenceBackend`` in ``reference/`` like the campaign in
``cross_validation/fd/``. See ``docs/dev/reference_implementations.md`` for the per-model methodology
and accuracy numbers.

Not itself a test module (no ``test_`` prefix, not collected by pytest) -- just the
pieces every file in this directory would otherwise duplicate: locating the ephemeris
files, and the white (unweighted) time-domain overlap-loss metric these tests use instead
of ``tests.helpers.metrics.overlap_loss`` (which requires a PSD and a frequency axis).
Each test module still does its own ``pytest.importorskip("lal")`` / ``"lalpulsar"`` and
``pytest.mark.skipif`` on the ephemeris files -- a module-level skip in a shared
``conftest.py`` does not degrade gracefully when the import itself is what's missing (it
raises during pytest's initial conftest loading instead of being reported as a skip), so
each module keeps its own skip guard rather than centralizing it there.
"""

import math
import os

import numpy as np


def find_ephemeris():
    """Locate Earth and Sun ephemeris files (both must exist), or ``(None, None)``.

    ``lalpulsar.InitBarycenter`` requires both files, so this only returns a pair
    when both are present (otherwise the caller should skip rather than hard-fail).
    """
    earth = os.environ.get("RIPPLE_EARTH_EPHEMERIS")
    if earth and os.path.exists(earth):
        # Use RIPPLE_SUN_EPHEMERIS if given, else guess the Sun file next to it.
        sun = os.environ.get("RIPPLE_SUN_EPHEMERIS") or earth.replace("earth", "sun")
        if os.path.exists(sun):
            return earth, sun
    candidates = [
        "earth00-40-DE405.dat.gz",
        "/usr/share/lalpulsar/earth00-40-DE405.dat.gz",
    ]
    for c in candidates:
        s = c.replace("earth", "sun")
        if os.path.exists(c) and os.path.exists(s):
            return c, s
    return None, None


def overlap_loss(h1, h2) -> float:
    """1 - normalized overlap between two real time series (white inner product).

    Uses the same numerically stable identity as
    ``tests.helpers.metrics.overlap_loss`` --
    ``(AB - C**2) / (sqrt(AB)*(sqrt(AB) + C))`` with ``A=<h1|h1>`` etc. -- but with a flat
    (white) time-domain inner product, the conventional figure of merit for these
    quasi-monochromatic continuous-wave strain series. Returns the mismatch; report
    ``log10`` of it via :func:`log10_str`.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    a = float(h1 @ h1)
    b = float(h2 @ h2)
    c = float(h1 @ h2)
    denom = math.sqrt(a * b)
    return max((a * b - c * c) / (denom * (denom + c)), 0.0)


def log10_str(loss: float) -> str:
    """``log10(loss)`` formatted, or 'N/A' for an exactly-zero (clamped) loss."""
    return f"{math.log10(loss):.2f}" if loss > 0.0 else "N/A (0)"
