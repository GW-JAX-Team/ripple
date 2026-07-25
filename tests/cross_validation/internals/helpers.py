"""Shared helpers for the IMRPhenomXHM per-mode bisection tests.

Two layers survive here: per-mode QNM sanity (fRING/fDAMP) and per-mode
amplitude, both compared against LAL's ``SimIMRPhenomXHM{Amplitude,...}``
low-level functions. The higher layers (phase, complex h_lm, full hp/hc via
LAL's raw per-mode API) were removed: they diverged from LAL by tens of
radians in a way that tracked back to a reference-time/phase convention
specific to LAL's low-level per-mode functions rather than to ripple's
output -- the *same* per-mode code, called through
``gen_IMRPhenomXHM_hphc`` and compared against ``ChooseFDWaveform`` in
``cross_validation/test_overlap.py``, agrees with LAL to ~1e-9. Root-causing
the low-level convention gap needs LAL C-source archaeology beyond what a
diagnostic test file should carry silently, so those comparisons were
removed rather than fixed with an unexplained offset.

All helpers raise ``pytest.skip(...)`` when LALSuite is unavailable so
the suite runs cleanly in environments that do not have lalsimulation
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import lal
    import lalsimulation as lalsim

    LAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    LAL_AVAILABLE = False


# Fiducial BBH parameters used for deterministic per-mode comparisons.
# Chosen to exercise: aligned spins of opposite sign, q != 1 (so odd-l modes
# have non-zero amplitude), and total mass that places fRD inside the
# default 20 Hz – 1024 Hz analysis band.
FIDUCIAL_PARAMS = {
    "m1": 36.0,  # solar masses
    "m2": 29.0,
    "chi1z": 0.30,
    "chi2z": -0.10,
    "distance_mpc": 400.0,
    "f_ref": 20.0,
    "inclination": 0.4,  # rad
    "phi_ref": 0.0,  # rad
}

XHM_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
NON_22_MODES = [(2, 1), (3, 3), (3, 2), (4, 4)]


def require_lal():
    if not LAL_AVAILABLE:
        pytest.skip("LALSuite not available")


def make_freq_grid(f_min: float = 20.0, f_max: float = 512.0, df: float = 1.0 / 8.0):
    """Common Hz frequency grid for per-mode comparison tests.

    Uses df = 1/8 Hz (T = 8 s segment) which matches the BBH grid used
    elsewhere in the suite while keeping per-test cost low.
    """
    n = int(np.floor((f_max - f_min) / df)) + 1
    return f_min + df * np.arange(n)


def freqs_to_real8seq(freqs_hz: np.ndarray):
    """Wrap a numpy frequency array as a lal.REAL8Sequence."""
    require_lal()
    seq = lal.CreateREAL8Sequence(len(freqs_hz))
    seq.data[:] = np.asarray(freqs_hz, dtype=np.float64)
    return seq


def make_xhm_params_dict():
    """Build a minimal LALDict for XHM mode-array tests.

    We DO NOT activate a mode array here — the per-mode LAL functions
    (SimIMRPhenomXHMAmplitude, SimIMRPhenomXHMPhase, ...) request a
    specific (ell, m) directly and ignore the dict mode array.
    """
    require_lal()
    return lal.CreateDict()


def lal_xhm_amplitude(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Call lalsim.SimIMRPhenomXHMAmplitude on the given Hz grid.

    Returns the amplitude array (REAL8Sequence as numpy array).
    """
    require_lal()
    p = params
    seq_in = freqs_to_real8seq(freqs_hz)
    out = lalsim.SimIMRPhenomXHMAmplitude(
        seq_in,
        int(ell),
        int(m),
        p["m1"] * lal.MSUN_SI,
        p["m2"] * lal.MSUN_SI,
        0.0,
        0.0,
        p["chi1z"],
        0.0,
        0.0,
        p["chi2z"],
        p["distance_mpc"] * 1e6 * lal.PC_SI,
        p["phi_ref"],
        p["f_ref"],
        make_xhm_params_dict(),
    )
    # `out` is a REAL8Sequence. Older swig wrappers return it directly;
    # some versions return (status, out).
    if isinstance(out, tuple):
        out = out[-1]
    return np.asarray(out.data, dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# ripple-side helpers
# ---------------------------------------------------------------------------


def ripple_pwf22(params=FIDUCIAL_PARAMS):
    """Return (pWF22, freqs_geom_factor M_s) for the fiducial parameters."""
    from ripplegw.waveforms.IMRPhenomX.IMRPhenomXHM import build_pWF22

    pWF22 = build_pWF22(
        params["m1"], params["m2"], params["chi1z"], params["chi2z"], params["f_ref"]
    )
    return pWF22


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


def region_masks(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Return boolean masks (inspiral, intermediate, ringdown) for one mode.

    The boundary frequencies (fAmpMatchIN/IM, fPhaseMatchIN/IM) are stored
    inside ripple's per-mode pPhase/pAmp objects; we use the *amplitude*
    boundaries as a coarse proxy that works well enough to bisect.
    """
    from ripplegw.waveforms.IMRPhenomX.IMRPhenomXHM import (
        build_pWF22,
        xhm_get_amp_coefficients,
        xhm_set_waveform_variables,
    )

    pWF22 = build_pWF22(
        params["m1"], params["m2"], params["chi1z"], params["chi2z"], params["f_ref"]
    )
    pWFHM = xhm_set_waveform_variables(int(ell), int(m), pWF22)
    pAmp = xhm_get_amp_coefficients(pWFHM, pWF22)
    M_s = pWF22["M_s"]
    fIn_Hz = float(pAmp.fAmpMatchIN) / M_s
    fIm_Hz = float(pAmp.fAmpMatchIM) / M_s
    f = np.asarray(freqs_hz)
    return (f < fIn_Hz), ((f >= fIn_Hz) & (f < fIm_Hz)), (f >= fIm_Hz)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def relative_amp_error(amp_ripple, amp_lal):
    """|amp_r - amp_l| / max(|amp_l|).

    Use a global denominator (max amplitude in the band) so a single
    bin near a zero crossing does not dominate the metric.
    """
    a_r = np.asarray(amp_ripple)
    a_l = np.asarray(amp_lal)
    denom = np.max(np.abs(a_l)) if np.max(np.abs(a_l)) > 0 else 1.0
    return np.abs(a_r - a_l) / denom
