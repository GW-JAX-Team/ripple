"""Shared helpers for the XHM / XPHM bisection tests.

These tests follow the pyramid in `debug_plan_xhm.md`:
each layer compares ripple to LAL at a progressively higher level of
abstraction (per-mode amplitude → phase → complex hlm → hp/hc).

All helpers raise ``pytest.skip(...)`` when LALSuite is unavailable so
the suite runs cleanly in environments that do not have lalsimulation
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
FIDUCIAL_PARAMS = dict(
    m1=36.0,  # solar masses
    m2=29.0,
    chi1z=0.30,
    chi2z=-0.10,
    distance_mpc=400.0,
    f_ref=20.0,
    inclination=0.4,  # rad
    phi_ref=0.0,  # rad
)

# Fiducial precessing parameters (for XPHM tests).
FIDUCIAL_PARAMS_PREC = dict(
    m1=36.0,
    m2=29.0,
    chi1x=0.10,
    chi1y=0.05,
    chi1z=0.30,
    chi2x=-0.05,
    chi2y=0.07,
    chi2z=-0.10,
    distance_mpc=400.0,
    f_ref=20.0,
    inclination=0.4,
    phi_ref=0.0,
)

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


def lal_xhm_phase(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Call lalsim.SimIMRPhenomXHMPhase on the given Hz grid."""
    require_lal()
    p = params
    seq_in = freqs_to_real8seq(freqs_hz)
    out = lalsim.SimIMRPhenomXHMPhase(
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
    if isinstance(out, tuple):
        out = out[-1]
    return np.asarray(out.data, dtype=np.float64).copy()


def lal_xhm_one_mode_freqseq(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Call lalsim.SimIMRPhenomXHMFrequencySequenceOneMode.

    Returns the complex h_{l,-m}(f) on the input grid.
    """
    require_lal()
    p = params
    seq_in = freqs_to_real8seq(freqs_hz)
    out = lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        seq_in,
        p["m1"] * lal.MSUN_SI,
        p["m2"] * lal.MSUN_SI,
        p["chi1z"],
        p["chi2z"],
        int(ell),
        int(m),
        p["distance_mpc"] * 1e6 * lal.PC_SI,
        p["phi_ref"],
        p["f_ref"],
        make_xhm_params_dict(),
    )
    if isinstance(out, tuple):
        out = out[-1]
    return np.asarray(out.data.data, dtype=np.complex128).copy()


def lal_xhm_full(freqs_hz, params=FIDUCIAL_PARAMS):
    """Call lalsim.SimIMRPhenomXHM (hp, hc) on a uniform grid in Hz.

    The LAL API requires (f_min, f_max, deltaF), so the input grid must
    be uniform.  The returned arrays are masked to the input grid range.
    """
    require_lal()
    p = params
    df = float(freqs_hz[1] - freqs_hz[0])
    f_min = float(freqs_hz[0])
    f_max = float(freqs_hz[-1])
    hp, hc = lalsim.SimIMRPhenomXHM(
        p["m1"] * lal.MSUN_SI,
        p["m2"] * lal.MSUN_SI,
        p["chi1z"],
        p["chi2z"],
        f_min,
        f_max,
        df,
        p["distance_mpc"] * 1e6 * lal.PC_SI,
        p["inclination"],
        p["phi_ref"],
        p["f_ref"],
        make_xhm_params_dict(),
    )
    n = len(hp.data.data)
    freqs_lal = np.arange(n) * df
    mask = (freqs_lal >= f_min - 0.5 * df) & (freqs_lal <= f_max + 0.5 * df)
    return np.asarray(hp.data.data[mask]), np.asarray(hc.data.data[mask])


def lal_xphm_one_mode_freqseq(freqs_hz, ell, m, params=FIDUCIAL_PARAMS_PREC):
    """Call SimIMRPhenomXPHMFrequencySequenceOneMode -> (h^J pos, h^J neg).

    Returns the positive-frequency J-frame strain.
    """
    require_lal()
    p = params
    seq_in = freqs_to_real8seq(freqs_hz)
    # mirror tests/utils.py XPHM construction (PrecVersion 222, MSA twist)
    lalparams = lal.CreateDict()
    ModeArray = lalsim.SimInspiralCreateModeArray()
    for el, em in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
        lalsim.SimInspiralModeArrayActivateMode(ModeArray, el, em)
    lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(lalparams, 1)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(lalparams, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalparams, 0.0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 222)

    hpos, hneg = lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode(
        seq_in,
        int(ell),
        int(m),
        p["m1"] * lal.MSUN_SI,
        p["m2"] * lal.MSUN_SI,
        p["chi1x"],
        p["chi1y"],
        p["chi1z"],
        p["chi2x"],
        p["chi2y"],
        p["chi2z"],
        p["distance_mpc"] * 1e6 * lal.PC_SI,
        p["inclination"],
        p["phi_ref"],
        p["f_ref"],
        lalparams,
    )
    return (
        np.asarray(hpos.data.data, dtype=np.complex128).copy(),
        np.asarray(hneg.data.data, dtype=np.complex128).copy(),
    )


def lal_xphm_full(freqs_hz, params=FIDUCIAL_PARAMS_PREC):
    """Call lalsim.SimIMRPhenomXPHM on a uniform Hz grid (PrecVersion=222)."""
    require_lal()
    p = params
    df = float(freqs_hz[1] - freqs_hz[0])
    f_min = float(freqs_hz[0])
    f_max = float(freqs_hz[-1])
    lalparams = lal.CreateDict()
    ModeArray = lalsim.SimInspiralCreateModeArray()
    for el, em in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
        lalsim.SimInspiralModeArrayActivateMode(ModeArray, el, em)
    lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(lalparams, 1)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(lalparams, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalparams, 0.0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 222)

    hp, hc = lalsim.SimIMRPhenomXPHM(
        p["m1"] * lal.MSUN_SI,
        p["m2"] * lal.MSUN_SI,
        p["chi1x"],
        p["chi1y"],
        p["chi1z"],
        p["chi2x"],
        p["chi2y"],
        p["chi2z"],
        p["distance_mpc"] * 1e6 * lal.PC_SI,
        p["inclination"],
        p["phi_ref"],
        f_min,
        f_max,
        df,
        p["f_ref"],
        lalparams,
    )
    n = len(hp.data.data)
    freqs_lal = np.arange(n) * df
    mask = (freqs_lal >= f_min - 0.5 * df) & (freqs_lal <= f_max + 0.5 * df)
    return np.asarray(hp.data.data[mask]), np.asarray(hc.data.data[mask])


# ---------------------------------------------------------------------------
# ripple-side helpers
# ---------------------------------------------------------------------------


def ripple_pwf22(params=FIDUCIAL_PARAMS):
    """Return (pWF22, freqs_geom_factor M_s) for the fiducial parameters."""
    from ripplegw.waveforms.IMRPhenomXHM import build_pWF22

    pWF22 = build_pWF22(
        params["m1"], params["m2"], params["chi1z"], params["chi2z"], params["f_ref"]
    )
    return pWF22


def ripple_xhm_one_mode_complex(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Return ripple's complex h_{l,m} on the given Hz grid.

    Includes the same overall amp0 distance prefactor and (-1)^l sign that
    LAL's SimIMRPhenomXHMFrequencySequenceOneMode applies, so the result
    is directly comparable to LAL.
    """
    import jax.numpy as jnp
    from ripplegw.waveforms.IMRPhenomXHM import (
        XLALSimIMRPhenomXHMGethlmModes,
        build_pWF22,
    )
    from ripplegw.constants import MTSUN, MRSUN, MPC

    Mtot = params["m1"] + params["m2"]
    M_s = Mtot * MTSUN
    dist_m = params["distance_mpc"] * MPC
    amp0 = Mtot * MRSUN * Mtot * MTSUN / dist_m
    minus1l = 1 if ell % 2 == 0 else -1

    freqs_geom = jnp.array(freqs_hz) * M_s
    pWF22 = build_pWF22(
        params["m1"], params["m2"], params["chi1z"], params["chi2z"], params["f_ref"]
    )
    hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
        freqs_geom, pWF22, params["phi_ref"], [(int(ell), int(m))]
    )
    return np.asarray(hlm_dict[(int(ell), int(m))]) * amp0 * minus1l


def ripple_xhm_hphc(freqs_hz, params=FIDUCIAL_PARAMS):
    """Return ripple's hp, hc for the fiducial XHM parameters on a Hz grid."""
    import jax.numpy as jnp
    from ripplegw.waveforms.IMRPhenomXHM import gen_IMRPhenomXHM_hphc

    theta = jnp.array(
        [
            params["m1"],
            params["m2"],
            params["chi1z"],
            params["chi2z"],
            params["distance_mpc"],
            0.0,
            params["phi_ref"],
            params["inclination"],
        ]
    )
    hp, hc = gen_IMRPhenomXHM_hphc(jnp.array(freqs_hz), theta, params["f_ref"])
    return np.asarray(hp), np.asarray(hc)


def ripple_xphm_hphc(freqs_hz, params=FIDUCIAL_PARAMS_PREC):
    """Return ripple's hp, hc for fiducial XPHM parameters on a Hz grid."""
    import jax.numpy as jnp
    from ripplegw.waveforms.IMRPhenomXPHM import generate_xphm

    hp, hc = generate_xphm(
        params["m1"],
        params["m2"],
        params["chi1x"],
        params["chi1y"],
        params["chi1z"],
        params["chi2x"],
        params["chi2y"],
        params["chi2z"],
        params["distance_mpc"],
        params["inclination"],
        params["phi_ref"],
        jnp.array(freqs_hz),
        params["f_ref"],
    )
    return np.asarray(hp), np.asarray(hc)


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


def region_masks(freqs_hz, ell, m, params=FIDUCIAL_PARAMS):
    """Return boolean masks (inspiral, intermediate, ringdown) for one mode.

    The boundary frequencies (fAmpMatchIN/IM, fPhaseMatchIN/IM) are stored
    inside ripple's per-mode pPhase/pAmp objects; we use the *amplitude*
    boundaries as a coarse proxy that works well enough to bisect.
    """
    from ripplegw.waveforms.IMRPhenomXHM import (
        build_pWF22,
        xhm_set_waveform_variables,
        xhm_get_amp_coefficients,
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


@dataclass
class RegionStats:
    name: str
    n: int
    max_abs: float
    max_rel: float


def per_region_max_abs(values, masks, names=("inspiral", "intermediate", "ringdown")):
    """Return per-region max(|values|) for a real array."""
    out = []
    v = np.asarray(values)
    for nm, mask in zip(names, masks):
        if not np.any(mask):
            out.append(RegionStats(nm, 0, 0.0, 0.0))
            continue
        sel = v[mask]
        out.append(RegionStats(nm, int(mask.sum()), float(np.max(np.abs(sel))), 0.0))
    return out


def relative_amp_error(amp_ripple, amp_lal):
    """|amp_r - amp_l| / max(|amp_l|).

    Use a global denominator (max amplitude in the band) so a single
    bin near a zero crossing does not dominate the metric.
    """
    a_r = np.asarray(amp_ripple)
    a_l = np.asarray(amp_lal)
    denom = np.max(np.abs(a_l)) if np.max(np.abs(a_l)) > 0 else 1.0
    return np.abs(a_r - a_l) / denom


def unwrap_after_subtract(phase_diff):
    """Unwrap a 2*pi-ambiguous phase difference."""
    return np.unwrap(np.asarray(phase_diff))
