"""Compare individual higher-order modes with a reference backend.

``test_overlap.py`` compares the full mode-summed polarizations; this module
isolates one spherical-harmonic mode at a time.  A regression confined to a
single subdominant mode is easily hidden in the mode-summed overlap (the (2,2)
mode dominates the SNR), so testing mode-by-mode is considerably more
sensitive.

What "one mode" means here
--------------------------
For each requested ``(l, m)`` with ``m > 0``, both codebases are asked for the
polarizations built from that mode alone.  On the reference side this is a
``ModeArray`` containing only ``(l, m)``; on the ripple side the mode's
``h_lm`` is evaluated and projected with the harmonic factors LAL uses on that
same single-mode path (which, for IMRPhenomXHM, are *not* the ones the full
generator uses -- see "Polarization conventions" below).

For the precessing model (IMRPhenomXPHM) the selected mode is the
*co-precessing* frame mode, which is then twisted up into the inertial frame --
this matches LAL's ``ModeArray`` semantics for XPHM (``ModeArrayJframe``
selects inertial modes and is not used here).

Because the overlap is scale-invariant, what this test is really sensitive to
is the mode's frequency-dependent amplitude and phase, not its overall
normalization.

Adding a waveform
-----------------
Register one :class:`HOMWaveform` in ``HOM_WAVEFORMS`` giving the modes each
codebase implements plus two callables: a reference single-mode generator and
a ripple single-mode generator factory.  Add a ``[lal_modes.<name>]`` block to
``tolerances.toml``.  Nothing else here needs to change.

Selecting modes
---------------
``--hom-modes 22,33,44`` restricts which modes are parametrized (default:
22,21,33,32,44,43).  Select a model the usual pytest way, ``-k IMRPhenomXHM``.
Modes a waveform does not implement are not silently dropped: they are skipped
with a reason naming the mode and the codebase (ripple, reference, or both)
that lacks it, so ``-rs`` reports every gap.

Polarization conventions: LAL's ModeArray path is not the full-waveform path
---------------------------------------------------------------------------
For ``IMRPhenomXHM``, LAL assembles single-mode polarizations differently from
the way it assembles the full waveform, so ``_ripple_xhm_mode`` follows the
*single-mode* convention rather than restricting ``gen_IMRPhenomXHM_hphc``:

* Default (unrestricted) call -- both codebases symmetrize the ``(l, -m)``
  partner, giving an inclination-dependent ``hx/h+``.  ripple's full XHM
  reproduces LAL's here exactly, amplitude included.
* Restricted ``ModeArray`` call -- LAL keeps only the ``(l, m)`` harmonic::

      h+ = 0.5 * (-i)^m * (-1)^l * F_{l,m}(iota) * h_lm,     hx = -i h+

  (verified against LAL for every mode at several inclinations).

``IMRPhenomHM`` symmetrizes on both paths and ripple's
``get_phenomHMFD_mode_projection`` matches it as-is; ``IMRPhenomXPHM`` reaches
the polarizations through the twist-up rather than ``IMRPhenomXHMFDAddMode``,
which is consistent between the two codebases.

Comparing the symmetrized ripple factors against LAL's unsymmetrized
single-mode output leaves ``h+`` agreeing up to a positive scale (invisible to
a scale-invariant overlap) while ``hx`` picks up a negative factor for
iota > pi/2, i.e. an overlap of exactly -1 and an overlap loss of 2.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripplegw.constants import MPC, MRSUN, MSUN, MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from tests.cross_validation.runner import (
    TestResult as _TestResult,  # aliased: pytest would try to collect a 'Test*' class
    default_grid,
    get_tolerance,
    load_psd,
    load_tolerances,
    plot_results,
    write_results,
)
from tests.helpers.grids import Grid
from tests.helpers.metrics import get_nyquist_mask, overlap_loss
from tests.helpers.params import random_params_batch

Mode = tuple[int, int]

# Per-mode limits live under their own backend key so the [lal.*] table (which
# test_tolerance_table.py checks against the docs) is untouched.
_TOLERANCE_BACKEND = "lal_modes"
_TOLERANCES = load_tolerances()

DEFAULT_MODES: tuple[Mode, ...] = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))


# ============================================================================
# Waveform registry
# ============================================================================


@dataclass(frozen=True)
class HOMWaveform:
    """What the per-mode test needs to know about one waveform.

    Args:
        name: Registered ripple waveform name; also the reference approximant.
        ripple_modes: ``(l, m)`` modes ripple implements, in the order its
            generator uses internally.
        reference_modes: ``(l, m)`` modes the reference backend implements.
        reference_single_mode: ``(params, mode, grid) -> (hp, hc)``.
        ripple_single_mode: ``(grid, mode) -> fn(params) -> (hp, hc)``, where
            the returned callable is JIT-compiled and vmap-able over a batch of
            ripple-native parameter dicts.
    Per-mode threshold overrides come from ``[lal_modes.<name>.modes]`` in
    ``tolerances.toml``, keyed by the compact ``"lm"`` string.
    """

    name: str
    ripple_modes: tuple[Mode, ...]
    reference_modes: tuple[Mode, ...]
    reference_single_mode: Callable
    ripple_single_mode: Callable

    def threshold(self, mode: Mode) -> float:
        """Per-mode overlap-loss limit, falling back to the waveform's own."""
        entry = _TOLERANCES.get(_TOLERANCE_BACKEND, {}).get(self.name, {})
        override = entry.get("modes", {}).get(f"{mode[0]}{mode[1]}")
        if override is not None:
            return float(override)
        return get_tolerance(_TOLERANCES, _TOLERANCE_BACKEND, self.name, "overlap_loss")

    def missing_from(self, mode: Mode) -> list[str]:
        """Codebases that do not implement ``mode`` (may be empty)."""
        missing = []
        if mode not in self.ripple_modes:
            missing.append("ripple")
        if mode not in self.reference_modes:
            missing.append("reference")
        return missing


# ============================================================================
# Reference (LAL) single-mode generation
# ============================================================================


def _mode_array(mode: Mode):
    """A LAL ModeArray containing a single ``(l, m)``."""
    import lalsimulation as lalsim

    array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(array, mode[0], mode[1])
    return array


def _trim(hp, hc, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Slice LAL's 0-based series down to the analysis band, as LALBackend does."""
    freqs = np.arange(len(hp.data.data)) * grid.df
    mask = (freqs > grid.f_l) & (freqs < grid.f_u)
    return hp.data.data[mask], hc.data.data[mask]


def _masses(params: dict) -> tuple[float, float]:
    m1, m2 = Mc_eta_to_ms(np.array([params["M_c"], params["eta"]]))
    return float(m1), float(m2)


def _make_aligned_reference(approximant_name: str) -> Callable:
    """Single-mode reference generator for an aligned-spin approximant.

    Deliberately not routed through ``LALBackend.generate``: that path has no
    mode selection, and adding one there would change the mode-summed test's
    call signature for a case only this module needs.
    """

    def generator(params: dict, mode: Mode, grid: Grid):
        import lal
        import lalsimulation as lalsim

        m1, m2 = _masses(params)
        laldict = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertModeArray(laldict, _mode_array(mode))
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            0.0,
            0.0,
            float(params["s1_z"]),
            0.0,
            0.0,
            float(params["s2_z"]),
            float(params["d_L"]) * 1e6 * lal.PC_SI,
            float(params["iota"]),
            float(params["phase_c"]),
            0,
            0,
            0,
            grid.df,
            grid.f_l,
            grid.f_u,
            grid.f_ref,
            laldict,
            lalsim.SimInspiralGetApproximantFromString(approximant_name),
        )
        return _trim(hp, hc, grid)

    return generator


def _reference_xphm_mode(params: dict, mode: Mode, grid: Grid):
    """Single co-precessing-mode (hp, hc) from LAL's IMRPhenomXPHM.

    Settings mirror ``LALBackend``: TwistPhenomHM=0 (XHM co-precessing seed, as
    in ripple), multibanding off, and PrecVersion=222 so an MSA-init failure
    raises instead of silently falling back to NNLO angles.
    """
    import lal
    import lalsimulation as lalsim

    m1, m2 = _masses(params)
    p = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertModeArray(p, _mode_array(mode))
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(p, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(p, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(p, 0.0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 222)
    hp, hc = lalsim.SimIMRPhenomXPHM(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        float(params["s1_x"]),
        float(params["s1_y"]),
        float(params["s1_z"]),
        float(params["s2_x"]),
        float(params["s2_y"]),
        float(params["s2_z"]),
        float(params["d_L"]) * 1e6 * lal.PC_SI,
        float(params["iota"]),
        float(params["phase_c"]),
        grid.f_l,
        grid.f_u,
        grid.df,
        grid.f_ref,
        p,
    )
    return _trim(hp, hc, grid)


# ============================================================================
# ripple single-mode generation
# ============================================================================

# ripple's IMRPhenomHM builds its per-mode ringdown tables assuming
# ModeArray[1] == (2, 2) (init_PhenomHM_Storage reads f_rd_array[1] for the
# 22-mode reference), so the full ModeArray is always evaluated and the mode of
# interest selected afterwards.  The other models accept a free mode list.
_HM_MODES: tuple[Mode, ...] = ((2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4))
_XHM_MODES: tuple[Mode, ...] = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
_XPHM_MODES: tuple[Mode, ...] = ((2, 1), (2, 2), (3, 2), (3, 3), (4, 4))


def _amp0(m1, m2, d_L):
    """LAL's XLALSimPhenomUtilsFDamp0 prefactor, as the ripple generators use it."""
    Mtot = m1 + m2
    return Mtot * MRSUN * Mtot * MTSUN / (d_L * MPC)


def _sminus2(ell: int) -> Callable:
    """ripple's spin-weighted spherical harmonic helper for this ``l``."""
    from ripplegw.utils.spherical_harmonics import (
        compute_sminus2_l2,
        compute_sminus2_l3,
        compute_sminus2_l4,
    )

    table = {2: compute_sminus2_l2, 3: compute_sminus2_l3, 4: compute_sminus2_l4}
    if ell not in table:
        raise ValueError(f"ripple has no spin-weighted harmonics for l={ell}")
    return table[ell]


def _ripple_hm_mode(grid: Grid, mode: Mode) -> Callable:
    """Jitted ``fn(params) -> (hp, hc)`` for one IMRPhenomHM mode.

    The harmonic projection reproduces ``gen_IMRPhenomHM`` restricted to a
    single mode.
    """
    from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomHM import (
        XLALSimIMRPhenomHMGethlmModes,
        get_phenomHMFD_mode_projection,
    )

    ell, emm = mode
    index = _HM_MODES.index(mode)
    mode_array = jnp.array(_HM_MODES, dtype=jnp.int32)
    minus1l = -1 if ell % 2 else 1
    axis = grid.axis

    @jax.jit
    def waveform(params):
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hlm = XLALSimIMRPhenomHMGethlmModes(
            axis,
            m1 * MSUN,
            m2 * MSUN,
            0.0,
            0.0,
            params["s1_z"],
            0.0,
            0.0,
            params["s2_z"],
            params["phase_c"],
            grid.df,
            grid.f_ref,
            {"ModeArray": mode_array},
        )
        projection = get_phenomHMFD_mode_projection(params["iota"], minus1l, ell, emm)
        projected = (
            projection[:, None] * hlm[index][None, :] * _amp0(m1, m2, params["d_L"])
        )
        return projected[0], projected[1]

    return waveform


def _ripple_xhm_mode(grid: Grid, mode: Mode) -> Callable:
    """Jitted ``fn(params) -> (hp, hc)`` for one IMRPhenomXHM mode.

    Mode content comes from ``XLALSimIMRPhenomXHMGethlmModes``; the harmonic
    projection follows LAL's restricted-``ModeArray`` convention rather than
    the symmetrized one used by ``gen_IMRPhenomXHM_hphc`` (module docstring).
    """
    from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import (
        XLALSimIMRPhenomXHMGethlmModes,
        build_pWF22,
    )

    ell, emm = mode
    minus1l = 1 if ell % 2 == 0 else -1
    sminus2 = _sminus2(ell)
    axis = grid.axis

    @jax.jit
    def waveform(params):
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        M_s = (m1 + m2) * MTSUN
        pWF22 = build_pWF22(m1, m2, params["s1_z"], params["s2_z"], grid.f_ref)
        hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
            axis * M_s, pWF22, phi0=params["phase_c"], ell_mm_pairs=[mode]
        )
        hlm = hlm_dict[mode] * _amp0(m1, m2, params["d_L"]) * minus1l

        # LAL's single-mode ModeArray convention: unlike the default
        # full-waveform call, which symmetrizes the (l, -m) partner as
        # gen_IMRPhenomXHM_hphc does, the restricted-ModeArray path keeps only
        # the (l, m) harmonic, giving
        #   h+ = 0.5 * (-i)^m * (-1)^l * F_{l,m}(iota) * h_lm,   hx = -i h+.
        # Using the symmetrized factors here instead compares two different
        # decompositions and flips the sign of hx for iota > pi/2.
        factorp = 0.5 * ((-1j) ** emm) * minus1l * sminus2(params["iota"], emm)
        hp = factorp * hlm
        return hp, -1j * hp

    return waveform


def _ripple_xphm_mode(grid: Grid, mode: Mode) -> Callable:
    """Jitted ``fn(params) -> (hp, hc)`` for one IMRPhenomXPHM mode.

    The selected mode is the co-precessing-frame mode; every other mode is
    zeroed before the twist-up, so the result is that mode's full contribution
    to the inertial-frame polarizations -- the same quantity LAL returns for a
    single-entry ``ModeArray``.
    """
    from ripplegw.waveforms.cbc.IMRPhenomX.initialize_MSA_system import (
        IMRPhenomX_Initialize_MSA_System,
    )
    from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXHM import (
        XLALSimIMRPhenomXHMGethlmModes,
        build_pWF22,
    )
    from ripplegw.waveforms.cbc.IMRPhenomX.IMRPhenomXPHM import twistup

    index = _XPHM_MODES.index(mode)
    axis = grid.axis

    @jax.jit
    def waveform(params):
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        Mf = axis * (m1 + m2) * MTSUN
        msa = IMRPhenomX_Initialize_MSA_System(
            mass_1=m1,
            mass_2=m2,
            chi1x=params["s1_x"],
            chi1y=params["s1_y"],
            chi1z=params["s1_z"],
            chi2x=params["s2_x"],
            chi2y=params["s2_y"],
            chi2z=params["s2_z"],
            reference_frequency=grid.f_ref,
        )
        pWF22 = build_pWF22(
            m1,
            m2,
            params["s1_z"],
            params["s2_z"],
            grid.f_ref,
            msa_SAv2=msa[15],
            msa_S1L_pav=msa[32],
            msa_S2L_pav=msa[33],
        )
        hlm_dict = XLALSimIMRPhenomXHMGethlmModes(
            Mf, pWF22, phi0=0.0, ell_mm_pairs=list(_XPHM_MODES)
        )
        # Keep only the mode under test; the twist-up mixes each co-precessing
        # mode into the inertial frame independently, so zeroing the others
        # isolates this mode's contribution exactly.
        hlms = jnp.stack(
            [
                (-1 if ell % 2 else 1) * hlm_dict[(ell, emm)]
                if i == index
                else jnp.zeros_like(hlm_dict[(ell, emm)])
                for i, (ell, emm) in enumerate(_XPHM_MODES)
            ]
        ) * _amp0(m1, m2, params["d_L"])
        return twistup(
            Mf,
            m1,
            m2,
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["phase_c"],
            params["iota"],
            grid.f_ref,
            hlms,
        )

    return waveform


# ============================================================================
# The registry -- add new HOM waveforms here
# ============================================================================

HOM_WAVEFORMS: dict[str, HOMWaveform] = {
    wf.name: wf
    for wf in [
        HOMWaveform(
            name="IMRPhenomHM",
            ripple_modes=_HM_MODES,
            reference_modes=((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)),
            reference_single_mode=_make_aligned_reference("IMRPhenomHM"),
            ripple_single_mode=_ripple_hm_mode,
        ),
        HOMWaveform(
            name="IMRPhenomXHM",
            ripple_modes=_XHM_MODES,
            reference_modes=((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)),
            reference_single_mode=_make_aligned_reference("IMRPhenomXHM"),
            ripple_single_mode=_ripple_xhm_mode,
        ),
        HOMWaveform(
            name="IMRPhenomXPHM",
            # ModeArray for XPHM selects co-precessing modes, which come from
            # XHM (TwistPhenomHM=0), hence the same set as IMRPhenomXHM.
            ripple_modes=_XPHM_MODES,
            reference_modes=((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)),
            reference_single_mode=_reference_xphm_mode,
            ripple_single_mode=_ripple_xphm_mode,
        ),
    ]
}


# ============================================================================
# Mode parsing and parametrization
# ============================================================================


def parse_modes(spec: str) -> tuple[Mode, ...]:
    """Parse a ``--hom-modes`` string into ``(l, m)`` pairs.

    Accepts whitespace- and/or comma-separated tokens in either the compact
    two-digit form (``22``, ``33``) or an explicit ``l:m`` form (``4:3``, and
    the only option for ``l >= 10``).

    Raises:
        ValueError: If a token cannot be parsed, or has m <= 0 or m > l.
    """
    modes: list[Mode] = []
    for token in spec.replace(",", " ").split():
        if ":" in token:
            ell_str, _, emm_str = token.partition(":")
        elif len(token) == 2 and token.isdigit():
            ell_str, emm_str = token[0], token[1]
        else:
            raise ValueError(
                f"Cannot parse mode {token!r}: expected 'lm' (e.g. 33) or 'l:m' "
                "(e.g. 4:3). Commas separate modes, so write '2:2' rather than "
                "'2,2' for the explicit form."
            )
        try:
            ell, emm = int(ell_str), int(emm_str)
        except ValueError:
            raise ValueError(
                f"Cannot parse mode {token!r}: l and m must be integers"
            ) from None
        if ell < 2 or emm <= 0 or emm > ell:
            raise ValueError(
                f"Invalid mode {token!r}: need l >= 2 and 0 < m <= l "
                "(the (l, -m) partner is included automatically)"
            )
        if (ell, emm) not in modes:
            modes.append((ell, emm))
    if not modes:
        raise ValueError("--hom-modes was given but parsed to an empty mode list")
    return tuple(modes)


def pytest_generate_tests(metafunc):
    """Parametrize over the (waveform, mode) grid, honouring ``--hom-modes``."""
    if {"waveform_name", "mode"} - set(metafunc.fixturenames):
        return
    spec = metafunc.config.getoption("--hom-modes")
    try:
        modes = parse_modes(spec) if spec else DEFAULT_MODES
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    metafunc.parametrize(
        "waveform_name,mode",
        [
            pytest.param(name, mode, id=f"{name}-{mode[0]}{mode[1]}")
            for name in HOM_WAVEFORMS
            for mode in modes
        ],
    )


# ============================================================================
# Residuals
# ============================================================================


def compute_mode_residuals(
    h_ripple: np.ndarray, h_reference: np.ndarray, amp_floor: float = 1e-8
) -> dict:
    """Frequency-resolved residual of ripple against the reference for one mode.

    The overlap collapses a whole waveform to one number; this keeps the
    frequency dependence, which distinguishes a constant phase offset from a
    drift that grows through the inspiral, or an amplitude error confined to
    the ringdown.

    Outside the generated band the reference is zero, where a ratio is 0/0 and
    a phase is undefined, so every ratio-derived quantity is masked to
    ``|h_ref| > amp_floor * max|h_ref|`` and is NaN elsewhere -- the arrays
    stay aligned with the frequency axis while plots and statistics skip the
    dead bins.

    Returns:
        Dict with ``band`` (usable-bin mask), both amplitudes, both unwrapped
        in-band phases, and three residuals: ``rel_amp``
        ``= (|h_r| - |h_l|) / |h_l|``, ``dphase`` ``= unwrapped
        arg(h_r) - arg(h_l)`` in radians, and ``frac`` ``= |h_r - h_l| / |h_l|``
        (combined amplitude and phase error), plus ``max_frac`` and
        ``max_abs_dphase`` over the band.
    """
    h_r = np.asarray(h_ripple)
    h_l = np.asarray(h_reference)
    amp_r, amp_l = np.abs(h_r), np.abs(h_l)
    peak = amp_l.max() if amp_l.size else 0.0
    band = amp_l > amp_floor * peak if peak > 0 else np.zeros(amp_l.shape, dtype=bool)

    def blank():
        return np.full(amp_l.shape, np.nan)

    rel_amp, frac, dphase = blank(), blank(), blank()
    phase_r, phase_l = blank(), blank()
    if band.any():
        rel_amp[band] = (amp_r[band] - amp_l[band]) / amp_l[band]
        frac[band] = np.abs(h_r[band] - h_l[band]) / amp_l[band]
        # Unwrap only across contiguous in-band bins; unwrapping through the
        # masked-out region would inject spurious 2*pi jumps.
        phase_r[band] = np.unwrap(np.angle(h_r[band]))
        phase_l[band] = np.unwrap(np.angle(h_l[band]))
        dphase[band] = np.unwrap(np.angle(h_r[band]) - np.angle(h_l[band]))

    return {
        "band": band,
        "amp_ripple": amp_r,
        "amp_reference": amp_l,
        "phase_ripple": phase_r,
        "phase_reference": phase_l,
        "rel_amp": rel_amp,
        "dphase": dphase,
        "frac": frac,
        "max_frac": float(np.nanmax(frac)) if band.any() else np.nan,
        "max_abs_dphase": float(np.nanmax(np.abs(dphase))) if band.any() else np.nan,
    }


def _select_best_worst(losses: np.ndarray, valid: np.ndarray):
    """Best and worst sample indices by overlap loss, or None when none usable."""
    usable = valid & np.isfinite(losses)
    if not usable.any():
        return None
    indices = np.where(usable)[0]
    order = np.argsort(losses[indices])
    return int(indices[order[0]]), int(indices[order[-1]])


def _plot_residuals(
    result: _TestResult,
    mode: Mode,
    grid: Grid,
    ripple_waveform: Callable,
    params_batch: dict,
    reference_p: dict,
    nyquist: jnp.ndarray,
    best_worst: tuple[int, int],
    path: Path,
) -> None:
    """h+ frequency evolution and ripple-minus-reference residuals, two samples.

    One column per sample (best and worst mismatch), four stacked panels each
    sharing a frequency axis: amplitude, amplitude residual, unwrapped phase,
    phase residual.  The axis is the dimensionless ``Mf = M_total * MTSUN * f``,
    so merger/ringdown lands at comparable positions in both columns even
    though the samples have different total masses; Mf is per-sample, hence the
    per-column x axis.

    Only h+ is shown: for XHM and XPHM ``hx = -i h+`` on both sides, and for HM
    the two polarizations differ by a fixed real factor at a given inclination,
    so the residuals carry no extra information.
    """
    import matplotlib.pyplot as plt

    axis = np.asarray(grid.axis)
    mask = np.asarray(nyquist)
    worst = result.worst
    columns = [("best", best_worst[0]), ("worst", best_worst[1])]
    if best_worst[0] == best_worst[1]:
        columns = [("only sample", best_worst[0])]

    fig, axes = plt.subplots(
        4,
        len(columns),
        figsize=(7.0 * len(columns), 11),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 2, 3, 2]},
        sharex="col",
    )
    fig.suptitle(
        f"{result.waveform} ({mode[0]},{mode[1]}) mode: $h_+$ frequency "
        f"evolution and ripple $-$ {result.reference} residual",
        fontsize=13,
    )

    for col, (label, idx) in enumerate(columns):
        sample = {k: np.asarray(v)[idx] for k, v in params_batch.items()}
        hp_l = np.asarray(reference_p[idx]) * mask
        hp_r = np.asarray(ripple_waveform(sample)[0]) * mask
        res = compute_mode_residuals(hp_r, hp_l)
        band = res["band"]

        m1, m2 = _masses(sample)
        Mf = axis * (m1 + m2) * MTSUN
        header = (
            f"{label}: sample {idx}, $1-\\mathcal{{O}}$ = {worst[idx]:.2e}\n"
            f"$m_1$={m1:.1f}, $m_2$={m2:.1f} $M_\\odot$, "
            f"$\\iota$={float(sample['iota']):.2f} rad"
        )

        ax = axes[0, col]
        ax.loglog(Mf[band], res["amp_reference"][band], color="k", lw=1.6, label="LAL")
        ax.loglog(
            Mf[band],
            res["amp_ripple"][band],
            color="tab:red",
            lw=1.0,
            ls="--",
            label="ripple",
        )
        ax.set_ylabel(r"$|h_+(Mf)|$")
        ax.set_title(header, fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1, col]
        ax.semilogx(Mf[band], res["rel_amp"][band], color="tab:blue", lw=1.0)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel(r"$\frac{|h_r| - |h_l|}{|h_l|}$")
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.semilogx(
            Mf[band], res["phase_reference"][band], color="k", lw=1.6, label="LAL"
        )
        ax.semilogx(
            Mf[band],
            res["phase_ripple"][band],
            color="tab:red",
            lw=1.0,
            ls="--",
            label="ripple",
        )
        ax.set_ylabel(r"$\arg h_+(Mf)$ [rad]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3, col]
        ax.semilogx(Mf[band], res["dphase"][band], color="tab:blue", lw=1.0)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel(r"$\Delta\arg h_+$ [rad]")
        ax.set_xlabel(r"$Mf$")
        ax.grid(alpha=0.3)
        ax.text(
            0.02,
            0.95,
            f"max $|h_r-h_l|/|h_l|$ = {res['max_frac']:.2e}\n"
            f"max $|\\Delta\\arg|$ = {res['max_abs_dphase']:.2e} rad",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightyellow",
                "alpha": 0.8,
            },
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ============================================================================
# The test
# ============================================================================


def _fmt_modes(modes: Iterable[Mode]) -> str:
    return ", ".join(f"({ell},{emm})" for ell, emm in sorted(modes))


@pytest.mark.accuracy
def test_reference_mode_overlap(
    waveform_name,
    mode,
    reference,
    n_samples,
    segment_duration_override,
    accuracy_outdir,
    make_plots,
    cross_val_results,
):
    """Compare one (l, m) mode against the reference backend.

    Modes either codebase does not implement are skipped with a reason naming
    the missing side.
    """
    if reference.name != "lal":
        pytest.skip(
            f"per-mode comparison is only implemented for lal, not {reference.name!r}"
        )
    if not reference.supports(waveform_name):
        pytest.skip(f"{reference.name} backend does not support {waveform_name!r}")

    spec = HOM_WAVEFORMS[waveform_name]
    missing = spec.missing_from(mode)
    if missing:
        pytest.skip(
            f"Mode ({mode[0]},{mode[1]}) is not implemented in "
            f"{' and '.join(missing)} for {waveform_name}. "
            f"ripple implements {_fmt_modes(spec.ripple_modes)}; "
            f"{reference.name} implements {_fmt_modes(spec.reference_modes)}."
        )

    import ripplegw

    wf = ripplegw.waveform(waveform_name)
    grid = default_grid(wf, T_override=segment_duration_override)
    threshold = spec.threshold(mode)

    psd_freqs, psd_values = load_psd()
    psd = jnp.interp(grid.axis, jnp.asarray(psd_freqs), jnp.asarray(psd_values))
    nyquist = get_nyquist_mask(grid.axis)

    params_batch = random_params_batch(wf, n_samples, seed=42)

    # ---- reference side ---------------------------------------------------
    reference_p: dict[int, np.ndarray] = {}
    reference_c: dict[int, np.ndarray] = {}
    errors: dict = {}
    valid = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        sample = {k: np.asarray(v)[i] for k, v in params_batch.items()}
        try:
            hp, hc = spec.reference_single_mode(sample, mode, grid)
        except Exception as exc:  # noqa: BLE001 - recorded per sample
            errors[str(i)] = str(exc)
            continue
        if len(hp) != len(grid.axis):
            errors[str(i)] = (
                f"reference returned {len(hp)} bins, expected {len(grid.axis)}"
            )
            continue
        reference_p[i] = np.asarray(hp)
        reference_c[i] = np.asarray(hc)
        valid[i] = True

    if not reference_p:
        pytest.skip(
            f"No valid {reference.name} samples for {waveform_name} "
            f"({mode[0]},{mode[1]}); errors: {list(errors.items())[:3]}"
        )

    # ---- ripple side + overlap -------------------------------------------
    ripple_waveform = spec.ripple_single_mode(grid, mode)
    loss_p = np.full(n_samples, np.nan)
    loss_c = np.full(n_samples, np.nan)
    for i in sorted(reference_p):
        sample = {k: jnp.asarray(np.asarray(v)[i]) for k, v in params_batch.items()}
        hp_r, hc_r = ripple_waveform(sample)
        hp_l = jnp.asarray(reference_p[i]) * nyquist
        hc_l = jnp.asarray(reference_c[i]) * nyquist
        loss_p[i] = float(overlap_loss(hp_r * nyquist, hp_l, psd, grid.axis))
        loss_c[i] = float(overlap_loss(hc_r * nyquist, hc_l, psd, grid.axis))

    result = _TestResult(
        waveform=f"{waveform_name}_{mode[0]}{mode[1]}",
        reference=reference.name,
        n_samples=n_samples,
        grid=grid,
        overlap_loss_p=loss_p,
        overlap_loss_c=loss_c,
        valid_mask=valid,
        errors=errors,
        threshold=threshold,
    )

    outdir = Path(accuracy_outdir)
    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        print(f"  Figure saved to: {plot_results(result, outdir)}")
        best_worst = _select_best_worst(result.worst, valid)
        if best_worst is not None:
            residual_file = (
                results_file.parent / f"{reference.name}_{result.waveform}_residual.png"
            )
            _plot_residuals(
                result,
                mode,
                grid,
                ripple_waveform,
                params_batch,
                reference_p,
                nyquist,
                best_worst,
                residual_file,
            )
            print(f"  Residual figure saved to: {residual_file}")

    finite = result.testable
    cross_val_results.append(
        {
            "waveform": f"{waveform_name} ({mode[0]},{mode[1]})",
            "reference": reference.name,
            "n_samples": n_samples,
            "n_failed": len(result.errors),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "max": result.max_loss,
            "threshold": threshold,
            "passed": result.passed,
        }
    )

    if result.errors:
        pytest.fail(
            f"{len(result.errors)}/{n_samples} samples failed to generate for "
            f"{waveform_name} ({mode[0]},{mode[1]}) against {reference.name}: "
            f"{list(result.errors.items())[:5]}"
        )
    assert finite.size > 0, f"No testable samples for {waveform_name} {mode}"
    assert result.max_loss < threshold, (
        f"{waveform_name} ({mode[0]},{mode[1]}): max overlap loss "
        f"{result.max_loss:.2e} exceeds threshold {threshold:.2e} for "
        f"reference {reference.name!r}"
    )
