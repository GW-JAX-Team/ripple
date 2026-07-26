"""LALSuite reference backend.

Wraps ``lalsimulation`` behind the ``ReferenceBackend`` protocol. Converts
ripple-native parameter dicts (``M_c``, ``eta``, ``s1_z``, ...) to LAL's own
convention internally, so callers never need to know LAL's argument order.

Three families need calls beyond the generic ``SimInspiralChooseFDWaveform``
path:

- **IMRPhenomXP** sets ``PhenomXPrecVersion=222``, which raises on MSA-init
  failure instead of silently falling back to NNLO (223) -- callers treat
  that exception as an MSA fallback, not a real failure.
- **IMRPhenomXPHM** uses ``SimIMRPhenomXPHM`` directly to configure the mode
  array, ``TwistPhenomHM=0`` (XHM co-precessing seed, matching ripple), and
  the same ``PrecVersion=222``.
- Tidal models insert ``TidalLambda{1,2}`` and the ``dQuadMon`` universal
  relation ripple assumes internally.
"""

import numpy as np

from ripplegw.conversions import Mc_eta_to_ms, lambda_tildes_to_lambdas
from tests.cross_validation.reference import register_backend
from tests.helpers.grids import Grid

try:
    import lal
    import lalsimulation as lalsim

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised via available()
    lal = None
    lalsim = None
    _IMPORT_ERROR = exc

# ripple waveform name -> LAL approximant name (identical for every built-in
# CBC family; kept as an explicit map so a future name mismatch is visible).
_APPROXIMANT_NAMES = {
    "TaylorF2": "TaylorF2",
    "IMRPhenomD": "IMRPhenomD",
    "IMRPhenomD_NRTidalv2": "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM": "IMRPhenomHM",
    "IMRPhenomPv2": "IMRPhenomPv2",
    "IMRPhenomXAS": "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3": "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM": "IMRPhenomXHM",
    "IMRPhenomXP": "IMRPhenomXP",
    "IMRPhenomXPHM": "IMRPhenomXPHM",
}

_TIDAL = {"TaylorF2", "IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3"}
_PRECESSING = {"IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"}


def _m1_m2(params: dict) -> tuple[float, float]:
    m1, m2 = Mc_eta_to_ms(np.array([params["M_c"], params["eta"]]))
    return float(m1), float(m2)


def _lambda_1_2(params: dict, m1: float, m2: float) -> tuple[float, float]:
    if "lambda_tilde" in params:
        l1, l2 = lambda_tildes_to_lambdas(
            np.array([params["lambda_tilde"], params["delta_lambda_tilde"], m1, m2])
        )
        return float(l1), float(l2)
    return float(params["lambda_1"]), float(params["lambda_2"])


def _mask_to_grid(hp, hc, df: float, grid: Grid) -> dict[str, np.ndarray]:
    freqs = np.arange(len(hp.data.data)) * df
    mask = (freqs > grid.f_l) & (freqs < grid.f_u)
    return {"p": hp.data.data[mask], "c": hc.data.data[mask]}


@register_backend
class LALBackend:
    name = "lal"

    @classmethod
    def available(cls) -> bool:
        return _IMPORT_ERROR is None

    def supports(self, waveform: str) -> bool:
        return waveform in _APPROXIMANT_NAMES

    def constants(self) -> dict[str, float]:
        return {
            "MSUN": lal.MSUN_SI,
            "MRSUN": lal.MRSUN_SI,
            "MTSUN": lal.MTSUN_SI,
            "G": lal.G_SI,
            "C": lal.C_SI,
            "PI": lal.PI,
            "TWO_PI": lal.TWOPI,
            "MPC": 1e6 * lal.PC_SI,
            "EULERGAMMA": lal.GAMMA,
        }

    def generate(
        self, waveform: str, params: dict[str, float], grid: Grid
    ) -> dict[str, np.ndarray]:
        if not self.supports(waveform):
            raise ValueError(f"LALBackend does not support {waveform!r}")

        m1, m2 = _m1_m2(params)
        m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        distance = params["d_L"] * 1e6 * lal.PC_SI
        phi_ref, inclination = params["phase_c"], params["iota"]
        approximant = lalsim.SimInspiralGetApproximantFromString(
            _APPROXIMANT_NAMES[waveform]
        )
        df, f_l, f_u, f_ref = grid.df, grid.f_l, grid.f_u, grid.f_ref

        if waveform == "IMRPhenomXP":
            lalparams = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 222)
            hp, hc = lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                lalparams,
                approximant,
            )
        elif waveform == "IMRPhenomXPHM":
            p = lal.CreateDict()
            mode_array = lalsim.SimInspiralCreateModeArray()
            for ell, m in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
                lalsim.SimInspiralModeArrayActivateMode(mode_array, ell, m)
            lalsim.SimInspiralWaveformParamsInsertModeArray(p, mode_array)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(p, 0)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(p, 0)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(p, 0.0)
            lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 222)
            hp, hc = lalsim.SimIMRPhenomXPHM(
                m1_kg,
                m2_kg,
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                distance,
                inclination,
                phi_ref,
                f_l,
                f_u,
                df,
                f_ref,
                p,
            )
        elif waveform in _PRECESSING:  # IMRPhenomPv2
            hp, hc = lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                None,
                approximant,
            )
        elif waveform in _TIDAL:
            l1, l2 = _lambda_1_2(params, m1, m2)
            laldict = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
            quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
            quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
            hp, hc = lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                0.0,
                0.0,
                params["s1_z"],
                0.0,
                0.0,
                params["s2_z"],
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                laldict,
                approximant,
            )
        else:
            hp, hc = lalsim.SimInspiralChooseFDWaveform(
                m1_kg,
                m2_kg,
                0.0,
                0.0,
                params["s1_z"],
                0.0,
                0.0,
                params["s2_z"],
                distance,
                inclination,
                phi_ref,
                0,
                0,
                0,
                df,
                f_l,
                f_u,
                f_ref,
                None,
                approximant,
            )

        return _mask_to_grid(hp, hc, df, grid)
