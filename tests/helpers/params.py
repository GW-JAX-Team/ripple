"""Registry-driven parameter dicts for ripple waveform tests.

``canonical_params`` returns one physically sensible, deterministic value for
every name in ``wf.parameter_names``. The union of parameter names across all
built-in models is 20, so one small table below covers every registered
waveform. ``random_params_batch`` draws ``n`` such dicts (as batched arrays)
for the accuracy campaign in ``cross_validation/``.

Both raise ``KeyError`` immediately on an unrecognised parameter name, naming
this file as the place to extend -- that is the entire test-side cost of
adding a waveform with a genuinely new parameter.
"""

import jax.numpy as jnp
import numpy as np

from ripplegw.conversions import lambdas_to_lambda_tildes, ms_to_Mc_eta

# ---------------------------------------------------------------------------
# Regime-scoped defaults (canonical_params) and sampling bounds (random_params_batch)
# ---------------------------------------------------------------------------

_REGIME_MASSES = {"bbh": (30.0, 25.0), "bns": (1.4, 1.3)}

_REGIME_DEFAULTS = {
    "bbh": {
        "s1_z": 0.5,
        "s2_z": -0.3,
        "s1_x": 0.1,
        "s1_y": 0.2,
        "s2_x": -0.1,
        "s2_y": 0.15,
        "d_L": 400.0,
    },
    "bns": {
        "s1_z": 0.05,
        "s2_z": -0.02,
        "lambda_1": 500.0,
        "lambda_2": 400.0,
        "d_L": 100.0,
    },
}

_COMMON_DEFAULTS = {
    "phase_c": 0.5,
    "phase": 0.5,
    "iota": 0.8,
    "Q": 10.0,
    "f_0": 100.0,
    "hrss": 1e-21,
    "e": 0.3,
    # Continuous-wave (CW) models: not regime-scoped (bbh/bns doesn't apply),
    # so these live alongside the other common defaults rather than in
    # _REGIME_DEFAULTS. tp_ssb is an absolute GPS time and must stay close to
    # tests/helpers/config.py's default_config()'s start_gps (1_000_000_000).
    "alpha": 1.3,
    "delta": -0.5,
    "f0": 12.3,
    "phi0": 1.1,
    "aplus": 1.0,
    "across": 0.64,
    "f1": -1.1e-9,
    "f2": 2.0e-18,
    "asini": 1.44,
    "ecc": 0.18,
    "period": 6.3 * 3600.0,
    "argp": 1.05,
    "tp_ssb": 1_000_000_000.0 + 1234.0,
}

# (m_min, m_max), (chi_min, chi_max), (lambda_min, lambda_max), (d_L_min, d_L_max)
_REGIME_BOUNDS = {
    "bbh": {
        "m": (5.0, 100.0),
        "chi": (-0.99, 0.99),
        "lambda": (0.0, 0.0),
        "d_L": (100.0, 3000.0),
    },
    "bns": {
        "m": (0.5, 3.0),
        "chi": (-0.05, 0.05),
        "lambda": (0.0, 5000.0),
        "d_L": (30.0, 500.0),
    },
}


def regime(wf) -> str:
    """``"bns"`` for tidal models, ``"bbh"`` otherwise -- picks the mass/spin/distance scale."""
    return "bns" if wf.waveform_metadata.get("is_tidal") else "bbh"


def canonical_params(wf, **overrides) -> dict[str, float]:
    """One physically sensible value per name in ``wf.parameter_names``."""
    reg = regime(wf)
    m1, m2 = _REGIME_MASSES[reg]
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    values = {"M_c": float(Mc), "eta": float(eta)}
    values.update(_REGIME_DEFAULTS[reg])
    values.update(_COMMON_DEFAULTS)
    if "lambda_tilde" in wf.parameter_names:
        lt, dlt = lambdas_to_lambda_tildes(
            jnp.array([values["lambda_1"], values["lambda_2"], m1, m2])
        )
        values["lambda_tilde"] = float(lt)
        values["delta_lambda_tilde"] = float(dlt)
    values.update(overrides)
    try:
        return {k: values[k] for k in wf.parameter_names}
    except KeyError as exc:
        raise KeyError(
            f"No default registered for parameter {exc.args[0]!r}, needed by "
            f"{type(wf).__name__}.parameter_names. Add it to _REGIME_DEFAULTS "
            "or _COMMON_DEFAULTS in tests/helpers/params.py."
        ) from None


def random_params_batch(
    wf, n: int, *, seed: int | None = None
) -> dict[str, np.ndarray]:
    """``n`` random ripple-native parameter dicts, batched as arrays.

    Mass/spin/distance ranges are regime-scoped (BBH vs BNS) the same way as
    ``canonical_params``. ``M_c``/``eta`` come straight from independently
    sampled ``m1``, ``m2`` -- there is no mass-ordering convention to
    maintain, unlike a raw ``(m1, m2)`` LAL-style parameterisation.
    """
    reg = regime(wf)
    bounds = _REGIME_BOUNDS[reg]
    rng = np.random.default_rng(seed)

    m1 = rng.uniform(*bounds["m"], n)
    m2 = rng.uniform(*bounds["m"], n)
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    values: dict[str, np.ndarray] = {"M_c": np.asarray(Mc), "eta": np.asarray(eta)}

    names = set(wf.parameter_names)
    if {"s1_x", "s1_y", "s1_z"} <= names:
        # Precessing: sample spin vectors uniformly on a ball of the regime's
        # magnitude range (angles uniform on the sphere, magnitude uniform).
        for i in (1, 2):
            mag = rng.uniform(*bounds["chi"], n)
            costheta = rng.uniform(-1, 1, n)
            phi = rng.uniform(0, 2 * np.pi, n)
            sintheta = np.sqrt(1 - costheta**2)
            values[f"s{i}_x"] = mag * sintheta * np.cos(phi)
            values[f"s{i}_y"] = mag * sintheta * np.sin(phi)
            values[f"s{i}_z"] = mag * costheta
    else:
        values["s1_z"] = rng.uniform(*bounds["chi"], n)
        values["s2_z"] = rng.uniform(*bounds["chi"], n)

    if "lambda_1" in names or "lambda_tilde" in names:
        l1 = rng.uniform(*bounds["lambda"], n)
        l2 = rng.uniform(*bounds["lambda"], n)
        if "lambda_tilde" in names:
            lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
            values["lambda_tilde"] = np.asarray(lt)
            values["delta_lambda_tilde"] = np.asarray(dlt)
        else:
            values["lambda_1"] = l1
            values["lambda_2"] = l2

    if "d_L" in names:
        values["d_L"] = rng.uniform(*bounds["d_L"], n)
    if "phase_c" in names:
        values["phase_c"] = rng.uniform(0, 2 * np.pi, n)
    if "iota" in names:
        values["iota"] = rng.uniform(0, np.pi, n)

    try:
        return {k: values[k] for k in wf.parameter_names}
    except KeyError as exc:
        raise KeyError(
            f"random_params_batch does not know how to sample {exc.args[0]!r}, "
            f"needed by {type(wf).__name__}.parameter_names. Extend "
            "tests/helpers/params.py."
        ) from None
