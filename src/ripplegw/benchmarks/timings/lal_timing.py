"""
Command-line interface for timing LALSuite gravitational waveform generation on CPU.

Intentionally has no dependency on ripplegw or JAX so it can run in a minimal
CPU environment (lalsuite + numpy only). JSON output format matches timing.py
so that compare_lal.py can compare the two backends directly.
"""

import argparse
import json
import logging
import socket
import subprocess
import time
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Same location timing.py writes to and compare_lal.py reads from.
_DEFAULT_OUTDIR = (
    Path(__file__).parent.parent.parent.parent.parent / "timings" / "outdir"
)

lal: Any = None
lalsim: Any = None
try:
    lal = import_module("lal")
    lalsim = import_module("lalsimulation")
    HAS_LAL = True
except ImportError:
    HAS_LAL = False


def _require_lal() -> tuple[Any, Any]:
    """Return LALSuite modules, or raise a clear error when they are unavailable."""
    if lal is None or lalsim is None:
        raise ImportError(
            "LALSuite is not available. Install lalsuite to run the LAL benchmark."
        )
    return lal, lalsim


# ── parameter generation (numpy, no JAX) ────────────────────────────────────


def _generate_bbh_parameters(n, seed=42):
    rng = np.random.default_rng(seed)
    mass_1 = rng.uniform(10, 100, n)
    mass_2 = rng.uniform(0.5, 1.0, n) * mass_1
    a_1 = rng.uniform(0, 0.99, n)
    a_2 = rng.uniform(0, 0.99, n)
    v1 = rng.uniform(-1, 1, (n, 3))
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = rng.uniform(-1, 1, (n, 3))
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "spin_1x": a_1 * v1[:, 0],
        "spin_1y": a_1 * v1[:, 1],
        "spin_1z": a_1 * v1[:, 2],
        "spin_2x": a_2 * v2[:, 0],
        "spin_2y": a_2 * v2[:, 1],
        "spin_2z": a_2 * v2[:, 2],
        "luminosity_distance": rng.uniform(100, 2000, n),
        "theta_jn": rng.uniform(0, np.pi, n),
        "phase": rng.uniform(0, 2 * np.pi, n),
        "geocent_time": rng.uniform(0, 1, n),
    }


def _generate_bns_parameters(n, seed=42):
    rng = np.random.default_rng(seed)
    mass_1 = rng.uniform(1.2, 3.0, n)
    mass_2 = rng.uniform(0.5, 1.0, n) * mass_1
    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": rng.uniform(-0.4, 0.4, n),
        "a_2": rng.uniform(-0.4, 0.4, n),
        "lambda_1": rng.uniform(0, 5000, n),
        "lambda_2": rng.uniform(0, 5000, n),
        "luminosity_distance": rng.uniform(100, 2000, n),
        "theta_jn": rng.uniform(0, np.pi, n),
        "phase": rng.uniform(0, 2 * np.pi, n),
        "geocent_time": rng.uniform(0, 1, n),
    }


def _generate_precessing_bns_parameters(n, seed=42):
    rng = np.random.default_rng(seed)
    mass_1 = rng.uniform(1.2, 3.0, n)
    mass_2 = rng.uniform(0.5, 1.0, n) * mass_1
    a_1 = rng.uniform(0, 0.4, n)
    a_2 = rng.uniform(0, 0.4, n)
    v1 = rng.uniform(-1, 1, (n, 3))
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = rng.uniform(-1, 1, (n, 3))
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "spin_1x": a_1 * v1[:, 0],
        "spin_1y": a_1 * v1[:, 1],
        "spin_1z": a_1 * v1[:, 2],
        "spin_2x": a_2 * v2[:, 0],
        "spin_2y": a_2 * v2[:, 1],
        "spin_2z": a_2 * v2[:, 2],
        "lambda_1": rng.uniform(0, 5000, n),
        "lambda_2": rng.uniform(0, 5000, n),
        "luminosity_distance": rng.uniform(100, 2000, n),
        "theta_jn": rng.uniform(0, np.pi, n),
        "phase": rng.uniform(0, 2 * np.pi, n),
        "geocent_time": rng.uniform(0, 1, n),
    }


def _get_waveform_type(waveform):
    if waveform in ("IMRPhenomXP_NRTidalv3",):
        return "precessing_bns"
    if waveform in ("TaylorF2", "IMRPhenomD_NRTidalv2", "IMRPhenomXAS_NRTidalv3"):
        return "bns"
    return "bbh"


def _get_git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── theta array builders ─────────────────────────────────────────────────────


def _theta_aligned(p, i):
    return [
        p["mass_1"][i],
        p["mass_2"][i],
        p["a_1"][i],
        p["a_2"][i],
        p["luminosity_distance"][i],
        p["geocent_time"][i],
        p["phase"][i],
        p["theta_jn"][i],
    ]


def _theta_tidal(p, i):
    return [
        p["mass_1"][i],
        p["mass_2"][i],
        p["a_1"][i],
        p["a_2"][i],
        p["lambda_1"][i],
        p["lambda_2"][i],
        p["luminosity_distance"][i],
        p["geocent_time"][i],
        p["phase"][i],
        p["theta_jn"][i],
    ]


def _theta_precessing(p, i):
    return [
        p["mass_1"][i],
        p["mass_2"][i],
        p["spin_1x"][i],
        p["spin_1y"][i],
        p["spin_1z"][i],
        p["spin_2x"][i],
        p["spin_2y"][i],
        p["spin_2z"][i],
        p["luminosity_distance"][i],
        p["geocent_time"][i],
        p["phase"][i],
        p["theta_jn"][i],
    ]


def _theta_precessing_tidal(p, i):
    return [
        p["mass_1"][i],
        p["mass_2"][i],
        p["spin_1x"][i],
        p["spin_1y"][i],
        p["spin_1z"][i],
        p["spin_2x"][i],
        p["spin_2y"][i],
        p["spin_2z"][i],
        p["lambda_1"][i],
        p["lambda_2"][i],
        p["luminosity_distance"][i],
        p["geocent_time"][i],
        p["phase"][i],
        p["theta_jn"][i],
    ]


# ── LAL call ─────────────────────────────────────────────────────────────────


def _call_lal_single(theta, waveform_name, f_l, f_u, f_ref, df):
    """Generate a single waveform with LALSuite and return (hp, hc) arrays."""
    lal, lalsim = _require_lal()
    approximant = lalsim.SimInspiralGetApproximantFromString(waveform_name)

    if waveform_name == "IMRPhenomXPHM":
        m1_kg, m2_kg = theta[0] * lal.MSUN_SI, theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        distance, phi_ref, inclination = (
            theta[8] * 1e6 * lal.PC_SI,
            theta[10],
            theta[11],
        )
        p = lal.CreateDict()
        MA = lalsim.SimInspiralCreateModeArray()
        for el, em in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
            lalsim.SimInspiralModeArrayActivateMode(MA, el, em)
        lalsim.SimInspiralWaveformParamsInsertModeArray(p, MA)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(p, 0)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(p, 0)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(p, 0.0)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 222)
        hp, hc = lalsim.SimIMRPhenomXPHM(
            m1_kg,
            m2_kg,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            distance,
            inclination,
            phi_ref,
            f_l,
            f_u,
            df,
            f_ref,
            p,
        )

    elif waveform_name in ("IMRPhenomXP", "IMRPhenomXP_NRTidalv3"):
        m1_kg, m2_kg = theta[0] * lal.MSUN_SI, theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        p = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 222)
        if waveform_name == "IMRPhenomXP_NRTidalv3":
            l1, l2 = theta[8], theta[9]
            distance, phi_ref, inclination = (
                theta[10] * 1e6 * lal.PC_SI,
                theta[12],
                theta[13],
            )
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(p, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(p, l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(
                p, lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1) - 1
            )
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(
                p, lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2) - 1
            )
        else:
            distance, phi_ref, inclination = (
                theta[8] * 1e6 * lal.PC_SI,
                theta[10],
                theta[11],
            )
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
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
            p,
            approximant,
        )

    elif waveform_name == "IMRPhenomPv2":
        m1_kg, m2_kg = theta[0] * lal.MSUN_SI, theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        distance, phi_ref, inclination = (
            theta[8] * 1e6 * lal.PC_SI,
            theta[10],
            theta[11],
        )
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
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

    else:
        is_tidal = waveform_name in (
            "TaylorF2",
            "IMRPhenomD_NRTidalv2",
            "IMRPhenomXAS_NRTidalv3",
        )
        m1_kg, m2_kg = theta[0] * lal.MSUN_SI, theta[1] * lal.MSUN_SI
        s1z, s2z = theta[2], theta[3]
        if is_tidal:
            l1, l2 = theta[4], theta[5]
            distance, phi_ref, inclination = (
                theta[6] * 1e6 * lal.PC_SI,
                theta[8],
                theta[9],
            )
            laldict = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(
                laldict, lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1) - 1
            )
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(
                laldict, lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2) - 1
            )
        else:
            distance, phi_ref, inclination = (
                theta[4] * 1e6 * lal.PC_SI,
                theta[6],
                theta[7],
            )
            laldict = None
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            s1z,
            0.0,
            0.0,
            s2z,
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

    freqs = np.arange(len(hp.data.data)) * df
    mask = (freqs > f_l) & (freqs < f_u)
    return hp.data.data[mask], hc.data.data[mask]


# ── timing ───────────────────────────────────────────────────────────────────


def _build_theta(waveform_name, waveform_type, params, i):
    if waveform_type == "precessing_bns":
        return _theta_precessing_tidal(params, i)
    if waveform_type == "bns":
        return _theta_tidal(params, i)
    if waveform_name in ("IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"):
        return _theta_precessing(params, i)
    return _theta_aligned(params, i)


def time_lal_waveform(waveform_name, params, waveform_type, config):
    f_l = config["minimum_frequency"]
    f_u = config["maximum_frequency"]
    f_ref = config["reference_frequency"]
    df = 1.0 / config["duration"]
    n_waveforms = config["n_waveforms"]
    n_runs = config["n_runs"]

    exec_times = []
    for run_idx in range(n_runs):
        start = time.time()
        for i in range(n_waveforms):
            theta = _build_theta(waveform_name, waveform_type, params, i)
            _call_lal_single(theta, waveform_name, f_l, f_u, f_ref, df)
        t = time.time() - start
        exec_times.append(t)
        logger.info(
            "  Run %d: %.6f s  (%.3f ms/waveform)",
            run_idx + 1,
            t,
            t / n_waveforms * 1000,
        )
    return exec_times


def run_timing(args):
    _require_lal()

    config = {
        "waveform": args.waveform,
        "device": "cpu",
        "n_waveforms": args.n_waveforms,
        "n_runs": args.n_runs,
        "precision": "float64",
        "duration": args.duration,
        "minimum_frequency": args.f_min,
        "maximum_frequency": args.f_max,
        "reference_frequency": args.f_ref,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_hash": _get_git_hash(),
    }

    logger.info("=" * 60)
    logger.info(
        "LAL Timing  |  %s  |  N=%d  |  %d runs",
        args.waveform,
        args.n_waveforms,
        args.n_runs,
    )
    logger.info("=" * 60)

    waveform_type = _get_waveform_type(args.waveform)
    if waveform_type == "precessing_bns":
        params = _generate_precessing_bns_parameters(args.n_waveforms)
    elif waveform_type == "bns":
        params = _generate_bns_parameters(args.n_waveforms)
    else:
        params = _generate_bbh_parameters(args.n_waveforms)

    exec_times = time_lal_waveform(args.waveform, params, waveform_type, config)

    arr = np.array(exec_times)
    mean_exec = float(np.mean(arr))
    std_exec = float(np.std(arr, ddof=1)) if len(exec_times) > 1 else 0.0
    mean_tpw_ms = mean_exec / args.n_waveforms * 1000
    std_tpw_ms = std_exec / args.n_waveforms * 1000
    mean_wps = args.n_waveforms / mean_exec
    std_wps = args.n_waveforms * std_exec / (mean_exec**2)

    logger.info("Mean time/waveform: %.3f ms (+/- %.3f ms)", mean_tpw_ms, std_tpw_ms)
    logger.info("Throughput: %.1f waveforms/s (+/- %.1f)", mean_wps, std_wps)

    results = {
        **config,
        "backend": "lal",
        "device_name": "cpu",
        "hostname": socket.gethostname(),
        "timed_run_times_s": [float(t) for t in exec_times],
        "mean_execution_time_s": float(mean_exec),
        "std_execution_time_s": float(std_exec),
        "min_execution_time_s": float(np.min(arr)),
        "max_execution_time_s": float(np.max(arr)),
        "time_per_waveform_ms": float(mean_tpw_ms),
        "time_per_waveform_std_ms": float(std_tpw_ms),
        "waveforms_per_second": float(mean_wps),
        "waveforms_per_second_std": float(std_wps),
    }

    output_path = (
        Path(args.output)
        if args.output
        else _DEFAULT_OUTDIR / f"{args.waveform}_lal_cpu.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", output_path)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Time LALSuite waveform generation on CPU (no JAX required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "waveform",
        choices=[
            "TaylorF2",
            "IMRPhenomD",
            "IMRPhenomD_NRTidalv2",
            "IMRPhenomHM",
            "IMRPhenomPv2",
            "IMRPhenomXAS",
            "IMRPhenomXAS_NRTidalv3",
            "IMRPhenomXHM",
            "IMRPhenomXP",
            "IMRPhenomXPHM",
            "IMRPhenomXP_NRTidalv3",
        ],
    )
    parser.add_argument("--n-waveforms", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--f-min", type=float, default=5.0)
    parser.add_argument("--f-max", type=float, default=2048.0)
    parser.add_argument("--f-ref", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path (default: timings/outdir/<waveform>_lal_cpu.json)",
    )
    args = parser.parse_args()
    run_timing(args)


if __name__ == "__main__":
    main()
