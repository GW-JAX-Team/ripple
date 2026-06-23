"""
Command-line interface for timing LALSuite gravitational waveform generation on CPU.

Mirrors the structure of timing.py so that JSON outputs are directly comparable
by postprocess.py. Each timed run generates n_waveforms waveforms sequentially;
per-waveform statistics are reported in the same fields as the ripple benchmark.
"""

import argparse
import json
import logging
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ripplegw.benchmarks.utils import (
    generate_bbh_parameters,
    generate_bns_parameters,
    generate_precessing_bns_parameters,
    get_git_hash,
)
from ripplegw.benchmarks.timings.timing import get_waveform_type

logger = logging.getLogger(__name__)

try:
    import lal
    import lalsimulation as lalsim

    HAS_LAL = True
except ImportError:
    HAS_LAL = False


def _require_lal():
    if not HAS_LAL:
        raise ImportError(
            "LALSuite is not available. Install lalsuite to run the LAL benchmark."
        )


def _call_lal_single(theta, waveform_name, f_l, f_u, f_ref, df):
    """Call LAL to generate a single waveform.

    Dispatch logic mirrors tests/utils.py::get_lal_waveform so that the
    same physical conventions (PrecVersion=222, XPHM direct call, tidal
    dict insertion) are used for both cross-validation and benchmarking.

    Args:
        theta: Parameter array (see _build_theta_* helpers for layout).
        waveform_name: LALSuite approximant name.
        f_l: Lower frequency bound (Hz).
        f_u: Upper frequency bound (Hz).
        f_ref: Reference frequency (Hz).
        df: Frequency resolution (Hz).

    Returns:
        Tuple (hp, hc) as numpy arrays on the frequency grid [f_l, f_u).
    """
    approximant = lalsim.SimInspiralGetApproximantFromString(waveform_name)

    if waveform_name == "IMRPhenomXPHM":
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        distance = theta[8] * 1e6 * lal.PC_SI
        phi_ref = theta[10]
        inclination = theta[11]

        p = lal.CreateDict()
        ModeArray = lalsim.SimInspiralCreateModeArray()
        for el, em in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
            lalsim.SimInspiralModeArrayActivateMode(ModeArray, el, em)
        lalsim.SimInspiralWaveformParamsInsertModeArray(p, ModeArray)
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
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        s1x, s1y, s1z_val = theta[2], theta[3], theta[4]
        s2x, s2y, s2z_val = theta[5], theta[6], theta[7]
        p = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 222)

        if waveform_name == "IMRPhenomXP_NRTidalv3":
            l1, l2 = theta[8], theta[9]
            distance = theta[10] * 1e6 * lal.PC_SI
            phi_ref = theta[12]
            inclination = theta[13]
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(p, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(p, l2)
            quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
            quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(p, quad1 - 1)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(p, quad2 - 1)
        else:
            distance = theta[8] * 1e6 * lal.PC_SI
            phi_ref = theta[10]
            inclination = theta[11]

        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            s1x,
            s1y,
            s1z_val,
            s2x,
            s2y,
            s2z_val,
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

    elif waveform_name in ("IMRPhenomPv2",):
        # Generic precessing non-tidal
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        distance = theta[8] * 1e6 * lal.PC_SI
        phi_ref = theta[10]
        inclination = theta[11]
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
        # Non-precessing: theta = [m1, m2, s1z, s2z, (l1, l2,) dist, tc, phic, inc]
        is_tidal = waveform_name in (
            "TaylorF2",
            "IMRPhenomD_NRTidalv2",
            "IMRPhenomXAS_NRTidalv3",
        )
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        s1z = theta[2]
        s2z = theta[3]

        if is_tidal:
            l1, l2 = theta[4], theta[5]
            distance = theta[6] * 1e6 * lal.PC_SI
            phi_ref = theta[8]
            inclination = theta[9]
            laldict = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
            quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
            quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
        else:
            distance = theta[4] * 1e6 * lal.PC_SI
            phi_ref = theta[6]
            inclination = theta[7]
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

    freqs_lal = np.arange(len(hp.data.data)) * df
    mask = (freqs_lal > f_l) & (freqs_lal < f_u)
    return hp.data.data[mask], hc.data.data[mask]


def _build_theta_aligned(params, i):
    return np.array(
        [
            float(params["mass_1"][i]),
            float(params["mass_2"][i]),
            float(params["a_1"][i]),
            float(params["a_2"][i]),
            float(params["luminosity_distance"][i]),
            float(params["geocent_time"][i]),
            float(params["phase"][i]),
            float(params["theta_jn"][i]),
        ]
    )


def _build_theta_tidal(params, i):
    return np.array(
        [
            float(params["mass_1"][i]),
            float(params["mass_2"][i]),
            float(params["a_1"][i]),
            float(params["a_2"][i]),
            float(params["lambda_1"][i]),
            float(params["lambda_2"][i]),
            float(params["luminosity_distance"][i]),
            float(params["geocent_time"][i]),
            float(params["phase"][i]),
            float(params["theta_jn"][i]),
        ]
    )


def _build_theta_precessing(params, i):
    return np.array(
        [
            float(params["mass_1"][i]),
            float(params["mass_2"][i]),
            float(params["spin_1x"][i]),
            float(params["spin_1y"][i]),
            float(params["spin_1z"][i]),
            float(params["spin_2x"][i]),
            float(params["spin_2y"][i]),
            float(params["spin_2z"][i]),
            float(params["luminosity_distance"][i]),
            float(params["geocent_time"][i]),
            float(params["phase"][i]),
            float(params["theta_jn"][i]),
        ]
    )


def _build_theta_precessing_tidal(params, i):
    return np.array(
        [
            float(params["mass_1"][i]),
            float(params["mass_2"][i]),
            float(params["spin_1x"][i]),
            float(params["spin_1y"][i]),
            float(params["spin_1z"][i]),
            float(params["spin_2x"][i]),
            float(params["spin_2y"][i]),
            float(params["spin_2z"][i]),
            float(params["lambda_1"][i]),
            float(params["lambda_2"][i]),
            float(params["luminosity_distance"][i]),
            float(params["geocent_time"][i]),
            float(params["phase"][i]),
            float(params["theta_jn"][i]),
        ]
    )


def time_lal_waveform(waveform_name, params, waveform_type, config):
    """Time LAL waveform generation sequentially over n_waveforms parameter sets.

    Returns:
        tuple: (exec_times) list of per-run total times (each over n_waveforms calls).
    """
    f_l = config["minimum_frequency"]
    f_u = config["maximum_frequency"]
    f_ref = config["reference_frequency"]
    df = 1.0 / config["duration"]
    n_waveforms = config["n_waveforms"]
    n_runs = config["n_runs"]

    if waveform_type == "precessing_bns":
        build_theta = _build_theta_precessing_tidal
    elif waveform_type == "bns":
        build_theta = _build_theta_tidal
    elif waveform_type in ("precessing_bbh",):
        build_theta = _build_theta_precessing
    else:
        if waveform_name in ("IMRPhenomPv2", "IMRPhenomXP", "IMRPhenomXPHM"):
            build_theta = _build_theta_precessing
        else:
            build_theta = _build_theta_aligned

    exec_times = []
    for run_idx in range(n_runs):
        start = time.time()
        for i in range(n_waveforms):
            theta = build_theta(params, i)
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
    """Main timing function for the LAL benchmark."""
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
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
    }

    logger.info("=" * 60)
    logger.info("LAL Timing Configuration")
    logger.info("=" * 60)
    logger.info("Waveform: %s", args.waveform)
    logger.info("Number of waveforms: %d", args.n_waveforms)
    logger.info("Duration: %s s", args.duration)
    logger.info("Frequency range: %s - %s Hz", args.f_min, args.f_max)
    logger.info("Reference frequency: %s Hz", args.f_ref)
    logger.info("Git hash: %s", config["git_hash"])
    logger.info("=" * 60)

    waveform_type = get_waveform_type(args.waveform)

    if waveform_type == "precessing_bns":
        params = generate_precessing_bns_parameters(args.n_waveforms)
    elif waveform_type == "bns":
        params = generate_bns_parameters(args.n_waveforms)
    else:
        params = generate_bbh_parameters(args.n_waveforms)

    # Convert JAX arrays to numpy for indexing in the timing loop
    params = {k: np.array(v) for k, v in params.items()}

    logger.info("Generated %d parameter sets", args.n_waveforms)
    logger.info("\n%s", "=" * 60)
    logger.info(
        "Timed runs (%d repetitions, %d waveforms each)", args.n_runs, args.n_waveforms
    )
    logger.info("=" * 60)

    exec_times = time_lal_waveform(args.waveform, params, waveform_type, config)

    exec_times_arr = np.array(exec_times)
    mean_exec = float(np.mean(exec_times_arr))
    std_exec = float(np.std(exec_times_arr, ddof=1)) if len(exec_times) > 1 else 0.0
    min_exec = float(np.min(exec_times_arr))
    max_exec = float(np.max(exec_times_arr))
    mean_tpw_ms = mean_exec / args.n_waveforms * 1000
    std_tpw_ms = std_exec / args.n_waveforms * 1000
    mean_wps = args.n_waveforms / mean_exec
    std_wps = args.n_waveforms * std_exec / (mean_exec**2)

    logger.info("\n%s", "=" * 60)
    logger.info("Timing Results")
    logger.info("=" * 60)
    logger.info("Timed runs (%d repetitions):", args.n_runs)
    logger.info("  Mean execution time: %.6f s", mean_exec)
    logger.info("  Std  execution time: %.6f s", std_exec)
    logger.info("  Min  execution time: %.6f s", min_exec)
    logger.info("  Max  execution time: %.6f s", max_exec)
    logger.info(
        "Mean time per waveform: %.3f ms  (+/- %.3f ms)", mean_tpw_ms, std_tpw_ms
    )
    logger.info("Mean waveforms per second: %.1f  (+/- %.1f)", mean_wps, std_wps)
    logger.info("=" * 60)

    results = {
        **config,
        "backend": "lal",
        "device_name": "cpu",
        "hostname": socket.gethostname(),
        "timed_run_times_s": [float(t) for t in exec_times],
        "mean_execution_time_s": float(mean_exec),
        "std_execution_time_s": float(std_exec),
        "min_execution_time_s": float(min_exec),
        "max_execution_time_s": float(max_exec),
        "time_per_waveform_ms": float(mean_tpw_ms),
        "time_per_waveform_std_ms": float(std_tpw_ms),
        "waveforms_per_second": float(mean_wps),
        "waveforms_per_second_std": float(std_wps),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        outdir = (
            Path(__file__).parent.parent.parent.parent.parent / "timings" / "outdir"
        )
        outdir.mkdir(exist_ok=True)
        output_path = outdir / f"{args.waveform}_lal_cpu.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to: %s", output_path)


def main():
    """Parse arguments and run LAL timing benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Time LALSuite gravitational waveform generation on CPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "waveform",
        type=str,
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
        help="Waveform approximant to time",
    )

    parser.add_argument(
        "--n-waveforms",
        type=int,
        default=100,
        help="Number of waveforms to generate per timed run",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of timed runs",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Waveform duration in seconds (sets frequency resolution df=1/duration)",
    )

    parser.add_argument(
        "--f-min", type=float, default=5.0, help="Minimum frequency in Hz"
    )

    parser.add_argument(
        "--f-max", type=float, default=2048.0, help="Maximum frequency in Hz"
    )

    parser.add_argument(
        "--f-ref", type=float, default=50.0, help="Reference frequency in Hz"
    )

    parser.add_argument(
        "--output", type=str, help="Output JSON file path (default: auto-generated)"
    )

    args = parser.parse_args()
    run_timing(args)


if __name__ == "__main__":
    main()
