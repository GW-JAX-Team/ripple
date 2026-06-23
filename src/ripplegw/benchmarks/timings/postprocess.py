"""
Postprocessing utilities for visualizing ripple waveform timing benchmarks.

Reads ripple timing JSON files from timings/outdir and creates bar charts
showing per-waveform evaluation time and throughput for each approximant.
For the ripple vs LAL cross-backend comparison, use compare_lal.py instead.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    plt: Any = None
    np: Any = None
    HAS_MATPLOTLIB = False

DEFAULT_RESULTS_DIR = Path.cwd() / "timings" / "outdir"
DEFAULT_OUTPUT_DIR = Path.cwd() / "timings" / "figures"


def load_timing_results(results_dir: Path) -> List[Dict]:
    """Load all ripple timing JSON files (float64) from the results directory."""
    results = []
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        if data.get("backend", "ripple") != "ripple":
            continue
        if data.get("precision") != "float64":
            continue
        results.append(data)

    if not results:
        logger.warning("No ripple float64 JSON files found in %s", results_dir)
    return results


def organize_results_by_waveform(results: List[Dict]) -> Dict[str, Dict]:
    """Organize results by waveform name."""
    organized: Dict[str, Dict] = {}
    for r in results:
        organized[r["waveform"]] = r
    return organized


def create_time_per_waveform_plot(
    organized: Dict[str, Dict],
    output_path: Path,
    device_name: str,
    n_waveforms: int,
):
    """Bar chart of time per waveform for all ripple models (float64)."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot.")
        return

    waveforms = sorted(organized)
    times = [organized[w].get("time_per_waveform_ms", float("nan")) for w in waveforms]
    stds = [organized[w].get("time_per_waveform_std_ms", 0.0) or 0.0 for w in waveforms]

    x = np.arange(len(waveforms))
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(
        x,
        times,
        yerr=stds,
        capsize=4,
        color="#3498db",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    ax.set_ylabel("Time per waveform (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Waveform evaluation time (ripple, float64)\nN = {n_waveforms}, device = {device_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, std in zip(bars, stds):
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + std,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def create_throughput_plot(
    organized: Dict[str, Dict],
    output_path: Path,
    device_name: str,
    n_waveforms: int,
):
    """Bar chart of throughput (waveforms/s) for all ripple models (float64)."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot.")
        return

    waveforms = sorted(organized)
    wps = [organized[w].get("waveforms_per_second", float("nan")) for w in waveforms]
    wps_stds = [
        organized[w].get("waveforms_per_second_std", 0.0) or 0.0 for w in waveforms
    ]

    x = np.arange(len(waveforms))
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(
        x,
        wps,
        yerr=wps_stds,
        capsize=4,
        color="#3498db",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    ax.set_ylabel("Waveforms per second", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Waveform throughput (ripple, float64)\nN = {n_waveforms}, device = {device_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, std in zip(bars, wps_stds):
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + std,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def run_postprocess(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return

    results = load_timing_results(results_dir)
    if not results:
        return

    organized = organize_results_by_waveform(results)
    first = results[0]
    device_name = first.get("device_name", "Unknown")
    n_waveforms = first.get("n_waveforms", 0)

    logger.info(
        "Device: %s  N=%d  Waveforms: %s",
        device_name,
        n_waveforms,
        ", ".join(sorted(organized)),
    )

    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots.")
        return

    create_time_per_waveform_plot(
        organized,
        output_dir / f"time_per_waveform_{device_name}.png",
        device_name,
        n_waveforms,
    )
    create_throughput_plot(
        organized,
        output_dir / f"throughput_{device_name}.png",
        device_name,
        n_waveforms,
    )
    logger.info("Postprocessing complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Postprocess ripple waveform timing results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_postprocess(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
