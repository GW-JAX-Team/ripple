"""
Postprocessing script comparing ripple (GPU, float64) vs LAL (CPU, float64).

Reads timing JSON files from timings/outdir and produces three plots:
  - time_per_waveform: grouped bars (ripple vs LAL), log y-axis
  - throughput: grouped bars (ripple vs LAL), log y-axis
  - speedup: per-model ripple speedup factor over LAL
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


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


def organize(results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Organize by waveform → {"ripple": data, "lal": data}.

    Only float64 ripple results are kept; non-float64 entries are skipped.
    """
    organized: Dict[str, Dict[str, Dict]] = {}
    for r in results:
        wf = r["waveform"]
        backend = r.get("backend", "ripple")
        if backend == "lal":
            label = "lal"
        elif backend == "ripple" and r.get("precision") == "float64":
            label = "ripple"
        else:
            continue
        organized.setdefault(wf, {})[label] = r
    return organized


def _label_bars(ax, bars, fmt=".3g", offset_factor=1.15):
    for bar in bars:
        h = bar.get_height()
        if not math.isnan(h) and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h * offset_factor,
                f"{h:{fmt}}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _bar_data(organized, waveforms, key, std_key):
    ripple = [organized[w].get("ripple", {}).get(key, float("nan")) for w in waveforms]
    lal = [organized[w].get("lal", {}).get(key, float("nan")) for w in waveforms]
    ripple_std = [
        organized[w].get("ripple", {}).get(std_key, 0.0) or 0.0 for w in waveforms
    ]
    lal_std = [organized[w].get("lal", {}).get(std_key, 0.0) or 0.0 for w in waveforms]
    return ripple, lal, ripple_std, lal_std


def create_time_per_waveform_plot(
    organized, output_path, ripple_device, ripple_n, lal_n
):
    waveforms = sorted(organized)
    ripple, lal, ripple_std, lal_std = _bar_data(
        organized,
        waveforms,
        "time_per_waveform_ms",
        "time_per_waveform_std_ms",
    )
    x = np.arange(len(waveforms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(
        x - width / 2,
        ripple,
        width,
        yerr=ripple_std,
        capsize=4,
        label=f"ripple float64 ({ripple_device} GPU, N={ripple_n})",
        color="#3498db",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    b2 = ax.bar(
        x + width / 2,
        lal,
        width,
        yerr=lal_std,
        capsize=4,
        label=f"LAL float64 (CPU, N={lal_n})",
        color="#2ecc71",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    ax.set_yscale("log")
    ax.set_ylabel("Time per waveform (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Waveform evaluation time: ripple (GPU) vs LAL (CPU)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    _label_bars(ax, b1)
    _label_bars(ax, b2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def create_throughput_plot(organized, output_path, ripple_device, ripple_n, lal_n):
    waveforms = sorted(organized)
    ripple, lal, ripple_std, lal_std = _bar_data(
        organized,
        waveforms,
        "waveforms_per_second",
        "waveforms_per_second_std",
    )
    x = np.arange(len(waveforms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(
        x - width / 2,
        ripple,
        width,
        yerr=ripple_std,
        capsize=4,
        label=f"ripple float64 ({ripple_device} GPU, N={ripple_n})",
        color="#3498db",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    b2 = ax.bar(
        x + width / 2,
        lal,
        width,
        yerr=lal_std,
        capsize=4,
        label=f"LAL float64 (CPU, N={lal_n})",
        color="#2ecc71",
        alpha=0.85,
        error_kw=dict(ecolor="black", lw=1.5),
    )
    ax.set_yscale("log")
    ax.set_ylabel("Waveforms per second", fontsize=12, fontweight="bold")
    ax.set_title(
        "Waveform throughput: ripple (GPU) vs LAL (CPU)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    _label_bars(ax, b1, fmt=".3g", offset_factor=1.5)
    _label_bars(ax, b2, fmt=".3g", offset_factor=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def create_speedup_plot(organized, output_path, ripple_device):
    waveforms = sorted(organized)
    speedups = []
    for wf in waveforms:
        r_tpw = organized[wf].get("ripple", {}).get("time_per_waveform_ms")
        l_tpw = organized[wf].get("lal", {}).get("time_per_waveform_ms")
        if r_tpw and l_tpw and r_tpw > 0:
            speedups.append(l_tpw / r_tpw)
        else:
            speedups.append(float("nan"))

    x = np.arange(len(waveforms))
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(x, speedups, color="#e74c3c", alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Speedup (LAL / ripple)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"ripple ({ripple_device} GPU, float64) speedup over LAL (CPU, float64)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    for bar, s in zip(bars, speedups):
        if not math.isnan(s) and s > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                s * 1.2,
                f"{s:.0f}×",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def run_compare(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return

    results = load_results(results_dir)
    if not results:
        logger.warning("No JSON files found in %s", results_dir)
        return

    organized = organize(results)
    if not organized:
        logger.warning("No float64 results found.")
        return

    ripple_device, ripple_n, lal_n = "GPU", 0, 0
    for wf_data in organized.values():
        if "ripple" in wf_data:
            ripple_device = wf_data["ripple"].get("device_name", "GPU")
            ripple_n = wf_data["ripple"].get("n_waveforms", 0)
        if "lal" in wf_data:
            lal_n = wf_data["lal"].get("n_waveforms", 0)

    logger.info("Waveforms: %s", ", ".join(sorted(organized)))
    logger.info("ripple device: %s  N=%d | LAL N=%d", ripple_device, ripple_n, lal_n)

    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots.")
        return

    create_time_per_waveform_plot(
        organized,
        output_dir / f"compare_time_{ripple_device}.png",
        ripple_device,
        ripple_n,
        lal_n,
    )
    create_throughput_plot(
        organized,
        output_dir / f"compare_throughput_{ripple_device}.png",
        ripple_device,
        ripple_n,
        lal_n,
    )
    create_speedup_plot(
        organized,
        output_dir / f"compare_speedup_{ripple_device}.png",
        ripple_device,
    )
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Compare ripple (GPU) vs LAL (CPU) waveform timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_compare(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
