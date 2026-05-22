"""Compare baseline and current XHM/XPHM mismatch runs.

This script expects two run directories produced by
`scripts/baseline_xhm_xphm_mismatch.py` and generates:

1. Overlay plots of the sorted mismatch distributions.
2. Overlay plots of the hp/hc mismatch distributions.
3. A JSON summary of the before/after statistics and improvement factors.

Usage:
    uv run python scripts/compare_baseline_xhm_xphm_mismatch.py \
        --baseline baseline_results/pre_fix \
        --current baseline_results/post_fix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WAVEFORMS = ("IMRPhenomXHM", "IMRPhenomXPHM")
METRICS = ("mismatch", "mismatch_hp", "mismatch_hc")


def _load_run(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    summary_path = run_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    samples: dict[str, dict[str, np.ndarray]] = {}
    for waveform in WAVEFORMS:
        data = np.load(run_dir / f"{waveform}_samples.npz")
        samples[waveform] = {metric: np.asarray(data[metric]) for metric in METRICS}
        samples[waveform]["msa_fallback_mask"] = np.asarray(data["msa_fallback_mask"])
    return summary, samples


def _finite_sorted(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    return np.sort(finite)[::-1]


def _positive_floor(values: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    return np.maximum(np.abs(values), floor)


def _improvement_factor(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if after == 0.0:
        return None
    return float(before / after)


def _summary_entry(before: dict, after: dict) -> dict:
    return {
        "before": before,
        "after": after,
        "delta_failed": before["n_failed"] - after["n_failed"],
        "max_reduction_factor": _improvement_factor(before["max"], after["max"]),
        "mean_reduction_factor": _improvement_factor(before["mean"], after["mean"]),
        "median_reduction_factor": _improvement_factor(before["median"], after["median"]),
    }


def _plot_waveform(
    waveform: str,
    threshold: float,
    baseline_samples: dict[str, np.ndarray],
    current_samples: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    for label, sample_set, color in (
        ("baseline", baseline_samples, "tab:red"),
        ("current", current_samples, "tab:blue"),
    ):
        sorted_values = _positive_floor(_finite_sorted(sample_set["mismatch"]))
        x = np.arange(1, len(sorted_values) + 1)
        ax.step(x, sorted_values, where="mid", label=label, color=color)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Sorted sample rank")
    ax.set_ylabel("Mismatch")
    ax.set_title(f"{waveform}: overall mismatch")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1]
    for metric, linestyle in (("mismatch_hp", "-"), ("mismatch_hc", ":")):
        for label, sample_set, color in (
            ("baseline", baseline_samples, "tab:red"),
            ("current", current_samples, "tab:blue"),
        ):
            sorted_values = _positive_floor(_finite_sorted(sample_set[metric]))
            x = np.arange(1, len(sorted_values) + 1)
            ax.step(
                x,
                sorted_values,
                where="mid",
                linestyle=linestyle,
                color=color,
                label=f"{label} {metric.replace('mismatch_', '')}",
            )
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Sorted sample rank")
    ax.set_ylabel("Mismatch")
    ax.set_title(f"{waveform}: hp/hc mismatch")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncols=2, fontsize=8)

    fig.suptitle(f"{waveform} mismatch comparison", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("baseline_results/pre_fix"),
        help="Directory containing the baseline run artifacts.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("baseline_results/post_fix"),
        help="Directory containing the current run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_results/comparisons/pre_fix_vs_post_fix"),
        help="Directory to write plots and summary JSON.",
    )
    args = parser.parse_args()

    baseline_summary, baseline_samples = _load_run(args.baseline)
    current_summary, current_samples = _load_run(args.current)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, object] = {
        "baseline_dir": str(args.baseline),
        "current_dir": str(args.current),
        "waveforms": {},
    }

    for waveform in WAVEFORMS:
        baseline_stats = baseline_summary["waveforms"][waveform]
        current_stats = current_summary["waveforms"][waveform]
        threshold = float(current_stats["threshold"])

        _plot_waveform(
            waveform,
            threshold,
            baseline_samples[waveform],
            current_samples[waveform],
            args.output_dir / f"{waveform}_comparison.png",
        )

        comparison["waveforms"][waveform] = {
            metric: _summary_entry(baseline_stats[metric], current_stats[metric])
            for metric in METRICS
        }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote comparison summary: {summary_path}")
    for waveform in WAVEFORMS:
        stats = comparison["waveforms"][waveform]["mismatch"]
        print(
            f"{waveform}: failed {stats['before']['n_failed']} -> {stats['after']['n_failed']}, "
            f"max {stats['before']['max']:.3e} -> {stats['after']['max']:.3e}"
        )


if __name__ == "__main__":
    main()
