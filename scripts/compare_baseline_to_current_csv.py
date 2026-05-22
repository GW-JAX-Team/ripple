"""Compare saved baseline artifacts or CSV results to current mismatch CSV results.

This is useful when the current sweep comes from
`tests/cross_validation/test_lal_mismatch.py`, which writes CSV results under
`tests/cross_validation/results/<run_tag>/`.

Usage:
    uv run python scripts/compare_baseline_to_current_csv.py \
        --baseline baseline_results/pre_fix \
        --current-run-tag n200_T32

    uv run python scripts/compare_baseline_to_current_csv.py \
        --baseline-run-tag pre_fix_n200_T32 \
        --current-run-tag n200_T32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
WAVEFORMS = ("IMRPhenomXHM", "IMRPhenomXPHM")
METRICS = ("mismatch", "mismatch_hp", "mismatch_hc")


def _load_baseline_samples(run_dir: Path, waveform: str) -> dict[str, np.ndarray]:
    data = np.load(run_dir / f"{waveform}_samples.npz")
    return {metric: np.asarray(data[metric]) for metric in METRICS}


def _load_current_csv(run_tag: str, waveform: str) -> dict[str, np.ndarray]:
    csv_path = (
        REPO_ROOT / "tests" / "cross_validation" / "results" / run_tag / f"mismatch_{waveform}.csv"
    )
    df = pd.read_csv(csv_path)
    return {metric: df[metric].to_numpy() for metric in METRICS}


def _finite_sorted(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    return np.sort(np.abs(finite))[::-1]


def _plot_waveform(
    waveform: str,
    baseline: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    threshold: float,
    out_path: Path,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    for label, sample_set, color in (
        ("baseline", baseline, "tab:red"),
        ("current", current, "tab:blue"),
    ):
        vals = np.maximum(_finite_sorted(sample_set["mismatch"]), 1e-18)
        ax.step(np.arange(1, len(vals) + 1), vals, where="mid", label=label, color=color)
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
            ("baseline", baseline, "tab:red"),
            ("current", current, "tab:blue"),
        ):
            vals = np.maximum(_finite_sorted(sample_set[metric]), 1e-18)
            ax.step(
                np.arange(1, len(vals) + 1),
                vals,
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

    fig.suptitle(f"{waveform}: baseline vs current CSV results", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    def _stats(values: np.ndarray) -> dict:
        finite = np.asarray(values)[np.isfinite(values)]
        return {
            "n_finite": int(finite.size),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "n_failed": int(np.sum(finite > threshold)),
        }

    return {"baseline": _stats(baseline["mismatch"]), "current": _stats(current["mismatch"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("baseline_results/pre_fix"),
        help="Directory containing baseline *_samples.npz artifacts.",
    )
    parser.add_argument(
        "--baseline-run-tag",
        type=str,
        default=None,
        help="Optional run tag under tests/cross_validation/results/ to use as the baseline instead of --baseline.",
    )
    parser.add_argument(
        "--current-run-tag",
        type=str,
        required=True,
        help="Run tag under tests/cross_validation/results/, e.g. n200_T32.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Mismatch threshold for summary statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summary. Defaults to baseline_results/comparisons/pre_fix_vs_<run_tag>_csv.",
    )
    args = parser.parse_args()

    baseline_label = args.baseline_run_tag or "pre_fix"
    output_dir = args.output_dir or Path(
        f"baseline_results/comparisons/{baseline_label}_vs_{args.current_run_tag}_csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "baseline_dir": str(args.baseline) if args.baseline_run_tag is None else None,
        "baseline_run_tag": args.baseline_run_tag,
        "current_run_tag": args.current_run_tag,
        "threshold": args.threshold,
        "waveforms": {},
    }

    for waveform in WAVEFORMS:
        if args.baseline_run_tag is None:
            baseline = _load_baseline_samples(args.baseline, waveform)
        else:
            baseline = _load_current_csv(args.baseline_run_tag, waveform)
        current = _load_current_csv(args.current_run_tag, waveform)
        stats = _plot_waveform(
            waveform,
            baseline,
            current,
            args.threshold,
            output_dir / f"{waveform}_comparison.png",
        )
        summary["waveforms"][waveform] = stats

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {summary_path}")
    for waveform in WAVEFORMS:
        cur = summary["waveforms"][waveform]["current"]
        print(
            f"{waveform}: current n_failed={cur['n_failed']}, "
            f"max={cur['max']:.3e}, mean={cur['mean']:.3e}"
        )


if __name__ == "__main__":
    main()
