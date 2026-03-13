"""
Postprocessing script for visualizing waveform timing benchmarks.

This script reads timing JSON files and creates comparison bar charts
for different waveform approximants, comparing float32 vs float64 performance.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_timing_results(
    results_dir: Path, gpu_filter: Optional[str] = None
) -> List[Dict]:
    """Load all timing JSON files from the results directory.

    Args:
        results_dir: Directory containing timing JSON files
        gpu_filter: Optional GPU name to filter results (e.g., "H100", "cpu")

    Returns:
        List of timing result dictionaries
    """
    results = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return results

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

            # Apply GPU filter if specified
            if gpu_filter is not None and data.get("device_name") != gpu_filter:
                continue

            results.append(data)

    return results


def organize_results_by_waveform(results: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Organize results by waveform and precision.

    Args:
        results: List of timing result dictionaries

    Returns:
        Nested dict: {waveform: {precision: data}}
    """
    organized = {}

    for result in results:
        waveform = result["waveform"]
        precision = result["precision"]

        if waveform not in organized:
            organized[waveform] = {}

        organized[waveform][precision] = result

    return organized


def create_time_per_waveform_plot(
    organized_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    gpu_name: str,
    n_waveforms: int,
    batch_size: int,
):
    """Create bar chart comparing time per waveform for float32 vs float64."""
    waveforms = sorted(organized_results.keys())
    float32_times = []
    float64_times = []

    for waveform in waveforms:
        data = organized_results[waveform]
        float32_times.append(
            data.get("float32", {}).get("time_per_waveform_ms", np.nan)
        )
        float64_times.append(
            data.get("float64", {}).get("time_per_waveform_ms", np.nan)
        )

    x = np.arange(len(waveforms))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2, float32_times, width, label="float32", color="#3498db", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, float64_times, width, label="float64", color="#e74c3c", alpha=0.8
    )

    ax.set_ylabel("Time per Waveform (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Waveform Evaluation Time\nN = {n_waveforms}, b = {batch_size}, GPU = {gpu_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    autolabel(bars1)
    autolabel(bars2)

    y_max = ax.get_ylim()[1]
    ax.annotate(
        "Better",
        xy=(len(waveforms) - 0.5, y_max * 0.15),
        xytext=(len(waveforms) - 0.5, y_max * 0.3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10,
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved time per waveform plot to: {output_path}")
    plt.close()


def create_throughput_plot(
    organized_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    gpu_name: str,
    n_waveforms: int,
    batch_size: int,
):
    """Create bar chart comparing throughput (waveforms/s) for float32 vs float64."""
    waveforms = sorted(organized_results.keys())
    float32_throughput = []
    float64_throughput = []

    for waveform in waveforms:
        data = organized_results[waveform]
        float32_throughput.append(
            data.get("float32", {}).get("waveforms_per_second", np.nan)
        )
        float64_throughput.append(
            data.get("float64", {}).get("waveforms_per_second", np.nan)
        )

    x = np.arange(len(waveforms))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        float32_throughput,
        width,
        label="float32",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        float64_throughput,
        width,
        label="float64",
        color="#e74c3c",
        alpha=0.8,
    )

    ax.set_ylabel("Waveforms per Second", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Waveform Throughput\nN = {n_waveforms}, b = {batch_size}, GPU = {gpu_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(waveforms, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    autolabel(bars1)
    autolabel(bars2)

    y_max = ax.get_ylim()[1]
    ax.annotate(
        "Better",
        xy=(len(waveforms) - 0.5, y_max * 0.85),
        xytext=(len(waveforms) - 0.5, y_max * 0.7),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10,
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved throughput plot to: {output_path}")
    plt.close()


def main():
    """Main postprocessing function."""
    parser = argparse.ArgumentParser(
        description="Postprocess waveform timing results and create comparison plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="outdir",
        help="Directory containing timing JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output plots",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Filter results by GPU name (e.g., H100, A100, cpu).",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    if args.gpu:
        print(f"Filtering by GPU: {args.gpu}")
    results = load_timing_results(results_dir, gpu_filter=args.gpu)

    if not results:
        print("No timing results found!")
        return

    print(f"Loaded {len(results)} timing results")

    organized = organize_results_by_waveform(results)

    first_result = results[0]
    gpu_name = first_result.get("device_name", "Unknown")
    n_waveforms = first_result.get("n_waveforms", 0)
    batch_size = first_result.get("batch_size", 0)

    print("\nGenerating plots for:")
    print(f"  GPU: {gpu_name}")
    print(f"  N waveforms: {n_waveforms}")
    print(f"  Batch size: {batch_size}")
    print(f"  Waveforms analyzed: {', '.join(sorted(organized.keys()))}")

    create_time_per_waveform_plot(
        organized, output_dir / f"time_per_waveform_{gpu_name}.png", gpu_name, n_waveforms, batch_size
    )

    create_throughput_plot(
        organized, output_dir / f"throughput_{gpu_name}.png", gpu_name, n_waveforms, batch_size
    )

    print("\nPostprocessing complete!")


if __name__ == "__main__":
    main()
