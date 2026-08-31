#!/usr/bin/env python3
"""Publication figure for the composed Transformer certificate runtime."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, finish_axes, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT = RESULTS / "transformer_v3_streaming_direct_analytic_audit.json"
CONTINUATION = RESULTS / "transformer_seed_366_matched_continuation.json"
REPLICATES = tuple(
    RESULTS
    / (
        "transformer_v3_streaming_direct_analytic_seed_366_gate_1_"
        f"anchor_1120_replicate-{index}.json"
    )
    for index in range(1, 4)
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    configure_paper_plots()
    audit = load(AUDIT)
    continuation = load(CONTINUATION)
    rows = [load(path) for path in REPLICATES]
    component_keys = (
        "streaming_centerline",
        "signed_correction_and_corrected_path",
        "analytic_neural_jets",
        "direct_image_green",
        "analytic_closure_and_event",
    )
    component_labels = ("centerline", "signed correction", "neural jets", "Green", "event")
    component_colors = (
        COLORS["blue"],
        COLORS["teal"],
        COLORS["ochre"],
        COLORS["vermilion"],
        COLORS["gray"],
    )
    medians = np.asarray(
        [
            statistics.median(float(row["timings_seconds"][key]) for row in rows)
            for key in component_keys
        ]
    )
    total = float(audit["median_end_to_end_seconds"])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.45, 1.0]})

    left = 0.0
    for value, label, color in zip(medians, component_labels, component_colors):
        axes[0].barh([0], [value], left=left, height=0.44, color=color, label=label)
        if value >= 0.22:
            axes[0].text(
                left + value / 2.0,
                0,
                f"{value:.2f}",
                color="white",
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight="bold",
            )
        left += value
    axes[0].set_xlim(0.0, max(total, float(medians.sum())) * 1.08)
    axes[0].set_ylim(-0.32, 0.42)
    axes[0].set_yticks([])
    axes[0].set_xlabel("median wall time (seconds)")
    axes[0].set_title("(a) Complete proof pipeline, $H=26$")
    axes[0].text(total, 0.34, f"{total:.3f} s end to end", ha="right", va="bottom", fontsize=7.1)
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.66),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.2,
    )

    values = np.asarray(
        [
            total,
            float(continuation["median_full_seconds"]),
            float(continuation["median_short_seconds"]),
        ]
    )
    labels = ("GREENCERT proof", "direct, 300 updates", "direct, 26 updates")
    y = np.arange(len(values))[::-1]
    colors = (COLORS["blue"], COLORS["ochre"], COLORS["gray"])
    markers = ("o", "s", "^")
    for ypos, value, color, marker in zip(y, values, colors, markers):
        axes[1].plot(value, ypos, marker=marker, ms=6.0, color=color, linestyle="none")
        axes[1].hlines(ypos, 0.08, value, color=color, linewidth=1.2, alpha=0.8)
        axes[1].text(value * 1.08, ypos, f"{value:.3f} s", va="center", fontsize=6.9)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.08, float(values.max()) * 1.35)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("median wall time (seconds, log scale)")
    axes[1].set_title("(b) Matched continuation controls")
    finish_axes(axes, grid_axis="x")

    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.35, wspace=0.36)
    pdf, _ = save_paper_figure(
        fig,
        "paper_composed_runtime",
        title="Composed GREENCERT runtime and matched continuation controls",
    )
    print(pdf)


if __name__ == "__main__":
    main()
