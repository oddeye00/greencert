#!/usr/bin/env python3
"""Publication figure for the frozen corrected-path scalability audits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    configure_paper_plots()
    panel = load("transformer_v3_relinearized_prefix_panel_audit.json")
    staged = load("transformer_direct_image_green_panel_audit.json")
    streaming = load("transformer_streaming_centerline_benchmark.json")

    rows = sorted(
        panel["rows"],
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
        ),
    )
    old = np.asarray(
        [row["directional_baseline_green_gram_applications"] for row in rows],
        dtype=float,
    )
    new = np.asarray(
        [row["relinearized_green_gram_applications"] for row in rows],
        dtype=float,
    )
    labels = [str(index) for index in range(1, len(rows) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.42))

    x = np.arange(len(rows))
    axes[0].plot(x, old, marker="o", ms=3.8, lw=1.4, color=COLORS["gray"], label="sealed directional")
    axes[0].plot(x, new, marker="s", ms=3.8, lw=1.5, color=COLORS["teal"], label="corrected prefix")
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks([4, 8, 16, 32, 64], labels=["4", "8", "16", "32", "64"])
    axes[0].set_xticks(x, labels, rotation=0)
    axes[0].set_xlabel("case (sorted by seed, gate)")
    axes[0].set_ylabel("Green Gram applications")
    axes[0].set_title("(a) Every frozen evaluable case")
    axes[0].grid(axis="y", which="major", alpha=0.18)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(
        0.03,
        0.06,
        "560 $\\to$ 64 total\n8.75$\\times$ reduction",
        transform=axes[0].transAxes,
        fontsize=6.7,
    )

    route = staged["route_distribution"]
    prefix = staged["prefix_distribution"]
    positions = np.asarray([0.0, 1.0, 2.4, 3.4, 4.4])
    values = np.asarray(
        [
            route["direct_image"],
            route["gram_fallback"],
            prefix["4"],
            prefix["8"],
            prefix["16"],
        ]
    )
    colors = [COLORS["ochre"], COLORS["blue"], COLORS["teal"], COLORS["vermilion"], COLORS["gray"]]
    axes[1].bar(positions, values, width=0.72, color=colors)
    axes[1].set_xticks(positions, ["direct", "Gram", "$m=4$", "$m=8$", "$m=16$"])
    axes[1].set_ylim(0, 16.5)
    axes[1].set_ylabel("issued cases")
    axes[1].set_title("(b) Staged release route")
    axes[1].axvline(1.7, color=COLORS["light"], lw=1.0)
    for xpos, value in zip(positions, values):
        axes[1].text(xpos, value + 0.45, str(int(value)), ha="center", fontweight="bold")
    stream_rows = sorted(streaming["rows"], key=lambda row: int(row["horizon"]))
    horizons = np.asarray([row["horizon"] for row in stream_rows], dtype=float)
    speed = np.asarray([row["speedup"] for row in stream_rows], dtype=float)
    memory = np.asarray(
        [row["estimated_centerline_memory_reduction"] for row in stream_rows],
        dtype=float,
    )
    axes[2].plot(horizons, speed, marker="o", color=COLORS["blue"], lw=1.7, label="time")
    axes[2].plot(horizons, memory, marker="s", color=COLORS["ochre"], lw=1.7, label="memory")
    axes[2].set_yscale("log")
    axes[2].set_xticks(horizons.astype(int))
    axes[2].set_xlabel("sealed horizon $H$")
    axes[2].set_ylabel("reduction factor")
    axes[2].set_title("(c) Exact causal-prefix streaming")
    axes[2].grid(alpha=0.18, which="both")
    axes[2].legend(frameon=False)
    for horizon, timing, mem in zip(horizons, speed, memory):
        axes[2].annotate(f"{timing:.2f}$\\times$", (horizon, timing), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8)
        axes[2].annotate(f"{mem:.1f}$\\times$", (horizon, mem), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8)

    fig.tight_layout(pad=0.35, w_pad=1.0)
    pdf, _ = save_paper_figure(
        fig,
        "paper_relinearized_prefix_panel",
        title="Corrected-prefix release and causal streaming costs",
    )
    print(pdf)


if __name__ == "__main__":
    main()
