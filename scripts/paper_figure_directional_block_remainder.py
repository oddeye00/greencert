#!/usr/bin/env python3
"""Matplotlib figure for the frozen directional-block remainder audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, finish_axes, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def main() -> None:
    configure_paper_plots()
    directional = load("transformer_directional_block_remainder_diagnostic.json")
    event = load("transformer_directional_three_sweep_event_audit.json")
    rows = sorted(directional["rows"], key=key)
    events = {key(row): row for row in event["rows"]}
    x = np.arange(len(rows))
    tightening = np.asarray(
        [1.0 / float(row["directional_to_scalar_sequence_ratio"]) for row in rows]
    )
    labels = [
        f"{row['candidate']['seed']}/{int(100 * row['candidate']['threshold'])}"
        for row in rows
    ]
    colors = []
    for row in rows:
        if row["development_row"]:
            colors.append(COLORS["ochre"])
        elif row["closure_passed"]:
            colors.append(COLORS["teal"])
        else:
            colors.append(COLORS["gray"])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.52), gridspec_kw={"width_ratios": [1.55, 1.0]})
    axes[0].bar(x, tightening, width=0.72, color=colors, edgecolor="none")
    holdout_median = 2898.7919299228784
    axes[0].axhline(
        holdout_median,
        color=COLORS["blue"],
        lw=1.2,
        ls="--",
        label=f"holdout median {holdout_median:,.0f}$\\times$",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(700, 15000)
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].set_ylabel("scalar / directional Taylor bound")
    axes[0].set_xlabel("seed / accuracy gate (%)")
    axes[0].set_title("(a) Three known directions are worth keeping")
    axes[0].legend(loc="upper left")

    scalar_y, directional_y = 1.0, 0.0
    for index, row in enumerate(rows):
        old_pass = bool(row["parent_closure_passed"])
        new_pass = bool(row["closure_passed"])
        axes[1].scatter(
            index,
            scalar_y,
            marker="o" if old_pass else "x",
            s=28,
            linewidths=1.2,
            color=COLORS["teal"] if old_pass else COLORS["gray"],
            zorder=3,
        )
        axes[1].scatter(
            index,
            directional_y,
            marker="o" if new_pass else "x",
            s=28,
            linewidths=1.2,
            color=(
                COLORS["ochre"]
                if row["development_row"] and new_pass
                else COLORS["teal"] if new_pass else COLORS["gray"]
            ),
            zorder=3,
        )
        if new_pass:
            bracket = events[key(row)]["bracket"]
            axes[1].annotate(
                str(bracket[0]),
                (index, directional_y),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                fontsize=6.3,
                color=COLORS["ink"],
            )
    axes[1].set_xlim(-0.7, len(rows) - 0.3)
    axes[1].set_ylim(-0.38, 1.35)
    axes[1].set_yticks([directional_y, scalar_y], ["directional block", "scalar fourth"])
    axes[1].set_xticks(x, [str(index + 1) for index in x])
    axes[1].set_xlabel("case (same order as panel a)")
    axes[1].set_title("(b) Closure: 1/15 $\\to$ 4/15")
    axes[1].text(
        0.02,
        0.96,
        "circle = closes; x = abstains\nlabels are issued event offsets",
        transform=axes[1].transAxes,
        va="top",
        fontsize=6.6,
    )
    finish_axes(axes)
    fig.tight_layout(pad=0.35, w_pad=1.25)
    pdf, _ = save_paper_figure(
        fig,
        "paper_directional_block_remainder",
        title="Directional block remainder tightening and closure",
    )
    print(pdf)


if __name__ == "__main__":
    main()

