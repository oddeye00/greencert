#!/usr/bin/env python3
"""Create the response-centered Transformer/anytime paper figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, finish_axes, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
AUDIT = ROOT / "results" / "transformer_v3_confirmation_audit.json"
BENCHMARK = ROOT / "results" / "transformer_v3_online_policy_matched_audit.json"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    rows = audit["rows"]
    assert len(rows) == 19
    assert audit["v3_issued"] == audit["v3_covered"] == 11

    configure_paper_plots()
    teal = COLORS["teal"]
    navy = COLORS["blue"]
    gold = COLORS["ochre"]
    gray = COLORS["gray"]
    red = COLORS["vermilion"]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.24))
    fig.subplots_adjust(left=0.065, right=0.992, bottom=0.25, top=0.82, wspace=0.42)

    # (a) Every frozen v3 candidate, colored by first issuing power.
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["candidate"]["seed"],
            rows[i]["candidate"]["threshold"],
        ),
    )
    x = np.arange(len(order))
    leads = np.asarray([rows[i]["actual_persistent_event"] for i in order])
    issued = np.asarray([rows[i]["certificate_issued"] for i in order])
    powers = [rows[i]["earliest_issuing_power"] for i in order]
    axes[0].scatter(
        x[~issued],
        leads[~issued],
        s=25,
        facecolors="white",
        edgecolors=gray,
        linewidths=1.0,
        label="abstain",
        zorder=2,
    )
    power_colors = {1: gold, 2: teal, 3: navy, 4: red}
    for power in (1, 2, 3, 4):
        mask = np.asarray([value == power for value in powers])
        axes[0].scatter(
            x[mask],
            leads[mask],
            s=28,
            color=power_colors[power],
            edgecolors="white",
            linewidths=0.45,
            label=f"issue q={power}",
            zorder=3,
        )
    conversion = next(
        position
        for position, index in enumerate(order)
        if rows[index]["candidate"]["seed"] == 372
    )
    axes[0].scatter(
        [conversion],
        [leads[conversion]],
        s=72,
        marker="*",
        facecolors="none",
        edgecolors=red,
        linewidths=1.0,
        zorder=4,
    )
    axes[0].set_title("(a) Outcome-sealed candidates")
    axes[0].set_xlabel("candidate (seed/gate order)")
    axes[0].set_ylabel("revealed event lead")
    axes[0].set_xticks([0, 4, 9, 14, 18])
    axes[0].set_ylim(0, 305)
    axes[0].legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.1,
    )

    # (b) Certificates available as the shared probe block advances.
    q_grid = np.arange(1, 9)
    cumulative = np.asarray([sum(p is not None and p <= q for p in powers) for q in q_grid])
    axes[1].step(q_grid, cumulative, where="post", color=teal, linewidth=2.0)
    axes[1].scatter(q_grid[:4], cumulative[:4], color=teal, s=22, zorder=3)
    axes[1].hlines(
        audit["baseline_issued"],
        1,
        8,
        color=gray,
        linestyle="--",
        linewidth=1.2,
        label="fixed-radius q=8: 10",
    )
    axes[1].annotate(
        "11/11 covered",
        xy=(4, 11),
        xytext=(4.45, 9.1),
        arrowprops={"arrowstyle": "-", "color": navy, "lw": 0.8},
        color=navy,
    )
    axes[1].set_title("(b) Anytime issuance")
    axes[1].set_xlabel("maximum inspected Gram power")
    axes[1].set_ylabel("issued certificates")
    axes[1].set_xticks(q_grid)
    axes[1].set_ylim(0, 12)
    axes[1].legend(loc="lower right", frameon=False)

    # (c) Same executable, candidate, streams, and bracket.
    online = np.asarray(
        [benchmark["online_operator_seconds"], benchmark["online_end_to_end_seconds"]]
    )
    full = np.asarray(
        [
            benchmark["full_q8_operator_seconds"],
            benchmark["online_end_to_end_seconds"]
            * benchmark["measured_end_to_end_speedup"],
        ]
    )
    categories = np.arange(2)
    width = 0.34
    axes[2].bar(categories - width / 2, full, width, color=gray, label="forced q=8")
    axes[2].bar(categories + width / 2, online, width, color=teal, label="online stop")
    speedups = full / online
    for index, speedup in enumerate(speedups):
        axes[2].text(
            index,
            max(full[index], online[index]) + 9,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            color=navy,
            fontweight="bold",
        )
    axes[2].set_title("(c) Matched replay time")
    axes[2].set_ylabel("seconds")
    axes[2].set_xticks(categories, ["operator", "end-to-end"])
    axes[2].set_ylim(0, 380)
    axes[2].legend(loc="upper left", frameon=False)

    finish_axes(axes)
    pdf, png = save_paper_figure(
        fig,
        "paper_transformer_v3_anytime",
        title="Response-centered Transformer certificates and anytime replay",
        dpi=300,
    )
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
