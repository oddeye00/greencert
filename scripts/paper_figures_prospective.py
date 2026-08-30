#!/usr/bin/env python3
"""Publication figures for the prospective primary confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "prospective_v2_primary.json"
INTERVAL = ROOT / "results" / "prospective_v2_interval.json"
INK = COLORS["ink"]
V1 = COLORS["gray"]
V2 = COLORS["teal"]
EVENT = COLORS["vermilion"]
NATURAL = COLORS["pale"]
GRID = COLORS["light"]


def trigger_rows(payload: dict) -> list[dict]:
    rows = [row for row in payload["rows"] if row["anchor"] is not None]
    if len(rows) != 27:
        raise ValueError(f"expected 27 prospective triggers, found {len(rows)}")
    return sorted(rows, key=lambda row: (row["seed"], row["threshold"]))


def row_label(row: dict) -> str:
    return f"s{row['seed']} / {int(round(100 * row['threshold']))}%"


def horizon_figure(payload: dict) -> None:
    rows = trigger_rows(payload)
    y = np.arange(len(rows))
    h1 = np.asarray([row["v1_reached_horizon"] for row in rows], dtype=float)
    h2 = np.asarray([row["v2_reached_horizon"] for row in rows], dtype=float)
    leads = np.asarray([row["actual_lead"] for row in rows], dtype=float)
    inside = leads <= 250

    fig, ax = plt.subplots(figsize=(7.15, 5.75), constrained_layout=True)
    for yi, left, right in zip(y, h1, h2, strict=True):
        ax.plot([left, right], [yi, yi], color=GRID, lw=1.2, zorder=1)
    ax.scatter(
        h1,
        y,
        s=24,
        color=V1,
        edgecolor="white",
        linewidth=0.4,
        label="v1 horizon",
        zorder=3,
    )
    ax.scatter(
        h2,
        y,
        s=27,
        color=V2,
        edgecolor="white",
        linewidth=0.4,
        label="recentered v2 horizon",
        zorder=4,
    )
    ax.scatter(
        leads[inside],
        y[inside],
        s=34,
        marker="D",
        facecolors="none",
        edgecolors=EVENT,
        linewidth=1.2,
        label="observed event lead",
        zorder=5,
    )
    if np.any(~inside):
        ax.scatter(
            np.full(np.sum(~inside), 258.0),
            y[~inside],
            s=38,
            marker=">",
            color=EVENT,
            label="event after 250 steps",
            zorder=5,
        )
    for yi, row in zip(y, rows, strict=True):
        if row["v2_certificate_issued"]:
            ax.axhspan(yi - 0.43, yi + 0.43, color=V2, alpha=0.055, lw=0)

    ax.axvline(250, color=INK, lw=0.8, ls=(0, (3, 2)), alpha=0.7)
    ax.text(247, -1.35, "fixed H = 250", ha="right", va="bottom", fontsize=7)
    ax.set_yticks(y, [row_label(row) for row in rows])
    ax.set_xlim(0, 264)
    ax.set_ylim(len(rows) - 0.35, -1.7)
    ax.set_xlabel("steps after prospective trigger")
    ax.set_ylabel("seed / nominal gate")
    ax.set_title("All 27 outcome-sealed prospective-primary triggers")
    ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        frameon=False,
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    save_paper_figure(
        fig,
        "paper_prospective_horizons",
        title="Prospective v1 and recentered v2 certificate horizons",
    )


def bracket_figure(payload: dict, interval: dict) -> None:
    rows_by_key = {
        (int(row["seed"]), float(row["threshold"]), int(row["anchor"])): row
        for row in trigger_rows(payload)
    }
    issued = sorted(
        interval["rows"], key=lambda row: (row["seed"], row["threshold"])
    )
    if len(issued) != 16:
        raise ValueError(f"expected 16 outward brackets, found {len(issued)}")

    fig, ax = plt.subplots(figsize=(7.15, 5.0), constrained_layout=True)
    y = np.arange(len(issued))
    for yi, row in zip(y, issued, strict=True):
        key = (int(row["seed"]), float(row["threshold"]), int(row["anchor"]))
        full = rows_by_key[key]
        lo, hi = row["outward_bracket"]
        if row["seed"] == 17:
            ax.axhspan(yi - 0.44, yi + 0.44, color=NATURAL, lw=0, zorder=0)
        if lo == hi:
            ax.scatter([lo], [yi], marker="s", s=35, color=V2, zorder=4)
        else:
            ax.plot(
                [lo, hi], [yi, yi], color=V2, lw=5.0, solid_capstyle="butt", zorder=3
            )
            ax.scatter([lo, hi], [yi, yi], marker="|", s=55, color=V2, zorder=4)
        ax.scatter(
            [row["actual_lead"]],
            [yi + 0.16],
            marker="D",
            s=31,
            facecolors="white",
            edgecolors=EVENT,
            linewidth=1.2,
            zorder=5,
        )
        if full["v1_certificate_issued"]:
            v1lo, v1hi = full["v1_bracket"]
            ax.plot(
                [v1lo, v1hi],
                [yi - 0.16, yi - 0.16],
                color=V1,
                lw=2.0,
                solid_capstyle="butt",
                zorder=2,
            )

    ax.scatter([], [], marker="s", s=35, color=V2, label="outward v2 bracket")
    ax.scatter(
        [],
        [],
        marker="D",
        s=31,
        facecolors="white",
        edgecolors=EVENT,
        linewidth=1.2,
        label="observed first passage",
    )
    ax.plot([], [], color=V1, lw=2.0, label="v1 bracket when issued")
    ax.set_yticks(y, [row_label(row) for row in issued])
    ax.set_ylim(len(issued) - 0.45, -0.75)
    ax.set_xlim(0, 245)
    ax.set_xlabel("steps after prospective trigger")
    ax.set_ylabel("seed / nominal gate")
    ax.set_title("All 16 outward-retained brackets and observed first passages")
    ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    save_paper_figure(
        fig,
        "paper_prospective_brackets",
        title="Outward-retained brackets and observed first passages",
    )


def main() -> None:
    configure_paper_plots()
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    interval = json.loads(INTERVAL.read_text(encoding="utf-8"))
    horizon_figure(payload)
    bracket_figure(payload, interval)
    print(ROOT / "figures" / "paper_prospective_horizons.pdf")
    print(ROOT / "figures" / "paper_prospective_brackets.pdf")


if __name__ == "__main__":
    main()
