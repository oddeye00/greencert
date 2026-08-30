#!/usr/bin/env python3
"""Publication figure for the sealed fresh signed-Green confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGGREGATE = RESULTS / "transformer_green_confirmation_audit.json"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json"
INK = COLORS["ink"]
TEAL = COLORS["teal"]
TEAL_DARK = COLORS["teal"]
GRAY = COLORS["gray"]
LIGHT = COLORS["light"]
RED = COLORS["vermilion"]
AMBER = COLORS["ochre"]


def load() -> tuple[list[dict], dict[tuple[int, float, int], dict]]:
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    seal = json.loads(CERTIFICATE_SEAL.read_text(encoding="utf-8"))
    rows = aggregate["rows"]
    if len(rows) != 23 or aggregate["summary"]["issued"] != 9:
        raise RuntimeError("fresh aggregate does not match the sealed 23/9 study")
    certificates = {}
    for entry in seal["certificate_files"]:
        candidate = entry["candidate"]
        key = (
            int(candidate["seed"]),
            float(candidate["threshold"]),
            int(candidate["anchor"]),
        )
        certificates[key] = json.loads((ROOT / entry["path"]).read_text(encoding="utf-8"))
    return rows, certificates


def key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return int(candidate["seed"]), float(candidate["threshold"]), int(candidate["anchor"])


def main() -> None:
    configure_paper_plots()
    rows, certificates = load()
    issued = [row for row in rows if row["certificate_issued"]]
    abstained = [
        row
        for row in rows
        if not row["certificate_issued"] and not row["construction_abstention"]
    ]
    construction = [row for row in rows if row["construction_abstention"]]
    if len(issued) != 9 or len(construction) != 1:
        raise RuntimeError("unexpected issuance/construction counts")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)
    ax = axes[0]
    diagonal = np.array([0.0, 300.0])
    ax.plot(diagonal, diagonal, color=LIGHT, lw=1.5, zorder=0)
    ax.scatter(
        [row["actual_event"] for row in abstained],
        [row["predicted_event"] for row in abstained],
        s=27,
        facecolors="white",
        edgecolors=GRAY,
        linewidth=1.0,
        label="certificate abstained",
        zorder=2,
    )
    ax.scatter(
        [row["actual_event"] for row in issued],
        [row["predicted_event"] for row in issued],
        s=34,
        color=TEAL,
        edgecolors="white",
        linewidth=0.5,
        label="certificate issued",
        zorder=3,
    )
    ax.scatter(
        [row["actual_event"] for row in construction],
        [row["predicted_event"] for row in construction],
        s=36,
        marker="x",
        color=AMBER,
        linewidth=1.3,
        label="construction abstention",
        zorder=4,
    )
    ax.text(
        0.04,
        0.94,
        "23/23 exact offsets\n9 singleton certificates",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        color=TEAL_DARK,
    )
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("revealed persistent-event lead")
    ax.set_ylabel("sealed clock prediction")
    ax.set_title("(a) Prospective timing on all frozen candidates", loc="left")
    ax.grid(color=LIGHT, lw=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        frameon=False,
        borderaxespad=0.4,
        handletextpad=0.45,
    )

    ax = axes[1]
    probed_pass = []
    probed_fail = []
    early = []
    for row in rows:
        certificate = certificates[key(row)]
        if row["construction_abstention"]:
            continue
        if certificate["green_probe"] is None:
            early.append(
                (
                    int(row["actual_event"]),
                    float(certificate["minimum_closure_lhs_using_kappa_ge_1"]),
                )
            )
        else:
            item = (
                int(row["actual_event"]),
                float(certificate["closure_lhs_2_kappa_M_Z"]),
            )
            (probed_pass if row["certificate_issued"] else probed_fail).append(item)

    ax.axhspan(1e-3, 1.0, color=TEAL, alpha=0.055, lw=0)
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.scatter(
        [item[0] for item in probed_pass],
        [item[1] for item in probed_pass],
        s=35,
        color=TEAL,
        edgecolors="white",
        linewidth=0.5,
        label=r"issued: $2\widehat\kappa MZ\leq1$",
        zorder=3,
    )
    ax.scatter(
        [item[0] for item in probed_fail],
        [item[1] for item in probed_fail],
        s=29,
        marker="x",
        color=RED,
        linewidth=1.1,
        label=r"Green closure failed",
        zorder=3,
    )
    ax.scatter(
        [item[0] for item in early],
        [item[1] for item in early],
        s=34,
        marker="^",
        facecolors="white",
        edgecolors=AMBER,
        linewidth=1.1,
        label=r"early abstain: lower bound $2MZ>1$",
        zorder=3,
    )
    ax.text(
        0.97,
        0.06,
        "1 additional construction abstention",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=GRAY,
    )
    ax.text(6, 1.15, "closure boundary", fontsize=7.0, color=INK)
    ax.set_yscale("log")
    ax.set_xlim(0, 300)
    ax.set_ylim(4e-3, 4e3)
    ax.set_xlabel("revealed persistent-event lead")
    ax.set_ylabel("nonlinear closure statistic")
    ax.set_title("(b) Closure criterion determines certificate issuance", loc="left")
    ax.grid(axis="y", which="both", color=LIGHT, lw=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=False,
        borderaxespad=0.4,
        handletextpad=0.45,
    )

    pdf, png = save_paper_figure(
        fig,
        "paper_transformer_green_confirmation",
        title="Prospective Transformer timing and nonlinear closure",
    )
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
