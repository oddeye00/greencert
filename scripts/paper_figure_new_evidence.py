#!/usr/bin/env python3
"""Publication figures for the real-data confirmation and new mechanism audits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, configure_paper_plots, save_paper_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def real_data_figure() -> None:
    audit = load(RESULTS / "real_dataset_confirmation" / "final_audit.json")
    outward = load(RESULTS / "real_dataset_outward_joined.json")
    rows = audit["rows"]
    issued = [row for row in rows if row["certificate_issued"]]
    thresholds = (0.90, 0.925, 0.95)
    labels = ("90%", "92.5%", "95%")
    colors = (COLORS["blue"], COLORS["teal"], COLORS["ochre"])

    candidates = [sum(abs(row["threshold"] - t) < 1e-12 for row in rows) for t in thresholds]
    issued_counts = [
        sum(row["certificate_issued"] and abs(row["threshold"] - t) < 1e-12 for row in rows)
        for t in thresholds
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.28))

    x = np.arange(3)
    axes[0].bar(x - 0.2, [24, 24, 24], width=0.2, color=COLORS["light"], label="cases")
    axes[0].bar(x, candidates, width=0.2, color=COLORS["gray"], label="trigger anchors")
    axes[0].bar(x + 0.2, issued_counts, width=0.2, color=colors, label="issued")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 26)
    axes[0].set_ylabel("seed--threshold cases")
    axes[0].set_title("(a) Candidates and issuance")
    for index, value in enumerate(issued_counts):
        axes[0].text(index + 0.2, value + 0.7, str(value), ha="center", color=colors[index], fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.31))

    rng = np.random.default_rng(1701)
    for index, (threshold, label, color) in enumerate(zip(thresholds, labels, colors)):
        leads = np.asarray(
            [row["actual_event"] for row in issued if abs(row["threshold"] - threshold) < 1e-12],
            dtype=float,
        )
        jitter = rng.uniform(-0.14, 0.14, size=len(leads))
        axes[1].scatter(
            np.full(len(leads), index) + jitter,
            leads,
            s=19,
            alpha=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )
        axes[1].plot([index - 0.2, index + 0.2], [np.median(leads)] * 2, color="black", lw=1.4)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("certified lead (updates)")
    axes[1].set_title("(b) Certified lead by gate")
    axes[1].grid(axis="y", alpha=0.18)

    outward_rows = [row for row in outward["rows"] if row["outward_issued"]]
    for threshold, label, color in zip(thresholds, labels, colors):
        selected = [row for row in outward_rows if abs(row["threshold"] - threshold) < 1e-12]
        actual = np.asarray([row["actual_event"] for row in selected], dtype=float)
        certified = np.asarray([row["outward_bracket"][0] for row in selected], dtype=float)
        axes[2].scatter(actual, certified, s=22, color=color, alpha=0.82, label=label, edgecolor="white", linewidth=0.35)
    maximum = max(row["actual_event"] for row in outward_rows)
    axes[2].plot([0, maximum + 8], [0, maximum + 8], linestyle="--", color="black", lw=0.9)
    axes[2].set_xlim(-4, maximum + 8)
    axes[2].set_ylim(-4, maximum + 8)
    axes[2].set_xlabel("revealed event lead")
    axes[2].set_ylabel("192-bit bracket location")
    axes[2].set_title("(c) Independent 192-bit replay")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(alpha=0.15)

    fig.tight_layout(pad=0.35, w_pad=1.0)
    save_paper_figure(
        fig,
        "paper_real_data_confirmation",
        title="WDBC certificate issuance, lead, and independent outward replay",
    )


def mechanism_figure() -> None:
    baseline = load(RESULTS / "transformer_unsigned_right_inverse_audit.json")
    sweep = load(RESULTS / "transformer_sweep_ablation.json")
    scaling = load(RESULTS / "transformer_batched_scaling_benchmark.json")
    comparable = [row for row in baseline["rows"] if row["green_operator_available"]]
    ratios = np.asarray([row["unsigned_to_signed_response_ratio"] for row in comparable])

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.34))

    axes[0].bar([0, 1], [9, 1], width=0.58, color=[COLORS["teal"], COLORS["gray"]])
    axes[0].set_xticks(
        [0, 1],
        ["signed\n" + r"$\Vert K_Hs\Vert$", "unsigned\n" + r"$\kappa\Vert s\Vert$"],
    )
    axes[0].set_ylim(0, 10.5)
    axes[0].set_ylabel("issued / 18 matched cases")
    axes[0].set_title("(a) Direction changes issuance")
    axes[0].text(0, 9.25, "9", ha="center", fontweight="bold", color=COLORS["teal"])
    axes[0].text(1, 1.25, "1", ha="center", fontweight="bold", color=COLORS["gray"])
    axes[0].text(
        0.5,
        6.4,
        f"median unsigned/signed\nratio: {np.median(ratios):.0f}$\\times$",
        ha="center",
        va="center",
        fontsize=6.7,
    )

    summaries = sweep["summary_by_sweep"]
    sweeps = np.asarray([row["sweep"] for row in summaries])
    defects = np.asarray([row["median_maximum_scaled_defect_norm"] for row in summaries])
    axes[1].plot(sweeps, defects, marker="o", color=COLORS["blue"], lw=1.8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(sweeps)
    axes[1].set_xlabel("variational sweeps")
    axes[1].set_ylabel("median max scaled defect")
    axes[1].set_title("(b) Variational defect contraction")
    axes[1].grid(alpha=0.18, which="both")
    for x, row in zip(sweeps, summaries):
        axes[1].annotate(
            f"{row['exact_event_clocks']}/3 exact",
            (x, row["median_maximum_scaled_defect_norm"]),
            xytext=(0, -12 if x == 0 else (7 if x % 2 == 0 else -12)),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
        )

    profiles = scaling["profiles"]
    parameters = np.asarray([row["parameter_count"] for row in profiles], dtype=float)
    serial_hours = np.asarray(
        [row["projection_h300"]["matched_serial_core_seconds"] / 3600 for row in profiles]
    )
    batched_hours = np.asarray(
        [row["projection_h300"]["projected_batched_core_seconds"] / 3600 for row in profiles]
    )
    axes[2].plot(
        parameters, serial_hours, marker="o", color=COLORS["gray"], label="serial probes"
    )
    axes[2].plot(
        parameters, batched_hours, marker="s", color=COLORS["ochre"], label="block-batched"
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("parameters")
    axes[2].set_ylabel("projected H=300 core hours")
    axes[2].set_title("(c) Batched operator scaling")
    axes[2].grid(alpha=0.18, which="both")
    axes[2].legend(frameon=False)
    axes[2].annotate(
        f"1M: {serial_hours[-1]:.2f} $\\to$ {batched_hours[-1]:.2f} h\n1.39 GiB peak",
        (parameters[-1], batched_hours[-1]),
        xytext=(-6, -32),
        textcoords="offset points",
        ha="right",
        fontsize=7.2,
    )

    fig.tight_layout(pad=0.35, w_pad=1.0)
    save_paper_figure(
        fig,
        "paper_mechanism_scaling",
        title="Signed propagation, variational contraction, and batched scaling",
    )


def main() -> None:
    configure_paper_plots()
    real_data_figure()
    mechanism_figure()
    print(ROOT / "figures" / "paper_real_data_confirmation.pdf")
    print(ROOT / "figures" / "paper_mechanism_scaling.pdf")


if __name__ == "__main__":
    main()
