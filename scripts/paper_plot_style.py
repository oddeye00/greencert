#!/usr/bin/env python3
"""Shared, deterministic Matplotlib style for every GREENCERT paper figure."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# Muted, colorblind-safe colors.  Shape and line style remain redundant encodings.
COLORS = {
    "ink": "#263442",
    "blue": "#356A9A",
    "teal": "#2A7F62",
    "ochre": "#B98218",
    "vermilion": "#B75445",
    "gray": "#818B98",
    "light": "#DCE2E8",
    "pale": "#F3F5F7",
}

_RELEASE_DATE = datetime(2026, 8, 29, tzinfo=timezone.utc)


def configure_paper_plots() -> None:
    """Reset Matplotlib and apply one restrained two-column-paper style."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.linewidth": 0.75,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_axes(axes: Iterable[plt.Axes], *, grid_axis: str = "y") -> None:
    """Apply a light data grid without adding decorative panel furniture."""
    for axis in axes:
        axis.grid(
            axis=grid_axis,
            color=COLORS["light"],
            linewidth=0.45,
            alpha=0.75,
        )


def save_paper_figure(
    fig: plt.Figure,
    stem: str,
    *,
    title: str,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Write deterministic vector and raster copies with explicit provenance."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    creator = f"Matplotlib {mpl.__version__}"
    fig.savefig(
        pdf,
        metadata={
            "Title": title,
            "Author": "Ian Rhee",
            "Subject": "GREENCERT research figure",
            "Keywords": "GREENCERT, neural training, certification",
            "Creator": creator,
            "Producer": creator,
            "CreationDate": _RELEASE_DATE,
            "ModDate": _RELEASE_DATE,
        },
    )
    fig.savefig(
        png,
        dpi=dpi,
        metadata={
            "Title": title,
            "Author": "Ian Rhee",
            "Software": creator,
            "Creation Time": _RELEASE_DATE.isoformat(),
        },
    )
    plt.close(fig)
    return pdf, png
