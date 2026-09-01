"""Figures drawn specifically for the slide deck.

The SIH template is white with dark blue titles, so everything here is rendered on white.
A results plot that arrives in the wrong theme looks borrowed rather than produced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#12233d"
BLUE = "#0e7fbf"
MUTED = "#41546f"
FAINT = "#e6ebf2"
ACCENT = "#c2255c"
GREEN = "#0f9d6b"


def pipeline_diagram(path: str | Path, width: float = 12.2, height: float = 3.5) -> Path:
    """The S0 to S8 flow, with both feedback edges drawn rather than described."""
    stages = [
        ("S0", "Ingest", ""),
        ("S1", "Pre-align", "kills scale\n+ viewpoint"),
        ("S2", "Re-illuminate", "N1"),
        ("S3", "Two tracks\n+ gate", "N2"),
        ("S4", "Sub-pixel\n+ covariance", "N4"),
        ("S5", "MAGSAC++\n+ local model", ""),
        ("S6", "Uniformity", "N3"),
        ("S7", "Products", ""),
        ("S8", "Evaluation", "N5"),
    ]

    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.set_xlim(0, len(stages) * 1.34)
    ax.set_ylim(-1.15, 2.05)
    ax.axis("off")

    box_w, box_h = 1.16, 0.86
    centres = []

    for i, (sid, title, note) in enumerate(stages):
        x = i * 1.34
        highlight = bool(note) and note not in ("kills scale\n+ viewpoint",)
        ax.add_patch(FancyBboxPatch(
            (x, -box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=1.6 if highlight else 1.0,
            edgecolor=BLUE if highlight else "#9fb0c6",
            facecolor="#eef6fc" if highlight else "#f7f9fc",
        ))
        ax.text(x + box_w / 2, 0.24, sid, ha="center", va="center",
                fontsize=8.5, color=BLUE, fontweight="bold", family="monospace")
        ax.text(x + box_w / 2, -0.09, title, ha="center", va="center",
                fontsize=8.6, color=INK, linespacing=1.25)
        if note:
            colour = ACCENT if highlight else MUTED
            ax.text(x + box_w / 2, -0.78, note, ha="center", va="center",
                    fontsize=7.4, color=colour, fontweight="bold" if highlight else "normal",
                    linespacing=1.2)
        centres.append(x + box_w / 2)

        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w, 0), (x + 1.34, 0),
                arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color="#8ea2ba",
            ))

    # The two feedback edges, drawn above the row so they read as returns. The label sits
    # clear of its own arc: an arc3 curve peaks at roughly rad/2 of the chord length, so
    # the text goes above that rather than through it.
    def feedback(a: int, b: int, rad: float, label: str) -> None:
        ax.add_patch(FancyArrowPatch(
            (centres[a], box_h / 2), (centres[b], box_h / 2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=9, linewidth=1.2,
            color=ACCENT, linestyle="--",
        ))
        apex = box_h / 2 + abs(rad) * abs(centres[a] - centres[b]) / 2.0
        ax.text((centres[a] + centres[b]) / 2, apex + 0.11, label,
                ha="center", va="bottom", fontsize=7.2, color=ACCENT)

    feedback(5, 1, 0.30, "S5 to S1: re-project from the corrected footprint")
    feedback(6, 3, 0.26, "S6 to S3: re-seed empty cells")

    fig.tight_layout(pad=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def illumination_strip(path: str | Path, images: dict[str, np.ndarray], ncc: float) -> Path:
    """Two renderings of one crater field under opposite Suns, and the number between them."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.5), facecolor="white")
    for ax, key, cap in zip(axes, ("east", "west"), ("Sun from the east", "Sun from the west")):
        ax.imshow(images[key], cmap="gray", interpolation="bilinear")
        ax.set_title(cap, fontsize=8, color=INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c3ccd9")
    fig.suptitle(f"Same ground, same viewpoint.   Correlation = {ncc:+.3f}",
                 fontsize=9, color=ACCENT, y=0.045, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(path)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path


def accuracy_bars(path: str | Path, rows: list[tuple[str, float]], title: str) -> Path:
    """Tie-point RMSE by method, on a log axis because the range spans three decades."""
    fig, ax = plt.subplots(figsize=(6.2, 2.42), facecolor="white")
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colours = [GREEN if "SETU" in n and "without" not in n else ("#f0a04b" if "without" in n else "#9fb0c6")
               for n in names]

    y = np.arange(len(rows))
    ax.barh(y, vals, color=colours, height=0.68, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.4, color=INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Tie-point RMSE against exact truth, pixels (log scale)", fontsize=7.5, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=7)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="x", color=FAINT, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3ccd9")

    for yi, v in zip(y, vals):
        ax.text(v * 1.16, yi, f"{v:.3g}", va="center", fontsize=7.2, color=INK)

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=210, facecolor="white")
    plt.close(fig)
    return path
