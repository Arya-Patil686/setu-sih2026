"""The two headline curves, plus the diagnostics that support them.

The problem statement names illumination variation and scale variation as its two
challenges. These plots answer both directly, on data whose ground truth is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setu.eval.runner import PairResult

#: Two palettes. The dark one matches the web demo; the light one matches the SIH slide
#: template, whose background is white. A figure that arrives in the wrong theme reads as
#: borrowed from somewhere else, which is the opposite of the impression a results plot
#: should give.
THEMES = {
    "dark": dict(bg="#05070c", fg="#e8eef7", muted="#8b9bb4", grid="#141d2b", spine="#1e2a3c"),
    "light": dict(bg="#ffffff", fg="#12233d", muted="#41546f", grid="#e6ebf2", spine="#c3ccd9"),
}
INK = THEMES["dark"]["bg"]
FG = THEMES["dark"]["fg"]
MUTED = THEMES["dark"]["muted"]
SERIES = ["#0e7fbf", "#e07b00", "#7c4dbf", "#0f9d6b", "#c2255c", "#2563eb", "#b91c1c"]
SERIES_DARK = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#fb7185"]


def _palette(theme: str) -> tuple[dict, list[str]]:
    t = THEMES.get(theme, THEMES["dark"])
    return t, (SERIES_DARK if theme == "dark" else SERIES)


def _style(ax, xlabel: str, ylabel: str, title: str = "", theme: str = "dark") -> None:
    t, _ = _palette(theme)
    ax.set_facecolor(t["bg"])
    ax.set_xlabel(xlabel, color=t["muted"], fontsize=10)
    ax.set_ylabel(ylabel, color=t["muted"], fontsize=10)
    if title:
        ax.set_title(title, color=t["fg"], fontsize=12, pad=12, loc="left")
    ax.tick_params(colors=t["muted"], labelsize=9)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(t["spine"])
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _group(results: Sequence[PairResult], x_attr: str, y_attr: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean of `y_attr` per unique value of `x_attr`, per method."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    by_method: dict[str, list[PairResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    for method, rows in by_method.items():
        buckets: dict[float, list[float]] = {}
        for r in rows:
            x, y = getattr(r, x_attr), getattr(r, y_attr)
            if not np.isfinite(x):
                continue
            # A failed run is a data point, not a missing one: dropping it would let a
            # method that fails on the hard half of the sweep look good on the easy half.
            buckets.setdefault(float(x), []).append(float(y) if np.isfinite(y) else np.nan)
        if not buckets:
            continue
        xs = np.array(sorted(buckets))
        ys = np.array([np.nanmean(buckets[x]) if np.isfinite(buckets[x]).any() else np.nan for x in xs])
        out[method] = (xs, ys)
    return out


def sweep_plot(
    results: Sequence[PairResult],
    x_attr: str,
    y_attr: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str | Path,
    log_x: bool = False,
    log_y: bool = False,
    methods: Sequence[str] | None = None,
    hline: float | None = None,
    hline_label: str = "",
    theme: str = "dark",
    figsize: tuple[float, float] = (7.6, 4.6),
    labels: dict[str, str] | None = None,
) -> Path:
    """One sweep curve per method."""
    t, series = _palette(theme)
    grouped = _group(results, x_attr, y_attr)
    if methods:
        grouped = {m: v for m, v in grouped.items() if m in methods}

    fig, ax = plt.subplots(figsize=figsize, facecolor=t["bg"])
    for i, (method, (xs, ys)) in enumerate(sorted(grouped.items())):
        emph = method == "setu_full"
        ax.plot(xs, ys, marker="o", markersize=5.5 if emph else 3.5,
                linewidth=2.8 if emph else 1.4,
                color=series[i % len(series)],
                label=(labels or {}).get(method, method),
                zorder=5 if emph else 2, alpha=1.0 if emph else 0.85)

    if hline is not None:
        ax.axhline(hline, color=t["muted"], linestyle="--", linewidth=1, zorder=1)
        if hline_label:
            ax.text(ax.get_xlim()[1], hline, f" {hline_label}", color=t["muted"],
                    fontsize=8, va="bottom", ha="right")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    _style(ax, xlabel, ylabel, title, theme)

    leg = ax.legend(frameon=False, fontsize=8.5, ncol=2)
    for text in leg.get_texts():
        text.set_color(t["fg"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor=t["bg"])
    plt.close(fig)
    return path


def rmse_vs_sun_elevation(results, path, methods=None, **kw) -> Path:
    """Headline curve one. Answers the problem statement's illumination challenge."""
    return sweep_plot(
        results, "d_sun_elev", "rmse_true_px",
        "Difference in solar elevation between source and reference (degrees)",
        "Registration RMSE against exact truth (px)",
        "Accuracy against illumination difference",
        path, log_y=True, methods=methods, hline=1.0, hline_label="1 px", **kw,
    )


def rmse_vs_scale(results, path, methods=None, **kw) -> Path:
    """Headline curve two. Answers the problem statement's scale challenge."""
    return sweep_plot(
        results, "scale_ratio", "rmse_true_px",
        "Ground sampling distance ratio between source and reference",
        "Registration RMSE against exact truth (px)",
        "Accuracy against scale ratio",
        path, log_x=True, log_y=True, methods=methods, hline=1.0, hline_label="1 px", **kw,
    )


def inliers_vs_sun_elevation(results, path, methods=None, **kw) -> Path:
    return sweep_plot(
        results, "d_sun_elev", "inlier_ratio",
        "Difference in solar elevation (degrees)", "Inlier ratio",
        "Inlier ratio against illumination difference", path, methods=methods, **kw,
    )


def calibration_plot(sigmas, residual_norms, path: str | Path) -> Path:
    """Predicted sigma against realised residual - the N4 claim, plotted."""
    from setu.eval.metrics import sigma_calibration

    cal = sigma_calibration(sigmas, residual_norms)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), facecolor=INK)
    if cal["curve"]:
        xs = [c["sigma_pred"] for c in cal["curve"]]
        ys = [c["residual_rms"] for c in cal["curve"]]
        lim = max(max(xs), max(ys)) * 1.15
        ax.plot([0, lim], [0, lim], color="#475569", linestyle="--", linewidth=1.2, label="perfect calibration")
        ax.plot(xs, ys, marker="o", color=SERIES[0], linewidth=2.2, label="measured")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    _style(ax, "Predicted sigma (px)", "Realised residual RMS (px)",
           f"Uncertainty calibration  ·  mean error {cal['calibration_error']:.3f} px")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(FG)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=INK)
    plt.close(fig)
    return path


def tiepoint_map(
    image: np.ndarray,
    tiepoints: Sequence[Any],
    path: str | Path,
    lattice: tuple[int, int] = (8, 8),
    title: str = "Tie points coloured by residual",
) -> Path:
    """Tie points over the image, coloured by residual. The strongest single visual."""
    fig, ax = plt.subplots(figsize=(7.2, 7.0), facecolor=INK)
    ax.imshow(image, cmap="gray", interpolation="nearest")

    xs = np.array([t.src_sample for t in tiepoints])
    ys = np.array([t.src_line for t in tiepoints])
    res = np.array([t.residual_norm for t in tiepoints])
    res = np.where(np.isfinite(res), res, 0.0)

    sc = ax.scatter(xs, ys, c=res, cmap="turbo", s=26, edgecolors="black", linewidths=0.4,
                    vmin=0, vmax=float(np.percentile(res, 95)) if res.size else 1.0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("residual (px)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    h, w = image.shape[:2]
    for i in range(1, lattice[0]):
        ax.axhline(i * h / lattice[0], color="#22d3ee", linewidth=0.5, alpha=0.35)
    for j in range(1, lattice[1]):
        ax.axvline(j * w / lattice[1], color="#22d3ee", linewidth=0.5, alpha=0.35)

    ax.set_title(title, color=FG, fontsize=12, loc="left", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=INK)
    plt.close(fig)
    return path


def residual_vector_field(image, tiepoints, path, scale: float = 40.0) -> Path:
    """Residual vectors, exaggerated, over the image. Shows systematic distortion at a glance."""
    fig, ax = plt.subplots(figsize=(7.2, 7.0), facecolor=INK)
    ax.imshow(image, cmap="gray", interpolation="nearest", alpha=0.75)

    xs = np.array([t.src_sample for t in tiepoints])
    ys = np.array([t.src_line for t in tiepoints])
    us = np.array([t.residual_x if np.isfinite(t.residual_x) else 0.0 for t in tiepoints])
    vs = np.array([t.residual_y if np.isfinite(t.residual_y) else 0.0 for t in tiepoints])

    ax.quiver(xs, ys, us * scale, vs * scale, color="#22d3ee", angles="xy",
              scale_units="xy", scale=1, width=0.0035)
    ax.set_title(f"Residual vector field (exaggerated {scale:g}x)", color=FG, fontsize=12, loc="left", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=INK)
    plt.close(fig)
    return path
