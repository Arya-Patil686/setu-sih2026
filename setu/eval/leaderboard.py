"""Section 10 - the leaderboard, as a markdown table you can paste into the deck."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from setu.eval.runner import PairResult

#: Column order. Accuracy first, then how much was found, then how well it was spread.
COLUMNS: list[tuple[str, str, str]] = [
    ("rmse_inliers_px", "Tie-point RMSE (px)", "{:.3f}"),
    ("rmse_true_px", "Model RMSE vs truth (px)", "{:.3f}"),
    ("precision_3px", "Precision @3px", "{:.1%}"),
    ("inlier_ratio", "Inlier ratio", "{:.1%}"),
    ("n_inliers", "Inliers", "{:.0f}"),
    ("coverage_matched", "Coverage @150 pts", "{:.2f}"),
    ("clark_evans_matched", "Clark-Evans R @150", "{:.2f}"),
    ("seconds", "Time (s)", "{:.1f}"),
]


def _cell(entry: dict[str, Any], key: str, fmt: str, with_ci: bool) -> str:
    # A method that failed on every pair gets said so, not left blank. "No registration"
    # is a result - it is what SIFT does on a 90-degree azimuth change - and a dash would
    # read as a gap in the experiment rather than as the finding it is.
    if entry.get("n_ok", 0) == 0:
        return "no registration"
    stat = entry.get(key)
    if not isinstance(stat, dict) or not np.isfinite(stat.get("value", np.nan)):
        return "-"
    value = fmt.format(stat["value"])
    if not with_ci or stat.get("n", 0) < 2:
        return value
    lo, hi = stat.get("ci_lo"), stat.get("ci_hi")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return value
    return f"{value}<br><sub>[{fmt.format(lo)}, {fmt.format(hi)}]</sub>"


def render_leaderboard(
    summary: dict[str, dict[str, Any]],
    title: str = "Leaderboard",
    order: Sequence[str] | None = None,
    with_ci: bool = True,
    notes: Sequence[str] = (),
) -> str:
    """Render the aggregated summary as a markdown table."""
    methods = list(order) if order else sorted(
        summary, key=lambda m: summary[m].get("rmse_true_px", {}).get("value", np.inf)
    )
    methods = [m for m in methods if m in summary]

    header = "| Method | " + " | ".join(c[1] for c in COLUMNS) + " |"
    divider = "|---|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [f"### {title}", "", header, divider]

    for m in methods:
        entry = summary[m]
        cells = [_cell(entry, key, fmt, with_ci) for key, _, fmt in COLUMNS]
        name = f"**{m}**" if m == "setu_full" else m
        if entry.get("n_failed"):
            name += f" <sub>({entry['n_failed']}/{entry['n_pairs']} failed)</sub>"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "**Tie-point RMSE** is the error of the delivered correspondences against exact "
        "truth, over inliers, and it is the number the problem statement's sub-pixel "
        "requirement refers to. **Model RMSE** is the error of the fitted transform; it "
        "falls roughly as the square root of the point count, so a method returning "
        "thousands of noisy points can score well on it while its individual tie points "
        "are useless. Uniformity is measured after subsampling every method to the same "
        "150 points, because coverage of an 8x8 lattice is otherwise a measure of density "
        "rather than of distribution. Square brackets are bootstrap 95% confidence "
        "intervals over pairs."
    )
    for n in notes:
        lines.append(f"\n{n}")
    return "\n".join(lines)


def render_ablation_table(summary: dict[str, dict[str, Any]]) -> str:
    """The ablation ladder - the table that shows what each novelty is worth."""
    from setu.eval.baselines import ABLATION_LABELS

    order = ["setu_no_reillum", "setu_no_structural", "setu_no_gate",
             "setu_no_refine", "setu_no_uniform", "setu_full"]
    present = [m for m in order if m in summary]
    if not present:
        return ""

    lines = [
        "### Ablation table",
        "",
        "| Configuration | Tie-point RMSE (px) | Model RMSE (px) | Precision @3px | Inlier ratio | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for m in present:
        e = summary[m]
        lines.append(
            f"| {ABLATION_LABELS.get(m, m)} | "
            + " | ".join(_cell(e, k, f, False) for k, f in
                         [("rmse_inliers_px", "{:.3f}"), ("rmse_true_px", "{:.3f}"),
                          ("precision_3px", "{:.1%}"), ("inlier_ratio", "{:.1%}"),
                          ("coverage", "{:.2f}")])
            + " |"
        )

    full = summary.get("setu_full", {})
    for m in present:
        if m == "setu_full":
            continue
        delta = _delta(summary[m], full, "rmse_inliers_px")
        if delta:
            lines.append("")
            lines.append(f"- **{ABLATION_LABELS.get(m, m)}**: tie-point RMSE {delta}.")
    return "\n".join(lines)


def _delta(without: dict[str, Any], full: dict[str, Any], key: str) -> str | None:
    a = without.get(key, {}).get("value")
    b = full.get(key, {}).get("value")
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return None
    return f"{a:.3f} px against {b:.3f} px, a factor of {a / b:.1f}"


def write_leaderboard(
    path: str | Path,
    summary: dict[str, dict[str, Any]],
    results: Sequence[PairResult] | None = None,
    title: str = "SETU leaderboard",
    context: dict[str, Any] | None = None,
) -> Path:
    """Write `leaderboard.md` next to the raw per-pair rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = [f"# {title}", ""]
    if context:
        parts.append("| Setting | Value |")
        parts.append("|---|---|")
        for k, v in context.items():
            parts.append(f"| {k} | {v} |")
        parts.append("")

    parts.append(render_leaderboard(summary, "All methods"))
    ablation = render_ablation_table(summary)
    if ablation:
        parts += ["", ablation]

    path.write_text("\n".join(parts))

    if results is not None:
        rows = [r.to_dict() for r in results]
        path.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    return path
