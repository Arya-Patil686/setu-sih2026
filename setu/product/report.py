"""S7 - the self-contained QA report.

One HTML file with every image embedded as a data URI, so it survives being emailed,
copied onto a stick, or opened on a laptop with no network. A report that needs a server
to render is a report that will not be looked at.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from setu.types import RunResult, _json_default

CSS = """
:root{--ink:#05070c;--panel:#0b1018;--line:#1a2333;--fg:#e8eef7;--muted:#8b9bb4;
--accent:#22d3ee;--good:#34d399;--warn:#f59e0b;--bad:#f87171;}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--fg);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:48px 28px 96px}
h1{font-size:34px;letter-spacing:-.02em;margin:0 0 6px;font-weight:650}
h2{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
margin:56px 0 18px;font-weight:600}
.sub{color:var(--muted);margin:0 0 8px;font-size:15px}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;font-size:12.5px}
.grid{display:grid;gap:14px}
.cards{grid-template-columns:repeat(auto-fit,minmax(178px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px}
.card .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card .v{font-size:27px;font-weight:600;margin-top:8px;letter-spacing:-.02em;
font-variant-numeric:tabular-nums}
.card .u{font-size:12.5px;color:var(--muted);margin-top:4px}
.imgs{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;
overflow:hidden}
figure img{width:100%;display:block;background:#000}
figcaption{padding:11px 15px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
td.num{font-variant-numeric:tabular-nums;text-align:right}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
.pass{background:rgba(52,211,153,.14);color:var(--good)}
.fail{background:rgba(248,113,113,.14);color:var(--bad)}
.note{background:rgba(34,211,238,.05);border-left:2px solid var(--accent);
padding:14px 18px;border-radius:0 10px 10px 0;color:var(--muted);font-size:13.5px;margin:16px 0}
details{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:14px 18px;margin-top:14px}
summary{cursor:pointer;color:var(--muted);font-size:13px}
pre{overflow-x:auto;color:var(--muted);font-size:12px;line-height:1.55}
.stage{display:flex;gap:14px;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--line)}
.stage .id{font-family:ui-monospace,monospace;font-size:11px;color:var(--accent);
min-width:52px;letter-spacing:.05em}
.stage .lbl{flex:1}
.stage .t{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12.5px}
footer{margin-top:72px;padding-top:22px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}
@media(max-width:640px){.wrap{padding:32px 18px 64px}h1{font-size:26px}}
"""


def _data_uri(path: Path) -> str | None:
    """Embed a file as a data URI so the report stays self-contained."""
    if not path.exists():
        return None
    mime = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg"}.get(path.suffix.lower())
    if mime is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fig(path: Path | None, caption: str) -> str:
    if path is None:
        return ""
    uri = _data_uri(path)
    if uri is None:
        return ""
    return f'<figure><img src="{uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def _card(key: str, value: Any, unit: str = "") -> str:
    if value is None:
        value = "-"
    elif isinstance(value, float):
        value = f"{value:.4g}"
    return f'<div class="card"><div class="k">{key}</div><div class="v">{value}</div><div class="u">{unit}</div></div>'


def _tag(ok: bool) -> str:
    return f'<span class="tag {"pass" if ok else "fail"}">{"pass" if ok else "below target"}</span>'


def build_report(result: RunResult, written: dict[str, str], base: Path) -> str:
    """Render the QA report for one run."""
    m = result.metrics
    u = m.get("uniformity", {}) or {}
    g = m.get("gate", {}) or {}
    src, ref = result.source, result.reference
    si, ri = src.get("illumination", {}), ref.get("illumination", {})

    def f(path_key: str) -> Path | None:
        p = written.get(path_key)
        return Path(p) if p else None

    d_elev = abs(float(si.get("sun_elev_deg", 0)) - float(ri.get("sun_elev_deg", 0)))
    d_az = abs(float(si.get("sun_az_deg", 0)) - float(ri.get("sun_az_deg", 0))) % 360
    d_az = min(d_az, 360 - d_az)

    cards = "".join([
        _card("Tie points", m.get("n_inliers"), f"{m.get('n_tiepoints', 0)} before outlier rejection"),
        _card("Inlier ratio", f"{(m.get('inlier_ratio') or 0) * 100:.1f}%", "after MAGSAC++"),
        _card("Model RMSE", m.get("rmse_px"), f"px  ·  {m.get('rmse_m', '-')} m"),
        _card("CE90", m.get("ce90_px"), f"px  ·  {m.get('ce90_m', '-')} m"),
        _card("Median sigma", m.get("median_sigma_px"), "px per tie point"),
        _card("Coverage", u.get("coverage_ratio"), f"of {u.get('n_valid_cells', 0)} valid cells"),
        _card("Clark-Evans R", u.get("clark_evans_R"), "1.0 random, >1 dispersed"),
        _card("Runtime", m.get("runtime_s"), "seconds end to end"),
    ])

    stages = "".join(
        f'<div class="stage"><span class="id">{s.get("stage")}</span>'
        f'<span class="lbl">{s.get("label")}</span>'
        f'<span class="t">{s.get("seconds", 0):.2f}s</span></div>'
        for s in result.stages
    )

    reillum = next((s for s in result.stages if s.get("stage") == "S2a"), {})
    ncc_real = reillum.get("ncc_source_vs_real_reference")
    ncc_rend = reillum.get("ncc_source_vs_rendered_reference")
    reillum_block = ""
    if ncc_real is not None and ncc_rend is not None:
        reillum_block = f"""
        <div class="note"><strong>Re-illumination.</strong> Correlation between the source and
        the <em>real</em> reference was {ncc_real:.3f}; against the reference re-rendered at the
        source's own solar geometry it is {ncc_rend:.3f}. The two images were acquired
        {d_elev:.0f}&deg; apart in solar elevation and {d_az:.0f}&deg; apart in azimuth.</div>"""

    unif_rows = "".join(f"""
        <tr><td>{name}</td><td class="num">{value}</td><td>{target}</td><td>{_tag(bool(ok))}</td></tr>"""
        for name, value, target, ok in [
            ("Coverage ratio", u.get("coverage_ratio", "-"), "&ge; 0.90", u.get("coverage_pass", False)),
            ("Occupancy chi-square p", u.get("chi2_p", "-"), "&gt; 0.05", u.get("chi2_pass", False)),
            ("Clark-Evans R", u.get("clark_evans_R", "-"), "1.0 - 1.4", u.get("clark_evans_pass", False)),
        ])

    gate_rows = "".join(f'<tr><td>{k.replace("n_", "").replace("_", " ")}</td><td class="num">{v}</td></tr>'
                        for k, v in g.items() if isinstance(v, (int, float)) and k.startswith("n_"))

    figs = "".join([
        _fig(f("source_png"), "Source, ortho-projected onto the working grid (S1)"),
        _fig(f("reference_png"), "Reference, ortho-projected onto the same grid (S1)"),
        _fig(f("rendered_reference_png"), "Reference re-rendered at the source's solar geometry (S2a, novelty N1)"),
        _fig(f("registered_png"), "Registered product (S7)"),
        _fig(f("checkerboard_png"), "Checkerboard against the reference - broken edges mean misregistration"),
        _fig(f("swipe_png"), "Swipe comparison"),
        _fig(f("blink_gif"), "Blink comparison, before and after registration"),
        _fig(f("tiepoint_map_png"), "Tie points coloured by residual, over the lattice (S6)"),
        _fig(f("residual_field_png"), "Residual vector field, exaggerated"),
    ])

    gm = (result.transform.get("global") or {})
    lm = (result.transform.get("local") or {})
    pre = (result.transform.get("prealign") or {})

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SETU QA report - {result.run_id}</title><style>{CSS}</style></head>
<body><div class="wrap">

<h1>SETU registration report</h1>
<p class="sub mono">{result.run_id}</p>
<p class="sub">{src.get('sensor')} <strong>{src.get('pid')}</strong> registered against
{ref.get('sensor')} <strong>{ref.get('pid')}</strong> &middot; generated {result.created_utc}</p>

<h2>Headline numbers</h2>
<div class="grid cards">{cards}</div>

<div class="note"><strong>What these numbers are.</strong> Model RMSE is the residual of the
fitted transform on its own inliers. It is not an independent accuracy figure, and on this run
it is smaller than the per-tie-point error by roughly the square root of the tie-point count.
The per-point uncertainty column of <span class="mono">tiepoints.csv</span> is the honest
per-correspondence figure, and the leave-one-out RMSE below is the honest model figure.</div>

{reillum_block}

<h2>Pipeline</h2>
{stages}

<h2>Uniformity (S6, novelty N3)</h2>
<table><thead><tr><th>Statistic</th><th style="text-align:right">Value</th><th>Target</th><th>Status</th></tr></thead>
<tbody>{unif_rows}</tbody></table>

<h2>Agreement gate (S3, novelty N2)</h2>
<table><thead><tr><th>Count</th><th style="text-align:right">Value</th></tr></thead>
<tbody>{gate_rows}</tbody></table>

<h2>Transform</h2>
<table><tbody>
<tr><td>Global model</td><td class="num">{gm.get('kind', '-')}</td></tr>
<tr><td>Inlier threshold (adaptive, from the per-point covariances)</td><td class="num">{gm.get('threshold_px', '-')} px</td></tr>
<tr><td>Local model</td><td class="num">{lm.get('kind', 'none')}</td></tr>
<tr><td>Local fit RMSE</td><td class="num">{lm.get('fit_rmse_px', '-')} px</td></tr>
<tr><td>Local leave-one-out RMSE</td><td class="num">{lm.get('loocv_rmse_px', '-')} px</td></tr>
<tr><td>Local model applied</td><td class="num">{lm.get('applied', '-')}</td></tr>
<tr><td>Working GSD</td><td class="num">{m.get('gsd_work_m', '-')} m</td></tr>
<tr><td>Residual after pre-alignment</td><td class="num">{pre.get('residual_px', '-')} px</td></tr>
<tr><td>Sensor model tier</td><td class="num">{(pre.get('source_model') or {}).get('tier', '-')}</td></tr>
<tr><td>Uncertainty variance factor</td><td class="num">{m.get('sigma_variance_factor', '-')}</td></tr>
</tbody></table>

<h2>Imagery</h2>
<div class="grid imgs">{figs}</div>

<h2>Files</h2>
<table><thead><tr><th>Artefact</th><th>File</th></tr></thead><tbody>
{''.join(f'<tr><td>{k.replace("_", " ")}</td><td class="mono">{Path(v).name}</td></tr>' for k, v in sorted(written.items()))}
</tbody></table>

<details><summary>Resolved configuration and full metrics</summary>
<pre>{json.dumps({"metrics": result.metrics, "config": result.config}, indent=2, default=_json_default)}</pre>
</details>

<footer>SETU &middot; Smart India Hackathon 2026, problem statement 26166 (ISRO / Department of Space)<br>
Every number on this page was produced by <span class="mono">setu/eval/</span> from this run.
Nothing here is a screenshot of something measured elsewhere.</footer>
</div></body></html>"""


def write_report(path: str | Path, result: RunResult, written: dict[str, str], base: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(result, written, base))
    return path
