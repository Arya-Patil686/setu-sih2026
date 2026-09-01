"""Section 10 - the command-line contract.

    setu register   one source against one reference, into a run directory
    setu bench      generate the controlled benchmark
    setu eval       run every method over a manifest and write the leaderboard
    setu serve      start the API and the web demo
    setu info       what this environment can actually do

Every command writes a machine-readable `metrics.json` and a human-readable
`report.html`; `setu eval` additionally writes `leaderboard.md`, which is a markdown
table that goes straight into the deck.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="SETU - sub-pixel multi-sensor registration of Chandrayaan-2 imagery.")
bench_app = typer.Typer(no_args_is_help=True, help="Generate the controlled benchmark.")
app.add_typer(bench_app, name="bench")

console = Console()


def _stage_printer():
    def progress(stage: str, label: str, detail: dict) -> None:
        if detail.get("status") == "done":
            console.print(f"  [cyan]{stage:<5}[/] {label:<46} [dim]{detail.get('seconds', 0):6.2f}s[/]")
    return progress


# --------------------------------------------------------------------- register

@app.command()
def register(
    source: Path = typer.Option(..., "--source", help="Source product (PDS4 xml, PDS3 lbl, or GeoTIFF)."),
    reference: Path = typer.Option(..., "--reference", help="Reference product."),
    out: Path = typer.Option(..., "--out", help="Run directory to write."),
    dem: Optional[Path] = typer.Option(None, "--dem", help="Shape model (SLDEM2015 tile or NAC DTM)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Experiment YAML."),
    source_sensor: Optional[str] = typer.Option(None, "--source-sensor"),
    reference_sensor: Optional[str] = typer.Option(None, "--reference-sensor"),
    no_products: bool = typer.Option(False, "--no-products", help="Metrics only; skip rasters and previews."),
) -> None:
    """Register one source image against one reference image."""
    from setu.bench.terrain import load_dem
    from setu.config import SetuConfig
    from setu.io.registry import read_product
    from setu.pipeline import Pipeline
    from setu.product.warp import warp_with_local
    from setu.product.writers import write_run

    cfg = SetuConfig.load(config)
    console.print(f"[bold]SETU[/] register  ·  config [cyan]{cfg.name}[/]")

    src = read_product(source, **({"sensor": source_sensor} if source_sensor else {}))
    ref = read_product(reference, **({"sensor": reference_sensor} if reference_sensor else {}))
    console.print(f"  source    {src.sensor:<12} {src.pid}  {src.array.shape}  {src.gsd_m} m/px")
    console.print(f"  reference {ref.sensor:<12} {ref.pid}  {ref.array.shape}  {ref.gsd_m} m/px")

    dem_arr = dem_gsd = None
    if dem is not None:
        patch = load_dem(dem)
        dem_arr, dem_gsd = patch.dem, patch.gsd_m
        console.print(f"  terrain   {patch.source}  {patch.dem.shape}  {patch.gsd_m:.1f} m/px")
    else:
        console.print("  [yellow]no DEM supplied: re-illumination (novelty N1) is disabled for this run[/]")

    t0 = time.time()
    pipeline = Pipeline(cfg, _stage_printer())
    result = pipeline.run(src, ref, dem=dem_arr, dem_gsd_m=dem_gsd, run_id=out.name)

    gm = result.transform.get("global")
    registered = None
    if gm and not no_products:
        import numpy as np

        pre = pipeline.stages[1].detail
        registered = warp_with_local(
            src.pan(), np.linalg.inv(np.array(gm["matrix"])),
            (int(pre["grid"]["height"]), int(pre["grid"]["width"])),
            resample=cfg.product.resample,
        )

    written = write_run(out, result, registered=registered, config=cfg, write_products=not no_products)
    _print_metrics(result.metrics)
    console.print(f"\n[green]done[/] in {time.time() - t0:.1f}s  ->  [cyan]{out}[/]")
    console.print(f"  report  {written.get('report_html')}")


def _print_metrics(m: dict) -> None:
    u = m.get("uniformity", {}) or {}
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    for label, value in [
        ("tie points (inliers)", f"{m.get('n_inliers')} of {m.get('n_tiepoints')}"),
        ("inlier ratio", f"{(m.get('inlier_ratio') or 0) * 100:.1f}%"),
        ("model RMSE", f"{m.get('rmse_px')} px  /  {m.get('rmse_m')} m"),
        ("CE90", f"{m.get('ce90_px')} px  /  {m.get('ce90_m')} m"),
        ("median sigma per point", f"{m.get('median_sigma_px')} px"),
        ("coverage ratio", f"{u.get('coverage_ratio')}  (target 0.90)"),
        ("Clark-Evans R", f"{u.get('clark_evans_R')}  (target 1.0-1.4)"),
        ("runtime", f"{m.get('runtime_s')} s"),
    ]:
        table.add_row(label, str(value))
    console.print()
    console.print(table)


# ------------------------------------------------------------------------ bench

@bench_app.command("generate")
def bench_generate(
    out: Path = typer.Option(..., "--out", help="Directory to write the benchmark into."),
    dem: Optional[Path] = typer.Option(None, "--dem", help="DEM tile; synthesised if absent."),
    sun_elev: str = typer.Option("10,20,30,45,60,75", "--sun-elev"),
    sun_az: str = typer.Option("0,45,90,135,180,225,270,315", "--sun-az"),
    scale: str = typer.Option("1,2,4,8,16", "--scale"),
    emission: str = typer.Option("0,10,25", "--emission"),
    terrain: str = typer.Option("highland", "--terrain", help="highland, mare or selfsimilar."),
    size: int = typer.Option(1536, "--size", help="DEM edge in pixels."),
    gsd: float = typer.Option(5.0, "--gsd", help="DEM ground sampling distance, metres."),
    tile: int = typer.Option(512, "--tile"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap the number of pairs."),
) -> None:
    """Generate the controlled benchmark with exact ground truth."""
    from setu.bench.generate import full_grid, write_bench
    from setu.bench.terrain import get_terrain

    nums = lambda s: [float(x) for x in s.split(",") if x.strip()]  # noqa: E731

    patch = get_terrain(dem, size=size, gsd_m=gsd, terrain=terrain)
    console.print(f"[bold]SETU[/] bench generate  ·  terrain [cyan]{patch.source}[/] "
                  f"{patch.dem.shape} at {patch.gsd_m:.1f} m/px")

    pairs = full_grid(patch, nums(sun_az), nums(sun_elev), nums(emission), nums(scale),
                      tile_px=tile, limit=limit)
    doc = write_bench(out, pairs)
    console.print(f"[green]wrote[/] {doc['n_pairs']} pairs  ->  [cyan]{out}[/]")


# ------------------------------------------------------------------------- eval

@app.command("eval")
def evaluate(
    out: Path = typer.Option(..., "--out", help="Directory for the leaderboard and plots."),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Benchmark manifest.json."),
    methods: str = typer.Option("all", "--methods", help="Comma-separated, or 'all'."),
    config: Optional[Path] = typer.Option(None, "--config"),
    sun_sweep: bool = typer.Option(True, "--sun-sweep/--no-sun-sweep"),
    scale_sweep_flag: bool = typer.Option(True, "--scale-sweep/--no-scale-sweep"),
    terrain: str = typer.Option("highland", "--terrain"),
    size: int = typer.Option(1536, "--size"),
    bootstrap: int = typer.Option(1000, "--bootstrap"),
) -> None:
    """Run every method over the benchmark and write the leaderboard."""
    from setu.bench.generate import load_bench_pair, make_pair, scale_sweep
    from setu.bench.terrain import synthetic_terrain
    from setu.config import SetuConfig
    from setu.eval.baselines import ABLATIONS, CLASSICAL, DEEP
    from setu.eval.leaderboard import render_leaderboard, write_leaderboard
    from setu.eval.plots import rmse_vs_scale, rmse_vs_sun_elevation
    from setu.eval.runner import aggregate, run_suite
    from setu.types import IlluminationState

    cfg = SetuConfig.load(config or "configs/synthetic.yaml")
    method_list = (CLASSICAL + DEEP + list(ABLATIONS)) if methods == "all" else \
        [m.strip() for m in methods.split(",") if m.strip()]

    pairs = []
    if manifest is not None:
        doc = json.loads(Path(manifest).read_text())
        base = Path(manifest).parent
        pairs = [load_bench_pair(base / e["file"], e) for e in doc["pairs"] if "file" in e]
        console.print(f"loaded {len(pairs)} pairs from {manifest}")
    else:
        patch = synthetic_terrain(size, 5.0, terrain, seed=26166)
        if sun_sweep:
            pairs += [
                make_pair(patch,
                          IlluminationState(sun_az_deg=135.0, sun_elev_deg=e, source="synthetic"),
                          IlluminationState(sun_az_deg=135.0, sun_elev_deg=70.0, source="synthetic"),
                          scale_ratio=1.0, tile_px=512, warp_kind="affine",
                          pair_id=f"sun{e:g}", seed=100 + i)
                for i, e in enumerate([10, 20, 30, 40, 50, 60, 70])
            ]
        if scale_sweep_flag:
            big = synthetic_terrain(max(size, 3072), 5.0, terrain, seed=26166)
            pairs += list(scale_sweep(big, ratios=[1, 2, 4, 8, 16], tile_px=512, warp_kind="affine"))
        console.print(f"generated {len(pairs)} benchmark pairs ({terrain})")

    if not pairs:
        console.print("[red]no pairs to evaluate[/]")
        raise typer.Exit(1)

    total = len(pairs) * len(method_list)
    console.print(f"running {len(method_list)} methods over {len(pairs)} pairs  ({total} runs)")

    t0 = time.time()
    results = run_suite(
        pairs, method_list, cfg,
        progress=lambda m, p, d, t: console.print(f"  [dim]{d:4d}/{t}[/] {m:<20} {p}"),
    )
    summary = aggregate(results, n_boot=bootstrap)

    write_leaderboard(out / "leaderboard.md", summary, results,
                      title="SETU evaluation", context={"pairs": len(pairs), "methods": len(method_list)})
    if sun_sweep:
        rmse_vs_sun_elevation(results, out / "rmse_vs_sun_elevation.png")
    if scale_sweep_flag:
        rmse_vs_scale(results, out / "rmse_vs_scale.png")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    console.print(f"\n[green]done[/] in {time.time() - t0:.0f}s  ->  [cyan]{out}[/]")
    console.print(render_leaderboard(summary, "Results", with_ci=False))


# ------------------------------------------------------------------------ serve

@app.command()
def serve(
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the API and serve the web demo."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is not installed.[/]  pip install 'setu[api]'")
        raise typer.Exit(1)

    console.print(f"[bold]SETU[/] serving on [cyan]http://{host}:{port}[/]")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


# ------------------------------------------------------------------------- info

@app.command()
def info() -> None:
    """Report what this environment can actually do.

    Printed rather than assumed, because which matcher was available is part of what a
    result means.
    """
    from setu.geom.sensor_model import ale_available
    from setu.io.isis import isis_available
    from setu.match.deep import track_a_status

    console.print("[bold]SETU[/] environment\n")
    status = track_a_status()

    table = Table(show_header=True, header_style="dim")
    table.add_column("Capability")
    table.add_column("Status")
    table.add_column("Consequence", style="dim")

    table.add_row("torch", "yes" if status["torch"] else "no",
                  "track A unavailable without it; track B still runs" if not status["torch"] else
                  f"device: {status['device']}")
    for name, ok in status["matchers"].items():
        table.add_row(f"  {name}", "[green]yes[/]" if ok is True else "[yellow]no[/]",
                      "selected" if name == status["selected"] else "")
    table.add_row("ISIS (kalasiris)", "yes" if isis_available() else "no", "optional, Tier A only")
    table.add_row("ale (CSM driver)", "yes" if ale_available() else "no",
                  "Tier B corner-fit model is the default and is sufficient")

    console.print(table)
    console.print(f"\ntrack A matcher selected: [cyan]{status['selected'] or 'none'}[/]")
    if status["selected"] != "matchanything_roma":
        console.print("[dim]MatchAnything weights are not bundled. Point SETU_MATCHANYTHING_DIR at your own\n"
                      "download of zju3dv/MatchAnything to enable them.[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main() or 0)
