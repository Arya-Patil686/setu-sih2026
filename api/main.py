"""FastAPI service around the SETU pipeline.

Deliberately synchronous. A job queue is the first thing the specification says to cut if
time runs out, and for the scene sizes the demo uses a registration completes in seconds
- so the queue would add a failure mode and a deployment dependency without buying
anything a judge would notice.

The service also serves the built web demo, so `setu serve` starts one process that
answers both the API and the page.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from setu.config import SetuConfig
from setu.types import IlluminationState

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "api"
RUNS.mkdir(parents=True, exist_ok=True)
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(
    title="SETU",
    version="0.1.0",
    description="Sub-pixel multi-sensor registration of Chandrayaan-2 imagery. SIH 2026, PS 26166.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness, plus what this environment can actually do.

    The capability report is part of the answer rather than a separate endpoint: a result
    means something different depending on which matcher was available to produce it.
    """
    from setu.geom.sensor_model import ale_available
    from setu.match.deep import track_a_status

    status = track_a_status()
    return {
        "ok": True,
        "version": "0.1.0",
        "device": status["device"],
        "track_a": {"selected": status["selected"], "available": status["matchers"]},
        "tier_a_sensor_model": ale_available(),
    }


@app.get("/api/config")
def configs() -> dict[str, Any]:
    """The shipped experiment configurations and what each is for."""
    out = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = SetuConfig.load(path)
        out.append({"file": path.name, "name": cfg.name, "description": cfg.description.strip()})
    return {"configs": out}


@app.get("/api/reference-policy")
def reference_policy() -> dict[str, Any]:
    """Which reference each payload should be registered against, and why."""
    from setu.io.registry import REFERENCE_POLICY

    return {
        sensor: {
            "preferred": list(p.preferred),
            "acceptable_gsd_m": list(p.acceptable_gsd_m),
            "rationale": p.rationale,
        }
        for sensor, p in REFERENCE_POLICY.items()
    }


@app.post("/api/benchmark")
def benchmark(
    src_sun_az: float = Form(135.0),
    src_sun_elev: float = Form(20.0),
    ref_sun_az: float = Form(135.0),
    ref_sun_elev: float = Form(45.0),
    scale_ratio: float = Form(1.0),
    terrain: str = Form("highland"),
    reilluminate: bool = Form(True),
    tile_px: int = Form(512),
) -> JSONResponse:
    """Generate a benchmark pair with exact truth, register it, and return the numbers.

    This is what the live demo calls. Because the pair is synthesised here, the accuracy
    that comes back is measured against a transform that is known rather than fitted.
    """
    from setu.bench.generate import apply_h, make_pair
    from setu.bench.terrain import synthetic_terrain
    from setu.pipeline import Pipeline

    if terrain not in ("highland", "mare", "selfsimilar"):
        raise HTTPException(400, f"unknown terrain {terrain!r}")
    if not 1.0 <= scale_ratio <= 32.0:
        raise HTTPException(400, "scale_ratio must lie between 1 and 32")

    size = 3072 if scale_ratio > 4 else 1536
    patch = synthetic_terrain(size, 5.0, terrain, seed=26166)
    pair = make_pair(
        patch,
        illum_src=IlluminationState(sun_az_deg=src_sun_az, sun_elev_deg=src_sun_elev, source="synthetic"),
        illum_ref=IlluminationState(sun_az_deg=ref_sun_az, sun_elev_deg=ref_sun_elev, source="synthetic"),
        scale_ratio=scale_ratio, tile_px=tile_px, warp_kind="affine", seed=7,
    )

    cfg = SetuConfig.load(ROOT / "configs" / "synthetic.yaml")
    cfg.illum.reilluminate = reilluminate

    t0 = time.perf_counter()
    result = Pipeline(cfg).run(
        *pair.to_products(),
        dem=pair.dem_ref if reilluminate else None,
        dem_gsd_m=pair.gsd_ref_m,
        synthetic=True,
    )
    elapsed = time.perf_counter() - t0

    inliers = [t for t in result.tiepoints if t.inlier]
    truth_rmse = None
    if inliers:
        s = np.array([[t.src_sample, t.src_line] for t in inliers])
        r = np.array([[t.ref_sample, t.ref_line] for t in inliers])
        truth_rmse = float(np.sqrt(np.mean(np.sum((r - apply_h(pair.H_true, s)) ** 2, axis=1))))

    return JSONResponse({
        "run_id": result.run_id,
        "seconds": round(elapsed, 2),
        "metrics": json.loads(result.to_json())["metrics"],
        "rmse_vs_truth_px": round(truth_rmse, 4) if truth_rmse is not None else None,
        "rmse_vs_truth_m": round(truth_rmse * pair.gsd_ref_m, 3) if truth_rmse is not None else None,
        "stages": result.stages,
        "d_sun_az": round(pair.d_sun_az, 2),
        "d_sun_elev": round(pair.d_sun_elev, 2),
        "scale_ratio": pair.scale_ratio,
        "tiepoints": [
            {
                "x": round(t.src_sample, 2), "y": round(t.src_line, 2),
                "rx": round(t.ref_sample, 2), "ry": round(t.ref_line, 2),
                "residual": round(float(t.residual_norm), 4) if np.isfinite(t.residual_norm) else None,
                "sigma": round(float(t.sigma_rms), 4) if np.isfinite(t.sigma_rms) else None,
                "track": t.track, "inlier": bool(t.inlier), "reseeded": bool(t.reseeded),
            }
            for t in result.tiepoints
        ],
    })


@app.post("/api/register")
async def register(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    dem: UploadFile | None = File(None),
    config: str = Form("default"),
) -> JSONResponse:
    """Register two uploaded products.

    Files are written to a temporary directory and read back through the normal reader
    registry, so an upload takes exactly the same path as a file on disk.
    """
    from setu.bench.terrain import load_dem
    from setu.io.registry import read_product
    from setu.pipeline import Pipeline
    from setu.product.writers import write_run

    cfg_path = ROOT / "configs" / (config if config.endswith(".yaml") else f"{config}.yaml")
    if not cfg_path.exists():
        raise HTTPException(400, f"unknown config {config!r}")
    cfg = SetuConfig.load(cfg_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = {}
        for name, upload in (("source", source), ("reference", reference), ("dem", dem)):
            if upload is None:
                continue
            dest = tmp_path / (upload.filename or name)
            dest.write_bytes(await upload.read())
            paths[name] = dest

        try:
            src = read_product(paths["source"])
            ref = read_product(paths["reference"])
        except Exception as exc:
            raise HTTPException(400, f"could not read the uploaded products: {exc}") from exc

        dem_arr = dem_gsd = None
        if "dem" in paths:
            patch = load_dem(paths["dem"])
            dem_arr, dem_gsd = patch.dem, patch.gsd_m

        try:
            result = Pipeline(cfg).run(src, ref, dem=dem_arr, dem_gsd_m=dem_gsd)
        except Exception as exc:
            raise HTTPException(422, f"registration failed: {type(exc).__name__}: {exc}") from exc

        out_dir = RUNS / result.run_id
        written = write_run(out_dir, result, config=cfg, write_products=False)

    return JSONResponse({
        "run_id": result.run_id,
        "metrics": json.loads(result.to_json())["metrics"],
        "stages": result.stages,
        "files": {k: Path(v).name for k, v in written.items()},
        "download": f"/api/runs/{result.run_id}",
    })


@app.get("/api/runs/{run_id}/{filename}")
def download(run_id: str, filename: str) -> FileResponse:
    """Serve one artefact from a completed run."""
    # Resolve and confine: a run id is user input and must not escape the runs directory.
    target = (RUNS / run_id / filename).resolve()
    if not str(target).startswith(str(RUNS.resolve())) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/api/runs/{run_id}")
def run_manifest(run_id: str) -> dict[str, Any]:
    manifest = (RUNS / run_id / "manifest.json").resolve()
    if not str(manifest).startswith(str(RUNS.resolve())) or not manifest.is_file():
        raise HTTPException(404, "not found")
    return json.loads(manifest.read_text())


# The built web demo is mounted last so that it never shadows an /api route.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
