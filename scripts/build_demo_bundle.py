"""Build the data and imagery the web demo renders.

Everything the site shows is produced here from real runs: the illumination
demonstration, three complete registrations at increasing difficulty, and the evaluation
sweeps. Nothing on the page is drawn by hand or transcribed, so any figure a judge points
at can be traced back to a `metrics.json`.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from setu.bench.generate import apply_h, make_pair
from setu.bench.terrain import synthetic_terrain
from setu.config import SetuConfig
from setu.illum.render import render_dem, render_similarity
from setu.pipeline import Pipeline
from setu.product.warp import _norm, checkerboard, to_png_bytes, warp_global
from setu.types import IlluminationState

OUT = ROOT / "web" / "public" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
MAX_PX = 620


def json_safe(obj):
    """Recursively replace non-finite floats with None.

    `json.dumps` happily writes bare `NaN` and `Infinity`, which are not valid JSON and
    which `JSON.parse` rejects outright. A single NaN buried in a bootstrap interval is
    therefore enough to make the whole results file unreadable in a browser, and it fails
    silently because the fetch succeeds and only the parse throws.
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def dump_json(path: Path, payload) -> None:
    """Write JSON a browser can actually parse."""
    path.write_text(json.dumps(json_safe(payload), indent=1, allow_nan=False, default=str))


def save(name: str, image: np.ndarray, max_px: int = MAX_PX) -> str:
    (OUT / name).write_bytes(to_png_bytes(image, max_px))
    return f"demo/{name}"


def save_rgb(name: str, rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("png encode failed")
    (OUT / name).write_bytes(buf.tobytes())
    return f"demo/{name}"


# --------------------------------------------------------- illumination demo

def illumination_demo() -> dict:
    """The one fact the whole project turns on, rendered from one terrain patch.

    Same ground, same viewpoint, same everything except where the Sun is. The
    correlation between the two is strongly *negative*: at opposite azimuths every
    crater's lit and shaded sides swap, so a gradient-based descriptor sees an inverted
    image rather than a changed one.
    """
    patch = synthetic_terrain(768, 5.0, "highland", seed=26166)
    crop = slice(140, 640), slice(140, 640)

    a = render_dem(patch.dem, 5.0, IlluminationState(sun_az_deg=90, sun_elev_deg=18)).image[crop]
    b = render_dem(patch.dem, 5.0, IlluminationState(sun_az_deg=270, sun_elev_deg=18)).image[crop]
    c = render_dem(patch.dem, 5.0, IlluminationState(sun_az_deg=90, sun_elev_deg=62)).image[crop]

    from setu.illum.structural import structural_transform

    pa = structural_transform(a, "pc")
    pb = structural_transform(b, "pc")

    return {
        "images": {
            "sun_east": save("illum_east.png", a),
            "sun_west": save("illum_west.png", b),
            "sun_high": save("illum_high.png", c),
            "pc_east": save("illum_pc_east.png", pa),
            "pc_west": save("illum_pc_west.png", pb),
        },
        "ncc_opposite_azimuth": round(float(render_similarity(a, b)["ncc"]), 4),
        "ncc_elevation_change": round(float(render_similarity(a, c)["ncc"]), 4),
        "ncc_structural_opposite": round(float(render_similarity(pa, pb)["ncc"]), 4),
        "shadow_fraction_low_sun": round(
            float(render_dem(patch.dem, 5.0, IlluminationState(sun_az_deg=90, sun_elev_deg=18)).shadow_fraction), 4),
        "shadow_fraction_high_sun": round(
            float(render_dem(patch.dem, 5.0, IlluminationState(sun_az_deg=90, sun_elev_deg=62)).shadow_fraction), 4),
    }


# ------------------------------------------------------------ scene registration

SCENES = [
    {"key": "ohrc_nac", "label": "OHRC against LRO NAC",
         "blurb": "A 30-degree change in solar azimuth at low sun, the case the problem statement calls illumination variation.",
         "src": (105.0, 22.0), "ref": (135.0, 22.0), "scale": 1.0, "terrain": "highland",
         "src_sensor": "OHRC", "ref_sensor": "NAC_L"},
    {"key": "hard_sun", "label": "Opposed illumination",
         "blurb": "The stress case: the Sun has moved 90 degrees in azimuth and 45 degrees in elevation between the two acquisitions.",
         "src": (45.0, 15.0), "ref": (135.0, 60.0), "scale": 1.0, "terrain": "highland",
         "src_sensor": "OHRC", "ref_sensor": "NAC_L"},
    {"key": "tmc2_scale", "label": "TMC-2 against NAC, 4x scale",
         "blurb": "A fourfold difference in ground sampling distance, absorbed by pre-alignment rather than by the matcher. Accuracy is quoted in reference pixels, so 4x coarser source pixels means the equivalent source-frame error is a quarter of the number shown.",
         "src": (135.0, 30.0), "ref": (135.0, 50.0), "scale": 4.0, "terrain": "highland",
         "src_sensor": "TMC2_NADIR", "ref_sensor": "NAC_L"},
]


def run_scene(scene: dict, cfg: SetuConfig) -> dict:
    size = 4096 if scene["scale"] > 2 else 1536
    patch = synthetic_terrain(size, 5.0, scene["terrain"], seed=26166)

    pair = make_pair(
        patch,
        illum_src=IlluminationState(sun_az_deg=scene["src"][0], sun_elev_deg=scene["src"][1], source="synthetic"),
        illum_ref=IlluminationState(sun_az_deg=scene["ref"][0], sun_elev_deg=scene["ref"][1], source="synthetic"),
        scale_ratio=scene["scale"], tile_px=512, min_src_px=256, warp_kind="affine",
        src_sensor=scene["src_sensor"], ref_sensor=scene["ref_sensor"],
        pair_id=scene["key"], seed=7,
    )

    src, ref = pair.to_products()
    src.sensor = scene["src_sensor"]      # type: ignore[assignment]
    ref.sensor = scene["ref_sensor"]      # type: ignore[assignment]

    stages: list[dict] = []
    result = Pipeline(cfg, lambda st, lb, d: None).run(
        src, ref, dem=pair.dem_ref, dem_gsd_m=pair.gsd_ref_m,
        run_id=scene["key"], synthetic=True,
    )
    stages = result.stages

    k = scene["key"]
    images = {
        "source": save(f"{k}_source.png", pair.source),
        "reference": save(f"{k}_reference.png", pair.reference),
    }

    # The rendered reference, recomputed for display exactly as S2a produced it.
    from setu.illum.render import reilluminate_reference

    rendered = reilluminate_reference(
        pair.dem_ref, pair.gsd_ref_m, pair.illum_src,
        reference_image=pair.reference, source_image=pair.source,
    )
    images["rendered"] = save(f"{k}_rendered.png", rendered.image)
    images["shadow"] = save(f"{k}_shadow.png", rendered.shadow.astype(np.float32))

    gm = result.transform.get("global")
    registered = None
    if gm:
        registered = warp_global(pair.source, np.array(gm["matrix"]),
                                 pair.reference.shape[:2], resample="lanczos")
        images["registered"] = save(f"{k}_registered.png", registered)
        images["checkerboard"] = save(f"{k}_checker.png", checkerboard(registered, pair.reference, tile=52))
        images["before"] = save(f"{k}_before.png", _overlay(_resize(pair.source, pair.reference.shape), pair.reference))
        images["after"] = save(f"{k}_after.png", _overlay(registered, pair.reference))

    inliers = [t for t in result.tiepoints if t.inlier]
    tiepoints = [
        {
            "x": round(t.src_sample / pair.source.shape[1], 5),
            "y": round(t.src_line / pair.source.shape[0], 5),
            "rx": round(t.ref_sample / pair.reference.shape[1], 5),
            "ry": round(t.ref_line / pair.reference.shape[0], 5),
            "r": round(float(t.residual_norm) if np.isfinite(t.residual_norm) else 0.0, 4),
            "sx": round(float(t.sigma_rms) if np.isfinite(t.sigma_rms) else 0.0, 4),
            "conf": round(float(t.conf), 3),
            "track": t.track,
            "inlier": bool(t.inlier),
            "reseeded": bool(t.reseeded),
        }
        for t in result.tiepoints
    ]

    # Accuracy against exact truth, which is the only number that matters here.
    truth_err = float("nan")
    if inliers:
        s = np.array([[t.src_sample, t.src_line] for t in inliers])
        r = np.array([[t.ref_sample, t.ref_line] for t in inliers])
        truth_err = float(np.sqrt(np.mean(np.sum((r - apply_h(pair.H_true, s)) ** 2, axis=1))))

    s2a = next((x for x in stages if x["stage"] == "S2a"), {})

    metrics = dict(result.metrics)
    metrics["rmse_vs_truth_px"] = round(truth_err, 4) if np.isfinite(truth_err) else None
    metrics["rmse_vs_truth_m"] = (
        round(truth_err * pair.gsd_ref_m, 3) if np.isfinite(truth_err) else None
    )

    return {
        "key": k,
        "label": scene["label"],
        "blurb": scene["blurb"],
        "run_id": result.run_id,
        "source": {**result.source, "sensor": scene["src_sensor"]},
        "reference": {**result.reference, "sensor": scene["ref_sensor"]},
        "metrics": metrics,
        "stages": stages,
        "images": images,
        "tiepoints": tiepoints,
        "truth": {
            "ncc_real": s2a.get("ncc_source_vs_real_reference"),
            "ncc_rendered": s2a.get("ncc_source_vs_rendered_reference"),
            "d_sun_elev": round(pair.d_sun_elev, 1),
            "d_sun_az": round(pair.d_sun_az, 1),
            "scale_ratio": pair.scale_ratio,
            "gsd_src_m": pair.gsd_src_m,
            "gsd_ref_m": pair.gsd_ref_m,
        },
    }


def _resize(img: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(np.asarray(img, np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-colour overlay: cyan for one image, warm for the other.

    Misregistration reads instantly as colour fringing, and alignment as neutral grey.
    """
    an, bn = _norm(a), _norm(b)
    rgb = np.zeros((*an.shape, 3), np.float32)
    rgb[..., 0] = bn                       # R from the reference
    rgb[..., 1] = 0.5 * (an + bn)
    rgb[..., 2] = an                       # B from the source
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------- eval copy

def copy_eval() -> dict | None:
    src = ROOT / "runs" / "eval_full"
    metrics = src / "metrics.json"
    if not metrics.exists():
        return None

    doc = json.loads(metrics.read_text())
    rows_path = src / "leaderboard.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []

    keep = ("method", "pair_id", "ok", "d_sun_elev", "d_sun_az", "scale_ratio",
            "rmse_inliers_px", "rmse_true_px", "rmse_points_px", "inlier_ratio",
            "coverage", "coverage_matched", "clark_evans_matched", "n_inliers",
            "precision_3px", "seconds")
    doc["rows"] = [{k: r.get(k) for k in keep} for r in rows]
    doc.pop("config", None)

    for png in src.glob("*.png"):
        shutil.copy(png, OUT / png.name)

    dump_json(OUT / "eval.json", doc)
    for md in src.glob("leaderboard*.md"):
        shutil.copy(md, OUT / md.name)
    return doc


def main() -> None:
    t0 = time.time()
    cfg = SetuConfig.load(ROOT / "configs" / "synthetic.yaml")

    print("illumination demonstration ...", flush=True)
    illum = illumination_demo()
    print(f"  NCC at opposite azimuth: {illum['ncc_opposite_azimuth']}", flush=True)

    scenes = []
    for scene in SCENES:
        print(f"registering scene: {scene['key']} ...", flush=True)
        try:
            scenes.append(run_scene(scene, cfg))
            m = scenes[-1]["metrics"]
            print(f"  {m['n_inliers']} inliers, RMSE vs truth {m.get('rmse_vs_truth_px')} px", flush=True)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("copying evaluation results ...", flush=True)
    ev = copy_eval()
    print(f"  {'ok' if ev else 'not found - run experiments/run_sweeps.py first'}", flush=True)

    dump_json(OUT / "demo.json", {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "illumination": illum,
        "scenes": scenes,
    })

    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
