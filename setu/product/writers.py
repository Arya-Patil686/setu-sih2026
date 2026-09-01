"""S7 - the run directory.

Every command writes the same set of artefacts into one directory: the registered
raster, the tie points in two formats, the transform, the metrics, a PDS4-style label and
a self-contained QA report. The tie-point columns are exactly those the specification
lists, in that order, so a downstream consumer can rely on them.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from setu.types import RunResult, TiePoint, _json_default

#: Tie-point columns, in the order Section S7 fixes them.
TIEPOINT_COLUMNS = [
    "id", "src_line", "src_sample", "ref_line", "ref_sample",
    "src_lon", "src_lat", "ref_lon", "ref_lat",
    "conf", "track", "sigma_x", "sigma_y", "sigma_xy",
    "residual_x", "residual_y", "inlier", "reseeded", "cell_id",
]


def _row(t: TiePoint) -> dict[str, Any]:
    def num(v: float, nd: int = 6) -> Any:
        return round(float(v), nd) if v is not None and np.isfinite(v) else ""

    return {
        "id": t.tid,
        "src_line": num(t.src_line, 4), "src_sample": num(t.src_sample, 4),
        "ref_line": num(t.ref_line, 4), "ref_sample": num(t.ref_sample, 4),
        "src_lon": num(t.src_lon), "src_lat": num(t.src_lat),
        "ref_lon": num(t.ref_lon), "ref_lat": num(t.ref_lat),
        "conf": num(t.conf, 4), "track": t.track,
        "sigma_x": num(t.sigma_x, 6), "sigma_y": num(t.sigma_y, 6), "sigma_xy": num(t.sigma_xy, 8),
        "residual_x": num(t.residual_x, 4), "residual_y": num(t.residual_y, 4),
        "inlier": int(bool(t.inlier)), "reseeded": int(bool(t.reseeded)),
        "cell_id": int(t.cell_id),
    }


def write_tiepoints_csv(path: str | Path, tiepoints: Sequence[TiePoint]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIEPOINT_COLUMNS)
        writer.writeheader()
        for t in tiepoints:
            writer.writerow(_row(t))
    return path


def write_tiepoints_geojson(path: str | Path, tiepoints: Sequence[TiePoint], crs_name: str | None = None) -> Path:
    """Tie points as a FeatureCollection, for QGIS.

    Geometry is the *source* selenographic position where one is known, because that is
    the frame the problem statement asks for accuracy in. Points without a geolocation
    fall back to their pixel coordinates, flagged so nobody mistakes one for the other.
    """
    features = []
    for t in tiepoints:
        has_lonlat = np.isfinite(t.src_lon) and np.isfinite(t.src_lat)
        coords = [float(t.src_lon), float(t.src_lat)] if has_lonlat else [float(t.src_sample), float(t.src_line)]
        props = _row(t)
        props["coordinates_are"] = "selenographic_lon_lat" if has_lonlat else "source_pixel_xy"
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": coords}, "properties": props})

    doc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs_name:
        doc["crs"] = {"type": "name", "properties": {"name": crs_name}}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=_json_default))
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def write_blink_gif(path: str | Path, frames: Sequence[np.ndarray], fps: int = 3) -> Path | None:
    """Animated before-and-after comparison.

    Non-technical evaluators respond to this more than to any table: a blink comparison
    makes a half-pixel misregistration visible as a shimmer that a number does not convey.
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(path, imgs, duration=1.0 / max(fps, 1), loop=0)
    return path


def write_png(path: str | Path, image: np.ndarray, max_px: int = 1400) -> Path:
    from setu.product.warp import to_png_bytes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_png_bytes(image, max_px))
    return path


def attach_geolocation(tiepoints: Sequence[TiePoint], grid: Any) -> None:
    """Fill the lon/lat columns from the working grid, in place."""
    from setu.geom.ortho import grid_to_lonlat

    if not tiepoints:
        return
    src = np.array([[t.src_sample, t.src_line] for t in tiepoints])
    ref = np.array([[t.ref_sample, t.ref_line] for t in tiepoints])
    try:
        src_ll = grid_to_lonlat(grid, src)
        ref_ll = grid_to_lonlat(grid, ref)
    except Exception:
        return
    for i, t in enumerate(tiepoints):
        t.src_lon, t.src_lat = float(src_ll[i, 0]), float(src_ll[i, 1])
        t.ref_lon, t.ref_lat = float(ref_ll[i, 0]), float(ref_ll[i, 1])


def write_run(
    out_dir: str | Path,
    result: RunResult,
    registered: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    source_ortho: np.ndarray | None = None,
    rendered_reference: np.ndarray | None = None,
    grid: Any | None = None,
    config: Any | None = None,
    write_products: bool = True,
) -> dict[str, str]:
    """Write the complete run directory and return a manifest of what was produced."""
    from setu.product.pds4_label import write_label
    from setu.product.report import write_report
    from setu.product.warp import blink_frames, checkerboard, swipe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if grid is not None:
        attach_geolocation(result.tiepoints, grid)

    written["tiepoints_csv"] = str(write_tiepoints_csv(out / "tiepoints.csv", result.tiepoints))
    written["tiepoints_geojson"] = str(write_tiepoints_geojson(
        out / "tiepoints.geojson", result.tiepoints,
        crs_name=(result.transform.get("grid") or {}).get("crs"),
    ))
    written["transform_json"] = str(write_json(out / "transform.json", result.transform))
    written["metrics_json"] = str(write_json(out / "metrics.json", {
        "run_id": result.run_id, "created_utc": result.created_utc,
        "source": result.source, "reference": result.reference,
        "metrics": result.metrics, "stages": result.stages,
    }))
    written["config_yaml"] = str(_write_config(out / "config.yaml", config, result))

    if write_products and registered is not None:
        written["registered_png"] = str(write_png(out / "registered.png", registered))
        if grid is not None:
            written["registered_tif"] = str(_write_geotiff(out / "registered.tif", registered, grid))

        if reference is not None:
            written["checkerboard_png"] = str(write_png(out / "checkerboard.png",
                                                        checkerboard(registered, reference)))
            written["swipe_png"] = str(write_png(out / "swipe.png", swipe(registered, reference)))
            gif = write_blink_gif(out / "blink.gif", blink_frames(_small(registered), _small(reference)))
            if gif:
                written["blink_gif"] = str(gif)

    if source_ortho is not None:
        written["source_png"] = str(write_png(out / "source_ortho.png", source_ortho))
    if reference is not None:
        written["reference_png"] = str(write_png(out / "reference_ortho.png", reference))
    if rendered_reference is not None:
        written["rendered_reference_png"] = str(write_png(out / "reference_rendered.png", rendered_reference))

    if registered is not None:
        written["pds4_label"] = str(write_label(
            out / "label.xml",
            product_id=result.run_id, raster_path="registered.tif",
            width=int(registered.shape[1]), height=int(registered.shape[0]),
            source=result.source, reference=result.reference,
            metrics=result.metrics, transform=result.transform,
        ))

    written["report_html"] = str(write_report(out / "report.html", result, written, out))
    write_json(out / "manifest.json", written)
    return written


def _small(image: np.ndarray, max_px: int = 700) -> np.ndarray:
    import cv2

    from setu.product.warp import _norm

    img = _norm(image)
    h, w = img.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img


def _write_geotiff(path: Path, array: np.ndarray, grid: Any) -> Path:
    from setu.io.geotiff import write_geotiff

    return write_geotiff(path, array, grid.transform, grid.crs, nodata=0.0, cog=True)


def _write_config(path: Path, config: Any, result: RunResult) -> Path:
    import yaml

    resolved = config.resolved() if config is not None and hasattr(config, "resolved") else result.config
    path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return path
