"""Run the two headline sweeps of Section 6.2 and write the leaderboard, plots and metrics.

Both sweeps vary exactly one thing. The illumination sweep holds the geometry, the
terrain and the warp fixed and moves only the source sun elevation; the scale sweep holds
the illumination fixed and moves only the ground sampling ratio. Anything that changes in
the result is therefore attributable to the variable named on the x-axis.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from setu.bench.generate import make_pair, scale_sweep
from setu.bench.terrain import synthetic_terrain
from setu.config import SetuConfig
from setu.eval.leaderboard import render_ablation_table, render_leaderboard, write_leaderboard
from setu.eval.plots import inliers_vs_sun_elevation, rmse_vs_scale, rmse_vs_sun_elevation
from setu.eval.runner import aggregate, run_suite
from setu.types import IlluminationState

OUT = Path(__file__).resolve().parent.parent / "runs" / "eval_full"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "sift", "orb", "intfeat", "rift", "cfog",
    "disk_lightglue", "loftr",
    "setu_no_reillum", "setu_no_gate", "setu_no_refine", "setu_no_uniform", "setu_full",
]

# The reference sits at a solar elevation that gives it usable shading in every
# experiment. Putting it at grazing or overhead sun would confound "how different are
# the two illuminations" with "does the reference have any texture at all".
REF_ELEV = 45.0
REF_AZ = 135.0
SRC_ELEVS = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0]
#: Azimuth offsets applied to the source at a fixed, favourable elevation. This is the
#: cleaner test of novelty N1: shadow *length* is identical on both sides, only shadow
#: *direction* changes, so nothing varies except the illumination difference itself.
SRC_AZ_OFFSETS = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
AZ_ELEV = 25.0
TERRAINS = ["highland", "mare"]
#: The azimuth sweep additionally runs on a deliberately self-similar crater field. That
#: is the terrain where a detector-free network produces confident wrong matches, so it is
#: the condition under which the agreement gate has anything to do. On easy terrain every
#: variant scores 100% precision and the ablation says nothing.
AZ_TERRAINS = ["highland", "mare", "selfsimilar"]
SCALES = [1, 2, 4, 8, 16]


def build_azimuth_pairs():
    """Azimuth sweep: solar elevation fixed, only the sun's direction moves.

    The headline illumination experiment. At a fixed 25 degrees of solar elevation both
    images carry the same amount of shadow; sweeping the azimuth from 0 to 180 degrees
    rotates that shadow until, at 180, every crater's lit and shaded sides have swapped.
    Nothing else in the pair changes.
    """
    pairs = []
    for terrain in AZ_TERRAINS:
        patch = synthetic_terrain(1536, 5.0, terrain, seed=26166)
        for i, offset in enumerate(SRC_AZ_OFFSETS):
            pairs.append(make_pair(
                patch,
                illum_src=IlluminationState(sun_az_deg=REF_AZ + offset, sun_elev_deg=AZ_ELEV,
                                            source="synthetic"),
                illum_ref=IlluminationState(sun_az_deg=REF_AZ, sun_elev_deg=AZ_ELEV,
                                            source="synthetic"),
                scale_ratio=1.0, tile_px=512, warp_kind="affine",
                pair_id=f"{terrain}_az{offset:g}", seed=200 + i,
            ))
    return pairs


def build_sun_pairs():
    """Elevation sweep: only the source sun elevation moves.

    Note the confound, which is inherent rather than an oversight: the difference in
    solar elevation cannot be varied without varying one of the two absolute elevations,
    and absolute elevation controls how much shadow there is to match on. Holding the
    reference at 45 degrees keeps that confound on the source side only, and the azimuth
    sweep above is the experiment that isolates illumination difference cleanly.
    """
    pairs = []
    for terrain in TERRAINS:
        patch = synthetic_terrain(1536, 5.0, terrain, seed=26166)
        for i, elev in enumerate(SRC_ELEVS):
            pairs.append(make_pair(
                patch,
                illum_src=IlluminationState(sun_az_deg=REF_AZ, sun_elev_deg=elev, source="synthetic"),
                illum_ref=IlluminationState(sun_az_deg=REF_AZ, sun_elev_deg=REF_ELEV, source="synthetic"),
                scale_ratio=1.0, tile_px=512, warp_kind="affine",
                pair_id=f"{terrain}_sun{elev:g}", seed=100 + i,
            ))
    return pairs


def build_scale_pairs():
    """Scale sweep: only the ground sampling ratio moves.

    The source is held at 256 px across the whole sweep so that a poor result at a large
    ratio cannot be blamed on there being too few source pixels left to match. That means
    the reference grows with the ratio, which is why this needs the 4096 px terrain.
    """
    patch = synthetic_terrain(4096, 5.0, "highland", seed=26166)
    return list(scale_sweep(patch, ratios=SCALES, sun_az=135.0, src_elev=30.0,
                            ref_elev=45.0, tile_px=512, min_src_px=256, warp_kind="affine"))


def main() -> None:
    cfg = SetuConfig.load("configs/synthetic.yaml")
    t0 = time.time()

    def progress(method, pair_id, done, total):
        print(f"  [{done:4d}/{total}] {method:18s} {pair_id}", flush=True)

    print("Building azimuth sweep pairs ...", flush=True)
    az_pairs = build_azimuth_pairs()
    print(f"  {len(az_pairs)} pairs, delta sun azimuth "
          f"{min(p.d_sun_az for p in az_pairs):.0f} to {max(p.d_sun_az for p in az_pairs):.0f} deg", flush=True)
    print("Running azimuth sweep ...", flush=True)
    az_results = run_suite(az_pairs, METHODS, cfg, progress=progress)

    print("Building elevation sweep pairs ...", flush=True)
    sun_pairs = build_sun_pairs()
    print(f"  {len(sun_pairs)} pairs, delta sun elevation "
          f"{min(p.d_sun_elev for p in sun_pairs):.0f} to {max(p.d_sun_elev for p in sun_pairs):.0f} deg", flush=True)
    print("Running elevation sweep ...", flush=True)
    sun_results = run_suite(sun_pairs, METHODS, cfg, progress=progress)

    print("Building scale sweep pairs ...", flush=True)
    scale_pairs = build_scale_pairs()
    print("Running scale sweep ...", flush=True)
    scale_results = run_suite(scale_pairs, METHODS, cfg, progress=progress)

    all_results = list(az_results) + list(sun_results) + list(scale_results)
    summary_all = aggregate(all_results)
    summary_az = aggregate(az_results)
    summary_sun = aggregate(sun_results)
    summary_scale = aggregate(scale_results)

    write_leaderboard(
        OUT / "leaderboard.md", summary_all, all_results,
        title="SETU evaluation - controlled benchmark with exact ground truth",
        context={
            "pairs": len(az_pairs) + len(sun_pairs) + len(scale_pairs),
            "methods": len(METHODS),
            "azimuth sweep": f"delta sun azimuth 0 to 180 deg at {AZ_ELEV:g} deg elevation",
            "elevation sweep": f"source sun elevation {SRC_ELEVS[0]:g} to {SRC_ELEVS[-1]:g} deg against a {REF_ELEV:g} deg reference",
            "scale sweep": f"GSD ratio {SCALES[0]}x to {SCALES[-1]}x",
            "terrain": ", ".join(TERRAINS),
            "ground truth": "exact - both images rendered from one DEM under a known warp",
        },
    )
    (OUT / "leaderboard_azimuth.md").write_text(
        render_leaderboard(summary_az, "Azimuth sweep (delta sun azimuth 0-180 deg, elevation fixed)")
        + "\n\n" + render_ablation_table(summary_az)
    )
    (OUT / "leaderboard_illumination.md").write_text(
        render_leaderboard(summary_sun, "Elevation sweep (source 10-75 deg against a 45 deg reference)")
        + "\n\n" + render_ablation_table(summary_sun)
    )
    (OUT / "leaderboard_scale.md").write_text(
        render_leaderboard(summary_scale, f"Scale sweep (GSD ratio {SCALES[0]}x-{SCALES[-1]}x)")
    )

    from setu.eval.plots import sweep_plot

    rmse_vs_sun_elevation(sun_results, OUT / "rmse_vs_sun_elevation.png")
    inliers_vs_sun_elevation(sun_results, OUT / "inliers_vs_sun_elevation.png")
    rmse_vs_scale(scale_results, OUT / "rmse_vs_scale.png")
    sweep_plot(az_results, "d_sun_az", "rmse_inliers_px",
               "Difference in solar azimuth between source and reference (degrees)",
               "Tie-point RMSE against exact truth (px)",
               "Accuracy against illumination difference (elevation fixed at 25 deg)",
               OUT / "rmse_vs_sun_azimuth.png", log_y=True, hline=1.0, hline_label="1 px")
    sweep_plot(az_results, "d_sun_az", "inlier_ratio",
               "Difference in solar azimuth (degrees)", "Inlier ratio",
               "Inlier ratio against illumination difference",
               OUT / "inliers_vs_sun_azimuth.png")

    (OUT / "metrics.json").write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - t0, 1),
        "methods": METHODS,
        "n_pairs": len(az_pairs) + len(sun_pairs) + len(scale_pairs),
        "config": cfg.resolved(),
        "summary_all": summary_all,
        "summary_azimuth": summary_az,
        "summary_illumination": summary_sun,
        "summary_scale": summary_scale,
    }, indent=2, default=str))

    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT}", flush=True)
    print(render_leaderboard(summary_all, "All methods, all pairs", with_ci=False), flush=True)


if __name__ == "__main__":
    main()
