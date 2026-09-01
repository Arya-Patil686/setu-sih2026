"""The multi-modal sweep: an IIRS-like thermal source against a visible reference.

"Multi-modal" is the first thing the problem statement asks for, and it is the hardest of
the three. IIRS beyond about 2.5 um is measuring surface temperature rather than reflected
sunlight, so the scene is not a differently-lit version of the reference. It is a different
physical quantity, and warm slopes do not sit where bright slopes do.

The source here is rendered visible-band imagery pushed through `thermal_like`, which
inverts the shading term and adds a thermal-inertia component, then degraded through the
IIRS noise and column-striping model. The reference stays visible. Ground truth is exact
either way, so the difficulty is real but the measurement is not guesswork.

Scale is set to 4x, which is the ratio the reference policy actually produces for IIRS:
80 m against a ~20 m Kaguya TC orthomosaic. Matching 80 m against 0.5 m NAC is not
attempted, by design.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setu.bench.generate import make_pair
from setu.bench.terrain import synthetic_terrain
from setu.config import SetuConfig
from setu.eval.leaderboard import render_leaderboard, write_leaderboard
from setu.eval.plots import sweep_plot
from setu.eval.runner import aggregate, run_suite
from setu.types import IlluminationState

OUT = Path(__file__).resolve().parent.parent / "runs" / "eval_multimodal"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["sift", "orb", "rift", "cfog", "loftr", "setu_no_reillum", "setu_full"]
AZ_OFFSETS = [0.0, 45.0, 90.0, 135.0, 180.0]
TERRAINS = ["highland", "mare"]
REF_AZ, ELEV = 135.0, 25.0


def build_pairs(cross_modal: bool, tag: str):
    """Pairs for one modality condition.

    `cross_modal=False` is IIRS as the pipeline actually handles it: the pseudo-panchromatic
    band synthesised from the 900 to 1600 nm reflected-solar window, so the scene is still
    morphologically a panchromatic image, carried across a real sensor gap - IIRS at an SNR
    of 35 with column striping, against a framing camera at 95.

    `cross_modal=True` is the stress case *beyond* what the design targets: the long-wave
    thermal regime, where the source is measuring temperature rather than reflected light.
    The specification is explicit that correcting those bands is not required for
    registration, and the pseudo-pan synthesis exists precisely to avoid them. It is run
    anyway, because knowing where a method stops working is worth as much as knowing where
    it does.
    """
    pairs = []
    for terrain in TERRAINS:
        patch = synthetic_terrain(1536, 5.0, terrain, seed=26166)
        for i, off in enumerate(AZ_OFFSETS):
            pairs.append(make_pair(
                patch,
                illum_src=IlluminationState(sun_az_deg=REF_AZ + off, sun_elev_deg=ELEV,
                                            source="synthetic"),
                illum_ref=IlluminationState(sun_az_deg=REF_AZ, sun_elev_deg=ELEV,
                                            source="synthetic"),
                scale_ratio=4.0, tile_px=512, min_src_px=256,
                cross_modal=cross_modal, src_sensor="IIRS", ref_sensor="KAGUYA_TC",
                warp_kind="affine", pair_id=f"{terrain}_{tag}_az{off:g}", seed=300 + i,
            ))
    return pairs


def main() -> None:
    cfg = SetuConfig.load("configs/iirs_tc.yaml")
    # The controlled benchmark is co-gridded by construction, so the synthetic pre-alignment
    # path applies; everything else comes from the IIRS experiment file.
    cfg.prealign.raise_on_acceptance_failure = False

    t0 = time.time()
    conditions = [
        ("reflected_solar", False,
         "IIRS pseudo-panchromatic (900-1600 nm) against a visible reference"),
        ("thermal", True,
         "Thermal-regime source against a visible reference (beyond the design target)"),
    ]

    everything = {}
    for tag, cross_modal, label in conditions:
        pairs = build_pairs(cross_modal, tag)
        print(f"\n{label}", flush=True)
        print(f"  {len(pairs)} pairs, source {pairs[0].gsd_src_m:g} m against "
              f"reference {pairs[0].gsd_ref_m:g} m", flush=True)

        results = run_suite(
            pairs, METHODS, cfg,
            progress=lambda m, p, d, t: print(f"  [{d:3d}/{t}] {m:18s} {p}", flush=True),
        )
        summary = aggregate(results)
        everything[tag] = summary

        write_leaderboard(
            OUT / f"leaderboard_{tag}.md", summary, results,
            title=label,
            context={
                "pairs": len(pairs),
                "scale ratio": "4x (80 m IIRS against ~20 m Kaguya TC)",
                "sun azimuth difference": "0 to 180 deg at 25 deg elevation",
                "sensor gap": "IIRS SNR 35 with column striping, against a framing camera at SNR 95",
            },
        )
        sweep_plot(results, "d_sun_az", "rmse_inliers_px",
                   "Difference in solar azimuth (degrees)",
                   "Tie-point RMSE against exact truth (px)", label,
                   OUT / f"rmse_{tag}.png", log_y=True, hline=1.0, hline_label="1 px")

        print(render_leaderboard(summary, label, with_ci=False), flush=True)

    (OUT / "metrics.json").write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methods": METHODS,
        "conditions": {t: lbl for t, _, lbl in conditions},
        "summary": everything,
    }, indent=2, default=str))

    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
