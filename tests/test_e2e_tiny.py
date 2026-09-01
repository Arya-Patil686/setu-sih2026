"""End-to-end run on a tiny synthetic pair, with exact ground truth.

This is the test the CI gate of Section 12 requires: it exercises S0 through S7 and
checks the result against a transform that is known exactly, so it fails if any stage
silently stops contributing.
"""

import numpy as np
import pytest

from setu.config import SetuConfig
from setu.eval.metrics import rmse_vs_truth
from setu.pipeline import Pipeline


@pytest.fixture(scope="module")
def run(tiny_pair):
    cfg = SetuConfig.load("configs/synthetic.yaml")
    cfg.uniform.target_points = 150
    src, ref = tiny_pair.to_products()
    return Pipeline(cfg).run(src, ref, dem=tiny_pair.dem_ref,
                             dem_gsd_m=tiny_pair.gsd_ref_m, synthetic=True)


def test_every_stage_ran(run):
    assert {s["stage"] for s in run.stages} >= {"S0", "S1", "S2a", "S2b", "S3A", "S3B", "S3G", "S4", "S5", "S6"}


def test_reillumination_improves_agreement_with_the_source(run):
    """N1's claim, checked on every run: the rendered reference must correlate better."""
    s2a = next(s for s in run.stages if s["stage"] == "S2a")
    assert s2a["reillumination"] is True
    assert s2a["ncc_source_vs_rendered_reference"] > s2a["ncc_source_vs_real_reference"]


def test_produces_tie_points_with_uncertainty(run):
    inliers = [t for t in run.tiepoints if t.inlier]
    assert len(inliers) >= 20
    assert sum(1 for t in inliers if np.isfinite(t.sigma_rms)) >= 10


def test_recovers_the_known_transform_to_sub_pixel(run, tiny_pair):
    gm = run.transform["global"]
    assert gm is not None
    error = rmse_vs_truth(np.array(gm["matrix"]), tiny_pair.H_true, tiny_pair.source.shape)
    assert error < 1.0, f"recovered transform is {error:.3f} px from truth"


def test_tie_points_are_accurate_against_truth(tiny_pair, run):
    """Per-correspondence accuracy, which is the honest sub-pixel claim."""
    from setu.bench.generate import apply_h

    inliers = [t for t in run.tiepoints if t.inlier and not t.reseeded]
    src = np.array([[t.src_sample, t.src_line] for t in inliers])
    ref = np.array([[t.ref_sample, t.ref_line] for t in inliers])
    err = np.hypot(*(ref - apply_h(tiny_pair.H_true, src)).T)
    assert np.median(err) < 1.5
    assert (err < 3.0).mean() > 0.8


def test_metrics_are_complete_and_finite(run):
    m = run.metrics
    for key in ("n_inliers", "inlier_ratio", "rmse_px", "rmse_m", "uniformity", "gate", "runtime_s"):
        assert key in m, f"missing metric: {key}"
    assert np.isfinite(m["rmse_px"])
    assert 0.0 <= m["inlier_ratio"] <= 1.0


def test_model_rmse_is_labelled_as_a_fit_residual(run):
    """Guards the honesty rule: the fit residual must never be presented as accuracy."""
    assert "note_on_rmse" in run.metrics
    assert "not" in run.metrics["note_on_rmse"].lower()


def test_run_directory_is_written(run, tmp_path):
    from setu.product.writers import write_run

    written = write_run(tmp_path / "run", run, write_products=False)
    for key in ("tiepoints_csv", "tiepoints_geojson", "transform_json", "metrics_json", "report_html"):
        assert key in written
    import pathlib
    assert pathlib.Path(written["report_html"]).stat().st_size > 2000
    header = pathlib.Path(written["tiepoints_csv"]).read_text().splitlines()[0]
    assert header.startswith("id,src_line,src_sample,ref_line,ref_sample")
    assert "sigma_x,sigma_y,sigma_xy" in header
