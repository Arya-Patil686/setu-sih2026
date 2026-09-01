"""Section 7.2 - every baseline the evaluation protocol requires.

All of them run through the same `Matcher` interface and the same robust fit as SETU
itself, so the comparison is of correspondence quality and nothing else. A baseline that
needed special handling in the runner would not be a baseline, it would be a
demonstration.

The ablations matter more than the classical baselines. "SIFT: 4 percent, SETU: 84
percent" teaches a judge only that SIFT is bad on the Moon. "MatchAnything alone: 62
percent, SETU: 84 percent" teaches them what this project actually contributes.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from setu.illum.structural import cfog, cfog_similarity, phase_congruency
from setu.match.base import MatchSet, Matcher, to_uint8


class SIFTBaseline(Matcher):
    """SIFT + FLANN + ratio test. The universal default, and the floor."""

    name = "sift"

    def __init__(self, n_features: int = 4000, ratio: float = 0.75) -> None:
        self.n_features = n_features
        self.ratio = ratio

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        a, b = to_uint8(src), to_uint8(ref)
        sift = cv2.SIFT_create(nfeatures=self.n_features)
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return MatchSet.empty(self.name, matcher=self.name)

        index = dict(algorithm=1, trees=5)
        flann = cv2.FlannBasedMatcher(index, dict(checks=64))
        pairs = flann.knnMatch(da, db, k=2)

        src_pts, ref_pts, conf = [], [], []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                src_pts.append(ka[m.queryIdx].pt)
                ref_pts.append(kb[m.trainIdx].pt)
                conf.append(float(1.0 - m.distance / max(n.distance, 1e-9)))
        return MatchSet(np.array(src_pts).reshape(-1, 2), np.array(ref_pts).reshape(-1, 2),
                        np.array(conf), self.name, {"matcher": self.name})


class ORBBaseline(Matcher):
    """ORB + brute-force Hamming + ratio test. The 'fast' default."""

    name = "orb"

    def __init__(self, n_features: int = 5000, ratio: float = 0.8) -> None:
        self.n_features = n_features
        self.ratio = ratio

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        a, b = to_uint8(src), to_uint8(ref)
        orb = cv2.ORB_create(nfeatures=self.n_features)
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return MatchSet.empty(self.name, matcher=self.name)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = bf.knnMatch(da, db, k=2)
        src_pts, ref_pts, conf = [], [], []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                src_pts.append(ka[m.queryIdx].pt)
                ref_pts.append(kb[m.trainIdx].pt)
                conf.append(float(1.0 - m.distance / 256.0))
        return MatchSet(np.array(src_pts).reshape(-1, 2), np.array(ref_pts).reshape(-1, 2),
                        np.array(conf), self.name, {"matcher": self.name})


class IntFeatBaseline(Matcher):
    """IntFeat, the MoonMetaSync hybrid: SIFT and ORB keypoints under one descriptor.

    Published on real OHRC-to-TMC-2 pairs, which makes it the one lunar-specific
    baseline available. Its own authors report that the hybrid did not beat plain SIFT,
    so it is included as the documented lunar floor rather than as a strong competitor.
    """

    name = "intfeat"

    def __init__(self, n_features: int = 3000, ratio: float = 0.78) -> None:
        self.n_features = n_features
        self.ratio = ratio

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        a, b = to_uint8(src), to_uint8(ref)
        sift = cv2.SIFT_create(nfeatures=self.n_features)
        orb = cv2.ORB_create(nfeatures=self.n_features)

        def detect(img):
            kp = list(sift.detect(img, None)) + list(orb.detect(img, None))
            if not kp:
                return [], None
            # One descriptor over the union of both detectors' keypoints, which is the
            # integrated-feature idea: ORB finds corners SIFT misses at low sun, and a
            # single descriptor keeps them comparable.
            return sift.compute(img, kp)

        ka, da = detect(a)
        kb, db = detect(b)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return MatchSet.empty(self.name, matcher=self.name)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        pairs = flann.knnMatch(da, db, k=2)
        src_pts, ref_pts, conf = [], [], []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                src_pts.append(ka[m.queryIdx].pt)
                ref_pts.append(kb[m.trainIdx].pt)
                conf.append(float(1.0 - m.distance / max(n.distance, 1e-9)))
        return MatchSet(np.array(src_pts).reshape(-1, 2), np.array(ref_pts).reshape(-1, 2),
                        np.array(conf), self.name, {"matcher": self.name})


class RIFTBaseline(Matcher):
    """Phase congruency detection with maximum-index-map description - track B alone.

    This is SETU's own structural track run without re-illumination, without the deep
    track and without the gate, which makes it both a strong classical multi-modal
    baseline and the cleanest possible ablation.
    """

    name = "rift"

    def __init__(self, lattice: tuple[int, int] = (10, 10), per_cell: int = 12) -> None:
        self.lattice = lattice
        self.per_cell = per_cell

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        from setu.match.structural import StructuralMatcher

        ms = StructuralMatcher(lattice=self.lattice, per_cell=self.per_cell,
                               window=kw.get("window", 64)).match(src, ref)
        ms.track = self.name
        ms.meta["matcher"] = self.name
        return ms


class CFOGTemplateBaseline(Matcher):
    """Dense CFOG template matching on a regular grid. The classical area-based method.

    No detector at all: a lattice of points is laid down and each is matched by dense
    correlation. That makes it unusually uniform by construction, which is a useful
    contrast with the detector-based methods in the uniformity columns of the table.
    """

    name = "cfog"

    def __init__(self, grid: int = 16, patch: int = 48, window: int = 64) -> None:
        self.grid = grid
        self.patch = patch
        self.window = window

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        from setu.match.structural import _correlate_tensor, _parabolic_peak, _peak_sharpness

        window = int(kw.get("window", self.window))
        h, w = src.shape[:2]
        src_t, ref_t = cfog(src), cfog(ref)
        half_p, half_w = self.patch // 2, window // 2
        margin = half_p + half_w + 2

        if h < 2 * margin or w < 2 * margin:
            return MatchSet.empty(self.name, matcher=self.name)

        ys = np.linspace(margin, h - margin, self.grid)
        xs = np.linspace(margin, w - margin, self.grid)
        src_pts, ref_pts, conf = [], [], []

        for y in ys:
            for x in xs:
                r0, c0 = int(y) - half_p, int(x) - half_p
                sr0, sc0 = r0 - half_w, c0 - half_w
                if sr0 < 0 or sc0 < 0 or sr0 + self.patch + window > ref_t.shape[0] or sc0 + self.patch + window > ref_t.shape[1]:
                    continue
                surface = _correlate_tensor(
                    ref_t[sr0:sr0 + self.patch + window, sc0:sc0 + self.patch + window],
                    src_t[r0:r0 + self.patch, c0:c0 + self.patch],
                )
                if surface.size == 0:
                    continue
                pk = np.unravel_index(int(np.argmax(surface)), surface.shape)
                dy, dx = _parabolic_peak(surface, pk)
                peak = float(surface[pk])
                if peak <= 0:
                    continue
                src_pts.append((float(x), float(y)))
                ref_pts.append((sc0 + half_p + pk[1] + dx, sr0 + half_p + pk[0] + dy))
                conf.append(min(peak, 1.0))

        return MatchSet(np.array(src_pts).reshape(-1, 2), np.array(ref_pts).reshape(-1, 2),
                        np.array(conf), self.name, {"matcher": self.name})


class PCOnlyBaseline(Matcher):
    """Phase congruency plus plain NCC template matching, the simplest structural method."""

    name = "pc_ncc"

    def match(self, src: np.ndarray, ref: np.ndarray, **kw: Any) -> MatchSet:
        from setu.match.structural import StructuralMatcher

        a = phase_congruency(src)["pc"]
        b = phase_congruency(ref)["pc"]
        ms = StructuralMatcher(metric="ncc", window=kw.get("window", 64)).match(a, b)
        ms.track = self.name
        ms.meta["matcher"] = self.name
        return ms


def build_baseline(name: str, **kw: Any) -> Matcher | None:
    """Construct one baseline by name, including the deep ones."""
    from setu.match.deep import build_matcher as build_deep

    classical = {
        "sift": SIFTBaseline,
        "orb": ORBBaseline,
        "intfeat": IntFeatBaseline,
        "rift": RIFTBaseline,
        "cfog": CFOGTemplateBaseline,
        "pc_ncc": PCOnlyBaseline,
    }
    if name in classical:
        return classical[name]()

    deep_aliases = {
        "superpoint_lightglue": "disk_lightglue",
        "lightglue": "disk_lightglue",
        "matchanything": "matchanything_eloftr",
    }
    return build_deep(deep_aliases.get(name, name), **kw)


#: Baselines that consume raw imagery, in the order they appear in the leaderboard.
CLASSICAL = ["sift", "orb", "intfeat", "rift", "cfog"]
DEEP = ["disk_lightglue", "loftr", "matchanything_eloftr"]

#: The ablation ladder. Each entry names the SETU component it switches off, so that the
#: leaderboard reads as an argument rather than as a list of configurations.
ABLATIONS: dict[str, dict[str, Any]] = {
    "setu_no_reillum": {"illum.reilluminate": False},
    "setu_no_gate": {"match.track_b": False},
    "setu_no_structural": {"illum.structural": "none"},
    "setu_no_uniform": {"uniform.reseed": False, "uniform.per_cell_quota": 10_000},
    "setu_no_refine": {"refine.lsm": False, "refine.upsample_factor": 1},
    "setu_full": {},
}

ABLATION_LABELS = {
    "setu_no_reillum": "SETU without sun-synchronised re-illumination (ablates N1)",
    "setu_no_gate": "SETU without the agreement gate, track A only (ablates N2)",
    "setu_no_structural": "SETU without the structural transform",
    "setu_no_uniform": "SETU without uniformity enforcement (ablates N3)",
    "setu_no_refine": "SETU without sub-pixel refinement (ablates N4)",
    "setu_full": "SETU, complete",
}
