"""The matcher interface every track and every baseline satisfies.

One adapter class per method, all returning the same triple, so that the evaluation
harness can run SIFT, LoFTR, MatchAnything and SETU's own tracks through identical code.
A baseline that needed special handling in the runner would not be a fair baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MatchSet:
    """Correspondences from one matcher.

    Coordinates are (x, y) in pixels, `kpts_src` in the source image frame and
    `kpts_ref` in the reference frame. `conf` is whatever confidence the underlying
    method reports, rescaled to [0, 1] by the adapter so the gate can compare tracks.
    """

    kpts_src: np.ndarray            # (N, 2) float64, (x, y)
    kpts_ref: np.ndarray            # (N, 2) float64, (x, y)
    conf: np.ndarray                # (N,) float64 in [0, 1]
    track: str = "A"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kpts_src = np.asarray(self.kpts_src, dtype=np.float64).reshape(-1, 2)
        self.kpts_ref = np.asarray(self.kpts_ref, dtype=np.float64).reshape(-1, 2)
        self.conf = np.asarray(self.conf, dtype=np.float64).reshape(-1)
        if not (len(self.kpts_src) == len(self.kpts_ref) == len(self.conf)):
            raise ValueError(
                f"ragged match set: {len(self.kpts_src)} src, {len(self.kpts_ref)} ref, {len(self.conf)} conf"
            )

    def __len__(self) -> int:
        return len(self.kpts_src)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def top(self, n: int) -> MatchSet:
        """The n highest-confidence correspondences."""
        if len(self) <= n:
            return self
        idx = np.argsort(self.conf)[::-1][:n]
        return MatchSet(self.kpts_src[idx], self.kpts_ref[idx], self.conf[idx], self.track, dict(self.meta))

    def filter(self, mask: np.ndarray) -> MatchSet:
        m = np.asarray(mask, dtype=bool)
        return MatchSet(self.kpts_src[m], self.kpts_ref[m], self.conf[m], self.track, dict(self.meta))

    def offset(self, d_src: tuple[float, float] = (0.0, 0.0), d_ref: tuple[float, float] = (0.0, 0.0)) -> MatchSet:
        """Shift both point sets, used to lift tile-local coordinates back to full-image."""
        return MatchSet(
            self.kpts_src + np.asarray(d_src, dtype=np.float64),
            self.kpts_ref + np.asarray(d_ref, dtype=np.float64),
            self.conf, self.track, dict(self.meta),
        )

    @staticmethod
    def empty(track: str = "A", **meta: Any) -> MatchSet:
        return MatchSet(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), track, meta)

    @staticmethod
    def concat(sets: list[MatchSet], track: str | None = None) -> MatchSet:
        sets = [s for s in sets if not s.is_empty]
        if not sets:
            return MatchSet.empty(track or "A")
        merged: dict[str, Any] = {}
        for s in sets:
            merged.update(s.meta)
        return MatchSet(
            np.vstack([s.kpts_src for s in sets]),
            np.vstack([s.kpts_ref for s in sets]),
            np.concatenate([s.conf for s in sets]),
            track or sets[0].track,
            merged,
        )

    def deduplicate(self, radius_px: float = 2.0) -> MatchSet:
        """Collapse near-duplicate source points, keeping the highest confidence.

        Tiles overlap by half their width, so the same feature is found several times.
        Keeping all copies would inflate the match count and make a clustered
        distribution look denser than it is.
        """
        if len(self) < 2:
            return self
        from scipy.spatial import cKDTree

        order = np.argsort(self.conf)[::-1]
        tree = cKDTree(self.kpts_src[order])
        keep = np.ones(len(order), dtype=bool)
        for i in range(len(order)):
            if not keep[i]:
                continue
            for j in tree.query_ball_point(self.kpts_src[order][i], radius_px):
                if j > i:
                    keep[j] = False
        idx = order[keep]
        return MatchSet(self.kpts_src[idx], self.kpts_ref[idx], self.conf[idx], self.track, dict(self.meta))


class Matcher(ABC):
    """Base class for every correspondence method."""

    name: str = "matcher"
    requires_gpu: bool = False

    @abstractmethod
    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        """Return correspondences between two single-channel float32 images."""

    def available(self) -> bool:
        """Whether this matcher can run in the current environment."""
        return True

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available(), "requires_gpu": self.requires_gpu}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Robust 8-bit view for the OpenCV detectors, which will not take float32.

    Percentile stretching rather than min/max: a single hot pixel or a deep shadow
    would otherwise compress the whole scene into a few grey levels.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi - lo < 1e-12:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
