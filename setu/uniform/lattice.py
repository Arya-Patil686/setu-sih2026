"""S6 - the lattice over the true overlap polygon.

The lattice is laid over the *intersection of the two validity masks*, not over the
bounding boxes. That distinction matters: two strips crossing at an angle have a
bounding-box intersection far larger than their real common ground, and a coverage
statistic computed on the bounding box would count empty cells that contain no imagery
at all - flattering the number in exactly the wrong direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from setu.types import TiePoint


@dataclass
class Lattice:
    """An m x n grid over the overlap, with per-cell validity."""

    m: int
    n: int
    shape: tuple[int, int]
    valid_cells: np.ndarray           # (m*n,) bool - cell intersects the overlap
    overlap_area_px: float
    cell_area_px: float

    @property
    def n_cells(self) -> int:
        return self.m * self.n

    @property
    def n_valid(self) -> int:
        return int(self.valid_cells.sum())

    def cell_of(self, x: float, y: float) -> int:
        h, w = self.shape
        col = int(np.clip(x / max(w, 1e-9) * self.n, 0, self.n - 1))
        row = int(np.clip(y / max(h, 1e-9) * self.m, 0, self.m - 1))
        return row * self.n + col

    def assign(self, tiepoints: Sequence[TiePoint]) -> None:
        for t in tiepoints:
            t.cell_id = self.cell_of(t.src_sample, t.src_line)

    def occupancy(self, tiepoints: Sequence[TiePoint], inliers_only: bool = True) -> np.ndarray:
        occ = np.zeros(self.n_cells, dtype=int)
        for t in tiepoints:
            if inliers_only and not t.inlier:
                continue
            cid = t.cell_id if t.cell_id >= 0 else self.cell_of(t.src_sample, t.src_line)
            occ[cid] += 1
        return occ

    def empty_cells(self, tiepoints: Sequence[TiePoint]) -> list[int]:
        """Valid cells holding no inlier - the re-seeding targets."""
        occ = self.occupancy(tiepoints)
        return [c for c in range(self.n_cells) if self.valid_cells[c] and occ[c] == 0]

    def cell_bounds(self, cell_id: int) -> tuple[int, int, int, int]:
        """(row0, col0, row1, col1) of a cell in image coordinates."""
        h, w = self.shape
        r, c = divmod(cell_id, self.n)
        return (
            int(r * h / self.m), int(c * w / self.n),
            int((r + 1) * h / self.m), int((c + 1) * w / self.n),
        )

    def coverage_ratio(self, tiepoints: Sequence[TiePoint]) -> float:
        """Fraction of *valid* cells holding at least one inlier."""
        if self.n_valid == 0:
            return 0.0
        occ = self.occupancy(tiepoints)
        return float(((occ > 0) & self.valid_cells).sum() / self.n_valid)


def auto_lattice_size(target_points: int, minimum: int = 4, maximum: int = 16) -> tuple[int, int]:
    """Cells per side of roughly sqrt(target_points / 4), so each cell can hold ~4 points."""
    side = int(np.clip(round(np.sqrt(max(target_points, 1) / 4.0)), minimum, maximum))
    return side, side


def overlap_mask(mask_src: np.ndarray, mask_ref_warped: np.ndarray) -> np.ndarray:
    """True overlap: the intersection of the two validity masks."""
    return np.asarray(mask_src, dtype=bool) & np.asarray(mask_ref_warped, dtype=bool)


def build_lattice(
    shape: tuple[int, int],
    lattice: tuple[int, int] = (8, 8),
    overlap: np.ndarray | None = None,
    min_valid_fraction: float = 0.25,
) -> Lattice:
    """Lay the lattice and mark which cells actually contain overlapping imagery.

    A cell counts as valid only when at least `min_valid_fraction` of it is inside the
    overlap. A cell clipped to a sliver by the strip edge cannot reasonably be expected
    to yield a tie point, and holding it against the coverage score would be a
    self-inflicted penalty.
    """
    m, n = lattice
    h, w = shape[:2]
    valid = np.ones(m * n, dtype=bool)
    overlap_area = float(h * w)

    if overlap is not None:
        ov = np.asarray(overlap, dtype=bool)
        overlap_area = float(ov.sum())
        for cid in range(m * n):
            r0, c0, r1, c1 = Lattice(m, n, (h, w), valid, overlap_area, 0.0).cell_bounds(cid)
            cell = ov[r0:r1, c0:c1]
            valid[cid] = bool(cell.size and cell.mean() >= min_valid_fraction)

    return Lattice(
        m=m, n=n, shape=(h, w), valid_cells=valid,
        overlap_area_px=overlap_area, cell_area_px=float(h * w) / (m * n),
    )


def overlap_polygon(mask: np.ndarray, simplify_px: float = 2.0) -> Polygon:
    """Vector outline of the overlap mask, for the GeoJSON output and the report."""
    import cv2

    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [
        Polygon(c.reshape(-1, 2)).buffer(0)
        for c in contours
        if len(c) >= 4 and cv2.contourArea(c) > 16
    ]
    if not polys:
        h, w = mask.shape[:2]
        return box(0, 0, w, h)
    return unary_union(polys).simplify(simplify_px)
