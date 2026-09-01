"""S3 track B - the handcrafted structural matcher.

Phase-congruency detection, maximum-index-map description and template refinement, in
the RIFT lineage. This track exists because a pretrained network, however good, has
never seen a body with no atmosphere, negligible albedo variation and self-similar
cratered texture, and will produce confident wrong matches on a repetitive crater field.
Track B fails differently, which is what makes the agreement gate of S3 worth anything.

Three choices here are deliberate and are what make the track usable rather than
merely present:

  * Detection is per lattice cell from the outset. Detecting globally and subsampling
    afterwards gives a clustered point set no amount of post-hoc filtering recovers,
    and uniformity is an explicit requirement of the problem statement.
  * Orientation is handled by trying all six cyclic shifts of the index map rather than
    by estimating a dominant orientation, which on cross-modal data is unreliable
    precisely where it matters most.
  * Feature description only gets the neighbourhood. Template matching gets the pixel.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from setu.illum.structural import cfog, mim_response, phase_congruency
from setu.match.base import Matcher, MatchSet

EPS = 1e-10


# ------------------------------------------------------------------- detect

def detect_pc_keypoints(
    pc_map: np.ndarray,
    lattice: tuple[int, int] = (12, 12),
    per_cell: int = 12,
    border: int = 48,
    min_pc: float = 0.05,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Top-q phase-congruency maxima per lattice cell, corner-like and edge-like.

    Both kinds are kept, as in RIFT. Corners are more distinctive but on a cratered
    surface they are scarce and cluster on a few fresh craters; edge maxima follow rims
    and give the spatial spread that the uniformity objective needs.
    """
    pc = np.asarray(pc_map, dtype=np.float32)
    h, w = pc.shape
    m, n = lattice

    # Corner response on the PC map rather than on the image: a corner in phase
    # congruency is a structural corner, and survives the contrast inversion that
    # destroys an intensity-based Harris response.
    harris = cv2.cornerHarris(pc, blockSize=5, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    # Edge-like maxima: local peaks of PC itself.
    local_max = cv2.dilate(pc, np.ones((5, 5), np.uint8))
    is_peak = (pc >= local_max - 1e-6) & (pc > min_pc)

    valid = np.zeros((h, w), dtype=bool)
    valid[border:h - border, border:w - border] = True
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return np.zeros((0, 2), dtype=np.float64)

    pts: list[tuple[float, float]] = []
    row_edges = np.linspace(0, h, m + 1).astype(int)
    col_edges = np.linspace(0, w, n + 1).astype(int)

    for i in range(m):
        for j in range(n):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]
            cell_valid = valid[r0:r1, c0:c1]
            if not cell_valid.any():
                continue

            # Half the quota to each detector, so neither can crowd the other out.
            n_corner = max(1, per_cell // 2)
            for score, count in ((harris[r0:r1, c0:c1], n_corner),
                                 (np.where(is_peak[r0:r1, c0:c1], pc[r0:r1, c0:c1], -np.inf), per_cell - n_corner)):
                s = np.where(cell_valid, score, -np.inf)
                flat = s.ravel()
                k = min(count, int(np.isfinite(flat).sum()))
                if k <= 0:
                    continue
                idx = np.argpartition(flat, -k)[-k:]
                for t in idx:
                    if not np.isfinite(flat[t]):
                        continue
                    rr, cc = divmod(int(t), c1 - c0)
                    pts.append((float(c0 + cc), float(r0 + rr)))

    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return _suppress_duplicates(np.asarray(pts, dtype=np.float64), radius=3.0)


def _suppress_duplicates(pts: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Drop points closer together than `radius`, keeping the first of each group."""
    if len(pts) < 2:
        return pts
    tree = cKDTree(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(pts[i], radius):
            if j > i:
                keep[j] = False
    return pts[keep]


# ----------------------------------------------------------------- describe

def mim_descriptors(
    mim_idx: np.ndarray,
    keypoints: np.ndarray,
    patch: int = 96,
    grid: int = 6,
    n_orient: int = 6,
    all_shifts: bool = False,
) -> np.ndarray:
    """RIFT-style descriptors from the maximum index map.

    The patch is divided into grid x grid subregions and each contributes a histogram
    over the n_orient index values, giving a grid*grid*n_orient vector.

    With `all_shifts`, every cyclic rotation of the orientation axis is returned as a
    separate descriptor. Rotating the image by one filter orientation permutes the index
    values cyclically, so comparing against all shifts is exactly equivalent to trying
    every candidate rotation - without ever estimating one.
    """
    idx = np.asarray(mim_idx, dtype=np.int32)
    h, w = idx.shape
    half = patch // 2
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    n_kp = len(kp)
    dim = grid * grid * n_orient

    if n_kp == 0:
        return np.zeros((0, n_orient, dim) if all_shifts else (0, dim), dtype=np.float32)

    out = np.zeros((n_kp, n_orient, dim) if all_shifts else (n_kp, dim), dtype=np.float32)
    sub = patch // grid
    onehot = np.eye(n_orient, dtype=np.float32)

    for i, (x, y) in enumerate(kp):
        c0, r0 = int(round(x)) - half, int(round(y)) - half
        c1, r1 = c0 + patch, r0 + patch
        if c0 < 0 or r0 < 0 or c1 > w or r1 > h:
            continue

        block = idx[r0:r1, c0:c1]
        # (grid, sub, grid, sub) -> per-subregion histograms in one reduction.
        hist = onehot[block].reshape(grid, sub, grid, sub, n_orient).sum(axis=(1, 3))
        base = hist.reshape(grid * grid, n_orient)

        if all_shifts:
            for s in range(n_orient):
                v = np.roll(base, s, axis=1).ravel()
                out[i, s] = v / (np.linalg.norm(v) + EPS)
        else:
            v = base.ravel()
            out[i] = v / (np.linalg.norm(v) + EPS)

    return out


# -------------------------------------------------------------------- match

def match_descriptors(
    desc_src: np.ndarray,
    desc_ref: np.ndarray,
    lowe_ratio: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour match with a Lowe ratio test, over all orientation shifts.

    The ratio is looser than the usual 0.75 because cross-modal descriptors are less
    discriminative; at 0.75 the test rejects most of the true matches along with the
    false ones. The template refinement that follows is what recovers the precision the
    looser ratio gives away.
    """
    if len(desc_src) == 0 or len(desc_ref) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)

    # Descriptors are L2-normalised, so a dot product is the cosine similarity and the
    # whole match reduces to one matrix multiply.
    if desc_src.ndim == 3:
        n_src, n_shift, dim = desc_src.shape
        sim = (desc_src.reshape(-1, dim) @ desc_ref.T).reshape(n_src, n_shift, -1)
        sim = sim.max(axis=1)                      # best over orientation shifts
    else:
        sim = desc_src @ desc_ref.T

    if sim.shape[1] < 2:
        best = sim.argmax(axis=1)
        return np.arange(len(sim)), best, sim.max(axis=1)

    order = np.argpartition(-sim, 1, axis=1)[:, :2]
    top = np.take_along_axis(sim, order, axis=1)
    swap = top[:, 0] < top[:, 1]
    order[swap] = order[swap][:, ::-1]
    top[swap] = top[swap][:, ::-1]

    # Cosine similarity: convert to a distance so the ratio has its usual meaning.
    d1 = 1.0 - top[:, 0]
    d2 = 1.0 - top[:, 1]
    good = d1 < lowe_ratio * np.maximum(d2, EPS)

    src_idx = np.nonzero(good)[0]
    return src_idx, order[good, 0], top[good, 0]


# ------------------------------------------------------------------- refine

def _skip(refined_ref: list, peaks: list, sharpness: list, xr: float, yr: float) -> None:
    """Record a candidate that could not be refined, keeping the three lists aligned."""
    refined_ref.append((xr, yr))
    peaks.append(0.0)
    sharpness.append(0.0)


def template_refine(
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    window: int = 48,
    patch: int = 32,
    metric: str = "cfog",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine each candidate by searching a window around the descriptor's guess.

    This is the step that makes track B usable. Feature description gets you into the
    right neighbourhood; a dense similarity search over a bounded window gets you the
    pixel. The window is bounded by S1's residual estimate, so the search is small and
    a repeated crater elsewhere in the scene is never a candidate.
    """
    src = np.asarray(src_repr, dtype=np.float32)
    ref = np.asarray(ref_repr, dtype=np.float32)

    if metric == "cfog":
        src_t = cfog(src) if src.ndim == 2 else src
        ref_t = cfog(ref) if ref.ndim == 2 else ref
    else:
        src_t = src[:, :, None] if src.ndim == 2 else src
        ref_t = ref[:, :, None] if ref.ndim == 2 else ref

    half_p, half_w = patch // 2, window // 2
    refined_ref, peaks, sharpness = [], [], []

    for (xs, ys), (xr, yr) in zip(np.atleast_2d(pts_src), np.atleast_2d(pts_ref)):
        c0, r0 = int(round(xs)) - half_p, int(round(ys)) - half_p
        if c0 < 0 or r0 < 0 or r0 + patch > src_t.shape[0] or c0 + patch > src_t.shape[1]:
            _skip(refined_ref, peaks, sharpness, xr, yr)
            continue
        templ = src_t[r0:r0 + patch, c0:c0 + patch]

        sc0, sr0 = int(round(xr)) - half_p - half_w, int(round(yr)) - half_p - half_w
        sc1, sr1 = sc0 + patch + window, sr0 + patch + window
        if sc0 < 0 or sr0 < 0 or sr1 > ref_t.shape[0] or sc1 > ref_t.shape[1]:
            _skip(refined_ref, peaks, sharpness, xr, yr)
            continue
        search = ref_t[sr0:sr1, sc0:sc1]

        surface = _correlate_tensor(search, templ)
        if surface.size == 0:
            _skip(refined_ref, peaks, sharpness, xr, yr)
            continue

        pk = np.unravel_index(int(np.argmax(surface)), surface.shape)
        dy, dx = _parabolic_peak(surface, pk)
        refined_ref.append((sc0 + half_p + pk[1] + dx, sr0 + half_p + pk[0] + dy))

        peak_val = float(surface[pk])
        peaks.append(peak_val)
        sharpness.append(_peak_sharpness(surface, pk, peak_val))

    return (
        np.asarray(refined_ref, dtype=np.float64).reshape(-1, 2),
        np.asarray(peaks, dtype=np.float64),
        np.asarray(sharpness, dtype=np.float64),
    )


def _correlate_tensor(search: np.ndarray, templ: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of a multi-channel template over a search window.

    Channels are summed, which for the L2-normalised CFOG tensor is the dense
    orientation similarity of the original formulation.
    """
    if search.shape[2] != templ.shape[2]:
        return np.zeros((0, 0), np.float32)
    acc = None
    for ch in range(templ.shape[2]):
        t = np.ascontiguousarray(templ[:, :, ch])
        s = np.ascontiguousarray(search[:, :, ch])
        if t.std() < 1e-8:
            continue
        r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        acc = r if acc is None else acc + r
    return (acc / max(templ.shape[2], 1)) if acc is not None else np.zeros((0, 0), np.float32)


def _parabolic_peak(surface: np.ndarray, pk: tuple[int, int]) -> tuple[float, float]:
    """Sub-pixel peak offset from a 1-D parabola through each axis."""
    r, c = pk
    h, w = surface.shape
    dy = dx = 0.0
    if 0 < r < h - 1:
        a, b, cc = surface[r - 1, c], surface[r, c], surface[r + 1, c]
        denom = a - 2 * b + cc
        if abs(denom) > 1e-12:
            dy = float(np.clip(0.5 * (a - cc) / denom, -1.0, 1.0))
    if 0 < c < w - 1:
        a, b, cc = surface[r, c - 1], surface[r, c], surface[r, c + 1]
        denom = a - 2 * b + cc
        if abs(denom) > 1e-12:
            dx = float(np.clip(0.5 * (a - cc) / denom, -1.0, 1.0))
    return dy, dx


def _peak_sharpness(surface: np.ndarray, pk: tuple[int, int], peak_val: float, exclude: int = 3) -> float:
    """peak / second_peak, with the neighbourhood of the peak excluded.

    This ratio is what the agreement gate's single-track branch tests. A broad, ambiguous
    correlation surface - the signature of a repeated crater - scores near 1 and is
    rejected however high its absolute peak.
    """
    masked = surface.copy()
    r0, r1 = max(0, pk[0] - exclude), min(surface.shape[0], pk[0] + exclude + 1)
    c0, c1 = max(0, pk[1] - exclude), min(surface.shape[1], pk[1] + exclude + 1)
    masked[r0:r1, c0:c1] = -np.inf
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return float("inf")
    second = float(finite.max())
    return float(peak_val / second) if second > 1e-6 else float("inf")


# ------------------------------------------------------------------ matcher

class StructuralMatcher(Matcher):
    """Track B end to end: PC detection, MIM description, template refinement."""

    name = "setu_track_b"

    def __init__(
        self,
        lattice: tuple[int, int] = (12, 12),
        per_cell: int = 12,
        patch: int = 96,
        lowe_ratio: float = 0.85,
        window: int = 48,
        metric: str = "cfog",
        n_orient: int = 6,
        pc_k: float = 2.0,
        pc_noise_adaptive: bool = True,
    ) -> None:
        self.lattice = lattice
        self.per_cell = per_cell
        self.patch = patch
        self.lowe_ratio = lowe_ratio
        self.window = window
        self.metric = metric
        self.n_orient = n_orient
        self.pc_k = pc_k
        self.pc_noise_adaptive = pc_noise_adaptive

    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        lattice = kwargs.get("lattice", self.lattice)
        window = int(kwargs.get("window", self.window))
        mask = kwargs.get("mask")

        pc_src = phase_congruency(src, k=self.pc_k, noise_adaptive=self.pc_noise_adaptive)["pc"]
        pc_ref = phase_congruency(ref, k=self.pc_k, noise_adaptive=self.pc_noise_adaptive)["pc"]
        mim_src, _ = mim_response(src, n_orient=self.n_orient)
        mim_ref, _ = mim_response(ref, n_orient=self.n_orient)

        border = max(self.patch // 2 + 2, window // 2 + self.patch // 4 + 2)
        kp_src = detect_pc_keypoints(pc_src, lattice, self.per_cell, border=border, mask=mask)
        kp_ref = detect_pc_keypoints(pc_ref, lattice, self.per_cell, border=border)
        if len(kp_src) == 0 or len(kp_ref) == 0:
            return MatchSet.empty("B", reason="no phase-congruency keypoints survived detection")

        d_src = mim_descriptors(mim_src, kp_src, self.patch, n_orient=self.n_orient, all_shifts=True)
        d_ref = mim_descriptors(mim_ref, kp_ref, self.patch, n_orient=self.n_orient, all_shifts=False)

        si, ri, sim = match_descriptors(d_src, d_ref, self.lowe_ratio)
        if len(si) == 0:
            return MatchSet.empty("B", n_kp_src=len(kp_src), n_kp_ref=len(kp_ref),
                                  reason="no descriptor match passed the ratio test")

        p_src, p_ref = kp_src[si], kp_ref[ri]
        p_ref_ref, peaks, sharp = template_refine(
            src, ref, p_src, p_ref, window=window, patch=32, metric=self.metric
        )

        keep = peaks > 0.0
        conf = np.clip(sim[keep] * np.clip(peaks[keep], 0.0, 1.0), 0.0, 1.0)

        return MatchSet(
            p_src[keep], p_ref_ref[keep], conf, track="B",
            meta={
                "n_kp_src": int(len(kp_src)), "n_kp_ref": int(len(kp_ref)),
                "n_putative": int(len(si)),
                "peak": peaks[keep], "sharpness": sharp[keep],
                "lattice": list(lattice), "window_px": window,
            },
        )
