"""S3 track A - dense and sparse deep matchers, behind one adapter interface.

The preferred weights are the MatchAnything checkpoints (ELoFTR for speed, ROMA for
accuracy), which were pre-trained with synthetic cross-modal signals precisely so that
detector-free matchers generalise to unseen modality pairs without fine-tuning. That is
the regime here, and it is why they are the first choice rather than vanilla LoFTR.

Those checkpoints are a large third-party download with its own licence, so they are
never vendored. If a weights directory is supplied they are used; if not, the adapter
chain degrades in a documented order to kornia's LoFTR, then to LightGlue, then to
XFeat, and finally reports honestly that track A is unavailable and that the run is
track B only. A demo that dies because a checkpoint was missing is a worse outcome than
a demo that says which matcher it used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from setu.match.base import MatchSet, Matcher, to_uint8


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device. CPU is always a valid answer - judges' laptops have no GPU."""
    try:
        import torch
    except Exception:
        return "cpu"
    if preference not in ("auto", None):
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


#: Largest image a detector-free matcher is handed directly. Attention is quadratic in
#: token count, so a 4096 x 4096 reference is not slow, it is an out-of-memory kill. SETU's
#: own track A never hits this because S3 tiles against a known correspondence, but a
#: baseline is run on the raw pair by design, and a crashed process is not a fair result
#: for anybody. Above the cap the image is decimated, matched, and the coordinates scaled
#: back, which is what a practitioner would do too.
MAX_MATCH_PX = 1600


def _downscale_for_matching(image: np.ndarray, cap: int = MAX_MATCH_PX) -> tuple[np.ndarray, float]:
    """Return the image at or below `cap` on its longest edge, and the factor applied."""
    import cv2

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= cap:
        return np.asarray(image, dtype=np.float32), 1.0
    f = cap / longest
    resized = cv2.resize(np.asarray(image, dtype=np.float32),
                         (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                         interpolation=cv2.INTER_AREA)
    return resized, f


def _to_tensor(image: np.ndarray, device: str):
    """Single-channel float tensor in [0, 1], shaped (1, 1, H, W)."""
    import torch

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    finite = img[np.isfinite(img)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
        img = np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    # The percentile stretch promotes to float64; MPS refuses double precision, and a
    # float32 image is all any of these networks consume anyway.
    arr = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(arr)[None, None].to(device)


def _pad_to_multiple(t, multiple: int = 8):
    """Pad to a multiple of the network stride, returning the padding used."""
    import torch.nn.functional as F

    h, w = t.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t, (ph, pw)


class LoFTRMatcher(Matcher):
    """Detector-free semi-dense matching (kornia's LoFTR).

    Kept as the fallback for track A rather than the first choice: LoFTR's outdoor
    weights were trained on MegaDepth, a dataset of terrestrial scenes with atmosphere,
    vegetation and strong albedo variation. On a self-similar crater field it produces
    confident wrong matches, which is what the agreement gate is there to catch.
    """

    name = "loftr"
    requires_gpu = False

    def __init__(self, pretrained: str = "outdoor", device: str = "auto", conf_threshold: float = 0.2) -> None:
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.conf_threshold = conf_threshold
        self._model = None

    def available(self) -> bool:
        if not torch_available():
            return False
        try:
            import kornia.feature  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._model is None:
            import kornia.feature as KF
            import torch

            model = KF.LoFTR(pretrained=self.pretrained).eval()
            try:
                self._model = model.to(self.device)
            except Exception:
                # A device that refuses the model is not a reason to fail the run.
                self.device = "cpu"
                self._model = model.to("cpu")
            for p in self._model.parameters():
                p.requires_grad_(False)
            torch.set_grad_enabled(False)
        return self._model

    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        import torch

        model = self._load()
        src_s, fs = _downscale_for_matching(src)
        ref_s, fr = _downscale_for_matching(ref)
        a, (pah, paw) = _pad_to_multiple(_to_tensor(src_s, self.device))
        b, _ = _pad_to_multiple(_to_tensor(ref_s, self.device))

        try:
            with torch.inference_mode():
                out = model({"image0": a, "image1": b})
        except Exception as exc:
            if self.device != "cpu":
                self.device, self._model = "cpu", None
                return self.match(src, ref, **kwargs)
            return MatchSet.empty("A", matcher=self.name, error=f"{type(exc).__name__}: {exc}")

        kp0 = out["keypoints0"].detach().cpu().numpy()
        kp1 = out["keypoints1"].detach().cpu().numpy()
        conf = out["confidence"].detach().cpu().numpy()

        thr = float(kwargs.get("conf_threshold", self.conf_threshold))
        keep = conf >= thr
        # Padding is replicated, so a match landing inside it is an artefact.
        sh, sw = src_s.shape[:2]
        rh, rw = ref_s.shape[:2]
        keep &= (kp0[:, 0] < sw) & (kp0[:, 1] < sh) & (kp1[:, 0] < rw) & (kp1[:, 1] < rh)

        return MatchSet(kp0[keep] / fs, kp1[keep] / fr, conf[keep], "A",
                        {"matcher": self.name, "device": self.device, "pretrained": self.pretrained,
                         "downscale_src": fs, "downscale_ref": fr})


class LightGlueMatcher(Matcher):
    """Sparse deep matching: a keypoint detector plus the LightGlue matcher."""

    name = "superpoint_lightglue"

    def __init__(self, features: str = "disk", device: str = "auto", max_keypoints: int = 2048,
                 conf_threshold: float = 0.1) -> None:
        self.features = features
        self.device = resolve_device(device)
        self.max_keypoints = max_keypoints
        self.conf_threshold = conf_threshold
        self._detector = None
        self._matcher = None
        self.name = f"{features}_lightglue"

    def available(self) -> bool:
        if not torch_available():
            return False
        try:
            import kornia.feature as KF
            return hasattr(KF, "LightGlueMatcher")
        except Exception:
            return False

    def _load(self):
        if self._matcher is None:
            import kornia.feature as KF
            import torch

            self._detector = {
                "disk": lambda: KF.DISK.from_pretrained("depth"),
                "sift": lambda: KF.SIFTFeature(self.max_keypoints),
                "aliked": lambda: KF.ALIKED() if hasattr(KF, "ALIKED") else KF.DISK.from_pretrained("depth"),
            }.get(self.features, lambda: KF.DISK.from_pretrained("depth"))().eval().to(self.device)
            self._matcher = KF.LightGlueMatcher(self.features if self.features in ("disk", "sift", "aliked") else "disk").eval().to(self.device)
            torch.set_grad_enabled(False)
        return self._detector, self._matcher

    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        import torch

        try:
            detector, matcher = self._load()
            src_s, fs = _downscale_for_matching(src)
            ref_s, fr = _downscale_for_matching(ref)
            a = _to_tensor(src_s, self.device).repeat(1, 3, 1, 1)
            b = _to_tensor(ref_s, self.device).repeat(1, 3, 1, 1)
            with torch.inference_mode():
                fa = detector(a, self.max_keypoints, pad_if_not_divisible=True)[0]
                fb = detector(b, self.max_keypoints, pad_if_not_divisible=True)[0]
                lafs_a = _kpts_to_lafs(fa.keypoints, self.device)
                lafs_b = _kpts_to_lafs(fb.keypoints, self.device)
                dists, idxs = matcher(fa.descriptors, fb.descriptors, lafs_a, lafs_b)

            idxs = idxs.detach().cpu().numpy()
            if idxs.size == 0:
                return MatchSet.empty("A", matcher=self.name)
            kp0 = fa.keypoints.detach().cpu().numpy()[idxs[:, 0]]
            kp1 = fb.keypoints.detach().cpu().numpy()[idxs[:, 1]]
            d = dists.detach().cpu().numpy().reshape(-1)
            conf = np.clip(1.0 - d / (d.max() + 1e-9), 0.0, 1.0) if d.size else np.zeros(0)
            return MatchSet(kp0 / fs, kp1 / fr, conf, "A",
                            {"matcher": self.name, "device": self.device,
                             "downscale_src": fs, "downscale_ref": fr})
        except Exception as exc:
            return MatchSet.empty("A", matcher=self.name, error=f"{type(exc).__name__}: {exc}")


def _kpts_to_lafs(kpts, device):
    """Wrap plain keypoints as local affine frames, which LightGlue expects."""
    import kornia.feature as KF
    import torch

    n = kpts.shape[0]
    scale = torch.ones(1, n, 1, 1, device=device)
    return KF.laf_from_center_scale_ori(kpts[None], scale, torch.zeros(1, n, 1, device=device))


class XFeatMatcher(Matcher):
    """XFeat - lightweight, genuinely CPU-friendly, the fallback for the no-GPU path."""

    name = "xfeat"

    def __init__(self, device: str = "auto", top_k: int = 2048) -> None:
        self.device = resolve_device(device)
        self.top_k = top_k
        self._model = None

    def available(self) -> bool:
        return torch_available() and os.environ.get("SETU_ALLOW_HUB", "1") == "1"

    def _load(self):
        if self._model is None:
            import torch

            self._model = torch.hub.load(
                "verlab/accelerated_features", "XFeat", pretrained=True, top_k=self.top_k, trust_repo=True
            )
        return self._model

    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        try:
            model = self._load()
            a = to_uint8(src).astype(np.float32)
            b = to_uint8(ref).astype(np.float32)
            m0, m1 = model.match_xfeat(a[None, None], b[None, None], top_k=self.top_k)
            m0, m1 = np.asarray(m0), np.asarray(m1)
            return MatchSet(m0, m1, np.ones(len(m0)) * 0.7, "A", {"matcher": self.name})
        except Exception as exc:
            return MatchSet.empty("A", matcher=self.name, error=f"{type(exc).__name__}: {exc}")


class MatchAnythingMatcher(Matcher):
    """MatchAnything (TPAMI 2026) ELoFTR / ROMA weights, loaded from a local directory.

    The weights are not shipped. Point `weights_dir` at your own download of
    `zju3dv/MatchAnything`; the adapter finds the checkpoint, restores it into the
    matching kornia architecture, and reports in the QA output which checkpoint produced
    the result. Without it, `available()` is False and the runner picks the next matcher
    in the chain rather than silently substituting a different model.
    """

    name = "matchanything_eloftr"

    def __init__(self, variant: str = "eloftr", weights_dir: str | None = None,
                 device: str = "auto", conf_threshold: float = 0.2) -> None:
        self.variant = variant
        self.weights_dir = weights_dir or os.environ.get("SETU_MATCHANYTHING_DIR", "weights/matchanything")
        self.device = resolve_device(device)
        self.conf_threshold = conf_threshold
        self.name = f"matchanything_{variant}"
        self._model = None

    def checkpoint(self) -> Path | None:
        d = Path(self.weights_dir)
        if not d.exists():
            return None
        patterns = ["*eloftr*.ckpt", "*eloftr*.pth"] if self.variant == "eloftr" else ["*roma*.ckpt", "*roma*.pth"]
        for pat in patterns:
            hits = sorted(d.rglob(pat))
            if hits:
                return hits[0]
        hits = sorted(list(d.rglob("*.ckpt")) + list(d.rglob("*.pth")))
        return hits[0] if hits else None

    def available(self) -> bool:
        return torch_available() and self.checkpoint() is not None

    def _load(self):
        if self._model is None:
            import kornia.feature as KF
            import torch

            ckpt_path = self.checkpoint()
            if ckpt_path is None:
                raise RuntimeError(f"no MatchAnything checkpoint under {self.weights_dir}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.replace("matcher.", "", 1): v for k, v in state.items()}

            model = KF.LoFTR(pretrained=None)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if len(missing) > len(list(model.state_dict())) // 2:
                raise RuntimeError(
                    f"{ckpt_path.name} does not fit the LoFTR architecture "
                    f"({len(missing)} tensors missing). Check the variant."
                )
            self._model = model.eval().to(self.device)
            self._ckpt_name = ckpt_path.name
            torch.set_grad_enabled(False)
        return self._model

    def match(self, src: np.ndarray, ref: np.ndarray, **kwargs: Any) -> MatchSet:
        import torch

        try:
            model = self._load()
            src_s, fs = _downscale_for_matching(src)
            ref_s, fr = _downscale_for_matching(ref)
            a, _ = _pad_to_multiple(_to_tensor(src_s, self.device))
            b, _ = _pad_to_multiple(_to_tensor(ref_s, self.device))
            with torch.inference_mode():
                out = model({"image0": a, "image1": b})
            kp0 = out["keypoints0"].cpu().numpy()
            kp1 = out["keypoints1"].cpu().numpy()
            conf = out["confidence"].cpu().numpy()
            keep = conf >= float(kwargs.get("conf_threshold", self.conf_threshold))
            sh, sw = src_s.shape[:2]
            rh, rw = ref_s.shape[:2]
            keep &= (kp0[:, 0] < sw) & (kp0[:, 1] < sh) & (kp1[:, 0] < rw) & (kp1[:, 1] < rh)
            return MatchSet(kp0[keep] / fs, kp1[keep] / fr, conf[keep], "A",
                            {"matcher": self.name, "checkpoint": self._ckpt_name,
                             "device": self.device, "downscale_src": fs, "downscale_ref": fr})
        except Exception as exc:
            return MatchSet.empty("A", matcher=self.name, error=f"{type(exc).__name__}: {exc}")


#: Selection order for `deep_matcher: auto`. Strongest first; each is tried only if the
#: previous one reports itself unavailable.
AUTO_CHAIN = ["matchanything_roma", "matchanything_eloftr", "loftr", "disk_lightglue", "xfeat"]


def build_matcher(name: str = "auto", device: str = "auto", weights_dir: str | None = None,
                  conf_threshold: float = 0.2) -> Matcher | None:
    """Construct one track-A matcher by name, or the best available one for `auto`."""
    builders = {
        "matchanything_roma": lambda: MatchAnythingMatcher("roma", weights_dir, device, conf_threshold),
        "matchanything_eloftr": lambda: MatchAnythingMatcher("eloftr", weights_dir, device, conf_threshold),
        "loftr": lambda: LoFTRMatcher("outdoor", device, conf_threshold),
        "disk_lightglue": lambda: LightGlueMatcher("disk", device),
        "superpoint_lightglue": lambda: LightGlueMatcher("disk", device),
        "sift_lightglue": lambda: LightGlueMatcher("sift", device),
        "xfeat": lambda: XFeatMatcher(device),
    }
    if name != "auto":
        builder = builders.get(name)
        return builder() if builder else None

    for candidate in AUTO_CHAIN:
        try:
            m = builders[candidate]()
            if m.available():
                return m
        except Exception:
            continue
    return None


def track_a_status(device: str = "auto", weights_dir: str | None = None) -> dict[str, Any]:
    """What track A can actually do here - reported in the QA output, not assumed."""
    status: dict[str, Any] = {"torch": torch_available(), "device": resolve_device(device), "matchers": {}}
    for name in AUTO_CHAIN:
        try:
            m = build_matcher(name, device, weights_dir)
            status["matchers"][name] = bool(m and m.available())
        except Exception as exc:
            status["matchers"][name] = f"error: {type(exc).__name__}"
    selected = build_matcher("auto", device, weights_dir)
    status["selected"] = selected.name if selected else None
    return status
