from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    size: int = 224
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.0
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    hsv_lower: tuple[int, int, int] = (25, 40, 40)
    hsv_upper: tuple[int, int, int] = (90, 255, 255)


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


def preprocess_bgr_to_rgb(image_bgr: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """
    OpenCV preprocessing: resize, light blur, HSV leaf mask, morphology,
    masked color image, CLAHE on L channel (LAB), return RGB uint8 HxWx3.
    """
    cfg = cfg or PreprocessConfig()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR image")

    work = image_bgr.astype(np.uint8, copy=False)
    h, w = work.shape[:2]
    scale = cfg.size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(work, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)

    k = _odd_ksize(cfg.gaussian_ksize)
    blurred = cv2.GaussianBlur(resized, (k, k), cfg.gaussian_sigma)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    segmented = cv2.bitwise_and(blurred, blurred, mask=mask)
    lab = cv2.cvtColor(segmented, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip), tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb.astype(np.uint8)
