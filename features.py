"""
PyroWatch - Feature Extraction Module
======================================
Extracts a fixed-length feature vector from a BGR image for use by
both the rule-based classifier and the sklearn RandomForest classifier.

Features extracted
------------------
0  smoke_ratio   : fraction of pixels with low saturation + mid brightness
1  fire_ratio    : fraction of HSV-orange pixels (flame signature)
2  lap_mean      : mean absolute Laplacian (edge sharpness)
3  mean_sat      : mean saturation normalised to [0,1]
4  sat_std       : saturation standard deviation normalised to [0,1]
5  val_mean      : mean brightness normalised to [0,1]
6  val_std       : brightness standard deviation normalised to [0,1]
7  soft_bonus    : 1.25 when Laplacian < 13 (smoke has diffuse edges)
8  smoke_conf    : composite smoke confidence score
9  fire_conf     : composite fire confidence score
"""

import cv2
import numpy as np
from typing import Dict, Tuple

FEATURE_NAMES: Tuple[str, ...] = (
    "smoke_ratio",
    "fire_ratio",
    "lap_mean",
    "mean_sat",
    "sat_std",
    "val_mean",
    "val_std",
    "soft_bonus",
    "smoke_conf",
    "fire_conf",
)

# HSV ranges
_FIRE_LOWER = np.array([0,  50, 100])
_FIRE_UPPER = np.array([30, 255, 255])


def extract(frame: np.ndarray) -> Dict[str, float]:
    """
    Compute all visual features from a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    dict mapping feature name -> float value.
    Also includes 'smoke_mask' and 'fire_mask' as np.ndarray for drawing.
    """
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # ── Smoke mask: low saturation + mid brightness ───────────────────────────
    raw_smoke = ((sat < 70) & (val > 65) & (val < 230)).astype(np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    smoke_mask = cv2.morphologyEx(raw_smoke, cv2.MORPH_CLOSE, kernel)
    smoke_ratio = float(smoke_mask.sum()) / (h * w)

    # ── Fire mask: orange/red HSV range ──────────────────────────────────────
    fire_mask  = cv2.inRange(hsv, _FIRE_LOWER, _FIRE_UPPER)
    fire_ratio = float(fire_mask.sum()) / (255.0 * h * w)

    # ── Edge sharpness (Laplacian) ────────────────────────────────────────────
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = float(np.abs(lap).mean())

    # ── Saturation and brightness statistics ─────────────────────────────────
    mean_sat = float(sat.mean()) / 255.0
    sat_std  = float(sat.std())  / 255.0
    val_mean = float(val.mean()) / 255.0
    val_std  = float(val.std())  / 255.0

    # ── Composite scores ──────────────────────────────────────────────────────
    soft_bonus  = 1.25 if lap_mean < 13 else 1.0
    smoke_conf  = float(min(1.0, smoke_ratio * 3.2 * soft_bonus))
    fire_conf_r = float(min(1.0, fire_ratio * 15.0))
    fire_conf   = float(fire_conf_r * (1.0 - 0.4 * smoke_conf))

    return {
        "smoke_ratio": smoke_ratio,
        "fire_ratio":  fire_ratio,
        "lap_mean":    lap_mean,
        "mean_sat":    mean_sat,
        "sat_std":     sat_std,
        "val_mean":    val_mean,
        "val_std":     val_std,
        "soft_bonus":  soft_bonus,
        "smoke_conf":  smoke_conf,
        "fire_conf":   fire_conf,
        # masks kept for drawing — not part of the numeric vector
        "_smoke_mask": smoke_mask,
        "_fire_mask":  fire_mask,
    }


def to_vector(feats: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dict to a fixed-length numpy array for sklearn.

    Parameters
    ----------
    feats : dict
        Output of extract().

    Returns
    -------
    np.ndarray of shape (10,), dtype float32.
    """
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
