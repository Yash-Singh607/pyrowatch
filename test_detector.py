"""
PyroWatch - Unit Tests
=======================
Tests for feature extraction, classification logic, and detector output.

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


from features import extract, to_vector, FEATURE_NAMES
from detector import (
    SmokeFireDetector,
    _status_label,
    _contour_box,
    SMOKE_THRESH,
    FIRE_THRESH,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clear_frame() -> np.ndarray:
    """Solid dark-green frame — represents a clear forest scene."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (10, 40, 10)   # BGR dark green
    return img


@pytest.fixture
def smoke_frame() -> np.ndarray:
    """Gray mid-brightness frame — represents heavy smoke coverage."""
    img = np.full((480, 640, 3), 155, dtype=np.uint8)   # neutral gray
    return img


@pytest.fixture
def fire_frame() -> np.ndarray:
    """Orange-saturated frame — represents fire pixels in HSV range."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (0, 145, 245)   # BGR → orange in HSV
    return img


@pytest.fixture
def detector() -> SmokeFireDetector:
    """Detector using OpenCV heuristic (no model file needed)."""
    return SmokeFireDetector(model_path=None)


# ── Feature extraction tests ──────────────────────────────────────────────────

class TestFeatureExtraction:

    def test_returns_all_keys(self, clear_frame):
        feats = extract(clear_frame)
        for name in FEATURE_NAMES:
            assert name in feats, f"Missing feature key: {name}"

    def test_smoke_ratio_zero_for_clear(self, clear_frame):
        feats = extract(clear_frame)
        assert feats["smoke_ratio"] == pytest.approx(0.0, abs=0.01), \
            "Clear frame should have near-zero smoke ratio"

    def test_smoke_ratio_high_for_smoke(self, smoke_frame):
        feats = extract(smoke_frame)
        assert feats["smoke_ratio"] > 0.5, \
            "Smoke frame should have high smoke ratio"

    def test_fire_ratio_high_for_fire(self, fire_frame):
        feats = extract(fire_frame)
        assert feats["fire_ratio"] > 0.01, \
            "Fire frame should have elevated fire ratio"

    def test_fire_ratio_zero_for_clear(self, clear_frame):
        feats = extract(clear_frame)
        assert feats["fire_ratio"] < 0.01, \
            "Clear frame should have near-zero fire ratio"

    def test_to_vector_shape(self, clear_frame):
        feats = extract(clear_frame)
        vec   = to_vector(feats)
        assert vec.shape == (len(FEATURE_NAMES),), \
            f"Expected shape ({len(FEATURE_NAMES)},), got {vec.shape}"

    def test_to_vector_dtype(self, clear_frame):
        feats = extract(clear_frame)
        vec   = to_vector(feats)
        assert vec.dtype == np.float32

    def test_all_features_finite(self, smoke_frame):
        feats = extract(smoke_frame)
        vec   = to_vector(feats)
        assert np.all(np.isfinite(vec)), "All features must be finite"


# ── Status label tests ────────────────────────────────────────────────────────

class TestStatusLabel:

    def test_fire_label(self):
        assert _status_label(0.0, FIRE_THRESH + 0.1) == "FIRE DETECTED"

    def test_smoke_label(self):
        assert _status_label(SMOKE_THRESH + 0.1, 0.0) == "SMOKE DETECTED"

    def test_possible_label(self):
        assert _status_label(0.18, 0.0) == "SMOKE POSSIBLE"

    def test_clear_label(self):
        assert _status_label(0.0, 0.0) == "ALL CLEAR"

    def test_fire_takes_priority_over_smoke(self):
        # When both are high, fire takes priority (checked first)
        assert _status_label(SMOKE_THRESH + 0.1, FIRE_THRESH + 0.1) == "FIRE DETECTED"


# ── Detector output structure tests ───────────────────────────────────────────

class TestDetectorOutput:

    def test_result_keys_present(self, detector, clear_frame):
        result = detector.detect(clear_frame)
        for key in ("smoke_pct", "fire_pct", "safe_pct",
                    "alert", "status", "detections", "method"):
            assert key in result, f"Missing result key: {key}"

    def test_clear_frame_no_alert(self, detector, clear_frame):
        result = detector.detect(clear_frame)
        assert result["alert"] is False
        assert result["status"] == "ALL CLEAR"

    def test_smoke_frame_triggers_alert(self, detector, smoke_frame):
        result = detector.detect(smoke_frame)
        assert result["alert"] is True
        assert "SMOKE" in result["status"]

    def test_fire_frame_triggers_alert(self, detector, fire_frame):
        result = detector.detect(fire_frame)
        assert result["alert"] is True

    def test_percentages_sum_approximately(self, detector, clear_frame):
        result = detector.detect(clear_frame)
        total  = result["smoke_pct"] + result["fire_pct"] + result["safe_pct"]
        assert 0 <= total <= 200, "Percentage sum out of expected range"

    def test_percentages_non_negative(self, detector, smoke_frame):
        result = detector.detect(smoke_frame)
        assert result["smoke_pct"] >= 0
        assert result["fire_pct"]  >= 0
        assert result["safe_pct"]  >= 0

    def test_method_is_string(self, detector, clear_frame):
        result = detector.detect(clear_frame)
        assert isinstance(result["method"], str)
        assert len(result["method"]) > 0

    def test_detections_is_list(self, detector, smoke_frame):
        result = detector.detect(smoke_frame)
        assert isinstance(result["detections"], list)

    def test_detection_box_fields(self, detector, smoke_frame):
        result = detector.detect(smoke_frame)
        for det in result["detections"]:
            assert "class"      in det
            assert "confidence" in det
            assert "box_px"     in det
            assert "box_pct"    in det
            for corner in ("x1", "y1", "x2", "y2"):
                assert corner in det["box_px"]


# ── Contour box helper tests ──────────────────────────────────────────────────

class TestContourBox:

    def test_empty_mask_returns_none(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        assert _contour_box(mask, 640, 480) is None

    def test_filled_mask_returns_box(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 255
        box  = _contour_box(mask, 640, 480)
        assert box is not None
        assert box["x1"] < box["x2"]
        assert box["y1"] < box["y2"]

    def test_box_coordinates_within_frame(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[50:430, 50:590] = 255
        box  = _contour_box(mask, 640, 480)
        assert box["x1"] >= 0
        assert box["y1"] >= 0
        assert box["x2"] <= 640
        assert box["y2"] <= 480


# ── Annotate output tests ─────────────────────────────────────────────────────

class TestAnnotate:

    def test_annotate_returns_same_shape(self, detector, clear_frame):
        result    = detector.detect(clear_frame)
        annotated = detector.annotate(clear_frame, result)
        assert annotated.shape == clear_frame.shape

    def test_annotate_does_not_modify_original(self, detector, smoke_frame):
        original  = smoke_frame.copy()
        result    = detector.detect(smoke_frame)
        detector.annotate(smoke_frame, result)
        np.testing.assert_array_equal(smoke_frame, original)
