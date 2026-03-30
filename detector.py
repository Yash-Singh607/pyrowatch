"""
PyroWatch - Detection Engine
==============================
Supports two detection backends:
  1. RandomForest classifier (default when model exists)
  2. Multi-feature OpenCV heuristic (fallback, no training needed)

Both backends share the same feature extraction pipeline defined in features.py.

Usage:
    python src/detector.py --source data/sample_images/smoke_001.jpg --save
    python src/detector.py --source data/sample_images/ --save --json
    python src/detector.py --source video.mp4 --save
    python src/detector.py --source 0 --show
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


from features import extract, to_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PyroWatch")

# ── Decision thresholds ───────────────────────────────────────────────────────
SMOKE_THRESH: float = 0.25
FIRE_THRESH:  float = 0.25
BOX_THRESH:   float = 0.12

# Class index mapping used by RandomForest
_CLASS_NAMES = ["Clear", "Fire", "Smoke"]   # alphabetical (sklearn default)


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class SmokeFireDetector:
    """
    Wildfire smoke and fire detector.

    If a trained RandomForest model (models/rf_classifier.pkl) is available,
    it is used as the primary classifier. Otherwise the OpenCV heuristic
    pipeline is used automatically.

    Parameters
    ----------
    model_path : str or None
        Path to rf_classifier.pkl or a YOLOv8 .pt file.
        Pass None to force the OpenCV heuristic.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.use_rf   = False
        self.use_yolo = False
        self._rf_model: Any = None
        self._rf_le:    Any = None

        if model_path and Path(model_path).exists():
            if model_path.endswith(".pkl"):
                self._load_rf(model_path)
            elif model_path.endswith(".pt"):
                self._load_yolo(model_path)
        else:
            log.info("Using OpenCV heuristic classifier.")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_rf(self, path: str) -> None:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._rf_model = bundle["model"]
        self._rf_le    = bundle["label_encoder"]
        self.use_rf    = True
        log.info(f"RandomForest model loaded: {path}")

    def _load_yolo(self, path: str) -> None:
        try:
            from ultralytics import YOLO      # type: ignore
            self._yolo_model = YOLO(path)
            self.use_yolo    = True
            log.info(f"YOLOv8 model loaded: {path}")
        except ImportError:
            log.warning("ultralytics not installed — falling back to OpenCV.")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray  shape (H, W, 3), uint8, BGR

        Returns
        -------
        dict with keys:
            smoke_pct, fire_pct, safe_pct  : float (0–100)
            alert                           : bool
            status                          : str
            detections                      : list[dict]
            method                          : str
        """
        if self.use_yolo:
            return self._detect_yolo(frame)
        if self.use_rf:
            return self._detect_rf(frame)
        return self._detect_opencv(frame)

    def annotate(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw bounding boxes and HUD bar onto a copy of frame.

        Parameters
        ----------
        frame  : original BGR frame
        result : output of detect()

        Returns
        -------
        Annotated BGR frame (copy).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for det in result["detections"]:
            if det["confidence"] < BOX_THRESH:
                continue
            b     = det["box_px"]
            label = det["class"].upper()
            conf  = det["confidence"]
            color = (20, 55, 220) if label == "FIRE" else (200, 165, 0)
            cv2.rectangle(out,
                          (b["x1"], b["y1"]),
                          (b["x2"], b["y2"]),
                          color, 2)
            tag = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(out,
                          (b["x1"], b["y1"] - th - 8),
                          (b["x1"] + tw + 6, b["y1"]),
                          color, -1)
            cv2.putText(out, tag,
                        (b["x1"] + 3, b["y1"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # HUD bar
        alert  = result["alert"]
        hud_c  = (20, 20, 180) if alert else (20, 120, 30)
        cv2.rectangle(out, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.rectangle(out, (0, 0), (w, 58), hud_c, 2)
        cv2.putText(out,
                    f"PYROWATCH  |  {result['status']}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (20, 20, 200) if alert else (20, 200, 60),
                    2, cv2.LINE_AA)
        cv2.putText(out,
                    f"Smoke:{result['smoke_pct']:.0f}%  "
                    f"Fire:{result['fire_pct']:.0f}%  "
                    f"Safe:{result['safe_pct']:.0f}%  "
                    f"[{result['method']}]",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, ts,
                    (w - 185, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (140, 140, 140), 1, cv2.LINE_AA)
        return out

    # ── OpenCV heuristic ──────────────────────────────────────────────────────

    def _detect_opencv(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w  = frame.shape[:2]
        feats = extract(frame)

        smoke_conf = feats["smoke_conf"]
        fire_conf  = feats["fire_conf"]
        safe_conf  = max(0.0, 1.0 - smoke_conf - fire_conf * 0.6)

        alert  = smoke_conf >= SMOKE_THRESH or fire_conf >= FIRE_THRESH
        status = _status_label(smoke_conf, fire_conf)

        detections: List[Dict[str, Any]] = []
        if smoke_conf >= BOX_THRESH:
            box = _contour_box(feats["_smoke_mask"], w, h)
            if box:
                detections.append(_make_det("smoke", smoke_conf, box, w, h))
        if fire_conf >= BOX_THRESH:
            box = _contour_box(feats["_fire_mask"], w, h)
            if box:
                detections.append(_make_det("fire", fire_conf, box, w, h))

        return _build_result(smoke_conf, fire_conf, safe_conf,
                             alert, status, detections, "opencv")

    # ── RandomForest ──────────────────────────────────────────────────────────

    def _detect_rf(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w  = frame.shape[:2]
        feats = extract(frame)
        vec   = to_vector(feats).reshape(1, -1)

        proba      = self._rf_model.predict_proba(vec)[0]
        pred_idx   = int(proba.argmax())
        pred_class = self._rf_le.classes_[pred_idx]

        # Map class probabilities to smoke/fire/safe scores
        classes    = list(self._rf_le.classes_)
        smoke_conf = float(proba[classes.index("Smoke")] if "Smoke" in classes else 0)
        fire_conf  = float(proba[classes.index("Fire")]  if "Fire"  in classes else 0)
        safe_conf  = float(proba[classes.index("Clear")] if "Clear" in classes else 0)

        alert  = pred_class in ("Smoke", "Fire")
        status = (f"FIRE DETECTED"  if pred_class == "Fire"  else
                  f"SMOKE DETECTED" if pred_class == "Smoke" else
                  "ALL CLEAR")

        detections: List[Dict[str, Any]] = []
        if smoke_conf >= BOX_THRESH:
            box = _contour_box(feats["_smoke_mask"], w, h)
            if box:
                detections.append(_make_det("smoke", smoke_conf, box, w, h))
        if fire_conf >= BOX_THRESH:
            box = _contour_box(feats["_fire_mask"], w, h)
            if box:
                detections.append(_make_det("fire", fire_conf, box, w, h))

        return _build_result(smoke_conf, fire_conf, safe_conf,
                             alert, status, detections, "random_forest")

    # ── YOLOv8 ───────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> Dict[str, Any]:
        results    = self._yolo_model(frame, verbose=False)
        r          = results[0]
        h, w       = frame.shape[:2]
        detections: List[Dict[str, Any]] = []
        smoke_conf, fire_conf = 0.0, 0.0

        for box in r.boxes:
            cls  = r.names[int(box.cls)].lower()
            conf = float(box.conf)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            bpx = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            detections.append(_make_det(cls, conf, bpx, w, h))
            if cls == "smoke": smoke_conf = max(smoke_conf, conf)
            if cls == "fire":  fire_conf  = max(fire_conf,  conf)

        safe_conf = max(0.0, 1.0 - smoke_conf - fire_conf * 0.5)
        alert  = smoke_conf >= SMOKE_THRESH or fire_conf >= FIRE_THRESH
        status = _status_label(smoke_conf, fire_conf)
        return _build_result(smoke_conf, fire_conf, safe_conf,
                             alert, status, detections, "yolov8")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _contour_box(
    mask: np.ndarray, w: int, h: int
) -> Optional[Dict[str, int]]:
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return {"x1": x, "y1": y, "x2": x + bw, "y2": y + bh}


def _make_det(
    cls: str,
    conf: float,
    box_px: Dict[str, int],
    w: int,
    h: int,
) -> Dict[str, Any]:
    return {
        "class":      cls,
        "confidence": round(float(conf), 3),
        "box_px":     box_px,
        "box_pct": {
            "x": round(box_px["x1"] / w * 100, 1),
            "y": round(box_px["y1"] / h * 100, 1),
            "w": round((box_px["x2"] - box_px["x1"]) / w * 100, 1),
            "h": round((box_px["y2"] - box_px["y1"]) / h * 100, 1),
        },
    }


def _build_result(
    smoke: float, fire: float, safe: float,
    alert: bool, status: str,
    detections: List[Dict[str, Any]],
    method: str,
) -> Dict[str, Any]:
    return {
        "smoke_pct":  round(smoke * 100, 1),
        "fire_pct":   round(fire  * 100, 1),
        "safe_pct":   round(safe  * 100, 1),
        "alert":      bool(alert),
        "status":     status,
        "detections": detections,
        "method":     method,
    }


def _status_label(smoke: float, fire: float) -> str:
    if fire  >= FIRE_THRESH:  return "FIRE DETECTED"
    if smoke >= SMOKE_THRESH: return "SMOKE DETECTED"
    if smoke >= 0.15:         return "SMOKE POSSIBLE"
    return "ALL CLEAR"


class _NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)


# ── Runners ───────────────────────────────────────────────────────────────────

def run_image(
    detector: SmokeFireDetector,
    path: str,
    save: bool,
    out_dir: str,
) -> Dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None:
        log.error(f"Cannot read: {path}")
        return {}
    frame  = cv2.resize(frame, (640, 480))
    t0     = time.time()
    result = detector.detect(frame)
    result["inference_ms"] = round((time.time() - t0) * 1000, 1)
    result["source"]       = str(path)

    tag = "*** ALERT ***" if result["alert"] else "OK"
    log.info(f"[{tag}] {Path(path).name} | {result['status']} | "
             f"smoke={result['smoke_pct']}% fire={result['fire_pct']}% "
             f"| {result['inference_ms']} ms")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, Path(path).stem + "_detected.jpg")
        cv2.imwrite(out_path, detector.annotate(frame, result))
        result["saved_to"] = out_path
        log.info(f"Saved → {out_path}")

    return result


def run_folder(
    detector: SmokeFireDetector,
    folder: str,
    save: bool,
    out_dir: str,
) -> List[Dict[str, Any]]:
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )
    if not images:
        log.error(f"No images found in {folder}")
        return []
    log.info(f"Processing {len(images)} images from {folder} ...")
    results = [run_image(detector, str(p), save, out_dir) for p in images]
    alerts  = sum(1 for r in results if r.get("alert"))
    log.info(f"Summary: {len(images)} images, {alerts} alerts triggered.")
    return results


def run_video(
    detector: SmokeFireDetector,
    source: Any,
    save: bool,
    out_dir: str,
    show: bool,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open: {source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    writer: Optional[cv2.VideoWriter] = None

    if save:
        os.makedirs(out_dir, exist_ok=True)
        tag      = "webcam" if isinstance(source, int) else Path(str(source)).stem
        out_path = os.path.join(out_dir, f"{tag}_detected.mp4")
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (640, 480)
        )
        log.info(f"Saving video → {out_path}")

    frames, alerts = 0, 0
    log.info("Running. Press Q to quit." if show else "Running. Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame     = cv2.resize(frame, (640, 480))
            result    = detector.detect(frame)
            annotated = detector.annotate(frame, result)
            frames   += 1

            if result["alert"]:
                alerts += 1
                if alerts == 1 or alerts % 30 == 0:
                    log.warning(f"ALERT frame {frames}: {result['status']}")

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("PyroWatch", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    log.info(f"Done: {frames} frames, {alerts} alert frames.")


def save_json(data: Any, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"results_{ts}.json")
    with open(path, "w") as f:
        json.dump(
            data if isinstance(data, list) else [data],
            f, indent=2, cls=_NpEncoder
        )
    log.info(f"JSON saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyroWatch — Wildfire Smoke & Fire Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/detector.py --source data/sample_images/smoke_001.jpg --save
  python src/detector.py --source data/sample_images/ --save --json
  python src/detector.py --source video.mp4 --save
  python src/detector.py --source 0 --show
  python src/detector.py --source data/sample_images/ --model models/rf_classifier.pkl --save
        """,
    )
    p.add_argument("--source",  required=True,
                   help="Image, folder, video file, or webcam index (0/1/...)")
    p.add_argument("--model",   default=None,
                   help="Path to rf_classifier.pkl or YOLOv8 .pt (optional)")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated outputs to --out-dir")
    p.add_argument("--show",    action="store_true",
                   help="Display live window (requires a screen)")
    p.add_argument("--json",    action="store_true",
                   help="Write JSON results file")
    p.add_argument("--out-dir", default="outputs",
                   help="Output directory (default: outputs/)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    detector = SmokeFireDetector(model_path=args.model)
    src      = args.source

    if src.isdigit():
        run_video(detector, int(src), args.save, args.out_dir, args.show)
        return

    src_path = Path(src)

    if src_path.is_dir():
        results = run_folder(detector, src, args.save, args.out_dir)
        if args.json:
            save_json(results, args.out_dir)
        return

    if src_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        run_video(detector, src, args.save, args.out_dir, args.show)
        return

    if src_path.is_file():
        result = run_image(detector, src, args.save, args.out_dir)
        if args.json:
            save_json(result, args.out_dir)
        return

    log.error(f"Source not found or unrecognised: {src}")
    sys.exit(1)


if __name__ == "__main__":
    main()
