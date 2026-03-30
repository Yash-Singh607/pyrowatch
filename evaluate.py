"""
PyroWatch - Evaluation Script
================================
Evaluates detection performance on a labeled image folder.
Produces:
  - Console metrics table (Accuracy, Precision, Recall, F1)
  - outputs/confusion_matrix.png
  - outputs/eval_report.json

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/rf_classifier.pkl
    python src/evaluate.py --img-dir data/sample_images --out-dir outputs
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


from detector import SmokeFireDetector

# Label conventions derived from filename prefix
LABEL_MAP = {
    "clear":       "Clear",
    "smoke":       "Smoke",
    "heavy_smoke": "Smoke",
    "fire":        "Fire",
}
CLASS_ORDER = ["Clear", "Smoke", "Fire"]


# ── Label helpers ─────────────────────────────────────────────────────────────

def filename_to_label(filename: str) -> Optional[str]:
    stem = Path(filename).stem.lower()
    for prefix, label in LABEL_MAP.items():
        if stem.startswith(prefix):
            return label
    return None


def result_to_pred(result: Dict) -> str:
    fire  = result.get("fire_pct",  0)
    smoke = result.get("smoke_pct", 0)
    if fire  >= 25: return "Fire"
    if smoke >= 25: return "Smoke"
    return "Clear"


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    out_path: str,
) -> None:
    """Save a colour-coded confusion matrix to out_path."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="PyroWatch — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


# ── Feature distribution plot ─────────────────────────────────────────────────

def plot_feature_distributions(
    smoke_ratios: List[float],
    fire_ratios: List[float],
    labels: List[str],
    out_path: str,
) -> None:
    """Plot smoke_ratio and fire_ratio distributions per class."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Clear": "#4CAF50", "Smoke": "#607D8B", "Fire": "#F44336"}

    for cls in CLASS_ORDER:
        idxs = [i for i, l in enumerate(labels) if l == cls]
        if not idxs:
            continue
        sr = [smoke_ratios[i] for i in idxs]
        fr = [fire_ratios[i]  for i in idxs]
        axes[0].hist(sr, bins=10, alpha=0.7, label=cls, color=colors[cls])
        axes[1].hist(fr, bins=10, alpha=0.7, label=cls, color=colors[cls])

    axes[0].set(title="Smoke ratio distribution", xlabel="smoke_ratio", ylabel="Count")
    axes[1].set(title="Fire ratio distribution",  xlabel="fire_ratio",  ylabel="Count")
    for ax in axes:
        ax.legend()

    fig.suptitle("PyroWatch — Feature Distributions by Class", fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Feature distributions saved → {out_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model_path: Optional[str] = None,
    img_dir: str = "data/sample_images",
    out_dir: str = "outputs",
) -> Dict:
    """
    Run end-to-end evaluation on a labeled image directory.

    Parameters
    ----------
    model_path : optional path to rf_classifier.pkl or YOLOv8 .pt
    img_dir    : directory of images with class-prefixed filenames
    out_dir    : directory for plots and JSON report

    Returns
    -------
    dict with all metric values
    """
    detector = SmokeFireDetector(model_path=model_path)

    images = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and filename_to_label(p.name) is not None
    )

    if not images:
        print(f"\n[ERROR] No labeled images found in '{img_dir}'.")
        print("  Run first:  python src/generate_samples.py\n")
        sys.exit(1)

    y_true: List[str]  = []
    y_pred: List[str]  = []
    rows:   List[Dict] = []
    smoke_ratios: List[float] = []
    fire_ratios:  List[float] = []

    print(f"\n{'='*68}")
    print(f"  PyroWatch Evaluation  |  {len(images)} images  "
          f"|  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*68}")
    print(f"  {'Image':<30} {'True':>6} {'Pred':>6}  "
          f"{'Smoke%':>7} {'Fire%':>6}  {'':>4}")
    print(f"  {'-'*64}")

    for p in images:
        true_label = filename_to_label(p.name)
        frame      = cv2.imread(str(p))
        if frame is None:
            continue
        frame  = cv2.resize(frame, (640, 480))
        result = detector.detect(frame)
        pred   = result_to_pred(result)

        y_true.append(true_label)
        y_pred.append(pred)
        smoke_ratios.append(result["smoke_pct"] / 100)
        fire_ratios.append( result["fire_pct"]  / 100)
        match = "OK" if pred == true_label else "FAIL"

        rows.append({
            "image":     p.name,
            "true":      true_label,
            "pred":      pred,
            "smoke_pct": result["smoke_pct"],
            "fire_pct":  result["fire_pct"],
            "correct":   pred == true_label,
        })
        print(f"  {p.name:<30} {true_label:>6} {pred:>6}  "
              f"{result['smoke_pct']:>6.1f}% {result['fire_pct']:>5.1f}%  {match}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    labels_present = [c for c in CLASS_ORDER if c in y_true]

    acc      = accuracy_score(y_true, y_pred)
    macro_p  = precision_score(y_true, y_pred, labels=labels_present,
                               average="macro", zero_division=0)
    macro_r  = recall_score(   y_true, y_pred, labels=labels_present,
                               average="macro", zero_division=0)
    macro_f1 = f1_score(       y_true, y_pred, labels=labels_present,
                               average="macro", zero_division=0)

    per_class_report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=labels_present,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{'='*68}")
    print("  RESULTS")
    print(f"{'='*68}")
    print(f"  Accuracy        : {acc:.1%}")
    print(f"  Macro Precision : {macro_p:.1%}")
    print(f"  Macro Recall    : {macro_r:.1%}")
    print(f"  Macro F1        : {macro_f1:.1%}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*48}")
    for cls in labels_present:
        m = per_class_report[cls]
        print(f"  {cls:<10} {m['precision']:>10.1%} {m['recall']:>8.1%} "
              f"{m['f1-score']:>8.1%} {int(m['support']):>9}")
    print(f"{'='*68}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plot_confusion_matrix(
        cm, labels_present,
        os.path.join(out_dir, "confusion_matrix.png"),
    )
    plot_feature_distributions(
        smoke_ratios, fire_ratios, y_true,
        os.path.join(out_dir, "feature_distributions.png"),
    )

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "timestamp":       datetime.now().isoformat(),
        "n_images":        len(y_true),
        "model":           model_path or "opencv",
        "accuracy":        round(acc,      4),
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "macro_f1":        round(macro_f1, 4),
        "per_class": {
            cls: {
                "precision": round(per_class_report[cls]["precision"], 4),
                "recall":    round(per_class_report[cls]["recall"],    4),
                "f1":        round(per_class_report[cls]["f1-score"],  4),
                "support":   int(per_class_report[cls]["support"]),
            }
            for cls in labels_present
        },
        "confusion_matrix": cm.tolist(),
        "per_image": rows,
    }

    rpath = os.path.join(out_dir, "eval_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {rpath}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyroWatch Evaluation")
    ap.add_argument("--model",   default=None,
                    help="Path to rf_classifier.pkl or YOLOv8 .pt (optional)")
    ap.add_argument("--img-dir", default="data/sample_images",
                    help="Labeled image directory (default: data/sample_images)")
    ap.add_argument("--out-dir", default="outputs",
                    help="Output directory for plots and JSON (default: outputs)")
    args = ap.parse_args()
    evaluate(args.model, args.img_dir, args.out_dir)
