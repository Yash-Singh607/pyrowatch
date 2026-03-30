"""
PyroWatch - Model Training Script
===================================
Trains a RandomForest classifier on extracted visual features.
Performs an 80/20 stratified train/test split, prints per-class metrics,
and saves the trained model to models/rf_classifier.pkl.

Usage:
    python src/train.py
    python src/train.py --img-dir data/sample_images --out-dir models
"""

import argparse
import pickle
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


from features import extract, to_vector, FEATURE_NAMES

# Label map derived from filename prefix
LABEL_MAP = {
    "clear":       "Clear",
    "smoke":       "Smoke",
    "heavy_smoke": "Smoke",
    "fire":        "Fire",
}


def load_dataset(img_dir: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load all labeled images from img_dir, extract feature vectors.

    Parameters
    ----------
    img_dir : str
        Directory containing images named with class prefix (e.g. smoke_001.jpg).

    Returns
    -------
    X : np.ndarray of shape (N, 10)
    y : list of string labels (N,)
    paths : list of file paths (for traceability)
    """
    images = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    X, y, paths = [], [], []
    skipped = 0

    for p in images:
        stem = p.stem.lower()
        label = None
        for prefix, cls in LABEL_MAP.items():
            if stem.startswith(prefix):
                label = cls
                break
        if label is None:
            skipped += 1
            continue

        frame = cv2.imread(str(p))
        if frame is None:
            skipped += 1
            continue
        frame = cv2.resize(frame, (640, 480))

        feats = extract(frame)
        vec   = to_vector(feats)
        X.append(vec)
        y.append(label)
        paths.append(str(p))

    if skipped:
        print(f"  Skipped {skipped} unlabeled/unreadable files.")

    return np.array(X, dtype=np.float32), y, paths


def train(img_dir: str = "data/sample_images",
          out_dir: str = "models",
          test_size: float = 0.20,
          seed: int = 42) -> None:
    """
    Full training pipeline: load → split → train → evaluate → save.

    Parameters
    ----------
    img_dir   : directory of labeled images
    out_dir   : where to save rf_classifier.pkl
    test_size : fraction held out for final test evaluation
    seed      : random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print("  PyroWatch — RandomForest Training")
    print(f"{'='*60}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading images from '{img_dir}' ...")
    X, y, paths = load_dataset(img_dir)

    if len(X) == 0:
        print("[ERROR] No labeled images found. Run generate_samples.py first.")
        sys.exit(1)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Total images : {len(X)}")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls:<10}: {cnt} images")

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── 3. Train / test split (stratified) ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        stratify=y_enc,
        random_state=seed,
    )
    print(f"\nSplit: {len(X_train)} train  |  {len(X_test)} test  "
          f"(80/20 stratified)")

    # ── 4. Train RandomForest ─────────────────────────────────────────────────
    print("\nTraining RandomForest classifier ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 5. Cross-validation on training set ───────────────────────────────────
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"  5-fold CV F1 (train): {cv_scores.mean():.3f} "
          f"(+/- {cv_scores.std():.3f})")

    # ── 6. Test set evaluation ────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = (y_pred == y_test).mean()

    print(f"\n{'='*60}")
    print("  TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.1%}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3,
    ))

    # ── 7. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " ".join(f"{c:>8}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<7} " + " ".join(f"{v:>8}" for v in row))

    # ── 8. Feature importances ────────────────────────────────────────────────
    print("\n  Feature importances:")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 40)
        print(f"  {name:<14} {imp:.4f}  {bar}")

    # ── 9. Save model ─────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "rf_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "label_encoder": le}, f)
    print(f"\n  Model saved → {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyroWatch — Train RandomForest classifier")
    ap.add_argument("--img-dir",   default="data/sample_images",
                    help="Directory of labeled images")
    ap.add_argument("--out-dir",   default="models",
                    help="Directory to save trained model")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="Fraction of data for final test (default: 0.20)")
    ap.add_argument("--seed",      type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()
    train(args.img_dir, args.out_dir, args.test_size, args.seed)
