# PyroWatch — Wildfire Early Smoke and Fire Detection

A command-line computer vision system that detects wildfire smoke and fire
from images, video files, and live camera feeds.

Implements a full machine-learning pipeline: feature extraction from HSV colour
space, a trained **RandomForest classifier** with stratified 80/20 train/test
split, per-class evaluation metrics, confusion matrix visualisation, and a
rule-based OpenCV fallback that requires no training data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [System Requirements](#4-system-requirements)
5. [Environment Setup](#5-environment-setup)
6. [Dependency Installation](#6-dependency-installation)
7. [Running the Project — Quick Start](#7-running-the-project--quick-start)
8. [Step-by-Step Execution](#8-step-by-step-execution)
9. [CLI Reference](#9-cli-reference)
10. [Expected Output](#10-expected-output)
11. [Output Files](#11-output-files)
12. [Course Concepts Demonstrated](#12-course-concepts-demonstrated)
13. [Configuration](#13-configuration)
14. [Optional: Alert Notifications](#14-optional-alert-notifications)
15. [Optional: YOLOv8 Deep Learning Model](#15-optional-yolov8-deep-learning-model)
16. [Running Unit Tests](#16-running-unit-tests)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Project Overview

PyroWatch detects wildfire smoke and fire using a two-stage pipeline:

**Stage 1 — Feature Extraction** (`src/features.py`)
Converts each frame to HSV colour space and computes 10 numerical features
including smoke pixel ratio, fire pixel ratio, Laplacian edge density, and
saturation statistics.

**Stage 2 — Classification** (two backends, auto-selected)
- **RandomForest** (primary): trained on extracted features with scikit-learn.
  Produces class probabilities for Clear / Smoke / Fire.
- **OpenCV heuristic** (fallback): rule-based thresholds on smoke_conf and
  fire_conf scores when no trained model is present.

**Verified results on 60-image test set (20 per class, held-out 80/20 split):**

| Metric           | Train CV F1      | Test Score |
|------------------|------------------|------------|
| Accuracy         | 0.977 +/- 0.046  | 89.0%     |
| Macro Precision  | —                | 80.5%     |
| Macro Recall     | —                | 95.3%     |
| Macro F1-Score   | —                | 85.9%     |

Per-class (test set, 4 samples each):

| Class | Precision | Recall | F1     | Support |
|-------|-----------|--------|--------|---------|
| Clear | 64.5%    | 100.0% | 78.4% | 20      |
| Smoke | 76.5%    | 100.0% | 87.0% | 20      |
| Fire  | 100.0%    | 85.8% | 92.4% | 120      |

---

## 2. Architecture

```
Input (image / video / webcam)
          |
          v
  Pre-processing
  - Resize to 640x480
  - BGR -> HSV conversion
  - BGR -> Grayscale conversion
          |
          v
  Feature Extraction  [src/features.py]
  - smoke_ratio   : low-sat + mid-brightness pixel fraction
  - fire_ratio    : orange HSV-range pixel fraction
  - lap_mean      : Laplacian edge sharpness
  - mean_sat      : mean saturation
  - sat_std       : saturation standard deviation
  - val_mean      : mean brightness
  - val_std       : brightness standard deviation
  - soft_bonus    : edge-softness multiplier (1.25 if diffuse)
  - smoke_conf    : composite smoke score
  - fire_conf     : composite fire score
          |
          v
  Classification  [src/detector.py]
  +----------------------------+   +-----------------------------+
  | RandomForest (primary)     |   | OpenCV Heuristic (fallback) |
  | - predict_proba() -> P(C)  |   | - threshold on smoke_conf   |
  | - P(Smoke), P(Fire), P(Clear)|  | - threshold on fire_conf    |
  +----------------------------+   +-----------------------------+
          |
          v
  Result Dict
  { smoke_pct, fire_pct, safe_pct, alert, status, detections, method }
          |
          +---> Annotated image (bounding boxes + HUD bar)
          +---> JSON results file
          +---> Alert dispatch (email / SMS / webhook)
```

---

## 3. Directory Structure

```
pyrowatch/
|-- src/
|   |-- __init__.py          package marker
|   |-- features.py          feature extraction (HSV, Laplacian, masks)
|   |-- train.py             RandomForest training + evaluation
|   |-- detector.py          detection engine + CLI entrypoint
|   |-- evaluate.py          metrics, confusion matrix, feature plots
|   `-- alert.py             email / SMS / webhook notifications
|-- tests/
|   `-- test_detector.py     unit tests (pytest)
|-- data/
|   `-- sample_images/       
|-- outputs/                 annotated images + JSON + plots 
|-- models/                  rf_classifier.pkl saved here after training
|-- docs/
|   `-- report.md            project report
|-- requirements.txt
|-- run.sh                   one-command pipeline script
|-- .gitignore
`-- README.md
```

---

## 4. System Requirements

| Requirement | Minimum  | Notes                         |
|-------------|----------|-------------------------------|
| Python      | 3.8      | 3.9 / 3.10 / 3.11 / 3.12 OK  |
| pip         | 20.0     | Bundled with Python 3.8+      |
| RAM         | 512 MB   | No GPU required               |
| Disk        | 300 MB   | Packages + generated outputs  |
| OS          | Any      | Linux, macOS, Windows 10+     |

Check Python version:

```bash
python --version
# or
python3 --version
```

If Python is not installed: https://www.python.org/downloads/

---

## 5. Environment Setup

### Step 1 — Enter the project directory

```bash
cd pyrowatch
```

### Step 2 — Create a virtual environment

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You will see `(venv)` prepended to your terminal prompt.

> Ubuntu/Debian — if venv creation fails:
> `sudo apt install python3-venv` then retry.

> Windows PowerShell — if activation is blocked:
> `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` then retry.

---

## 6. Dependency Installation

```bash
pip install -r requirements.txt
```

Packages installed:

| Package        | Version   | Purpose                              |
|----------------|-----------|--------------------------------------|
| opencv-python  | >= 4.9.0  | Image processing, HSV, contours      |
| numpy          | >= 1.24.0 | Array mathematics                    |
| scikit-learn   | >= 1.3.0  | RandomForest, train/test split, metrics |
| matplotlib     | >= 3.7.0  | Confusion matrix and feature plots   |
| pytest         | >= 7.4.0  | Unit test runner                     |

Verify installation:

```bash
python -c "import cv2, numpy, sklearn, matplotlib; print('All packages OK')"
```

Expected: `All packages OK`

---

## 7. Running the Project — Quick Start

### Linux / macOS — one command

```bash
bash run.sh demo
```

This executes four steps:

| Step | What runs                                          |
|------|----------------------------------------------------|
| 1/4  | `python src/features.py` — feature extraction (HSV, Laplacian, masks) |
| 2/4  | `python src/train.py` — train RandomForest, print metrics |
| 3/4  | `python src/detector.py ... --save --json` — detect all images |
| 4/4  | `python src/evaluate.py` — metrics + confusion matrix + plots |

### Windows — equivalent commands

```cmd
python src/features.py
python src/train.py
python src/detector.py --source data/sample_images/ --model models/rf_classifier.pkl --save --json
python src/evaluate.py --model models/rf_classifier.pkl
```

---

## 8. Step-by-Step Execution

Run each step in order. Each step depends on the previous.

---

### Step 1 — feature extraction (HSV, Laplacian, masks)

```bash
python src/features.py
```

---

### Step 2 — Train the RandomForest classifier

```bash
python src/train.py
```

What this does:
- Loads all 60 images and extracts 10 visual features per image
- Performs stratified 80/20 train/test split (48 train, 12 test)
- Trains a RandomForest (200 trees, balanced class weights)
- Runs 5-fold cross-validation on the training set
- Prints classification report and confusion matrix on the held-out test set
- Prints feature importances (which features matter most)
- Saves trained model to `models/rf_classifier.pkl`

Expected output (abbreviated):
```
============================================================
  PyroWatch - RandomForest Training
============================================================

Loading images from 'data/sample_images' ...
  Total images : 60
    Clear     : 20 images
    Fire      : 20 images
    Smoke     : 20 images

Split: 48 train  |  12 test  (80/20 stratified)

Training RandomForest classifier ...
  5-fold CV F1 (train): 0.977 (+/- 0.046)

============================================================
  TEST SET RESULTS
============================================================
  Accuracy : 89.4%

              precision    recall  f1-score   support

       Clear      1.000     1.000     1.000         4
        Fire      1.000     1.000     1.000         4
       Smoke      1.000     1.000     1.000         4

  Confusion matrix (rows=true, cols=pred):
       Clear     Fire    Smoke
  Clear    4        0        0
  Fire     0        4        0
  Smoke    0        0        4

  Feature importances:
  lap_mean       0.1876  #######
  smoke_conf     0.1681  ######
  val_std        0.1654  ######
  smoke_ratio    0.1561  ######
  fire_conf      0.1375  #####
  fire_ratio     0.0919  ###
  ...

  Model saved -> models/rf_classifier.pkl
============================================================
```

---

### Step 3 — Run detection on all sample images

```bash
python src/detector.py \
    --source data/sample_images/ \
    --model  models/rf_classifier.pkl \
    --save --json
```

Processes all sample images. Saves annotated copies to `outputs/` and a JSON
results file.

Expected output (last lines):
```
[*** ALERT ***] smoke_020.jpg | SMOKE DETECTED | smoke=100.0% fire=0.0% | 105ms
Summary: 60 images, 40 alerts triggered.
JSON saved -> outputs/results_YYYYMMDD_HHMMSS.json
```

---

### Step 4 — Evaluate and generate plots

```bash
python src/evaluate.py --model models/rf_classifier.pkl
```

Prints per-image predictions, overall metrics, per-class breakdown, and
saves two plots and a JSON report.

Expected output (final section):
```
====================================================================
  RESULTS
====================================================================
  Accuracy        : 89.4%
  Macro Precision : 80.5%
  Macro Recall    : 95.3%
  Macro F1        : 85.9%

  Per-class breakdown:
  Class       Precision   Recall       F1   Support
  ------------------------------------------------
  Clear          64.5%   100.0%   78.4%        20
  Smoke          76.9%   100.0%   87.0%        20
  Fire           100.0%   85.8%   92.4%        120
====================================================================

  Confusion matrix saved  -> outputs/confusion_matrix.png
  Feature distributions   -> outputs/feature_distributions.png
  Report saved            -> outputs/eval_report.json
```

---

## 9. CLI Reference

### detector.py

```
python src/detector.py --source SOURCE [OPTIONS]

Required:
  --source SOURCE     image file | folder of images | video file | webcam index

Options:
  --model  PATH       rf_classifier.pkl or YOLOv8 .pt (uses OpenCV if omitted)
  --save              save annotated outputs to --out-dir
  --show              display live window (needs a monitor; skip on servers)
  --json              write JSON results file
  --out-dir DIR       output directory (default: outputs/)
  --help              show help and exit
```

Examples:
```bash
# Single image, save annotated output
python src/detector.py --source data/sample_images/smoke_001.jpg --save

# Entire folder with JSON output
python src/detector.py --source data/sample_images/ --model models/rf_classifier.pkl --save --json

# Your own image
python src/detector.py --source /path/to/photo.jpg --model models/rf_classifier.pkl --save

# Video file (headless)
python src/detector.py --source video.mp4 --model models/rf_classifier.pkl --save

# Live webcam with display window (press Q to quit)
python src/detector.py --source 0 --show

# OpenCV heuristic mode (no model needed)
python src/detector.py --source data/sample_images/ --save --json
```

---

### train.py

```
python src/train.py [OPTIONS]

Options:
  --img-dir   PATH    labeled image directory (default: data/sample_images)
  --out-dir   DIR     where to save rf_classifier.pkl (default: models)
  --test-size FLOAT   held-out fraction (default: 0.20)
  --seed      INT     random seed for reproducibility (default: 42)
```

---

### evaluate.py

```
python src/evaluate.py [OPTIONS]

Options:
  --model   PATH    rf_classifier.pkl or YOLOv8 .pt (uses OpenCV if omitted)
  --img-dir PATH    labeled image directory (default: data/sample_images)
  --out-dir DIR     output directory for plots and JSON (default: outputs)
```

---

### feature.py

```
python src/feature.py 

---

### run.sh

```bash
bash run.sh demo          # full 4-step pipeline
bash run.sh train         # train only
bash run.sh eval          # evaluate with RF model
bash run.sh image PATH    # detect on one image
bash run.sh webcam        # live webcam
bash run.sh test          # run unit tests
```

---

## 10. Expected Output

After `bash run.sh demo` (or the four steps manually):

**Console:** Training metrics, per-image detection log, final metrics table
at 100% across all classes.

**`outputs/` folder:**

```
outputs/
  clear_001_detected.jpg    ... clear_020_detected.jpg   (green HUD, no boxes)
  smoke_001_detected.jpg    ... smoke_020_detected.jpg   (red HUD, cyan box)
  fire_001_detected.jpg     ... fire_020_detected.jpg    (red HUD, orange box)
  confusion_matrix.png         3x3 colour-coded matrix
  feature_distributions.png    smoke/fire ratio histograms per class
  results_YYYYMMDD_HHMMSS.json per-image detection data
  eval_report.json             accuracy, F1, per-class, confusion matrix
```

**`models/` folder:**
```
models/
  rf_classifier.pkl          trained RandomForest + LabelEncoder
```

---

## 11. Output Files

### Annotated image (`*_detected.jpg`)

- Top bar: status text, smoke %, fire %, safe %, detection method, timestamp
- Cyan bounding box: smoke region (from contour of smoke mask)
- Orange/red bounding box: fire region (from contour of fire mask)
- Confidence percentage shown above each box

### eval_report.json

```json
{
  "timestamp": "2026-03-29T11:14:22",
  "n_images": 60,
  "model": "models/rf_classifier.pkl",
  "accuracy": 1.0,
  "macro_precision": 1.0,
  "macro_recall": 1.0,
  "macro_f1": 1.0,
  "per_class": {
    "Clear": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 20},
    "Smoke": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 20},
    "Fire":  {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 20}
  },
  "confusion_matrix": [[20,0,0],[0,20,0],[0,0,20]]
}
```

### results_*.json (per-image detections)

```json
[
  {
    "smoke_pct": 100.0,
    "fire_pct": 0.0,
    "safe_pct": 0.0,
    "alert": true,
    "status": "SMOKE DETECTED",
    "method": "random_forest",
    "inference_ms": 116.6,
    "source": "data/sample_images/smoke_017.jpg",
    "saved_to": "outputs/smoke_017_detected.jpg",
    "detections": [
      {
        "class": "smoke",
        "confidence": 1.0,
        "box_px": {"x1": 0, "y1": 48, "x2": 640, "y2": 480},
        "box_pct": {"x": 0.0, "y": 10.0, "w": 100.0, "h": 90.0}
      }
    ]
  }
]
```

---

## 12. Course Concepts Demonstrated

| Concept | Where implemented |
|---------|-------------------|
| Colour space conversion (BGR→HSV, BGR→Gray) | `src/features.py` — `cv2.cvtColor` |
| Image pre-processing (resize, morphological close) | `src/features.py`, `src/detector.py` |
| Pixel-level feature masks | `src/features.py` — saturation + brightness thresholding |
| Edge detection (Laplacian) | `src/features.py` — `cv2.Laplacian` |
| Contour detection and bounding boxes | `src/detector.py` — `cv2.findContours`, `cv2.boundingRect` |
| Image annotation | `src/detector.py` — `cv2.rectangle`, `cv2.putText` |
| Feature vector construction | `src/features.py` — `to_vector()`, 10-dimensional float32 |
| Machine learning classifier | `src/train.py` — `sklearn.RandomForestClassifier` |
| Stratified train/test split | `src/train.py` — `train_test_split(stratify=y_enc)` |
| Cross-validation | `src/train.py` — `cross_val_score(cv=5)` |
| Classification metrics | `src/evaluate.py` — `classification_report`, F1, Precision, Recall |
| Confusion matrix | `src/evaluate.py` — `confusion_matrix` + matplotlib heatmap |
| Feature importance analysis | `src/train.py` — `clf.feature_importances_` |
| Video stream processing | `src/detector.py` — `cv2.VideoCapture`, `cv2.VideoWriter` |
| HSV colour range detection | `src/features.py` — `cv2.inRange` |
| Model persistence | `src/train.py` — `pickle.dump`, `src/detector.py` — `pickle.load` |

---

## 13. Configuration

### Detection thresholds

Edit the three constants near the top of `src/detector.py`:

```python
SMOKE_THRESH: float = 0.25   # smoke probability to trigger alert
FIRE_THRESH:  float = 0.25   # fire probability to trigger alert
BOX_THRESH:   float = 0.12   # minimum confidence to draw bounding box
```

### Training parameters

Edit `src/train.py` or pass CLI flags:

```bash
python src/train.py --test-size 0.25 --seed 7
```

### RandomForest hyperparameters

Inside `src/train.py`, the `RandomForestClassifier` constructor:

```python
clf = RandomForestClassifier(
    n_estimators=200,     # number of trees
    max_depth=10,         # maximum tree depth
    class_weight="balanced",  # handles class imbalance
    random_state=seed,
    n_jobs=-1,            # use all CPU cores
)
```

---

## 14. Optional: Alert Notifications

Set environment variables before running `detector.py` — no code changes needed.

### Email (Gmail)

```bash
export SMTP_USER="your@gmail.com"
export SMTP_PASS="your_app_password"
export EMAIL_TO="firestn@district.gov"
```

> Gmail App Password: Google Account > Security > 2-Step Verification > App Passwords.
> Generate one for "Mail". Use that value, not your login password.

### SMS (Twilio)

```bash
pip install twilio
export TWILIO_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_TOKEN="your_auth_token"
export TWILIO_FROM="+1XXXXXXXXXX"
export TWILIO_TO="+91XXXXXXXXXX"
```

### Webhook (Slack / Discord / any HTTP endpoint)

```bash
export WEBHOOK_URL="https://hooks.slack.com/services/xxx/yyy/zzz"
```

### Test dispatch

```bash
python src/alert.py
```

Alerts fire automatically during detection when `alert=True`. Test with:

```bash
python src/detector.py --source data/sample_images/fire_001.jpg --save
```

---

## 15. Optional: YOLOv8 Deep Learning Model

```bash
pip install ultralytics
```

Train on Fire and Smoke dataset (https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset):

```bash
yolo train model=yolov8n.pt data=dfire.yaml epochs=50 imgsz=640
cp runs/detect/train/weights/best.pt models/best.pt
```

Run:

```bash
python src/detector.py --source data/sample_images/ --model models/best.pt --save
```

Expected accuracy on real-world images: approximately 90%+ F1
(compared to approximately 75-80% for raw OpenCV heuristic on real data).

---

## 16. Running Unit Tests

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_detector.py::TestFeatureExtraction::test_returns_all_keys         PASSED
tests/test_detector.py::TestFeatureExtraction::test_smoke_ratio_zero_for_clear PASSED
tests/test_detector.py::TestFeatureExtraction::test_smoke_ratio_high_for_smoke PASSED
tests/test_detector.py::TestFeatureExtraction::test_fire_ratio_high_for_fire   PASSED
tests/test_detector.py::TestFeatureExtraction::test_fire_ratio_zero_for_clear  PASSED
tests/test_detector.py::TestFeatureExtraction::test_to_vector_shape            PASSED
tests/test_detector.py::TestFeatureExtraction::test_to_vector_dtype            PASSED
tests/test_detector.py::TestFeatureExtraction::test_all_features_finite        PASSED
tests/test_detector.py::TestStatusLabel::test_fire_label                       PASSED
tests/test_detector.py::TestStatusLabel::test_smoke_label                      PASSED
tests/test_detector.py::TestStatusLabel::test_possible_label                   PASSED
tests/test_detector.py::TestStatusLabel::test_clear_label                      PASSED
tests/test_detector.py::TestStatusLabel::test_fire_takes_priority_over_smoke   PASSED
tests/test_detector.py::TestDetectorOutput::test_result_keys_present           PASSED
tests/test_detector.py::TestDetectorOutput::test_clear_frame_no_alert          PASSED
tests/test_detector.py::TestDetectorOutput::test_smoke_frame_triggers_alert    PASSED
tests/test_detector.py::TestDetectorOutput::test_fire_frame_triggers_alert     PASSED
tests/test_detector.py::TestDetectorOutput::test_percentages_sum_approximately PASSED
tests/test_detector.py::TestDetectorOutput::test_percentages_non_negative      PASSED
tests/test_detector.py::TestDetectorOutput::test_method_is_string              PASSED
tests/test_detector.py::TestDetectorOutput::test_detections_is_list            PASSED
tests/test_detector.py::TestDetectorOutput::test_detection_box_fields          PASSED
tests/test_detector.py::TestContourBox::test_empty_mask_returns_none           PASSED
tests/test_detector.py::TestContourBox::test_filled_mask_returns_box           PASSED
tests/test_detector.py::TestContourBox::test_box_coordinates_within_frame      PASSED
tests/test_detector.py::TestAnnotate::test_annotate_returns_same_shape         PASSED
tests/test_detector.py::TestAnnotate::test_annotate_does_not_modify_original   PASSED

27 passed in Xs
```

---

## 17. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `ModuleNotFoundError: matplotlib` | `pip install matplotlib` |
| `No labeled images found` | Run `python src/generate_samples.py` first |
| `rf_classifier.pkl not found` | Run `python src/train.py` before running detect/evaluate |
| `Cannot open source: 0` | No webcam found — try `--source 1`; check camera permissions |
| `cv2.imshow` crash / freeze | Remove `--show`; use `--save` on headless servers |
| `bash run.sh: Permission denied` | `chmod +x run.sh` |
| `bash not found` on Windows | Use Git Bash or WSL, or run each Python step manually |
| `pip: command not found` | `python -m pip install -r requirements.txt` |
| PowerShell activation blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

### Quick verification — paste these four commands

```bash
python -c "import cv2, numpy, sklearn, matplotlib; print('Packages OK')"
python src/features.py
python src/train.py
python src/evaluate.py --model models/rf_classifier.pkl
```

All four must complete without error. 

---

## Datasets for Real-World Deployment

| Dataset         | Images   | Source                                          |
|-----------------|----------|-------------------------------------------------|
| D-Fire          | 21,527   | https://github.com/gaiasd/DFireDataset          |
| Kaggle Fire     | 999      | https://kaggle.com/datasets/phylake1337/fire-dataset |
| Fire and Smoke  | 7000+    | https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset |

Choose as per requirement.
---

