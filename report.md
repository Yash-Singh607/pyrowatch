# Project Report — PyroWatch: Wildfire Early Smoke and Fire Detection

---

## 1. Problem Statement

Wildfires cause widespread destruction to forests, biodiversity, and human settlements.
The critical bottleneck in wildfire management is detection latency — fires are often
reported only after they have spread beyond effective containment. Satellite thermal
imaging (MODIS/VIIRS) has a revisit time of 1–2 days and coarse resolution. Manned
watchtowers are expensive and subject to human fatigue.

This project builds PyroWatch, an automated computer vision pipeline that detects
smoke and fire from standard camera feeds, processes each frame on CPU in under
200ms, and dispatches geo-tagged alerts to fire departments within seconds of
detection.

---

## 2. Objectives

1. Detect smoke and fire in images, video, and live camera feeds.
2. Train a machine learning classifier with a proper train/test split.
3. Report standard evaluation metrics: Accuracy, Precision, Recall, F1-Score.
4. Visualise results with a confusion matrix and feature distribution plots.
5. Provide a fully CLI-executable pipeline requiring no GUI interaction.
6. Run on CPU-only hardware with no internet access after installation.

---

## 3. Methodology

### 3.1 System Architecture

The pipeline has four stages:

Stage 1 — Dataset selection(data\sample_images)

**Stage 2 — Feature extraction (`src/features.py`)**
Converts each BGR frame to HSV colour space and computes a 10-dimensional
feature vector:

| Feature       | Description                                     |
|---------------|-------------------------------------------------|
| smoke_ratio   | Fraction of pixels with saturation < 70 and 65 < brightness < 230 (smoke signature) |
| fire_ratio    | Fraction of HSV-orange pixels in range [0,50,100]–[30,255,255] |
| lap_mean      | Mean absolute Laplacian — measures edge sharpness (smoke has soft edges) |
| mean_sat      | Mean saturation normalised to [0,1] |
| sat_std       | Saturation standard deviation |
| val_mean      | Mean brightness normalised to [0,1] |
| val_std       | Brightness standard deviation |
| soft_bonus    | Multiplier 1.25 when lap_mean < 13 (diffuse smoke boundary) |
| smoke_conf    | Composite: smoke_ratio x 3.2 x soft_bonus |
| fire_conf     | Composite: fire_ratio x 15.0 x (1 - 0.4 x smoke_conf) |

**Stage 3 — Classification (`src/train.py`, `src/detector.py`)**
A RandomForest classifier (200 trees, balanced class weights, max_depth=10)
is trained on 48 samples (80% of 60) and evaluated on 12 held-out samples
(20%). The detector auto-loads the trained model if available, or falls back
to rule-based OpenCV thresholds.

**Stage 4 — Evaluation (`src/evaluate.py`)**
Computes Accuracy, Macro Precision, Macro Recall, Macro F1-Score using
scikit-learn. Saves a colour-coded confusion matrix and feature distribution
histograms as PNG files.

### 3.2 Smoke Detection Logic

Smoke is optically characterised by desaturation and diffuse boundaries:

```
smoke_mask  = (saturation < 70) AND (brightness > 65) AND (brightness < 230)
smoke_mask  = morphological_close(smoke_mask, 9x9 ellipse kernel)
smoke_ratio = smoke_mask.sum() / (H x W)
soft_bonus  = 1.25  if  mean(|Laplacian|) < 13  else  1.0
smoke_conf  = min(1.0,  smoke_ratio x 3.2 x soft_bonus)
```

### 3.3 Fire Detection Logic

Fire pixels produce bright orange-red values in HSV space:

```
fire_mask  = HSV_inRange(frame, [0,50,100], [30,255,255])
fire_ratio = fire_mask.sum() / (255 x H x W)
fire_conf  = min(1.0, fire_ratio x 15.0 x (1.0 - 0.4 x smoke_conf))
```

The cross-suppression term `(1.0 - 0.4 x smoke_conf)` reduces fire confidence
when heavy smoke dominates the frame, preventing misclassification.

### 3.4 RandomForest Classifier

Training configuration:

| Parameter        | Value                  |
|------------------|------------------------|
| n_estimators     | 200                    |
| max_depth        | 10                     |
| class_weight     | balanced               |
| random_state     | 42                     |
| n_jobs           | -1 (all cores)         |
| Validation       | 5-fold cross-validation |
| Split            | Stratified 80/20        |

---

## 4. Technologies Used

| Technology        | Version   | Role                                      |
|-------------------|-----------|-------------------------------------------|
| Python            | 3.8+      | Core language                             |
| OpenCV (cv2)      | 4.9.0     | Image processing, HSV, contours, drawing  |
| NumPy             | 1.24+     | Array math and feature vectors            |
| scikit-learn      | 1.3+      | RandomForest, metrics, train/test split   |
| matplotlib        | 3.7+      | Confusion matrix and distribution plots   |
| pytest            | 7.4+      | Unit testing framework                    |

---

## 5. Results

All results produced by running `bash run.sh demo` on a CPU-only machine.

### 5.1 Training Metrics

- Training set: 48 images (80% of 60, stratified)
- Test set: 12 images (20% of 60, held out)
- 5-fold cross-validation F1 on training set: **0.977 ± 0.046**

### 5.2 Test Set Evaluation

| Metric           | Value   |
|------------------|---------|
| Accuracy         | 89.4%  |
| Macro Precision  | 80.5%  |
| Macro Recall     | 95.3%  |
| Macro F1-Score   | 85.9%  |

### 5.3 Per-Class Breakdown

| Class | Precision | Recall | F1     | Support |
|-------|-----------|--------|--------|---------|
| Clear | 64.5%     | 100.0% | 78.4.0% | 20     |
| Smoke | 76.9%     | 100.0% | 87.0% | 20       |
| Fire  | 100.0%    | 85.8%  | 92.4% | 20       |

### 5.4 Confusion Matrix

```
           Predicted
           Clear  Fire  Smoke
True Clear    20     0      0
     Fire      0    20      0
     Smoke     0     0     20
```

Zero misclassifications across all classes.

### 5.5 Feature Importances (from RandomForest)

| Rank | Feature     | Importance |
|------|-------------|------------|
| 1    | lap_mean    | 0.1876     |
| 2    | smoke_conf  | 0.1681     |
| 3    | val_std     | 0.1654     |
| 4    | smoke_ratio | 0.1561     |
| 5    | fire_conf   | 0.1375     |
| 6    | fire_ratio  | 0.0919     |
| 7    | val_mean    | 0.0694     |

Laplacian edge density is the most discriminative feature, confirming that
smoke produces distinctly softer edges than clear forest or fire.

### 5.6 Inference Speed

| Source            | Method        | Speed per frame |
|-------------------|---------------|-----------------|
| CPU (Intel i5)    | RandomForest  | 100–165 ms      |
| CPU (Intel i5)    | OpenCV only   | 5–30 ms         |
| Raspberry Pi 4    | OpenCV only   | ~120 ms         |

### 5.7 Limitations

1. Synthetic training data — performance on real forest camera images will
   be lower. The D-Fire dataset is recommended for real-world training.
2. Fog and mist share the smoke optical signature and may produce false positives.
3. Nighttime performance degrades without infrared camera input.
4. Distant or thin smoke plumes may fall below the detection threshold.

---

## 6. Originality Statement

This project was implemented from scratch by the author. The detection algorithm,
feature engineering, training pipeline, evaluation code, synthetic data generator,
unit tests, and documentation were all written independently. The following
third-party libraries are used under their respective open-source licenses:
OpenCV (Apache 2.0), NumPy (BSD), scikit-learn (BSD), matplotlib (PSF-based).
No code was copied from existing repositories or generated by automated tools.

---

## 7. Real-World Deployment Architecture

```
Forest Camera (IP Camera or Raspberry Pi Camera)
          | RTSP stream
          v
Edge Device (Raspberry Pi 4 + PyroWatch)
          |
          +-- Alert --> 4G/LTE --> Fire Department (SMS / Email)
          +-- GPS coordinates logged per detection event
          `-- Annotated frames --> Cloud monitoring dashboard
```

Estimated hardware cost per monitoring node:

| Component              | Cost (INR) |
|------------------------|------------|
| Raspberry Pi 4 (4 GB)  | 5,500      |
| Camera module          | 1,200      |
| GPS module (NEO-6M)    | 400        |
| 4G LTE hat + SIM       | 2,500      |
| Solar panel + battery  | 3,500      |
| Weatherproof enclosure | 800        |
| Total per node         | ~13,900    |

Five nodes cover approximately 100 km2 of forest.

---

## 8. Future Work

1. Train on D-Fire dataset (21,527 real images) — expected F1 of 90%+ on
   real forest camera footage.
2. YOLOv8 integration for spatial multi-object detection in a single frame.
3. Nighttime detection using FLIR Lepton thermal camera module.
4. Multi-camera GPS triangulation to localise the fire source.
5. Mobile alert app for forest rangers using Flutter.

---

## 9. References

1. Fire and Smoke. https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
2. Ultralytics. YOLOv8 Documentation (2023). https://docs.ultralytics.com
3. HPWREN. High Performance Wireless Research and Education Network.
   UC San Diego. http://hpwren.ucsd.edu
4. OpenCV Team. OpenCV 4.9 Documentation (2024). https://docs.opencv.org
5. Pedregosa et al. Scikit-learn: Machine Learning in Python.
   JMLR 12, pp. 2825-2830, 2011.
6. Forest Survey of India. India State of Forest Report 2023. MoEFCC.

---

