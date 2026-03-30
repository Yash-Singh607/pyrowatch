#!/usr/bin/env bash
# PyroWatch - Pipeline runner
#
# Usage:
#   bash run.sh demo      full pipeline (generate -> train -> detect -> evaluate)
#   bash run.sh train     train only
#   bash run.sh eval      evaluate with trained RF model
#   bash run.sh image PATH
#   bash run.sh webcam
#   bash run.sh test      run unit tests

set -e
MODE="${1:-demo}"

echo ""
echo "========================================"
echo "   PyroWatch -- Wildfire Smoke Detector "
echo "========================================"
echo ""

case "$MODE" in

  demo)
    echo "[1/4] Generating 60 sample images..."
    python generate_samples.py

    echo ""
    echo "[2/4] Training RandomForest classifier..."
    python train.py

    echo ""
    echo "[3/4] Running detector on all samples..."
    python detector.py \
        --source data/sample_images/ \
        --model  models/rf_classifier.pkl \
        --save --json

    echo ""
    echo "[4/4] Evaluation metrics + confusion matrix..."
    python evaluate.py \
        --model  models/rf_classifier.pkl \
        --out-dir outputs/

    echo ""
    echo "Done. Check outputs/ for annotated images, confusion_matrix.png,"
    echo "feature_distributions.png, and eval_report.json"
    ;;

  train)
    python train.py "${@:2}"
    ;;

  eval)
    python evaluate.py --model models/rf_classifier.pkl "${@:2}"
    ;;

  image)
    [ -z "$2" ] && { echo "Usage: bash run.sh image <path>"; exit 1; }
    python detector.py --source "$2" --model models/rf_classifier.pkl --save
    ;;

  webcam)
    python detector.py --source 0 --show
    ;;

  test)
    python -m pytest tests/ -v
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run.sh [demo|train|eval|image <path>|webcam|test]"
    exit 1
    ;;
esac
