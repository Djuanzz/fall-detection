#!/bin/bash
# train_ntu25.sh
# =============
# Training pipeline untuk NTU RGB+D 25-joint binary fall detection.
#
# LANGKAH 1: Siapkan dataset
#   bash train_ntu25.sh prepare /path/to/nturgbd_skeleton
#
# LANGKAH 2: Training (semua modality)
#   bash train_ntu25.sh train
#
# LANGKAH 3: Inference manual
#   bash train_ntu25.sh infer /path/to/file.skeleton weights/path.pt

set -e
GPU=0

case "$1" in
  prepare)
    SKEL_DIR="${2:-dataset/nturgbd_skeleton}"
    echo "=== Menyiapkan dataset NTU RGB+D 25-joint ==="
    echo "Skeleton dir: $SKEL_DIR"
    python scripts/prepare_dataset_ntu25.py \
        --skeleton_dir "$SKEL_DIR" \
        --out_dir      dataset/ntu25_data \
        --max_frames   150 \
        --split_method random \
        --val_split    0.2
    echo "Dataset siap di dataset/ntu25_data/"
    ;;

  train)
    echo "=== Training: Joint (Balanced) ==="
    python main.py \
        --config  config/fall-detection-ntu/balanced.yaml \
        --work-dir work_dir/fall_ntu25_balanced \
        --device "$GPU"

    echo "=== Training: Bone (Balanced) ==="
    python main.py \
        --config  config/fall-detection-ntu/balanced_bone.yaml \
        --work-dir work_dir/fall_ntu25_balanced_bone \
        --device "$GPU"

    echo "=== Training: Motion (Balanced) ==="
    python main.py \
        --config  config/fall-detection-ntu/balanced_motion.yaml \
        --work-dir work_dir/fall_ntu25_balanced_motion \
        --device "$GPU"
    ;;

  train-joint)
    python main.py \
        --config  config/fall-detection-ntu/balanced.yaml \
        --work-dir work_dir/fall_ntu25_balanced \
        --device "$GPU"
    ;;

  train-full)
    echo "=== Training: Full dataset dengan class weight ==="
    python main.py \
        --config  config/fall-detection-ntu/full.yaml \
        --work-dir work_dir/fall_ntu25_full \
        --device "$GPU"
    ;;

  infer)
    SKEL="${2:?Argumen 2: path file .skeleton}"
    WEIGHTS="${3:?Argumen 3: path weights .pt}"
    python scripts/manual_inference_ntu.py \
        --skeleton "$SKEL" \
        --weights  "$WEIGHTS" \
        --config   config/fall-detection-ntu/balanced.yaml
    ;;

  realtime)
    SKEL="${2:?Argumen 2: path file .skeleton}"
    WEIGHTS="${3:?Argumen 3: path weights .pt}"
    python scripts/realtime_ntu_inference.py \
        --skeleton "$SKEL" \
        --weights  "$WEIGHTS" \
        --config   config/fall-detection-ntu/balanced.yaml \
        --mode file \
        --fps 10
    ;;

  *)
    echo "Cara pakai:"
    echo "  bash train_ntu25.sh prepare  <skeleton_dir>     # siapkan dataset"
    echo "  bash train_ntu25.sh train                       # training semua modality"
    echo "  bash train_ntu25.sh train-joint                 # joint only"
    echo "  bash train_ntu25.sh train-full                  # full dataset"
    echo "  bash train_ntu25.sh infer    <skeleton> <weights>  # inferensi satu file"
    echo "  bash train_ntu25.sh realtime <skeleton> <weights>  # simulasi real-time"
    ;;
esac
