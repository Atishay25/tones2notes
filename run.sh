#!/bin/bash

DATASET_DIR="data/MAPS"
WORKSPACE=$(pwd)
CHECKPOINT_PATH="checkpoints/main/CCNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=8/15000_iterations.pth"

#python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE

python3 src/main.py train --workspace=$WORKSPACE --model_type='CCNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=8 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=50000 --cuda

#python3 src/results.py infer_prob --workspace=$WORKSPACE --model_type='CCNN' --checkpoint_path=$CHECKPOINT_PATH --augmentation='none' --dataset='maps' --split='test' --cuda

#python3 src/results.py calculate_metrics --workspace=$WORKSPACE --model_type='CCNN' --augmentation='none' --dataset='maps' --split='test'