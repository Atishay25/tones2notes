#!/bin/bash

DATASET_DIR="data/MAPS"
WORKSPACE=$(pwd)

#python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE

python3 src/main.py train --workspace=$WORKSPACE --model_type='CCNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=8 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=50000 --cuda