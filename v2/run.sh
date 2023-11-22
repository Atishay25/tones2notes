#!/bin/bash

DATASET_DIR="../data/MAPS"
WORKSPACE=$(pwd)

#python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE

python3 main.py train --workspace=$WORKSPACE --model_type='Net' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda