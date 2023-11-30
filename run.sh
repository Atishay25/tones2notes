#!/bin/bash

DATASET_DIR="data/MAPS"         # path to dataset
WORKSPACE=$(pwd)                # current working directory

CHECKPOINT_PATH="7000_iterations.pth"

#python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE

# Give model_type as
#   'CCNN' - For Concurrent CNN 
#   'CRNN' - For Concurrent RNN
#   'CRNN_Conditioning' - For CRNN with conditioning (inter-feature dependencies, this is the final model) 

# loss_type as
#   'regress_onset_offset_frame_velocity_bce' - Using regressed features
#   'onset_offset_frame_velocity_bce' - Non-regressed
#python3 src/main.py train --workspace=$WORKSPACE --model_type='CRNN_Conditioning' --loss_type='onset_offset_frame_velocity_bce' --max_note_shift=0 --batch_size=8 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=50000 --cuda

python3 src/results.py infer_prob --workspace=$WORKSPACE --model_type='CRNN_Conditioning' --checkpoint_path=$CHECKPOINT_PATH --dataset='maps' --split='test' --post_processor_type='regression' --cuda 

python3 src/results.py calculate_metrics --workspace=$WORKSPACE --model_type='CRNN_Conditioning' --dataset='maps' --split='test' --post_processor_type='regression' > j10.txt
