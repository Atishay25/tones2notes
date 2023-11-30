#!/bin/bash

CHECKPOINT_PATH="checkpoints/main/CCNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=8/20000_iterations.pth"
MODEL_TYPE="CCNN"
python3 src/transcribe.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path='./data/MAPS/ENSTDkCl/MUS/MAPS_MUS-bk_xmas1_ENSTDkCl.wav' --cuda