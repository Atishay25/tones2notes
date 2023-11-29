# tones2notes

### Datasets

Found 3 possible good datasets for Music Transcription -
-   MAPS : https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR
- MusicNet : https://zenodo.org/records/5120004#.YXDPwKBlBpQ
- MAESTRO : https://magenta.tensorflow.org/datasets/maestro 

## Instructions
- For loading features from dataset and storing them into .h5 files
    ```
    python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE
    ```
- For training the model (includes both processing features and training on them)
    ```
    python3 src/main.py train --workspace=$WORKSPACE --model_type='CCNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=8 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=50000 --cuda
    ```
- For transcribing any given audio
    ```
    ./transcribe.sh
    ```
    with required audio_path and model checkpoint


### v1 
- For running `train_model.py`
```
 CUDA_VISIBLE_DEVICES=1,2,4 python3 -W ignore train_model.py MAPS mymodel --channels 1 3 --label-type frame_onset --epoch 10 --steps 3000 --timesteps 128 --early-stop 6 --train-batch-size 8  --val-batch-size 8 
```

## Resources

- See [this](https://github.com/BShakhovsky/PolyphonicPianoTranscription) for notebooks and information about the datasets
- Main : https://github.com/bytedance/piano_transcription
- Main's simpler version : https://github.com/qiuqiangkong/music_transcription_MAPS
- Main paper which we are implementing is [this](https://github.com/bytedance/piano_transcription/blob/master/paper/High-resolution%20Piano%20Transcription%20with%20Pedals%20by%20Regressing%20Precise%20Onsets%20and%20Offsets%20Times_v0.2.pdf)

### Status




- In v2 folder, starting with new implementation of Main
- Go through the paper


https://github.com/Atishay25/tones2notes/assets/110190084/48c37086-663a-475c-84c1-47769e38a464



