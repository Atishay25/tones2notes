# tones2notes

Orewa Monkey D Luffy, Kaizokou Nari Oto Kuda!!

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

## Resources

- See [this](https://github.com/BShakhovsky/PolyphonicPianoTranscription) for notebooks and information about the datasets
- Main : https://github.com/bytedance/piano_transcription
- Main's simpler version : https://github.com/qiuqiangkong/music_transcription_MAPS
- Main paper which we are implementing is [this](https://github.com/bytedance/piano_transcription/blob/master/paper/High-resolution%20Piano%20Transcription%20with%20Pedals%20by%20Regressing%20Precise%20Onsets%20and%20Offsets%20Times_v0.2.pdf)

https://github.com/Atishay25/tones2notes/assets/110190084/48c37086-663a-475c-84c1-47769e38a464

## Transcription Results
- L theme from Death Note. The Original music is [this](https://www.youtube.com/watch?v=qR6dzwQahOM)
    https://github.com/Atishay25/tones2notes/assets/96432271/b5e7aff0-c105-4a3c-9cbd-6c6b0d84c3a7
- Another song

