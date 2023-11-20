# tones2notes

### Datasets

Found 3 possible good datasets for Music Transcription -
-   MAPS : https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR
- MusicNet : https://zenodo.org/records/5120004#.YXDPwKBlBpQ
- MAESTRO : https://magenta.tensorflow.org/datasets/maestro 



### Commands



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
