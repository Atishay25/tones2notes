# tones2notes

### Dataset

Currently using the Nottingham music dataset. Refer to [this](https://github.com/jukedeck/nottingham-dataset/) repository for its usage


## Commands



- For training
```
 CUDA_VISIBLE_DEVICES=1,2,4 python3 -W ignore train_model.py MAPS mymodel --channels 1 3 --label-type frame_onset --epoch 10 --steps 3000 --timesteps 128 --early-stop 6 --train-batch-size 8  --val-batch-size 8 
```

