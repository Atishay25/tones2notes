import data_generator as dg
import numpy as np

meta = dict()
meta['hdf5_name'] = 'MAPS_MUS-mz_330_1_SptkBGCl.h5'
meta['start_time'] = 2.0

td = dg.MapsDataset(hdf5s_dir="./hdf5s", segment_seconds=10, frames_per_second=100,max_note_shift=0,augmentor='none')


tf = td[meta]
#for i in tf.keys():
#    print(i, tf[i].shape)
#    print(np.count_nonzero(tf[i]))

dd = dg.Sampler(hdf5s_dir='./hdf5s', split='train', 
        segment_seconds=10, hop_seconds=1, 
        batch_size=32, mini_data=0)
