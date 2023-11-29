import numpy as np
import os
import h5py
import config
import librosa
import argparse

from processing import read_maps_midi
from sklearn.model_selection import train_test_split

def pack_maps(args):
    dir = args.dir
    sample_rate = config.sample_rate
    pianos_train = ['AkPnBcht','AkPnCGdD','SptkBGCl','AkPnBsdf','AkPnStgb','SptkBGAm','StbgTGd2']
    pianos_test =  ['ENSTDkCl', 'ENSTDkAm']
    workspace = args.workspace

    hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')
    count = 0
    train_data = []
    for p in pianos_train:
        sub_dir = os.path.join(dir, p, 'MUS')
        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) if os.path.splitext(name)[-1] == '.mid']
        for audio_name in audio_names:
            train_data.append((p, audio_name))
            

    audio_train, audio_val = train_test_split(train_data, test_size=0.2, random_state=42)
    for audio_name in audio_train+audio_val:
        sub_dir = os.path.join(dir, audio_name[0], 'MUS')
        audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name[1]))
        midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name[1]))

        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        midi_dict = read_maps_midi(midi_path)
        
        packed_hdf5_path = os.path.join(hdf5s_dir, '{}.h5'.format(audio_name[1]))
        packed_dir_name = os.path.dirname(packed_hdf5_path)
        if not os.path.exists(packed_dir_name):
            os.makedirs(packed_dir_name)
        
        split = 'validation' if audio_name in audio_val else 'train'
        print(count, audio_name[1], split)
        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('split', data=split.encode(), dtype='S20')
            hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name[1]).encode(), dtype='S100')
            hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name[1]).encode(), dtype='S100')
            hf.attrs.create('duration', data=midi_dict['midi_event_time'][:][-1], dtype=np.float32)
            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=(audio * 32767.).astype(np.int16), dtype=np.int16)
        count += 1

    for p in pianos_test:
        sub_dir = os.path.join(dir, p, 'MUS')
        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) if os.path.splitext(name)[-1] == '.mid']
        for audio_name in audio_names:
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            midi_dict = read_maps_midi(midi_path)

            packed_hdf5_path = os.path.join(hdf5s_dir, '{}.h5'.format(audio_name))
            packed_dir_name = os.path.dirname(packed_hdf5_path)
            if not os.path.exists(packed_dir_name):
                os.makedirs(packed_dir_name)

            split = 'test'
            print(count, audio_name, split)
            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data=split.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name[1]).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name[1]).encode(), dtype='S100')
                hf.attrs.create('duration', data=midi_dict['midi_event_time'][:][-1], dtype=np.float32)
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=(audio * 32767.).astype(np.int16), dtype=np.int16)
            count += 1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=str, required=True, help='Directory of dataset.')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    args = parser.parse_args()
    pack_maps(args)