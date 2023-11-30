import os
import numpy as np
import h5py
import librosa
from processing import traverse_folder, TargetProcessor, plot_waveform_midi_targets
import config

# Class to be used by Dataloader
# Takes audio + midi features from .h5 files of each audio segment as input
# Returns the waveform (X) and targets (y) for each audio segment
# max_note_shift is the number of semitone for pitch augmentation
class MapsDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, max_note_shift=0):
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.max_note_shift = max_note_shift
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.random_state = np.random.RandomState(42)
        # processing MIDI events to targets
        self.target_processor = TargetProcessor(self.segment_seconds, self.frames_per_second, self.begin_note, self.classes_num)

    # to prepare input and target of a segment and provide it to dataloader during training
    # meta is a dictionary with keys {'hdf5_name': name of file, 'start_time': start time of segment}
    # returns data dictionary with features
    def __getitem__(self, meta):
        hdf5_name = meta[0]
        start_time = meta[1]
        hdf5_path = os.path.join(self.hdf5s_dir,hdf5_name)
        data_dict = {}
        note_shift = self.random_state.randint(low=-self.max_note_shift, high=self.max_note_shift + 1)      # pitch augmentation

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples
            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples
            waveform = ((hf['waveform'][start_sample : end_sample])/32767.).astype(np.float32)
            if note_shift != 0:         # augmenting pitch
                waveform = librosa.effects.pitch_shift(waveform, self.sample_rate, note_shift, bins_per_octave=12)
            data_dict['waveform'] = waveform
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            (target_dict, note_events) = self.target_processor.process(start_time, midi_events_time, midi_events, note_shift=note_shift)

        for key in target_dict.keys():      # copy target values (y) in data 
            data_dict[key] = target_dict[key]
        debugging = False
        if debugging:           # set debugging = True to generate the LogMel spectrum and other features of processed data
            plot_waveform_midi_targets(data_dict, start_time, note_events)
        return data_dict

# This Class generate samples of segments each of length segment_seconds
# and provides it to the Maps_Dataset class for processing as input
class Sampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, batch_size, mini_data, random_seed=42):
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []
        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([audio_name, start_time])
                        start_time += self.hop_seconds
                    n += 1
                    if mini_data and n == 10:
                        break
        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    # yields segments based on batch size
    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:      # create a list of segments of size = batch_size
                index = self.segment_indexes[self.pointer]
                self.pointer += 1
                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)
                batch_segment_list.append(self.segment_list[index])
                i += 1
            yield batch_segment_list

    # return current length of segment_list 
    def __len__(self):
        return len(self.segment_list)
        
    # dictionary to be for generating segments
    def state_dict(self):   
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


# similar Class as above, for testing & evaluation phase
class TestSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, batch_size, mini_data, random_seed=42):
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20            # Number of mini-batches to validate
        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []
        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([audio_name, start_time])
                        start_time += self.hop_seconds
                    n += 1
                    if mini_data and n == 10:
                        break
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        pointer = 0
        iteration = 0
        while True:
            if iteration == self.max_evaluate_iteration:
                break
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1
            iteration += 1
            yield batch_segment_list

    def __len__(self):
        return -1

# Join input and targets og segments to a batch
# Note that each key of returned dictionary consists of features
# of shape = (batch_size, num_classes) and the waveform
# is of the shape = (batch_size, segment_samples)
def collate_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict