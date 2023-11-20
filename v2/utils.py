import os
import logging
import h5py
import soundfile
import librosa
import audioread
import numpy as np
import pandas as pd
import csv
import datetime
import collections
import pickle
from mido import MidiFile

def read_maps_midi(midi_path):
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
    assert len(midi_file.tracks) == 1

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[0]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


