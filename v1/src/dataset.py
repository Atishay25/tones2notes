import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
np.random.seed(42)
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class MIDI_Dataset:
    def __init__(self, path):
        self.path = path
        self.all_notes = []
        self.num_files = 0

    def read_data(self):
        data_dir = pathlib.Path(self.path)
        filenames = glob.glob(str(data_dir/'**/*.mid*'))
        self.num_files = len(filenames)
        print('Number of files:', self.num_files)
        for f in filenames[:self.num_files]:
            notes = self.midi_to_notes(f)
            self.all_notes.append(notes)
        self.all_notes = pd.concat(self.all_notes)
        n_notes = len(self.all_notes)
        print("Number of notes parsed : ", n_notes)

    def midi_to_notes(self,midi_file):
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            notes['filename'].append(midi_file)
            prev_start = start
        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
    

if __name__ == "__main__":
    dataset = MIDI_Dataset('./data/MIDI')
    dataset.read_data()
    print(dataset.all_notes.head())
