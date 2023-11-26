import config, os
from processing import note_detection_with_onset_offset_regress
import numpy as np
import pickle
import h5py
import mir_eval
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor

from processing import (get_filename, traverse_folder, onsets_frames_note_detection)

class OnsetsFramesPostProcessor(object):
    def __init__(self, frames_per_second, classes_num):
        """Postprocess the Googl's onsets and frames system output. Only used
        for comparison.

        Args:
          frames_per_second: int
          classes_num: int
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        
        self.frame_threshold = 0.5
        self.onset_threshold = 0.1
        self.offset_threshold = 0.3

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]
        """

        # Post process piano note outputs to piano note events information
        est_on_off_note_vels = self.output_dict_to_note_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity]"""
        
        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        return est_note_events

    def output_dict_to_note_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]
        """

        # Sharp onsets and offsets
        output_dict = self.sharp_output_dict(
            output_dict, onset_threshold=self.onset_threshold, 
            offset_threshold=self.offset_threshold)

        # Post process output_dict to piano notes
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict, frame_threshold=self.frame_threshold) 

        return est_on_off_note_vels

    def sharp_output_dict(self, output_dict, onset_threshold, offset_threshold):
        """Sharp onsets and offsets. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]
        [0., 0., 1., 0., 0., 0.]

        Args:
          output_dict: {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            ...}
          onset_threshold: float
          offset_threshold: float

        Returns:
          output_dict: {
            'onset_output': (frames_num, classes_num), 
            'offset_output': (frames_num, classes_num)}
        """
        if 'reg_onset_output' in output_dict.keys():
            output_dict['onset_output'] = self.sharp_output(
                output_dict['reg_onset_output'], 
                threshold=onset_threshold)

        if 'reg_offset_output' in output_dict.keys():
            output_dict['offset_output'] = self.sharp_output(
                output_dict['reg_offset_output'], 
                threshold=offset_threshold)

        return output_dict

    def sharp_output(self, input, threshold=0.3):
        """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

        Args:
          input: (frames_num, classes_num)

        Returns:
          output: (frames_num, classes_num)
        """
        (frames_num, classes_num) = input.shape
        output = np.zeros_like(input)

        for piano_note in range(classes_num):
            loct = None
            for i in range(1, frames_num - 1):
                if input[i, piano_note] > threshold and input[i, piano_note] > input[i - 1, piano_note] and input[i, piano_note] > input[i + 1, piano_note]:
                    loct = i
                else:
                    if loct is not None:
                        output[loct, piano_note] = 1
                        loct = None

        return output

    def output_dict_to_detected_notes(self, output_dict, frame_threshold):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """

        est_tuples = []
        est_midi_notes = []

        for piano_note in range(self.classes_num):
            
            est_tuples_per_note = onsets_frames_note_detection(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                threshold=frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 3)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)
        
        if len(est_midi_notes) == 0:
            return []
        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            velocities = est_tuples[:, 2]
        
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            """(notes, 3), the three columns are onset_times, offset_times and velocity."""

            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

            return est_on_off_note_vels

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, 
        offset_threshold, frame_threshold):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]
        """

        # Post process piano note outputs to piano note events information
        est_on_off_note_vels = self.output_dict_to_note_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        return est_note_events

    def output_dict_to_note_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output  

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        return est_on_off_note_vels

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
 
        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                frame_threshold=self.frame_threshold)
            
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events
    
def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440

class ScoreCalculator(object):
    def __init__(self, hdf5s_dir, probs_dir, split, post_process_type='regression'):
        self.split = split
        self.probs_dir = probs_dir
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.velocity_scale = config.velocity_scale
        self.velocity = True

        self.evaluate_frame = True
        self.onset_tolerance = 0.05
        self.offset_ratio = 0.2
        self.offset_min_tolerance = 0.05

        self.post_processor_type = post_process_type

        (hdf5_names, self.hd5f_paths) = traverse_folder(hdf5s_dir)

    def __call__(self, params):
        stats_dict = self.metrics(params)
        return np.mean(stats_dict['f1'])

    def metrics(self, params):
        n = 0
        list_args = []

        for n, hdf5_path in enumerate(self.hd5f_paths):
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == self.split:
                    list_args.append([n, hdf5_path, params])

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.calculate_score_per_song, list_args)

        stats_list = list(results)
        stats_dict = {}
        for key in stats_list[0].keys():
            stats_dict[key] = [e[key] for e in stats_list if key in e.keys()]

        return stats_dict
    
    def calculate_score_per_song(self, args):
        n = args[0]
        hdf5_path = args[1]
        [onset_threshold, offset_threshold, frame_threshold] = args[2]

        return_dict = {}

        prob_path = os.path.join(self.probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
        total_dict = pickle.load(open(prob_path, 'rb'))

        ref_on_off_pairs = total_dict['ref_on_off_pairs']
        ref_midi_notes = total_dict['ref_midi_notes']
        output_dict = total_dict

        if self.evaluate_frame:
            y_pred = (np.sign(total_dict['frame_output']- frame_threshold) + 1) / 2
            y_pred[np.where(y_pred==0.5)] = 0
            y_true = total_dict['frame_roll']
            y_pred = y_pred[0 : y_true.shape[0], :]
            y_true = y_true[0 : y_pred.shape[0], :]

            tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
            return_dict['frame_precision'] = tmp[0][1]
            return_dict['frame_recall'] = tmp[1][1]
            return_dict['frame_f1'] = tmp[2][1]

        if self.post_processor_type == 'regression':
            post_processor = RegressionPostProcessor(self.frames_per_second,
                classes_num=self.classes_num, onset_threshold=onset_threshold,
                offset_threshold=offset_threshold, frame_threshold=frame_threshold)
        
        elif self.post_processor_type == 'onsets_framse':
            post_processor = OnsetsFramesPostProcessor(self.frames_per_second, classes_num=self.classes_num)

        
        est_on_off_note_vels = post_processor.output_dict_to_note_arrays(output_dict)
        
        est_on_offs = est_on_off_note_vels[:, 0 : 2]
        est_midi_notes = est_on_off_note_vels[:, 2]
        est_vels = est_on_off_note_vels[:, 3] * self.velocity_scale
    
        if self.velocity:
            (note_precision, note_recall, note_f1, _) = (
                mir_eval.transcription_velocity.precision_recall_f1_overlap(
                       ref_intervals=ref_on_off_pairs,
                       ref_pitches=note_to_freq(ref_midi_notes),
                       ref_velocities=total_dict['ref_velocity'],
                       est_intervals=est_on_offs,
                       est_pitches=note_to_freq(est_midi_notes),
                       est_velocities=est_vels,
                       onset_tolerance=self.onset_tolerance, 
                       offset_ratio=self.offset_ratio, 
                       offset_min_tolerance=self.offset_min_tolerance ))
        else:
            (note_precision, note_recall, note_f1, _) = \
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs, 
                    ref_pitches=note_to_freq(ref_midi_notes), 
                    est_intervals=est_on_offs, 
                    est_pitches=note_to_freq(est_midi_notes), 
                    onset_tolerance=self.onset_tolerance, 
                    offset_ratio=self.offset_ratio, 
                    offset_min_tolerance=self.offset_min_tolerance)
        
        print('note f1: {:.3f}'.format(note_f1))

        return_dict['note_prcision'] = note_precision
        return_dict['note_recall'] = note_recall
        return_dict['note_f1'] = note_f1
        return return_dict 