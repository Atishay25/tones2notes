import config, os
from processing import note_detection_with_onset_offset_regress
import numpy as np
import pickle
import h5py
import mir_eval
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor

from processing import get_filename, traverse_folder, onsets_frames_note_detection


# Onset Post Processor (without regression)
class OnsetsFramesPostProcessor(object):
    def __init__(self, frames_per_second, classes_num):
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        self.frame_threshold = 0.5
        self.onset_threshold = 0.1
        self.offset_threshold = 0.3

    # post-process the model output dict to MIDI events
    # each midi-event is of form : {'onset_time': 4,5, 'offset_time': 12.43, 'midi_note': 51, 'velocity': 39}
    def output_dict_to_midi_events(self, output_dict):      
        est_on_off_note_vels = self.output_dict_to_note_arrays(output_dict)     # piano note outputs to note events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)   # change note format into midi event format
        return est_note_events

    def output_dict_to_note_arrays(self, output_dict):    # finding note events from output probabilities
        # using sharp onsets and offsets (non-regressed)
        output_dict = self.sharp_output_dict(output_dict, onset_threshold=self.onset_threshold, offset_threshold=self.offset_threshold)
        # output_dict to piano notes
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict, frame_threshold=self.frame_threshold) 
        return est_on_off_note_vels

    # calculating sharp output note
    # for example, [0,0.2,0.4,0.2,0] ---> [0,0,1,0,0]
    def sharp_output_dict(self, output_dict, onset_threshold, offset_threshold):
        if 'reg_onset_output' in output_dict.keys():
            output_dict['onset_output'] = self.sharp_output(output_dict['reg_onset_output'], threshold=onset_threshold)
        if 'reg_offset_output' in output_dict.keys():
            output_dict['offset_output'] = self.sharp_output(output_dict['reg_offset_output'], threshold=offset_threshold)
        return output_dict

    # helper function for the above function
    def sharp_output(self, input, threshold=0.3):
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

    def output_dict_to_detected_notes(self, output_dict, frame_threshold):  # getting notes from output dict
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

        est_tuples = np.array(est_tuples)           # (notes, 5) shape
                                                    # 5 values stored are onset, offset, onset_shift, offset_shift, normalized velocity
        est_midi_notes = np.array(est_midi_notes)   # (notes,)
        
        if len(est_midi_notes) == 0:
            return []
        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            velocities = est_tuples[:, 2]

            # (notes, 3), the three columns are onset_times, offset_times and velocity
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)   
            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)
            return est_on_off_note_vels

    def detected_notes_to_events(self, est_on_off_note_vels):     # change format from founded notes to midi event format
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})
        return midi_events


# The Regression based Post processor, as proposed by the paper
class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, offset_threshold, frame_threshold):
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale

    def output_dict_to_midi_events(self, output_dict):      # model output probabilities to MIDI events
        est_on_off_note_vels = self.output_dict_to_note_arrays(output_dict)
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)
        return est_note_events

    # regression based modeling 
    def output_dict_to_note_arrays(self, output_dict):
        # regression outputs to binarized outputs
        # for example, onset = [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output              # Values are binary
        output_dict['onset_shift_output'] = onset_shift_output  

        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output            
        output_dict['offset_shift_output'] = offset_shift_output

        # process matrices results to event results
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)
        return est_on_off_note_vels

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1       # implementation of the delta formula
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False
        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
        for piano_note in range(classes_num):
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

        est_tuples = np.array(est_tuples)
        est_midi_notes = np.array(est_midi_notes) 
        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)
        return est_on_off_note_vels

    def detected_notes_to_events(self, est_on_off_note_vels):
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})
        return midi_events
    
def note_to_freq(piano_note):     # piano note to freq mapping
    return 2 ** ((piano_note - 39) / 12) * 440


# Calculating Evaluation results on the Test dataset
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

    def __call__(self, params):           # returns the mean f1-score for full test set
        stats_dict = self.metrics(params)
        return np.mean(stats_dict['f1'])

    def metrics(self, params):
        n = 0
        list_args = []
        for n, hdf5_path in enumerate(self.hd5f_paths):       # read .h5 files of test data
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == self.split:
                    list_args.append([n, hdf5_path, params])
        with ProcessPoolExecutor() as executor:               # calculate score for each song
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
        elif self.post_processor_type == 'onsets_frames':
            post_processor = OnsetsFramesPostProcessor(self.frames_per_second, classes_num=self.classes_num)
        est_on_off_note_vels = post_processor.output_dict_to_note_arrays(output_dict)
        est_on_offs = est_on_off_note_vels[:, 0 : 2]
        est_midi_notes = est_on_off_note_vels[:, 2]
        est_vels = est_on_off_note_vels[:, 3] * self.velocity_scale
        if self.velocity:         # we putted nonzero velocity where notes were present, so calculating note based scores at velocity != 0
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
                       offset_min_tolerance=self.offset_min_tolerance))
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
        print('note f1: {:.3f}'.format(note_f1))          # printing f1score for each song of test dataset
        return_dict['note_prcision'] = note_precision
        return_dict['note_recall'] = note_recall
        return_dict['note_f1'] = note_f1
        return return_dict 