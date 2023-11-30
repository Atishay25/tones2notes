import os
import librosa
import audioread
import soundfile as sf
import numpy as np
import datetime
import pickle
import config
from mido import MidiFile

def create_folder(fd):          # create folder if not already exists
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):         # get filename of the calling file
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def read_maps_midi(midi_path):          # read MIDI file 
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

def traverse_folder(folder):        # helper function to traverse folder, used for reading .h5 files
    paths = []
    names = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
    return names, paths

# Main preprocessor used for preprocessing each audio and MIDI
class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events, note_shift=0):     # process the given midi events
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        note_events = []
        buffer_dict = {}
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        for i in range(ex_bgn_idx, fin_idx):
            attribute_list = midi_events[i].split(' ')
            if attribute_list[0] in ['note_on', 'note_off']:
                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))

        for note_event in note_events:
            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))
                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1
                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)
                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):       # get regression targets 
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0    

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll
            }
        return target_dict, note_events
    
    def get_regression(self, input):            # regressing on features based on delta and parameter J
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20         # J = 5 used here 
        output = (1. - output)
        return output
    

# Converting MIDI events into a MIDI file
def write_events_to_midi(start_time, note_events, midi_path):
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)
    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat
    # Using 2 Tracks
    # Track 0  - for meta data of the file
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1 - for actual transcripted data
    track1 = MidiTrack()
    message_roll = []
    for note_event in note_events:
        # onset
        message_roll.append({'time': note_event['onset_time'], 'midi_note': note_event['midi_note'], 'velocity': note_event['velocity']})
        # offset
        message_roll.append({'time': note_event['offset_time'], 'midi_note': note_event['midi_note'], 'velocity': 0})

    # sorting MIDI messages based on time
    message_roll.sort(key=lambda note_event: note_event['time'])
    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)
    midi_file.save(midi_path)
    

# Plotting output of the features generated by model
def plot_waveform_midi_targets(data_dict, start_time, note_events):
    import matplotlib.pyplot as plt
    create_folder('debug')
    audio_path = 'debug/debug.wav'
    midi_path = 'debug/debug.mid'
    fig_path = 'debug/debug.png'
    sf.write(audio_path, data_dict['waveform'], samplerate=config.sample_rate)
    write_events_to_midi(start_time, note_events, midi_path)
    x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=160, window='hann', center=True)
    x = np.abs(x) ** 2
    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(30, 30))
    fontsize = 20
    axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto')
    axs[2].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto')
    axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[5].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[6].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[7].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Log spectrogram', fontsize=fontsize)
    axs[1].set_title('onset_roll', fontsize=fontsize)
    axs[2].set_title('offset_roll', fontsize=fontsize)
    axs[3].set_title('reg_onset_roll', fontsize=fontsize)
    axs[4].set_title('reg_offset_roll', fontsize=fontsize)
    axs[5].set_title('frame_roll', fontsize=fontsize)
    axs[6].set_title('velocity_roll', fontsize=fontsize)
    axs[7].set_title('mask_roll', fontsize=fontsize)
    plt.savefig(fig_path)
    print('Write out to {}, {}, {}!'.format(audio_path, midi_path, fig_path))


# Container to store stats during training and dump them into pickle files
class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.statistics_dict = {'train': [], 'validation': [], 'test': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))
        resume_statistics_dict = {'train': [], 'validation': [], 'test': []}
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
        self.statistics_dict = resume_statistics_dict

# Onset frames note detection (Non-regressed)
def onsets_frames_note_detection(frame_output, onset_output, offset_output, velocity_output, threshold):
    output_tuples = []
    loct = None
    for i in range(onset_output.shape[0]):
        # Use onset_output is used to detect the presence of notes
        if onset_output[i] > threshold:
            if loct:
                output_tuples.append([loct, i, velocity_output[loct]])
            loct = i
        if loct and i > loct:
            # Use frame_output is used to detect the offset of notes
            if frame_output[i] <= threshold:
                output_tuples.append([loct, i, velocity_output[loct]])
                loct = None
    output_tuples.sort(key=lambda pair: pair[0])
    return output_tuples

# process predicions to get note events
# by detecting onset, offset outputs and then 
# detecting onsets with frame and offset outputs
def note_detection_with_onset_offset_regress(frame_output, onset_output, onset_shift_output, offset_output, offset_shift_output, velocity_output,frame_threshold):
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:        # onset found
            if bgn:
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:     # if onset found, searching for offset
            if frame_output[i] <= frame_threshold and not frame_disappear:
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                offset_occur = i        # offset found

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    fin = offset_occur
                else:
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    # sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples

# loading audio 
def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,dtype=np.float32, res_type='kaiser_best'):
    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = int(np.round(sr_native * offset)) * n_channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))* n_channels)
        n = 0
        for frame in input_file:
            frame = librosa.core.audio.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)
            if n < s_start:     # offset is after current frame
                continue
            if s_end < n_prev:
                break
            if s_end < n:       # file end in current frame, crop the frame
                frame = frame[:s_end - n_prev]
            if n_prev <= s_start <= n:      # beginning frame
                frame = frame[(s_start - n_prev):]
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.core.audio.to_mono(y)
        if sr is not None:
            y = librosa.core.audio.resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)
        else:
            sr = sr_native
    y = np.ascontiguousarray(y, dtype=dtype)       # into numpy array of dtype (floats)
    return (y, sr)