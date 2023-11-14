import os
import csv
import glob
import math
import pickle
import librosa
import logging
import numpy as np
import pretty_midi
from tqdm import trange
from scipy import signal, fftpack
import h5py
import soundfile as sf

from keras.models import load_model
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src.config import HarmonicNum

class LabelFmt:
    def __init__(self, start_time, end_time, note, instrument=1,  start_beat=0., end_beat=10., note_value=""):
        """
        Instrument number
            1  Piano
            7  Harpsichord
            41 Violin
            42 Viola
            43 Cello
            44 Contrabass
            61 Horn     
            69 Oboe
            71 Bassoon
            72 Clarinet
            74 Flute
        """
        self.start_time = start_time # in second, float
        self.end_time = end_time # in second, float
        self.instrument = instrument # piano, int
        self.note = note # midi number, int
        self.start_beat = start_beat # float
        self.end_beat = end_beat # float
        self.note_value = note_value 

    def get(self):
        return [self.start_time, self.end_time, self.instrument, self.note, self.start_beat, self.end_beat, self.note_value]
    

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, np.round(N/2).astype('int'), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
                            
    tfr = abs(fftpack.fft(tfr, n=N, axis=0))  
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/(q+1e-9)
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    
    tfrL = np.dot(freq_band_transformation[:, :len(ceps)], ceps)
    return tfrL, central_freq

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)


    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs*tc)
                ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc/fr)
                tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)
    
    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]
    

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    q = np.arange(HighQuefIdx)/float(fs)
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 


def fetch_harmonic(
        data, 
        cenf, 
        ith_har, 
        start_freq=27.5, 
        num_per_octave=48, 
        is_reverse=False
    ):
    
    ith_har += 1
    if ith_har != 0 and is_reverse:
        ith_har = 1/ith_har
    
    #harmonic_series = [12, 19, 24, 28, 31]     
    bins_per_note = int(num_per_octave / 12)           
    total_bins = int(bins_per_note * 88)
    
    hid = min(range(len(cenf)), key=lambda i: abs(cenf[i]-ith_har*start_freq))
    
    harmonic = np.zeros((total_bins, data.shape[1]))
    upper_bound = min(len(cenf)-1, hid+total_bins)
    harmonic[:(upper_bound-hid)] = data[hid:upper_bound]
    
    return harmonic


def parallel_extract(x, samples, MaxSample, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    freq_width = MaxSample * Hop
    Round = np.ceil(samples/MaxSample).astype('int')
    tmpL0, tmpLF, tmpLQ, tmpZ = {}, {}, {}, {}
    
    max_workers = min(os.cpu_count(), Round)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {}
        for i in range(Round):
            tmpX = x[i*freq_width:(i+1)*freq_width]
            future = executor.submit(CFP_filterbank, tmpX, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
            future_to_segment[future] = i

        for future in concurrent.futures.as_completed(future_to_segment):
            seg_id = future_to_segment[future]
            try:
                tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = future.result()
                tmpL0[seg_id] = tfrL0
                tmpLF[seg_id] = tfrLF
                tmpLQ[seg_id] = tfrLQ
                tmpZ[seg_id] = tfrLF*tfrLQ
            except Exception as exc:
                print("Something generated an exception: {}".format(exc))
                raise exc
    
    return tmpL0, tmpLF, tmpLQ, tmpZ, f, q, t, CenFreq

def feature_extraction(
        filename,
        hop=0.02, # in seconds
        w=7939,
        fr=2.0,
        fc=27.5,
        tc=1/4487.0,
        g=[0.24, 0.6, 1],
        NumPerOctave=48,
        Down_fs=44100
    ):
                       
    x, fs = sf.read(filename)
    if len(x.shape)>1:
       x = np.mean(x, axis = 1)
    x = signal.resample_poly(x, Down_fs, fs)
    fs = Down_fs # sampling frequency
    Hop = round(Down_fs*hop)
    x = x.astype('float32')
    h = signal.blackmanharris(w) # window size
    g = np.array(g)

    MaxSample = 2000
    samples = np.floor(len(x)/Hop).astype('int')
    print("# Sample: ", samples)
    if samples > MaxSample:
        tmpL0, tmpLF, tmpLQ, tmpZ, f, q, t, CenFreq = parallel_extract(x, samples, MaxSample, fr, fs, Hop, h, fc, tc, g, NumPerOctave)

        tfrL0 = tmpL0.pop(0)
        tfrLF = tmpLF.pop(0)
        tfrLQ = tmpLQ.pop(0)
        Z = tmpZ.pop(0)
        rr = len(tmpL0)
        for i in range(1, rr+1, 1):
            tfrL0 = np.concatenate((tfrL0, tmpL0.pop(i)), axis=1)
            tfrLF = np.concatenate((tfrLF, tmpLF.pop(i)), axis=1)
            tfrLQ = np.concatenate((tfrLQ, tmpLQ.pop(i)), axis=1)
            Z = np.concatenate((Z, tmpZ.pop(i)), axis=1)
    else:
        tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
        Z = tfrLF * tfrLQ

    return Z, tfrL0, tfrLF, tfrLQ, t, CenFreq, f

def process_feature_song_list(
        dataset_name, 
        song_list,
        harmonic=False,
        num_harmonic=0
    ):

    fs = 44100
    if harmonic:
        freq_range = [1.0, fs/2]
    else:
        freq_range = [27.5, 4487.0]
    hdf_out = h5py.File(dataset_name+".hdf", "w")
    
    for idx, song in enumerate(song_list):
        print("Extracting({}/{}): {}".format(idx+1, len(song_list), song))
        
        out  = feature_extraction(song, fc=freq_range[0], tc=(1/freq_range[1]), Down_fs=fs)
        cenf = out[5]
        #z, spec, gcos, ceps, cenf = out[0:5]

        piece = np.transpose(np.array(out[0:4]), axes=(2, 1, 0))
        
        if harmonic:
            # Harmonic spectrum
            har = []
            for i in range(num_harmonic+1):
                har.append(fetch_harmonic(out[1], cenf, i))
            har_s = np.transpose(np.array(har), axes=(2, 1, 0))
            
            # Harmonic GCoS
            har = []
            for i in range(num_harmonic+1):
                har.append(fetch_harmonic(out[2], cenf, i))
            har_g = np.transpose(np.array(har), axes=(2, 1, 0))

            # Harmonic cepstrum
            har = []
            for i in range(num_harmonic+1):
                har.append(fetch_harmonic(out[3], cenf, i, is_reverse=True))
            har_c = np.transpose(np.array(har), axes=(2, 1, 0))
            
            piece = np.dstack((har_s, har_g, har_c))
        
        key = os.path.basename(song)
        key = key.replace(".wav", "")
        hdf_out.create_dataset(key, data=piece, compression="gzip", compression_opts=5)
    
    hdf_out.close()  


class BaseFeatureExt:
    def __init__(self, wav_path:list, label_path:list, label_ext:str, save_path="./train", piece_per_file=40, file_prefix="train", harmonic=False):
        self.file_prefix = file_prefix
        self.wav_path = wav_path
        self.label_path = label_path
        self.num_per_file = piece_per_file
        self.label_ext = label_ext
        self.harmonic = harmonic
        self.save_path = save_path
        self.t_unit = 0.02

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def load_label(self, file_path, sample_rate=44100):
        raise NotImplementedError
        
    def process(self):
        with open(os.path.join(self.save_path, "SongList.csv"), "w", newline='') as csvList:
            writer = csv.DictWriter(csvList, fieldnames=["File name", "id", "Save path"])
            writer.writeheader()
            
            for i, (wav_paths, label_paths)  in enumerate(self.batch_generator()):
                sub_name = os.path.join(self.save_path, ("{}_{}_{}".format(self.file_prefix, self.num_per_file, str(i+1))))

                # Process audio features
                process_feature_song_list(sub_name, wav_paths, harmonic=self.harmonic, num_harmonic=HarmonicNum)
        
                # Process labels
                self.process_labels(sub_name, label_paths)
                
                # write record to file
                for wav in wav_paths:
                    writer.writerow(
                        {
                            "File name": sub_name,
                            "id": os.path.basename(wav),
                            "Save path": self.save_path
                        }
                    )

    def batch_generator(self):
        # Parse audio files
        all_wavs = []
        for path in self.wav_path:
            all_wavs += glob.glob(os.path.join(path, "*.wav"))
        # Parse label files
        name_path_map = {}
        for path in self.label_path:
            files = glob.glob(os.path.join(path, "*{}".format(self.label_ext)))
            for ff in files:
                name = self.name_transform(os.path.basename(ff))
                name_path_map[name] = ff
        names = [os.path.basename(wav).replace(".wav", "") for wav in all_wavs]
        all_labels = []
        for name in names:
            if name not in name_path_map:
                logging.error("Cannot found corresponding groundtruth file: %s", name)
                continue
            all_labels.append(name_path_map[name])

        # Start to generate batch
        iters = math.ceil(len(all_wavs)/self.num_per_file)
        for i in range(iters):
            print("Iters: {}/{}".format(i+1, iters))
            lower_b = i*self.num_per_file
            upper_b = (i+1)*self.num_per_file
            yield all_wavs[lower_b:upper_b], all_labels[lower_b:upper_b]

    def name_transform(self, original):
        """
        Transform a ground truth file name to the correponding audio name (without extension).
        """
        return original.replace(self.label_ext, "")
    
    def process_labels(self, sub_name, files):
        '''
        Stored structure:
            labels: Frame x Pitch x Instrument x 2
                Frame: Each frame is <t_unit> second long
                Pitch: Maximum length is 88, equals to the keys of piano
                Instrument: Available types please refer to LabelFormat.py. 
                    It's in Dict type. Only if there exists a instrument will be write to the Dict key-value pair.
                Last dimension: Type of list. [onset_prob, offset_prob]
        '''
        lowest_pitch = librosa.note_to_midi('A0')
        highest_pitch = librosa.note_to_midi('C8')
        pitches = highest_pitch - lowest_pitch + 1
            
        labels = {}
        contains = {} # To summarize a specific instrument is included in pieces
        for idx in trange(len(files), leave=False):
            gt_path = files[idx]
            content, last_sec = self.load_label(gt_path)
            frame_num = int(round(last_sec, 2)/self.t_unit)
            
            label = [{} for i in range(frame_num)]
            for cc in content:
                start_time, end_time, instrument, note, start_beat, end_beat, note_value = cc.get()
                start_f, end_f = int(round(start_time, 2)/self.t_unit), int(round(end_time, 2)/self.t_unit)
                pitch = note-lowest_pitch

                # Put onset probability to the pitch of the instrument
                onsets_v = 1.0
                onsets_len = 2 # 2 frames long
                ii = 0
                for i in range(start_f, end_f):
                    if pitch not in label[i]:
                        label[i][pitch] = {}
                    label[i][pitch][instrument] = [onsets_v, 0] # [onset_prob, offset_prob]
                    ii += 1
                    if ii > onsets_len:
                        onsets_v /= (ii-onsets_len)
                
                # Put offset probability to the pitch of the instrument
                offset_v = 1.0
                offset_len = 4 # 2 frames long
                ii = 0
                for i in range(end_f-1, start_f, -1):
                    label[i][pitch][instrument][1] = offset_v
                    ii += 1
                    if ii >= offset_len:
                        offset_v /= (ii-onsets_len) 

                    # Below are some statistical information generation                    
                    # instrument is contained in pieces
                    if instrument not in contains:
                        contains[instrument] = []
                    name = os.path.basename(gt_path)
                    name = name.rsplit(".", 1)[0]
                    if name not in contains[instrument]:
                        contains[instrument].append(name)
                        contains[instrument].append(idx)
            
            key = self.name_transform(os.path.basename(gt_path))
            labels[key] = label

        pickle.dump(labels, open(sub_name+".pickle", "wb"), pickle.HIGHEST_PROTOCOL)
        return contains
    

class MAPSFeatureExt(BaseFeatureExt):
    def load_label_midi(self, file_path, **kwargs):
        midi = pretty_midi.PrettyMIDI(file_path)
        inst = midi.instruments[0]

        content = []
        last_sec = 0
        for note in inst.notes:
            onset = note.start
            offset = note.end
            pitch = note.pitch
            content.append(LabelFmt(onset, offset, pitch))
            last_sec = max(last_sec, offset)

        return content, last_sec


    def load_label(self, file_path, sample_rate=44100):
        with open(file_path, "r") as ll_file:
            lines = ll_file.readlines()

        content = []
        last_sec = 0
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                continue
            onset, offset, note = lines[i].split("\t")
            onset, offset, note = float(onset), float(offset), int(note.strip())
            content.append(LabelFmt(onset, offset, note)) 
            last_sec = max(last_sec, offset)
        
        return content, last_sec