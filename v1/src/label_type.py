import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

import sys
sys.path.append("./")

from src.utils import label_conversion


class BaseLabelType:
    def __init__(self, mode, timesteps=128):
        self.mode = mode
        self.timesteps = timesteps
        self.l_conv = lambda label, tid, **kwargs: label_conversion(label, tid, timesteps=timesteps, **kwargs)

        self.mode_mapping = {
            "frame": {"conversion_func": self.get_frame, "out_classes": 2},
            "frame_onset": {"conversion_func": self.get_frame_onset, "out_classes": 3},
            "frame_onset_offset": {"conversion_func": self.get_frame_onset_offset, "out_classes": 4}
        }
        self._update_mode()
        if mode not in self.mode_mapping:
            raise ValueError(f"Available mode: {self.mode_mapping.keys()}. Provided: {mode}")
    
    # Override this function if you have implemented a new mode
    def _update_mode(self):
        pass

    def get_conversion_func(self):
        return self.mode_mapping[self.mode]["conversion_func"]

    def get_out_classes(self)->int:
        return self.mode_mapping[self.mode]["out_classes"]

    def get_frame(self, label, tid)->np.ndarray:
        return self.l_conv(label, tid, mpe=True)

    def get_frame_onset(self, label, tid)->np.ndarray:
        frame = self.get_frame(label, tid)
        onset = self.l_conv(label, tid, onsets=True, mpe=True)[:,:,1]

        frame[:,:,1] -= onset
        frm_on = np.dstack([frame, onset])
        frm_on[:,:,0] = 1-np.sum(frm_on[:,:,1:], axis=2)

        return frm_on

    def get_frame_onset_offset(self, label, tid)->np.ndarray:
        frm_on = self.get_frame_onset(label, tid)
        offset = self.l_conv(label, tid, offsets=True)

        tmp = frm_on[:,:,1] - frm_on[:,:,2] - offset
        tmp[tmp>0] = 0
        offset += tmp
        frm_on[:,:,1] = frm_on[:,:,1] - frm_on[:,:,2] - offset

        frm_on_off = np.dstack([frm_on, offset])
        frm_on_off[:,:,0] = 1-np.sum(frm_on_off[:,:,1:], axis=2)

        return frm_on_off
