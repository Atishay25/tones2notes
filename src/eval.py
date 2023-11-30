import numpy as np
from sklearn import metrics
from pytorch_utils import forward_dataloader

# Mean Absolute Error 
def mae(target, output, mask):
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)

# Evaluate each segment based on metrics 
class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, dataloader):
        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        # Frame and onset evaluation
        if 'frame_output' in output_dict.keys():
            statistics['frame_ap'] = metrics.average_precision_score(
                output_dict['frame_roll'].flatten(), 
                output_dict['frame_output'].flatten(), average='macro')
        
        if 'onset_output' in output_dict.keys():
            statistics['onset_macro_ap'] = metrics.average_precision_score(
                output_dict['onset_roll'].flatten(), 
                output_dict['onset_output'].flatten(), average='macro')

        if 'offset_output' in output_dict.keys():
            statistics['offset_ap'] = metrics.average_precision_score(
                output_dict['offset_roll'].flatten(), 
                output_dict['offset_output'].flatten(), average='macro')

        if 'reg_onset_output' in output_dict.keys():        # mask takes those indices where any one of them is nonzero
            mask = (np.sign(output_dict['reg_onset_output'] + output_dict['reg_onset_roll'] - 0.01) + 1) / 2
            statistics['reg_onset_mae'] = mae(output_dict['reg_onset_output'], 
                output_dict['reg_onset_roll'], mask)

        if 'reg_offset_output' in output_dict.keys():
            mask = (np.sign(output_dict['reg_offset_output'] + output_dict['reg_offset_roll'] - 0.01) + 1) / 2
            statistics['reg_offset_mae'] = mae(output_dict['reg_offset_output'], 
                output_dict['reg_offset_roll'], mask)

        if 'velocity_output' in output_dict.keys():
            statistics['velocity_mae'] = mae(output_dict['velocity_output'], output_dict['velocity_roll'] / 128, output_dict['onset_roll'])
            
        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)
        return statistics