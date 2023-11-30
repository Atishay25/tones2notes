import numpy as np
import torch

# helper functions
def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    return x.to(device)

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

# function to input the data into model
def forward_dataloader(model, dataloader, batch_size, return_target=True):
    output_dict = {}
    device = next(model.parameters()).device
    for n, batch_data_dict in enumerate(dataloader):
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
        for key in batch_output_dict.keys():
            if '_list' not in key:
                append_to_dict(output_dict, key, 
                    batch_output_dict[key].data.cpu().numpy())
        if return_target:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type or 'reg_distance' in target_type or \
                    'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, 
                        batch_data_dict[target_type])
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict

# function to input data into model in batches
def forward(model, x, batch_size):    
    output_dict = {}
    device = next(model.parameters()).device
    pointer = 0
    while True:
        if pointer >= len(x):
            break
        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
        for key in batch_output_dict.keys():
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict
