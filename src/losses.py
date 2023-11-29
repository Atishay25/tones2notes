import torch

# Loss function to be used during Training


# Binary Cross Entropy Loss
# wherever mask is 0, those indices won't be taken
def bce(output, target, mask):
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    bc_error = torch.sum(matrix * mask) / torch.sum(mask)
    return bc_error

# High-resolution regression loss
# includes onset loss, offset loss, velocity loss and frame loss with Regressed rolls
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss

# onsets and frames system piano note loss
# Does not take regression rolls into account
def onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss

def get_loss_func(loss_type):
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce
    elif loss_type == 'onset_offset_frame_velocity_bce':
        return onset_offset_frame_velocity_bce
    else:
        raise Exception('Incorrect loss_type!')