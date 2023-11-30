import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time

import torch 
import torch.optim as optim
import torch.utils.data

import config
from losses import get_loss_func 
from data_generator import MapsDataset, Sampler, TestSampler, collate_fn
from eval import SegmentEvaluator
from processing import StatisticsContainer, create_folder
from pytorch_utils import move_data_to_device
from models import Net, CCNN, CRNN, CRNN_Conditioning


def train(args):
    workspace = args.workspace
    model_type = args.model_type
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    loss_type = args.loss_type
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8
    loss_func= get_loss_func(loss_type)         # can change loss function if want
    hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')
    checkpoints_dir = os.path.join(workspace, 'checkpoints', model_type,
        'loss_type={}'.format(loss_type), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    statistics_path = os.path.join(workspace, 'statistics', model_type,
        'loss_type={}'.format(loss_type), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    if 'cuda' in str(device):
        device = 'cuda'
    else:
        device = 'cpu'

    Model = eval(model_type)        
    model  = Model(frames_per_second=frames_per_second, classes_num=classes_num)

    # Dataset objects
    train_dataset = MapsDataset(hdf5s_dir=hdf5s_dir, segment_seconds=segment_seconds, frames_per_second=frames_per_second, max_note_shift=max_note_shift)
    evaluate_dataset = MapsDataset(hdf5s_dir=hdf5s_dir, segment_seconds=segment_seconds, frames_per_second=frames_per_second, max_note_shift=0)

    # Samplers to generate batches of segments
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, batch_size=batch_size, mini_data=mini_data)
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, batch_size=batch_size, mini_data=mini_data)
    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, batch_size=batch_size, mini_data=mini_data)
    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, batch_size=batch_size, mini_data=mini_data)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    
    evaluator = SegmentEvaluator(model, batch_size)                 # evaluation on validation sets
    statistics_container = StatisticsContainer(statistics_path)     # stores evaluation stats during training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)      
    
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', 
            model_type, 'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']
    else:
        iteration = 0
    model = torch.nn.DataParallel(model)
    if 'cuda' in str(device):
        model.to(device)
    train_bgn_time = time.time()
    for batch_data_dict in train_loader:
        if iteration % 500 == 0:            # Evaluate Model at intervals on train, validation, test data
            print('--------------------------------')
            print('Iteration', iteration)
            train_fin_time = time.time()
            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            validate_statistics = evaluator.evaluate(validate_loader)
            test_statistics = evaluator.evaluate(test_loader)
            print('Train Stats', evaluate_train_statistics)
            print('Val Stats', validate_statistics)
            print('Test Stats', test_statistics)
            statistics_container.append(iteration, evaluate_train_statistics, data_type='train')        # Store the results in a file for stats
            statistics_container.append(iteration, validate_statistics, data_type='validation')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            print('Train time: {:.3f} s, validate time: {:.3f} s'.format(train_time, validate_time))
            train_bgn_time = time.time()
        if iteration % 1000 == 0:                # Saving model at regular iterations for checkpoints
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict()
            }
            checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(iteration))
            torch.save(checkpoint, checkpoint_path)
            print('Model saved to ', checkpoint_path)
        
        if iteration % reduce_iteration == 0 and iteration > 0:         # reducing learning rate at higher iterations
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        for key in batch_data_dict.keys():        # Move data to device
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
         
        model.train()
        batch_output_dict = model(batch_data_dict['waveform'])
        loss = loss_func(model, batch_output_dict, batch_data_dict)
        print('Iteration: ',iteration, '\tLoss: ',loss.item())
        optimizer.zero_grad()       
        loss.backward()             # back prop
        optimizer.step()
        if iteration == early_stop:         # stop the loop
            break
        iteration += 1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True, choices=['CCNN','CRNN','CRNN_Conditioning'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True, choices=['regress_onset_offset_frame_velocity_bce','onset_offset_frame_velocity_bce'])
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')
