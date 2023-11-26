import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import argparse
import torch
import time
import h5py
import pickle
import config

from post_process import (ScoreCalculator)
from processing import (create_folder, get_filename, traverse_folder, TargetProcessor)
from transcribe import PianoTranscription

def inference(args):
    workspace = args.workspace
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    augmentation = args.augmentation
    dataset = args.dataset
    split = args.split
    post_processor_type = args.post_processor_type
    
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note

    hdf5s = os.path.join(workspace, 'hdf5s', dataset)
    probs = os.path.join(workspace, 'probs',
        'model_type={}'.format(model_type),
        'augmentation={}'.format(augmentation),
        'dataset={}'.format(dataset),
        'split={}'.format(split))
    create_folder(probs)

    transcriptor = PianoTranscription(model_type, device=device,
        checkpoint_path=checkpoint_path, segment_samples=segment_samples,
        post_processor_type=post_processor_type)
    
    (names, paths) = traverse_folder(hdf5s)

    for path in paths:
        with h5py.File(path, 'r') as hf:
            if hf.attrs['split'].decode() == split:

                audio = (hf['waveform'][:] / 32767.).astype(np.float32)
                midi_events = [e.decode() for e in hf['midi_event'][:]]
                midi_events_time = hf['midi_event_time'][:]

                targets_processor = TargetProcessor(
                    segment_seconds=len(audio)/sample_rate,
                    frames_per_second=frames_per_second, begin_note=begin_note, classes_num=classes_num
                )

                (target_dict, note_events) = targets_processor.process(start_time=0,
                                                midi_events_time=midi_events_time,
                                                midi_events=midi_events)
                
                ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                ref_velocity = np.array([event['velocity'] for event in note_events])

                transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
                output_dict = transcribed_dict['output_dict']

                total_dict = output_dict.copy()
                total_dict['frame_roll'] = target_dict['frame_roll']
                total_dict['ref_on_off_pairs'] = ref_on_off_pairs
                total_dict['ref_midi_notes'] = ref_midi_notes
                total_dict['ref_velocity'] = ref_velocity

                prob_path = os.path.join(probs, '{}.pkl'.format(get_filename(path)))
                create_folder(os.path.dirname(prob_path))
                pickle.dump(total_dict, open(prob_path, 'wb'))

def calculate_metrics(args, thresholds=None):
    workspace = args.workspace
    model_type = args.model_type
    augmentation = args.augmentation
    dataset = args.dataset
    split = args.split
    post_processor_type = args.post_processor_type

    hdf5s = os.path.join(workspace, 'hdf5s', dataset)
    probs = os.path.join(workspace, 'probs',
        'model_type={}'.format(model_type),
        'augmentation={}'.format(augmentation),
        'dataset={}'.format(dataset),
        'split={}'.format(split))
    
    score_calculaotr = ScoreCalculator(hdf5s, probs, split=split, post_processor_type=post_processor_type)

    if not thresholds:
        thresholds = [0.3, 0.3, 0.3]
    
    start_time = time.time()
    stats_dict = score_calculaotr.metrics(thresholds)
    print('Time: {:.3f}'.format(time.time() - start_time))

    for key in stats_dict.keys():
        print('{}: {:.4f}'.format(key, np.mean(stats_dict[key])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_infer_prob = subparsers.add_parser('infer_prob')
    parser_infer_prob.add_argument('--workspace', type=str, required=True)
    parser_infer_prob.add_argument('--model_type', type=str, required=True)
    parser_infer_prob.add_argument('--augmentation', type=str, required=True)
    parser_infer_prob.add_argument('--checkpoint_path', type=str, required=True)
    parser_infer_prob.add_argument('--dataset', type=str, required=True, choices=['maestro', 'maps'])
    parser_infer_prob.add_argument('--split', type=str, required=True)
    parser_infer_prob.add_argument('--post_processor_type', type=str, default='regression')
    parser_infer_prob.add_argument('--cuda', action='store_true', default=False)

    parser_metrics = subparsers.add_parser('calculate_metrics')
    parser_metrics.add_argument('--workspace', type=str, required=True)
    parser_metrics.add_argument('--model_type', type=str, required=True)
    parser_metrics.add_argument('--augmentation', type=str, required=True)
    parser_metrics.add_argument('--dataset', type=str, required=True, choices=['maestro', 'maps'])
    parser_metrics.add_argument('--split', type=str, required=True)
    parser_metrics.add_argument('--post_processor_type', type=str, default='regression')

    args = parser.parse_args()

    if args.mode == 'infer_prob':
        inference(args)

    elif args.mode == 'calculate_metrics':
        calculate_metrics(args)

    else:
        raise Exception('Incorrct argument!')