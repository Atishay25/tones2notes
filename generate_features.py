import os
import argparse

import src.config as conf
from src.config import MAPSDatasetInfo

import src.feature.feature_processor as fp

d_conf = {
    'MAPS':{
        'dataset_info':conf.MAPSDatasetInfo,
        'processor': fp.MAPSFeatureExt
    }
}

def create_parser():
    parser = argparse.ArgumentParser("Feature Processor")
    parser.add_argument("dataset", help="One of Maps, MusicNet, or Maestro", 
                        type=str, choices=["MAPS", "MusicNet", "Maestro", "Su", "Su-10", "URMP", "Bach", "Rhythm"])
    parser.add_argument("dataset_path", help="Path to the downloaded dataset",
                        type=str)
    parser.add_argument("-p", "--phase", help="Generate training feature or testing feature. Default: %(default)s",
                        type=str, default="train", choices=["train", "test"])
    parser.add_argument("-n", "--piece-per-file", help="Number of pieces should be included in one generated file",
                        type=int, default=40)
    parser.add_argument("-s", "--save-path", help="Path to save the generated feature and label files", 
                        type=str, default="./train_feature")
    parser.add_argument("-a", "--harmonic", help="Generate harmonic features",
                        action="store_true")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    d_info = d_conf[args.dataset]["dataset_info"](args.dataset_path)
    proc_cls = d_conf[args.dataset]['processor']

    paths = d_info.train_wavs if args.phase == 'train' else d_info.test_wavs
    wav_paths = [os.path.join(d_info.base_path, path) for path in paths]
    paths = d_info.train_labels if args.phase=="train" else d_info.test_labels
    label_paths = [os.path.join(d_info.base_path, path) for path in paths] 

    processor = proc_cls(
        wav_paths, 
        label_paths, 
        d_info.label_ext,
        save_path=args.save_path,
        file_prefix=args.phase,
        piece_per_file=args.piece_per_file,
        harmonic=args.harmonic
    )
    processor.process()