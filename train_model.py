import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
import json

import argparse
import tensorflow as tf
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
gpus = tf.config.list_physical_devices('GPU')
#gpus = tf.config.get_visible_devices('GPU')
print(gpus)
if gpus:
    for gpu in gpus:
        # Set memory growth for each GPU
        tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.experimental.set_memory_growth(gpus[3], True)
from keras import callbacks
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from src.dataflow import dataflow
from src.config import HarmonicNum
from src.models.mymodel import sparse_loss, mctl_loss
from src.utils import ModelInfo
from src.label_type import BaseLabelType

dataset_paths = {
    "MAPS" : "data/MAPS"
}

dataflow_cls = {
    "MAPS" : dataflow.MAPSDataflow
}

default_model_path = "./model"

def create_parser():
    parser = argparse.ArgumentParser(
        "Polyphonic music transcription project done by MCTLab, IIS Sinica.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("dataset", help="One of the Maestro, MusicNet, or Maps", choices=["Maestro", "MusicNet", "MAPS"],
                        type=str)
    parser.add_argument("output_model_name", help="Name for trained model. If --input-model is given, \
                        then this flag has no effect.", type=str)
    parser.add_argument("-d", "--dataset-path", help="Path to the root of the dataset that has preprocessed feature",
                        type=str)
    parser.add_argument("--use-harmonic", help="Wether to use HCFP feature to train the model",
                        action="store_true")
    parser.add_argument("--label-type", help="Type of label to be transformed to",
                        type=str, choices=["frame", "frame_onset", "multi_instrument_frame", "multi_instrument_note"], 
                        default="frame_onset")
    parser.add_argument("--loss-function", help="Use specific type of loss functions.",
                         type=str, default="bce", choices=["focal", "smooth", "bce"])
    # Channel types
    #   0: Z
    #   1: Spec
    #   2: GCoS
    #   3: Ceps
    parser.add_argument("-c", "--channels", help="Use specific channels of feature to train ",
                        type=int, nargs="+", default=[1, 3]) 
    parser.add_argument("--use-ram", help="Wether to load the whole dataset into ram",
                        action="store_true")
    parser.add_argument("-t", "--timesteps", help="Time width for each input feature",
                        type=int, default=256)
    
    # Arguments about the training progress
    parser.add_argument("-e", "--epochs", help="Number of epochs to train",
                        type=int, default=10)
    parser.add_argument("-s", "--steps", help="Training steps for each epoch",
                        type=int, default=2000)
    parser.add_argument("-vs", "--val-steps", help="Validation steps",
                        type=int, default=500)
    parser.add_argument("-i", "--input-model", help="If given, then will continue to train on a pre-trained model")
    parser.add_argument("-b", "--train-batch-size", help="Batch size for training phase",
                        type=int, default=8)
    parser.add_argument("-vb", "--val-batch-size", help="Batch size for validation phase",
                        type=int, default=16)
    parser.add_argument("--early-stop", help="Early stop the training after given # epochs",
                        type=int, default=4)
    return parser

def train(
        model, 
        generator_train, 
        generator_val,
        epochs=1,
        callbacks=None, 
        steps=6000, 
        v_steps=3000
    ):
    
    model.fit(
        x=generator_train, 
        validation_data=generator_val,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_steps=v_steps,
        callbacks=callbacks,
        max_queue_size=100,
        batch_size=8,
        use_multiprocessing=False
    )

    return model


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    if args.dataset not in dataflow_cls:
        raise TypeError
    
    # Hyper parameters that will be stored for future reuse
    hparams = {}

    # Parameters that will be passed to dataflow
    df_params = {}

    # Information about the model (load/store/create)
    minfo = ModelInfo()
    
    # Handling root path to the dataset
    d_path = dataset_paths[args.dataset]
    if args.dataset_path is not None:
        assert(os.path.isdir(args.dataset_path))
        d_path = args.dataset_path
    
    # Number of channels that model need to know about
    minfo.input_channels = args.channels
    
    # Type of feature to use
    minfo.feature_type = "CFP"
    
    # Output model name
    minfo.name = args.output_model_name
    
    # Feature length on time dimension
    minfo.timesteps = args.timesteps

    # Label type
    minfo.label_type = args.label_type
    l_type = BaseLabelType(args.label_type, timesteps=minfo.timesteps)
#
    ## Number of output classes
    minfo.output_classes = l_type.get_out_classes()

    # Continue to fine-tune on a pre-trained model
    if args.input_model is not None:
        # load configuration from previous training stage
        model = minfo.load_model(args.input_model)

    # Check whether to use harmonic feature
    if args.use_harmonic:
        tmp_ch = []
        for ch in minfo.input_channels:
            tmp_ch += list(range((ch-1)*HarmonicNum, ch*HarmonicNum))
        minfo.input_channels = tmp_ch
        minfo.feature_type = "HCFP"
    
    df_params["b_sz"]      = args.train_batch_size
    df_params["phase"]     = "train"
    df_params["use_ram"]   = args.use_ram
    df_params["channels"]  = minfo.input_channels
    df_params["timesteps"] = minfo.timesteps
    df_params["out_classes"]  = minfo.output_classes
    df_params["dataset_path"] = d_path
    df_params["label_conversion_func"] = l_type.get_conversion_func()
    print(args.train_batch_size, args.val_batch_size)
    print("Loading training data")
    df_cls = dataflow_cls[args.dataset]
    train_df = df_cls(**df_params)


    print(train_df.feature_num)
    for i in train_df.features:
        print(i)


    print("Loading validation data")
    df_params["b_sz"]  = args.val_batch_size
    df_params["phase"] = "val"
    val_df = df_cls(**df_params)


    hparams["channels"]       = minfo.input_channels
    hparams["timesteps"]      = minfo.timesteps
    hparams["feature_type"]   = minfo.feature_type
    #hparams["output_classes"] = minfo.output_classes
    
    print("Creating/loading model")
    # Create model
    if args.input_model is None:
        #model = minfo.create_model(model_type="attn")
        model = minfo.create_model(model_type="aspp")
    # Loss function
    loss_func_mapping = {
        "focal": sparse_loss,
        #"smooth": lambda label, pred: mctl_loss(label, pred, out_classes=minfo.output_classes),
        "bce": tf.keras.losses.BinaryCrossentropy()
    }
    loss_func = loss_func_mapping[args.loss_function]

    # Store other training information
    minfo.dataset = args.dataset
    minfo.epochs = args.epochs
    minfo.steps = args.steps
    minfo.loss_function = args.loss_function
    minfo.train_batch_size = args.train_batch_size
    minfo.val_batch_size = args.val_batch_size
    minfo.early_stop = args.early_stop

    # Save model and configurations
    print(minfo)
    minfo.save_model(model, default_model_path)
    
    # Use multi-gpu to train the model
    #if False:
    #    para_model = multi_gpu_model(model, gpus=2, cpu_merge=False)
    #    para_model.compile(optimizer="adam", loss={'prediction': loss_func}, metrics=['accuracy'])
    #    model = para_model
    #else:
    #    model.compile(optimizer="adam", loss={'prediction': loss_func}, metrics=['accuracy'])
    model.compile(optimizer="adam", loss={'prediction': loss_func}, metrics=['accuracy'])
    # create callbacks
    earlystop   = callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop)
    checkpoint  = callbacks.ModelCheckpoint(os.path.join(default_model_path, minfo.name, "weights.h5"), 
                                            monitor="val_loss", save_best_only=False, save_weights_only=True)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join("tensorboard", args.output_model_name),
                                        write_images=True)
    callback_list = [checkpoint, earlystop, tensorboard]
    
    print("Start training")
    # Start training
    train(model, train_df, val_df,
          epochs    = args.epochs,
          callbacks = callback_list,
          steps     = args.steps,
          v_steps   = args.val_steps)