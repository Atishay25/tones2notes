import os
import librosa
import argparse
from pathlib import Path
from transcribe import PianoTranscription
from synthviz import create_video

# reference for video making : https://github.com/minzwon/sota-music-tagging-models/blob/master/predict.py

parser = argparse.ArgumentParser()
parser.add_argument('--audio_file', type=str, required=True, help='Path to audio file')
args = parser.parse_args()
audioFile = (args.audio_file).split('/')[-1]
midiFile =  audioFile.split('.')[0] + '_transcipted.mid'
videoFile = audioFile.split('.')[0] + '_transcripted.mp4'
vf = os.path.join(Path.cwd(), 'results', videoFile)

tc = PianoTranscription('CRNN_Conditioning', device='cuda', checkpoint_path='./model_checkpoints/CRNN_Conditioning_regressedLoss.pth')
audio, _ = librosa.core.load(args.audio_file, sr=16000)
tc.transcribe(audio, midiFile)
create_video(input_midi=midiFile, video_filename=vf)
print(f"Created Video of size {os.path.getsize(vf)} bytes at path {vf}")
print(Path(vf))