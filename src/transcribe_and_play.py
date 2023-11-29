import os
from pathlib import Path

import cog
import librosa

from transcribe import PianoTranscription
from synthviz import create_video

# adapted from example: https://github.com/minzwon/sota-music-tagging-models/blob/master/predict.py


#@cog.input("audio_input", type=Path, help="Input audio file")
#def predict(self, audio_input):
#    midi_intermediate_filename = "transcription.mid"
#    video_filename = os.path.join(Path.cwd(), "output.mp4")
#    audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
#    # Transcribe audio
#    self.transcriptor.transcribe(audio, midi_intermediate_filename)
#    # 'Visualization' output option
#    create_video(
#        input_midi=midi_intermediate_filename, video_filename=video_filename
#    )
#    print(
#        f"Created video of size {os.path.getsize(video_filename)} bytes at path {video_filename}"
#    )
#    # Return path to video
#    return Path(video_filename)
    
tc = PianoTranscription('CCNN', device='cuda', checkpoint_path='./../checkpoints/main/CCNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=8/25000_iterations.pth')
midifile = "op_wano.mid"
vf = os.path.join(Path.cwd(), "output_op_wano.mp4")
audio, _ = librosa.core.load("./../op_wano.mp3", sr=16000)
tc.transcribe(audio, midifile)
create_video(input_midi=midifile, video_filename=vf)
print(f"OHOHOHOHOHOH Created Video of size {os.path.getsize(vf)} bytes at path {vf}")
print(Path(vf))

    

    
