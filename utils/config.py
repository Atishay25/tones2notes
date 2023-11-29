sample_rate = 16000         # Downsampling to 16kHz
classes_num = 88            # Number of Piano Notes, Piano Notes are in range [21,108]
begin_note = 21             # MIDI note of A0, the lowest piano Note
segment_seconds = 10.0	    # Training segment duration of 10 seconds
hop_seconds = 1.0         
frames_per_second = 100
velocity_scale = 128