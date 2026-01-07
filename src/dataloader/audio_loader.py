from pydub import AudioSegment
import os

input_dir = "./data/voice_samples"

for file in os.listdir(input_dir):
    if file.endswith(".opus"):
        path = os.path.join(input_dir, file)

        # Let ffmpeg auto-detect the input format
        sound = AudioSegment.from_file(path)

        # Convert to WAV with TTS-friendly settings
        out_path = path.replace(".opus", ".wav")
        sound.set_frame_rate(22050).set_channels(1).export(out_path, format="wav")

        print("Converted:", out_path)
