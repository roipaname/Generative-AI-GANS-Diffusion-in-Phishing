import os
from pydub import AudioSegment
from TTS.api import TTS
import torch

# -------------------------------
# CONFIG
# -------------------------------
VOICE_FOLDER = "./data/voice_samples"
OUTPUT_FOLDER = "./outputs/voice_output"
TEXT = "I love my boyfriend"
LANGUAGE = "en" 

# Ensure outputs folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# STEP 1: Convert all .opus to .wav
# -------------------------------
wav_files = []
for file in os.listdir(VOICE_FOLDER):
    path = os.path.join(VOICE_FOLDER, file)
    if file.endswith(".opus"):
        # convert to WAV
        sound = AudioSegment.from_file(path)
        wav_path = path.replace(".opus", ".wav")
        sound.set_frame_rate(22050).set_channels(1).export(wav_path, format="wav")
        print("Converted:", wav_path)
        wav_files.append(wav_path)
    elif file.endswith(".wav"):
        wav_files.append(path)

# -------------------------------
# STEP 2: Combine multiple .wav files into one speaker sample
# -------------------------------
combined = AudioSegment.empty()
for w in wav_files:
    combined += AudioSegment.from_file(w)

combined_path = os.path.join(VOICE_FOLDER, "combined_speaker.wav")
combined.export(combined_path, format="wav")
print("✅ Combined speaker sample:", combined_path)

# -------------------------------
# STEP 3: Load TTS model
# -------------------------------
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)

# -------------------------------
# STEP 4: Generate speech with speaker_wav and attention mask
# -------------------------------
out_path = os.path.join(OUTPUT_FOLDER, "cloned_speech.wav")

# Force torch attention mask usage (suppress warnings)
with torch.no_grad():
    tts.tts_to_file(
        text=TEXT,
        speaker_wav=combined_path,
        file_path=out_path,
        language=LANGUAGE
    )

print("✅ Speech generated:", out_path)
