import torch
from TTS.api import TTS
import simpleaudio as sa

# Preload model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

# Preload combined speaker WAV
SPEAKER_WAV = "./data/voice_samples/combined_speaker.wav"

def speak_alert(text: str):
    out_path = "./outputs/cloned_speech.wav"
    with torch.no_grad():
        tts.tts_to_file(text=text, speaker_wav=SPEAKER_WAV, file_path=out_path, language="en")
    
    # Play audio
    wave_obj = sa.WaveObject.from_wave_file(out_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


