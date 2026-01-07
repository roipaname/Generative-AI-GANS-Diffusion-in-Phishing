# speech_to_text.py
import streamlit as st
import speech_recognition as sr
import tempfile
import wave
import pyaudio

# Create a Streamlit session state to track recording
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
    st.session_state.audio_frames = []

def start_recording():
    """Begin capturing audio."""
    st.session_state.is_recording = True
    st.session_state.audio_frames = []
    st.info("🎙️ Recording... Click 'Stop' when finished.")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    while st.session_state.is_recording:
        data = stream.read(1024)
        st.session_state.audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def stop_recording():
    """Stop the current recording and save it to temp file."""
    if not st.session_state.audio_frames:
        st.warning("No audio captured.")
        return None

    st.session_state.is_recording = False

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wf = wave.open(temp_wav.name, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes(b''.join(st.session_state.audio_frames))
    wf.close()

    st.success("🛑 Recording stopped.")
    return temp_wav.name

def transcribe_audio(file_path):
    """Convert audio file to text using Google Speech Recognition."""
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            st.success("✅ Transcription complete.")
            return text
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand audio.")
            return ""
        except sr.RequestError:
            st.error("⚠️ Speech Recognition API unavailable.")
            return ""
