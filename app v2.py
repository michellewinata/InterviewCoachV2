"""
InterviewCoach Streamlit - live transcript edition.

Requires the original project files:
  analyzer.py
  asr_engine.py
  questions.py

Run:
  streamlit run app_streamlit.py

Browser microphone is captured with streamlit-webrtc. Audio is sent to the
existing StreamingTranscriber in short chunks and the transcript is refreshed
while the session is active.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from analyzer import analyze, format_report
from asr_engine import SAMPLE_RATE, StreamingTranscriber
from questions import QUESTIONS

LANGUAGES = {"Indonesia": "id", "English": "en"}
MODEL_OPTIONS = ["tiny", "base", "small", "medium"]
CHUNK_SECONDS = 2.0


class LiveAudioProcessor:
    """WebRTC audio callback: capture browser mic frames into a thread-safe queue."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # PyAV usually provides planar or packed audio. Convert to mono float32.
        audio = frame.to_ndarray()
        if audio.ndim == 2:
            # Planar audio often has shape (channels, samples).
            if audio.shape[0] <= 8:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=1)

        if np.issubdtype(audio.dtype, np.integer):
            max_value = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max_value
        else:
            audio = audio.astype(np.float32, copy=False)

        self.audio_queue.put((audio.copy(), int(frame.sample_rate or SAMPLE_RATE)))
        return frame


@st.cache_resource(show_spinner=False)
def load_transcriber(model_size: str):
    return StreamingTranscriber(model_size=model_size)


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio.astype(np.float32, copy=False)
    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(int(from_rate), int(to_rate))
    return resample_poly(audio, to_rate // divisor, from_rate // divisor).astype(np.float32)


def save_session(audio, question, language, model_size, transcript, result, engine):
    os.makedirs("sessions", exist_ok=True)
    name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    wav_path = os.path.join("sessions", name + ".wav")
    json_path = os.path.join("sessions", name + ".json")
    sf.write(wav_path, audio, SAMPLE_RATE)

    data = {
        "question": question["text"],
        "language": language,
        "model": model_size,
        "transcript": transcript,
        "analysis": result,
        "latency": engine.latency_summary(),
        "words": engine.get_words(),
