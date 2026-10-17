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
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return wav_path, json_path


st.set_page_config(page_title="InterviewCoach", page_icon="🎙️", layout="wide")
st.title("🎙️ InterviewCoach")
st.caption("Latihan interview dengan live speech-to-text Bahasa Indonesia / English.")

with st.sidebar:
    st.header("Pengaturan")
    language_name = st.selectbox("Bahasa jawaban", list(LANGUAGES.keys()))
    model_size = st.selectbox(
        "Model Whisper",
        MODEL_OPTIONS,
        index=1,
        help="tiny/base lebih ringan; small/medium lebih berat.",
    )

language_code = LANGUAGES[language_name]
question_list = QUESTIONS[language_code]
question_texts = [q["text"] for q in question_list]
question_index = st.selectbox(
    "Pertanyaan interview",
    range(len(question_texts)),
    format_func=lambda i: question_texts[i],
)
question = question_list[question_index]
st.subheader("Pertanyaan")
st.markdown(f"**{question['text']}**")

st.markdown("### Live transcript")
st.write("Klik START untuk mulai bicara. Transkrip diperbarui saat audio diproses, kira-kira tiap 2 detik.")

# Session state persists across Streamlit reruns.
defaults = {
    "live_audio": [],
    "live_final_text": "",
    "live_partial_text": "",
    "live_started_at": None,
    "live_last_step": 0,
    "live_error": None,
    "live_finished": False,
    "live_model": None,
    "live_language": None,
    "live_question_index": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

ctx = webrtc_streamer(
    key="interviewcoach-live-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=LiveAudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
    audio_receiver_size=1024,
    async_processing=True,
)

# Initialize a fresh transcription session when the WebRTC recorder starts.
if ctx.state.playing and st.session_state.live_started_at is None:
    try:
        with st.spinner(f"Memuat model Whisper '{model_size}'..."):
            engine = load_transcriber(model_size)
            engine.language = language_code
            engine.reset()
        st.session_state.live_started_at = time.time()
        st.session_state.live_audio = []
        st.session_state.live_final_text = ""
        st.session_state.live_partial_text = ""
        st.session_state.live_last_step = 0
        st.session_state.live_error = None
        st.session_state.live_finished = False
        st.session_state.live_model = model_size
        st.session_state.live_language = language_code
        st.session_state.live_question_index = question_index
    except Exception as exc:
        st.session_state.live_error = str(exc)

engine = None
if st.session_state.live_started_at is not None:
    try:
        engine = load_transcriber(st.session_state.live_model or model_size)
        engine.language = st.session_state.live_language or language_code
    except Exception as exc:
        st.session_state.live_error = str(exc)

# Drain browser audio frames, convert to the ASR sample rate, and feed the
# existing streaming transcriber. Do not use sounddevice: the mic is in browser.
if ctx.state.playing and ctx.audio_processor and engine:
    try:
        received = []
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.15)
            received.extend(frames)
        except queue.Empty:
            frames = []

        for frame in received:
            chunk = frame.to_ndarray()
            if chunk.ndim == 2:
                # WebRTC frame layout is commonly (channels, samples).
                chunk = chunk.mean(axis=0) if chunk.shape[0] <= 8 else chunk.mean(axis=1)
            if np.issubdtype(chunk.dtype, np.integer):
                chunk = chunk.astype(np.float32) / float(np.iinfo(chunk.dtype).max)
            else:
                chunk = chunk.astype(np.float32, copy=False)

            rate = int(frame.sample_rate or SAMPLE_RATE)
            chunk = resample_audio(chunk, rate, SAMPLE_RATE)
            if len(chunk):
                st.session_state.live_audio.append(chunk)
                engine.add_audio(chunk)

        total_samples = sum(len(c) for c in st.session_state.live_audio)
        steps_done = int(total_samples / (CHUNK_SECONDS * SAMPLE_RATE))
        if steps_done > st.session_state.live_last_step:
            st.session_state.live_last_step = steps_done
            if engine.step():
                st.session_state.live_final_text = engine.get_committed_text()
                st.session_state.live_partial_text = engine.get_partial_text()

    except Exception as exc:
        st.session_state.live_error = str(exc)

final_placeholder = st.empty()
partial_placeholder = st.empty()
metrics_placeholder = st.empty()

final_placeholder.markdown(
    f"**Final transcript**\n\n{st.session_state.live_final_text or '_Menunggu ucapan..._'}"
)
partial_placeholder.markdown(
    f"<span style='color:gray'><b>Partial transcript</b><br>{st.session_state.live_partial_text or '...'}</span>",
    unsafe_allow_html=True,
)

if st.session_state.live_started_at is not None:
    duration = time.time() - st.session_state.live_started_at
    metrics_placeholder.caption(
        f"Status: {'Merekam' if ctx.state.playing else 'Perekaman berhenti'} · "
        f"Durasi: {int(duration // 60):02d}:{int(duration % 60):02d}"
    )

if st.session_state.live_error:
    st.error(f"ASR error: {st.session_state.live_error}")

# Explicitly finalize after the user stops WebRTC.
if not ctx.state.playing and st.session_state.live_started_at is not None and not st.session_state.live_finished:
    try:
        engine = load_transcriber(st.session_state.live_model or model_size)
        engine.language = st.session_state.live_language or language_code
        engine.step(final=True)
        transcript = engine.get_text()
        words = engine.get_words()
        audio = (
            np.concatenate(st.session_state.live_audio).astype(np.float32)
            if st.session_state.live_audio else np.array([], dtype=np.float32)
        )
        duration_seconds = len(audio) / SAMPLE_RATE

        st.session_state.live_final_text = transcript
        st.session_state.live_partial_text = ""
        st.session_state.live_finished = True

        st.subheader("Final transcript")
        st.text_area("Transkripsi lengkap", transcript, height=180)

        q_idx = st.session_state.live_question_index
        final_question = question_list[q_idx]
        result = analyze(
            words,
            st.session_state.live_language,
            final_question["keywords"],
            final_question["ideal_seconds"],
            duration_seconds,
        )
        st.subheader("Laporan analisis")
        st.text(format_report(result, final_question["text"]))

        if len(audio):
            wav_path, json_path = save_session(
                audio,
                final_question,
                language_name,
                st.session_state.live_model or model_size,
                transcript,
                result,
                engine,
            )
            st.success("Sesi selesai dan tersimpan.")
            with open(wav_path, "rb") as f:
                st.download_button("Unduh audio WAV", f.read(), file_name=Path(wav_path).name, mime="audio/wav")
            with open(json_path, "rb") as f:
                st.download_button("Unduh laporan JSON", f.read(), file_name=Path(json_path).name, mime="application/json")
    except Exception as exc:
        st.error(f"Gagal menyelesaikan transkripsi/analisis: {exc}")

if st.session_state.live_started_at is not None and not st.session_state.live_finished:
    st.info("Jika teks tidak berubah, pastikan browser memberi izin mikrofon dan koneksi WebRTC berhasil.")
    # Refresh the Streamlit script to render updates while the browser is recording.
    time.sleep(0.25)
    st.rerun()

st.divider()
st.caption(
    "Catatan: live transcript bergantung pada streaming WebRTC dan implementasi StreamingTranscriber "
    "di asr_engine.py. Uji terlebih dahulu secara lokal sebelum deploy."
)
