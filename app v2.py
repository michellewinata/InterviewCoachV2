"""
InterviewCoach - Streamlit web app
Supports Indonesian and English interview practice with local Whisper ASR.

Run locally:
    streamlit run app_streamlit.py

Expected companion files in the same directory:
    asr_engine.py
    analyzer.py
    questions.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from analyzer import analyze, format_report
from asr_engine import SAMPLE_RATE, StreamingTranscriber
from questions import QUESTIONS

LANGUAGES = {"Indonesia": "id", "English": "en"}
MODEL_OPTIONS = ["tiny", "base", "small", "medium"]


@st.cache_resource(show_spinner=False)
def load_transcriber(model_size: str):
    """Cache the ASR model so it is not loaded again on every rerun."""
    return StreamingTranscriber(model_size=model_size)


def transcribe_audio(audio: np.ndarray, engine: StreamingTranscriber):
    """Send recorded audio to the existing streaming engine in short chunks."""
    engine.reset()
    chunk_samples = max(1, int(2.0 * SAMPLE_RATE))

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk):
            engine.add_audio(chunk.astype(np.float32, copy=False))
            engine.step()

    # Finalize any remaining audio and return the final transcript/word timings.
    engine.step(final=True)
    return engine.get_text(), engine.get_words()


def save_session(audio, question, language, model_size, transcript, result, engine):
    """Save a WAV and JSON report under ./sessions."""
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


st.set_page_config(
    page_title="InterviewCoach",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ InterviewCoach")
st.caption("Latihan interview Bahasa Indonesia / English dengan speech recognition lokal.")

with st.sidebar:
    st.header("Pengaturan")
    language_name = st.selectbox("Bahasa jawaban", list(LANGUAGES.keys()))
    model_size = st.selectbox(
        "Model Whisper",
        MODEL_OPTIONS,
        index=1,
        help="tiny lebih ringan; small/medium biasanya lebih akurat tetapi lebih lambat.",
    )
    st.info(
        "Model ASR dijalankan di server aplikasi. Pada hosting gratis, proses pertama "
        "bisa cukup lama karena model perlu diunduh dan dimuat."
    )

language_code = LANGUAGES[language_name]
question_list = QUESTIONS[language_code]
question_texts = [q["text"] for q in question_list]
selected_question_text = st.selectbox("Pertanyaan interview", question_texts)
selected_index = question_texts.index(selected_question_text)
question = question_list[selected_index]

st.subheader("Pertanyaan")
st.markdown(f"**{question['text']}**")

st.markdown("### Rekam jawaban")
st.write(
    "Klik tombol rekam di bawah, jawab pertanyaan, lalu hentikan rekaman. "
    "Jika browser tidak mendukung perekaman mikrofon, unggah file audio WAV."
)

recorded_audio = None
if hasattr(st, "audio_input"):
    recorded_audio = st.audio_input("Rekam jawaban menggunakan mikrofon")
else:
    st.warning(
        "Versi Streamlit yang terpasang belum menyediakan perekam audio. "
        "Unggah file audio sebagai gantinya."
    )

uploaded_audio = st.file_uploader(
    "Atau unggah audio (WAV/FLAC/OGG)",
    type=["wav", "flac", "ogg"],
)

audio_source = recorded_audio if recorded_audio is not None else uploaded_audio

if audio_source is not None:
    st.audio(audio_source)

    if st.button("Transkripsikan dan analisis jawaban", type="primary"):
        try:
            with st.spinner(f"Memuat model Whisper '{model_size}'..."):
                engine = load_transcriber(model_size)
                engine.language = language_code

            with st.spinner("Membaca audio dan membuat transkripsi..."):
                audio_source.seek(0)
                audio, input_sample_rate = sf.read(audio_source, dtype="float32")

                # Convert stereo to mono.
                if audio.ndim == 2:
                    audio = np.mean(audio, axis=1)

                # Resample if needed; scipy is a dependency of many audio stacks,
                # but report a clear error if it is unavailable.
                if input_sample_rate != SAMPLE_RATE:
                    try:
                        from scipy.signal import resample_poly
                        from math import gcd
                        divisor = gcd(int(input_sample_rate), int(SAMPLE_RATE))
                        up = SAMPLE_RATE // divisor
                        down = int(input_sample_rate) // divisor
                        audio = resample_poly(audio, up, down).astype(np.float32)
                    except Exception as exc:
                        st.error(
                            f"Sample rate audio ({input_sample_rate} Hz) berbeda dari "
                            f"sample rate model ({SAMPLE_RATE} Hz), dan konversi gagal: {exc}"
                        )
                        st.stop()

                if len(audio) == 0:
                    st.error("File audio kosong. Silakan rekam atau unggah audio lain.")
                    st.stop()

                started = time.time()
                engine.language = language_code
                transcript, words = transcribe_audio(audio, engine)
                elapsed = time.time() - started

            st.subheader("Transkripsi")
            if transcript.strip():
                st.text_area("Hasil transkripsi", transcript, height=180)
            else:
                st.warning(
                    "Transkripsi kosong. Coba rekam dengan suara lebih jelas atau gunakan "
                    "model Whisper yang lebih besar."
                )

            duration_seconds = len(audio) / SAMPLE_RATE
            result = analyze(
                words,
                language_code,
                question["keywords"],
                question["ideal_seconds"],
                duration_seconds,
            )

            st.subheader("Laporan analisis")
            st.text(format_report(result, question["text"]))

            try:
                wav_path, json_path = save_session(
                    audio,
                    question,
                    language_name,
                    model_size,
                    transcript,
                    result,
                    engine,
                )
                st.success("Sesi berhasil disimpan.")
                st.caption(f"Audio: {wav_path} | Laporan: {json_path}")

                with open(wav_path, "rb") as f:
                    st.download_button(
                        "Unduh audio WAV",
                        data=f.read(),
                        file_name=Path(wav_path).name,
                        mime="audio/wav",
                    )
                with open(json_path, "rb") as f:
                    st.download_button(
                        "Unduh laporan JSON",
                        data=f.read(),
                        file_name=Path(json_path).name,
                        mime="application/json",
                    )
            except Exception as exc:
                st.warning(f"Analisis selesai, tetapi penyimpanan sesi gagal: {exc}")

            with st.expander("Detail performa"):
                st.write(f"Durasi audio: {duration_seconds:.1f} detik")
                st.write(f"Waktu transkripsi: {elapsed:.1f} detik")
                st.write(f"Jumlah kata terdeteksi: {len(words)}")
                try:
                    st.json(engine.latency_summary())
                except Exception:
                    pass

        except Exception as exc:
            st.error(f"Gagal memproses audio: {exc}")
            st.exception(exc)
else:
    st.info("Rekam jawaban atau unggah audio untuk memulai analisis.")

st.divider()
st.caption("InterviewCoach • Transkripsi diproses oleh model Whisper di server aplikasi.")
