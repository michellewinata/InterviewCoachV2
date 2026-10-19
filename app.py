import io
import os
import re
import json
import wave
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import numpy as np
import scipy.signal
import vosk

# ── Optional Whisper Support ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="InterviewCoach - Native AI Speech Practice",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 60%, #1E88E5 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(13,71,161,0.15);
    }
    
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E3F2FD;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(33,150,243,0.06);
    }
    .stat-val {
        font-size: 26px;
        font-weight: 800;
        color: #0D47A1;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stat-lbl {
        font-size: 13px;
        color: #546E7A;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .question-box {
        background: #F0F7FF;
        border-left: 5px solid #2196F3;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 17px;
        font-weight: 600;
        color: #0D47A1;
    }
    
    .filler-badge {
        background-color: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFE0B2;
        padding: 2px 7px;
        border-radius: 6px;
        font-weight: 700;
    }
    
    .pause-short {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #C8E6C9;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 800;
        margin: 0 4px;
    }
    
    .pause-long {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #FFCDD2;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 800;
        margin: 0 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Datasets: Questions & Fillers ─────────────────────────────────────────────
QUESTIONS: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "Common & Personal": [
            "Tell me about yourself.",
            "What is your greatest professional strength?",
            "What do you consider to be your biggest weakness?",
            "Why are you interested in this position?",
            "Why do you want to work at our company?",
            "Where do you see yourself in five years?",
            "What are you passionate about outside of work?",
            "How would your colleagues describe your working style?"
        ],
        "Teamwork & Collaboration": [
            "Tell me about a time you worked successfully in a team.",
            "How do you handle disagreements or conflicts with colleagues?",
            "Describe a situation where you had to work with a difficult coworker.",
            "How do you prioritize tasks when collaborating on group projects?",
            "Give an example of going above and beyond to support a team member.",
            "How do you give and receive constructive feedback?",
            "Tell me about a time you stepped up to lead a project.",
            "How do you ensure clear communication in a remote or hybrid environment?"
        ],
        "Problem Solving": [
            "Describe a complex problem you solved at work and how you approached it.",
            "Tell me about a mistake you made and what you learned from it.",
            "How do you approach a task you've never done before?",
            "Give an example of a creative solution you devised under constraints.",
            "Describe a time you had to make a critical decision with incomplete data.",
            "How do you handle unexpected changes or roadblocks in a project?",
            "Tell me about a time you had to meet an extremely tight deadline.",
            "How do you prioritize when juggling multiple urgent tasks?"
        ],
        "Leadership & Management": [
            "How would you describe your leadership style?",
            "How do you motivate a team facing challenges or tight deadlines?",
            "Give an example of how you persuaded stakeholders to adopt your idea.",
            "How do you handle an underperforming team member?",
            "Describe a difficult decision you had to make as a leader.",
            "How do you foster trust and transparency within your team?",
            "Tell me about a project you managed from inception to completion."
        ],
        "Situational Questions": [
            "What would you do if you strongly disagreed with a manager's decision?",
            "How do you manage stress when workloads become overwhelming?",
            "What would you do if you witnessed a colleague acting unethically?",
            "How would you handle onboarding at a completely new company?",
            "If a project falls behind schedule, how do you get it back on track?",
            "What would you do if you received negative feedback from a key client?",
            "How do you work effectively under a micromanager?"
        ],
        "Career & Growth": [
            "What professional accomplishment are you most proud of?",
            "How do you stay up-to-date with emerging industry trends?",
            "What are your long-term career aspirations?",
            "How do you handle setbacks and transform them into growth?",
            "What technical or soft skills are you actively developing right now?",
            "Tell me about a mentor who had a lasting impact on your journey.",
            "How do you balance daily deliverables with long-term skill building?"
        ],
        "Culture & Values": [
            "What kind of work environment allows you to do your best work?",
            "What intrinsic factors motivate you on difficult days?",
            "How do you maintain work-life balance and avoid burnout?",
            "What does meaningful career success look like to you?",
            "What questions do you have for our team and company culture?"
        ]
    },
    "id": {
        "Umum & Pribadi": [
            "Ceritakan tentang diri Anda.",
            "Apa kekuatan profesional terbesar Anda?",
            "Apa kelemahan terbesar yang sedang Anda perbaiki?",
            "Mengapa Anda tertarik dengan posisi ini?",
            "Mengapa Anda ingin bergabung dengan perusahaan kami?",
            "Di mana Anda melihat diri Anda lima tahun ke depan?",
            "Apa minat dan passion Anda di luar pekerjaan?",
            "Bagaimana rekan kerja menggambarkan etos kerja Anda?"
        ],
        "Kerja Tim & Kolaborasi": [
            "Ceritakan pengalaman Anda bekerja sukses dalam sebuah tim.",
            "Bagaimana Anda menangani konflik atau perbedaan pendapat dengan rekan kerja?",
            "Gambarkan situasi saat Anda harus bekerja sama dengan rekan yang sulit.",
            "Bagaimana cara Anda memprioritaskan tugas saat bekerja dalam kelompok?",
            "Berikan contoh saat Anda memberikan bantuan ekstra untuk rekan tim.",
            "Bagaimana cara Anda memberikan dan menerima umpan balik konstruktif?",
            "Ceritakan saat Anda berinisiatif memimpin ketika tim membutuhkan arahan.",
            "Bagaimana Anda menjaga komunikasi efektif saat bekerja secara remote?"
        ],
        "Pemecahan Masalah": [
            "Gambarkan masalah sulit yang pernah Anda selesaikan di pekerjaan.",
            "Ceritakan kesalahan kerja yang pernah Anda buat dan apa pembelajarannya.",
            "Bagaimana cara Anda memulai tugas yang benar-benar baru bagi Anda?",
            "Berikan contoh solusi kreatif yang Anda ciptakan di bawah keterbatasan.",
            "Ceritakan saat Anda harus mengambil keputusan penting dengan data terbatas.",
            "Bagaimana Anda merespons perubahan mendadak di tengah proyek?",
            "Ceritakan pengalaman Anda menyelesaikan pekerjaan dengan tenggat waktu ketat.",
            "Bagaimana Anda mengatur prioritas ketika semua tugas terasa mendesak?"
        ],
        "Kepemimpinan & Manajemen": [
            "Bagaimana gaya kepemimpinan yang biasa Anda terapkan?",
            "Bagaimana cara Anda memotivasi tim yang sedang mengalami kejenuhan?",
            "Berikan contoh saat Anda berhasil meyakinkan orang lain tentang ide Anda.",
            "Bagaimana pendekatan Anda terhadap anggota tim yang kinerjanya menurun?",
            "Gambarkan keputusan tersulit yang pernah Anda ambil sebagai pemimpin.",
            "Bagaimana Anda membangun rasa saling percaya dan keterbukaan dalam tim?",
            "Ceritakan sebuah proyek yang Anda kelola dari awal hingga sukses."
        ],
        "Pertanyaan Situasional": [
            "Apa yang Anda lakukan jika tidak sependapat dengan arahan atasan?",
            "Bagaimana strategi Anda mengelola stres saat beban kerja sangat tinggi?",
            "Apa tindakan Anda jika melihat rekan kerja melakukan pelanggaran etika?",
            "Bagaimana cara Anda beradaptasi dalam minggu-minggu pertama di tempat baru?",
            "Jika proyek terlambat dari target, langkah apa yang langsung Anda ambil?",
            "Bagaimana cara Anda menyikapi komplain atau kritik pedas dari klien?",
            "Bagaimana cara Anda tetap produktif di bawah atasan yang micromanage?"
        ],
        "Karir & Pertumbuhan": [
            "Pencapaian profesional apa yang paling membanggakan bagi Anda sejauh ini?",
            "Bagaimana cara Anda terus mengikuti perkembangan tren industri terbaru?",
            "Apa rencana karir jangka panjang yang ingin Anda capai?",
            "Bagaimana cara Anda bangkit dari kegagalan dan menjadikannya pelajaran?",
            "Keterampilan baru apa yang saat ini sedang aktif Anda pelajari?",
            "Ceritakan tentang sosok mentor yang paling berpengaruh bagi karir Anda.",
            "Bagaimana Anda membagi waktu antara pekerjaan harian dan pengembangan diri?"
        ],
        "Budaya & Nilai": [
            "Lingkungan kerja seperti apa yang membuat Anda dapat berkinerja optimal?",
            "Apa yang paling memotivasi Anda untuk memberikan usaha terbaik setiap hari?",
            "Bagaimana cara Anda menjaga keseimbangan hidup dan mencegah burnout?",
            "Bagi Anda pribadi, apa arti kesuksesan dalam berkarir?",
            "Apakah ada hal yang ingin Anda tanyakan mengenai tim dan budaya kami?"
        ]
    }
}

FILLERS: Dict[str, List[str]] = {
    "en": [
        "um", "uh", "er", "ah", "like", "you know", "basically", "literally", "actually",
        "honestly", "right", "so", "well", "kind of", "sort of", "i mean", "you see",
        "i guess", "anyway", "stuff", "whatever", "totally", "seriously", "definitely",
        "absolutely", "obviously", "clearly", "essentially", "fundamentally"
    ],
    "id": [
        "em", "eh", "ee", "anu", "gitu", "ya", "kan", "tuh", "jadi", "sebenarnya", "pokoknya",
        "intinya", "sebenernya", "kayak", "iyakan", "loh", "dong", "sih", "deh", "udah",
        "soalnya", "makanya", "nah", "gimana ya", "begitu", "hal"
    ]
}

STOP_WORDS_EN = {
    "the", "and", "a", "an", "in", "to", "of", "for", "with", "at", "by", "from", "on", "about",
    "into", "through", "during", "before", "after", "above", "below", "between", "under", "this",
    "that", "these", "those", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "i", "me", "my", "myself", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "also"
}

STOP_WORDS_ID = {
    "yang", "di", "dan", "itu", "dengan", "untuk", "tidak", "ini", "dari", "dalam", "akan", "pada",
    "juga", "saya", "ke", "karena", "tersebut", "bisa", "ada", "mereka", "lebih", "sudah", "atau",
    "saat", "oleh", "secara", "ia", "setelah", "dua", "sampai", "serta", "saja", "banyak", "telah",
    "sangat", "bagi", "kita", "kamu", "dia", "mereka", "anda", "kami", "apa", "siapa", "mengapa",
    "kapan", "bagaimana", "mana", "jika", "kalau", "namun", "tetapi", "lalu", "kemudian", "bahkan",
    "agar", "supaya", "hanya", "lagi", "pun", "walau", "meski", "seperti", "sebagai", "bukan"
}


# ── VOSK Model Management ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vosk_model(lang: str = "en") -> Optional[vosk.Model]:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if lang == "en":
        folder_name = "vosk-model-small-en-us-0.15"
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    else:
        folder_name = "vosk-model-small-en-us-0.15"  # fallback
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        
    model_path = models_dir / folder_name
    if not model_path.exists():
        with st.spinner("Downloading offline VOSK model (~40 MB)... Please wait a few seconds..."):
            zip_dest = models_dir / f"{folder_name}.zip"
            urllib.request.urlretrieve(url, zip_dest)
            with zipfile.ZipFile(zip_dest, "r") as zf:
                zf.extractall(models_dir)
            if zip_dest.exists():
                zip_dest.unlink()
                
    vosk.SetLogLevel(-1)
    return vosk.Model(str(model_path))


@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str = "base") -> Any:
    if not HAS_WHISPER:
        return None
    try:
        # compute_type="int8" is lightweight and fast on CPU
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        st.warning(f"Whisper initialization note: {e}")
        return None


# ── Audio Processing Helpers ──────────────────────────────────────────────────
def convert_to_16k_mono_pcm(wav_bytes: bytes) -> Tuple[bytes, float]:
    """Reads WAV bytes and converts to 16kHz mono 16-bit PCM."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, 'rb') as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            
    if sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16)
    elif sw == 4:
        arr = (np.frombuffer(raw, dtype=np.int32) / 65536).astype(np.int16)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)
        
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
        
    if fr != 16000:
        num_samples = int(len(arr) * 16000 / fr)
        arr = scipy.signal.resample(arr, num_samples).astype(np.int16)
        
    duration = len(arr) / 16000.0
    return arr.tobytes(), duration


def transcribe_vosk(model: vosk.Model, pcm_bytes: bytes) -> List[Dict[str, Any]]:
    """Transcribes PCM bytes using VOSK with word-level timestamps."""
    rec = vosk.KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    
    words_list = []
    chunk_size = 4000
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        if rec.AcceptWaveform(chunk):
            res = json.loads(rec.Result())
            if "result" in res:
                words_list.extend(res["result"])
                
    final_res = json.loads(rec.FinalResult())
    if "result" in final_res:
        words_list.extend(final_res["result"])
        
    return words_list


def transcribe_whisper(model: Any, wav_bytes: bytes) -> List[Dict[str, Any]]:
    """Transcribes audio using faster-whisper with word-level timestamps."""
    with io.BytesIO(wav_bytes) as bio:
        segments, _ = model.transcribe(bio, language="id", word_timestamps=True)
        words_list = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    words_list.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "conf": w.probability
                    })
            else:
                words_list.append({
                    "word": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "conf": 1.0
                })
        return words_list


# ── Analysis Engine ───────────────────────────────────────────────────────────
def analyze_speech(
    words_list: List[Dict[str, Any]],
    duration: float,
    lang: str
) -> Dict[str, Any]:
    """Analyzes speech, detects pauses (/ and //), counts fillers, and extracts keywords."""
    clean_words = []
    report_tokens = []
    pauses = []
    
    prev_end: Optional[float] = None
    
    for item in words_list:
        w = item.get("word", "").strip()
        if not w:
            continue
        start = item.get("start", 0.0)
        end = item.get("end", start)
        
        # Pause detection between words
        if prev_end is not None:
            gap = start - prev_end
            if gap >= 3.0:
                report_tokens.append("//")
                pauses.append({"type": "Long (>3s)", "time": f"{int(prev_end//60)}:{int(prev_end%60):02d}", "duration": round(gap, 1)})
            elif gap >= 1.5:
                report_tokens.append("/")
                pauses.append({"type": "Short (1.5-3s)", "time": f"{int(prev_end//60)}:{int(prev_end%60):02d}", "duration": round(gap, 1)})
                
        clean_words.append(w)
        report_tokens.append(w)
        prev_end = end
        
    clean_transcript = " ".join(clean_words).strip()
    report_raw = " ".join(report_tokens).strip()
    
    # Calculate Metrics
    total_words = len(clean_words)
    dur = max(duration, 1.0)
    wpm = round((total_words / dur) * 60)
    
    # Filler Word Counting
    fillers_ref = FILLERS.get(lang, FILLERS["en"])
    filler_counts = {}
    lower_text = clean_transcript.lower()
    
    for f in fillers_ref:
        matches = re.findall(rf"\b{re.escape(f)}\b", lower_text)
        if matches:
            filler_counts[f] = len(matches)
            
    total_fillers = sum(filler_counts.values())
    filler_pct = round((total_fillers / total_words) * 100) if total_words > 0 else 0
    
    # Highlight Fillers and Pauses in Final Transcript Report
    highlighted_report = report_raw
    highlighted_report = re.sub(r' // ', ' <span class="pause-long" title="Long pause (>3s)">//</span> ', highlighted_report)
    highlighted_report = re.sub(r' / ', ' <span class="pause-short" title="Short pause (1.5-3s)">/</span> ', highlighted_report)
    
    # Sort fillers by length descending to match multi-word fillers first
    sorted_fillers = sorted(fillers_ref, key=len, reverse=True)
    for f in sorted_fillers:
        pattern = rf"\b({re.escape(f)})\b"
        highlighted_report = re.sub(pattern, r'<span class="filler-badge">\1</span>', highlighted_report, flags=re.IGNORECASE)
        
    # Keyword Extraction
    stops = STOP_WORDS_EN if lang == "en" else STOP_WORDS_ID
    all_tokens = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", lower_text)
    freq = {}
    for tok in all_tokens:
        if tok not in stops and tok not in fillers_ref:
            freq[tok] = freq.get(tok, 0) + 1
            
    keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "clean_transcript": clean_transcript,
        "highlighted_report": highlighted_report,
        "duration": duration,
        "total_words": total_words,
        "wpm": wpm,
        "total_fillers": total_fillers,
        "filler_pct": filler_pct,
        "filler_counts": filler_counts,
        "pauses": pauses,
        "keywords": keywords,
    }


# ── Main Application UI ───────────────────────────────────────────────────────
def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:28px;">🎙️ InterviewCoach</h1>
        <p style="margin:6px 0 0 0; opacity:0.9; font-size:15px;">
            Practice job interviews with local on-device speech recognition, filler word detection & pause analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.subheader("⚙️ Settings & Configuration")
        
        lang_choice = st.radio(
            "🌐 Select Language / Bahasa:",
            options=["English (EN)", "Bahasa Indonesia (ID)"],
            index=0
        )
        lang = "en" if "English" in lang_choice else "id"
        
        st.divider()
        st.subheader("🎯 Practice Mode")
        practice_mode = st.radio(
            "Choose mode:",
            options=["Free Speech", "Choose Question (50 Questions)", "Random Question Roll"],
            index=1
        )
        
        st.divider()
        st.caption("🔒 **100% Local On-Device AI**\nNo audio is sent to Google, OpenAI, or external cloud speech APIs.")

    # Practice Mode Selection Box
    selected_question = ""
    lang_questions = QUESTIONS[lang]
    
    if practice_mode == "Choose Question (50 Questions)":
        st.markdown("### 📋 Select Interview Question")
        cat = st.selectbox("Category:", list(lang_questions.keys()))
        selected_question = st.selectbox("Question:", lang_questions[cat])
        st.markdown(f'<div class="question-box">💡 <b>Interview Question:</b><br>{selected_question}</div>', unsafe_allow_html=True)
        
    elif practice_mode == "Random Question Roll":
        st.markdown("### 🎲 Random Interview Question")
        if "random_q" not in st.session_state or st.button("🎲 Roll New Question"):
            all_qs = [q for cat_qs in lang_questions.values() for q in cat_qs]
            st.session_state["random_q"] = np.random.choice(all_qs)
        selected_question = st.session_state["random_q"]
        st.markdown(f'<div class="question-box">💡 <b>Your Question:</b><br>{selected_question}</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="question-box">🗣️ <b>Free Speech Mode:</b> Answer freely or introduce yourself.</div>', unsafe_allow_html=True)

    # Audio Recording / Upload Section
    st.markdown("### 🎤 Record Your Response")
    st.write("Click the microphone button to record your answer, then click again to stop:")

    audio_data = None
    if hasattr(st, "audio_input"):
        audio_data = st.audio_input("Record audio response:")
    else:
        audio_data = st.file_uploader("Upload audio recording (WAV / MP3):", type=["wav", "mp3", "ogg"])

    if audio_data is not None:
        raw_bytes = audio_data.read()
        st.audio(raw_bytes)
        
        if st.button("🚀 Analyze Interview Response", type="primary"):
            with st.spinner("Processing speech locally with on-device model..."):
                try:
                    pcm_bytes, duration = convert_to_16k_mono_pcm(raw_bytes)
                except Exception as e:
                    st.error(f"Error decoding audio format: {e}. Please record or upload a standard WAV file.")
                    return
                
                # Perform ASR
                words_list = []
                if lang == "en":
                    vosk_model = get_vosk_model("en")
                    if vosk_model:
                        words_list = transcribe_vosk(vosk_model, pcm_bytes)
                else:
                    # Indonesian
                    whisper_model = get_whisper_model("base")
                    if whisper_model:
                        words_list = transcribe_whisper(whisper_model, raw_bytes)
                    else:
                        # Fallback to VOSK if whisper not available
                        vosk_model = get_vosk_model("en")
                        if vosk_model:
                            words_list = transcribe_vosk(vosk_model, pcm_bytes)
                
                if not words_list:
                    st.warning("No clear speech detected in the audio. Please speak clearly into the microphone.")
                    return
                
                # Run Performance Report Engine
                report = analyze_speech(words_list, duration, lang)
                
                st.success("✅ Analysis Complete!")
                st.divider()
                
                # Performance Report Section
                st.header("📊 End-of-Session Performance Report")
                
                # Stats Metrics Grid
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-val">{int(report['duration']//60)}:{int(report['duration']%60):02d}</div>
                        <div class="stat-lbl">⏱️ Duration</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-val">{report['total_words']}</div>
                        <div class="stat-lbl">💬 Words Spoken</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    wpm_color = "#2E7D32" if 120 <= report['wpm'] <= 160 else "#E65100"
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-val" style="color:{wpm_color}">{report['wpm']}</div>
                        <div class="stat-lbl">⚡ WPM (Speed)</div>
                    </div>""", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-val" style="color:#C62828">{report['total_fillers']}</div>
                        <div class="stat-lbl">🔴 Fillers ({report['filler_pct']}%)</div>
                    </div>""", unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""<div class="stat-card">
                        <div class="stat-val">{len(report['pauses'])}</div>
                        <div class="stat-lbl">⏸️ Pauses (&gt;1.5s)</div>
                    </div>""", unsafe_allow_html=True)

                st.write("")
                
                # Filler Words Breakdown & Keywords
                rep_c1, rep_c2 = st.columns(2)
                with rep_c1:
                    st.subheader("🔴 Filler Words Breakdown")
                    if report['filler_counts']:
                        f_cols = st.columns(3)
                        for idx, (f_word, count) in enumerate(report['filler_counts'].items()):
                            with f_cols[idx % 3]:
                                st.markdown(f'<span class="filler-badge">{f_word}</span> × {count}', unsafe_allow_html=True)
                    else:
                        st.info("🎉 Great job! No filler words detected in your response.")
                        
                with rep_c2:
                    st.subheader("🔑 Top Keywords Extracted")
                    if report['keywords']:
                        kw_str = " &nbsp; ".join([f"<b>{kw}</b> ({cnt})" for kw, cnt in report['keywords']])
                        st.markdown(kw_str, unsafe_allow_html=True)
                    else:
                        st.info("No significant keywords extracted.")
                
                # Pauses Breakdown
                st.subheader("⏸️ Detected Pauses")
                if report['pauses']:
                    p_cols = st.columns(min(4, len(report['pauses'])))
                    for idx, p in enumerate(report['pauses']):
                        with p_cols[idx % 4]:
                            badge_cls = "pause-long" if ">3s" in p['type'] else "pause-short"
                            st.markdown(f'<span class="{badge_cls}">{p["type"]}</span> at <b>{p["time"]}</b> ({p["duration"]}s)', unsafe_allow_html=True)
                else:
                    st.info("No long pauses (>1.5s) detected. Good speaking flow!")

                # Final Transcript with Highlighted Fillers and Pause Markers
                st.subheader("📝 Final Transcript with Highlighted Fillers & Pause Markers")
                st.caption("Legend: <span class='filler-badge'>Orange</span> = Filler words &nbsp;|&nbsp; <span class='pause-short'>/</span> = Short pause (1.5-3.0s) &nbsp;|&nbsp; <span class='pause-long'>//</span> = Long pause (>3.0s)", unsafe_allow_html=True)
                
                st.markdown(
                    f"""<div style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:12px; padding:20px; font-size:16px; line-height:1.9;">
                        {report['highlighted_report']}
                    </div>""",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
