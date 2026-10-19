"""
InterviewCoach on Modal
=======================
- GPU class  : faster-whisper running OpenAI Whisper large-v3-turbo (CTranslate2, fp16)
- Web app    : FastAPI on a cheap CPU container; serves static/index.html + /api/*

Deploy:
    pip install modal
    modal setup
    modal deploy modal_app.py

Dev (hot reload, temporary URL):
    modal serve modal_app.py
"""
import io
import threading

import modal

APP_NAME = "interviewcoach"
MODEL_NAME = "large-v3-turbo"   # faster-whisper alias for openai/whisper-large-v3-turbo
MODEL_DIR = "/models"
GPU = "L4"                      # "T4" is cheaper and also works; "A10G"/"L40S" are faster
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Whisper tends to "clean up" speech. A disfluent prompt nudges it to keep fillers (um, uh, ee, anu...).
PROMPTS = {
    "en": "Umm, let me think like, hmm... Okay, so, uh, basically, you know, I mean, I'm kind of, like, thinking.",
    "id": "Ee, jadi, eee, anu, gitu, kan, ya... nah, intinya, maksudnya, sih, kayak, begitu, yah.",
}

app = modal.App(APP_NAME)


def _download_model():
    """Runs once at image build time so containers start with weights already on disk."""
    from faster_whisper.utils import download_model

    download_model(MODEL_NAME, cache_dir=MODEL_DIR)


gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("faster-whisper==1.1.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_model)
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]")
    .add_local_dir("static", remote_path="/assets")  # must stay the last build step
)


# ─────────────────────────────────────────────────────────────────────────────
#  GPU: Whisper large-v3-turbo
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(image=gpu_image, gpu=GPU, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=8)
class Transcriber:
    @modal.enter()
    def load(self):
        import numpy as np
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            MODEL_NAME, device="cuda", compute_type="float16", download_root=MODEL_DIR
        )
        self.lock = threading.Lock()  # one GPU decode at a time; others queue
        # Warm up CUDA kernels so the first real request is fast
        segs, _ = self.model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(segs)

    @modal.method()
    def ping(self) -> str:
        return "ok"

    @modal.method()
    def transcribe(self, audio: bytes, lang: str = "en", live: bool = False) -> dict:
        """
        live=True  -> short rolling chunk for the live preview (greedy decode + VAD, fast)
        live=False -> full recording for the final report (beam search, no VAD so fillers/pauses survive)
        Returns {text, words:[{w,s,e}], duration, language}
        """
        prompt = PROMPTS.get(lang, PROMPTS["en"])
        with self.lock:
            segments, info = self.model.transcribe(
                io.BytesIO(audio),
                language=lang,
                task="transcribe",
                beam_size=1 if live else 5,
                word_timestamps=True,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                vad_filter=live,
                vad_parameters={"min_silence_duration_ms": 500},
                no_speech_threshold=0.6,
                hallucination_silence_threshold=2.0,
            )
            words = []
            for seg in segments:  # generator: decoding happens while iterating
                txt = seg.text.strip()
                if not txt:
                    continue
                # drop silence hallucinations and prompt echo
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue
                if txt.lower() in prompt.lower():
                    continue
                for w in seg.words or []:
                    token = w.word.strip()
                    if token:
                        words.append({"w": token, "s": round(w.start, 2), "e": round(w.end, 2)})

        return {
            "text": " ".join(w["w"] for w in words),
            "words": words,
            "duration": round(info.duration, 2),
            "language": info.language,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Web: static UI + API (CPU only; calls the GPU class remotely)
# ─────────────────────────────────────────────────────────────────────────────
@app.function(image=web_image, scaledown_window=120)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse

    api = FastAPI(title="InterviewCoach")

    with open("/assets/index.html", encoding="utf-8") as f:
        index_html = f.read()

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return index_html

    @api.post("/api/warmup")
    async def warmup():
        """Fire-and-forget: boots the GPU container while the user is still picking a question."""
        await Transcriber().ping.spawn.aio()
        return {"ok": True}

    @api.post("/api/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        lang: str = Form("en"),
        live: str = Form("0"),
    ):
        if lang not in PROMPTS:
            raise HTTPException(400, "lang must be 'en' or 'id'")
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "audio too large (max 25 MB)")
        try:
            return await Transcriber().transcribe.remote.aio(data, lang, live == "1")
        except Exception as e:  # undecodable audio, GPU error, etc.
            raise HTTPException(500, f"transcription failed: {e}")

    return api
