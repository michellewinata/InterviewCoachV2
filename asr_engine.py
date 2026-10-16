"""
Engine transkripsi streaming (lokal, pakai faster-whisper).

Whisper aslinya bukan model streaming, jadi kita bikin "pseudo-streaming":
  1. Audio dari mic terus masuk ke buffer.
  2. Tiap beberapa detik, seluruh isi buffer ditranskrip ulang -> teks yang tampil di layar
     diperbarui (ini yang disebut partial / updated transcription).
  3. Kalau buffer sudah terlalu panjang (> MAX_BUFFER_SECONDS), bagian yang sudah "aman"
     dipindah ke committed (final) lalu buffer dipotong, supaya inference tidak makin lama.
"""

import time
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000        # Whisper butuh 16 kHz mono
MAX_BUFFER_SECONDS = 15    # batas panjang buffer sebelum dipotong
MIN_AUDIO_SECONDS = 1.0    # audio lebih pendek dari ini nggak diproses
SILENCE_RMS = 0.005        # di bawah ini dianggap hening (biar Whisper nggak halusinasi)

# Whisper cenderung "membersihkan" ucapan dan membuang filler.
# Prompt awal berisi filler bikin dia lebih mau menuliskannya.
FILLER_PROMPTS = {
    "en": "Um, uh, so, like, you know, hmm, I mean, well...",
    "id": "Eee, emm, anu, jadi, hmm, ya, gitu, kayak, apa ya, maksudnya...",
}


class StreamingTranscriber:
    def __init__(self, model_size="base", language="id", device="cpu",
                 compute_type="int8", use_filler_prompt=True):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language          # "en" atau "id", bisa diganti sebelum sesi mulai
        self.use_filler_prompt = use_filler_prompt
        self.reset()

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start = 0.0   # posisi awal buffer di timeline sesi (detik)
        self.committed = []       # kata yang sudah final
        self.partial = []         # kata yang masih bisa berubah
        self.latencies = []       # waktu inference tiap update (detik)
        self.rtfs = []            # real-time factor tiap update
        self.last_latency = 0.0
        self.last_rtf = 0.0

    def add_audio(self, chunk):
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])

    # ------------------------------------------------------------------
    def step(self, final=False):
        """Transkripsi isi buffer sekarang. Return True kalau teks diperbarui."""
        buffer_seconds = len(self.buffer) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(self.buffer ** 2))) if len(self.buffer) else 0.0

        if buffer_seconds < MIN_AUDIO_SECONDS or rms < SILENCE_RMS:
            if final:  # sesi selesai -> apa yang tersisa dijadikan final
                self.committed += self.partial
                self.partial = []
            return False

        prompt = FILLER_PROMPTS.get(self.language) if self.use_filler_prompt else None

        start = time.perf_counter()
        segments, _ = self.model.transcribe(
            self.buffer,
            language=self.language,
            beam_size=1,                        # greedy, lebih cepat
            word_timestamps=True,               # dipakai untuk WPM & deteksi pause
            condition_on_previous_text=False,   # cegah teks berulang
            initial_prompt=prompt,
        )
        segments = list(segments)  # generator baru benar-benar jalan di sini
        elapsed = time.perf_counter() - start

        self.last_latency = elapsed
        self.last_rtf = elapsed / buffer_seconds
        self.latencies.append(elapsed)
        self.rtfs.append(self.last_rtf)

        # penting: hitung waktu kata SEBELUM buffer_start diubah
        seg_words = [self._segment_words(seg) for seg in segments]
        all_words = [w for words in seg_words for w in words]

        if final:
            self.committed += all_words
            self.partial = []
            return True

        n_commit, cut = self._find_cut(segments, seg_words, all_words, buffer_seconds)
        if n_commit == 0:
            self.partial = all_words
        else:
            self.committed += all_words[:n_commit]
            self.partial = all_words[n_commit:]
            self.buffer = self.buffer[int(cut * SAMPLE_RATE):]
            self.buffer_start += cut
        return True

    def _segment_words(self, seg):
        words = []
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append({
                    "word": text,
                    "start": w.start + self.buffer_start,
                    "end": w.end + self.buffer_start,
                })
        return words

    def _find_cut(self, segments, seg_words, all_words, buffer_seconds):
        """Tentukan berapa kata yang di-commit dan di detik keberapa buffer dipotong."""
        if buffer_seconds <= MAX_BUFFER_SECONDS:
            return 0, 0.0

        # kasus normal: commit semua segmen kecuali yang terakhir
        if len(segments) >= 2:
            n = sum(len(w) for w in seg_words[:-1])
            return n, segments[-1].start

        # kasus jarang: Whisper ngasih 1 segmen panjang -> potong per kata,
        # sisakan ~3 detik terakhir
        if buffer_seconds > MAX_BUFFER_SECONDS * 1.5:
            batas = self.buffer_start + buffer_seconds - 3
            lama = [w for w in all_words if w["end"] < batas]
            if lama:
                return len(lama), lama[-1]["end"] - self.buffer_start
        return 0, 0.0

    # ------------------------------------------------------------------
    def get_committed_text(self):
        return " ".join(w["word"] for w in self.committed)

    def get_partial_text(self):
        return " ".join(w["word"] for w in self.partial)

    def get_text(self):
        return (self.get_committed_text() + " " + self.get_partial_text()).strip()

    def get_words(self):
        return self.committed + self.partial

    def latency_summary(self, latencies=None, rtfs=None):
        lat = self.latencies if latencies is None else latencies
        rtf = self.rtfs if rtfs is None else rtfs
        if not lat:
            return {"updates": 0, "mean_latency": 0.0, "max_latency": 0.0, "mean_rtf": 0.0}
        return {
            "updates": len(lat),
            "mean_latency": float(np.mean(lat)),
            "max_latency": float(np.max(lat)),
            "mean_rtf": float(np.mean(rtf)),
        }
