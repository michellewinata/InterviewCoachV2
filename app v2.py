"""
InterviewCoach - aplikasi latihan interview (English / Bahasa Indonesia)
dengan real-time speech recognition lokal.

Cara jalan:
    python app.py                 # model default: base
    python app.py --model small   # lebih akurat, tapi lebih berat
"""

import argparse
import json
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf

from analyzer import analyze, format_report
from asr_engine import SAMPLE_RATE, StreamingTranscriber
from questions import QUESTIONS

STEP_SECONDS = 2.0   # transkripsi diperbarui tiap 2 detik audio baru
LANGUAGES = {"Indonesia": "id", "English": "en"}


class InterviewCoachApp:
    def __init__(self, root, model_size):
        self.root = root
        self.model_size = model_size
        self.engine = None

        self.recording = False
        self.audio_q = queue.Queue()   # audio dari mic -> worker
        self.ui_q = queue.Queue()      # worker -> tampilan
        self.stop_flag = threading.Event()
        self.all_audio = []
        self.stream = None
        self.start_time = 0.0
        self.recording_seconds = 0.0

        root.title("InterviewCoach - Real-Time Speech Recognition")
        root.geometry("920x800")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_ui()
        self.refresh_questions()

        # load model di thread terpisah supaya jendela nggak freeze
        self.status_var.set(f"Memuat model Whisper '{model_size}'...")
        threading.Thread(target=self.load_model, daemon=True).start()
        self.root.after(100, self.poll_ui_queue)

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Bahasa jawaban:").grid(row=0, column=0, sticky="w")
        self.lang_var = tk.StringVar(value="Indonesia")
        self.lang_box = ttk.Combobox(top, textvariable=self.lang_var, values=list(LANGUAGES),
                                     state="readonly", width=12)
        self.lang_box.grid(row=0, column=1, padx=8, sticky="w")
        self.lang_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_questions())

        ttk.Label(top, text="Pertanyaan:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.question_var = tk.StringVar()
        self.question_box = ttk.Combobox(top, textvariable=self.question_var,
                                         state="readonly", width=90)
        self.question_box.grid(row=1, column=1, padx=8, pady=(8, 0), sticky="w")
        self.question_box.bind("<<ComboboxSelected>>", lambda e: self.show_question())

        self.question_label = ttk.Label(self.root, text="", font=("Segoe UI", 14, "bold"),
                                        wraplength=860, padding=(10, 8))
        self.question_label.pack(fill="x")

        controls = ttk.Frame(self.root, padding=(10, 0))
        controls.pack(fill="x")
        self.start_btn = ttk.Button(controls, text="Mulai Menjawab", command=self.start_session,
                                    state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Selesai", command=self.stop_session,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        info = ttk.Frame(self.root, padding=10)
        info.pack(fill="x")
        self.status_var = tk.StringVar()
        self.timer_var = tk.StringVar(value="Waktu: 00:00")
        self.latency_var = tk.StringVar(value="Latency: - | RTF: -")
        ttk.Label(info, textvariable=self.status_var, width=40).pack(side="left")
        ttk.Label(info, textvariable=self.timer_var, width=16).pack(side="left")
        ttk.Label(info, textvariable=self.latency_var).pack(side="left")

        box1 = ttk.LabelFrame(self.root, text="Transkripsi Live (abu-abu = masih bisa berubah)",
                              padding=6)
        box1.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.transcript = tk.Text(box1, height=9, wrap="word", font=("Segoe UI", 12),
                                  state="disabled")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_config("final", foreground="black")
        self.transcript.tag_config("partial", foreground="gray")

        box2 = ttk.LabelFrame(self.root, text="Laporan Analisis", padding=6)
        box2.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.report = tk.Text(box2, height=14, wrap="word", font=("Consolas", 10),
                              state="disabled")
        self.report.pack(fill="both", expand=True)

    def lang_code(self):
        return LANGUAGES[self.lang_var.get()]

    def refresh_questions(self):
        """Isi dropdown pertanyaan sesuai bahasa yang dipilih."""
        texts = [q["text"] for q in QUESTIONS[self.lang_code()]]
        self.question_box["values"] = texts
        self.question_box.current(0)
        self.show_question()

    def show_question(self):
        self.question_label.config(text=self.question_var.get())

    def current_question(self):
        return QUESTIONS[self.lang_code()][self.question_box.current()]

    def set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def show_transcript(self, final_text, partial_text):
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("end", final_text + " ", "final")
        self.transcript.insert("end", partial_text, "partial")
        self.transcript.see("end")
        self.transcript.config(state="disabled")

    def set_inputs_enabled(self, enabled):
        state = "readonly" if enabled else "disabled"
        self.lang_box.config(state=state)
        self.question_box.config(state=state)

    # --------------------------------------------------------------- model
    def load_model(self):
        try:
            self.engine = StreamingTranscriber(model_size=self.model_size)
            self.ui_q.put(("ready",))
        except Exception as e:
            self.ui_q.put(("error", f"Gagal memuat model: {e}"))

    # ------------------------------------------------------------- sesi
    def start_session(self):
        self.engine.language = self.lang_code()
        self.engine.reset()
        self.all_audio = []
        self.audio_q = queue.Queue()
        self.stop_flag.clear()

        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                         blocksize=1600, callback=self.audio_callback)
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Mikrofon", f"Mikrofon tidak bisa dibuka:\n{e}")
            return

        self.recording = True
        self.start_time = time.time()
        threading.Thread(target=self.worker_loop, daemon=True).start()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.set_inputs_enabled(False)
        self.status_var.set("● Merekam... silakan menjawab")
        self.set_text(self.report, "")
        self.show_transcript("", "")
        self.update_timer()

    def stop_session(self):
        if not self.recording:
            return
        self.recording = False
        self.recording_seconds = time.time() - self.start_time
        self.stream.stop()
        self.stream.close()
        self.stop_flag.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("Memproses transkripsi akhir...")

    def audio_callback(self, indata, frames, time_info, status):
        # dipanggil sounddevice tiap ada blok audio baru: cukup masukkan ke antrean
        self.audio_q.put(indata[:, 0].copy())

    def worker_loop(self):
        """Thread terpisah: ambil audio dari antrean, transkripsi tiap STEP_SECONDS."""
        step_samples = int(STEP_SECONDS * SAMPLE_RATE)
        samples_since_step = 0

        while True:
            try:
                chunk = self.audio_q.get(timeout=0.1)
                self.engine.add_audio(chunk)
                self.all_audio.append(chunk)
                samples_since_step += len(chunk)
            except queue.Empty:
                if self.stop_flag.is_set():
                    break

            if samples_since_step >= step_samples:
                samples_since_step = 0
                if self.engine.step():
                    self.ui_q.put(("partial", self.engine.get_committed_text(),
                                   self.engine.get_partial_text(),
                                   self.engine.last_latency, self.engine.last_rtf))

        self.engine.step(final=True)   # proses sisa audio, semua jadi final
        self.ui_q.put(("final",))

    def update_timer(self):
        if self.recording:
            s = int(time.time() - self.start_time)
            self.timer_var.set(f"Waktu: {s // 60:02d}:{s % 60:02d}")
            self.root.after(200, self.update_timer)

    # ------------------------------------------------------ update tampilan
    def poll_ui_queue(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                kind = msg[0]
                if kind == "ready":
                    self.status_var.set("Model siap. Klik 'Mulai Menjawab'.")
                    self.start_btn.config(state="normal")
                elif kind == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Error", msg[1])
                elif kind == "partial":
                    _, final_text, partial_text, latency, rtf = msg
                    self.show_transcript(final_text, partial_text)
                    self.latency_var.set(f"Latency: {latency:.2f} dtk | RTF: {rtf:.2f}")
                elif kind == "final":
                    self.finish_session()
        except queue.Empty:
            pass
        self.root.after(100, self.poll_ui_queue)

    def finish_session(self):
        words = self.engine.get_words()
        self.show_transcript(self.engine.get_text(), "")

        q = self.current_question()
        result = analyze(words, self.lang_code(), q["keywords"], q["ideal_seconds"],
                         self.recording_seconds)
        self.set_text(self.report, format_report(result, q["text"]))

        path = self.save_session(q, result)
        self.status_var.set(f"Selesai. Sesi disimpan di {path}")
        self.start_btn.config(state="normal")
        self.set_inputs_enabled(True)

    def save_session(self, question, result):
        """Simpan audio (.wav) + data sesi (.json), berguna untuk bukti & evaluasi WER."""
        os.makedirs("sessions", exist_ok=True)
        name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        if self.all_audio:
            audio = np.concatenate(self.all_audio)
            sf.write(os.path.join("sessions", name + ".wav"), audio, SAMPLE_RATE)

        data = {
            "question": question["text"],
            "language": self.lang_code(),
            "model": self.model_size,
            "transcript": self.engine.get_text(),
            "analysis": result,
            "latency": self.engine.latency_summary(),
            "words": self.engine.get_words(),
        }
        with open(os.path.join("sessions", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return os.path.join("sessions", name)

    def on_close(self):
        if self.recording and self.stream is not None:
            self.stream.stop()
            self.stream.close()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                        help="ukuran model Whisper: tiny / base / small / medium")
    args = parser.parse_args()

    root = tk.Tk()
    InterviewCoachApp(root, args.model)
    root.mainloop()
