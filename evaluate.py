"""
Evaluasi teknis: WER + latency, dengan simulasi streaming dari file audio.

Contoh:
    # baseline (clean), Bahasa Indonesia
    python evaluate.py --folder data/clean --lang id --model base --out hasil_clean.csv

    # kondisi eksperimen: ditambah noise (SNR 10 dB)
    python evaluate.py --folder data/clean --lang id --model base --noise-snr 10 --out hasil_noise10.csv

    # untuk rekaman berbahasa Inggris
    python evaluate.py --folder data/clean_en --lang en --out hasil_en.csv

Struktur folder: tiap audio punya transkrip referensi dengan nama sama, misal
    data/clean/jawaban1.wav  +  data/clean/jawaban1.txt   (ditulis manual)
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
from faster_whisper.audio import decode_audio

from asr_engine import SAMPLE_RATE, StreamingTranscriber

STEP_SECONDS = 2.0   # sama dengan di app.py
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac")


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # buang tanda baca
    return text.split()


def align(ref, hyp):
    """Edit distance tingkat kata + backtrace, supaya tahu mana substitusi/deletion/insertion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # deletion
                          d[i][j - 1] + 1,          # insertion
                          d[i - 1][j - 1] + cost)   # match / substitution

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            kind = "correct" if ref[i - 1] == hyp[j - 1] else "substitution"
            ops.append((kind, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("deletion", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("insertion", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def add_noise(audio, snr_db):
    """Tambah white noise dengan SNR tertentu (dB). Seed tetap supaya hasil bisa diulang."""
    rng = np.random.default_rng(42)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def simulate_streaming(engine, audio):
    """Suapkan audio per STEP_SECONDS, sama seperti saat mic dipakai di app.py."""
    engine.reset()
    step = int(STEP_SECONDS * SAMPLE_RATE)
    for i in range(0, len(audio), step):
        engine.add_audio(audio[i:i + step])
        engine.step()
    lat, rtf = list(engine.latencies), list(engine.rtfs)   # sebelum pass final
    engine.step(final=True)
    return engine.get_text(), engine.latency_summary(lat, rtf)


def find_pairs(args):
    if args.audio:
        return [(args.audio, args.ref)]
    pairs = []
    for path in sorted(glob.glob(os.path.join(args.folder, "*"))):
        if path.lower().endswith(AUDIO_EXT):
            ref = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(ref):
                pairs.append((path, ref))
            else:
                print(f"[skip] {path}: transkrip referensi (.txt) tidak ada")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="satu file audio")
    parser.add_argument("--ref", help="transkrip referensi untuk --audio")
    parser.add_argument("--folder", help="folder berisi pasangan audio + .txt")
    parser.add_argument("--lang", default="id", choices=["id", "en"])
    parser.add_argument("--model", default="base")
    parser.add_argument("--noise-snr", type=float, default=None, help="tambah noise, SNR dalam dB")
    parser.add_argument("--out", default=None, help="simpan ringkasan ke CSV")
    args = parser.parse_args()

    if not args.folder and not (args.audio and args.ref):
        parser.error("isi --folder, atau --audio dan --ref")

    pairs = find_pairs(args)
    if not pairs:
        print("Tidak ada pasangan audio + referensi yang ditemukan.")
        return

    engine = StreamingTranscriber(model_size=args.model, language=args.lang)

    rows = []
    all_ops = []
    total_err = total_ref = 0

    for audio_path, ref_path in pairs:
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        if args.noise_snr is not None:
            audio = add_noise(audio, args.noise_snr)

        with open(ref_path, encoding="utf-8") as f:
            ref = normalize_text(f.read())

        text, lat = simulate_streaming(engine, audio)
        hyp = normalize_text(text)

        ops = align(ref, hyp)
        s = sum(1 for o in ops if o[0] == "substitution")
        d = sum(1 for o in ops if o[0] == "deletion")
        i = sum(1 for o in ops if o[0] == "insertion")
        wer = (s + d + i) / len(ref) if ref else 0.0

        total_err += s + d + i
        total_ref += len(ref)
        all_ops += [(os.path.basename(audio_path),) + o for o in ops if o[0] != "correct"]

        rows.append({
            "file": os.path.basename(audio_path),
            "duration_s": round(len(audio) / SAMPLE_RATE, 1),
            "ref_words": len(ref), "S": s, "D": d, "I": i,
            "WER": round(wer, 3),
            "mean_latency_s": round(lat["mean_latency"], 2),
            "max_latency_s": round(lat["max_latency"], 2),
            "mean_RTF": round(lat["mean_rtf"], 3),
        })
        print(f"{rows[-1]['file']}: WER={wer:.1%} (S={s} D={d} I={i} dari {len(ref)} kata) | "
              f"latency rata-rata {lat['mean_latency']:.2f} dtk | RTF {lat['mean_rtf']:.2f}")

    overall = total_err / total_ref if total_ref else 0.0
    kondisi = "clean" if args.noise_snr is None else f"noise SNR {args.noise_snr:g} dB"
    print(f"\n=== Ringkasan ({kondisi}, bahasa {args.lang}, model {args.model}) ===")
    print(f"WER total : {overall:.1%}  ({total_err} error / {total_ref} kata referensi)")
    print(f"Latency   : rata-rata {np.mean([r['mean_latency_s'] for r in rows]):.2f} dtk, "
          f"maks {max(r['max_latency_s'] for r in rows):.2f} dtk")
    print(f"RTF       : rata-rata {np.mean([r['mean_RTF'] for r in rows]):.2f} (< 1 berarti lebih cepat dari real-time)")

    print("\nContoh error untuk error analysis (maks 25):")
    print(f"{'file':<18}{'jenis':<14}{'referensi':<18}{'hasil ASR'}")
    for file, kind, r, h in all_ops[:25]:
        print(f"{file:<18}{kind:<14}{r or '-':<18}{h or '-'}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRingkasan per file disimpan di {args.out}")


if __name__ == "__main__":
    main()
