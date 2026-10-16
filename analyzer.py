"""
Analisis jawaban setelah sesi selesai.
Semua dihitung dari kata + timestamp hasil ASR, jadi nggak perlu model tambahan.
"""

import re

PAUSE_THRESHOLD = 1.0   # jeda >= 1 detik antar kata dianggap pause

# bunyi ragu-ragu: um, uh, umm, eee, ehm, hmm, dst. (berlaku untuk kedua bahasa)
HESITATION = re.compile(r"^(u+h+m*|u+m+|e{2,}|e+h+m*|e+m+|e+r+m*|h+m+|m{2,})$")

# kata/frasa yang sering jadi filler (silakan diubah sesuai kebutuhan)
FILLER_WORDS = {
    "en": {"like", "basically", "actually"},
    "id": {"anu", "kayak", "gitu", "ya"},
}
FILLER_PHRASES = {
    "en": ["you know", "i mean", "kind of", "sort of"],
    "id": ["apa ya", "maksudnya", "gimana ya"],
}


def normalize(word):
    return re.sub(r"[^\w']", "", word.lower())


def count_fillers(words, language):
    tokens = [normalize(w["word"]) for w in words]
    counts = {}
    i = 0
    while i < len(tokens):
        # cek frasa dulu (misal "you know" / "apa ya")
        cocok = False
        for phrase in FILLER_PHRASES.get(language, []):
            parts = phrase.split()
            if tokens[i:i + len(parts)] == parts:
                counts[phrase] = counts.get(phrase, 0) + 1
                i += len(parts)
                cocok = True
                break
        if cocok:
            continue

        t = tokens[i]
        if HESITATION.match(t) or t in FILLER_WORDS.get(language, set()):
            counts[t] = counts.get(t, 0) + 1
        i += 1
    return counts


def find_pauses(words, threshold=PAUSE_THRESHOLD):
    pauses = []
    for prev, cur in zip(words, words[1:]):
        gap = cur["start"] - prev["end"]
        if gap >= threshold:
            pauses.append({"at": prev["end"], "duration": gap})
    return pauses


def check_keywords(text, keywords):
    text = text.lower()
    found = [k for k in keywords if k in text]
    missing = [k for k in keywords if k not in text]
    return found, missing


def analyze(words, language, keywords, ideal_seconds, recording_seconds):
    result = {
        "total_words": len(words),
        "recording_seconds": recording_seconds,
        "speaking_seconds": 0.0,
        "wpm": 0.0,
        "filler_count": 0,
        "filler_percent": 0.0,
        "filler_detail": {},
        "pause_count": 0,
        "total_pause_seconds": 0.0,
        "longest_pause": 0.0,
        "pauses": [],
        "keywords_found": [],
        "keywords_missing": list(keywords),
        "feedback": [],
    }

    if not words:
        result["feedback"].append("Tidak ada ucapan yang terdeteksi. Cek mikrofon lalu coba lagi.")
        return result

    text = " ".join(w["word"] for w in words)

    # durasi bicara = dari kata pertama sampai kata terakhir
    speaking = words[-1]["end"] - words[0]["start"]
    result["speaking_seconds"] = speaking
    if speaking > 0:
        result["wpm"] = len(words) / (speaking / 60)

    fillers = count_fillers(words, language)
    result["filler_detail"] = fillers
    result["filler_count"] = sum(fillers.values())
    result["filler_percent"] = 100 * result["filler_count"] / len(words)

    pauses = find_pauses(words)
    result["pauses"] = pauses
    result["pause_count"] = len(pauses)
    result["total_pause_seconds"] = sum(p["duration"] for p in pauses)
    result["longest_pause"] = max((p["duration"] for p in pauses), default=0.0)

    found, missing = check_keywords(text, keywords)
    result["keywords_found"] = found
    result["keywords_missing"] = missing

    result["feedback"] = make_feedback(result, ideal_seconds)
    return result


def make_feedback(r, ideal_seconds):
    tips = []
    low, high = ideal_seconds

    # durasi jawaban
    if r["recording_seconds"] < low:
        tips.append(f"Jawabanmu cukup singkat ({r['recording_seconds']:.0f} dtk). "
                    f"Idealnya sekitar {low}-{high} detik, coba tambahkan contoh atau detail.")
    elif r["recording_seconds"] > high:
        tips.append(f"Jawabanmu agak panjang ({r['recording_seconds']:.0f} dtk). "
                    f"Idealnya sekitar {low}-{high} detik, coba lebih ringkas.")
    else:
        tips.append("Durasi jawaban sudah pas.")

    # kecepatan bicara (patokan kasar 100-160 kata/menit)
    if r["wpm"] > 160:
        tips.append(f"Kecepatan bicara {r['wpm']:.0f} WPM, agak cepat. Coba lebih santai.")
    elif r["wpm"] < 100:
        tips.append(f"Kecepatan bicara {r['wpm']:.0f} WPM, agak lambat. Coba lebih lancar.")
    else:
        tips.append(f"Kecepatan bicara {r['wpm']:.0f} WPM, sudah nyaman didengar.")

    # filler
    if r["filler_percent"] > 5:
        tips.append(f"Filler words cukup banyak ({r['filler_count']} kali, "
                    f"{r['filler_percent']:.1f}% dari total kata). Coba berhenti sejenak "
                    f"daripada mengisi dengan 'um/eee'.")
    else:
        tips.append("Filler words masih terkendali.")

    # pause
    if r["pause_count"] >= 3 or r["longest_pause"] >= 3:
        tips.append(f"Ada {r['pause_count']} jeda panjang (terlama {r['longest_pause']:.1f} dtk). "
                    f"Siapkan kerangka jawaban supaya alurnya lebih mengalir.")
    else:
        tips.append("Jeda antar kalimat masih wajar.")

    # keyword
    if r["keywords_missing"]:
        tips.append("Kata kunci yang belum muncul: " + ", ".join(r["keywords_missing"]) + ".")
    else:
        tips.append("Semua kata kunci sudah muncul di jawaban.")
    return tips


def format_report(r, question_text):
    lines = [
        f"Pertanyaan : {question_text}",
        "",
        f"Durasi rekaman      : {r['recording_seconds']:.1f} detik",
        f"Durasi bicara       : {r['speaking_seconds']:.1f} detik",
        f"Jumlah kata         : {r['total_words']}",
        f"Kecepatan bicara    : {r['wpm']:.0f} kata/menit (WPM)",
        f"Filler words        : {r['filler_count']} ({r['filler_percent']:.1f}%)",
    ]
    if r["filler_detail"]:
        detail = ", ".join(f"{k} x{v}" for k, v in r["filler_detail"].items())
        lines.append(f"   rincian          : {detail}")
    lines.append(f"Pause (>= {PAUSE_THRESHOLD:.0f} dtk)   : {r['pause_count']} kali, "
                 f"total {r['total_pause_seconds']:.1f} dtk, terlama {r['longest_pause']:.1f} dtk")
    lines.append(f"Keyword ditemukan   : {', '.join(r['keywords_found']) or '-'}")
    lines.append(f"Keyword belum muncul: {', '.join(r['keywords_missing']) or '-'}")
    lines.append("")
    lines.append("Feedback:")
    for t in r["feedback"]:
        lines.append(f"  - {t}")
    return "\n".join(lines)
