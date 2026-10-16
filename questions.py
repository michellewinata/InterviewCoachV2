# Bank pertanyaan interview (English & Bahasa Indonesia).
# keywords      : kata kunci yang diharapkan muncul di jawaban (dicek pakai substring, huruf kecil)
# ideal_seconds : durasi jawaban yang ideal (min, max) dalam detik

QUESTIONS = {
    "en": [
        {
            "text": "Tell me about yourself.",
            "keywords": ["university", "study", "project", "experience", "skill"],
            "ideal_seconds": (45, 120),
        },
        {
            "text": "Why do you want to work at our company?",
            "keywords": ["company", "learn", "contribute", "value", "goal"],
            "ideal_seconds": (30, 90),
        },
        {
            "text": "What is your greatest strength?",
            "keywords": ["strength", "example", "team", "skill", "result"],
            "ideal_seconds": (30, 90),
        },
        {
            "text": "Tell me about a challenge you faced in a project and how you solved it.",
            "keywords": ["project", "problem", "team", "solve", "result"],
            "ideal_seconds": (60, 150),
        },
        {
            "text": "Where do you see yourself in five years?",
            "keywords": ["goal", "learn", "grow", "career", "contribute"],
            "ideal_seconds": (30, 90),
        },
    ],
    "id": [
        {
            "text": "Ceritakan tentang diri kamu.",
            "keywords": ["kuliah", "universitas", "jurusan", "proyek", "pengalaman"],
            "ideal_seconds": (45, 120),
        },
        {
            "text": "Kenapa kamu tertarik bekerja di perusahaan ini?",
            "keywords": ["perusahaan", "belajar", "berkontribusi", "nilai", "tujuan"],
            "ideal_seconds": (30, 90),
        },
        {
            "text": "Apa kelebihan terbesar kamu?",
            "keywords": ["kelebihan", "contoh", "tim", "kemampuan", "hasil"],
            "ideal_seconds": (30, 90),
        },
        {
            "text": "Ceritakan tantangan yang pernah kamu hadapi di sebuah proyek dan cara kamu menyelesaikannya.",
            "keywords": ["proyek", "masalah", "tim", "solusi", "hasil"],
            "ideal_seconds": (60, 150),
        },
        {
            "text": "Kamu melihat dirimu di mana lima tahun lagi?",
            "keywords": ["tujuan", "belajar", "berkembang", "karier", "berkontribusi"],
            "ideal_seconds": (30, 90),
        },
    ],
}
