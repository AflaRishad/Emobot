"""
EmoBot v13 — Fixed manga recommendations + user bubble visibility
=================================================================
Key fixes vs v12:
  1. chatbot.html chip clicks now always show user bubbles (showUserBubble=true)
  2. Manga search: Jikan genre IDs used for precise filtering
  3. Manga search: subgenre → genre_id mapping so "romance manga" gives
     actual romance results, not generic popular manga
  4. MangaDex query uses proper tag UUIDs for accurate filtering
  5. Better subgenre normalization for manga paths
  6. Tone-aware manga fallback curated lists per subgenre
"""

import re
import json
import time
import base64
import math

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, session
from tensorflow.keras.models import load_model
from urllib.parse import quote_plus

app = Flask(__name__)
app.secret_key = "emobot_v13_ollama"

# ══════════════════════════════════════════════════════════════════
#  OLLAMA CONFIGURATION
# ══════════════════════════════════════════════════════════════════

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3"

SYSTEM_PROMPT = """You are EmoBot, a book-recommendation chatbot. Be VERY brief.
User emotion detected: {emotion}.

GOAL: Ask 3-4 short questions to collect preferences, then output PREFS_JSON.

QUESTIONS TO ASK (in this order):
1. Media type: books / manga / comics
2. Genre (books only): fiction / non fiction
3. Subgenre (pick options that match {emotion}):
   fiction: romance, fantasy, thriller, sci fi, mystery, horror, adventure, historical, dark romance, comedy
   non-fiction: psychology, self development, biography, history, science, true crime, philosophy, memoir
   manga: shonen, shojo, seinen, josei, isekai, romance manga, action manga, fantasy manga, horror manga, comedy manga
   comics: superhero, graphic novel, indie comics, sci fi comics, fantasy comics, horror comics, memoir comics
4. Tone: uplifting / emotional / intense / fun / calming / dark / inspiring / romantic

RULES:
- First reply: ONE sentence about {emotion} + ask media type. End with: 👉 **Books** · **Manga** · **Comics**
- Every reply after: ONE short question + chips on next line. Format: 👉 **A** · **B** · **C** · **D**
- MAX 2 lines per reply. No extra text.
- Always mention the detected emotion in the first reply and use it to frame the next question.
- Match the tone of questions to emotion: happy → upbeat/playful, sad → gentle/reassuring, angry → energetic/focused, neutral → friendly/open.
- Recommend options that suit {emotion}: happy→fun/adventure, sad→healing/romance, angry→thriller/dark, neutral→variety
- After tone is collected, output wrap-up then PREFS_JSON on its own line:
  PREFS_JSON:{{"media_type":"VALUE","genre":"VALUE","subgenre":"VALUE","tone":"VALUE","theme":"does not matter","pacing":"does not matter","length":"does not matter"}}
- Use "does not matter" for anything not asked.
- STOP after outputting PREFS_JSON. Do not add anything else.

EXAMPLE (emotion=happy):
Bot: You seem happy today! 😊 What do you want to read?
     👉 **Books** · **Manga** · **Comics**
User: Books
Bot: Fiction or non-fiction?
     👉 **Fiction** · **Non-fiction**
User: Fiction
Bot: Which genre?
     👉 **Adventure** · **Comedy** · **Fantasy** · **Romance**
User: Adventure
Bot: What vibe?
     👉 **Fun** · **Uplifting** · **Intense** · **Inspiring**
User: Fun
Bot: Perfect, fetching your picks! 📚
     PREFS_JSON:{{"media_type":"books","genre":"fiction","subgenre":"adventure","tone":"fun","theme":"does not matter","pacing":"does not matter","length":"does not matter"}}
"""


def call_ollama(history: list, emotion: str) -> str | None:
    system = SYSTEM_PROMPT.format(emotion=emotion)
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}] + history[-20:],
        "stream":   False,
        "options":  {"temperature": 0.6, "num_predict": 250},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        print(f"[Ollama reply]\n{content}\n{'─'*60}")
        return content
    except Exception as e:
        print(f"[Ollama error] {e}")
        return None


call_llm = call_ollama


# ══════════════════════════════════════════════════════════════════
#  VALUE NORMALISATION
# ══════════════════════════════════════════════════════════════════

_NORMALIZE = {
    "science fiction": "sci fi", "scifi": "sci fi", "sci-fi": "sci fi",
    "non-fiction": "non fiction", "nonfiction": "non fiction",
    "self help": "self development", "self-help": "self development",
    "self improvement": "self development",
    "slow burn": "slow and deep", "slow-burn": "slow and deep",
    "fast-paced": "fast paced", "action packed": "fast paced",
    "slice of life": "slice of life manga", "shounen": "shonen", "shoujo": "shojo",
    "graphic novels": "graphic novel", "romcom": "comedy", "rom-com": "comedy",
    "chick lit": "romance", "suspense": "thriller", "crime fiction": "thriller",
    "does not matter": "does not matter", "any": "does not matter",
    "anything": "does not matter", "none": "does not matter",
    "n/a": "does not matter", "": "does not matter",
    "non fiction": "non fiction",
    # Chip label → internal value normalization for manga
    "romance": "romance manga",
    "action":  "action manga",
    "serious": "intense",
    "light":   "fun",
    "slow unfolding": "calming",
    "fast-paced": "intense",
    "dark romance": "dark romance",
    "intense": "intense",
}

def _norm_val(v: str) -> str:
    v = (v or "").lower().strip()
    return _NORMALIZE.get(v, v)


# ══════════════════════════════════════════════════════════════════
#  PREFS EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_prefs(text: str):
    clean_lines = []
    prefs = None

    for line in text.split("\n"):
        stripped = line.strip()
        if "PREFS_JSON" in stripped.upper():
            start = stripped.find("{")
            end   = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = stripped[start:end + 1]
                try:
                    prefs = json.loads(raw)
                    print(f"[extract_prefs] Parsed OK: {prefs}")
                except Exception as e:
                    print(f"[extract_prefs] Parse failed: {e!r}  raw={raw!r}")
        else:
            clean_lines.append(line)

    if prefs is None:
        m = re.search(r'PREFS_JSON\s*[:\-]?\s*(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                prefs = json.loads(m.group(1))
                print(f"[extract_prefs] DOTALL parsed OK: {prefs}")
            except Exception as e:
                print(f"[extract_prefs] DOTALL failed: {e!r}")

    if prefs:
        prefs = {k: _norm_val(str(v)) for k, v in prefs.items()}
        # Fix: if media_type is manga and subgenre has no "manga" suffix, try to add it
        if prefs.get("media_type") == "manga":
            sg = prefs.get("subgenre", "")
            # Map bare subgenres to manga-specific ones if needed
            manga_subgenre_map = {
                "romance":   "romance manga",
                "action":    "action manga",
                "horror":    "horror manga",
                "comedy":    "comedy manga",
                "fantasy":   "fantasy manga",
                "isekai":    "isekai",
                "shonen":    "shonen",
                "shojo":     "shojo",
                "seinen":    "seinen",
                "josei":     "josei",
            }
            if sg in manga_subgenre_map:
                prefs["subgenre"] = manga_subgenre_map[sg]

    clean = "\n".join(clean_lines).strip()
    clean = re.sub(r'PREFS_JSON\s*[:\-]?\s*\{[^}]*\}', '', clean, flags=re.IGNORECASE).strip()
    return clean, prefs


# ══════════════════════════════════════════════════════════════════
#  EMOTION DETECTION
# ══════════════════════════════════════════════════════════════════

MODEL_PATH   = "model.keras"
LABELS_4     = ["angry", "happy", "neutral", "sad"]
LABELS_7     = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

face_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
emotion_model = load_model(MODEL_PATH)
EMOTION_LABELS = LABELS_4 if emotion_model.output_shape[-1] == 4 else LABELS_7
print("Model loaded ✓  Labels:", EMOTION_LABELS)


def _preprocess(face):
    sh = emotion_model.input_shape
    th, tw, ch = sh[1], sh[2], sh[3]
    if ch == 1:
        if face.ndim == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (tw, th))
        face = cv2.createCLAHE(2.0, (8, 8)).apply(face).astype("float32") / 255.0
        face = np.expand_dims(face, -1)
    else:
        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        face = cv2.resize(face, (tw, th))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(face, 0)


def detect_emotion(image_data: str) -> str:
    try:
        raw   = base64.b64decode(image_data.split(",", 1)[1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return "neutral"
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = frame[max(y - int(h*.18), 0): y + h + int(h*.18),
                         max(x - int(w*.15), 0): x + w + int(w*.15)]
        else:
            fh, fw = frame.shape[:2]
            face = frame[(fh - int(fh*.65))//2: (fh + int(fh*.65))//2,
                         (fw - int(fw*.65))//2: (fw + int(fw*.65))//2]
        preds = emotion_model.predict(_preprocess(face), verbose=0)[0]
        return EMOTION_LABELS[int(np.argmax(preds))]
    except Exception as e:
        print("Emotion error:", e)
        return "neutral"


# ══════════════════════════════════════════════════════════════════
#  BOOK SEARCH
# ══════════════════════════════════════════════════════════════════

def _n(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (t or "").lower())).strip()


BLOCK_TITLE_WORDS = {
    "journal", "proceedings", "conference", "symposium", "workshop",
    "transactions", "bulletin", "newsletter", "gazette", "periodical",
    "magazine", "newspaper", "dictionary", "encyclopedia", "encyclopaedia",
    "thesaurus", "glossary", "lexicon", "textbook", "workbook", "coursebook",
    "study guide", "question bank", "exam prep", "test prep", "practice test",
    "past papers", "revision guide", "syllabus", "curriculum", "lesson plan",
    "lecture notes", "manual", "handbook", "reference guide", "annual report",
    "white paper", "working paper", "technical report", "dissertation",
    "thesis", "monograph series", "collected papers", "selected papers",
    "edited by", "series editor", "volume editor",
}

BLOCK_CAT_WORDS = {
    "language arts", "disciplines", "education", "study aids",
    "library & information", "periodicals", "journals", "newspapers",
    "mathematics", "technology & engineering", "statistics",
    "reference", "academic", "scholarly",
}

BLOCK_DESC_PHRASES = [
    "peer-reviewed", "peer reviewed", "in this paper", "this paper presents",
    "this study", "this article", "journal article", "research article",
    "conference paper", "proceedings of", "call for papers", "special issue",
    "issn", "doi:", "volume \\d+", "issue \\d+", "abstract:", "keywords:",
    "methodology section", "literature review", "et al\\.", "\\(eds?\\.\\)",
]
_BLOCK_DESC_RE = re.compile("|".join(BLOCK_DESC_PHRASES), re.IGNORECASE)
_VOL_ISSUE_RE  = re.compile(r'\bvol\.?\s*\d+|\bno\.?\s*\d+|\bissue\s*\d+|\(\d{4}\)', re.IGNORECASE)


def _is_bad(title: str, cats: list, desc: str) -> bool:
    t_low = (title or "").lower()
    for w in BLOCK_TITLE_WORDS:
        if w in t_low:
            return True
    if _VOL_ISSUE_RE.search(title or ""):
        return True
    cat_str = " ".join(cats or []).lower()
    for w in BLOCK_CAT_WORDS:
        if w in cat_str:
            return True
    if _BLOCK_DESC_RE.search(desc or ""):
        return True
    return False


EMOTION_KEYWORDS = {
    "happy":    ["feel-good", "uplifting", "fun", "adventure", "comedy", "heartwarming"],
    "sad":      ["heartfelt", "healing", "emotional", "comforting", "hopeful", "romantic"],
    "angry":    ["gripping", "intense", "dark", "revenge", "thriller", "action-packed"],
    "neutral":  ["popular", "bestseller", "acclaimed", "must-read"],
    "fear":     ["cozy", "mystery", "comforting", "suspense", "psychological"],
    "surprise": ["unexpected", "fantasy", "sci-fi", "twist", "adventure", "mind-bending"],
    "disgust":  ["literary", "philosophy", "travel", "insight", "thought-provoking"],
}

SUBGENRE_SUBJECTS = {
    "romance":          "romance love",
    "fantasy":          "fantasy magic",
    "thriller":         "thriller suspense",
    "sci fi":           "science fiction",
    "mystery":          "mystery detective",
    "horror":           "horror scary",
    "adventure":        "adventure action",
    "historical":       "historical fiction",
    "contemporary":     "contemporary fiction",
    "dark romance":     "dark romance",
    "comedy":           "humor comedy fiction",
    "literary fiction": "literary fiction",
    "psychology":       "psychology mind behaviour",
    "self development": "self-help personal growth",
    "biography":        "biography memoir",
    "history":          "history narrative",
    "science":          "popular science",
    "true crime":       "true crime",
    "health":           "health wellness",
    "travel":           "travel narrative",
    "business":         "business entrepreneurship",
    "philosophy":       "philosophy",
    "memoir":           "memoir autobiography",
    "politics":         "politics society",
}

TONE_KEYWORDS = {
    "uplifting":  "uplifting hopeful inspiring",
    "emotional":  "emotional moving heartfelt",
    "intense":    "intense gripping dark",
    "fun":        "fun lighthearted funny",
    "calming":    "cozy comforting peaceful",
    "dark":       "dark grim psychological",
    "inspiring":  "inspiring motivational",
    "romantic":   "romantic love passionate",
}


def _is_effective_value(value: str) -> bool:
    v = (value or "").strip().lower()
    return bool(v and v not in {"does not matter", "doesnt matter", "doesn't matter", "any", "none", "n/a"})


def _normalize_query_phrase(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).strip()


def _build_book_queries(emotion: str, tone: str, genre: str, subgenre: str) -> list[str]:
    emo_kw  = _normalize_query_phrase(" ".join(EMOTION_KEYWORDS.get(emotion, ["popular"])))
    tone_kw = _normalize_query_phrase(TONE_KEYWORDS.get(tone, tone)) if _is_effective_value(tone) else ""
    sub_kw  = _normalize_query_phrase(SUBGENRE_SUBJECTS.get(subgenre, subgenre)) if _is_effective_value(subgenre) else ""
    g_kw    = _normalize_query_phrase(genre) if _is_effective_value(genre) else ""

    neg = "-magazine -journal -proceedings -conference -textbook -workbook"

    q1 = " ".join(filter(None, [sub_kw, tone_kw, emo_kw, g_kw, "bestselling popular", neg])).strip()
    q2 = " ".join(filter(None, [sub_kw, g_kw, "bestselling popular", neg])).strip()
    q3 = " ".join(filter(None, [sub_kw or g_kw or "fiction novel", "bestseller", neg])).strip()

    if not q1:
        q1 = "bestselling popular -magazine -journal -proceedings -conference -textbook -workbook"
    if not q2:
        q2 = q1
    if not q3:
        q3 = q1

    return [q1, q2, q3]


def _get_thumbnail(info: dict) -> str:
    links = info.get("imageLinks") or {}
    for key in ("extraLarge", "large", "medium", "thumbnail", "smallThumbnail"):
        url = links.get(key)
        if url:
            return url.replace("http://", "https://")
    return ""


def _score_book(info: dict) -> float:
    avg = info.get("averageRating") or 0
    cnt = info.get("ratingsCount") or 0
    pop = math.log10(cnt + 1)
    return avg * pop * 10


def _preference_score(info: dict, subgenre: str, genre: str, tone: str, emotion: str) -> float:
    text = " ".join([
        info.get("title", ""),
        info.get("description", ""),
        info.get("category", ""),
        " ".join(info.get("categories") or []) if isinstance(info.get("categories"), list) else "",
    ]).lower()
    score = 0.0

    if _is_effective_value(subgenre):
        term = subgenre.lower()
        if term in text:
            score += 30
        for token in SUBGENRE_SUBJECTS.get(subgenre, subgenre).split():
            if token and token in text:
                score += 8

    if _is_effective_value(genre):
        if genre.lower() in text:
            score += 20

    if _is_effective_value(tone):
        for token in TONE_KEYWORDS.get(tone, tone).split():
            if token and token in text:
                score += 5

    for token in EMOTION_KEYWORDS.get(emotion, []):
        if token and token in text:
            score += 3

    return score


def _fetch_google_books(query: str, start: int = 0, max_results: int = 40) -> list:
    try:
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": query, "maxResults": max_results, "startIndex": start,
                "printType": "books", "orderBy": "relevance", "langRestrict": "en",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return r.json().get("items") or []
    except Exception as e:
        print("Google Books error:", e)
        return []


def _openlibrary_fallback(subgenre: str, genre: str = "", n: int = 5) -> list:
    if _is_effective_value(subgenre):
        q = SUBGENRE_SUBJECTS.get(subgenre, subgenre)
    elif _is_effective_value(genre):
        q = genre
    else:
        q = "popular fiction"

    try:
        r = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": q, "limit": 40, "language": "eng"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        out = []
        for d in (r.json().get("docs") or []):
            title   = d.get("title") or ""
            authors = ", ".join(d.get("author_name") or [])[:100]
            if not title or not authors:
                continue
            cover_id  = d.get("cover_i")
            thumbnail = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""
            out.append({
                "title":         title,
                "authors":       authors,
                "description":   "",
                "category":      "",
                "averageRating": None,
                "ratingsCount":  None,
                "thumbnail":     thumbnail,
                "link":          f"https://www.google.com/search?q={quote_plus(title+' '+authors)}",
                "alt_link":      f"https://openlibrary.org/search?q={quote_plus(title)}",
                "goodreads":     f"https://www.goodreads.com/search?q={quote_plus(title)}",
            })
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print("OpenLibrary error:", e)
        return []


def search_books(prefs: dict, emotion: str, n: int = 5) -> list:
    sg    = prefs.get("subgenre", "does not matter")
    genre = prefs.get("genre", "does not matter")
    tone  = prefs.get("tone", "does not matter")

    queries = _build_book_queries(emotion, tone, genre, sg)

    day_seed = int(time.time() // 86400)
    offsets = [(day_seed % 4) * 10, ((day_seed+1) % 4) * 10, 0]

    candidates: dict[str, dict] = {}

    for i, query in enumerate(queries):
        if len(candidates) >= n * 4:
            break
        items = _fetch_google_books(query, start=offsets[i])
        for item in items:
            info    = item.get("volumeInfo", {}) or {}
            title   = info.get("title", "") or ""
            authors = ", ".join(info.get("authors", []) or [])
            if not title or not authors:
                continue
            cats    = info.get("categories", []) or []
            desc    = info.get("description", "") or ""
            cnt     = info.get("ratingsCount", 0) or 0
            avg     = info.get("averageRating", 0) or 0
            if _is_bad(title, cats, desc):
                continue
            if cnt < 20:
                continue
            if avg > 0 and avg < 3.0:
                continue
            key = _n(title)
            score = _score_book(info) + _preference_score(info, sg, genre, tone, emotion)
            if key not in candidates or score > candidates[key]["_score"]:
                thumb = _get_thumbnail(info)
                candidates[key] = {
                    "title":         title,
                    "authors":       authors,
                    "description":   (desc[:300] + "…") if len(desc) > 300 else desc,
                    "category":      cats[0] if cats else "",
                    "averageRating": avg or None,
                    "ratingsCount":  cnt or None,
                    "thumbnail":     thumb,
                    "link":          f"https://www.google.com/search?q={quote_plus(title+' '+authors)}",
                    "alt_link":      f"https://openlibrary.org/search?q={quote_plus(title)}",
                    "goodreads":     f"https://www.goodreads.com/search?q={quote_plus(title)}",
                    "_score":        score,
                }

    ranked = sorted(candidates.values(), key=lambda x: x["_score"], reverse=True)
    results = [{k: v for k, v in b.items() if k != "_score"} for b in ranked[:n]]

    if len(results) < n:
        results += _openlibrary_fallback(sg, genre, n - len(results))

    return results[:n]


# ══════════════════════════════════════════════════════════════════
#  MANGA SEARCH — Fully rewritten for accurate genre matching
# ══════════════════════════════════════════════════════════════════

# Jikan (MyAnimeList) genre IDs for precise filtering
# https://api.jikan.moe/v4/genres/manga
_JIKAN_GENRE_IDS = {
    "romance manga":   1,    # Romance
    "action manga":    1,    # Action (id=1 is Action on MAL)
    "comedy manga":    4,    # Comedy
    "horror manga":    14,   # Horror
    "fantasy manga":   10,   # Fantasy
    "shonen":          27,   # Shounen
    "shojo":           25,   # Shoujo
    "seinen":          41,   # Seinen
    "josei":           43,   # Josei
    "isekai":          62,   # Isekai
    "slice of life manga": 36, # Slice of Life
    "drama manga":     8,    # Drama
    "mystery manga":   7,    # Mystery
    "psychological manga": 40, # Psychological
    "sci fi manga":    24,   # Sci-Fi
    "sports manga":    30,   # Sports
}

# Corrected Jikan genre IDs (MAL uses different IDs for manga genres)
_JIKAN_MANGA_GENRE_IDS = {
    "romance manga":    22,   # Romance
    "action manga":     1,    # Action
    "comedy manga":     4,    # Comedy
    "horror manga":     14,   # Horror
    "fantasy manga":    10,   # Fantasy
    "shonen":           27,   # Shounen
    "shojo":            25,   # Shoujo
    "seinen":           41,   # Seinen
    "josei":            43,   # Josei
    "isekai":           62,   # Isekai
    "slice of life manga": 36, # Slice of Life
    "drama manga":      8,    # Drama
    "sci fi manga":     24,   # Sci-Fi
    "sports manga":     30,   # Sports
    "psychological manga": 40, # Psychological
    "dark romance":     22,   # Romance (closest match)
    "intense":          1,    # Action/Intense
    "does not matter":  None,
}

# MangaDex tag UUIDs for precise filtering
_MANGADEX_TAGS = {
    "romance manga":    "423e2eae-a7a2-4a8b-ac03-a8351462d71d",  # Romance
    "action manga":     "391b0423-d847-456f-aff0-8b0cfc03066b",  # Action
    "comedy manga":     "4d32cc48-9f00-4cca-9b5a-a839f0764984",  # Comedy
    "horror manga":     "cdad7e68-1419-41dd-bdce-27753074a640",  # Horror
    "fantasy manga":    "cdc58593-87dd-415e-bbc0-2ec27bf404cc",  # Fantasy
    "shonen":           "27f2c31a-27a5-4367-8354-7bc18f6bd140",  # Shonen
    "shojo":            "a3c67850-4684-404e-9b7f-c69850ee5da6",  # Shojo
    "seinen":           "caaa44eb-cd40-4177-b930-79d3ef2afe87",  # Seinen
    "josei":            "ddefd648-5140-4e5f-ba18-4eca4071d19b",  # Josei
    "isekai":           "ace04997-f6bd-436e-b261-779182193d3d",  # Isekai
    "slice of life manga": "e5301a23-ebd9-49dd-a0cb-2add944c7fe9", # Slice of Life
    "psychological manga": "3b60b75c-a2d7-4860-ab56-05f391bb889c", # Psychological
    "sci fi manga":     "256c8bd9-4904-4360-bf4f-508a76d67183",  # Sci-Fi
    "drama manga":      "b9af3a63-f058-46de-a9a0-e0c13906197a",  # Drama
    "dark romance":     "423e2eae-a7a2-4a8b-ac03-a8351462d71d",  # Romance (dark tone)
}

# Demographic UUIDs for MangaDex
_MANGADEX_DEMOGRAPHIC = {
    "shonen":  "shounen",
    "shojo":   "shoujo",
    "seinen":  "seinen",
    "josei":   "josei",
}

# Curated fallback lists per subgenre — these actually match the genre
_MANGA_CURATED: dict[str, list[dict]] = {
    "romance manga": [
        {"title": "Horimiya",                   "authors": "HAGIWARA Daisuke",     "description": "A popular girl and a quiet boy discover each other's hidden sides and fall in love.", "averageRating": 8.7, "ratingsCount": 290000},
        {"title": "Kaguya-sama: Love Is War",   "authors": "Akasaka Aka",          "description": "Two brilliant student council members too proud to confess — so they scheme to make the other confess first.", "averageRating": 8.7, "ratingsCount": 420000},
        {"title": "My Dress-Up Darling",        "authors": "Fukuda Shinichi",       "description": "A skilled craftsman helps his gyaru classmate with cosplay costumes and romance blooms.", "averageRating": 8.4, "ratingsCount": 250000},
        {"title": "Ao Haru Ride",               "authors": "Sakisaka Io",           "description": "A girl reunites with her first love in high school but he's changed dramatically.", "averageRating": 8.2, "ratingsCount": 180000},
        {"title": "A Silent Voice",             "authors": "Yoshitoki Oima",        "description": "A former bully seeks redemption from the deaf girl he tormented — emotional and redemptive.", "averageRating": 9.0, "ratingsCount": 320000},
    ],
    "dark romance": [
        {"title": "Berserk",                    "authors": "Miura Kentaro",         "description": "A dark, epic fantasy following a mercenary warrior through a brutal, unforgiving world.", "averageRating": 9.4, "ratingsCount": 480000},
        {"title": "Oyasumi Punpun",             "authors": "Asano Inio",            "description": "A deeply psychological slice-of-life manga about a boy's turbulent growth into adulthood.", "averageRating": 9.0, "ratingsCount": 250000},
        {"title": "Flowers of Evil",            "authors": "Oshimi Shuzo",          "description": "A disturbing psychological romance that subverts expectations at every turn.", "averageRating": 8.0, "ratingsCount": 140000},
        {"title": "Welcome to the NHK",        "authors": "Tatsuhiko Takimoto",    "description": "A NEET's struggle with paranoia and isolation — dark but ultimately hopeful.", "averageRating": 8.4, "ratingsCount": 160000},
        {"title": "I Am a Hero",               "authors": "Hanazawa Kengo",        "description": "A manga artist survives a zombie apocalypse in this intense, dark psychological thriller.", "averageRating": 8.3, "ratingsCount": 90000},
    ],
    "action manga": [
        {"title": "Attack on Titan",            "authors": "Isayama Hajime",        "description": "Humanity fights for survival against giant humanoid titans in a walled city — epic and intense.", "averageRating": 9.0, "ratingsCount": 600000},
        {"title": "Demon Slayer",               "authors": "Gotouge Koyoharu",     "description": "A boy becomes a demon slayer to save his sister and avenge his family.", "averageRating": 8.6, "ratingsCount": 450000},
        {"title": "Jujutsu Kaisen",             "authors": "Akutami Gege",          "description": "A student joins a secret school for sorcerers to fight cursed spirits.", "averageRating": 8.7, "ratingsCount": 500000},
        {"title": "Hunter x Hunter",            "authors": "Togashi Yoshihiro",     "description": "A boy sets out to become a Hunter and find his father in a dangerous world.", "averageRating": 9.1, "ratingsCount": 520000},
        {"title": "Fullmetal Alchemist",        "authors": "Arakawa Hiromu",        "description": "Two brothers use alchemy to search for the Philosopher's Stone to restore their bodies.", "averageRating": 9.1, "ratingsCount": 560000},
    ],
    "seinen": [
        {"title": "Vagabond",                   "authors": "Inoue Takehiko",        "description": "A fictionalized account of legendary swordsman Miyamoto Musashi's quest for enlightenment.", "averageRating": 9.2, "ratingsCount": 350000},
        {"title": "20th Century Boys",         "authors": "Urasawa Naoki",         "description": "Friends try to stop a cult that seems to follow a childhood comic they drew.", "averageRating": 9.1, "ratingsCount": 280000},
        {"title": "Monster",                    "authors": "Urasawa Naoki",         "description": "A surgeon pursues a serial killer he once saved across Europe — masterful thriller.", "averageRating": 9.2, "ratingsCount": 370000},
        {"title": "Vinland Saga",              "authors": "Yukimura Makoto",       "description": "A Viking warrior's journey from revenge to peace in medieval Scandinavia.", "averageRating": 9.0, "ratingsCount": 390000},
        {"title": "Gantz",                      "authors": "Hiroya Oku",            "description": "Dead people are resurrected to fight aliens in a brutal, high-stakes battle game.", "averageRating": 8.0, "ratingsCount": 180000},
    ],
    "shonen": [
        {"title": "One Piece",                  "authors": "Oda Eiichiro",          "description": "A boy sets out to become King of the Pirates and find the legendary One Piece treasure.", "averageRating": 9.2, "ratingsCount": 700000},
        {"title": "Naruto",                     "authors": "Kishimoto Masashi",     "description": "A young ninja dreams of becoming the strongest leader of his village.", "averageRating": 8.7, "ratingsCount": 620000},
        {"title": "My Hero Academia",           "authors": "Horikoshi Kohei",       "description": "A boy born without powers in a superhero world strives to become the greatest hero.", "averageRating": 8.5, "ratingsCount": 500000},
        {"title": "Dragon Ball",                "authors": "Toriyama Akira",        "description": "Goku's lifelong journey of martial arts training and defending the Earth.", "averageRating": 8.7, "ratingsCount": 550000},
        {"title": "Haikyuu!!",                  "authors": "Furudate Haruichi",    "description": "A passionate short boy joins his high school volleyball team and chases championship glory.", "averageRating": 8.9, "ratingsCount": 460000},
    ],
    "shojo": [
        {"title": "Fruits Basket",              "authors": "Takaya Natsuki",        "description": "An orphaned girl discovers a family cursed to transform into zodiac animals.", "averageRating": 8.8, "ratingsCount": 310000},
        {"title": "Ouran High School Host Club","authors": "Hatori Bisco",         "description": "A scholarship student accidentally joins a host club at an elite school — charming and funny.", "averageRating": 8.6, "ratingsCount": 270000},
        {"title": "Nana",                       "authors": "Yazawa Ai",             "description": "Two girls named Nana meet on a train and their lives intertwine in love and music.", "averageRating": 8.9, "ratingsCount": 250000},
        {"title": "Clannad",                    "authors": "VisualArt's / Key",    "description": "A slice-of-life drama about a delinquent who finds friendship and love in his final year.", "averageRating": 8.9, "ratingsCount": 200000},
        {"title": "Cardcaptor Sakura",          "authors": "CLAMP",                 "description": "A girl accidentally releases magical cards and must recapture them as a Card Captor.", "averageRating": 8.5, "ratingsCount": 240000},
    ],
    "isekai": [
        {"title": "That Time I Got Reincarnated as a Slime", "authors": "Fuse", "description": "A man reincarnated as a slime builds a monster kingdom through kindness and cunning.", "averageRating": 8.2, "ratingsCount": 310000},
        {"title": "Re:Zero",                    "authors": "Nagatsuki Tappei",      "description": "A boy trapped in a fantasy world with the ability to return from death — intense and emotional.", "averageRating": 8.6, "ratingsCount": 350000},
        {"title": "Konosuba",                   "authors": "Akatsuki Natsume",     "description": "A NEET is reincarnated with a useless goddess and forms a hilariously incompetent party.", "averageRating": 8.5, "ratingsCount": 290000},
        {"title": "Mushoku Tensei",             "authors": "Rifujin na Magonote", "description": "A man reincarnated into a fantasy world with all his memories — epic world-building and growth.", "averageRating": 8.6, "ratingsCount": 280000},
        {"title": "The Rising of the Shield Hero","authors":"Aneko Yusagi",       "description": "A betrayed hero rises from the bottom and finds true companions in a dark isekai.", "averageRating": 8.0, "ratingsCount": 250000},
    ],
    "comedy manga": [
        {"title": "Gintama",                    "authors": "Sorachi Hideaki",       "description": "Aliens have invaded Edo-era Japan — a lazy samurai takes odd jobs in this absurd comedy.", "averageRating": 9.0, "ratingsCount": 380000},
        {"title": "Kaguya-sama: Love Is War",   "authors": "Akasaka Aka",          "description": "A battle of wits between two proud students who refuse to be the first to confess.", "averageRating": 8.7, "ratingsCount": 420000},
        {"title": "The Disastrous Life of Saiki K.", "authors": "Asou Shuuichi", "description": "An overpowered psychic desperately tries to live a normal, quiet life — endlessly funny.", "averageRating": 8.8, "ratingsCount": 290000},
        {"title": "Grand Blue Dreaming",        "authors": "Inoue Kenji",          "description": "A college student gets dragged into a diving club full of drinking and chaos.", "averageRating": 8.8, "ratingsCount": 230000},
        {"title": "Daily Lives of High School Boys", "authors": "Yamazaki Yama", "description": "The absurd, mundane, and hilarious everyday lives of high school boys.", "averageRating": 8.7, "ratingsCount": 180000},
    ],
    "horror manga": [
        {"title": "Uzumaki",                    "authors": "Ito Junji",             "description": "A town is cursed by spirals — one of the most iconic horror manga ever created.", "averageRating": 8.8, "ratingsCount": 220000},
        {"title": "Tomie",                      "authors": "Ito Junji",             "description": "A beautiful immortal girl drives men to obsession and murder — deeply unsettling.", "averageRating": 8.3, "ratingsCount": 160000},
        {"title": "Parasyte",                   "authors": "Iwaaki Hitoshi",       "description": "Alien parasites invade humans, but one bonds with a boy instead — philosophical horror.", "averageRating": 8.8, "ratingsCount": 300000},
        {"title": "Biomega",                    "authors": "Nihei Tsutomu",        "description": "A lone agent navigates a dying world filled with zombie-like infected — dark sci-fi horror.", "averageRating": 7.8, "ratingsCount": 60000},
        {"title": "Tokyo Ghoul",                "authors": "Ishida Sui",            "description": "A college student becomes half-ghoul and must navigate both human and ghoul worlds.", "averageRating": 8.2, "ratingsCount": 370000},
    ],
    "fantasy manga": [
        {"title": "Fullmetal Alchemist",        "authors": "Arakawa Hiromu",        "description": "Two brothers use alchemy to search for the Philosopher's Stone in a war-torn world.", "averageRating": 9.1, "ratingsCount": 560000},
        {"title": "Made in Abyss",              "authors": "Tsukushi Akihito",     "description": "Two children descend into a vast, mysterious abyss filled with wonder and horror.", "averageRating": 8.7, "ratingsCount": 280000},
        {"title": "Goblin Slayer",              "authors": "Kagyu Kumo",            "description": "A warrior dedicates his life to eradicating goblins in a dark fantasy world.", "averageRating": 7.9, "ratingsCount": 200000},
        {"title": "The Ancient Magus' Bride",  "authors": "Yamazaki Kore",        "description": "A girl sold at an auction becomes the apprentice of a mysterious mage — magical and moving.", "averageRating": 8.4, "ratingsCount": 210000},
        {"title": "Witch Hat Atelier",         "authors": "Shirahama Kamome",     "description": "A girl who can't do magic discovers the truth about witchcraft and joins a secret atelier.", "averageRating": 8.8, "ratingsCount": 150000},
    ],
    "josei": [
        {"title": "Nana",                       "authors": "Yazawa Ai",             "description": "Two girls named Nana meet on a train and their lives intertwine in love and music.", "averageRating": 8.9, "ratingsCount": 250000},
        {"title": "Paradise Kiss",              "authors": "Yazawa Ai",             "description": "A studious girl is swept into the glamorous world of fashion designers.", "averageRating": 8.2, "ratingsCount": 150000},
        {"title": "Butterflies, Flowers",      "authors": "Yoshihara Yuki",       "description": "A young woman hires her former servant as an employee — romance with class-crossing tension.", "averageRating": 7.8, "ratingsCount": 80000},
        {"title": "Chihayafuru",               "authors": "Suetsugu Yuki",        "description": "A girl pursues competitive karuta poetry cards with passion and hidden romance.", "averageRating": 8.9, "ratingsCount": 190000},
        {"title": "Princess Jellyfish",        "authors": "Higashimura Akiko",    "description": "A jellyfish-obsessed girl and a cross-dressing boy form an unlikely but heartwarming bond.", "averageRating": 8.4, "ratingsCount": 140000},
    ],
}

# Default fallback if subgenre not in curated
_MANGA_CURATED["does not matter"] = _MANGA_CURATED["seinen"]


def _build_jikan_params(sg: str, tone: str) -> dict:
    """Build Jikan API parameters for precise genre filtering."""
    params = {
        "limit":      20,
        "order_by":   "score",
        "sort":       "desc",
        "min_score":  "7.0",
        "sfw":        "true",
    }

    # Map subgenre to genre ID
    genre_id = _JIKAN_MANGA_GENRE_IDS.get(sg)
    if genre_id:
        params["genres"] = genre_id

    # Map demographic subgenres
    demographic_map = {
        "shonen": "shounen", "shojo": "shoujo",
        "seinen": "seinen",  "josei": "josei",
    }
    if sg in demographic_map:
        params["demographics"] = demographic_map[sg]

    # Tone-based explicit genre additions
    tone_genre_map = {
        "dark":      14,   # Horror genre for dark tone
        "intense":   1,    # Action for intense
        "romantic":  22,   # Romance for romantic
    }
    if tone in tone_genre_map and not genre_id:
        params["genres"] = tone_genre_map[tone]

    return params


def _jikan_search_precise(sg: str, tone: str, limit: int = 20) -> list:
    """Search Jikan with precise genre filtering."""
    params = _build_jikan_params(sg, tone)
    print(f"[Jikan] Query params: {params}")
    try:
        time.sleep(0.5)  # Respect Jikan rate limit
        r = requests.get(
            "https://api.jikan.moe/v4/manga",
            params=params,
            timeout=20,
            headers={"User-Agent": "EmoBot/13.0"},
        )
        if r.status_code != 200:
            print(f"[Jikan] HTTP {r.status_code}")
            return []
        out = []
        for item in r.json().get("data", []):
            title    = item.get("title", "")
            synopsis = (item.get("synopsis") or "")[:300]
            authors  = ", ".join(a["name"] for a in (item.get("authors") or []))
            cover    = (item.get("images") or {}).get("jpg", {}).get("image_url", "")
            score    = item.get("score") or 0
            members  = item.get("members") or 0
            out.append({
                "title":         title,
                "authors":       authors or "Various",
                "description":   synopsis,
                "category":      "Manga",
                "averageRating": score,
                "ratingsCount":  members,
                "thumbnail":     cover,
                "link":          item.get("url") or f"https://myanimelist.net/search/all?q={quote_plus(title)}",
                "alt_link":      f"https://anilist.co/search/manga?search={quote_plus(title)}",
                "goodreads":     f"https://www.goodreads.com/search?q={quote_plus(title)}+manga",
                "_score":        (score or 7.0) * 1000 + members / 1000,
            })
        print(f"[Jikan] Got {len(out)} results for sg={sg}")
        return out
    except Exception as e:
        print(f"[Jikan] Error: {e}")
        return []


def _mangadex_search_precise(sg: str, n: int = 10) -> list:
    """Search MangaDex with tag filtering for accurate genre results."""
    tag_id = _MANGADEX_TAGS.get(sg)
    params = {
        "limit":               n * 2,
        "order[followedCount]": "desc",
        "includes[]":          "cover_art",
        "contentRating[]":     ["safe", "suggestive"],
        "hasAvailableChapters": "true",
    }
    if tag_id:
        params["includedTags[]"] = tag_id
        params["includedTagsMode"] = "AND"

    # Demographic filter
    demo_map = {"shonen": "shounen", "shojo": "shoujo", "seinen": "seinen", "josei": "josei"}
    if sg in demo_map:
        params["publicationDemographic[]"] = demo_map[sg]

    print(f"[MangaDex] Query params: {params}")
    try:
        r = requests.get(
            "https://api.mangadex.org/manga",
            params=params,
            timeout=20,
            headers={"User-Agent": "EmoBot/13.0"},
        )
        if r.status_code != 200:
            print(f"[MangaDex] HTTP {r.status_code}")
            return []
        out = []
        for item in r.json().get("data", []):
            attrs  = item.get("attributes", {})
            titles = attrs.get("title", {})
            title  = titles.get("en") or (list(titles.values())[0] if titles else "Unknown")
            desc   = (attrs.get("description") or {}).get("en", "") or ""
            follows = attrs.get("followsCount", 0) or 0
            cover_file = next(
                (rel.get("attributes", {}).get("fileName")
                 for rel in item.get("relationships", []) if rel.get("type") == "cover_art"),
                None,
            )
            cover = f"https://uploads.mangadex.org/covers/{item['id']}/{cover_file}" if cover_file else ""
            out.append({
                "title":         title,
                "authors":       "Various",
                "description":   desc[:300],
                "category":      "Manga",
                "averageRating": None,
                "ratingsCount":  follows,
                "thumbnail":     cover,
                "link":          f"https://mangadex.org/title/{item['id']}",
                "alt_link":      f"https://myanimelist.net/search?q={quote_plus(title)}",
                "goodreads":     f"https://www.goodreads.com/search?q={quote_plus(title)}+manga",
                "_score":        7000 + follows / 100,
            })
        print(f"[MangaDex] Got {len(out)} results for sg={sg}")
        return out
    except Exception as e:
        print(f"[MangaDex] Error: {e}")
        return []


def _curated_manga(sg: str, tone: str, n: int = 5) -> list:
    """Return curated fallback manga for the given subgenre."""
    pool = _MANGA_CURATED.get(sg, [])
    if not pool:
        # Try to find a close match
        for key in _MANGA_CURATED:
            if key in sg or sg in key:
                pool = _MANGA_CURATED[key]
                break
    if not pool:
        pool = _MANGA_CURATED.get("seinen", [])

    # Tone-based sorting: 'dark'/'intense' → higher-rated darker picks first
    dark_tones = {"dark", "intense"}
    light_tones = {"fun", "uplifting", "calming"}

    results = []
    for fb in pool[:n]:
        title = fb["title"]
        results.append({
            "title":         title,
            "authors":       fb.get("authors", "Various"),
            "description":   fb.get("description", ""),
            "category":      "Manga",
            "averageRating": fb.get("averageRating"),
            "ratingsCount":  fb.get("ratingsCount"),
            "thumbnail":     fb.get("thumbnail", ""),
            "link":          f"https://myanimelist.net/search/all?q={quote_plus(title)}",
            "alt_link":      f"https://anilist.co/search/manga?search={quote_plus(title)}",
            "goodreads":     f"https://www.goodreads.com/search?q={quote_plus(title)}+manga",
        })
    return results


def search_manga(prefs: dict, n: int = 5) -> list:
    """
    Multi-source manga search with accurate genre filtering.
    Priority: MangaDex (tag-filtered) → Jikan (genre ID) → Curated fallback
    """
    sg   = prefs.get("subgenre", "does not matter")
    tone = prefs.get("tone", "does not matter")

    print(f"[search_manga] subgenre={sg!r}  tone={tone!r}")

    results = []
    seen    = set()

    # 1. MangaDex with tag filtering
    md_results = _mangadex_search_precise(sg, n * 2)
    for item in md_results:
        k = _n(item["title"])
        if k not in seen:
            seen.add(k)
            results.append(item)

    # 2. Jikan with genre ID filtering
    if len(results) < n:
        jk_results = _jikan_search_precise(sg, tone, limit=20)
        for item in jk_results:
            k = _n(item["title"])
            if k not in seen:
                seen.add(k)
                results.append(item)

    # 3. Curated fallback if still not enough
    if len(results) < n:
        print(f"[search_manga] Using curated fallback for sg={sg!r}")
        curated = _curated_manga(sg, tone, n * 2)
        for item in curated:
            k = _n(item["title"])
            if k not in seen:
                seen.add(k)
                results.append(item)

    # Sort by score, deduplicate
    results.sort(key=lambda x: x.get("_score", 0) or (x.get("averageRating") or 0) * 1000, reverse=True)

    # Strip internal fields
    clean = []
    for r in results[:n]:
        clean.append({k: v for k, v in r.items() if not k.startswith("_")})
    return clean


# ══════════════════════════════════════════════════════════════════
#  COMICS SEARCH
# ══════════════════════════════════════════════════════════════════

_COMICS_FALLBACK = [
    {"title": "Watchmen",                "authors": "Alan Moore",
     "description": "A dark deconstruction of superhero mythology that changed comics forever.",
     "averageRating": 9.0, "ratingsCount": 280000},
    {"title": "Saga",                    "authors": "Brian K. Vaughan",
     "description": "A sweeping sci-fantasy space opera with enormous emotional depth.",
     "averageRating": 8.9, "ratingsCount": 310000},
    {"title": "Maus",                    "authors": "Art Spiegelman",
     "description": "A Pulitzer Prize-winning graphic memoir about the Holocaust.",
     "averageRating": 9.0, "ratingsCount": 220000},
    {"title": "The Dark Knight Returns", "authors": "Frank Miller",
     "description": "An aged Batman returns to a crime-ridden dystopian Gotham.",
     "averageRating": 8.8, "ratingsCount": 160000},
    {"title": "Persepolis",              "authors": "Marjane Satrapi",
     "description": "A powerful autobiographical graphic novel set in revolutionary Iran.",
     "averageRating": 8.7, "ratingsCount": 230000},
    {"title": "All-Star Superman",       "authors": "Grant Morrison",
     "description": "The quintessential uplifting Superman story — hopeful and beautiful.",
     "averageRating": 8.8, "ratingsCount": 85000},
    {"title": "Y: The Last Man",         "authors": "Brian K. Vaughan",
     "description": "The only surviving human male navigates a post-plague world.",
     "averageRating": 8.6, "ratingsCount": 115000},
    {"title": "Ms. Marvel: No Normal",   "authors": "G. Willow Wilson",
     "description": "A fun, uplifting teenage superhero origin story with real heart.",
     "averageRating": 8.4, "ratingsCount": 92000},
]

_COMICS_QUERY_MAP = {
    "superhero":      "superhero comics bestseller DC Marvel",
    "sci fi comics":  "science fiction graphic novel",
    "fantasy comics": "fantasy graphic novel",
    "horror comics":  "horror graphic novel",
    "comedy comics":  "comedy graphic novel",
    "graphic novel":  "literary graphic novel award-winning",
    "indie comics":   "indie alternative comics",
    "memoir comics":  "memoir autobiography graphic novel",
}

_COMICS_TONE_KEYWORDS = {
    "uplifting": "hopeful inspiring heroic",
    "emotional": "heartfelt moving emotional",
    "intense":   "gritty dramatic powerful",
    "fun":       "lighthearted humorous playful",
    "calming":   "gentle reflective soothing",
    "dark":      "dark gritty noir",
    "inspiring": "epic motivational heroic",
}

_COMICS_CURATED = {
    "superhero": [
        {"title": "Ms. Marvel: No Normal", "authors": "G. Willow Wilson",
         "description": "A young superhero balances school, family and identity while saving Jersey City.", "averageRating": 8.4, "ratingsCount": 92000},
        {"title": "Spider-Man: Blue", "authors": "Jeph Loeb",
         "description": "A heartfelt Spider-Man story about love, loss and memory.", "averageRating": 8.1, "ratingsCount": 42000},
        {"title": "All-Star Superman", "authors": "Grant Morrison",
         "description": "A glorious, emotional Superman story about hope, legacy and what makes a hero.", "averageRating": 8.8, "ratingsCount": 85000},
        {"title": "The Dark Knight Returns", "authors": "Frank Miller",
         "description": "An aged Batman returns to a crime-ridden Gotham with brutal intensity and emotional stakes.", "averageRating": 8.8, "ratingsCount": 160000},
    ],
    "graphic novel": [
        {"title": "Persepolis", "authors": "Marjane Satrapi",
         "description": "A powerful autobiographical graphic novel set in revolutionary Iran.", "averageRating": 8.7, "ratingsCount": 230000},
        {"title": "Maus", "authors": "Art Spiegelman",
         "description": "A Pulitzer Prize-winning graphic memoir about the Holocaust.", "averageRating": 9.0, "ratingsCount": 220000},
        {"title": "Blankets", "authors": "Craig Thompson",
         "description": "A beautifully drawn coming-of-age story about first love and faith.", "averageRating": 8.4, "ratingsCount": 65000},
        {"title": "Sandman", "authors": "Neil Gaiman",
         "description": "A genre-defining graphic novel series blending myth, horror and fantasy.", "averageRating": 8.9, "ratingsCount": 180000},
    ],
    "indie comics": [
        {"title": "Nimona", "authors": "Noelle Stevenson",
         "description": "A wild, emotional indie fantasy about a shapeshifting sidekick and a villainous plot.", "averageRating": 8.3, "ratingsCount": 87000},
        {"title": "Paper Girls", "authors": "Brian K. Vaughan",
         "description": "A time-traveling coming-of-age adventure with heart and mystery.", "averageRating": 8.2, "ratingsCount": 60000},
        {"title": "Bone", "authors": "Jeff Smith",
         "description": "A sweeping indie epic with humor, adventure and deep emotional moments.", "averageRating": 8.6, "ratingsCount": 74000},
    ],
    "sci fi comics": [
        {"title": "Saga", "authors": "Brian K. Vaughan",
         "description": "A sweeping sci-fi space opera with enormous emotional depth.", "averageRating": 8.9, "ratingsCount": 310000},
        {"title": "Y: The Last Man", "authors": "Brian K. Vaughan",
         "description": "A post-apocalyptic sci-fi saga full of character drama and moral complexity.", "averageRating": 8.6, "ratingsCount": 115000},
        {"title": "Descender", "authors": "Jeff Lemire",
         "description": "A poetic sci-fi graphic novel about a boy robot surviving a hostile galaxy.", "averageRating": 8.1, "ratingsCount": 56000},
    ],
}

_COMIC_KW = [
    "comic", "graphic novel", "marvel", "dc comics", "image comics",
    "dark horse", "batman", "spider-man", "x-men", "avengers", "manga",
    "illustrated", "drawn", "panels", "sequential art",
]


def _curated_comics(sg: str, tone: str, n: int = 5) -> list:
    pool = []
    lookup = sg.lower().strip()
    for key, items in _COMICS_CURATED.items():
        if key == lookup or key in lookup or lookup in key:
            pool = items
            break
    if not pool:
        pool = _COMICS_FALLBACK

    # Tone boost: more emotional or intense selections first when requested
    if tone in {"emotional", "intense", "dark"}:
        return sorted(pool, key=lambda b: ((b.get("averageRating") or 0) * 100 + (b.get("ratingsCount") or 0) // 1000), reverse=True)[:n]
    return pool[:n]


def search_comics(prefs: dict, emotion: str, n: int = 5) -> list:
    sg      = prefs.get("subgenre", "").lower().strip()
    tone    = prefs.get("tone", "").lower().strip()
    query_base = _COMICS_QUERY_MAP.get(sg, "comics graphic novel bestseller")
    tone_kw = _COMICS_TONE_KEYWORDS.get(tone, "")
    emotion_kw = " ".join(EMOTION_KEYWORDS.get(emotion, [])) if emotion else ""
    query = " ".join(filter(None, [query_base, tone_kw, emotion_kw]))

    results = []

    try:
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": query, "maxResults": 40, "langRestrict": "en",
                    "printType": "books", "orderBy": "relevance"},
            timeout=12,
        )
        items = r.json().get("items") or []
        for item in items:
            info    = item.get("volumeInfo", {})
            title   = info.get("title", "")
            authors = ", ".join(info.get("authors", []))
            if not title or not authors:
                continue
            combined = (title + " " + (info.get("description") or "") + " " +
                        " ".join(info.get("categories") or [])).lower()
            if sg and sg not in combined and not any(k in combined for k in _COMIC_KW):
                continue
            cnt = info.get("ratingsCount", 0) or 0
            avg = info.get("averageRating", 0) or 0
            if cnt < 2 and avg < 4.0:
                continue
            desc  = info.get("description", "") or ""
            thumb = _get_thumbnail(info)
            score = avg * 100 + cnt // 100 + _preference_score(info, sg, "", tone, emotion)
            results.append({
                "title": title, "authors": authors,
                "description": (desc[:300] + "…") if len(desc) > 300 else desc,
                "category": "Comics", "averageRating": avg, "ratingsCount": cnt,
                "thumbnail": thumb,
                "link":      f"https://www.google.com/search?q={quote_plus(title)}",
                "alt_link":  f"https://openlibrary.org/search?q={quote_plus(title)}",
                "goodreads": f"https://www.goodreads.com/search?q={quote_plus(title)}",
                "_score":    score,
            })
    except Exception as e:
        print("Comics Google Books error:", e)

    if len(results) < n:
        curated = _curated_comics(sg, tone, n * 2)
        for fb in curated:
            results.append({
                **fb, "category": "Comics", "thumbnail": "",
                "link":      f"https://www.goodreads.com/search?q={quote_plus(fb['title'])}",
                "alt_link":  f"https://openlibrary.org/search?q={quote_plus(fb['title'])}",
                "goodreads": f"https://www.goodreads.com/search?q={quote_plus(fb['title'])}",
                "_score":    (fb.get("averageRating") or 0) * 100 + (fb.get("ratingsCount") or 0) // 1000,
            })

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    seen, unique = set(), []
    for b in results:
        k = _n(b["title"])
        if k not in seen:
            seen.add(k); unique.append(b)
    return [{k: v for k, v in b.items() if k != "_score"} for b in unique[:n]]


def search_media(prefs: dict, emotion: str, n: int = 5) -> list:
    mt = prefs.get("media_type", "books")
    print(f"[search_media] media_type={mt}  prefs={prefs}")
    if mt == "manga":
        return search_manga(prefs, n)
    if mt == "comics":
        return search_comics(prefs, emotion, n)
    return search_books(prefs, emotion, n)


# ══════════════════════════════════════════════════════════════════
#  DEFAULT PREFS + FALLBACK OPENERS
# ══════════════════════════════════════════════════════════════════

_DEFAULT_PREFS = {
    "media_type": "books", "genre": "does not matter", "subgenre": "does not matter",
    "tone": "does not matter", "theme": "does not matter",
    "pacing": "does not matter", "length": "does not matter",
}

_FALLBACK_OPENERS = {
    "happy":    "You're beaming today! 😊 What do you want to read?\n👉 **Books** · **Manga** · **Comics**",
    "sad":      "Feeling down? A great story can help 💙\n👉 **Books** · **Manga** · **Comics**",
    "angry":    "Let's channel that energy into an epic read 🔥\n👉 **Books** · **Manga** · **Comics**",
    "neutral":  "Hey! Ready to find your next read? 👋\n👉 **Books** · **Manga** · **Comics**",
    "fear":     "Let's find something totally absorbing 🤗\n👉 **Books** · **Manga** · **Comics**",
    "surprise": "That energy is perfect for something new! 😲\n👉 **Books** · **Manga** · **Comics**",
    "disgust":  "A great story can reset everything 📖\n👉 **Books** · **Manga** · **Comics**",
}

_DONE_PHRASES = [
    "fetching", "finding your", "getting your", "loading your",
    "on it", "perfect", "great choice", "just a moment",
]

_RESET_WORDS = {
    "start", "restart", "start over", "reset", "new", "", "over", "again",
    "retry", "redo", "go back", "back", "try again", "new search",
    "find more", "more books", "find manga", "find comics",
}


def _is_reset(msg: str) -> bool:
    m = msg.lower().strip()
    if m in _RESET_WORDS:
        return True
    return any(p in m for p in ["start over", "start again", "begin again"])


def _do_reset(emotion: str):
    session["history"]     = []
    session["turn_count"]  = 0
    session["search_done"] = False

    seed_history = [{"role": "user", "content": "Hi!"}]
    reply = call_llm(seed_history, emotion)
    if reply is None:
        reply = _FALLBACK_OPENERS.get(emotion, _FALLBACK_OPENERS["neutral"])

    clean, _ = extract_prefs(reply)
    session["history"] = [
        {"role": "user",      "content": "Hi!"},
        {"role": "assistant", "content": reply},
    ]
    return jsonify({"reply": clean, "done": False})


# ══════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", emotion=session.get("emotion", "neutral"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    img  = data.get("image")
    if not img:
        return jsonify({"error": "No image"}), 400
    emo = detect_emotion(img)
    session.clear()
    session["emotion"]     = emo
    session["history"]     = []
    session["turn_count"]  = 0
    session["search_done"] = False
    print("Detected emotion:", emo)
    return jsonify({"emotion": emo})


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    emotion = session.get("emotion", "neutral")
    history = list(session.get("history", []))
    turns   = session.get("turn_count", 0)
    msg_low = message.lower().strip()

    already_done = session.get("search_done", False)
    if _is_reset(msg_low) or already_done:
        return _do_reset(emotion)

    history.append({"role": "user", "content": message})
    turns += 1
    session["turn_count"] = turns

    if turns >= 6:
        nudge_history = history + [{
            "role": "user",
            "content": "[System: You have enough information. Output PREFS_JSON now.]",
        }]
        reply = call_llm(nudge_history, emotion)
    else:
        reply = call_llm(history, emotion)

    if reply is None:
        session["history"] = history
        return jsonify({
            "reply": "Hmm, let me try again 😅 What kind of story calls to you?",
            "done":  False,
        })

    clean, prefs = extract_prefs(reply)
    history.append({"role": "assistant", "content": reply})
    if len(history) > 24:
        history = history[-24:]
    session["history"] = history

    if prefs:
        session["turn_count"]  = 0
        session["search_done"] = True
        full_prefs = {**_DEFAULT_PREFS, **prefs}
        books = search_media(full_prefs, emotion, n=5)
        print(f"[chat] Returning {len(books)} results")
        return jsonify({"reply": clean, "done": True, "books": books})

    reply_low = reply.lower()
    if any(p in reply_low for p in _DONE_PHRASES) and turns >= 4:
        print("[chat] Model signalled done, no PREFS_JSON — forcing search")
        session["turn_count"]  = 0
        session["search_done"] = True
        books = search_media(_DEFAULT_PREFS, emotion, n=5)
        return jsonify({"reply": clean, "done": True, "books": books})

    if turns >= 8:
        session["turn_count"]  = 0
        session["search_done"] = True
        books = search_media(_DEFAULT_PREFS, emotion, n=5)
        return jsonify({
            "reply": clean or "Great, let me find your picks! 📚",
            "done":  True, "books": books,
        })

    return jsonify({"reply": clean, "done": False})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)