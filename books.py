import re
import time
import requests
from urllib.parse import quote_plus


def _norm(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _google_search_link(title: str, authors: str) -> str:
    q = f"{title} {authors}".strip()
    return "https://www.google.com/search?q=" + quote_plus(q)


def _openlibrary_search_link(title: str, authors: str) -> str:
    q = f"{title} {authors}".strip()
    return "https://openlibrary.org/search?q=" + quote_plus(q)


def _goodreads_search_link(title: str, authors: str) -> str:
    q = f"{title} {authors}".strip()
    return "https://www.goodreads.com/search?q=" + quote_plus(q)


# ── Block lists ──────────────────────────────────────────────────────────────
BLOCK_TITLE_PHRASES = [
    "writing", "writer", "how to", "guide", "handbook", "manual", "workbook",
    "companion", "study", "textbook", "research", "criticism", "analysis",
    "journal", "proceedings", "conference", "report", "directory", "reference",
    "literary criticism", "craft of", "paper", "review paper", "systematic review",
    "literature review", "survey", "white paper", "thesis", "dissertation",
    "magazine", "periodical", "issue", "volume", "vol", "no ", "edition",
    "proceeding", "transactions", "newsletter", "bulletin",
]

BLOCK_CATEGORIES = [
    "language arts", "disciplines", "education", "business", "economics",
    "philosophy", "reference", "study aids", "library", "statistics", "law",
    "medical", "social science", "science", "technology", "mathematics",
    "computers", "engineering", "academic", "scholarly", "journals", "periodicals",
]

COMICS_CATEGORIES = ["comics", "graphic novels", "manga"]

BLOCK_DESC_PHRASES = [
    "peer reviewed", "peer reviewed journal", "special issue", "call for papers",
    "journal article", "research article", "this paper", "in this paper",
    "conference paper", "proceedings of", "volume", "issue", "issn", "doi", "citation",
]


def _blocked_strict(title: str, categories, description: str) -> bool:
    nt = _norm(title)
    nc = _norm(" ".join(categories or []))
    nd = _norm(description or "")
    for p in BLOCK_TITLE_PHRASES:
        if _norm(p) in nt:
            return True
    for c in BLOCK_CATEGORIES:
        if _norm(c) in nc:
            return True
    for c in COMICS_CATEGORIES:
        if _norm(c) in nc:
            return True
    for p in BLOCK_DESC_PHRASES:
        if _norm(p) in nd:
            return True
    return False


def _blocked_loose(title: str, description: str) -> bool:
    nt = _norm(title)
    nd = _norm(description or "")
    loose = [
        "writing", "how to", "guide", "handbook", "manual", "workbook", "companion",
        "criticism", "journal", "paper", "review", "magazine", "periodical",
        "proceedings", "conference", "report", "textbook", "study guide",
        "question bank", "exam", "syllabus", "notes", "doi", "issn",
    ]
    for p in loose:
        if _norm(p) in nt or _norm(p) in nd:
            return True
    return False


def _fiction_like(categories, title, description) -> bool:
    nc = _norm(" ".join(categories or []))
    nt = _norm(title)
    nd = _norm(description)
    if "fiction" in nc:
        return True
    hints = [
        "novel", "story", "saga", "series", "romance", "thriller", "mystery",
        "fantasy", "adventure", "dystopian", "coming of age",
    ]
    return any(w in nt or w in nd for w in hints)


def _tone_terms(emotion: str, tone: str) -> str:
    emo = (emotion or "neutral").lower().strip()
    t = (tone or "").lower().strip()
    emo_map = {
        "sad":     "emotional moving heartfelt",
        "angry":   "dark intense gripping",
        "happy":   "fun uplifting feel good",
        "neutral": "popular",
    }
    tone_map = {
        "uplifting":  "uplifting hopeful inspiring",
        "emotional":  "emotional heartfelt moving",
        "intense":    "intense gripping fast paced",
        "fun":        "fun funny lighthearted",
        "calming":    "cozy comforting",
        "dark":       "dark grim",
        "inspiring":  "inspiring motivational",
    }
    return (tone_map.get(t, t) + " " + emo_map.get(emo, "popular")).strip()


def _build_query(emotion, tone, genre, subgenre, theme, length, pacing, setting, variant=0) -> str:
    base = _tone_terms(emotion, tone)
    g = "books"
    if genre == "fiction":
        g = "fiction novel"
    elif genre == "non fiction":
        g = "nonfiction book"
    elif genre in ("any", None):
        g = "novel"

    sub = (subgenre or "").strip()
    th  = (theme or "").strip()
    ln  = "epic" if length == "long" else "short" if length == "short" else ""
    pc  = "fast paced" if pacing == "fast" else "slow burn" if pacing == "slow" else ""
    st  = (setting or "").strip()

    neg = " -magazine -journal -proceedings -conference -paper -review -textbook -workbook -manual -handbook -study"

    if variant == 0:
        return " ".join(f"{base} {g} {sub} {th} {ln} {pc} {st} bestselling popular{neg}".split())
    if variant == 1:
        return " ".join(f"{g} {sub} {th} bestselling popular{neg}".split())
    if variant == 2:
        return " ".join(f"bestselling {g} {sub}{neg}".split())
    return "bestselling novels" + neg


def _fetch_items(query: str, start_index: int, max_results: int = 40):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "startIndex": start_index,
        "printType": "books",
        "orderBy": "relevance",
        "langRestrict": "en",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        return r.json().get("items") or []
    except Exception:
        return []


def _score_popularity(info: dict) -> int:
    avg = info.get("averageRating") or 0
    cnt = info.get("ratingsCount") or 0
    s = 5 + int(avg * 6)
    if cnt >= 200000: s += 40
    elif cnt >= 50000: s += 30
    elif cnt >= 20000: s += 24
    elif cnt >= 5000:  s += 18
    elif cnt >= 1000:  s += 12
    elif cnt >= 200:   s += 7
    elif cnt >= 50:    s += 3
    return s


def _dedupe_keep_best(cands):
    best = {}
    for c in cands:
        key = _norm(c.get("title", ""))
        if not key:
            continue
        if key not in best or c["_score"] > best[key]["_score"]:
            best[key] = c
    out = list(best.values())
    out.sort(key=lambda x: x["_score"], reverse=True)
    return out


def _get_thumbnail(info: dict) -> str:
    """Extract best available thumbnail URL from volumeInfo."""
    links = info.get("imageLinks") or {}
    # Prefer higher res, fall back gracefully
    for key in ("extraLarge", "large", "medium", "thumbnail", "smallThumbnail"):
        url = links.get(key)
        if url:
            # Force HTTPS
            return url.replace("http://", "https://")
    return ""


def _make_book(info: dict, score: int):
    title       = info.get("title", "") or ""
    authors_list = info.get("authors", []) or []
    authors      = ", ".join(authors_list) if authors_list else ""
    desc         = info.get("description", "") or ""
    cats         = info.get("categories", []) or []
    category     = cats[0] if cats else ""
    thumbnail    = _get_thumbnail(info)

    return {
        "title":         title,
        "authors":       authors,
        "description":   (desc[:260] + ("…" if len(desc) > 260 else "")) if desc else "",
        "category":      category,
        "averageRating": info.get("averageRating"),
        "ratingsCount":  info.get("ratingsCount"),
        "thumbnail":     thumbnail,
        "link":          _google_search_link(title, authors),
        "alt_link":      _openlibrary_search_link(title, authors),
        "goodreads":     _goodreads_search_link(title, authors),
        "_score":        score,
    }


def _openlibrary_fallback(query: str, limit: int = 5):
    url = "https://openlibrary.org/search.json"
    try:
        r = requests.get(url, params={"q": query, "limit": 40}, timeout=15)
        if r.status_code != 200:
            return []
        docs = r.json().get("docs") or []
        out = []
        for d in docs:
            title = d.get("title") or ""
            if not title:
                continue
            authors = ", ".join(d.get("author_name") or [])[:80]
            if _blocked_loose(title, ""):
                continue

            # Build Open Library cover URL from cover_i
            cover_id = d.get("cover_i")
            thumbnail = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""

            out.append({
                "title":         title,
                "authors":       authors,
                "description":   "",
                "category":      "",
                "averageRating": None,
                "ratingsCount":  None,
                "thumbnail":     thumbnail,
                "link":          _google_search_link(title, authors),
                "alt_link":      _openlibrary_search_link(title, authors),
                "goodreads":     _goodreads_search_link(title, authors),
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def get_books_smart(emotion, tone, genre, subgenre, theme, length, pacing, setting):
    day_seed = int(time.time() // (24 * 3600))
    seed = abs(hash(f"{emotion}|{tone}|{genre}|{subgenre}|{theme}|{length}|{pacing}|{setting}|{day_seed}"))
    start_indices = [(seed % 6) * 20, ((seed + 1) % 6) * 20, ((seed + 2) % 6) * 20]

    want_fiction = (genre == "fiction")
    candidates = []

    for variant in [0, 1, 2, 3]:
        query = _build_query(emotion, tone, genre, subgenre, theme, length, pacing, setting, variant=variant)

        for start in start_indices:
            items = _fetch_items(query, start_index=start, max_results=40)
            for item in items:
                info  = item.get("volumeInfo", {}) or {}
                title = info.get("title", "") or ""
                if not title:
                    continue
                cats = info.get("categories", []) or []
                desc = info.get("description", "") or ""
                if _blocked_strict(title, cats, desc):
                    continue
                if want_fiction and not _fiction_like(cats, title, desc):
                    continue
                score = _score_popularity(info)
                candidates.append(_make_book(info, score))

        candidates = _dedupe_keep_best(candidates)
        if len(candidates) >= 5:
            break

    if len(candidates) < 5:
        loose_candidates = []
        query = _build_query(emotion, tone, genre, subgenre, theme, length, pacing, setting, variant=2)
        for start in start_indices:
            items = _fetch_items(query, start_index=start, max_results=40)
            for item in items:
                info  = item.get("volumeInfo", {}) or {}
                title = info.get("title", "") or ""
                if not title:
                    continue
                cats = info.get("categories", []) or []
                desc = info.get("description", "") or ""
                if _blocked_loose(title, desc):
                    continue
                if want_fiction and not _fiction_like(cats, title, desc):
                    continue
                score = _score_popularity(info)
                loose_candidates.append(_make_book(info, score))

        loose_candidates = _dedupe_keep_best(loose_candidates)
        candidates = _dedupe_keep_best(candidates + loose_candidates)

    chosen = candidates[:5]
    for b in chosen:
        b.pop("_score", None)

    if chosen:
        return chosen

    q = _build_query(emotion, tone, genre, subgenre, theme, length, pacing, setting, variant=1)
    fb = _openlibrary_fallback(q, limit=5)
    return fb if fb else []