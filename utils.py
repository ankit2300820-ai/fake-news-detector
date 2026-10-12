import re


def clean_text(text):
    """Simple text cleaning used for both training and prediction."""
    text = str(text)
    # Remove agency tag like "WASHINGTON (Reuters) -" so the model learns
    # writing style, not just the word "Reuters".
    text = re.sub(r"^.{0,80}?\(reuters\)\s*-?\s*", "", text, flags=re.I)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


# ---------- Hindi support ----------
def is_hindi(text):
    """True if most letters are Devanagari (Hindi)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters) > 0.3


def translate_to_english(text):
    """Hindi -> English using Google Translate (needs internet + deep-translator)."""
    from deep_translator import GoogleTranslator
    chunks = [text[i:i + 4500] for i in range(0, min(len(text), 13500), 4500)]
    return " ".join(GoogleTranslator(source="hi", target="en").translate(c) for c in chunks)


# ---------- URL support ----------
def extract_text_from_url(url):
    """Download a web page and return its title + paragraph text."""
    import requests
    from bs4 import BeautifulSoup
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    return extract_text_from_html(resp.text)


def extract_text_from_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return (title + ". " + " ".join(p for p in paras if len(p.split()) >= 6)).strip()


# ---------- Unique features: explain, red flags, trust score ----------
CLICKBAIT = ["you won't believe", "share this", "share before", "before they delete", "shocking", "breaking:",
             "they don't want you", "doctors hate", "miracle", "secret", "exposed", "wake up"]
SOURCES = ["said", "according to", "told", "reported", "statement", "spokesman", "official"]


def explain(vectorizer, model, text, top=5):
    """Words that pushed the Logistic Regression towards FAKE or REAL."""
    X = vectorizer.transform([clean_text(text)])
    words = vectorizer.get_feature_names_out()
    scores = [(words[i], X[0, i] * model.coef_[0][i]) for i in X.nonzero()[1]]
    fake = sorted([s for s in scores if s[1] < 0], key=lambda s: s[1])[:top]
    real = sorted([s for s in scores if s[1] > 0], key=lambda s: -s[1])[:top]
    return [w for w, _ in fake], [w for w, _ in real]


def red_flags(text):
    flags, lower, words = [], text.lower(), text.split()
    caps = [w for w in (x.strip(".,:;!?\"'") for x in words) if len(w) >= 3 and w.isupper() and w.isalpha()]
    if words and len(caps) / len(words) > 0.15:
        flags.append("Too many ALL-CAPS words")
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
    found = [p for p in CLICKBAIT if p in lower]
    if found:
        flags.append("Clickbait phrases: " + ", ".join(found[:3]))
    if len(words) >= 30 and not any(s in lower for s in SOURCES):
        flags.append("No quoted source or attribution")
    return flags


def trust_score(real_probability, n_flags):
    """0-100 score: model's REAL probability minus 7 points per red flag."""
    score = max(0, min(100, round(real_probability - 7 * n_flags)))
    level = "High trust" if score >= 70 else "Be careful" if score >= 40 else "Low trust"
    return score, level
