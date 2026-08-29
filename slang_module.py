import requests
import string
import math
from wordfreq import word_frequency

FREQUENCY_THRESHOLD = 3.0

def detect_oov_tokens(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    tokens = text.lower().translate(translator).split()
    oov = []
    for token in tokens:
        if not token.isalpha() or len(token) <= 2:
            continue
        freq = word_frequency(token, "en", minimum=0.0)
        zipf = math.log10(freq * 1e9) if freq > 0 else 0.0
        if zipf < FREQUENCY_THRESHOLD:
            oov.append(token)
    return oov

def lookup_urban_dictionary(term: str) -> str | None:
    try:
        resp = requests.get(
            "https://api.urbandictionary.com/v0/define",
            params={"term": term},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("list"):
            return data["list"][0]["definition"].replace("[", "").replace("]", "").strip()
    except requests.RequestException:
        pass
    return None

def get_slang_context(text: str) -> dict:
    oov_tokens = detect_oov_tokens(text)
    definitions = {term: lookup_urban_dictionary(term) for term in oov_tokens}
    return {
        "oov_tokens": oov_tokens,
        "definitions": definitions,
        "has_unresolved_terms": any(v is None for v in definitions.values()),
    }