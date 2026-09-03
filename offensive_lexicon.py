import string

# Curated from HurtLex + HateXplain offensive tokens - keep small for precision
LEXICON = {
    "stupid", "foolish", "idiot", "idiots", "moron", "morons", "dumb", "dumbass",
    "asshole", "assholes", "bitch", "bitches", "bastard", "jerk", "loser",
    "fuck", "fucking", "fucked", "shit", "shitty", "trash", "garbage",
    "clown", "whore", "slut", "douche", "douchebag", "sucks", "suck",
    "crap", "freak", "weirdo", "lame", "annoying", "hate", "kill", "die"
}
# normalize
LEXICON = {w.strip().lower() for w in LEXICON if w.strip()}

ALPHA = 0.20  # boost per hit, tune 0.05-0.25 on val 2015 - 0.20 flips 0.644/0.34 with 2 hits (stupid+idiot)
ALPHA_CAP = 0.95

def lexicon_boost(probs: dict, text: str, alpha: float = ALPHA) -> dict:
    """Soft boost for offensive if lexicon hits; renormalize."""
    translator = str.maketrans("", "", string.punctuation)
    tokens = text.lower().translate(translator).split()
    hits = sum(1 for t in tokens if t in LEXICON)
    if hits == 0:
        return probs
    # copy to avoid mutating upstream
    boosted = dict(probs)
    boosted["offensive"] = min(ALPHA_CAP, boosted.get("offensive", 0) + alpha * hits)
    s = sum(boosted.values())
    if s > 0:
        boosted = {k: v / s for k, v in boosted.items()}
    return boosted

def lexicon_hits(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    tokens = text.lower().translate(translator).split()
    return [t for t in tokens if t in LEXICON]
