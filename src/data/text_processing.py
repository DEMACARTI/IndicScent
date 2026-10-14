"""Text normalisation and script-based language identification for Indian-language memes.

Normalisation is deliberately conservative: sentiment cues such as emojis, "!!!",
negations ("not", "nahi", "না"), casing (XLM-R is cased) and Indic scripts are kept.
Zero-width joiner / non-joiner (U+200D / U+200C) are preserved because they change how
conjuncts render in Bengali and Devanagari.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Unicode blocks for scripts used in Indian languages (+ Latin for English/romanised text).
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Nepali, Sanskrit ...
    "Bengali": (0x0980, 0x09FF),      # Bengali, Assamese
    "Gurmukhi": (0x0A00, 0x0A7F),     # Punjabi
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),       # Urdu, Kashmiri
}
# Scripts that XLM-R saw substantial pre-training data for (CC-100), incl. romanised Indic.
SUPPORTED_SCRIPTS = set(SCRIPT_RANGES) | {"Latin"}

SCRIPT_TO_LANG_HINT = {
    "Devanagari": "Hindi/Marathi/Nepali (Devanagari)",
    "Bengali": "Bengali/Assamese",
    "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati",
    "Oriya": "Odia",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Arabic": "Urdu/Arabic script",
    "Latin": "English or romanised Indic (cannot be separated by script alone)",
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_WS_RE = re.compile(r"\s+")
# Collapse 4+ repeats of the same character to 3 ("sooooo" -> "sooo", "!!!!!" -> "!!!"):
# keeps the emphasis signal while limiting tokenizer fragmentation.
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_KEEP_FORMAT_CHARS = {"‌", "‍"}  # ZWNJ, ZWJ


def normalize_text(text: object) -> str:
    """Return a cleaned string; ``None``/NaN become ``""``."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    s = unicodedata.normalize("NFC", str(text))
    # Drop control/format characters (BOM, bidi marks, stray zero-width spaces) except ZWJ/ZWNJ.
    s = "".join(
        ch for ch in s
        if ch in _KEEP_FORMAT_CHARS or ch in "\n\t " or unicodedata.category(ch) not in {"Cc", "Cf"}
    )
    s = _URL_RE.sub("URL", s)
    s = _MENTION_RE.sub("@USER", s)
    s = _REPEAT_RE.sub(lambda m: m.group(1) * 3, s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def char_script(ch: str) -> str | None:
    cp = ord(ch)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    if ch.isalpha():
        try:
            if "LATIN" in unicodedata.name(ch):
                return "Latin"
        except ValueError:
            return "Other"
        return "Other"
    return None  # digits, punctuation, emoji, whitespace


def detect_scripts(text: str) -> dict:
    """Script-level language identification.

    Returns the dominant script, all scripts with share >= 15 %, a code-mixing flag and a
    language hint. Script detection cannot tell English from romanised Hindi/Bengali; the
    hint says so instead of guessing.
    """
    counts = Counter(s for s in map(char_script, text) if s)
    total = sum(counts.values())
    if total == 0:
        return {"dominant": "None", "scripts": [], "code_mixed": False,
                "language_hint": "no alphabetic text", "supported": True}
    shares = {k: v / total for k, v in counts.items()}
    major = sorted([k for k, v in shares.items() if v >= 0.15], key=lambda k: -shares[k])
    dominant = max(shares, key=shares.get)
    unsupported = [k for k in major if k not in SUPPORTED_SCRIPTS]
    return {
        "dominant": dominant,
        "scripts": major,
        "code_mixed": len(major) > 1,
        "language_hint": SCRIPT_TO_LANG_HINT.get(dominant, "unknown script"),
        "supported": not unsupported,
    }
