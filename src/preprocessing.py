"""Text preprocessing utilities for technical repair transcripts."""
import re

TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

def segment_sentences(text: str) -> list[str]:
    """Split a transcript into sentences using terminal punctuation."""
    return [part.strip() for part in SENTENCE_PATTERN.split(text.strip()) if part.strip()]

def tokenize(text: str) -> list[str]:
    """Tokenize words and punctuation without requiring downloaded corpora."""
    return TOKEN_PATTERN.findall(text)

def simple_lemma(token: str) -> str:
    """Small rule-based lemma helper; not a full linguistic lemmatizer."""
    word = token.lower()
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "for", "with"
}

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Optional utility for text exploration; do not use before POS tagging."""
    return [token for token in tokens if token.lower() not in BASIC_STOPWORDS]
