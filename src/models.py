"""Transparent rule-based POS tagging for technical repair transcripts.

The pipeline explicitly records:
1. lexical regex rules,
2. morphological regex rules,
3. DefaultTagger fallback,
4. contextual corrections.
"""
import re
from nltk.tag import RegexpTagger, DefaultTagger
from preprocessing import tokenize

# Stage A: lexical/structural regular-expression rules.
LEXICAL_PATTERNS = [
    (r"^[.!?,;:]+$", "PUNCT"),
    (r"^\d+(?:\.\d+)?$", "NUM"),
    (r"^(the|a|an|this|that|these|those|each|every|my|your|his|her|their)$", "DET"),
    (r"^(i|we|you|he|she|it|they|me|us|them)$", "PRON"),
    (r"^(is|am|are|was|were|be|been|being|has|have|had|do|does|did|will|can|could|should|must|may|might)$", "AUX"),
    (r"^(to|in|on|at|with|from|into|of|for|by|under|over|after|before|near)$", "ADP"),
    (r"^(and|or|but|then|because)$", "CONJ"),
    (r"^(damaged|broken|faulty|loose|worn|hot|cold|low|high|electrical|hydraulic|metal|intact)$", "ADJ"),
    (r"^(carefully|slowly|quickly|not)$", "ADV"),
    (r"^(check|connect|disconnect|replace|remove|tighten|loosen|install|inspect|measure|turn|open|close|attach|adjust|test|press|rotate|clean|repair|run|start|stop|calibrate|examine|change|fix)$", "VERB"),
]

# Stage B: morphology-based regular-expression rules. They are deliberately simple heuristics.
MORPHOLOGICAL_PATTERNS = [
    (r".*ly$", "ADV"),
    (r".*ing$", "VERB"),
    (r".*ed$", "VERB"),
    (r".*(ous|ful|able|ible|ive|al)$", "ADJ"),
]

DEFAULT_TAG = "NOUN"
DEFAULT_TAGGER = DefaultTagger(DEFAULT_TAG)

# The two regex stages form a true NLTK RegexpTagger cascade:
# lexical patterns are tried first, morphology patterns second, default fallback last.
LEXICAL_TAGGER = RegexpTagger(LEXICAL_PATTERNS, backoff=RegexpTagger(MORPHOLOGICAL_PATTERNS, backoff=DEFAULT_TAGGER))
# Alias retained for clear assignment terminology.
BASE_TAGGER = LEXICAL_TAGGER

AMBIGUOUS_WORDS = {"pump", "wire", "gauge", "torque", "pressure"}
VERB_WORDS = {
    "pump", "wire", "gauge", "torque", "check", "connect", "disconnect",
    "replace", "remove", "tighten", "loosen", "install", "inspect", "measure",
    "turn", "open", "close", "attach", "adjust", "test", "press", "rotate",
    "clean", "repair", "run", "start", "stop", "calibrate", "examine", "change", "fix"
}
DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "each", "every",
    "my", "your", "his", "her", "their"
}
MODALS = {"will", "can", "could", "should", "must", "may", "might"}
PRONOUNS = {"i", "we", "you", "he", "she", "it", "they"}

def _match_pattern(word: str, patterns):
    for pattern, tag in patterns:
        if re.match(pattern, word):
            return pattern, tag
    return None, None

def tag_tokens_with_trace(tokens: list[str], rule_order: str = "lexical_first") -> list[dict]:
    """Return token decisions. rule_order supports controlled rule-priority experiments.

    lexical_first: production order (lexical, then morphological, then default).
    morphology_first: morphology rules take priority over lexical rules.
    reverse_lexical: reverse lexical rule order, keeping morphology second.
    reverse_morphological: reverse morphological rule order, keeping lexical first.
    """
    if rule_order == "lexical_first":
        lexical_patterns = LEXICAL_PATTERNS
        morphological_patterns = MORPHOLOGICAL_PATTERNS
        pattern_groups = [("lexical", lexical_patterns), ("morphological", morphological_patterns)]
    elif rule_order == "morphology_first":
        lexical_patterns = LEXICAL_PATTERNS
        morphological_patterns = MORPHOLOGICAL_PATTERNS
        pattern_groups = [("morphological", morphological_patterns), ("lexical", lexical_patterns)]
    elif rule_order == "reverse_lexical":
        lexical_patterns = list(reversed(LEXICAL_PATTERNS))
        morphological_patterns = MORPHOLOGICAL_PATTERNS
        pattern_groups = [("lexical", lexical_patterns), ("morphological", morphological_patterns)]
    elif rule_order == "reverse_morphological":
        lexical_patterns = LEXICAL_PATTERNS
        morphological_patterns = list(reversed(MORPHOLOGICAL_PATTERNS))
        pattern_groups = [("lexical", lexical_patterns), ("morphological", morphological_patterns)]
    else:
        raise ValueError(f"Unknown rule_order: {rule_order}")
    lower_tokens = [token.lower() for token in tokens]
    trace = []

    for i, token in enumerate(tokens):
        word = lower_tokens[i]
        matched_group = None
        matched_pattern = None
        initial_tag = DEFAULT_TAG
        for group_name, patterns in pattern_groups:
            pattern, tag = _match_pattern(word, patterns)
            if tag is not None:
                matched_group, matched_pattern, initial_tag = group_name, pattern, tag
                break

        if matched_group == "lexical":
            source = "Regex: lexical rule"
            rule_detail = f"Lexical regex {matched_pattern!r} matched; assigned {initial_tag}."
        elif matched_group == "morphological":
            source = "Regex: morphological rule"
            rule_detail = f"Morphological regex {matched_pattern!r} matched; assigned {initial_tag}."
        else:
            source = "DefaultTagger fallback"
            matched_pattern = None
            rule_detail = f"No lexical or morphological regex matched; DefaultTagger assigned {DEFAULT_TAG}."

        trace.append({
            "token": token,
            "initial_tag": initial_tag,
            "final_tag": initial_tag,
            "source": source,
            "matched_pattern": matched_pattern,
            "rule_detail": rule_detail,
            "context_detail": "No contextual correction applied.",
            "context_applied": False,
        })

    # Stage C: explicit context-based correction pass.
    for i, token in enumerate(tokens):
        word = lower_tokens[i]
        previous = lower_tokens[i - 1] if i > 0 else ""
        next_word = lower_tokens[i + 1] if i + 1 < len(tokens) else ""
        new_tag = trace[i]["initial_tag"]
        reason = None

        if re.fullmatch(r"[.!?,;:]+", token):
            new_tag = "PUNCT"
            reason = "Punctuation token is assigned PUNCT."
        elif word in AMBIGUOUS_WORDS:
            if previous in DETERMINERS:
                new_tag = "NOUN"
                reason = f"'{tokens[i-1]}' is a determiner, so '{token}' is treated as a noun."
            elif previous in MODALS or previous == "to":
                new_tag = "VERB"
                reason = f"'{tokens[i-1]}' signals an action, so '{token}' is treated as a verb."
            elif i == 0 and word in VERB_WORDS and next_word in DETERMINERS:
                new_tag = "VERB"
                reason = "Sentence-initial command followed by a determiner suggests a verb."
            elif previous in PRONOUNS and word in VERB_WORDS:
                new_tag = "VERB"
                reason = f"The pronoun '{tokens[i-1]}' before this action word suggests a verb."
            elif next_word in DETERMINERS and previous not in DETERMINERS:
                new_tag = "VERB"
                reason = "Neighboring-word pattern suggests an instruction/action verb."

        if reason:
            trace[i]["context_detail"] = reason
            trace[i]["context_applied"] = new_tag != trace[i]["initial_tag"]
        trace[i]["final_tag"] = new_tag

    return trace

def tag_tokens(tokens: list[str], rule_order: str = "lexical_first") -> list[tuple[str, str]]:
    """Return standard (token, final POS tag) pairs; optional order is for experiments."""
    return [(row["token"], row["final_tag"]) for row in tag_tokens_with_trace(tokens, rule_order=rule_order)]

def tag_sentence(sentence: str) -> list[tuple[str, str]]:
    """Tokenize and tag a sentence."""
    return tag_tokens(tokenize(sentence))
