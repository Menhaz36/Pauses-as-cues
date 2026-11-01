"""
Text-only parts of the LLM post-correction stage: pause markers, local regions,
the edit-distance guard and sentence reconstruction. These are the functions of the
evaluation notebook (cells 18 and 24) that produced the paper's numbers; keep them
identical, scripts/check_against_paper_run.py verifies that they still are.
"""
import re

from app.config import LLM_CONTEXT_WORDS, LLM_MAX_ABS_EDITS, LLM_MAX_EDIT_RATIO

PAUSE_RE = re.compile(r"^\[\d+(?:\.\d+)?s_pause\]$", re.IGNORECASE)
BETWEEN_RE = re.compile(r"\S+\s+\[\d+(?:\.\d+)?s_pause\]\s+\S+")


def is_pause(token: str) -> bool:
    return bool(PAUSE_RE.fullmatch(token))


def contains_between_word_pause(text: str) -> bool:
    """True if a marker sits between two words (the utterance is sent to the LLM)."""
    return bool(BETWEEN_RE.search(str(text)))


def remove_pause_markers(text: str) -> str:
    return " ".join(t for t in str(text).split() if not is_pause(t))


def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^A-Z0-9'\s]", "", str(text).upper()).split()


def extract_pause_regions(text: str, k: int = LLM_CONTEXT_WORDS) -> list[dict]:
    """
    Selects k words on each side of every pause marker and merges overlapping selections.
    Returns token spans {start, end, input_region} over the whitespace-split text.
    """
    tokens = str(text).split(); n = len(tokens)
    selected = [False] * n
    for p in [i for i, t in enumerate(tokens) if is_pause(t)]:
        selected[p] = True
        for step in (-1, 1):
            seen, i = 0, p + step
            while 0 <= i < n and seen < k:
                selected[i] = True
                if not is_pause(tokens[i]): seen += 1
                i += step
    regions, i = [], 0
    while i < n:
        if not selected[i]: i += 1; continue
        start = i
        while i < n and selected[i]: i += 1
        regions.append({"start": start, "end": i - 1, "input_region": " ".join(tokens[start:i])})
    return regions


def word_edit_distance(a: str, b: str) -> int:
    a, b = normalize_words(remove_pause_markers(a)), normalize_words(remove_pause_markers(b))
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (wa != wb), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def clean_llm_output(text: str | None) -> str | None:
    if text is None: return None
    text = re.sub(r"^```(?:text)?", "", str(text).strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(corrected|output|corrected segment)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip("\"' ")


def validate_llm_output(
    original: str,
    candidate: str | None,
    max_abs_edits: int = LLM_MAX_ABS_EDITS,
    max_edit_ratio: float = LLM_MAX_EDIT_RATIO,
) -> tuple[str, str]:
    """
    Edit-distance guard: an output that differs from the region by more than
    `max_abs_edits` words or more than `max_edit_ratio` of its words is rejected and
    the original region is kept. Returns (accepted_output, validation_reason).
    """
    if candidate is None: return original, "llm_failure"
    candidate = clean_llm_output(candidate)
    if not candidate: return original, "empty_output"
    dist = word_edit_distance(original, candidate)
    if dist > max_abs_edits or dist / max(len(normalize_words(remove_pause_markers(original))), 1) > max_edit_ratio:
        return original, "excessive_rewrite"
    return candidate, "accepted"


def reconstruct_sentence(formatted_text: str, regions: list[dict]) -> str:
    """Substitutes each region's accepted output back into the formatted text."""
    tokens = str(formatted_text).split()
    for region in sorted(regions, key=lambda r: r["start"], reverse=True):
        tokens[region["start"]:region["end"] + 1] = str(region["accepted_output"]).split()
    return " ".join(tokens)
