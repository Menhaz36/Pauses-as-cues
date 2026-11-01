"""
Scoring used in the paper (Sect. 3.1 and 4): text is upper-cased and every character
other than letters, digits, apostrophes and whitespace is deleted; WER/CER/MER are
corpus-level (jiwer). Same code as the evaluation notebook, cells 16, 26 and 28.
"""
import re

import jiwer
import numpy as np

from app.pipeline.stages.regions import normalize_words


def norm_del(text) -> str:
    """Scoring normalization of the paper: punctuation deleted."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9'\s]", "", str(text).upper())).strip()


def norm_space(text) -> str:
    """Alternative: punctuation → space (counts hyphenated loops word by word)."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9'\s]", " ", str(text).upper())).strip()


def corpus_metrics(refs: list[str], hyps: list[str]) -> dict:
    """Corpus WER, CER and MER over already-normalized references and hypotheses."""
    refs = [r if r else "X" for r in refs]
    o = jiwer.process_words(refs, hyps)
    n = o.hits + o.substitutions + o.deletions
    return {"WER": (o.substitutions + o.deletions + o.insertions) / n,
            "CER": jiwer.cer(refs, hyps), "MER": jiwer.mer(refs, hyps), "ref_words": n}


def wer_details(reference: str, hypothesis: str) -> dict:
    """Per-utterance S/D/I counts (the tie-breaking that Table 3 was computed with)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int); back = np.empty((n + 1, m + 1), dtype=object)
    for i in range(1, n + 1): dp[i, 0], back[i, 0] = i, "D"
    for j in range(1, m + 1): dp[0, j], back[0, j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j], back[i, j] = dp[i - 1, j - 1], "C"
            else:
                dp[i, j], back[i, j] = min([(dp[i-1, j-1] + 1, "S"), (dp[i-1, j] + 1, "D"), (dp[i, j-1] + 1, "I")], key=lambda x: x[0])
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        op = back[i, j]
        if op == "C": i -= 1; j -= 1
        elif op == "S": S += 1; i -= 1; j -= 1
        elif op == "D": D += 1; i -= 1
        elif op == "I": I += 1; j -= 1
        else: break
    return {"S": S, "D": D, "I": I, "errors": S + D + I, "N": n, "wer": (S + D + I) / n if n else np.nan}
