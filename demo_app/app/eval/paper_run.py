"""
Loads the saved run behind the paper (evaluation notebook, MODE="fixed") and replays it
through the code in app/. Used by scripts/check_against_paper_run.py and
scripts/export_paper_results.py.
"""
import json
import os

import pandas as pd

from app import config
from app.eval.metrics import norm_del, wer_details
from app.pipeline.stages.pauseAppend import format_pause_aware_transcript
from app.pipeline.stages.regions import (
    contains_between_word_pause, extract_pause_regions, normalize_words,
    reconstruct_sentence, remove_pause_markers, validate_llm_output,
)


def load_run(run_dir: str, tau: float = config.PAUSE_THRESHOLD) -> pd.DataFrame:
    """One row per utterance: metadata, ASR hypothesis, CTC words, formatted text at τ."""
    ck = os.path.join(run_dir, "checkpoints")
    meta = pd.read_parquet(os.path.join(ck, "split.parquet"))
    asr = pd.read_parquet(os.path.join(ck, "asr_finetuned.parquet"))
    ctc = pd.read_parquet(os.path.join(ck, "ctc.parquet"))
    df = meta.merge(asr, on="sample_id").merge(ctc, on="sample_id")
    df["words"] = df.ctc_json.map(lambda s: json.loads(s) if isinstance(s, str) else None)
    df["formatted"] = [format_pause_aware_transcript(w, tau) if w is not None else "" for w in df.words]
    df["has_pause"] = df.formatted.map(contains_between_word_pause)
    return df


def load_llm_answers(run_dir: str, tau: float = config.PAUSE_THRESHOLD) -> dict:
    """Saved LLM answers by job key; the last record of a job wins (later retries)."""
    answers = {}
    with open(os.path.join(run_dir, "checkpoints", f"llm_jobs_pause_tau{tau:g}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line); answers[rec["job_key"]] = rec
    return answers


def replay_llm(df: pd.DataFrame, answers: dict) -> tuple[pd.DataFrame, dict, int]:
    """
    Rebuilds the LLM stage from the saved answers: regions, guard decisions and the
    post-corrected sentences. Returns (per-utterance results, regions by sample_id,
    number of regions that differ from the saved jobs).
    """
    sel = df[df.has_pause]
    regions_by_id, mismatched = {}, 0
    for row in sel.itertuples():
        regs = extract_pause_regions(row.formatted)
        for rid, reg in enumerate(regs):
            rec = answers.get(f"pause|{int(row.sample_id)}|{rid}")
            if rec is None or (rec["start"], rec["end"], rec["input_region"]) != (reg["start"], reg["end"], reg["input_region"]):
                mismatched += 1; continue
            reg["raw_llm_output"] = rec["raw_llm_output"]
            reg["accepted_output"], reg["validation_reason"] = validate_llm_output(reg["input_region"], rec["raw_llm_output"])
        regions_by_id[int(row.sample_id)] = regs

    out = sel[["sample_id", "transcription", "asr_transcription", "speech_status", "split", "speaker", "formatted"]].copy()
    out["baseline"] = out.formatted.map(remove_pause_markers)
    out["llm_output"] = [remove_pause_markers(reconstruct_sentence(t, regions_by_id[s])) for t, s in zip(out.formatted, out.sample_id)]
    # consistency filter of the notebook: the formatted words must equal the ASR words (365 → 363)
    out = out[[normalize_words(a) == normalize_words(b) for a, b in zip(out.asr_transcription, out.baseline)]].copy()
    b = pd.DataFrame([wer_details(r, h) for r, h in zip(out.transcription, out.baseline)], index=out.index)
    l = pd.DataFrame([wer_details(r, h) for r, h in zip(out.transcription, out.llm_output)], index=out.index)
    out["err_before"], out["err_after"], out["N"] = b.errors, l.errors, b.N
    return out, regions_by_id, mismatched


def eval_frame(df: pd.DataFrame, llm_out: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Normalized reference and the three systems' hypotheses (zero-shot only on the test split)."""
    base = pd.read_parquet(os.path.join(run_dir, "checkpoints", "asr_baseline.parquet"))
    ev = df[["sample_id", "transcription", "asr_transcription", "speech_status", "split", "speaker"]].copy()
    ev["ref"] = ev.transcription.fillna("").map(norm_del)
    ev = ev[ev.ref != ""]
    ev["asr"] = ev.asr_transcription.map(norm_del)
    llm_map = dict(zip(llm_out.sample_id, llm_out.llm_output))
    ev["asr_llm"] = [norm_del(llm_map[s]) if s in llm_map else a for s, a in zip(ev.sample_id, ev.asr)]
    ev = ev.merge(base.rename(columns={"asr_transcription": "zero_shot_raw"}), on="sample_id", how="left")
    ev["zero_shot"] = ev.zero_shot_raw.map(lambda x: norm_del(x) if isinstance(x, str) else None)
    return ev
