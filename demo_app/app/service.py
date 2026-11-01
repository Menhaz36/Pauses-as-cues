"""
Model service used by the API: loads the models once and compares the three systems of
the paper on an audio file — zero-shot Whisper-tiny, fine-tuned Whisper-tiny, and
fine-tuned + pause-guided LLM post-correction.
"""
import logging
import threading

import jiwer
from scipy.stats import binomtest

from app import config
from app.eval.metrics import corpus_metrics, norm_del, wer_details
from app.pipeline.stages.correction import correct_utterance
from app.pipeline.stages.ctc_result import ctc_alignment_result
from app.pipeline.stages.pauseAppend import pause_format_stage
from app.pipeline.stages.regions import normalize_words, remove_pause_markers
from app.pipeline.stages.transcription import transcribe_audio

logger = logging.getLogger(__name__)

SYSTEMS = {
    "zero_shot": "Zero-shot Whisper-tiny",
    "finetuned": "Fine-tuned Whisper-tiny",
    "pause_llm": "+ pause-guided LLM",
}


def alignment_ops(reference: str, hypothesis: str) -> list[dict]:
    """Word alignment of a normalized hypothesis against the reference, for a diff view."""
    out = jiwer.process_words(reference, hypothesis or "")
    ref, hyp = out.references[0], out.hypotheses[0]
    ops = []
    for chunk in out.alignments[0]:
        r = ref[chunk.ref_start_idx:chunk.ref_end_idx]
        h = hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type in ("equal", "substitute"):
            ops += [{"op": chunk.type, "ref": a, "hyp": b} for a, b in zip(r, h)]
        elif chunk.type == "delete":
            ops += [{"op": "delete", "ref": a, "hyp": None} for a in r]
        else:
            ops += [{"op": "insert", "ref": None, "hyp": b} for b in h]
    return ops


def score(reference: str, hypothesis: str) -> dict:
    ref, hyp = norm_del(reference), norm_del(hypothesis)
    m = corpus_metrics([ref], [hyp])
    d = wer_details(ref, hyp)
    return {"WER": m["WER"], "CER": m["CER"], "MER": m["MER"], "S": d["S"], "D": d["D"], "I": d["I"],
            "errors": d["errors"], "ref_words": d["N"], "alignment": alignment_ops(ref, hyp)}


class PipelineService:
    def __init__(self):
        self._lock = threading.Lock()   # the models are not safe to run concurrently
        self.finetuned = self.baseline = self.processor = self.ctc_model = self.llm = None
        self.llm_error = None

    def load(self):
        from app.models.wav2vec import get_ctcmodel, get_processor
        from app.models.whisper import get_baseline_asr_model, get_finetuned_asr_model

        config.hf_login()
        self.finetuned = get_finetuned_asr_model()
        self.baseline = get_baseline_asr_model()
        self.processor, self.ctc_model = get_processor(), get_ctcmodel()
        try:
            from app.models.llm import get_llm
            self.llm = get_llm()
        except Exception as e:    # the UI still works without the LLM stage
            self.llm_error = str(e)
            logger.warning(f"LLM unavailable: {e}")

    def status(self) -> dict:
        return {"models_loaded": self.finetuned is not None, "llm_available": self.llm is not None,
                "llm_error": self.llm_error, "llm_model": config.LLM_MODEL_ID,
                "default_tau": config.PAUSE_THRESHOLD}

    def analyze(self, audio_path: str, reference: str | None = None, tau: float = config.PAUSE_THRESHOLD,
                run_baseline: bool = True, run_llm: bool = True) -> dict:
        with self._lock:
            utt = transcribe_audio(self.finetuned, audio_path)
            texts = {"finetuned": utt.asr_text}
            if run_baseline:
                texts["zero_shot"] = self.baseline({"raw": utt.audio, "sampling_rate": config.SAMPLING_RATE}).get("text", "")
            ctc_alignment_result(utt, self.processor, self.ctc_model)
            pause_format_stage(utt, tau)
            llm_note = None
            if run_llm and self.llm is not None:
                correct_utterance(utt, self.llm)
                texts["pause_llm"] = utt.final_text
                # the paper's consistency filter (365 → 363): the corrected text is built from the CTC
                # words, so if they differ from the ASR words (e.g. quote marks the scoring keeps),
                # scoring it against the ASR text would credit the LLM for a normalization difference
                if utt.has_pause and normalize_words(utt.asr_text) != normalize_words(remove_pause_markers(utt.formatted_text)):
                    texts["pause_llm"] = utt.asr_text
                    llm_note = (f"The aligned words differ from the ASR words after normalization, so the LLM output "
                                f"({utt.final_text!r}) is not scored; the ASR text is kept, as in the paper's evaluation.")
            elif run_llm:
                llm_note = f"LLM stage skipped: {self.llm_error}"

        result = {
            "tau": tau,
            "duration_s": len(utt.audio) / config.SAMPLING_RATE,
            "systems": {k: {"label": SYSTEMS[k], "text": texts[k]} for k in SYSTEMS if k in texts},
            "words": utt.words, "alignment_failed": utt.words is None,
            "formatted_text": utt.formatted_text, "has_pause": utt.has_pause,
            "sent_to_llm": bool(run_llm and self.llm is not None and utt.has_pause),
            "regions": utt.regions, "llm_note": llm_note,
        }
        if reference and norm_del(reference):
            result["reference"] = reference
            for k, s in result["systems"].items():
                s["metrics"] = score(reference, s["text"])
        return result


def summarize(results: list[dict]) -> dict:
    """Corpus metrics per system over files that have a reference, plus the LLM sign test."""
    scored = [r for r in results if "reference" in r]
    corpus = {}
    for k in SYSTEMS:
        pairs = [(norm_del(r["reference"]), norm_del(r["systems"][k]["text"])) for r in scored if k in r["systems"]]
        if pairs:
            corpus[k] = {"label": SYSTEMS[k], "files": len(pairs), **corpus_metrics([p[0] for p in pairs], [p[1] for p in pairs])}
    llm = None
    both = [r for r in scored if "pause_llm" in r["systems"]]
    if both:
        before = [r["systems"]["finetuned"]["metrics"]["errors"] for r in both]
        after = [r["systems"]["pause_llm"]["metrics"]["errors"] for r in both]
        imp = sum(a < b for b, a in zip(before, after)); wor = sum(a > b for b, a in zip(before, after))
        llm = {"files": len(both), "sent_to_llm": sum(r["sent_to_llm"] for r in both),
               "errors_before": sum(before), "errors_after": sum(after), "improved": imp, "worsened": wor,
               "unchanged": len(both) - imp - wor,
               "sign_test_p": binomtest(imp, imp + wor, 0.5).pvalue if imp + wor else None}
    return {"files": len(results), "scored_files": len(scored), "corpus": corpus, "llm": llm}
