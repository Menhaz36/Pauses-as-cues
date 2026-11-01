import logging
import time

from app.config import LLM_CONTEXT_WORDS, LLM_RETRIES
from app.pipeline.prompts.template import final_prompt
from app.pipeline.stages.regions import (
    extract_pause_regions,
    reconstruct_sentence,
    remove_pause_markers,
    validate_llm_output,
)
from app.pipeline.utterance import Utterance

logger = logging.getLogger(__name__)


def response_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()


def _ask_llm(chain, segment: str, retries: int = LLM_RETRIES) -> tuple[str | None, str | None]:
    """Returns (text, error). Retries with exponential backoff (Groq rate limits)."""
    err = None
    for attempt in range(retries):
        try:
            return response_text(chain.invoke({"segment": segment})), None
        except Exception as e:
            err = str(e)[:300]
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{retries}): {err}")
            time.sleep(2 * 2 ** attempt)
    return None, err


def correct_utterance(utt: Utterance, llm, k: int = LLM_CONTEXT_WORDS) -> Utterance:
    """
    Pause-guided post-correction. Only utterances with a between-word pause are sent;
    each ±k-word region around the markers is corrected on its own, passed through the
    edit-distance guard, and substituted back. All other utterances keep the ASR text.
    """
    if not utt.has_pause:
        utt.final_text = utt.asr_text
        return utt

    chain = final_prompt | llm
    utt.regions = []
    for region in extract_pause_regions(utt.formatted_text, k):
        raw, err = _ask_llm(chain, region["input_region"])
        accepted, reason = validate_llm_output(region["input_region"], raw)
        utt.regions.append({**region, "raw_llm_output": raw, "llm_error": err,
                            "accepted_output": accepted, "validation_reason": reason})

    utt.final_text = remove_pause_markers(reconstruct_sentence(utt.formatted_text, utt.regions))
    logger.info(f"Post-corrected transcript: {utt.final_text}")
    return utt


def batch_correct_utterances(utts: list[Utterance], llm, k: int = LLM_CONTEXT_WORDS) -> list[Utterance]:
    # one call at a time: avoids bursts of 429s on the Groq free tier
    return [correct_utterance(utt, llm, k) for utt in utts]
