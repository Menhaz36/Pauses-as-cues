import logging

from app.config import PAUSE_THRESHOLD, LEADING_PAUSE_MARKER
from app.pipeline.stages.regions import contains_between_word_pause
from app.pipeline.utterance import Utterance

logger = logging.getLogger(__name__)

def format_pause_aware_transcript(
    ctc_result: list[dict],
    pause_threshold: float = PAUSE_THRESHOLD,
    leading: bool = LEADING_PAUSE_MARKER,
) -> str:
    """
    Inserts a marker [δs_pause] between two consecutive words whenever their gap δ ≥ τ
    (Eq. 3). Markers are single tokens so that region extraction can split on whitespace.
    """

    formatted_text = []

    for index, item in enumerate(ctc_result):

        # Silence before the first word (off in the paper's setting)
        if index == 0 and leading:
            start_pause = item.get("start", 0.0) or 0.0

            if start_pause >= pause_threshold:
                formatted_text.append(
                    f"[{start_pause:.1f}s_pause]"
                )

        # Word
        formatted_text.append(item["word"])

        # Silence after the word
        pause_after = item.get("pause_after", 0.0) or 0.0

        if pause_after >= pause_threshold:
            formatted_text.append(
                f"[{pause_after:.1f}s_pause]"
            )

    return " ".join(formatted_text)


def pause_format_stage(
    utt: Utterance,
    pause_threshold: float = PAUSE_THRESHOLD,
) -> Utterance:
    """Sets `formatted_text` and `has_pause`; a failed alignment gets no markers."""
    if utt.words is None:
        utt.formatted_text, utt.has_pause = utt.asr_text, False
    else:
        utt.formatted_text = format_pause_aware_transcript(utt.words, pause_threshold)
        utt.has_pause = contains_between_word_pause(utt.formatted_text)

    logger.info(f"Formatted transcript: {utt.formatted_text}")
    return utt


def batch_format_pause_aware_transcript(
      utts: list[Utterance],
      pause_threshold: float = PAUSE_THRESHOLD,
      ) -> list[Utterance]:
    return [pause_format_stage(utt, pause_threshold) for utt in utts]
