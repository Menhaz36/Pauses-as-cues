
from app.pipeline.ctc_align.CTCWordAligner import CTCWordAligner
from app.pipeline.utterance import Utterance
from app.config import CTC_PADDED_BATCH
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
logger = logging.getLogger(__name__)

# ctc-segmentation raises these when no valid path exists (e.g. degenerate
# repetitive hypotheses such as "k-k-k-..."); the utterance then gets no markers
ALIGNMENT_ERRORS = (AssertionError, IndexError)


def ctc_alignment_result(
    utt: Utterance,
    processor: Wav2Vec2Processor,
    ctc_model: Wav2Vec2ForCTC,
) -> Utterance:
    """
    Runs CTC forced alignment of the ASR hypothesis against the audio to produce
    word-level timestamps.

    Args:
        utt: Utterance from the transcription step (audio + ASR text).
        processor: Loaded Wav2Vec2Processor instance.
        ctc_model: Loaded Wav2Vec2ForCTC instance.

    Returns:
        The same Utterance with `words` set to the word-level alignment
        (word, start, end, pause_after, ...), or None if alignment failed.
    """
    aligner = CTCWordAligner(processor, ctc_model)
    utt.words = _align_or_none(aligner, utt)
    if utt.words is not None:
        logger.info(f"CTC alignment complete: {len(utt.words)} words aligned")
    return utt


def batch_ctc_alignment_result(
    utts: list[Utterance],
    processor: Wav2Vec2Processor,
    ctc_model: Wav2Vec2ForCTC,
    ) -> list[Utterance]:
    """
    Aligns each utterance individually (the paper's setting). Zero-padding a batch
    changes the wav2vec2-base outputs, so the padded batch path is used only when
    CTC_PADDED_BATCH is switched on explicitly.
    """
    aligner = CTCWordAligner(processor, ctc_model)

    if CTC_PADDED_BATCH:
        logger.warning("Padded CTC batching is on: word timings will differ from the paper's.")
        results = aligner.align([u.asr_text for u in utts], [u.audio for u in utts], batch=True)
        for utt, words in zip(utts, results):
            utt.words = words
    else:
        for utt in utts:
            utt.words = _align_or_none(aligner, utt)

    failed = sum(u.words is None for u in utts)
    logger.info(f"Batch CTC alignment complete: {len(utts) - failed} aligned, {failed} failed")
    return utts


def _align_or_none(aligner: CTCWordAligner, utt: Utterance) -> list[dict] | None:
    try:
        return aligner.align(utt.asr_text, utt.audio)
    except ALIGNMENT_ERRORS as e:
        logger.warning(
            f"CTC alignment failed for {utt.audio_path} [{type(e).__name__}]: {e} | text={utt.asr_text!r}"
        )
        return None
