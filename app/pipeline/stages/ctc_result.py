
from app.pipeline.ctc_align.CTCWordAligner import CTCWordAligner
# from CTCWordAligner import CTCWordAligner
# app/pipeline/ctc_pipeline.py
import logging
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
from typing import List, Tuple
logger = logging.getLogger(__name__)


def ctc_alignment_result(
    data: tuple[str, str],
    processor: Wav2Vec2Processor,
    ctc_model: Wav2Vec2ForCTC,
) -> list[dict]:
    """
    Runs CTC forced alignment on the transcribed audio to produce
    word-level timestamps.

    Args:
        data: (whisper_result, audio_path) tuple from the transcription step.
        processor: Loaded Wav2Vec2Processor instance.
        ctc_model: Loaded Wav2Vec2ForCTC instance.

    Returns:
        List of word-level alignment dicts (word, start, end, pause_after, etc.)
    """
    whisper_result, audio_path = data

    try:
        waveform, sr = torchaudio.load(audio_path)
        audio_arr = waveform.numpy().squeeze()

        logger.debug(
            f"Loaded audio '{audio_path}' — sr={sr}, "
            f"range=[{audio_arr.min():.4f}, {audio_arr.max():.4f}]"
        )

        aligner = CTCWordAligner(processor, ctc_model)
        ctc_res = aligner.align(whisper_result, audio_arr)

        logger.info(f"CTC alignment complete: {len(ctc_res)} words aligned")
        logger.info(f"CTC alignment result: {ctc_res}")
        logger.info(f"Type of CTC alignment result: {type(ctc_res)}")
        
        return ctc_res

    except Exception as e:
        logger.error(f"CTC alignment failed for {audio_path}: {e}")
        raise

def _convert_transcription_res_to_single_transcription_and_audio_array_list(
    data: list[tuple[str, str]]
) -> tuple[list[str],list[np.ndarray]]:
    transcriptions = []
    audio_arrays = []
    for whisper_result, audio_path in data:
        waveform, sr = torchaudio.load(audio_path)
        audio_arr = waveform.numpy().squeeze()
        transcriptions.append(whisper_result)
        logger.debug(
            f"Loaded audio '{audio_path}' — sr={sr}, "
            f"range=[{audio_arr.min():.4f}, {audio_arr.max():.4f}]"
        )
        audio_arrays.append(audio_arr)
    return (transcriptions, audio_arrays)


def batch_ctc_alignment_result(
    data: list[tuple[str, str]],
    processor: Wav2Vec2Processor,
    ctc_model: Wav2Vec2ForCTC,
    ) -> list[list[dict] | None]:

    transcriptions, audio_arrays = _convert_transcription_res_to_single_transcription_and_audio_array_list(data)
    try:
        aligner = CTCWordAligner(processor, ctc_model)
        ctc_res = aligner.align(transcriptions, audio_arrays,batch=True)

        logger.info(f"Batch CTC alignment complete: {len(ctc_res)} items aligned")
        return ctc_res
    except Exception as e:
        logger.error(f"Batch CTC alignment failed: {e}")
        raise
    