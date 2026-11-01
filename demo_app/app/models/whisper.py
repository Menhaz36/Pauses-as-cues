import torch
import transformers
from packaging import version

from app.config import (
    FINE_TUNED_MODEL_ID,
    BASELINE_MODEL_ID,
    SAMPLING_RATE,
    FEATURE_SIZE,
    get_device_and_dtype,
)
import logging
from transformers import pipeline,WhisperFeatureExtractor

logger = logging.getLogger(__name__)


# Select device and dtype
device, torch_dtype = get_device_and_dtype()


def asr_model(model_id:str=FINE_TUNED_MODEL_ID,device:int=-1,torch_dtype:torch.dtype=torch.float32):
    """
    Returns the ASR model pipeline for the given model_id.

    """
    logger.info(f"loading ASR model {model_id} on device {device} with dtype {torch_dtype}")

    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id,
            feature_size=FEATURE_SIZE,
            sampling_rate=SAMPLING_RATE,
        )

        # transformers renamed `torch_dtype` to `dtype` in 4.56
        dtype_kw = (
            {"dtype": torch_dtype}
            if version.parse(transformers.__version__) >= version.parse("4.56")
            else {"torch_dtype": torch_dtype}
        )

        asr=pipeline(
            "automatic-speech-recognition",
            model=model_id,
            feature_extractor=feature_extractor,
            device=device,
            return_timestamps=False,              # word timings come from CTC alignment instead
            generate_kwargs={"language": "english", "task": "transcribe"},  # greedy decoding
            model_kwargs={"attn_implementation": "sdpa"}, # Uses PyTorch Scaled Dot Product Attention
            **dtype_kw,
        )
        logger.info(f"whisper ASR model loaded: {model_id}")

        return asr

    except Exception as e:
        logger.error(f"Error loading ASR model {model_id}: {e}")
        raise e


def get_finetuned_asr_model():
    """
    Returns the finetuned ASR model pipeline.
    """
    return asr_model(FINE_TUNED_MODEL_ID,device=device,torch_dtype=torch_dtype) # type: ignore

def get_baseline_asr_model():
    """
    Returns the zero-shot baseline ASR model pipeline (Table 2, first row).
    """
    return asr_model(BASELINE_MODEL_ID,device=device,torch_dtype=torch_dtype) # pyright: ignore[reportArgumentType]
