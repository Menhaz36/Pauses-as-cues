from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor # type: ignore
from app.config import WAV2VEC2_MODEL_ID, get_device_and_dtype, torch_device_name
import logging

logger = logging.getLogger(__name__)


def get_processor(model_id: str = WAV2VEC2_MODEL_ID) -> Wav2Vec2Processor:

      try:
            logger.info(f"Loading Wav2Vec2 processor for model {model_id}")
            processor = Wav2Vec2Processor.from_pretrained(
                  model_id
                  )
            logger.info(f"Wav2Vec2 processor loaded: {model_id}")
            return processor

      except Exception as e:
            logger.error(f"Error loading Wav2Vec2 processor {model_id}: {e}")
            raise e


def get_ctcmodel(model_id: str = WAV2VEC2_MODEL_ID) -> Wav2Vec2ForCTC:

      try:
            device = torch_device_name(get_device_and_dtype()[0])
            logger.info(f"Loading Wav2Vec2 model {model_id} on {device}")
            model = Wav2Vec2ForCTC.from_pretrained(
                  model_id
                  ).to(device) # type: ignore
            logger.info(f"Wav2Vec2 model loaded: {model_id}")
            return model

      except Exception as e:
            logger.error(f"Error loading Wav2Vec2 model {model_id}: {e}")
            raise e
