from app import config
from app.models.whisper import asr_model, get_finetuned_asr_model
from app.models.llm import get_llm
from app.models.wav2vec import get_ctcmodel, get_processor
from app.config import hf_login
from app.pipeline.chain import full_pipeline_chain
import logging
import sys
import asyncio

logger = logging.getLogger(__name__)



hf_login()

asr_model=get_finetuned_asr_model()
# from huggingface_hub import try_to_load_from_cache
# def check_cache(model_id):
#       path = try_to_load_from_cache(model_id, "config.json")
#       if path:
#             logger.info(f"✓ {model_id} loaded from cache")
#             logger.info(f"  {path}")
#       else:
#             logger.info(f"↓ {model_id} will be downloaded")

    
# check_cache(config.FINE_TUNED_MODEL_ID)

llm=get_llm()

processor=get_processor()

ctc_model=get_ctcmodel()


speech_repair_chain=full_pipeline_chain(asr_model=asr_model,processor=processor,ctc_model=ctc_model,llm=llm)


async def process_audio(file_path: str) -> str:
    """Entry point called by main1.py """
    return await speech_repair_chain.ainvoke(file_path)


if __name__=="__main__":
      path = (
              sys.argv[1]
              if len(sys.argv) > 1
              else r"D:\D\IMPORTANT\Dysarthria\ASR\Sample_dysarthria_audio.wav"
          )
      
      result = asyncio.run(process_audio(path))
      print("Final Repaired Transcript:", result)













