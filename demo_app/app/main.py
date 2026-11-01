from app.models.whisper import get_finetuned_asr_model
from app.models.llm import get_llm
from app.models.wav2vec import get_ctcmodel, get_processor
from app.config import hf_login
from app.pipeline.chain import full_pipeline_chain
from app.pipeline.utterance import Utterance
import json
import logging
import sys
import asyncio

logger = logging.getLogger(__name__)



hf_login()

asr_model=get_finetuned_asr_model()

llm=get_llm()

processor=get_processor()

ctc_model=get_ctcmodel()


speech_repair_chain=full_pipeline_chain(asr_model=asr_model,processor=processor,ctc_model=ctc_model,llm=llm)


async def process_audio(file_path: str) -> Utterance:
    """Runs the full pipeline on one audio file and returns every intermediate result."""
    return await speech_repair_chain.ainvoke(file_path)


if __name__=="__main__":
      path = (
              sys.argv[1]
              if len(sys.argv) > 1
              else r"D:\D\IMPORTANT\Dysarthria\ASR\Sample_dysarthria_audio.wav"
          )

      result = asyncio.run(process_audio(path))
      print(json.dumps(result.to_dict(), indent=2))
      print("Final Post-corrected Transcript:", result.final_text)
