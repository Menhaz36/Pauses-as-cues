from langchain_core.runnables import RunnableLambda # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from functools import partial
from app.pipeline.stages.transcription import async_transcribe,batch_transcribe_audio
from app.pipeline.stages.ctc_result import ctc_alignment_result, batch_ctc_alignment_result
from app.pipeline.stages.pauseAppend import format_pause_aware_transcript, batch_format_pause_aware_transcript
from app.pipeline.prompts.template import final_prompt
from app.config import BATCH_SIZE, CHUNK_LENGTH_S
import os



def full_pipeline_chain(asr_model,processor,ctc_model,llm):
      """
            Creates a chain of runnables for the full audio processing pipeline.

            Args:
                  asr_model: The ASR model pipeline to use for transcription.
                  processor: Loaded Wav2Vec2Processor instance.
                  ctc_model: Loaded Wav2Vec2ForCTC instance.
            return :
                  RunnableLambda: A runnable that executes the full pipeline.
      """

      transcribe_runnable = RunnableLambda(
                  partial(async_transcribe, asr_model)
            )
      
      ctc_aligner_runnable =RunnableLambda(
                  partial(ctc_alignment_result, processor=processor, ctc_model=ctc_model)
            )

      pause_formatter_runnable=RunnableLambda(format_pause_aware_transcript)


      return (
            transcribe_runnable
            | ctc_aligner_runnable
            | pause_formatter_runnable
            | final_prompt
            | llm
            | StrOutputParser()
      )

def batch_full_pipeline_chain(asr_model,processor,ctc_model,llm):


      transcribe_runnable=RunnableLambda(
            partial(batch_transcribe_audio, asr_model, batch_size=BATCH_SIZE, chunk_length_s=CHUNK_LENGTH_S)
      )

      ctc_aligner_runnable=RunnableLambda(
            partial(batch_ctc_alignment_result,processor=processor, ctc_model=ctc_model)
      )

      pause_formatter_runnable=RunnableLambda(
            partial(batch_format_pause_aware_transcript)
      )

      repair_runnable = (
        final_prompt
        | llm
        | StrOutputParser()
      ).map()

      return (
            transcribe_runnable
            | ctc_aligner_runnable
            | pause_formatter_runnable
            | repair_runnable
      )

      