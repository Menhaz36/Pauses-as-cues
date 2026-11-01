from langchain_core.runnables import RunnableLambda # type: ignore
from functools import partial
from app.pipeline.stages.transcription import async_transcribe,batch_transcribe_audio
from app.pipeline.stages.ctc_result import ctc_alignment_result, batch_ctc_alignment_result
from app.pipeline.stages.pauseAppend import pause_format_stage, batch_format_pause_aware_transcript
from app.pipeline.stages.correction import correct_utterance, batch_correct_utterances
from app.config import BATCH_SIZE, CHUNK_LENGTH_S



def full_pipeline_chain(asr_model,processor,ctc_model,llm):
      """
            Creates a chain of runnables for the full audio processing pipeline:
            ASR → CTC alignment → pause formatting (τ = 0.30 s) → pause-guided LLM correction.

            Args:
                  asr_model: The ASR model pipeline to use for transcription.
                  processor: Loaded Wav2Vec2Processor instance.
                  ctc_model: Loaded Wav2Vec2ForCTC instance.
                  llm: Chat model used for post-correction.
            return :
                  RunnableLambda: A runnable that maps an audio path to an Utterance.
      """

      transcribe_runnable = RunnableLambda(
                  partial(async_transcribe, asr_model)
            )

      ctc_aligner_runnable =RunnableLambda(
                  partial(ctc_alignment_result, processor=processor, ctc_model=ctc_model)
            )

      pause_formatter_runnable=RunnableLambda(pause_format_stage)

      correction_runnable=RunnableLambda(
                  partial(correct_utterance, llm=llm)
            )


      return (
            transcribe_runnable
            | ctc_aligner_runnable
            | pause_formatter_runnable
            | correction_runnable
      )

def batch_full_pipeline_chain(asr_model,processor,ctc_model,llm):
      """Same pipeline for a list of audio paths; returns a list of Utterances."""

      transcribe_runnable=RunnableLambda(
            partial(batch_transcribe_audio, asr_model, batch_size=BATCH_SIZE, chunk_length_s=CHUNK_LENGTH_S)
      )

      ctc_aligner_runnable=RunnableLambda(
            partial(batch_ctc_alignment_result,processor=processor, ctc_model=ctc_model)
      )

      pause_formatter_runnable=RunnableLambda(
            partial(batch_format_pause_aware_transcript)
      )

      correction_runnable=RunnableLambda(
            partial(batch_correct_utterances, llm=llm)
      )

      return (
            transcribe_runnable
            | ctc_aligner_runnable
            | pause_formatter_runnable
            | correction_runnable
      )
