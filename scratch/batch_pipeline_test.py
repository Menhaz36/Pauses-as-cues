from app.pipeline.stages.transcription import batch_transcribe_audio
from app.pipeline.stages.ctc_result import batch_ctc_alignment_result
from app.models.llm import get_llm
from app.models.whisper import asr_model, get_finetuned_asr_model
from app.models.wav2vec import get_ctcmodel, get_processor
from app.pipeline.stages.pauseAppend import batch_format_pause_aware_transcript
import logging
logger = logging.getLogger(__name__)
import time
from app.pipeline.chain import batch_full_pipeline_chain


logger.info("loading asr models for batch processing")
start=time.time()
asr_model=get_finetuned_asr_model()
logger.info("asr model loaded in %f seconds", time.time()-start)

logger.info("loading wav2vec2 processor and ctc model for batch processing")
start=time.time()
processor=get_processor()
logger.info("wav2vec2 processor loaded in %f seconds", time.time()-start)

logger.info("loading wav2vec2 ctc model for batch processing")
start=time.time()
ctc_model=get_ctcmodel()
logger.info("wav2vec2 ctc model loaded in %f seconds", time.time()-start)

logger.info("loading llm for batch processing")
start=time.time()
llm=get_llm()
logger.info("llm loaded in %f seconds", time.time()-start)

audio_files = [
    r"D:\D\IMPORTANT\Dysarthria\ASR\Sample_dysarthria_audio.wav",
    r"D:\D\IMPORTANT\Dysarthria\ASR\wav_headMic_F03S01_0028.wav",
    r"D:\D\IMPORTANT\Dysarthria\ASR\wav_headMic_F04S02_0053.wav",
    r"D:\D\IMPORTANT\Dysarthria\ASR\wav_headMic_M03S02_0421.wav"
]


# logger.info("pipeline: trancription-->ctc_alignment-->pause_aware_formatting-->final_prompt-->llm-->StrOutputParser")
# logger.info("transcribing audio ")
# start=time.time()
# result=batch_transcribe_audio(asr_model,audio_files)
# logger.info("transcription completed in %f seconds", time.time()-start)
# logger.info("transcription result: %s", result)
# logger.info("Type of result: %s", type(result))

# logger.info("running CTC alignment on transcribed audio")
# start=time.time()
# result_ctc=batch_ctc_alignment_result(result,processor,ctc_model)
# logger.info("CTC alignment completed in %f seconds", time.time()-start)
# logger.info("CTC alignment result: %s", result_ctc)
# logger.info("Type of CTC alignment result: %s", type(result_ctc))

# logger.info("running pause-aware formatting on CTC alignment result")
# start=time.time()
# result_format=batch_format_pause_aware_transcript(result_ctc)
# logger.info("Pause-aware formatting completed in %f seconds", time.time()-start)
# logger.info("Formatted transcript: %s", result_format)



speech_repair_chain=batch_full_pipeline_chain(asr_model=asr_model,processor=processor,ctc_model=ctc_model,llm=llm)

start=time.time()
logger.info("running full pipeline on batch of audio files")
result=speech_repair_chain.invoke(audio_files)
logger.info("Full pipeline completed in %f seconds", time.time()-start)
logger.info("Final Repaired Transcript: %s", result)
logger.info("Type of Final Repaired Transcript: %s", type(result))
