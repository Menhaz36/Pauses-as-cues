import logging
import asyncio

from app.config import SAMPLING_RATE
from app.pipeline.stages.audio import load_audio
from app.pipeline.utterance import Utterance


logger = logging.getLogger(__name__)

def transcribe_audio(asr_model, audio_file_path: str) -> Utterance:
      """
            Transcribes the given audio file using the provided ASR model.

            Args:
                  asr_model: The ASR model pipeline to use for transcription.
                  audio_file_path (str): Path to the audio file to be transcribed.

            Returns:
                  Utterance: holds the 16 kHz waveform and the transcribed text.
      """
      try:
            logger.info(f"Transcribing audio file: {audio_file_path}")

            utt = Utterance(audio_path=audio_file_path, audio=load_audio(audio_file_path))
            result = asr_model({"raw": utt.audio, "sampling_rate": SAMPLING_RATE})
            utt.asr_text = result.get("text", "")

            logger.info(f"Transcription result: {utt.asr_text}")

            return utt

      except Exception as e:
            logger.error(f"Error transcribing audio file {audio_file_path}: {e}")
            raise e


async def async_transcribe(asr_model,audio_path:str)->Utterance:
      return await asyncio.to_thread(transcribe_audio,asr_model,audio_path)




def batch_transcribe_audio(asr_model,
                            audio_file_paths: list[str],
                            batch_size: int = 4,
                            chunk_length_s:int=30) -> list[Utterance]:
      if not audio_file_paths:
            return []
      try:
            logger.info(f"Transcribing batch of {len(audio_file_paths)} audio files (batch_size={batch_size})...")

            utts=[Utterance(audio_path=path, audio=load_audio(path)) for path in audio_file_paths]
            results=asr_model(
                  [{"raw": u.audio, "sampling_rate": SAMPLING_RATE} for u in utts],
                  batch_size=batch_size,
                  chunk_length_s=chunk_length_s
            )
            for utt,result in zip(utts,results):
                  utt.asr_text=result.get("text","")
            return utts

      except Exception as e:
            logger.error(f"Error transcribing audio batch: {e}")
            raise e
