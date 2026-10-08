import logging
import asyncio


logger = logging.getLogger(__name__)

def transcribe_audio(asr_model, audio_file_path: str) -> tuple[str, str]:
      """
            Transcribes the given audio file using the provided ASR model.

            Args:
                  asr_model: The ASR model pipeline to use for transcription.
                  audio_file_path (str): Path to the audio file to be transcribed.

            Returns:
                  tuple[str, str]: A tuple containing the transcribed text and the audio file path.
      """
      try:
            logger.info(f"Transcribing audio file: {audio_file_path}")

            result = asr_model(audio_file_path)
            transcription = result.get("text", "")

            logger.info(f"Transcription result: {transcription}")

            return (transcription,audio_file_path)
      
      except Exception as e:
            logger.error(f"Error transcribing audio file {audio_file_path}: {e}")
            raise e


async def async_transcribe(asr_model,audio_path:str)->tuple[str,str]:
      return await asyncio.to_thread(transcribe_audio,asr_model,audio_path)




def batch_transcribe_audio(asr_model,
                            audio_file_paths: list[str],
                            batch_size: int = 4,
                            chunk_length_s:int=30) -> list[tuple[str, str]]:
      if not audio_file_paths:
            return []
      try:
            logger.info(f"Transcribing batch of {len(audio_file_paths)} audio files (batch_size={batch_size})...")

            results=asr_model(
                  audio_file_paths,
                  batch_size=batch_size,
                  chunk_length_s=chunk_length_s
            )
            output=[]
            for path,result in zip(audio_file_paths,results):
                  transcription=result.get("text","")
                  output.append((transcription,path))
            return output
      
      except Exception as e:
            logger.error(f"Error transcribing audio batch: {e}")
            raise e


