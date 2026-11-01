import logging

import numpy as np
import soundfile as sf

from app.config import SAMPLING_RATE

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
      """
            Loads an audio file as a mono float32 waveform at `sampling_rate` (16 kHz).

            The same array is fed to Whisper and to the CTC aligner. The aligner converts frame
            indices to seconds assuming 16 kHz, so a file at another rate (or with two channels)
            would otherwise give wrong word timings and pause durations.
      """
      audio, sr = sf.read(audio_path, dtype="float32")
      if audio.ndim > 1:
            audio = audio.mean(axis=1)
      if sr != sampling_rate:
            import librosa  # local import: only needed for files that are not 16 kHz
            logger.debug(f"Resampling '{audio_path}' from {sr} Hz to {sampling_rate} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
      return audio
