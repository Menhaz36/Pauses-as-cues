from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Utterance:
    """
    One utterance as it moves through the pipeline. Each stage fills in its own fields,
    so every intermediate result (ASR text, word timings, markers, LLM regions) stays
    available for inspection and evaluation.
    """
    audio_path: str
    audio: np.ndarray | None = field(default=None, repr=False)   # mono, 16 kHz

    # ASR transcription
    asr_text: str = ""

    # CTC alignment: list of {word, start, end, duration, pause_after, confidence};
    # None when alignment failed (the utterance then gets no pause markers)
    words: list[dict] | None = None

    # pause formatting
    formatted_text: str = ""
    has_pause: bool = False

    # LLM post-correction: one dict per region
    # {start, end, input_region, raw_llm_output, llm_error, accepted_output, validation_reason}
    regions: list[dict] = field(default_factory=list)

    # final transcript: the corrected text if the utterance had a pause, else the ASR text
    final_text: str = ""

    def to_dict(self) -> dict:
        """Everything except the audio samples (JSON-serialisable)."""
        d = asdict(self)
        d.pop("audio")
        return d
