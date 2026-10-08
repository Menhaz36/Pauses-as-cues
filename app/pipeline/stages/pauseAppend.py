import logging

logger = logging.getLogger(__name__)

def format_pause_aware_transcript(
    ctc_result: list[dict],
    pause_threshold: float = 0.7,
) -> dict:

    formatted_text = []

    for index, item in enumerate(ctc_result):

        # Silence before the first word
        if index == 0:
            start_pause = item.get("start", 0.0) or 0.0

            if start_pause >= pause_threshold:
                formatted_text.append(
                    f"[{start_pause:.1f}s pause]"
                )

        # Word
        formatted_text.append(item["word"])

        # Silence after the word
        pause_after = item.get("pause_after", 0.0) or 0.0

        if pause_after >= pause_threshold:
            formatted_text.append(
                f"[{pause_after:.1f}s pause]"
            )

    final_string = " ".join(formatted_text)

    logger.info(
        f"Formatted transcript: {final_string}"
    )

    return {"input": final_string}


def batch_format_pause_aware_transcript(
      batch_ctc_result: list[list[dict] | None],
      pause_threshold: float = 0.7,
      ) -> list[dict]:
    result=[]
    for ctc_result in batch_ctc_result:
      if ctc_result is None:
            result.append({"input": ""})
            continue
      formatted=format_pause_aware_transcript(ctc_result,pause_threshold)
      result.append(formatted)
    return result
    
