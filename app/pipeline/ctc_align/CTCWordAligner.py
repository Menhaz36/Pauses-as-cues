import re
import torch
import ctc_segmentation
import logging
logger = logging.getLogger(__name__)

class CTCWordAligner:

    def __init__(
        self,
        processor,
        ctc_model,
        sample_rate=16000,
    ):
        self.processor = processor
        self.ctc_model = ctc_model
        self.sample_rate = sample_rate

        self.device = next(ctc_model.parameters()).device

        self.ctc_model.eval()

        self.char_list, self.blank_idx = self._build_char_list()


    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    def align(
        self,
        whisper_result,
        audio,
        batch=False,
    ):
        """
        Perform CTC word alignment.

        Single:
            align(text, audio_array)

        Batch:
            align(
                [text1, text2],
                [audio1, audio2],
                batch=True
            )
        """

        if batch:
            return self._align_batch(
                whisper_result,
                audio,
            )

        return self._align_single(
            whisper_result,
            audio,
        )


    # =========================================================
    # SINGLE ALIGNMENT
    # =========================================================

    def _align_single(
        self,
        text,
        audio_array,
    ):

        # 1. normalize text
        words = self._normalize_text(text)

        # 2. compute CTC log probabilities
        log_probs = self._compute_log_probs(
            audio_array
        )

        # 3. perform alignment
        return self._segment_one(
            audio_array,
            words,
            log_probs,
        )


    # =========================================================
    # BATCH ALIGNMENT
    # =========================================================

    def _align_batch(
        self,
        texts,
        audio_arrays,
    ):

        if len(texts) != len(audio_arrays):
            raise ValueError(
                "Number of texts and audio arrays must match."
            )

        # 1. normalize all texts
        normalized_texts = [
            self._normalize_text(text)
            for text in texts
        ]

        # 2. GPU/model forward pass is batched
        log_probs_list = self._compute_log_probs_batch(
            audio_arrays
        )

        # 3. CTC segmentation is still done individually
        results = []

        for audio_array, words, log_probs in zip(
            audio_arrays,
            normalized_texts,
            log_probs_list,
        ):
            try:
                result = self._segment_one(
                    audio_array,
                    words,
                    log_probs,
                )

            except (AssertionError, IndexError) as e:
                logger.warning(
                    "CTC alignment failed [%s]: %s | "
                    "audio=%.2fs | text=%s",
                    type(e).__name__,
                    e,
                    len(audio_array) / self.sample_rate,
                    " ".join(words),
                )
                result = None

            results.append(result)

        return results


    # =========================================================
    # SEGMENT ONE AUDIO
    # =========================================================

    def _segment_one(
        self,
        audio_array,
        words,
        log_probs,
    ):

        config = self._build_ctc_config(
            audio_len_samples=len(audio_array),
            lpz_len=log_probs.shape[0],
        )

        segments = self._run_ctc_alignment(
            words,
            log_probs,
            config,
        )

        return self._format_output(
            words,
            segments,
        )


    # =========================================================
    # CTC SEGMENTATION
    # =========================================================

    def _run_ctc_alignment(
        self,
        words,
        log_probs,
        config,
    ):

        ground_truth_mat, utt_begin_indices = (
            ctc_segmentation.prepare_text(
                config,
                words,
            )
        )

        timings, char_probs, _ = (
            ctc_segmentation.ctc_segmentation(
                config,
                log_probs,
                ground_truth_mat,
            )
        )

        segments = (
            ctc_segmentation.determine_utterance_segments(
                config=config,
                utt_begin_indices=utt_begin_indices,
                char_probs=char_probs,
                timings=timings,
                text=words,
            )
        )

        return segments


    # =========================================================
    # SINGLE LOG PROBS
    # =========================================================

    def _compute_log_probs(
        self,
        audio_array,
    ):

        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        input_values = inputs.input_values.to(
            self.device
        )

        with torch.inference_mode():

            logits = self.ctc_model(
                input_values
            ).logits

        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=-1,
        )

        return (
            log_probs
            .squeeze(0)
            .cpu()
            .numpy()
        )


    # =========================================================
    # BATCH LOG PROBS
    # =========================================================

    def _compute_log_probs_batch(
        self,
        audio_arrays,
    ):

        inputs = self.processor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_values = inputs.input_values.to(
            self.device
        )

        attention_mask = inputs.attention_mask.to(
            self.device
        )

        with torch.inference_mode():

            logits = self.ctc_model(
                input_values,
                attention_mask=attention_mask,
            ).logits

        # [batch, frames, vocab]
        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=-1,
        )

        # actual audio lengths before padding
        input_lengths = attention_mask.sum(
            dim=1
        )

        # actual Wav2Vec2 output frame lengths
        output_lengths = self._get_output_lengths(
            input_lengths
        )

        log_probs_list = []

        for i, output_length in enumerate(
            output_lengths
        ):

            frame_length = int(
                output_length.item()
            )

            # remove padded CTC frames
            sample_log_probs = log_probs[
                i,
                :frame_length,
                :
            ]

            log_probs_list.append(
                sample_log_probs
                .cpu()
                .numpy()
            )

        return log_probs_list


    # =========================================================
    # WAV2VEC OUTPUT LENGTH
    # =========================================================

    def _get_output_lengths(
        self,
        input_lengths,
    ):

        if hasattr(
            self.ctc_model,
            "_get_feat_extract_output_lengths",
        ):
            return (
                self.ctc_model
                ._get_feat_extract_output_lengths(
                    input_lengths
                )
            )

        # fallback
        output_lengths = input_lengths.clone()

        for kernel_size, stride in zip(
            self.ctc_model.config.conv_kernel,
            self.ctc_model.config.conv_stride,
        ):
            output_lengths = (
                (output_lengths - kernel_size)
                // stride
            ) + 1

        return output_lengths


    # =========================================================
    # CONFIG
    # =========================================================

    def _build_ctc_config(
        self,
        audio_len_samples,
        lpz_len,
    ):

        config = (
            ctc_segmentation
            .CtcSegmentationParameters(
                char_list=self.char_list,
                blank=self.blank_idx,
            )
        )

        config.index_duration = (
            audio_len_samples
            / lpz_len
            / self.sample_rate
        )

        return config


    # =========================================================
    # TOKENIZER VOCAB
    # =========================================================

    def _build_char_list(self):

        vocab = [
            self.processor.tokenizer.convert_ids_to_tokens(i)
            for i in range(
                len(self.processor.tokenizer)
            )
        ]

        word_delim = (
            self.processor
            .tokenizer
            .word_delimiter_token
        )

        char_list = [
            " " if token == word_delim
            else token.lower()
            if token.startswith("<")
            else token
            for token in vocab
        ]

        if "<pad>" not in char_list:
            raise ValueError(
                "Expected '<pad>' token in vocabulary."
            )

        blank_idx = char_list.index("<pad>")

        return char_list, blank_idx


    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(text):

        text = text.upper()

        text = re.sub(
            r"[^\w\s']",
            "",
            text,
        )
        text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)

        return text.split()


    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    @staticmethod
    def _format_output(
        words,
        segments,
    ):

        results = []

        for i, (
            word,
            (start, end, score),
        ) in enumerate(
            zip(words, segments)
        ):

            pause_after = None

            if i < len(segments) - 1:

                pause_after = max(
                    0.0,
                    round(
                        segments[i + 1][0] - end,
                        3,
                    ),
                )

            results.append({
                "word": word,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(
                    end - start,
                    3,
                ),
                "pause_after": pause_after,
                "confidence": round(
                    float(score),
                    3,
                ),
            })

        return results

















#purana code


























# import ctc_segmentation # type: ignore
# import torch # type: ignore
# import re
# class CTCWordAligner:
#     def __init__(self, processor, ctc_model, sample_rate=16000):
#         self.processor = processor
#         self.ctc_model = ctc_model
#         self.sample_rate = sample_rate
#         self.device = next(ctc_model.parameters()).device
#         self.char_list, self.blank_idx = self._build_char_list()

#     # def _build_char_list(self):
#     #     vocab = [
#     #         self.processor.tokenizer.convert_ids_to_tokens(i)
#     #         for i in range(len(self.processor.tokenizer))
#     #     ]
#     #     char_list = [
#     #         token.lower() if token.startswith("<") else token
#     #         for token in vocab
#     #     ]
#     #     if "<pad>" not in char_list:
#     #         raise ValueError(
#     #             f"Expected '<pad>' token in vocab for blank index, "
#     #             f"got vocab sample: {char_list[:10]}"
#     #         )
#     #     blank_idx = char_list.index("<pad>")
#     #     return char_list, blank_idx

#     def _build_char_list(self):
#         vocab = [
#             self.processor.tokenizer.convert_ids_to_tokens(i)
#             for i in range(len(self.processor.tokenizer))
#         ]
#         word_delim = self.processor.tokenizer.word_delimiter_token  # usually "|"
#         char_list = [
#             " " if token == word_delim
#             else token.lower() if token.startswith("<")
#             else token
#             for token in vocab
#         ]
#         if "<pad>" not in char_list:
#             raise ValueError(...)
#         blank_idx = char_list.index("<pad>")
#         return char_list, blank_idx

#     def _build_ctc_config(self, audio_len_samples, lpz_len):
#         config = ctc_segmentation.CtcSegmentationParameters(
#             char_list=self.char_list,
#             blank=self.blank_idx,
#         )
#         # Critical: derive the real per-frame duration from this specific
#         # forward pass, rather than relying on the library's default
#         # (which assumes a hybrid-ASR frame stride, not wav2vec2's).
#         config.index_duration = (audio_len_samples / lpz_len) / self.sample_rate
#         return config

#     def _normalize_text(self, text):
#         text = text.upper()
#         text = re.sub(r"[^\w\s']", "", text)# remove punctuation except apostrophes
#         return text.split()

#     def _compute_log_probs(self, audio_array):
#       inputs = self.processor(
#           audio_array,
#           sampling_rate=self.sample_rate,
#           return_tensors="pt",
#       )
#       input_values = inputs.input_values.to(self.device)
#       with torch.no_grad():
#           logits = self.ctc_model(input_values).logits
#       log_probs = (
#           torch.nn.functional.log_softmax(logits, dim=-1)
#           .squeeze(0)
#           .cpu()
#           .numpy()
#       )
#       return log_probs

#     def _align(self, words, log_probs, config):
#         ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
#             config,
#             words,
#         )
#         timing, char_probs, _ = ctc_segmentation.ctc_segmentation(
#             config,
#             log_probs,
#             ground_truth_mat,
#         )
#         return ctc_segmentation.determine_utterance_segments(
#             config=config,
#             utt_begin_indices=utt_begin_indices,
#             char_probs=char_probs,
#             timings=timing,
#             text=words,
#         )

#     def _format_output(self, words, segments):
#         results = []
#         for i, (word, (start, end, score)) in enumerate(zip(words, segments)):
#             pause_after = None
#             if i < len(segments) - 1:
#                 pause_after = max(
#                     0.0,
#                     round(segments[i + 1][0] - end, 3),
#                 )
#             results.append({
#                 "word": word,
#                 "start": round(start, 3),
#                 "end": round(end, 3),
#                 "duration": round(end - start, 3),
#                 "pause_after": pause_after,
#                 "confidence": round(float(score), 3),
#             })
#         return results

#     def align(self, text, audio_array):
#         words = self._normalize_text(text)
#         if not words:
#             return []

#         log_probs = self._compute_log_probs(audio_array)

#         # Build config per-utterance: index_duration depends on this
#         # specific audio length vs. this specific log_probs length,
#         # which can vary slightly with padding/framing.
#         config = self._build_ctc_config(
#             audio_len_samples=len(audio_array),
#             lpz_len=log_probs.shape[0],
#         )
#         print("delimiter token:", repr(self.processor.tokenizer.word_delimiter_token))
#         print("space in char_list:", " " in self.char_list)
#         print("blank_idx:", self.blank_idx, self.char_list[self.blank_idx])
#         print("log_probs shape:", log_probs.shape, "vocab size:", len(self.char_list))
#         print("index_duration:", config.index_duration)
#         print("words:", words[:10])
#         print("argmax sample:", log_probs.argmax(-1)[:50])

#         segments = self._align(words, log_probs, config)
#         return self._format_output(words, segments)














'''

Connectionist Temporal Classification (CTC) was used to align words from
the transcription to their utterance timestamps. Before alignment, the transcript was normalized
by converting it to uppercase, removing punctuation while preserving valid intra-word apostro-
phes, and splitting the sentence into individual words.
The operation was performed using the  ctc-segmentation toolkit and "facebook/wav2vec2-base-960h" model

First,the Wav2Vec2 processor coverted the audio waveform into the normalized input format expected by wave2vec2 model 
respresented as Pytorch Tensors using a sampling rate of 16Khz

The input was then processed by the wave2vec2 ctc model extract accoustic information and produce logits, where each timestamp contains a score for everypossible vocabulary.
The vocabulary contains 32 tokens in which one was blank token. The log softmax operation converted the
scores into frame wise probabilities

The frame-wise log-probabilities obtained from the Wav2Vec2 CTC model were aligned with the corresponding 
transcript using the ctc-segmentation toolkit.
The toolkit performed dynamic based ctc allignment between the predicted transcript and frame wise log probability
The algorithm searched for the most probable valid ctc alignment path across the acoustic frames while accounting for
 repeated tokens and CTC blank symbols
 The resulting token-level alignments were converted into word-level start and end timestamps, from which the pause 
 duration between consecutive words was calculated.





'''