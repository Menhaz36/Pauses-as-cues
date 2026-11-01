# Pauses as Error Cues in Dysarthric ASR

Code for *Analysing whether Inter-word Pauses are Reliable Cues for Locating ASR Errors in
Dysarthric Speech with Pause-Guided LLM Post-Correction* (Abdin, Sharma, Kundu).

Pipeline: audio → encoder-fine-tuned Whisper-tiny → CTC word alignment (wav2vec 2.0) →
pause markers at gaps ≥ τ = 0.30 s → gpt-oss-120b on ±4-word regions around the markers,
behind an edit-distance guard → post-corrected transcript.

## Layout

| Path | Contents |
|---|---|
| `app/` | Inference pipeline (one file or a batch of files) |
| `app/service.py` | Loads the models once; compares the three systems on a recording |
| `api/main.py` | FastAPI service |
| `ui/streamlit_app.py` | Streamlit front end (calls the API) |



## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in HF_TOKEN and GROQ_API_KEY
```

## Run the pipeline

```bash
python -m app.main path/to/audio.wav
python -m scratch.batch_pipeline_test      # batch of files
```

Each file returns an `Utterance` (`app/pipeline/utterance.py`) with every intermediate result:
ASR text, word timings, formatted text with markers, each LLM region with its raw answer and
guard decision, and `final_text`.

`final_text` is the corrected text for utterances with a pause and the unchanged ASR text for
all others. Corrected utterances are upper case, because regions are built from the CTC
words. Scoring normalizes case, so this does not affect WER.

## API and web UI

Start the API first; it loads the three models once (about 30 s), then the UI:

```bash
python -m uvicorn api.main:app --port 8000
python -m streamlit run ui/streamlit_app.py
```

The UI opens at http://localhost:8501 (set `API_URL` if the API runs elsewhere). Tabs:

* **Try it**: upload or record a sentence, optionally with its reference. Shows the transcripts of
  zero-shot Whisper-tiny, the fine-tuned model and the pause-guided LLM side by side, WER/CER/MER
  with a word diff against the reference, a word timeline with the pauses ≥ τ, and every LLM region
  with the guard's decision.
* **Evaluate a set**: several recordings plus references (typed in the table or uploaded as a CSV
  `filename,reference`). Corpus WER/CER/MER per system, how many recordings the LLM improved or
  worsened, and a sign test.
* **Paper results**: the paper's evaluation. Includes Table 2 by speaker group and by speaker,
  threshold selection with its bootstrap CI and the LOSO check (Table 1, Fig. 2), pause markers vs
  random positions, the gap distribution, and the LLM outcome per region and per speaker (Table 3).

The τ slider in the sidebar lets you try other thresholds. Results at τ ≠ 0.30 s are exploratory.

API endpoints (interactive docs at http://localhost:8000/docs):

## Notes

* The pause marker is the single token `[0.3s_pause]`. Region extraction splits on
  whitespace, so the marker must not contain a space.
* 17 of the 16,552 utterances fail CTC alignment (degenerate hypotheses such as
  `k-k-k-...`). They get no pause markers and keep their ASR text.
* Batched CTC with zero-padding changes the wav2vec2-base outputs, which is why utterances
  are aligned one at a time (`CTC_PADDED_BATCH = False`).

