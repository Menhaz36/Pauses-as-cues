# Pauses as Cues for ASR Errors in Dysarthric Speech

Code, notebooks and results for the paper

> M. Abdin, R. Sharma, B. Kundu. *Analysing whether Inter-word Pauses are Reliable Cues
> for Locating ASR Errors in Dysarthric Speech with Pause-Guided LLM Post-Correction.*

The study asks two questions on the TORGO corpus: do long inter-word pauses mark where
an ASR system makes errors, and can a large language model repair the errors near those
pauses? The pipeline is a fine-tuned Whisper-tiny recognizer, word timings from CTC
alignment with wav2vec 2.0, pause markers at gaps of at least 0.30 s, and gpt-oss-120b
correcting a short window around each marker.

## Repository layout

```
Pauses-as-cues/
├── demo_app/                          runnable pipeline, API and web interface
└── experiments/
    ├── exploratory_analysis/          corpus-wide analysis used to select the pause threshold
    └── recording_grouped_evaluation/  held-out evaluation reported in the paper
        ├── 01_grouped_finetuning/
        └── 02_heldout_test_evaluation/
```

### `demo_app/`

A working version of the pipeline that runs on single recordings or batches, with a
FastAPI service and a Streamlit interface. Setup and usage are described in
[`demo_app/README.md`](demo_app/README.md).

### `experiments/exploratory_analysis/`

The first, corpus-wide analysis on a file-level split of TORGO. It is used in the paper
only to choose the pause threshold (Table 1): the threshold of 0.30 s was selected by a
pre-specified rule and checked with leave-one-speaker-out validation. Because files of the
same recording can fall on both sides of a file-level split, these results are treated as
exploratory and are not the reported evaluation.

The folder contains the Kaggle notebooks (with their outputs), the saved intermediate
results, and the output tables and figures.

### `experiments/recording_grouped_evaluation/`

The evaluation reported in Sect. 4 of the paper (Tables 2 and 3). Recordings are grouped
so that no recording appears in more than one partition, and the test recordings are held
out from fine-tuning and checkpoint selection. It has two stages, each run as a separate
Kaggle notebook:

- `01_grouped_finetuning/` builds the recording-grouped split and fine-tunes Whisper-tiny.
- `02_heldout_test_evaluation/` tests the pause cue on the held-out recordings and runs the
  pause-guided LLM correction with its controls (markers hidden, and markers at random
  boundaries with the same editing budget).

The per-recording ASR and LLM-correction predictions, and the source of every number in
the paper, are in `02_heldout_test_evaluation/outputs/`. See
[`experiments/recording_grouped_evaluation/README.md`](experiments/recording_grouped_evaluation/README.md)
for where to find them.

Each experiment folder follows the same pattern: `notebook/` holds the Kaggle notebooks
and `outputs/` holds what they produced.

## Data and models

- **Corpus:** TORGO, loaded from the Hugging Face dataset `abnerh/TORGO-database`.
  No audio from the corpus is included in this repository.
- **Fine-tuned model (reported evaluation):** `Menhaz/whisper-tiny-torgo-grouped-iciss`
  on Hugging Face. Model weights are not stored here.
- **Aligner:** `facebook/wav2vec2-base-960h`.
- **LLM:** gpt-oss-120b through the Groq API.

## Reproducing the results

The notebooks were run on Kaggle. They read the Hugging Face and Groq credentials from
Kaggle Secrets (`HF_TOKEN`, `GROQ_API_KEY`); no keys are stored in this repository.
Run the notebooks in each stage in order. The saved outputs let the tables be checked
without re-running the models.
