# Recording-grouped evaluation

This folder holds the held-out evaluation reported in Sect. 4 of the paper
(Tables 2 and 3). It has two stages, each run as a separate Kaggle notebook.

| Stage | What it does |
|---|---|
| `01_grouped_finetuning/` | Builds the recording-grouped split (seed 42) and fine-tunes Whisper-tiny on it. The selected model (step 800) is on Hugging Face as `Menhaz/whisper-tiny-torgo-grouped-iciss`, revision `958b6c10786e6cf8e6f1dae03ad1db5d407c3339`. |
| `02_heldout_test_evaluation/` | Pause localisation and pause-guided LLM correction on the held-out test takes (run `20261112T053554Z_7b6c322ff5`). |

## Start here

| You want | Open |
|---|---|
| The numbers in Table 2 | `02_heldout_test_evaluation/outputs/paper_values/paper_table2.csv` |
| The numbers in Table 3 | `02_heldout_test_evaluation/outputs/paper_values/paper_table3.csv` |
| Other numbers quoted in the text | `02_heldout_test_evaluation/outputs/paper_values/paper_text_values.csv` |
| Per-recording ASR predictions | `02_heldout_test_evaluation/outputs/tables/exact_asr_predictions.csv` |
| Per-recording LLM-correction predictions | `02_heldout_test_evaluation/outputs/tables/exact_correction_predictions.csv` |
| Which file supports which part of the evaluation | `02_heldout_test_evaluation/outputs/reviewer_evidence.json` |

Every row in the three `paper_values` files has a `source` column naming the
run table it was read from. They are produced by
`02_heldout_test_evaluation/notebook/paper_values_from_grouped_run.ipynb`,
which only reads the saved run tables.

## Prediction files

`exact_asr_predictions.csv` has one row per test take (1,815 takes: 624 dysarthric,
1,191 control). It gives the reference (`transcription`), the zero-shot output
(`zero_shot`) and the fine-tuned output (`asr`), each also in normalized form.

`exact_correction_predictions.csv` has one row per test take and condition
(1,815 × 4 = 7,260 rows). The `arm` column gives the condition:

- `untouched`: fine-tuned output, no editing
- `visible`: LLM correction with pause markers shown
- `hidden`: same windows, markers removed
- `random`: markers at random boundaries with an equal editing budget

`marked` shows whether the take received a pause marker, and `matched` whether
it entered the measured-vs-random comparison.

## Folder layout

```
recording_grouped_evaluation/
├── README.md
├── 01_grouped_finetuning/
│   ├── notebook/        fine-tuning notebook (with outputs)
│   └── outputs/         split, training and validation records
└── 02_heldout_test_evaluation/
    ├── notebook/        evaluation notebook and paper-values notebook
    └── outputs/         complete run output, unchanged
        ├── paper_values/
        ├── tables/      all result tables, including the two prediction files
        ├── checkpoints/ per-batch ASR, alignment and LLM job records (JSON)
        └── ...          split, threshold, prompt and response records
```

The run output in `02_heldout_test_evaluation/outputs/` is kept exactly as the
pipeline wrote it, so the paths in `reviewer_evidence.json` and the SHA-256
values in `file_hashes.json` stay valid. Its six split and training files
(`split_manifest.csv`, `split_audit.json`, `training_summary.json`,
`validation_history.csv`, `asr_validation_predictions.csv`,
`pause_calibration_predictions.csv`) are identical copies of
`01_grouped_finetuning/outputs/`; their SHA-256 values are recorded in
`run_config.json`.

TORGO audio and model weights are not included. TORGO is available from its
original distributors; the model is on Hugging Face (above).
