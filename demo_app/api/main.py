"""
REST API for the pause-guided dysarthric ASR pipeline.

    uvicorn api.main:app --port 8000

Endpoints
    GET  /health          models loaded, LLM availability, default τ
    POST /analyze         one audio file (+ optional reference): the three systems side by side
    POST /evaluate        several audio files + references: per-file results and corpus metrics
    GET  /paper-results   the evaluation of the paper's run (tables behind Tables 1–3 and Fig. 2)
"""
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402
from app.service import PipelineService, summarize  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "paper_run"
service = PipelineService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.load()     # ~30 s on first start (model download / cache load)
    yield


app = FastAPI(title="Pauses as Error Cues in Dysarthric ASR", lifespan=lifespan)


def _analyze_upload(upload: UploadFile, reference: str | None, tau: float, run_baseline: bool, run_llm: bool) -> dict:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        path = tmp.name
    try:
        return service.analyze(path, reference, tau, run_baseline, run_llm)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process {upload.filename}: {e}")
    finally:
        os.remove(path)


@app.get("/health")
def health():
    return service.status()


@app.post("/analyze")
def analyze(
    file: UploadFile = File(..., description="WAV/FLAC/OGG audio"),
    reference: str | None = Form(None, description="Reference transcript, for WER/CER/MER"),
    tau: float = Form(config.PAUSE_THRESHOLD, ge=0.02, le=3.0, description="Pause threshold in seconds (paper: 0.30)"),
    run_baseline: bool = Form(True),
    run_llm: bool = Form(True),
):
    return {"filename": file.filename, **_analyze_upload(file, reference, tau, run_baseline, run_llm)}


@app.post("/evaluate")
def evaluate(
    files: list[UploadFile] = File(...),
    references: str = Form("{}", description='JSON object {"filename": "reference transcript"}'),
    tau: float = Form(config.PAUSE_THRESHOLD, ge=0.02, le=3.0),
    run_baseline: bool = Form(True),
    run_llm: bool = Form(True),
):
    try:
        refs = json.loads(references or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"references must be a JSON object: {e}")
    results = [{"filename": f.filename, **_analyze_upload(f, refs.get(f.filename), tau, run_baseline, run_llm)}
               for f in files]
    return {"results": results, "summary": summarize(results)}


@app.get("/paper-results")
def paper_results():
    if not RESULTS_DIR.exists():
        raise HTTPException(status_code=404, detail="results/paper_run is missing: run scripts/export_paper_results.py")
    out = {}
    for p in sorted(RESULTS_DIR.iterdir()):
        if p.suffix == ".csv":
            out[p.stem] = json.loads(pd.read_csv(p).to_json(orient="records"))
        elif p.suffix == ".json":
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out
