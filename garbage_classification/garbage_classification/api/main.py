"""
api/main.py
-----------
FastAPI backend for the Garbage Classification service.

Endpoints
---------
GET  /health              -> uptime / liveness check (used by the UI + Locust)
GET  /classes              -> list of class labels
POST /predict              -> single-image prediction (multipart/form-data file)
POST /upload                -> bulk upload of images for retraining, labeled by class
POST /retrain                -> triggers the retraining pipeline on uploaded data
GET  /insights              -> dataset statistics used by the UI's visualizations
GET  /model-info             -> which model version is currently deployed

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
import json
import io
import time
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import CLASS_NAMES
from src.prediction import predict_image, load_model
from src.retrain import full_retrain_pipeline
from src import database as db

APP_START_TIME = time.time()
MODEL_PATH = Path("models/garbage_model.keras")
UPLOADS_DIR = Path("data/uploads")
PREPROCESSED_DIR = Path("data/preprocessed")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

for cls in CLASS_NAMES:
    (UPLOADS_DIR / cls).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Garbage Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retraining status is tracked in-memory; a real deployment would persist
# this (DB row / Redis key) so it survives a process restart.
RETRAIN_STATUS = {"state": "idle", "last_result": None}


@app.get("/health")
def health():
    """Uptime endpoint -- the UI polls this to show 'model up-time'."""
    uptime_seconds = time.time() - APP_START_TIME
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "ok",
        "uptime_seconds": round(uptime_seconds, 1),
        "model_file_present": model_loaded,
    }


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.get("/model-info")
def model_info():
    if not MODEL_PATH.exists():
        raise HTTPException(404, "No trained model deployed yet.")
    stat = MODEL_PATH.stat()
    return {
        "model_path": str(MODEL_PATH),
        "size_mb": round(stat.st_size / 1e6, 2),
        "last_modified": time.ctime(stat.st_mtime),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not yet trained/deployed.")

    print("Step 1: Reading uploaded file")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("Step 2: Image loaded")
    except Exception:
        raise HTTPException(400, "Uploaded file is not a valid image.")

    image_array = np.array(image)
    print("Step 3: Starting prediction")

    result = predict_image(str(MODEL_PATH), image_array)

    print("Step 4: Prediction finished")

    return result


@app.post("/upload")
async def upload_bulk(
    label: str = Form(..., description="Class label for these images, e.g. 'plastic'"),
    files: List[UploadFile] = File(...),
):
    """Bulk upload of images (used for retraining). Files are saved under
    data/uploads/<label>/ AND logged as rows in data/uploads.db (SQLite) --
    see src/database.py."""
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")

    dest_dir = UPLOADS_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest_path = dest_dir / f.filename
        with dest_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        # Explicit "save to database" step: every uploaded file is logged as
        # a row (filename, label, path, timestamp) so retraining can later
        # query exactly which uploads are pending / which fed which model version.
        db.log_upload(filename=f.filename, label=label, file_path=str(dest_path))
        saved.append(f.filename)

    return {"label": label, "saved_count": len(saved), "filenames": saved}


@app.get("/uploads/log")
def uploads_log():
    """Full audit trail of every uploaded retraining image: when it was
    uploaded, whether it's been preprocessed, and which retrain run (if any)
    consumed it. Backed by data/uploads.db (SQLite)."""
    return {"uploads": db.get_all_uploads()}


def _run_retrain_job():
    try:
        RETRAIN_STATUS["state"] = "running"
        result = full_retrain_pipeline(
            uploads_dir=str(UPLOADS_DIR),
            preprocessed_dir=str(PREPROCESSED_DIR),
            train_dir=str(TRAIN_DIR),
            test_dir=str(TEST_DIR),
            model_path=str(MODEL_PATH),
        )
        RETRAIN_STATUS["state"] = "completed"
        RETRAIN_STATUS["last_result"] = result
    except Exception as e:
        RETRAIN_STATUS["state"] = "failed"
        RETRAIN_STATUS["last_result"] = {"error": str(e)}


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Kicks off retraining as a background task so the HTTP request returns
    immediately; the UI polls /retrain/status for progress. Pending work is
    determined from the database (rows not yet tied to a retrain run), which
    is the authoritative record of what's been uploaded."""
    pending = db.get_pending_uploads()
    if len(pending) == 0:
        raise HTTPException(400, "No newly uploaded images to retrain on yet.")

    if RETRAIN_STATUS["state"] == "running":
        raise HTTPException(409, "A retraining job is already running.")

    background_tasks.add_task(_run_retrain_job)
    RETRAIN_STATUS["state"] = "queued"
    return {"message": "Retraining started in the background.", "pending_images": len(pending)}


@app.get("/retrain/status")
def retrain_status():
    return RETRAIN_STATUS

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Garbage Classification API"
    }


@app.get("/insights")
def insights():
    """Dataset statistics for UI visualization."""

    stats_file = Path("data/dataset_stats.json")

    # If statistics file exists (production / Render)
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)

        return {
            "train_counts": stats.get("train_counts", {}),
            "test_counts": stats.get("test_counts", {}),
            "pending_upload_counts": {
                cls: len([
                    f for f in (UPLOADS_DIR / cls).glob("*")
                    if f.suffix.lower() in IMAGE_EXTS
                ])
                for cls in CLASS_NAMES
            },
            "preprocessed_counts": {},
            "pending_uploads_in_db": len(db.get_pending_uploads()),
        }


    # Local fallback (when folders exist)
    def count_images(base_dir):
        base = Path(base_dir)

        return {
            cls: len([
                f for f in (base / cls).glob("*")
                if f.suffix.lower() in IMAGE_EXTS
            ])
            for cls in CLASS_NAMES
            if (base / cls).exists()
        }


    return {
        "train_counts": count_images(TRAIN_DIR),
        "test_counts": count_images(TEST_DIR),
        "pending_upload_counts": count_images(UPLOADS_DIR),
        "preprocessed_counts": count_images(PREPROCESSED_DIR),
        "pending_uploads_in_db": len(db.get_pending_uploads()),
    }
