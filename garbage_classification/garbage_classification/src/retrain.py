"""
retrain.py
----------
Retraining trigger used by the API's /retrain endpoint and by the
"Retrain" button in the Streamlit UI.

Flow (each step is explicit and independently inspectable -- important for
demonstrating the retraining process clearly):
1. Uploaded images are saved as files under data/uploads/<class>/ AND
   logged as rows in the SQLite database (src/database.py) by the API's
   /upload endpoint.
2. preprocess_uploaded_batch() (src/preprocessing.py) validates, converts,
   and resizes every pending upload, writing cleaned copies to
   data/preprocessed/<class>/. Each cleaned file is marked
   `preprocessed = 1` in the database.
3. merge_preprocessed_into_train() moves the cleaned copies into
   data/train/<class>/ and marks those DB rows with the retrain run's
   timestamp (used_in_retrain_run), so we always know which DB-logged
   upload contributed to which model version.
4. retrain_model() rebuilds tf.data pipelines including the new images and
   fine-tunes the model, WARM-STARTING from the existing saved
   garbage_model.h5 (the custom pretrained model) rather than from
   ImageNet scratch -- faster, and mirrors real incremental retraining.
5. The new model overwrites models/garbage_model.h5 and the prediction
   module's cache is cleared so the API immediately serves the new weights.
"""

import shutil
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from .preprocessing import get_datasets, preprocess_uploaded_batch, CLASS_NAMES
from .model import compile_model, get_callbacks
from .prediction import clear_model_cache, CUSTOM_OBJECTS
from . import database as db


def merge_preprocessed_into_train(preprocessed_by_class: dict, uploads_dir: str,
                                   train_dir: str, run_timestamp: str):
    """Moves already-preprocessed images into the training set and marks
    their database rows (keyed by the ORIGINAL upload path) as consumed
    by this retrain run."""
    train_dir = Path(train_dir)
    moved = {}
    original_paths_used = []

    for cls, paths in preprocessed_by_class.items():
        dst_dir = train_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for p in paths:
            dest = dst_dir / p.name
            shutil.move(str(p), str(dest))
            # DB rows were logged against the original data/uploads/<class>/<file> path
            original_paths_used.append(str(Path(uploads_dir) / cls / p.name))
            moved_count += 1
        moved[cls] = moved_count

    db.mark_used_in_retrain(original_paths_used, run_timestamp)
    return moved


def retrain_model(train_dir: str, test_dir: str, model_path: str,
                   epochs: int = 5, log_dir: str = "models/retrain_logs"):
    """
    Warm-starts from the currently deployed CUSTOM model (garbage_model.h5,
    itself already pretrained via transfer learning from ImageNet) and
    fine-tunes it briefly on the now-larger training set. Returns a dict
    with before/after metrics on the held-out test set.
    """
    model_path = Path(model_path)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = get_datasets(train_dir, test_dir)

    # Warm start: load OUR previously trained model, not a fresh ImageNet base.
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    compile_model(model, lr=1e-5)  # small LR: fine-tuning an already-trained model, not training from scratch

    before = model.evaluate(test_ds, verbose=0, return_dict=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = str(model_path)  # overwrite in place; version history kept in log_dir
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(checkpoint_path),
    )

    after = model.evaluate(test_ds, verbose=0, return_dict=True)

    # Keep a timestamped copy for audit / rollback.
    versioned_copy = Path(log_dir) / f"model_{timestamp}.h5"
    model.save(versioned_copy)

    clear_model_cache()  # so the API serves the freshly retrained weights immediately

    return {
        "timestamp": timestamp,
        "epochs_run": len(history.history["loss"]),
        "metrics_before": before,
        "metrics_after": after,
        "versioned_model_path": str(versioned_copy),
    }


def full_retrain_pipeline(uploads_dir="data/uploads", preprocessed_dir="data/preprocessed",
                           train_dir="data/train", test_dir="data/test",
                           model_path="models/garbage_model.h5", epochs=5):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1+2: explicit preprocessing of pending uploads (validate/convert/resize)
    preprocessed = preprocess_uploaded_batch(uploads_dir, preprocessed_dir)
    for cls, paths in preprocessed.items():
        for p in paths:
            original_upload_path = Path(uploads_dir) / cls / p.name
            db.mark_preprocessed(str(original_upload_path))

    # Step 3: merge cleaned images into the training set, log which DB rows were used
    moved = merge_preprocessed_into_train(preprocessed, uploads_dir, train_dir, run_timestamp)

    # Step 4-5: retrain (warm-started from the custom model) and redeploy
    result = retrain_model(train_dir, test_dir, model_path, epochs=epochs)
    result["new_images_merged"] = moved
    result["run_timestamp"] = run_timestamp
    return result
