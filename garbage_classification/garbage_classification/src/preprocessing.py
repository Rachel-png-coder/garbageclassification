"""
preprocessing.py
----------------
Data acquisition + preprocessing utilities for the Garbage Classification project.

Responsibilities:
1. Split the raw Kaggle dataset (one folder per class) into train/ and test/
   while preserving class subfolders.
2. Build tf.data pipelines (with augmentation) for training and evaluation.
3. Provide a single-image preprocessing function reused by both the notebook
   and the FastAPI prediction endpoint (so train-time and inference-time
   preprocessing can NEVER drift apart).
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = (160, 160)          # MobileNetV2 works natively at 96/128/160/192/224
BATCH_SIZE = 32
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Splitting the raw Kaggle folder structure into data/train and data/test
# --------------------------------------------------------------------------- #
def split_dataset(raw_dir: str, out_dir: str, test_size: float = 0.15, seed: int = SEED):
    """
    raw_dir: path to the folder that directly contains class subfolders
             e.g. Garbage_Dataset_Classification/images/{cardboard,glass,...}
    out_dir: destination root that will contain out_dir/train/<class>/*.jpg
             and out_dir/test/<class>/*.jpg
    """
    random.seed(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    for split in ["train", "test"]:
        for cls in CLASS_NAMES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        files = sorted([f for f in cls_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        random.shuffle(files)

        n_test = int(len(files) * test_size)
        test_files = files[:n_test]
        train_files = files[n_test:]

        for f in train_files:
            shutil.copy(f, out_dir / "train" / cls / f.name)
        for f in test_files:
            shutil.copy(f, out_dir / "test" / cls / f.name)

        summary[cls] = {"train": len(train_files), "test": len(test_files)}

    return summary


# --------------------------------------------------------------------------- #
# 2. tf.data pipelines
# --------------------------------------------------------------------------- #
def _augment_layer():
    """Augmentation applied ONLY at train time."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def get_datasets(train_dir: str, test_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Returns (train_ds, val_ds, test_ds, class_names) as tf.data.Dataset objects.
    train_dir is further split 85/15 into train/val.
    Pixel values are NOT rescaled here -- MobileNetV2's own preprocess_input
    (applied inside the model, see model.py) expects [-1, 1] scaled inputs and
    handles that internally, so we only resize here.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        class_names=CLASS_NAMES,
    )

    class_names = train_ds.class_names
    augment = _augment_layer()

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                             num_parallel_calls=tf.data.AUTOTUNE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# --------------------------------------------------------------------------- #
# 3. Single-image preprocessing (shared by notebook + API)
# --------------------------------------------------------------------------- #
def preprocess_image(image_path_or_array, img_size=IMG_SIZE):
    """
    Accepts either a filepath (str) or an already-loaded numpy/PIL array and
    returns a (1, H, W, 3) float32 batch ready to feed model.predict().
    Resizing only -- rescaling to [-1,1] happens inside the model via the
    MobileNetV2 preprocess_input Lambda layer (see model.py) so this function
    stays identical whether called at train time or inference time.
    """
    if isinstance(image_path_or_array, (str, Path)):
        img = tf.keras.utils.load_img(image_path_or_array, target_size=img_size)
        arr = tf.keras.utils.img_to_array(img)
    else:
        arr = tf.image.resize(np.asarray(image_path_or_array), img_size).numpy()

    arr = np.expand_dims(arr, axis=0)
    return arr.astype("float32")


def preprocess_uploaded_image(src_path, out_dir, img_size=IMG_SIZE):
    """
    Explicit preprocessing step applied to a single newly-uploaded image
    BEFORE it's allowed anywhere near the training set:
      1. Validate it's actually a readable, non-corrupt image
      2. Convert to RGB (drop alpha / handle grayscale uploads)
      3. Resize to the model's input size
      4. Save the cleaned copy to out_dir, preserving the filename

    Returns the path to the preprocessed copy, or None if the file was
    invalid and got skipped (so a bad upload never corrupts a retrain run).
    """
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = tf.keras.utils.load_img(src_path)  # raises if unreadable/corrupt
    except Exception:
        return None

    img = img.convert("RGB")
    img = img.resize(img_size)

    out_path = out_dir / src_path.name
    img.save(out_path, format="JPEG", quality=95)
    return out_path


def preprocess_uploaded_batch(uploads_dir: str, preprocessed_dir: str, class_names=CLASS_NAMES,
                               img_size=IMG_SIZE):
    """
    Runs preprocess_uploaded_image() over every raw file currently sitting
    in data/uploads/<class>/, writing cleaned copies to
    data/preprocessed/<class>/. Returns {class: [preprocessed_paths]}.
    Used as an explicit, loggable step by src/retrain.py before any
    retraining happens.
    """
    uploads_dir = Path(uploads_dir)
    preprocessed_dir = Path(preprocessed_dir)
    results = {}

    for cls in class_names:
        src_dir = uploads_dir / cls
        if not src_dir.exists():
            continue
        raw_files = [f for f in src_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

        cleaned_paths = []
        for f in raw_files:
            out_path = preprocess_uploaded_image(f, preprocessed_dir / cls, img_size)
            if out_path is not None:
                cleaned_paths.append(out_path)
        results[cls] = cleaned_paths

    return results


if __name__ == "__main__":
    # Example usage when run standalone:
    #   python src/preprocessing.py
    RAW = "../Garbage_Dataset_Classification/images"
    OUT = "../data"
    if os.path.exists(RAW):
        s = split_dataset(RAW, OUT)
        print(s)
    else:
        print(f"Raw dataset not found at {RAW}. Update the path and re-run.")
