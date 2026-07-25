"""
prediction.py
-------------
Single-datapoint inference used by both the notebook (for sanity checks)
and the FastAPI /predict endpoint.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .preprocessing import preprocess_image, CLASS_NAMES, IMG_SIZE

_MODEL_CACHE = {}

# The model's first layer is a Lambda wrapping MobileNetV2's preprocess_input.
# Keras 3's safe deserializer refuses to resolve arbitrary functions by name
# unless they're passed explicitly here (or registered with
# @keras.saving.register_keras_serializable) -- without this, load_model()
# raises "Could not locate function 'preprocess_input'".
CUSTOM_OBJECTS = {"preprocess_input": preprocess_input}


def load_model(model_path: str):
    """Cached model loader -- avoids reloading the .h5 on every request."""
    model_path = str(model_path)
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = tf.keras.models.load_model(
            model_path, compile=False, custom_objects=CUSTOM_OBJECTS
        )
    return _MODEL_CACHE[model_path]


def predict_image(model_path: str, image_path_or_array, class_names=CLASS_NAMES):
    """
    Returns:
        {
          "predicted_class": "plastic",
          "confidence": 0.94,
          "probabilities": {"cardboard": 0.01, "glass": 0.00, ...}
        }
    """
    model = load_model(model_path)
    batch = preprocess_image(image_path_or_array, img_size=IMG_SIZE)
    probs = model.predict(batch, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
    }


def clear_model_cache():
    """Called after retraining so the API picks up the freshly trained model
    instead of serving stale cached weights."""
    _MODEL_CACHE.clear()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.prediction <model_path> <image_path>")
    else:
        result = predict_image(sys.argv[1], sys.argv[2])
        print(result)
