import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def enable_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_history(history_path: Path) -> Optional[Dict[str, list]]:
    if not history_path.exists():
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)

def preprocess_for_cnn(image: Image.Image) -> np.ndarray:
    img = image.convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1) 
    arr = np.expand_dims(arr, axis=0) 
    return arr

def preprocess_for_vgg(image: Image.Image, img_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    img = image.convert("L").resize(img_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.repeat(arr, repeats=3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict(model: tf.keras.Model, x: np.ndarray) -> Tuple[np.ndarray, int]:
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

