from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from utils import (
    CLASS_NAMES,
    enable_gpu_memory_growth,
    load_history,
    preprocess_for_cnn,
    preprocess_for_vgg,
    predict,
)

MODELS_DIR = Path("models")
HISTORY_DIR = Path("history")
CNN_MODEL_PATH = MODELS_DIR / "best_fashion_cnn.keras"
VGG_MODEL_PATH = MODELS_DIR / "best_vgg_finetuned.keras"
CNN_HISTORY_PATH = HISTORY_DIR / "history_cnn.json"
VGG_HISTORY_PATH = HISTORY_DIR / "history_vgg.json"

@st.cache_resource
def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path.as_posix())


def plot_history(history: dict, title_prefix: str) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure(figsize=(5.1, 3.8))
        plt.plot(list(epochs), history.get("loss", []), label="loss")
        if "val_loss" in history:
            plt.plot(list(epochs), history.get("val_loss", []), label="val_loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{title_prefix} loss", fontsize=10)
        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure(figsize=(5.1, 3.8))
        plt.plot(list(epochs), history.get("accuracy", []), label="accuracy")
        if "val_accuracy" in history:
            plt.plot(list(epochs), history.get("val_accuracy", []), label="val_accuracy")

        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{title_prefix} accuracy", fontsize=10)
        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

def main() -> None:
    st.set_page_config(page_title="CNN/VGG16 Image Classifier", layout="wide")

    enable_gpu_memory_growth()
    gpus = tf.config.list_physical_devices("GPU")

    st.title("Clothing Image Classification")
    st.caption("Fashion-MNIST dataset · CNN or VGG16")

    with st.sidebar:
        st.subheader("Model selection")
        model_choice = st.radio("Choose a model", ["VGG16","CNN",], index=0)

    missing = []
    if not CNN_MODEL_PATH.exists():
        missing.append(str(CNN_MODEL_PATH))
    if not VGG_MODEL_PATH.exists():
        missing.append(str(VGG_MODEL_PATH))
    if missing:
        st.error("Missing model files:\n" + "\n".join(missing))
        st.stop()

    uploaded = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to start prediction.")
        st.stop()

    from PIL import Image
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Prediction")

        if model_choice.startswith("VGG16"):
            model = load_model(VGG_MODEL_PATH)
            x = preprocess_for_vgg(image, img_size=(64, 64))
            probs, pred_idx = predict(model, x)
            history = load_history(VGG_HISTORY_PATH)
            model_name = "VGG16"
        else:
            model = load_model(CNN_MODEL_PATH)
            x = preprocess_for_cnn(image)
            probs, pred_idx = predict(model, x)
            history = load_history(CNN_HISTORY_PATH)
            model_name = "CNN"

        confidence = float(probs[pred_idx]) * 100
        color = "#4CAF50" if confidence > 70 else "#FF9800"
        
        st.markdown(
            f"""    
            <p>
                <b>Predicted class:</b>
                <span style="font-size:26px; font-weight:700; color:{color};">
                    {CLASS_NAMES[pred_idx]}
                </span>
                <span style="font-size:14px; color:gray;">
                    ({confidence:.1f}%)
                </span>
            </p>
            """,
            unsafe_allow_html=True
        )


        st.markdown("**Probabilities:**")

        def format_prob(p: float) -> str:
            pct = p * 100
            if pct < 0.01:
                return "<0.01 %"
            return f"{pct:.2f} %"

        df_probs = pd.DataFrame({
            "class": CLASS_NAMES,
            "probability_raw": [float(p) for p in probs],
        })

        df_probs = df_probs.sort_values("probability_raw", ascending=False)
        df_probs["probability"] = df_probs["probability_raw"].map(format_prob)
        df_probs = df_probs.drop(columns=["probability_raw"])

        st.dataframe(df_probs, use_container_width=True)


        fig = plt.figure()
        y = np.arange(len(CLASS_NAMES))
        plt.barh(y, probs)
        plt.yticks(y, CLASS_NAMES)
        plt.xlabel("probability")
        plt.title(f"{model_name} probabilities")
        st.pyplot(fig)

    st.divider()
    st.subheader("Training curves (loss/accuracy)")

    if history is None:
        st.warning(
            "History file not found. Export it from the notebook into `history/*.json` "
            "to display loss/accuracy curves."
        )
    else:
        plot_history(history, title_prefix=model_name)


if __name__ == "__main__":
    main()
