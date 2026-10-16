"""Streamlit deployment for the Kinyarwanda news-topic classifier."""
from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
API_DIR = ROOT_DIR / "api"
MODEL_DIR = ROOT_DIR / "models" / "kinnews"
sys.path.insert(0, str(API_DIR))

from nlp_utils import DeployedClassifier  # noqa: E402


@st.cache_resource
def load_classifier() -> DeployedClassifier:
    return DeployedClassifier(MODEL_DIR)


st.set_page_config(page_title="Kinyarwanda News Classifier", page_icon="📰")
st.title("Kinyarwanda News Topic Classifier")
st.write("Enter Kinyarwanda news text and the model will predict its topic.")

if not (MODEL_DIR / "config.json").exists():
    st.error("The trained Kinyarwanda model is missing from the deployment.")
    st.info("Run the notebook's model-saving section and include models/kinnews before deploying.")
    st.stop()

classifier = load_classifier()
text = st.text_area(
    "News text",
    height=220,
    max_chars=20000,
    placeholder="Andika cyangwa ushyire hano inyandiko y'amakuru...",
)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some Kinyarwanda news text.")
    else:
        result = classifier.predict(text)
        if result is None:
            st.warning("The text is empty after cleaning.")
        else:
            st.subheader(f"Predicted topic: {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.bar_chart(result["probabilities"])
