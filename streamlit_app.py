"""Streamlit deployment with the same UI behavior as the local HTML frontend."""
from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
API_DIR = ROOT_DIR / "api"
MODEL_DIR = ROOT_DIR / "models" / "kinnews"
sys.path.insert(0, str(API_DIR))

from nlp_utils import DeployedClassifier  # noqa: E402

MAX_TEXT_LENGTH = 20000


@st.cache_resource
def load_classifier() -> DeployedClassifier:
    return DeployedClassifier(MODEL_DIR)


def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="News topic classifier: Kinyarwanda", page_icon="📰", layout="centered")
st.markdown(
    """
    <style>
    :root { --bg:#f6f7f4; --card:#fff; --text:#1d2420; --muted:#5b665f;
            --line:#d9ded8; --green:#1f7a4d; --bar:#cfe6d8; --error-bg:#fdecea;
            --error:#a12622; --warn:#8a5a00; }
    .stApp { background:var(--bg); color:var(--text); }
    .block-container { max-width:760px; padding:2rem 1rem 3rem; }
    h1 { font-size:1.6rem !important; margin-bottom:.25rem !important; }
    .lead, .small { color:var(--muted); }
    .lead { margin-bottom:1.25rem; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:1rem; margin-bottom:1rem; }
    .card h2 { color:var(--muted); font-size:.95rem; margin:0 0 .15rem; }
    .topic { color:var(--green); font-size:2rem; font-weight:700; margin:0; }
    .model-info { color:var(--muted); font-size:.85rem; margin-top:.4rem; }
    .warning { color:var(--warn); margin-top:.4rem; }
    .bar-row { display:grid; grid-template-columns:120px 1fr 56px; align-items:center;
               gap:8px; padding:3px 0; }
    .bar-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar-name.top { font-weight:700; }
    .bar-track { background:var(--bar); border-radius:6px; height:12px; overflow:hidden; }
    .bar-fill { background:var(--green); height:100%; border-radius:6px; }
    .bar-pct { text-align:right; font-variant-numeric:tabular-nums; }
    div[data-testid="stTextArea"] textarea { min-height:180px; }
    div[data-testid="stButton"] button { border-radius:8px; }
    div[data-testid="stButton"] button[kind="primary"] { background:var(--green); border-color:var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)

if "text" not in st.session_state:
    st.session_state.text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = ""

st.markdown("# News topic classifier")
st.markdown(
    '<p class="lead">Write or paste a news text in <strong>Kinyarwanda</strong>. '
    "The model predicts its topic (politics, sport, economy, health, ...).</p>",
    unsafe_allow_html=True,
)

card_start()
st.markdown("**Kinyarwanda model**")
if not (MODEL_DIR / "config.json").exists():
    st.error("The trained Kinyarwanda model is missing from the deployment.")
    st.info("Run the notebook's model-saving section and include models/kinnews before deploying.")
    card_end()
    st.stop()

try:
    classifier = load_classifier()
    config = classifier.config
    st.markdown(
        f'<p class="model-info">Model quality on test articles: macro-F1 '
        f'{config["test_macro_f1"]:.2f}, accuracy {config["test_accuracy"]:.2f} '
        f'({len(classifier.labels)} topics)</p>',
        unsafe_allow_html=True,
    )
except Exception as error:
    st.error(f"Could not load the model: {error}")
    classifier = None
card_end()

card_start()
st.markdown("**News text**")
text = st.text_area(
    "News text",
    key="text",
    height=180,
    max_chars=MAX_TEXT_LENGTH,
    label_visibility="collapsed",
    placeholder="Write or paste the text here ...",
)
left, right = st.columns([1, 1])
with left:
    classify_clicked = st.button("Classify", type="primary", disabled=classifier is None, use_container_width=True)
with right:
    clear_clicked = st.button("Clear", use_container_width=True)
st.caption(f"{len(text)} / {MAX_TEXT_LENGTH} characters")
if st.session_state.error:
    st.error(st.session_state.error)
st.caption("Tip: use the Classify button to submit the text.")
card_end()

if clear_clicked:
    st.session_state.text = ""
    st.session_state.result = None
    st.session_state.error = ""
    st.rerun()

if classify_clicked:
    st.session_state.error = ""
    st.session_state.result = None
    if not text.strip():
        st.session_state.error = "Please write or paste some text first."
    elif classifier is None:
        st.session_state.error = "The model is not available. Please try again later."
    else:
        result = classifier.predict(text.strip())
        if result is None:
            st.session_state.error = "The text is empty after cleaning. Please send some real text."
        else:
            st.session_state.result = result
    st.rerun()

result = st.session_state.result
if result:
    card_start()
    st.markdown("<h2>Predicted topic</h2>", unsafe_allow_html=True)
    st.markdown(f'<p class="topic">{result["label"].capitalize()}</p>', unsafe_allow_html=True)
    st.markdown(f'Confidence: {result["confidence"] * 100:.1f}%')
    if result["confidence"] < 0.5:
        st.markdown(
            '<p class="warning">The model is not very sure about this text. Treat the result with care.</p>',
            unsafe_allow_html=True,
        )
    bars = []
    for index, (topic, probability) in enumerate(list(result["probabilities"].items())[:6]):
        width = max(0.0, min(100.0, probability * 100))
        top_class = " top" if index == 0 else ""
        bars.append(
            f'<div class="bar-row"><span class="bar-name{top_class}">{topic.capitalize()}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<span class="bar-pct">{probability * 100:.1f}%</span></div>'
        )
    st.markdown("".join(bars), unsafe_allow_html=True)
    with st.expander("What the model actually saw (after cleaning)"):
        st.code(result["cleaned_text"], language=None)
    card_end()

st.caption("The model can make mistakes. Do not rely on it alone for important decisions.")
