"""
Neural-Sense | Sentiment Classifier | Streamlit App
Authors : Ahmad Hassan  | Muhammad Fareed Ghani 

HOW TO RUN:
    conda activate neuralsense
    streamlit run app.py
"""

import re
import time
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Neural-Sense | Sentiment Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# MODEL LOADING
@st.cache_resource(show_spinner=False)
def load_model_and_vocab():
    import tensorflow as tf
    from tensorflow.keras.datasets import imdb

    model      = tf.keras.models.load_model("neural_sense_model.h5")
    meta       = np.load("neural_sense_meta.npz")
    vocab_size = int(meta["vocab_size"])
    max_len    = int(meta["max_len"])
    word_index = imdb.get_word_index()
    return model, word_index, vocab_size, max_len


# PREDICTION
def predict(text, model, word_index, vocab_size, max_len, thresh_pos, thresh_neg):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Clean and tokenise
    clean  = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = clean.split()

    # Encode — unknown words get index 2 (<UNK>), offset by +3 per IMDb convention
    encoded = []
    for t in tokens:
        idx = word_index.get(t, None)
        if idx is not None and idx + 3 < vocab_size:
            encoded.append(idx + 3)
        else:
            encoded.append(2)   # <UNK>

    if not encoded:
        return "Neutral / Mixed", 0.5, 0.0

    padded = pad_sequences([encoded], maxlen=max_len, padding="post", truncating="post")
    score  = float(model.predict(padded, verbose=0)[0][0])

    if score >= thresh_pos:
        return "Positive", score, score
    elif score <= thresh_neg:
        return "Negative", score, 1.0 - score
    else:
        return "Neutral / Mixed", score, 1.0 - abs(score - 0.5) * 2


# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:      #0b0f19;
    --surface: #111827;
    --border:  #1f2937;
    --accent:  #f0a500;
    --pos:     #00e676;
    --neg:     #ff5252;
    --neu:     #40c4ff;
    --text:    #e5e7eb;
    --muted:   #6b7280;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
h1, h2, h3, h4 { font-family: var(--mono) !important; color: var(--text) !important; }

textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 15px !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(240,165,0,.25) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #0b0f19 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    transition: opacity .2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: .85 !important; }

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-val { font-family: var(--mono); font-size: 1.7rem; font-weight: 700; color: var(--accent); }
.metric-key { font-size: 0.76rem; color: var(--muted); margin-top: 4px; }

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 20px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-card.pos::before { background: var(--pos); }
.result-card.neg::before { background: var(--neg); }
.result-card.neu::before { background: var(--neu); }

.result-label { font-family: var(--mono); font-size: 2.2rem; font-weight: 700; letter-spacing: -1px; margin-bottom: 4px; }
.result-label.pos { color: var(--pos); }
.result-label.neg { color: var(--neg); }
.result-label.neu { color: var(--neu); }

.result-sub { font-size: 0.88rem; color: var(--muted); margin-bottom: 20px; }

.bar-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }
.bar-wrap { background: #1f2937; border-radius: 6px; height: 10px; margin-bottom: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; }
.bar-fill.pos { background: linear-gradient(90deg, #00e676, #69f0ae); }
.bar-fill.neg { background: linear-gradient(90deg, #ff5252, #ff8a80); }
.bar-fill.neu { background: linear-gradient(90deg, #40c4ff, #80d8ff); }

.chip { display: inline-block; font-family: var(--mono); font-size: 0.73rem; padding: 3px 10px; border-radius: 20px; margin-right: 6px; margin-top: 8px; background: #1f2937; color: var(--muted); }

.history-row { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
.hist-label { font-family: var(--mono); font-weight: 700; font-size: 0.9rem; }
.hist-label.pos { color: var(--pos); }
.hist-label.neg { color: var(--neg); }
.hist-label.neu { color: var(--neu); }
.hist-meta { font-size: 0.76rem; color: var(--muted); display: inline; margin-left: 10px; font-family: var(--sans); font-weight: 400; }
.hist-text { color: var(--muted); font-size: 0.82rem; margin-top: 4px; }

.mono-tag { font-family: var(--mono); font-size: 0.72rem; color: var(--accent); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.divider { border-top: 1px solid var(--border); margin: 22px 0; }

.ex-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE
if "history"        not in st.session_state: st.session_state.history        = []
if "total_analyzed" not in st.session_state: st.session_state.total_analyzed = 0
if "input_text"     not in st.session_state: st.session_state.input_text     = ""


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 18px'>
        <div class='mono-tag'>Neural-Sense v1.0</div>
        <div style='font-family:var(--mono);font-size:1.3rem;font-weight:700;margin-top:6px;line-height:1.3;color:#e5e7eb'>
            Sentiment<br>Classifier
        </div>
        <div style='font-size:0.78rem;color:#6b7280;margin-top:6px'>BiLSTM · IMDb · TensorFlow</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='mono-tag'>Model Info</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.81rem;color:#9ca3af;line-height:2;margin-top:6px'>
        <b style='color:#e5e7eb'>Architecture</b><br>Bidirectional LSTM<br>
        <b style='color:#e5e7eb'>Dataset</b><br>IMDb (50,000 reviews)<br>
        <b style='color:#e5e7eb'>Embedding</b><br>Trainable (128-dim)<br>
        <b style='color:#e5e7eb'>Classes</b><br>Positive · Negative · Mixed<br>
        <b style='color:#e5e7eb'>Test Accuracy</b><br>≥ 82%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='mono-tag'>Thresholds</div>", unsafe_allow_html=True)
    threshold_pos = st.slider("Positive threshold", 0.50, 0.90, 0.60, 0.05)
    threshold_neg = st.slider("Negative threshold", 0.10, 0.50, 0.40, 0.05)

    st.markdown("---")
    if st.button("🗑  Clear History"):
        st.session_state.history        = []
        st.session_state.total_analyzed = 0
        st.rerun()

    st.markdown("""
    <div style='font-size:0.7rem;color:#374151;margin-top:18px;text-align:center'>
        Ahmad Hassan · M. Fareed Ghani
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:8px 0 4px'>
    <div class='mono-tag'>Deep Learning </div>
    <h1 style='font-family:var(--mono);font-size:2.4rem;font-weight:700;margin:8px 0 4px;letter-spacing:-1px'>
        🧠 Neural-Sense
    </h1>
    <div style='color:#6b7280;font-size:0.93rem'>
        Sentiment Classification of Consumer Reviews using Bidirectional LSTMs
    </div>
</div>
<div class='divider'></div>
""", unsafe_allow_html=True)


# LOAD MODEL
with st.spinner("Loading model…"):
    try:
        model, word_index, vocab_size, max_len = load_model_and_vocab()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error   = str(e)

if not model_loaded:
    st.error(f"**Could not load model.** Make sure `neural_sense_model.h5` and `neural_sense_meta.npz` are in the same folder.\n\n`{load_error}`")
    st.stop()

# STATS ROW
pos_count = sum(1 for h in st.session_state.history if h["label"] == "Positive")
neg_count = sum(1 for h in st.session_state.history if h["label"] == "Negative")
neu_count = sum(1 for h in st.session_state.history if h["label"] == "Neutral / Mixed")

c1, c2, c3, c4 = st.columns(4)
for col, val, key in [(c1, st.session_state.total_analyzed, "Analyzed"),
                       (c2, pos_count, "Positive"),
                       (c3, neg_count, "Negative"),
                       (c4, neu_count, "Mixed")]:
    with col:
        st.markdown(
            f"<div class='metric-box'><div class='metric-val'>{val}</div><div class='metric-key'>{key}</div></div>",
            unsafe_allow_html=True
        )

st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)

# EXAMPLE SENTENCES
examples = [
    "The acting was superb and the story was deeply moving.",
    "This was a complete waste of two hours. Terrible script.",
    "Oh yeah, because who doesn't love sitting through 3 hours of pain.",
    "The movie wasn't bad, it was actually great!",
    "A masterpiece if you enjoy watching paint dry.",
    "Not what I expected, but in the best possible way.",
    "Boring, predictable, and painfully long. Avoid at all costs.",
    "I hated the first half but the ending was absolutely incredible.",
]

with st.expander("✨  Try example sentences"):
    st.markdown("<div class='ex-label'>Click any sentence to load it into the input box</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            short = ex[:44] + "…" if len(ex) > 44 else ex
            if st.button(f'"{short}"', key=f"ex_{i}", use_container_width=True):
                st.session_state.input_text = ex
                st.rerun()


# TEXT INPUT
st.markdown("<div class='mono-tag' style='margin-top:14px'>Input</div>", unsafe_allow_html=True)

user_text = st.text_area(
    label="Enter your review",
    value=st.session_state.input_text,
    placeholder="Type or paste any review here…\ne.g. \"The movie wasn't bad, it was actually great!\"",
    height=130,
    label_visibility="collapsed",
)

# Keep session state in sync so example prefill works
st.session_state.input_text = user_text

word_count = len(user_text.split()) if user_text.strip() else 0
st.markdown(
    f"<div style='font-size:0.76rem;color:#374151;text-align:right'>{word_count} words · {len(user_text)} chars</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
analyse_clicked = st.button("⚡  Analyse Sentiment")

# PREDICTION + RESULT
if analyse_clicked:
    raw = user_text.strip()
    if not raw:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running inference…"):
            t0 = time.time()
            label, score, confidence = predict(
                raw, model, word_index, vocab_size, max_len,
                threshold_pos, threshold_neg
            )
            elapsed = (time.time() - t0) * 1000

        st.session_state.total_analyzed += 1
        st.session_state.history.insert(0, {
            "text": raw, "label": label,
            "score": score, "confidence": confidence
        })

        css_cls  = {"Positive": "pos", "Negative": "neg"}.get(label, "neu")
        emoji    = {"Positive": "😊", "Negative": "😞"}.get(label, "🤔")
        conf_pct = f"{confidence * 100:.1f}%"
        neg_pct  = f"{(1 - score) * 100:.1f}%"
        pos_pct  = f"{score * 100:.1f}%"
        bar_w    = f"{score * 100:.1f}%"

        st.markdown(
            f"<div class='result-card {css_cls}'>"
            f"<div class='result-label {css_cls}'>{emoji}&nbsp; {label}</div>"
            f"<div class='result-sub'>Confidence {conf_pct} &nbsp;·&nbsp; Inferred in {elapsed:.0f} ms</div>"
            f"<div class='bar-row'><span>Negative</span><span>Positive</span></div>"
            f"<div class='bar-wrap'><div class='bar-fill {css_cls}' style='width:{bar_w}'></div></div>"
            f"<div class='bar-row'><span>{neg_pct}</span><span>{pos_pct}</span></div>"
            f"<span class='chip'>Raw Score: {score:.4f}</span>"
            f"<span class='chip'>Vocab: {vocab_size:,} words</span>"
            f"<span class='chip'>Max Len: {max_len} tokens</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        if label == "Positive":
            st.success("The model detects an overall **positive** sentiment. The text likely contains appreciative or affirmative language.")
        elif label == "Negative":
            st.error("The model detects an overall **negative** sentiment. The text likely contains critical, disappointed, or hostile language.")
        else:
            st.info("The model is **uncertain** — the text may be sarcastic, mixed, or contain contradictory sentiment signals.")

# HISTORY
if st.session_state.history:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='mono-tag'>Analysis History</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    for entry in st.session_state.history[:10]:
        css_cls = {"Positive": "pos", "Negative": "neg"}.get(entry["label"], "neu")
        emoji   = {"Positive": "😊", "Negative": "😞"}.get(entry["label"], "🤔")
        trunc   = entry["text"][:90] + "…" if len(entry["text"]) > 90 else entry["text"]
        st.markdown(
            f"<div class='history-row'>"
            f"<div class='hist-label {css_cls}'>{emoji} {entry['label']}"
            f"<span class='hist-meta'>score {entry['score']:.3f} · conf {entry['confidence']*100:.0f}%</span>"
            f"</div>"
            f"<div class='hist-text'>\"{trunc}\"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# FOOTER
st.markdown("""
<div class='divider'></div>
<div style='text-align:center;font-size:0.76rem;color:#374151;padding:6px 0 18px'>
    <span style='font-family:var(--mono);color:#f0a500'>Neural-Sense</span>
    &nbsp;·&nbsp; Ahmad Hassan &amp; Muhammad Fareed Ghani 

</div>
""", unsafe_allow_html=True)