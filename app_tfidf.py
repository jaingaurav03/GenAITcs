# app_tfidf.py
import os
import re
from typing import List, Dict, Tuple

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ------------------------
# Data loading
# ------------------------
@st.cache_data(show_spinner=False)
def load_kb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    expected = {"id","title","text","category"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"KB is missing columns: {missing}")
    return df

# ------------------------
# Build TF-IDF index
# ------------------------
@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: List[str]):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X

# ------------------------
# Intent + PII
# ------------------------
INTENT_KEYWORDS = {
    "cards": ["card","limit","cvv","expiry","block","stolen","upgrade"],
    "accounts": ["balance","account","statement","passbook","transfer","imps","neft","rtgs"],
    "loans": ["loan","emi","prepay","foreclose","moratorium","interest"],
    "kyc": ["kyc","aadhaar","pan","address proof","re-kyc"],
    "fraud": ["fraud","unauthorized","chargeback","dispute","scam"],
}

PII_PATTERNS = [
    (re.compile(r"\b\d{12}\b"), "<AADHAAR_MASKED>"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "<PAN_MASKED>"),
    (re.compile(r"\b\d{16}\b"), "<CARD_MASKED>"),
    (re.compile(r"\b\d{10}\b"), "<PHONE_MASKED>"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "<EMAIL_MASKED>"),
]

def detect_intent(query: str) -> str:
    q = query.lower()
    best, score = "general", 0
    for intent, kws in INTENT_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in q)
        if s > score:
            best, score = intent, s
    return best

def redact_pii(text: str) -> str:
    out = text
    for pat, repl in PII_PATTERNS:
        out = pat.sub(repl, out)
    return out

def retrieve(query: str, vec, mat, docs: List[str], k: int = 3):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).flatten()
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Banking Assistant (TF-IDF)", page_icon="💬", layout="centered")
st.title("💬 AI-Powered Banking Assistant — TF‑IDF")
st.caption("Works offline. Optional LLMs can be added later.")

kb_path = st.text_input("Knowledge Base CSV path", value="bank_kb.csv")
try:
    kb = load_kb(kb_path)
    vec, mat = build_tfidf_index(kb["text"].tolist())
except Exception as e:
    st.error(f"Could not load KB: {e}")
    st.stop()

with st.expander("ℹ️ How it works"):
    st.markdown("""
1) **PII redaction** for card/Aadhaar/PAN/email/phone.  
2) **Intent detection** using lightweight keywords.  
3) **TF‑IDF retrieval** to surface the top‑k passages.  
4) **Template answer** citing the most relevant doc.
""")

q = st.text_input("Ask a banking question", placeholder="How can I increase my credit card limit?")
col1, col2 = st.columns(2)
with col1:
    k = st.slider("Passages (k)", 1, 5, 3)
with col2:
    show_texts = st.toggle("Show passage texts", value=True)

if q:
    red_q = redact_pii(q)
    intent = detect_intent(red_q)
    hits = retrieve(red_q, vec, mat, kb["text"].tolist(), k=k)

    st.write(f"**Intent:** `{intent}`  |  **PII‑safe query:** {red_q}")

    st.subheader("Top Passages")
    for rank, (i, score) in enumerate(hits, start=1):
        row = kb.iloc[i]
        st.markdown(f"**[{rank}] {row['title']}** — score: {score:.3f} — category: `{row['category']}`")
        if show_texts:
            with st.expander("Show text"):
                st.write(row["text"])

    # Compose simple answer from top hit
    top_i, top_score = hits[0]
    top_row = kb.iloc[top_i]
    st.markdown("---")
    st.subheader("Assistant Answer")
    st.write(f"Based on our knowledge base, here's what to do: **{top_row['text']}**\n\n_Source:_ [{top_row['title']}]")

    st.markdown("---")
    st.subheader("Feedback")
    fb = st.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True, index=0)
    comment = st.text_input("Optional comment")
    if st.button("Submit Feedback"):
        st.success("Thanks! Your feedback has been recorded (demo).")

st.markdown("---")
st.markdown("### Run")
st.code("""
pip install -U streamlit scikit-learn pandas
streamlit run app_tfidf.py
""", language="bash")
