# app_embeddings_faiss.py
"""
Optional: embeddings + FAISS retrieval (vector search).

Notes:
- Requires: sentence-transformers, faiss-cpu (or faiss-gpu), pandas, streamlit
- If packages are missing, the app shows instructions instead of crashing.
"""
import os, sys
import streamlit as st

missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    missing.append("sentence-transformers")
try:
    import faiss
except Exception:
    missing.append("faiss-cpu (or faiss-gpu)")

if missing:
    st.set_page_config(page_title="Banking Assistant (Embeddings)", page_icon="🧭")
    st.title("🧭 Embeddings + FAISS Demo")
    st.error("Missing packages: " + ", ".join(missing))
    st.code("""
pip install streamlit pandas sentence-transformers faiss-cpu
streamlit run app_embeddings_faiss.py
""")
    st.stop()

import numpy as np

@st.cache_data(show_spinner=False)
def load_kb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"id","title","text","category"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"KB is missing columns: {missing}")
    return df

@st.cache_resource(show_spinner=False)
def build_index(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    return model, index, embs

def search(query, model, index, k=3):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0])]

st.set_page_config(page_title="Banking Assistant (Embeddings)", page_icon="🧭", layout="centered")
st.title("🧭 AI-Powered Banking Assistant — Embeddings + FAISS")

kb_path = st.text_input("Knowledge Base CSV path", value="bank_kb.csv")
try:
    kb = load_kb(kb_path)
    model, index, embs = build_index(kb["text"].tolist())
except Exception as e:
    st.error(f"Could not set up index: {e}")
    st.stop()

q = st.text_input("Ask a banking question", placeholder="How do I block a stolen card?")
k = st.slider("Top-k", 1, 10, 3)

if q:
    hits = search(q, model, index, k=k)
    st.subheader("Top Matches")
    for rank, (i, score) in enumerate(hits, start=1):
        row = kb.iloc[i]
        st.markdown(f"**[{rank}] {row['title']}** — score: {score:.3f} — category: `{row['category']}`")
        with st.expander("Show text"):
            st.write(row["text"])

    top_i, top_score = hits[0]
    top_row = kb.iloc[top_i]
    st.markdown("---")
    st.subheader("Assistant Answer")
    st.write(f"Based on our knowledge base, here's what to do: **{top_row['text']}**\n\n_Source:_ [{top_row['title']}]")
