# AI-Powered Banking Customer Query Assistant

This project includes two Streamlit apps:
- `app_tfidf.py` — works fully offline using TF‑IDF retrieval
- `app_embeddings_faiss.py` — optional vector search with sentence-transformers + FAISS

## Quick Start (TF‑IDF)
```bash
pip install -U streamlit scikit-learn pandas
streamlit run app_tfidf.py
```

## Optional: Embeddings + FAISS
```bash
pip install -U streamlit pandas sentence-transformers faiss-cpu
streamlit run app_embeddings_faiss.py
```

## Files
- `bank_kb.csv` — sample knowledge base (replace with your real FAQs/policies)
- `requirements.txt` — minimal requirements
- `app_tfidf.py` — main demo app
- `app_embeddings_faiss.py` — optional enhanced retrieval

## Notes for Production
- Add authentication (SSO/OAuth), audit logging, stricter PII detection
- Replace CSV with DB or CMS; add versioning and approvals
- Use embeddings + vector DB (FAISS, PGVector, Pinecone) for better recall
- Add LLM guardrails + human handoff, monitoring, and CI checks
