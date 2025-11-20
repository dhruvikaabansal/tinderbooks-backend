# model_utils.py
import numpy as np
import pandas as pd
import joblib
import faiss
from typing import List

# --- paths to artifacts (make sure these files exist in artifacts/) ---
BOOKS_CSV = "artifacts/books_for_reco.csv"
EMB_PATH   = "artifacts/book_embs.npy"
FAISS_IDX  = "artifacts/books_faiss.index"
RERANKER   = "artifacts/reranker_lgbm_v2.joblib"
SCALER     = "artifacts/reranker_scaler_v2.joblib"

# --- load artifacts once at startup ---
books = pd.read_csv(BOOKS_CSV)
embs = np.load(EMB_PATH)
index = faiss.read_index(FAISS_IDX)
model = joblib.load(RERANKER)
scaler = joblib.load(SCALER)

# feature names - must match how you trained reranker
FEATURES = ["cos_sim","pop","overlap","author_match","series_match","author_pop"]

# precompute some arrays / maps
pop_arr = books['avg_rating'].fillna(3.0).to_numpy() if 'avg_rating' in books.columns else np.ones(len(books))
title_tokens = [set(str(t).lower().split()) for t in books['title'].fillna("")]
author_clean = books['authors'].astype(str).str.split(",").str[0].str.strip().fillna("unknown")
author_of = {i: a for i,a in author_clean.to_dict().items()}

def extract_series(title):
    if not isinstance(title, str): return ""
    if "(" in title and ")" in title:
        try:
            return title.split("(")[1].split(")")[0].lower()
        except:
            return ""
    return ""

series_of = {i: extract_series(t) for i,t in enumerate(books['title'].fillna(""))}
author_pop_arr = books['authors'].astype(str).apply(lambda x: 1).to_numpy()  # placeholder counts

def build_user_vec(liked_indices):
    if len(liked_indices)==0:
        return np.zeros((embs.shape[1],), dtype=np.float32)
    v = embs[liked_indices].mean(axis=0)
    norm = np.linalg.norm(v)
    if norm>0:
        v = v / norm
    return v.astype(np.float32)

def recommend_for_likes(liked_indices: List[int], top_k:int=10, cand_k:int=200):
    uvec = build_user_vec(liked_indices)
    D,I = index.search(uvec.reshape(1,-1), k=cand_k)
    cands = I[0].tolist()
    feats = []
    user_token_union = set()
    for i in liked_indices:
        if 0 <= i < len(title_tokens):
            user_token_union |= title_tokens[i]
    user_authors = set(author_of.get(i,"") for i in liked_indices)
    user_series  = set(series_of.get(i,"") for i in liked_indices if series_of.get(i,""))
    for c in cands:
        cos_sim = float(np.dot(uvec, embs[c]))
        pop = float(pop_arr[c]) if c < len(pop_arr) else 1.0
        overlap = len(user_token_union & title_tokens[c]) if c < len(title_tokens) else 0
        author_match = 1 if author_of.get(c,"") in user_authors else 0
        series_match = 1 if (series_of.get(c,"") and series_of.get(c,"") in user_series) else 0
        author_pop = float(author_pop_arr[c]) if c < len(author_pop_arr) else 0.0
        feats.append([cos_sim, pop, overlap, author_match, series_match, author_pop])
    X_scaled = scaler.transform(feats)
    import pandas as pd
    X_df = pd.DataFrame(X_scaled, columns=FEATURES)
    scores = model.predict_proba(X_df)[:,1]
    order = np.argsort(scores)[::-1][:top_k]
    recs = []
    for i in order:
        idx = cands[i]
        recs.append({
            "book_idx": int(idx),
            "score": float(scores[i]),
            "title": str(books.iloc[idx].title),
            "authors": str(books.iloc[idx].authors) if 'authors' in books.columns else "",
            "image": books.iloc[idx].small_image_url if 'small_image_url' in books.columns else None
        })
    return recs
