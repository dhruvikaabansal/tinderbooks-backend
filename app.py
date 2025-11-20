# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # <-- add this import
from pydantic import BaseModel
from typing import List
from model_utils import recommend_for_likes

app = FastAPI()

# Allow browser-based frontends to call this API (demo-only: allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # for demo only; replace "*" with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LikesRequest(BaseModel):
    liked_indices: List[int]
    k: int = 10

@app.post("/recommend")
async def recommend(req: LikesRequest):
    recs = recommend_for_likes(req.liked_indices, top_k=req.k)
    return {"recs": recs}

@app.get("/health")
def health():
    return {"status": "ok"}
