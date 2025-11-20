# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model_utils import recommend_for_likes

app = FastAPI()

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
