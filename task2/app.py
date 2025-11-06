# app.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from price_gap import find_price_gap_pair
from movies_client import search_movies

app = FastAPI(title="Price Gap + Movies API")


# --------------------------
# POST /api/price-gap-pair
# --------------------------

class PriceGapRequest(BaseModel):
    nums: list[int] = Field(..., description="List of integers")
    k: int = Field(..., ge=0, description="Non-negative difference")


class PriceGapResponse(BaseModel):
    i: Optional[int]
    j: Optional[int]
    value_i: Optional[int]
    value_j: Optional[int]


@app.post("/api/price-gap-pair", response_model=PriceGapResponse)
def price_gap_pair(payload: PriceGapRequest):

    pair = find_price_gap_pair(payload.nums, payload.k)
    if pair is None:
        return PriceGapResponse(i=None, j=None, value_i=None, value_j=None)

    i, j = pair
    return PriceGapResponse(
        i=i,
        j=j,
        value_i=payload.nums[i],
        value_j=payload.nums[j]
    )


# -------------------
# GET /api/movies
# -------------------

@app.get("/api/movies")
def movies(
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1)
):
    if not q:
        return {
            "results": [],
            "total_results": 0,
            "total_pages": 0,
            "page": page
        }

    try:
        return search_movies(q, page)
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream movie API error")
