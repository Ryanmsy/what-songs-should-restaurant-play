import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import urlencode

import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.environ.get(
    "ARTIFACT_PATH",
    os.path.join(HERE, "..", "artifact", "recommender_artifact.pkl"),
)

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
SPOTIFY_SCOPES = "user-read-private"  # widen once there's an actual use for the token

MODEL = {}


# The API supports two artifact shapes so ARTIFACT_PATH can point at either the
# original recommender_artifact.pkl or the newer recommender_artifact_v2.pkl
# (cuisine-genre filtering + real popularity penalty) without a code change.
# These helpers translate between the two schemas; everything else assumes
# the unified names below.
def _yelp_pc_cols(artifact):
    return artifact.get("yelp_pc_cols") or artifact["archetype_pcs"]


def _song_scaler(artifact):
    return artifact.get("scaler_song") or artifact["song_scaler"]


def _base_penalty(artifact, n_songs):
    if "penalty" in artifact:
        return artifact["penalty"]
    if "popularity_penalty" in artifact:
        return artifact["popularity_penalty"]
    return np.ones(n_songs)


def _matched_cuisine_genres(artifact, restaurant_row):
    cuisine_filters = artifact.get("cuisine_genre_filters")
    if not cuisine_filters:
        return set()
    genres = set()
    for col, genre_list in cuisine_filters.items():
        if restaurant_row.get(col, 0) == 1:
            genres.update(genre_list)
    return genres


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(ARTIFACT_PATH)
    MODEL["artifact"] = artifact
    # business_id is the stable public key - never expose the internal row position
    MODEL["business_id_to_row"] = {
        bid: i for i, bid in enumerate(artifact["restaurants_meta"]["business_id"])
    }
    yield
    MODEL.clear()


app = FastAPI(title="Restaurant Song Recommender", lifespan=lifespan)


class RestaurantSummary(BaseModel):
    business_id: str
    name: str


class RecommendRequest(BaseModel):
    business_id: str


class SongRecommendation(BaseModel):
    id: str
    name: str
    artists: str
    distance: float
    is_hub: bool = False
    genre: Optional[str] = None
    popularity: Optional[int] = None


class RecommendResponse(BaseModel):
    business_id: str
    restaurant_name: str
    dominant_spotify_pc: str
    dominant_spotify_label: str
    matched_cuisine_genres: List[str] = []
    recommendations: List[SongRecommendation]


class HealthResponse(BaseModel):
    status: str
    built_at: str
    n_restaurants: int
    n_songs: int


@app.get("/health", response_model=HealthResponse)
def health():
    artifact = MODEL["artifact"]
    return HealthResponse(
        status="ok",
        built_at=artifact.get("built_at", "unknown"),
        n_restaurants=len(artifact["restaurants_meta"]),
        n_songs=len(artifact["songs_meta"]),
    )


@app.get("/restaurants", response_model=List[RestaurantSummary])
def search_restaurants(q: str = Query(..., min_length=1), limit: int = 10):
    restaurants_meta = MODEL["artifact"]["restaurants_meta"]
    mask = restaurants_meta["name"].str.contains(q, case=False, na=False)
    matches = restaurants_meta.loc[mask, ["business_id", "name"]].head(limit)
    return [RestaurantSummary(**row) for row in matches.to_dict(orient="records")]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    artifact = MODEL["artifact"]
    row_idx = MODEL["business_id_to_row"].get(req.business_id)
    if row_idx is None:
        raise HTTPException(status_code=404, detail=f"No restaurant with business_id={req.business_id!r}")

    restaurants_meta = artifact["restaurants_meta"]
    songs_meta = artifact["songs_meta"]
    yelp_pc_cols = _yelp_pc_cols(artifact)
    spotify_pc_cols = artifact["spotify_pc_cols"]
    spotify_pc_labels = artifact["spotify_pc_labels"]
    song_scaler = _song_scaler(artifact)
    song_vecs_scaled = artifact["song_vecs_scaled"]
    base_penalty = _base_penalty(artifact, len(songs_meta))

    restaurant_row = restaurants_meta.loc[row_idx]
    y = restaurant_row[yelp_pc_cols].to_numpy(dtype=float)
    y_norm = y / (np.linalg.norm(y) + 1e-9)
    s_hat = artifact["W"] @ y_norm
    s_hat_scaled = song_scaler.transform(s_hat.reshape(1, -1))

    dominant_idx = int(np.argmax(np.abs(s_hat_scaled[0])))
    dominant_pc = spotify_pc_cols[dominant_idx]

    # Cuisine acts as a hard filter on the candidate pool (not a PCA target) -
    # see notes/progress_log.md entries 6-7 for why genre can't be reached by
    # averaging songs in audio-feature space.
    matched_genres = _matched_cuisine_genres(artifact, restaurant_row)
    if matched_genres and "track_genre" in songs_meta.columns:
        candidate_idx = np.where(songs_meta["track_genre"].isin(matched_genres).values)[0]
    else:
        candidate_idx = np.arange(len(songs_meta))

    distances = np.linalg.norm(song_vecs_scaled[candidate_idx] - s_hat_scaled, axis=1)
    distances = distances * base_penalty[candidate_idx]
    top_local = np.argsort(distances)[:5]
    top_idx = candidate_idx[top_local]

    hub_threshold = artifact.get("hub_threshold")
    in_degree = artifact.get("in_degree")
    has_genre = "track_genre" in songs_meta.columns
    has_popularity = "popularity" in songs_meta.columns

    recommendations = [
        SongRecommendation(
            id=str(songs_meta.loc[song_idx, "id"]),
            name=str(songs_meta.loc[song_idx, "name"]),
            artists=str(songs_meta.loc[song_idx, "artists"]),
            distance=float(distances[local_idx]),
            is_hub=bool(in_degree[song_idx] > hub_threshold) if in_degree is not None else False,
            genre=str(songs_meta.loc[song_idx, "track_genre"]) if has_genre else None,
            popularity=int(songs_meta.loc[song_idx, "popularity"]) if has_popularity else None,
        )
        for local_idx, song_idx in zip(top_local, top_idx)
    ]

    return RecommendResponse(
        business_id=req.business_id,
        restaurant_name=str(restaurant_row["name"]),
        dominant_spotify_pc=dominant_pc,
        dominant_spotify_label=spotify_pc_labels[dominant_pc],
        matched_cuisine_genres=sorted(matched_genres),
        recommendations=recommendations,
    )


@app.get("/login")
def spotify_login():
    """Redirect to Spotify's consent screen. Visit this route in a browser to start the flow."""
    if not SPOTIFY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="SPOTIFY_CLIENT_ID is not set")

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
    }
    return RedirectResponse(f"https://accounts.spotify.com/authorize?{urlencode(params)}")


@app.get("/callback")
def spotify_callback(code: Optional[str] = None, error: Optional[str] = None):
    """Spotify redirects here with ?code=... after the user approves access."""
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify authorization failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' query parameter")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    if not resp.ok:
        raise HTTPException(status_code=400, detail=f"Spotify token exchange failed: {resp.text}")

    token_data = resp.json()
    MODEL["spotify_access_token"] = token_data["access_token"]
    MODEL["spotify_token_expires_at"] = time.time() + token_data["expires_in"]

    return {"status": "connected", "expires_in": token_data["expires_in"]}
