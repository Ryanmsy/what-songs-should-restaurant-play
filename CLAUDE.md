# Restaurant Song Recommendation Engine

## Goal
Build a recommendation engine that suggests what songs a restaurant should play, based on the restaurant's attributes and vibe.

**Deadline:** End of July 2026

## Datasets
| File | Source | Description |
|------|--------|-------------|
| `tracks_features.csv` | Spotify | ~1.2M tracks with audio features (danceability, energy, tempo, valence, etc.) |
| `yelp_clean.csv` | Yelp | Restaurant business attributes (category, ambiance, noise level, etc.) |

## Current Status
- [ ] EDA & data cleaning — Spotify (`tracks_features.csv`)
- [ ] EDA & data cleaning — Yelp (`yelp_clean.csv`)
- [ ] Covariance matrix / PCA (redundancy analysis) ← **active**
- [ ] Feature engineering / vibe mapping (restaurant attributes → audio feature targets)
- [ ] Model / recommendation logic
- [ ] Evaluation

## Tech Stack
- Python 3.10 (pinned — `scikit-learn`/`scipy` were unverified on the 3.14 interpreter this project started with; `.venv` now targets 3.10 via `uv`), Jupyter notebooks
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

## Spotify Audio Feature Ranges (Spotify API spec)
| Feature | Range | Notes |
|---------|-------|-------|
| danceability | 0.0 – 1.0 | |
| energy | 0.0 – 1.0 | |
| key | 0 – 11 | -1 = no key detected |
| loudness | ~-60 – 0 dB | |
| mode | 0 or 1 | 0=minor, 1=major |
| speechiness | 0.0 – 1.0 | |
| acousticness | 0.0 – 1.0 | |
| instrumentalness | 0.0 – 1.0 | |
| liveness | 0.0 – 1.0 | |
| valence | 0.0 – 1.0 | |
| tempo | > 0 BPM | typical range 50–250 |
| time_signature | 1 – 7 | |

## Known Data Issues (found during EDA)
- `tracks_features.csv`: ~11 rows with nulls; rows where `tempo == 0.0` should be dropped
- Artists/artist_ids stored as stringified Python lists — need parsing if joining on artist
- `tracks_features.csv`: no popularity/familiarity column — can't filter or rank songs by popularity directly; any "avoid obscure songs" logic needs a proxy (e.g. a curated known-artist list, see notebook 05b)
- `tracks_features.csv`: ~23,461 rows (~4.1%) are exact duplicates — same `(name, artists)` pair appears under multiple Spotify track IDs, presumably from appearing on more than one release/album. Dedupe on `(name, artists)` before any nearest-neighbor / hub-detection logic, otherwise duplicated songs look artificially more "popular" and can appear more than once in the same recommendation list (found in notebook 05b)
- `yelp_clean.csv`: no true NaNs, but `NoiseLevel`, `Alcohol`, `RestaurantsAttire` use `"unknown"` as a missing sentinel
- `yelp_clean.csv`: binary columns stored as float (0.0/1.0) — cast to int for ML
- `yelp_clean.csv` Yelp data: 52,268 restaurants, 23 columns
- Yelp PCA (`yelp_pca.pkl`, 13 components): yPC4's dominant loading is `NoiseLevel` (+0.50) / `stars` (-0.44) — i.e. "loud/chaotic, lower-rated," **not** late-night, despite an earlier notebook mislabeling it as "Late-Night Dive." The actual late-night axis is yPC7 (`GoodForMeal.latenight` +0.63 dominant). Verify against `pca_yelp.components_` before trusting a PC's assumed meaning.

## Deferred — Revisit After MVP is Built
- `deployment_draft/artifact/requirements.txt` duplicates `pyproject.toml`'s dependency list — decide whether to keep them independent (lean Docker builds shouldn't need the whole dev toolchain) or consolidate to one source of truth
- `ANCHOR_ARTISTS` in `build_artifact.py` (also `05b_recommender.ipynb`) has placeholder lists for `pc1` (sit-down dinner), `pc4` (loud/chaotic), `pc6` (dive bar), `pc7` (late-night) that were never reviewed by a human — only `pc2`, `pc3`, `pc5`, `pc8` came from a curated list; review/replace the placeholders
- A few anchor artists have zero matches in the song catalog, likely because notebook 04 filters to 2000–2020 releases: Michael Bublé, Vance Joy, Guns N' Roses, The Rolling Stones, Creedence Clearwater Revival — swap for artists with matching-era catalog if those archetypes need stronger anchoring
- Hub-penalty threshold (5% of restaurants) and the planned EMA smoothing factor (`alpha`) for feedback-driven `W` updates are both unvalidated defaults picked for the MVP, not tuned against real usage — revisit once feedback data exists
- Streamlit MVP intentionally ships with no thumbs up/down UI — don't add it until the `/feedback` endpoint + DynamoDB + EMA update actually exist; adding the buttons earlier would be dead UI with nothing behind it

## Folder Structure
```
restaurant_recommendation/
├── data/
│   ├── raw/              ← original files, never modify
│   └── processed/        ← cleaned outputs from notebooks
├── notebooks/            ← numbered in pipeline order
├── notes/                ← learning docs, writeups
├── deployment_draft/     ← MVP serving layer (local only, no cloud yet)
│   ├── artifact/         ← build_artifact.py + recommender_artifact.pkl (bundled model, no notebook dependency)
│   ├── api/               ← FastAPI service (not yet built)
│   └── streamlit/         ← Streamlit client (not yet built)
└── CLAUDE.md
```

## Notebooks
- `notebooks/01_spotify_eda.ipynb` — EDA and cleaning → `data/processed/tracks_features_clean.csv`
- `notebooks/02_yelp_eda.ipynb` — EDA and cleaning → `data/processed/yelp_ml_ready.csv`
- `notebooks/03_covariance_pca.ipynb` — Covariance matrix (linear algebra), redundancy analysis, eigendecomposition preview
- `notebooks/04_pca_.ipynb` — Fits PCA on both datasets (`StandardScaler` → `PCA`, Spotify features get `log1p` on skewed columns first) → saves `spotify_pca.pkl`, `spotify_scaler.pkl`, `yelp_pca.pkl`, `yelp_scaler.pkl`, `song_pca.csv`, `restaurant_pca.csv`
- `notebooks/05b_recommender.ipynb` — Current recommender baseline: hand-authored `W` weight matrix (Yelp-PC → Spotify-PC), targets grounded in real anchor-artist songs where available (falls back to a projected audio-feature intent otherwise), boss-level extreme-restaurant evaluation. Supersedes `05_trialrecommender.ipynb` (kept as history/scratch — has the mislabeling and scale bugs noted above, already fixed here)

## End Goal (Linear Algebra → ML showcase)
| Concept | Where it appears |
|---------|-----------------|
| Covariance matrix, linear dependence | Notebook 03 |
| PCA / eigendecomposition / SVD | Notebook 03 → 04 |
| Cosine similarity, dot products | Notebook 05 (recommendation) |
| Clustering (vector spaces) | Notebook 05 |
| Deployment | Docker / containers (post-model) |