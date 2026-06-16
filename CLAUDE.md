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
- Python, Jupyter notebooks
- pandas, numpy, matplotlib, seaborn

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
- `yelp_clean.csv`: no true NaNs, but `NoiseLevel`, `Alcohol`, `RestaurantsAttire` use `"unknown"` as a missing sentinel
- `yelp_clean.csv`: binary columns stored as float (0.0/1.0) — cast to int for ML
- `yelp_clean.csv` Yelp data: 52,268 restaurants, 23 columns

## Folder Structure
```
restaurant_recommendation/
├── data/
│   ├── raw/              ← original files, never modify
│   └── processed/        ← cleaned outputs from notebooks
├── notebooks/            ← numbered in pipeline order
├── notes/                ← learning docs, writeups
└── CLAUDE.md
```

## Notebooks
- `notebooks/01_spotify_eda.ipynb` — EDA and cleaning → `data/processed/tracks_features_clean.csv`
- `notebooks/02_yelp_eda.ipynb` — EDA and cleaning → `data/processed/yelp_ml_ready.csv`
- `notebooks/03_covariance_pca.ipynb` — Covariance matrix (linear algebra), redundancy analysis, eigendecomposition preview

## End Goal (Linear Algebra → ML showcase)
| Concept | Where it appears |
|---------|-----------------|
| Covariance matrix, linear dependence | Notebook 03 |
| PCA / eigendecomposition / SVD | Notebook 03 → 04 |
| Cosine similarity, dot products | Notebook 05 (recommendation) |
| Clustering (vector spaces) | Notebook 05 |
| Deployment | Docker / containers (post-model) |