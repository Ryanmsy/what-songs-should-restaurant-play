import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join("data", "processed")

AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo'
]

# Same skewed columns log1p'd in Spotify.apply_log() (src/ingestion.py) — an
# archetype's raw feature values have to go through the identical transform
# before they land in the same PCA space as the real songs.
SKEWED = ['instrumentalness', 'acousticness', 'speechiness', 'liveness']

# Hand-authored per Yelp PC (see notebooks/05b_recommender.ipynb) — a target
# audio-feature "vibe" for each restaurant archetype. Deviations are applied on
# top of the dataset's average song; unset features keep the average value.
# Placeholder for pc1/pc4/pc6/pc7 (never anchor-artist-validated — see CLAUDE.md).
YELP_PC_LABELS = {
    "pc1": "sit-down dinner (table service + casual + happy hour)",
    "pc2": "brunch / breakfast",
    "pc3": "fine dining (upscale + classy + romantic, not casual)",
    "pc4": "loud / chaotic (high noise, lower stars)",
    "pc5": "hipster / trendy",
    "pc6": "dive bar (very divey, some late-night overlap)",
    "pc7": "late-night",
    "pc8": "good for groups (divey/hipster-leaning, no table service)",
}


def build_audio_intents(neutral: dict) -> dict:
    """Build one target audio-feature profile per Yelp-PC archetype by
    nudging the dataset-average song ('neutral') toward that vibe."""

    def archetype(**deviations):
        intent = dict(neutral)
        intent.update(deviations)
        return intent

    return {
        "pc1": archetype(acousticness=0.40, valence=0.50, tempo=110.0),
        "pc2": archetype(danceability=0.60, acousticness=0.45, instrumentalness=0.15,
                          liveness=0.15, valence=0.75),
        "pc3": archetype(danceability=0.30, energy=0.25, acousticness=0.65, instrumentalness=0.55,
                          liveness=0.15, loudness=-15.0, tempo=85.0),
        "pc4": archetype(energy=0.80, speechiness=0.12, acousticness=0.15, instrumentalness=0.05,
                          liveness=0.35, loudness=-5.0, tempo=130.0),
        "pc5": archetype(danceability=0.55, liveness=0.25, valence=0.50),
        "pc6": archetype(energy=0.70, acousticness=0.25, instrumentalness=0.15,
                          liveness=0.45, loudness=-7.0),
        "pc7": archetype(danceability=0.40, energy=0.35, acousticness=0.40, instrumentalness=0.40,
                          valence=0.30, loudness=-12.0, tempo=95.0),
        "pc8": archetype(danceability=0.70, energy=0.65, liveness=0.25, valence=0.65, loudness=-8.0),
    }


class Recommender:

    def __init__(self):
        self.songs: Optional[pd.DataFrame] = None
        self.restaurants: Optional[pd.DataFrame] = None
        self.pca_spotify = None
        self.scaler_spotify: Optional[StandardScaler] = None

        self.spotify_pc_cols: list[str] = []
        self.yelp_pc_cols: list[str] = []

        self.W: Optional[np.ndarray] = None
        self.song_scaler = StandardScaler()
        self.song_finder: Optional[NearestNeighbors] = None

    def load_data(self):
        """Load the PCA-transformed songs/restaurants and the fitted Spotify
        scaler/PCA saved by src/pca.py."""
        self.songs = pd.read_csv(os.path.join(PROCESSED_DIR, "song_pca.csv"))
        self.restaurants = pd.read_csv(os.path.join(PROCESSED_DIR, "restaurant_pca.csv"))
        self.scaler_spotify = joblib.load(os.path.join(PROCESSED_DIR, "spotify_scaler.pkl"))
        self.pca_spotify = joblib.load(os.path.join(PROCESSED_DIR, "spotify_pca.pkl"))

        self.spotify_pc_cols = [c for c in self.songs.columns if c.startswith("pc")]
        self.yelp_pc_cols = [c for c in self.restaurants.columns if c.startswith("pc")]
        return self.songs, self.restaurants

    def _neutral_profile(self) -> dict:
        """Reconstruct the 'average song' in raw audio-feature units by
        undoing the scaler's centering (and the log1p on skewed columns)."""
        if self.scaler_spotify is None:
            raise ValueError("No data loaded. Call load_data() first!")

        audio_features = AUDIO_FEATURES[: self.scaler_spotify.n_features_in_]
        mean_raw = self.scaler_spotify.mean_.copy()
        for feature in SKEWED:
            if feature in audio_features:
                i = audio_features.index(feature)
                mean_raw[i] = np.expm1(mean_raw[i])
        return dict(zip(audio_features, mean_raw)), audio_features

    def _audio_to_spotify_pc(self, audio_dict: dict, audio_features: list[str]) -> np.ndarray:
        """Push one hand-authored audio-feature profile through the same
        log1p -> scale -> PCA pipeline notebook 04 used on real songs, so it
        lands as a point in the same Spotify-PC space."""
        row = pd.DataFrame([[audio_dict[f] for f in audio_features]], columns=audio_features)
        row[SKEWED] = row[SKEWED].apply(np.log1p)
        scaled = self.scaler_spotify.transform(row)
        return self.pca_spotify.transform(scaled)[0]

    def build_W(self) -> np.ndarray:
        """Fit W, the linear map from Yelp-PC space to Spotify-PC space.

        Design pattern: LinearRegression used as a coordinate-system
        translator, not a predictor. Each Yelp-PC archetype (e.g. "dive bar")
        becomes one training row: a one-hot vector in Yelp-PC space paired
        with its hand-authored target point in Spotify-PC space.
        fit_intercept=False forces the fit through the origin, matching how
        PCA coordinates are already centered around zero.
        """
        if self.songs is None or self.restaurants is None:
            raise ValueError("No data loaded. Call load_data() first!")

        neutral, audio_features = self._neutral_profile()
        audio_intents = build_audio_intents(neutral)

        archetype_pcs = list(YELP_PC_LABELS.keys())
        n_yelp, n_spotify = len(self.yelp_pc_cols), len(self.spotify_pc_cols)

        X_train = np.zeros((len(archetype_pcs), n_yelp))
        Y_train = np.zeros((len(archetype_pcs), n_spotify))

        for row_i, pc in enumerate(archetype_pcs):
            col_i = self.yelp_pc_cols.index(pc)
            X_train[row_i, col_i] = 1.0
            Y_train[row_i] = self._audio_to_spotify_pc(audio_intents[pc], audio_features)

        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, Y_train)
        self.W = model.coef_
        return self.W

    def fit(self):
        """Load data, fit W, and index the songs for nearest-neighbor search."""
        self.load_data()
        self.build_W()

        song_vecs = self.songs[self.spotify_pc_cols].to_numpy(dtype=float)
        song_vecs_scaled = self.song_scaler.fit_transform(song_vecs)
        self.song_finder = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(song_vecs_scaled)
        return self

    def recommend(self, business_id: str, k: int = 5) -> pd.DataFrame:
        """Translate one restaurant's Yelp-PC vector into its top-k closest
        real songs in Spotify-PC space.

        Concept: cosine-normalizing the restaurant vector before applying W
        keeps only its *direction* (the vibe), discarding magnitude, since W
        was trained on unit one-hot vectors. NearestNeighbors then does a
        k-NN search (Euclidean distance) over real songs' scaled PCA
        coordinates to find the closest actual tracks to that target point.
        """
        if self.W is None or self.song_finder is None or self.restaurants is None:
            raise ValueError("Model not fitted. Call fit() first!")

        match = self.restaurants.loc[self.restaurants["business_id"] == business_id]
        if match.empty:
            raise ValueError(f"No restaurant with business_id={business_id!r}")

        y = match.iloc[0][self.yelp_pc_cols].to_numpy(dtype=float)
        y_norm = y / (np.linalg.norm(y) + 1e-9)
        target = self.song_scaler.transform((self.W @ y_norm).reshape(1, -1))

        distances, idx = self.song_finder.kneighbors(target, n_neighbors=k)
        idx, distances = idx[0], distances[0]

        return pd.DataFrame({
            "song": self.songs.loc[idx, "name"].values,
            "artist": self.songs.loc[idx, "artists"].values,
            "distance": distances.round(3),
        })

    def save_model(self, filepath: str):
        """Saves the entire fitted recommender (W, song scaler, NN index) to disk."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> "Recommender":
        return joblib.load(filepath)


def run_recommender():
    """Fits the recommender and saves it to data/processed/."""
    recommender = Recommender()
    recommender.fit()
    recommender.save_model(os.path.join(PROCESSED_DIR, "recommender.pkl"))
    return recommender


if __name__ == "__main__":
    run_recommender()
