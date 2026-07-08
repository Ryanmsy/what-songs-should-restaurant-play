"""
Builds the single deployable recommender artifact consumed by the API.

Reproduces the pipeline from notebooks/05b_recommender.ipynb (dedupe songs,
anchor-grounded W, hub-penalty) as a plain script so it can run outside
Jupyter. Re-run this whenever the underlying model changes; the API only
ever loads the output file, never the notebook.
"""
import os
import re
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "..", "data", "processed")
OUTPUT_PATH = os.path.join(HERE, "recommender_artifact.pkl")

MIN_ANCHOR_SONGS = 5
HUB_THRESHOLD_PCT = 0.05

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness", "tempo",
]
SKEWED = ["instrumentalness", "acousticness", "speechiness", "liveness"]

YELP_FEATURES = [
    "Ambience.romantic", "Ambience.divey", "Ambience.classy",
    "Ambience.hipster", "Ambience.trendy", "Ambience.upscale", "Ambience.casual",
    "HasTV", "HappyHour", "RestaurantsGoodForGroups",
    "GoodForMeal.breakfast", "GoodForMeal.brunch",
    "GoodForMeal.latenight", "GoodForMeal.dinner",
    "RestaurantsTableService", "NoiseLevel", "stars",
]

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

SPOTIFY_PC_LABELS = {
    "pc1": "energetic + loud + danceable",
    "pc2": "happy + danceable (acoustic-leaning)",
    "pc3": "live performance + speechy",
    "pc4": "fast tempo",
    "pc5": "spoken word / pure instrumental",
    "pc6": "live instrumental / jazz-like",
}

# Anchor artists per Yelp PC (see notebooks/05b_recommender.ipynb Step 5b for the
# relabeling rationale - pc1/pc4/pc6/pc7 are placeholder lists, review/replace).
ANCHOR_ARTISTS = {
    "pc1": ["Bruno Mars", "John Legend", "Michael Buble", "Jason Mraz", "Andra Day",
            "Alicia Keys", "Adele", "Ben Rector", "Colbie Caillat", "Gavin DeGraw"],
    "pc2": ["Leon Bridges", "Norah Jones", "Jack Johnson", "John Mayer", "The Lumineers",
            "Maggie Rogers", "Vance Joy", "Kacey Musgraves", "Fleet Foxes", "Hozier"],
    "pc3": ["Miles Davis", "John Coltrane", "Frank Sinatra", "Ella Fitzgerald", "Bill Evans",
            "Sade", "Diana Krall", "Nat King Cole", "Stan Getz", "Nina Simone"],
    "pc4": ["AC/DC", "Guns N' Roses", "Red Hot Chili Peppers", "Kings of Leon",
            "The White Stripes", "Foo Fighters", "Weezer", "blink-182", "Fall Out Boy", "Panic! At The Disco"],
    "pc5": ["Tame Impala", "Khruangbin", "Thundercat", "Vampire Weekend", "Mac DeMarco",
            "Arctic Monkeys", "Phoebe Bridgers", "LCD Soundsystem", "The Strokes", "Blood Orange"],
    "pc6": ["Tom Waits", "The Black Keys", "Alabama Shakes", "Gary Clark Jr.", "ZZ Top",
            "Jack White", "The Rolling Stones", "Bob Seger", "Creedence Clearwater Revival", "Whiskey Myers"],
    "pc7": ["FKA twigs", "Billie Eilish", "Frank Ocean", "James Blake", "Rhye",
            "Bon Iver", "Daniel Caesar", "SZA", "Jorja Smith", "Lianne La Havas"],
    "pc8": ["Daft Punk", "Disclosure", "Calvin Harris", "The Weeknd", "Dua Lipa",
            "Fred again..", "RÜFÜS DU SOL", "KAYTRANADA", "Jamie xx", "Peggy Gou"],
}


def load_base_data():
    songs = pd.read_csv(os.path.join(PROCESSED, "song_pca.csv"))
    restaurants = pd.read_csv(os.path.join(PROCESSED, "restaurant_pca.csv"))
    pca_spotify = joblib.load(os.path.join(PROCESSED, "spotify_pca.pkl"))
    scaler_spotify = joblib.load(os.path.join(PROCESSED, "spotify_scaler.pkl"))
    pca_yelp = joblib.load(os.path.join(PROCESSED, "yelp_pca.pkl"))
    return songs, restaurants, pca_spotify, scaler_spotify, pca_yelp


def dedupe_songs(songs):
    before = len(songs)
    songs = songs.drop_duplicates(subset=["name", "artists"]).reset_index(drop=True)
    print(f"Deduped songs: dropped {before - len(songs)} rows ({(before - len(songs)) / before:.1%})")
    return songs


def build_audio_to_spotify_pc(audio_features, scaler_spotify, pca_spotify):
    def audio_to_spotify_pc(audio_dict):
        row = pd.DataFrame([[audio_dict[f] for f in audio_features]], columns=audio_features)
        row[SKEWED] = row[SKEWED].apply(np.log1p)
        scaled = scaler_spotify.transform(row)
        return pca_spotify.transform(scaled)[0]
    return audio_to_spotify_pc


def build_neutral_profile(audio_features, scaler_spotify):
    mean_raw = scaler_spotify.mean_.copy()
    for f in SKEWED:
        i = audio_features.index(f)
        mean_raw[i] = np.expm1(mean_raw[i])
    return dict(zip(audio_features, mean_raw))


def build_audio_intents(neutral):
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


def build_anchor_targets(songs, song_vecs, archetype_pcs):
    y_train_anchor = {}
    for pc in archetype_pcs:
        artists = ANCHOR_ARTISTS[pc]
        mask = pd.Series(False, index=songs.index)
        zero_hits = []
        for artist in artists:
            m = songs["artists"].str.contains(re.escape(artist), case=False, na=False)
            if m.sum() == 0:
                zero_hits.append(artist)
            mask |= m
        n_matched = int(mask.sum())
        print(f"  {pc}: {n_matched} anchor songs matched" + (f"  (no matches: {zero_hits})" if zero_hits else ""))
        y_train_anchor[pc] = song_vecs[mask.values].mean(axis=0) if n_matched >= MIN_ANCHOR_SONGS else None
    return y_train_anchor


def build_W(archetype_pcs, yelp_pc_cols, n_yelp, n_spotify, y_train_anchor, audio_intents, audio_to_spotify_pc):
    n = len(archetype_pcs)
    X_train = np.zeros((n, n_yelp))
    Y_train = np.zeros((n, n_spotify))
    for row_i, pc in enumerate(archetype_pcs):
        col_i = yelp_pc_cols.index(pc)
        X_train[row_i, col_i] = 1.0
        if y_train_anchor.get(pc) is not None:
            Y_train[row_i] = y_train_anchor[pc]
        else:
            Y_train[row_i] = audio_to_spotify_pc(audio_intents[pc])
            print(f"  {pc}: used hand-authored fallback (insufficient anchor coverage)")

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, Y_train)
    return model.coef_


def build_hub_penalty(restaurant_vecs, W, scaler_song, song_vecs_scaled, n_songs):
    restaurant_vecs_normalized = restaurant_vecs / (np.linalg.norm(restaurant_vecs, axis=1, keepdims=True) + 1e-9)
    s_hat_all = restaurant_vecs_normalized @ W.T
    s_hat_all_scaled = scaler_song.transform(s_hat_all)

    nn = NearestNeighbors(n_neighbors=10, algorithm="auto", metric="euclidean")
    nn.fit(song_vecs_scaled)
    _, top10_idx = nn.kneighbors(s_hat_all_scaled)

    n_restaurants = len(restaurant_vecs)
    hub_threshold = HUB_THRESHOLD_PCT * n_restaurants
    song_in_degree = Counter(top10_idx.flatten())

    in_degree_arr = np.zeros(n_songs)
    for song_idx, count in song_in_degree.items():
        in_degree_arr[song_idx] = count

    penalty = np.ones(n_songs)
    over = in_degree_arr > hub_threshold
    penalty[over] = np.sqrt(in_degree_arr[over] / hub_threshold)

    print(f"Hub detection: {int(over.sum())} songs flagged as hubs (threshold={hub_threshold:.0f} restaurants)")
    return penalty, hub_threshold, in_degree_arr


def main():
    t0 = time.time()

    songs, restaurants, pca_spotify, scaler_spotify, pca_yelp = load_base_data()
    songs = dedupe_songs(songs)

    spotify_pc_cols = [c for c in songs.columns if c.startswith("pc")]
    yelp_pc_cols = [c for c in restaurants.columns if c.startswith("pc")]
    song_vecs = songs[spotify_pc_cols].values
    restaurant_vecs = restaurants[yelp_pc_cols].values
    n_spotify, n_yelp = len(spotify_pc_cols), len(yelp_pc_cols)

    yelp_features = YELP_FEATURES[: pca_yelp.n_features_in_]
    audio_features = AUDIO_FEATURES[: pca_spotify.n_features_in_]

    audio_to_spotify_pc = build_audio_to_spotify_pc(audio_features, scaler_spotify, pca_spotify)
    neutral = build_neutral_profile(audio_features, scaler_spotify)
    audio_intents = build_audio_intents(neutral)

    archetype_pcs = list(YELP_PC_LABELS.keys())  # pc1..pc8
    print("Matching anchor artists against the song catalog...")
    y_train_anchor = build_anchor_targets(songs, song_vecs, archetype_pcs)

    print("Fitting W (anchor-grounded, hand-authored fallback where needed)...")
    W = build_W(archetype_pcs, yelp_pc_cols, n_yelp, n_spotify, y_train_anchor, audio_intents, audio_to_spotify_pc)

    scaler_song = StandardScaler()
    song_vecs_scaled = scaler_song.fit_transform(song_vecs)

    print("Running population-wide hub detection (this queries all restaurants)...")
    penalty, hub_threshold, in_degree_arr = build_hub_penalty(
        restaurant_vecs, W, scaler_song, song_vecs_scaled, len(songs)
    )

    artifact = {
        "W": W,
        "yelp_pc_cols": yelp_pc_cols,
        "spotify_pc_cols": spotify_pc_cols,
        "scaler_song": scaler_song,
        "song_vecs_scaled": song_vecs_scaled,
        "penalty": penalty,
        "hub_threshold": hub_threshold,
        "in_degree": in_degree_arr,
        "songs_meta": songs[["id", "name", "artists"]].reset_index(drop=True),
        "restaurants_meta": restaurants[["business_id", "name"] + yelp_pc_cols].reset_index(drop=True),
        "yelp_pc_labels": YELP_PC_LABELS,
        "spotify_pc_labels": SPOTIFY_PC_LABELS,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    joblib.dump(artifact, OUTPUT_PATH)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nWrote {OUTPUT_PATH} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
