from src.ingestion import run_pipeline, Spotify, Yelp
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROCESSED_DIR = os.path.join("data", "processed")


class PCAReducer():
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.threshold)

    def fit_transform(self, df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
        """Scales and PCA-transforms the feature columns, then re-attaches
        id_cols so each PCA row can still be traced back to a real
        song/restaurant."""

        identifiers = df[id_cols].reset_index(drop=True)
        features = df.drop(columns=id_cols)

        # 1. Scale the data (using the tool saved in self)
        X_scaled = self.scaler.fit_transform(features)

        # 2. Fit and transform PCA (automatically stops at the variance threshold)
        final_array = self.pca.fit_transform(X_scaled)

        # 3. Print out how many components it took to reach the threshold
        print(f"Number of components kept: {self.pca.n_components_}")

        # 4. Return identifiers + PCA columns as a clean DataFrame
        columns = [f"pc{i+1}" for i in range(self.pca.n_components_)]
        pca_df = pd.DataFrame(final_array, columns=columns)
        return pd.concat([identifiers, pca_df], axis=1)

    def save_model(self, filepath: str):
        """Saves the entire trained tool (scaler + pca) to disk."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")


def run_pca():
    """Runs ingestion, fits PCA on both datasets, and saves the
    transformed data + fitted scaler/PCA objects to data/processed/."""

    yelp_df, spotify_df = run_pipeline()
    if yelp_df is None or spotify_df is None:
        raise ValueError("Ingestion pipeline returned no data.")

    song_reducer = PCAReducer()
    song_pca_df = song_reducer.fit_transform(spotify_df, id_cols=Spotify.ID_COLS)
    song_pca_df.to_csv(os.path.join(PROCESSED_DIR, "song_pca.csv"), index=False)
    joblib.dump(song_reducer.scaler, os.path.join(PROCESSED_DIR, "spotify_scaler.pkl"))
    joblib.dump(song_reducer.pca, os.path.join(PROCESSED_DIR, "spotify_pca.pkl"))
    print(f"song_pca.csv         {song_pca_df.shape}")

    restaurant_reducer = PCAReducer()
    restaurant_pca_df = restaurant_reducer.fit_transform(yelp_df, id_cols=Yelp.ID_COLS)
    restaurant_pca_df.to_csv(os.path.join(PROCESSED_DIR, "restaurant_pca.csv"), index=False)
    joblib.dump(restaurant_reducer.scaler, os.path.join(PROCESSED_DIR, "yelp_scaler.pkl"))
    joblib.dump(restaurant_reducer.pca, os.path.join(PROCESSED_DIR, "yelp_pca.pkl"))
    print(f"restaurant_pca.csv   {restaurant_pca_df.shape}")

    return song_pca_df, restaurant_pca_df


if __name__ == "__main__":
    run_pca()
