from src.ingestion import run_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

yelp_df, spotify_df = run_pipeline()

class PCAReducer():
    def __init__(self,thresold: float = 0.85):
        self.thresold = thresold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.thresold)


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales the data and applies PCA"""
        
        # 1. Scale the data (using the tool saved in self)
        X_scaled = self.scaler.fit_transform(df)

        # 2. Fit and transform PCA (automatically stops at 85% variance!)
        final_array = self.pca.fit_transform(X_scaled)
        
        # 3. Print out how many components it took to reach 85%
        print(f"Number of components kept: {self.pca.n_components_}")

        # 4. Return as a clean DataFrame
        columns = [f"PC{i+1}" for i in range(self.pca.n_components_)]
        return pd.DataFrame(final_array, columns=columns)