from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import numpy as np 
from typing import Optional
from abc import ABC, abstractmethod

class Ingestion(ABC):

    def __init__(self, dataset_name: str, filepath: str):
        self.dataset_name = dataset_name 
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    # --- SHARED METHODS (Every dataset does these exactly the same way) ---
    def import_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df 

    def cleaning(self):
        if self.df is None:
            raise ValueError("DataFrame is empty. Call import_data() first!")
        self.df = self.df.drop_duplicates(keep='first')
        self.df = self.df.dropna(how='all')
        return self.df  

    def clean_unknown_values(self):
        if self.df is None:
            raise ValueError("DataFrame is empty. Call import_data() first!")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].replace('unknown', float('nan'))
        return self.df

    def validate(self):
        if self.df is None:
            raise ValueError('df is empty')
        # Drop any remaining NaNs at the very end of the pipeline
        self.df = self.df.dropna() 
        return self.df 

    # --- THE BLANK SPACE (Every dataset must fill this in with its unique steps) ---
    @abstractmethod
    def apply_custom_transformations(self):
        """Override this method in subclasses to apply dataset-specific cleaning."""
        pass

    # --- THE TEMPLATE (The single boss that controls the order of operations) ---
    def run(self):
        print(f"Starting pipeline for {self.dataset_name}...")
        self.import_data()
        self.cleaning()
        self.clean_unknown_values()
        
        # This calls the unique steps for whichever class is currently running!
        self.apply_custom_transformations() 
        
        self.validate()
        print(f"Finished pipeline for {self.dataset_name}!")
        return self.df


class Spotify(Ingestion):

    ID_COLS = ['id', 'name', 'artists']

    def specific_missing_value(self):
        before = len(self.df)
        self.df = self.df[self.df['tempo'] != 0]
        self.df = self.df.drop_duplicates(subset='id', keep='first')
        print(f"  Spotify: Dropped {before - len(self.df)} rows with 0 tempo.")
        return self.df

    def filter_year(self):
        self.df = self.df[self.df['year'].between(2000, 2020)]
        return self.df

    def filter_audio_features(self):
        AUDIO_FEATURES = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo'
        ]
        self.df = self.df[self.ID_COLS + AUDIO_FEATURES]
        return self.df

    def apply_log(self):
        SKEWED = ['instrumentalness', 'acousticness', 'speechiness', 'liveness']
        self.df[SKEWED] = self.df[SKEWED].apply(np.log1p)
        return self.df

    # Here we fill in the blank required by the abstract method!
    def apply_custom_transformations(self):
        self.filter_year()
        self.specific_missing_value()
        self.filter_audio_features()
        self.apply_log()


class Yelp(Ingestion):

    ID_COLS = ['business_id', 'name']
    BINARY_COLS = [
        'Ambience.romantic', 'Ambience.divey', 'Ambience.classy', 'Ambience.hipster',
        'Ambience.trendy', 'Ambience.upscale', 'Ambience.casual',
        'HasTV', 'RestaurantsGoodForGroups', 'HappyHour',
        'GoodForMeal.breakfast', 'GoodForMeal.brunch', 'GoodForMeal.latenight', 'GoodForMeal.dinner',
        'RestaurantsTableService'
    ]

    def specific_missing_value(self):
        self.df = self.df.drop_duplicates(subset='business_id', keep='first')
        return self.df

    def numerical_outlier_values(self):
        numerical_cols = [c for c in self.df.select_dtypes(include=['int64', 'float64']).columns if c not in self.BINARY_COLS]
        for col in numerical_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[(self.df[col] >= Q1 - 1.5 * IQR) & (self.df[col] <= Q3 + 1.5 * IQR)]
        return self.df

    def categorical_encoding(self):
        if 'NoiseLevel' in self.df.columns:
            NOISE_ORDER = {'quiet': 0, 'average': 1, 'loud': 2, 'very_loud': 3}
            self.df['NoiseLevel'] = self.df['NoiseLevel'].map(NOISE_ORDER)

        cols_to_encode = [col for col in ['Alcohol', 'RestaurantsAttire'] if col in self.df.columns]
        if cols_to_encode:
            encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
            columns_dummies = encoder.fit_transform(self.df[cols_to_encode])
            self.df = pd.concat([self.df.drop(columns=cols_to_encode), columns_dummies], axis=1)
        return self.df

    def boolean_switch(self):
        bool_cols = self.df.select_dtypes(include='bool').columns.tolist()
        self.df[bool_cols] = self.df[bool_cols].astype(int)
        self.df[self.BINARY_COLS] = self.df[self.BINARY_COLS].astype(int)
        return self.df

    def filter_continous(self):
        YELP_CANDIDATES = [
        'Ambience.romantic', 'Ambience.divey', 'Ambience.classy',
        'Ambience.hipster', 'Ambience.trendy', 'Ambience.upscale', 'Ambience.casual',
        'HasTV', 'HappyHour', 'RestaurantsGoodForGroups',
        'GoodForMeal.breakfast', 'GoodForMeal.brunch',
        'GoodForMeal.latenight', 'GoodForMeal.dinner',
        'RestaurantsTableService', 'NoiseLevel', 'stars'
        ]
        YELP_FEATURES = [c for c in YELP_CANDIDATES if c in self.df.columns]
        self.df = self.df[self.ID_COLS + YELP_FEATURES].dropna()
        return self.df

    def drop_constant_columns(self):
        variance = self.df.var(numeric_only=True)
        low_var_cols = variance[variance < 0.01].index.tolist()
        if low_var_cols:
            self.df = self.df.drop(columns=low_var_cols)
        return self.df

    # Here we fill in the blank required by the abstract method!
    def apply_custom_transformations(self):
        self.specific_missing_value()
        self.numerical_outlier_values()
        self.categorical_encoding()
        self.boolean_switch()
        self.filter_continous()
        self.drop_constant_columns()


def run_pipeline():
    # Initialize and run Yelp Pipeline
    yelp_pipeline = Yelp(dataset_name="Yelp Reviews", filepath="data/raw/yelp_clean.csv")
    yelp_df = yelp_pipeline.run()

    # Initialize and run Spotify Pipeline
    spotify_pipeline = Spotify(dataset_name="Spotify Tracks", filepath="data/raw/tracks_features.csv")
    spotify_df = spotify_pipeline.run()

    return yelp_df, spotify_df

if __name__ == "__main__":
    yelp_data, spotify_data = run_pipeline()