from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import numpy as np 
from typing import Optional


class Ingestion:

    def __init__(self,dataset_name: str, filepath: str):
        self.dataset_name= dataset_name 
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None


    # Data ingestion
    def import_data(self):
        """ import data"""
        self.df = pd.read_csv(self.filepath)
        return self.df 

    def cleaning(self):
        """Drop all duplicates and empty rows."""
        if self.df is None:
            raise ValueError("DataFrame is empty. Call import_data() first!")
            
        self.df = self.df.drop_duplicates(keep='first')
        self.df = self.df.dropna(how='all')
        return self.df  

    def clean_unknown_values(self):
        """Replace 'unknown' strings with np.nan in categorical columns."""
        if self.df is None:
            raise ValueError("DataFrame is empty. Call import_data() first!")
            
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].replace('unknown', float('nan'))
        return self.df

    def run(self):
        self.import_data()
        self.cleaning()
        self.clean_unknown_values()
        return self.df

    

class Spotify(Ingestion):

    def specific_missing_value(self):
        """remove any songs where tempo is 0."""
        before = len(self.df)
        self.df = self.df[self.df['tempo'] != 0]

        self.df = self.df.drop_duplicates(subset='id', keep='first')

        print(f"Cleaned! Rows dropped: {before - len(self.df)}")
        return self.df

    def run(self):
        super().run()
        self.specific_missing_value()
        return self.df

class Yelp(Ingestion):

    # Shared with numerical_outlier_values() so IQR filtering never runs on these —
    # they're 0/1 flags, not continuous values, and IQR bounds collapse to a single
    # point on skewed binary data, silently dropping every minority-class row.
    BINARY_COLS = [
        'Ambience.romantic', 'Ambience.divey', 'Ambience.classy', 'Ambience.hipster',
        'Ambience.trendy', 'Ambience.upscale', 'Ambience.casual',
        'HasTV', 'RestaurantsGoodForGroups', 'HappyHour',
        'GoodForMeal.breakfast', 'GoodForMeal.brunch', 'GoodForMeal.latenight', 'GoodForMeal.dinner',
        'RestaurantsTableService'
    ]

    def specific_missing_value(self):
        """Drop duplicate restaurants by business_id."""
        self.df = self.df.drop_duplicates(subset='business_id', keep='first')
        return self.df

    def numerical_outlier_values(self):
        """Remove numerical outliers using the IQR method (continuous columns only)."""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [c for c in numerical_cols if c not in self.BINARY_COLS]

        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        return self.df

    def categorical_encoding(self):
        """ If columns == categorical then do categorical encoding
            If columns less than 2 then binary encoding
            if columns > 2 & < 5 then one hot encoding
            else stop process, ask human """

        # Noise
        if 'NoiseLevel' in self.df.columns:
            NOISE_ORDER = {'quiet': 0, 'average': 1, 'loud': 2, 'very_loud': 3}
            self.df['NoiseLevel'] = self.df['NoiseLevel'].map(NOISE_ORDER)

        cols_to_encode = ['Alcohol', 'RestaurantsAttire']

        # Ensure columns exist before trying to encode them
        existing_cols = [col for col in cols_to_encode if col in self.df.columns]

        if existing_cols:
            encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
            columns_dummies = encoder.fit_transform(self.df[existing_cols])

            # Drop the original columns and concatenate the new dummy columns
            self.df = self.df.drop(columns=existing_cols)
            self.df = pd.concat([self.df, columns_dummies], axis=1)

        return self.df

    def boolean_switch(self):
        bool_cols = self.df.select_dtypes(include='bool').columns.tolist()
        self.df[bool_cols] = self.df[bool_cols].astype(int)

        self.df[self.BINARY_COLS] = self.df[self.BINARY_COLS].astype(int)
        return self.df

    def run(self):
        super().run()
        self.specific_missing_value()
        self.numerical_outlier_values()
        self.categorical_encoding()
        self.boolean_switch()
        print(" Yelp pipeline complete!")
        return self.df


def run_pipeline():
    """Main execution entry point."""

    # Initialize and run Yelp Pipeline
    yelp_pipeline = Yelp(dataset_name="Yelp Reviews", filepath="data/raw/yelp_clean.csv")
    yelp_df = yelp_pipeline.run()

    # Initialize and run Spotify Pipeline
    spotify_pipeline = Spotify(dataset_name="Spotify Tracks", filepath="data/raw/tracks_features.csv")
    spotify_df = spotify_pipeline.run()

    return yelp_df, spotify_df


if __name__ == "__main__":
    # yelp_data, spotify_data = run_pipeline()
    pass