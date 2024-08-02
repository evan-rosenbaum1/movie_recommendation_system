import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original data
        
        # Movie Rating Count
        movie_rating_count = X.groupby('movie_id').size().reset_index(name='movie_rating_count')
        X = X.merge(movie_rating_count, on='movie_id', how='left')

        # Average User Rating
        average_user_rating = X.groupby('user_id')['true_r'].mean().reset_index(name='average_user_rating')
        X = X.merge(average_user_rating, on='user_id', how='left')

        # User Rating Count
        user_rating_count = X.groupby('user_id').size().reset_index(name='user_rating_count')
        X = X.merge(user_rating_count, on='user_id', how='left')

        # User-Movie Rating Count Ratio
        X['user_movie_rating_count_ratio'] = X['user_rating_count'] / (X['movie_rating_count'] + 1)

        return X
