import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GetYear(BaseEstimator, TransformerMixin):
    """sklearn transformer to get year of datetime column"""
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].dt.year.astype('Int64')
        return X
        
    def get_feature_names_out(self):
        return self.columns_