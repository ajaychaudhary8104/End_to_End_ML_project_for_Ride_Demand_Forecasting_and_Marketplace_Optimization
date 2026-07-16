from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
import numpy as np


class FrequencyEncoder(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.frequency_maps = {}

    def fit(self,X,y=None):

        for col in X.columns:

            self.frequency_maps[col] = (X[col].value_counts(normalize=True).to_dict())

        return self

    def transform(self, X):

        X = X.copy()

        for col in X.columns:

            X[col] = (X[col].map(self.frequency_maps[col]).fillna(0))

        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(list(self.frequency_maps.keys()))

        return np.array(input_features)