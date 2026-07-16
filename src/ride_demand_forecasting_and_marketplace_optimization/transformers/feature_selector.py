from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
    f_regression,
)

class RDFeatureSelector:

    def __init__(self, k=200):
        self.k = k
        self.selector = None

    def fit_transform(self , X , y):
        actual_k = min(self.k, X.shape[1])

        self.selector = SelectKBest(score_func=mutual_info_regression, k=actual_k)
        
        return self.selector.fit_transform(X, y)

    def transform(self, X):

        return self.selector.transform(X)