import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class ColumnExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, cols):
        self.cols = cols


    def transform(self, X, **transform_params):
        X_categorical = X[self.cols]
        X_categorical = X_categorical.fillna(method='ffill')
        return X_categorical.values


    def fit(self, X, y=None, **fit_params):
        return self
