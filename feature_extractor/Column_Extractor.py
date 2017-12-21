import numpy as np
import pandas as pd

class ColumnExtractor(object):

    def __init__(self, cols):
        self.cols = cols


    def transform(self, X):
        X_categorical = X[self.cols]
        X_categorical = X_categorical.fillna(method='ffill')
        return X_categorical.values


    def fit(self, X, y=None):
        return self