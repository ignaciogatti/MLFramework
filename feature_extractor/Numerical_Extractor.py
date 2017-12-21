import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from scipy import sparse
from scipy.sparse import save_npz


class NumericalFeatureExtraction( TransformerMixin, BaseEstimator):


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numerical_sparsed = sparse.csr_matrix(X)
        save_npz('numerical_features.npz', X_numerical_sparsed)
        return X_numerical_sparsed


    def get_params(self, deep=True):
        return {'col_to_extract': self.col_extract}

    def get_transformer(self):
        return {}
