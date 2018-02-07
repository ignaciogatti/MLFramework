from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from abc import ABC, abstractclassmethod

class Label_Encoder(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.lb = LabelEncoder()

    def transform(self, X, y=None, **fit_params):
        X_labeled = self.lb.transform(X)
        X_labeled = X_labeled.reshape(-1, 1)
        return X_labeled

    def fit(self, X, y=None):
        self.lb.fit(X)
        return self


class Multi_Label_Binarizer(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def transform(self, X, y=None,**fit_params):
        X_labeled = self.mlb.transform(X)
        #reshape X_labeled
        return X_labeled

    def fit(self, X, y=None):
        self.mlb.fit(X)