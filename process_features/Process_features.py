from sklearn.pipeline import FeatureUnion

class Process_feature_union:

    def __init__(self, features):
        self.features = features


    def get_feature_union(self):

        fu = FeatureUnion(self.features)
        return ('features', fu)