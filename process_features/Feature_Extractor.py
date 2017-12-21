from abc import ABC, abstractclassmethod

class Feature_extractor(ABC):

    def __init__(self, features):
        self.features = features

    @abstractclassmethod
    def get_pipeline(self, feat):
        pass


    def get_transformers(self):
        transformers = []
        for feat in self.features:
            transformers.append(self.get_pipeline(feat))
        return  transformers



