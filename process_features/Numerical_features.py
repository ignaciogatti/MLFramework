from  sklearn.pipeline import Pipeline
from feature_extractor.Numerical_Extractor import NumericalFeatureExtraction
from feature_extractor.Column_Extractor import ColumnExtractor
from process_features.Feature_Extractor import Feature_extractor

class Numerical_features_extractor(Feature_extractor):

    def get_pipeline(self, feat):

        p = Pipeline(['extract', ColumnExtractor(feat),
                      'numerical_feature', NumericalFeatureExtraction()])
        return p
