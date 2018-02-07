from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from feature_extractor.Column_Extractor import ColumnExtractor
from process_features.Feature_Extractor import Feature_extractor
from feature_extractor.Categorical_Encoder import Label_Encoder, Multi_Label_Binarizer


class Categorical_feature_extractor(Feature_extractor):


    def get_pipeline(self, feat):

        p = Pipeline([('extract', ColumnExtractor(feat)),
                      ('label_encoder', Label_Encoder()),
                      ('one_hot_encoder', OneHotEncoder())])
        return (feat, p)


class Multilabel_feature_extractor(Feature_extractor):

    def get_pipeline(self, feat):

        p = Pipeline([('extract', ColumnExtractor(feat)),
                      ('multilabel_encoder', Multi_Label_Binarizer()),
                      ('one_hot_encoder', OneHotEncoder())])
        return (feat, p)