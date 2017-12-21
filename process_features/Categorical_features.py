from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from feature_extractor.Column_Extractor import ColumnExtractor
from process_features.Feature_Extractor import Feature_extractor


class Categorical_features_extractor(Feature_extractor):


    def get_pipeline(self, feat):

        p = Pipeline(['extract', ColumnExtractor(feat),
                      'label_encoder', LabelEncoder(),
                      'one hot encoder', OneHotEncoder()])
        return p


class Multilabel_feature_extractor(Feature_extractor):

    def get_pipeline(self, feat):

        p = Pipeline(['extract', ColumnExtractor(feat),
                      'multilabel_encoder', MultiLabelBinarizer(),
                      'one hot encoder', OneHotEncoder()])
        return p