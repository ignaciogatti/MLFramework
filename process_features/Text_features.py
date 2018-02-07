from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extractor.Column_Extractor import ColumnExtractor
from process_features.Feature_Extractor import Feature_extractor

class Text_features_extractor(Feature_extractor):

    def get_pipeline(self, feat):

        p = Pipeline([('extract', ColumnExtractor(feat)),
                      ('tfidf', TfidfVectorizer())])
        return (feat ,p)
