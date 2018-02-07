import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from process_features.Categorical_features import Categorical_feature_extractor
from process_features.Process_features import Process_feature_union
from sklearn.ensemble import RandomForestClassifier
from feature_extractor.Column_Extractor import ColumnExtractor
from feature_extractor.Categorical_Encoder import Label_Encoder

columns_to_keep =['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked']

df_train = pd.read_csv('/home/ignacio/PycharmProjects/MLFramework/train.csv')

y = df_train['Survived']
X = df_train[columns_to_keep]
print('Full dataset')
print(X.shape)


cat_feat_extraction = Categorical_feature_extractor(features=['Pclass', 'Sex'])

cat_transformers = cat_feat_extraction.get_transformers()


features_process = Process_feature_union(features=cat_transformers)

features = features_process.get_feature_union()


titanic_pipeline = Pipeline([
     features,
    ('estimator', RandomForestClassifier())
])


