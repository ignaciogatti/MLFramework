import pandas as pd
from process_features.Categorical_features import Categorical_features_extractor


columns_to_keep =['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked']

df_train = pd.read_csv('/home/ignacio/PycharmProjects/MLFramework/train.csv')

y = df_train['Survived']
X = df_train[columns_to_keep]
print('Full dataset')
print(X.shape)

cat_feat_extraction = Categorical_features_extractor(features=['Pclass', 'Sex', 'Embarked'])

cat_feat_extraction.get_transformers()