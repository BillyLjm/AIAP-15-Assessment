import datapipeline
import transformers

import scipy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, pre_path: str, post_path: str):
        """
        Initialises the model and specifies the datasets that will be used for
        training and testing.

        :param pre_path: filepath to the pre-purchase survey result database
        :param post_path: filepath to the post-trip survey result database
        """
        # split into train test
        dpl = datapipeline.Datapipeline()
        X, y = dpl.transform_data(pre_path, post_path)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, stratify=y, test_size=0.4, random_state=1)

        # encode multi-class labels
        self.labenc = LabelEncoder()
        self.labenc.fit(y)

        # preprocess features
        pent_cols = ['Onboard Wifi Service', # columns to pca
            'Embarkation/Disembarkation time convenient',
            'Ease of Online booking', 'Gate location', 'Onboard Dining Service',
            'Online Check-in', 'Cabin Comfort', 'Onboard Entertainment',
            'Cabin service', 'Baggage handling', 'Port Check-in Service',
            'Onboard Service', 'Cleanliness']
        self.preprocess = Pipeline([
            ('dropper', ColumnTransformer([
                ('drop', 'drop', ['Logging', 'WiFi', 'Dining', 'Entertainment']),
                ('year', transformers.GetYear(), ['Date of Birth'])
            ], remainder='passthrough')),
            ('imputer', ColumnTransformer([
                ('imputeMean', SimpleImputer(strategy='mean'),
                    make_column_selector(dtype_include=np.number)),
                ('imputeFreq', SimpleImputer(strategy='most_frequent'),
                    make_column_selector(dtype_exclude=np.number)),
            ], remainder='passthrough')),
            ('encode', ColumnTransformer([
                ('ohe', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False),
                    make_column_selector(dtype_include='object')),
                ('pca', PCA(n_components = 5),
                    ['imputeMean__remainder__' + i for i in pent_cols]),
            ], remainder='passthrough'))
        ])
        self.preprocess.set_output(transform='pandas')

        # set ML model
        self.model = RandomForestClassifier()

    def train(self):
        """
        Trains the model of the training dataset, using grid search

        :returns: tuple of the best (hyperparameters, f1_macro score)
        """
        # grid search cv
        parameters = {
            'n_estimators': [10, 50, 100],
            'max_features': ['sqrt'],
            'max_depth' : np.arange(40, 91, 10),
            'criterion' :['gini']
        }
        clf = GridSearchCV(self.model, parameters,
            scoring='f1_macro', cv=StratifiedKFold(5),
            n_jobs=-1)

        # train preprocessing pipeline.
        self.preprocess.fit(self.X_train)

        # remember best grid search
        clf.fit(self.preprocess.transform(self.X_train),
            self.labenc.transform(self.y_train))
        self.model = clf.best_estimator_

        return clf.best_params_, clf.best_score_

    def test(self):
        """
        Description of the function.

        :return: mean squared error of model on test dataset
        """
        y_pred = self.labenc.inverse_transform(
            self.model.predict(
                self.preprocess.transform(self.X_test)
        ))
        return f1_score(y_pred, self.y_test, average='macro')

    def predict(self, X):
        """
        Description of the function.

        :param X: features to predict for
        :return: list of corresponding predictions
        """
        # This should use the trained model to predict the target for the test_data and return the test mse
        return self.labenc.inverse_transform(
            self.model.predict(
                self.preprocess.transform(X)
        ))

if __name__ == "__main__":
    print("---------------\n Random Forest \n---------------")
    clf = Model("data/cruise_pre.db", "data/cruise_post.db")

    params, score_train = clf.train()
    print(f"Hyperparameters used are {params}")

    score_test = clf.test()
    print(f"The test F1-macro score is {score_test}")