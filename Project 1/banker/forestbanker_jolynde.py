from banker import BankerBase , run
from random import choice
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV


class RandomForestClassifier(BankerBase):
    model = None

    def __init__(self):
        self.interest_rate = None

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y

    def parse_X(self, X):
        return X

    def build_forest(self, X, y):
        #model = RandomForestClassifier(n_estimators=130)

        # we optimize the model on the dataset, but it does so in the training so it will do this for every new dataset it
        # is trained on (https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
        param_grid = {
            'n_estimators': np.linspace(10, 200).astype(int),
            'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }

        # Estimator for use in random search
        estimator = RandomForestClassifier()

        # Create the random search model
        model = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                                scoring = 'roc_auc', cv = 3,
                                n_iter = 10, verbose = 1)
        #print(model.best_params_)
        return model

    def get_proba(self, X):
        return self.model.predict_proba(np.array(X).reshape(1,-1))[:,1]

    def expected_utility(self, X):
        p = self.get_proba(self.parse_X(X))
        gain = self.calculate_gain(X)
        expected_utilitiy = (gain*p.flatten())-(X['amount']*(1-p.flatten()))
        return expected_utilitiy

    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1).ravel())
        X = self.parse_X(X)
        self.model = self.build_forest(X, y)
        self.model.fit(X,y)
        self.model.best_estimator_
        print(self.model.best_estimator_)

    def get_best_params(self):
        print(self.model.best_params_)
        return self.model.best_params_

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        return actions

    def predict(self,Xtest):
        return self.model.predict(Xtest)

    def predict_proba(self, Xtest):
        return self.model.predict_proba(Xtest)

    def get_importances(self, X):
        importance = list(zip(X, self.model.feature_importances_))
        print(importance)
        return importance

if __name__ == '__main__':
    run()
