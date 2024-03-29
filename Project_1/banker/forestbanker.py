from banker import BankerBase , run
from random import choice
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier


class RandomForestClassifierBanker(BankerBase):
    model = None

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    #def set_interest_rate(self, interest_rate):
        #self.interest_rate = interest_rate

    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y

    def parse_X(self, X):
        return X

    def build_forest(self, X, y):
        base_cls = RandomForestClassifier()
        model = BaggingClassifier(base_estimator = base_cls,
                                    n_estimators = 130)
        return model

    def get_proba(self, X):
<<<<<<< HEAD
        ##print("proba")
        return self.model.predict_proba(X)[:,1]
=======
        return self.model.predict_proba(np.array(X).reshape(1,-1))[:,1]
>>>>>>> 4e43b88f3a895fb6d956cfc2000ecbcbe3a362a7

    def expected_utility(self, X):
        import numbers
        if isinstance(X.values[0], numbers.Number):
            X_vals = X.values.reshape(1,-1)
        else:
            X_vals = X.values
        p = self.get_proba(self.parse_X(X_vals))
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

    def get_best_action(self, X):
<<<<<<< HEAD
        actions = (self.expected_utility(X) > 0).astype(int)
        actions[actions == 0] = 2
        ##print('action')
        ##print(actions)

=======
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
>>>>>>> 4e43b88f3a895fb6d956cfc2000ecbcbe3a362a7
        return actions

    def predict(self,Xtest):
        return self.model.predict(Xtest)

    def predict_proba(self, Xtest):
        return self.model.predict_proba(Xtest)

    def get_importances(self, X):
        importance = list(zip(X, self.model.feature_importances_))
        return importance
        print(importance)

if __name__ == '__main__':
    run()
