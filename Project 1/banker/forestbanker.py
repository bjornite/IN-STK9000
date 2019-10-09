from banker import BankerBase , run
from random import choice
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class RandomForestClassifierBanker(BankerBase):
    model = None

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y

    def parse_X(self, X):
        return X

    def build_forest(self, X, y):
        model = RandomForestClassifier(n_estimators=130)
        return model

    def get_proba(self, X):
        ##print("proba")
        return self.model.predict_proba(np.array(X).reshape(1,-1))[:,1]

    def expected_utility(self, X):
        p = self.get_proba(self.parse_X(X))
        gain = self.calculate_gain(X)
        ##print("Gain")
        #print(gain)
        ##print("Gain x p")
        ##print(gain*p)
        ##print("perte seche")
        ##print(X['amount'])
        ##print("argent a perdre")
        ##print(X['amount']*(1-p.flatten()))
        expected_utilitiy = (gain*p.flatten())-(X['amount']*(1-p.flatten()))
        ##print("expected_utilitiy")
        #print(expected_utilitiy)
        return expected_utilitiy

    def calculate_gain(self, X):
        ##print("calculate_gain")
        ##print(X['amount']*((1 + self.interest_rate)**(X['duration']) - 1))
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1).ravel())
        X = self.parse_X(X)
        self.model = self.build_forest(X, y)
        self.model.fit(X,y)

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        ##print('action')
        ##print(actions)

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
