from imblearn.under_sampling import RandomUnderSampler
from banker import BankerBase, run
from random import choice
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
sys.stderr = stderr
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import BaggingClassifier


class RandomBanker(BankerBase):
    """Example Banker implementation. To implement your own, you need to take
    care of a number of details.

    1. Your class should inherit from BankerBase in `banker.py`. If so, it
       will be automatically discovered and scored when you call the run
       function from the same file.
    1. Your class needs to have a class member with the same name for each
       constructor argument (to be sklearn compliant).
       """

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        pass

    def get_best_action(self, X):
        return choice((1, 2))

    def predict(self, X):
        return [choice((1, 0)) for x in X.iterrows()]

class NeuralBanker(BankerBase):

    def __init__(self, interest_rate = 0.05, layer_sizes=[16,8],
                batch_size=32,
                epochs=10,
                optimizer="Adam",
                loss="binary_crossentropy",
                alpha = 0.001):
        self.interest_rate = interest_rate
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha

    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not callable(v)}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y

    def parse_X(self, X):
        #parsed_X = X.drop(["amount", "duration"], axis=1)
        return X.values.reshape(1, -1)

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        return actions

    def expected_utility(self, X):
        p = self.predict_proba(self.parse_X(X))
        gain = self.calculate_gain(X)
        expected_utilitiy = gain*p.flatten()-X['amount']*(1-p.flatten())
        return expected_utilitiy

    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def build_network(self, X, y):
        base_cls = Sequential()
        base_cls.add(BatchNormalization())
        for layer_size in self.layer_sizes:
            base_cls.add(Dense(layer_size, activation='elu',kernel_regularizer=regularizers.l2(self.alpha)))
        base_cls.add(Dense(1, activation='sigmoid'))
        base_cls.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        model = BaggingClassifier(base_estimator = base_cls,
                                    n_estimators = 10)
        return model

    def get_proba(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1))
        X, y = X.values, y
        self.model = self.build_network(X, y)
        self.model.fit(X, y)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def accuracyscore(self, Xtest, y_test):
        return self.model.evaluate(Xtest, y_test)
        print(accuracy)

if __name__ == '__main__':
    run()
