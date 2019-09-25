from imblearn.under_sampling import RandomUnderSampler
from banker import BankerBase, run
from random import choice
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
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
        model = Sequential()
        model.add(BatchNormalization())
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation='elu',kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def score(self, X, y):
        n_test_examples = len(X)
        utility = 0

        ## Example test function - this is only an unbiased test if the data has not been seen in training
        for t in range(n_test_examples):
            action = self.get_best_action(X.iloc[t])
            good_loan = y.iloc[t] # assume the labels are correct
            duration = X['duration'].iloc[t]
            amount = X['amount'].iloc[t]
            # If we don't grant the loan then nothing happens
            if (action==1):
                if (good_loan != 1):
                    utility -= amount
                else:
                    utility += amount*(pow(1 + self.interest_rate, duration) - 1)
        return utility
    
    def get_proba(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1))
        X, y = X.values, y
        self.model = self.build_network(X, y)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

class NeuralBankerGridSearch(BankerBase):
    
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        param_grid = {'layer_sizes': [[32, 16], [64, 16], [64,32,16,8]], 
        'batch_size': [8],
        'epochs': [3],
        'interest_rate': [self.interest_rate],
        'optimizer': ['Adam'],
        'loss': ['binary_crossentropy'],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        self.model = GridSearchCV(NeuralBanker(), param_grid, cv=5, n_jobs=6)
        self.model.fit(X, y)
        print(self.model.best_params_)

    def parse_X(self, X):
        #parsed_X = X.drop(["amount", "duration"], axis=1)
        return X.values.reshape(1, -1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        return actions

    def expected_utility(self, X):
        p = self.predict_proba(self.parse_X(X))
        #print(p.flatten())
        gain = self.calculate_gain(X)
        expected_utilitiy = gain*p.flatten()-X['amount']*(1-p.flatten())
        return expected_utilitiy
    
    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

if __name__ == '__main__':
    run()
