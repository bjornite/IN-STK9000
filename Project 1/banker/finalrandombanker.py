

from banker import BankerBase, run
from random import choice
import sys
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

"""
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# For NeuralBanker
import tensorflow as tf

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
sys.stderr = stderr
"""

# For kNNbanker
from sklearn import metrics
from sklearn.pipeline import make_pipeline

# For RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

    def __init__(self):
        self.interest_rate = None

    def set_interest_rate(self, interest_rate):
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
        self.interest_rate = None
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

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

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def accuracyscore(self, Xtest, y_test):
        return self.model.evaluate(Xtest, y_test)
        print(accuracy)

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

    def predict(self, Xtest):
        actions = (self.model.predict(Xtest) > 0.5).astype(int).flatten()
        return actions

    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def predict_classes(self, Xtest):
        return self.model.predict_classes(Xtest)

class kNNbanker(BankerBase):

    model = None
    norm = None

    def __init__(self):
        self.interest_rate = None

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

    def kNN(self, X, y):
        scaler = StandardScaler()
        base_cls = KNeighborsClassifier(n_neighbors = 15)
        knn = BaggingClassifier(base_estimator = base_cls,
                                n_estimators = 100)

        model = make_pipeline(scaler, knn)
        return model

    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y

    def parse_X(self, X):
        #parsed_X = X.drop(["amount", "duration"], axis=1)
        return X #.values.reshape(1, -1)

    def prediction(self, y):
        return self.model.predict(y)

    def get_proba(self, X):
        return self.model.predict_proba(np.array(X).reshape(1,-1))[0][1]

    def expected_utility(self, X):
        p = self.get_proba(self.parse_X(X.values.reshape(1, -1)))
        gain = self.calculate_gain(X)
        expected_utility = gain*p.flatten()-X['amount']*(1-p.flatten())
        return expected_utility

    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1).ravel())
        X = self.parse_X(X)
        self.model = self.kNN(X, y)
        self.model.fit(X,y)

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        return actions

    def predict_proba(self, Xtest):
        return self.model.predict_proba(Xtest)

    def predict(self,Xtest):
        return self.model.predict(Xtest)


class RandomForestClassifierBanker(BankerBase):
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
        base_cls = RandomForestClassifier()
        model = BaggingClassifier(base_estimator = base_cls,
                                    n_estimators = 130)
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
        return importance
        print(importance)




# This function take as parameters an array X_one_column with the corresponding column that we want to anonymize.
    # For instance X['age']. It wiil return the new array with interval of value and not numérical value.
def privacy_step(X_one_column):
    #pandas.options.mode.chained_assignment = None # This avoid the warn beacause, this function will write into the original frame.
    max = X_one_column.max()
    min = X_one_column.min()
    difference = max - min
    # Calculates the number of values in a step
    step = difference / 4
    # Replacement of each value with the corresponding interval
    for i in range(0,len(X_one_column)) :

        if min <= X_one_column[i] < min+step :
            step1 = "[{min} - {vars}[".format(min=min, vars=min+step)
            X_one_column[i]=step1

        elif min+step <= X_one_column[i] < min+2*step :
            step2 = "[{min} - {vars}[".format(min=min+step, vars=min+2*step)
            X_one_column[i]=step2

        elif min+2*step <= X_one_column[i] < min+3*step :
            step3 = "[{min} - {vars}[".format(min=min+2*step, vars=min+3*step)
            X_one_column[i]=step3

        elif min+3*step <= X_one_column[i] < max :
            step4 = "[{min} - {vars}]".format(min=min+3*step, vars=max)
            X_one_column[i]=step4
    return X_one_column

#### Laplace mechanism for centralised DP
# This function take as parameters an array X_one_column with the corresponding column that we want to anonymize and the epsilon. For instance X['age'].
# It wiil return the new array with data and noise for each value.
def privacy_epsilon(X_one_column,epsilon):
    max = X_one_column.max()
    min = X_one_column.min()
    central_sensitivity = max / len(X_one_column)
    local_noise = numpy.random.laplace(scale=central_sensitivity/epsilon, size=len(X_one_column))
    X_with_noise = X_one_column + local_noise
    return X_with_noise

# This function is for randomising responses. The function return an array with anonymized data.
# The principe is to flip a coin and if it comes heads, respond truthfully. 
# Otherwise, change the data randomly
def privacy_step_coin(X_one_column,p):
    #pandas.options.mode.chained_assignment = None # avoid warning
    New_X_one_column = X_one_column
    for i in range(0,len(New_X_one_column)) :
        n = 1
        coin = numpy.random.binomial(n,p)
        # if coin = 1 we do nothing because we say the truth
        if coin ==0:
            #we chose aleatory in the list of type of data.
            class_of_X = list(set(New_X_one_column))
            high_value_class = len(class_of_X)-1
            random_i = numpy.random.randint(low=0,high=high_value_class)
            New_X_one_column[i] =  class_of_X[random_i]
    return New_X_one_column





if __name__ == '__main__':
    run()
