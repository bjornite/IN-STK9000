from banker import BankerBase, run
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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

    model = None
    
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate
        
    def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y
    
    def build_network(self, X, y):
        model = Sequential()
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
        model.add(Dense(512, activation='elu', input_shape=[X.iloc[0].shape[0]]))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='MSE',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        return model
        
    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1))
        self.model = self.build_network(X, y)
        self.model.fit(X, y)
        
    def get_best_action(self, X):
        pred = self.model.predict(X)
        print(pred)
        return (pred).reshape(-1).astype(int)


if __name__ == '__main__':
    run()
