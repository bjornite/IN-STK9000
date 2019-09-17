from banker import BankerBase, run
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

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
    
    def __init__(self, interest_rate, layer_sizes=[16,8], optimizer="Adam", loss="binary_crossentropy"):
        self.interest_rate = interest_rate
        self.layer_sizes = layer_sizes
        self.optimizer = optimizer
        self.loss = loss
        
    def parse_y(self, y):
        y[np.where(y == 2)] = 0 
        return y
    
    def parse_X(self, X):
        #parsed_X = X.drop(["amount", "duration"], axis=1)
        return X

    def build_network(self, X, y):
        model = Sequential()
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
        model.add(Dense(self.layer_sizes[0], activation='elu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(self.layer_sizes[1], activation='elu',kernel_regularizer=regularizers.l2(0.001)))
        #model.add(Dropout(0.1))
        #model.add(Dense(8, activation='elu'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model
    
    def get_proba(self, X):
        return self.model.predict(X)
    
    def expected_utility(self, X):
        p = self.get_proba(self.parse_X(X))
        #print(p.flatten())
        gain = self.calculate_gain(X)
        expected_utilitiy = gain.values*p.flatten()-X['amount'].values*(1-p.flatten())
        return expected_utilitiy
    
    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)
        
    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1))
        X = self.parse_X(X)
        # Over-sampling to solve problem with unbalanced dataset
        #rom imblearn.over_sampling import SMOTE
        #os = SMOTE(random_state=0)
        #columns = X.columns
        #os_data_X,os_data_y=os.fit_sample(X, y)
        #os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
        #os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        os_data_X, os_data_y = rus.fit_resample(X, y)
        self.model = self.build_network(os_data_X, os_data_y)
        self.model.fit(os_data_X, os_data_y, epochs = 3, validation_split=0.05)
        
    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        return actions

if __name__ == '__main__':
    run()
