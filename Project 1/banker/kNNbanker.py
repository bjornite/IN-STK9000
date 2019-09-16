from banker import BankerBase, run
from random import choice
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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
    
class kNNbanker(BankerBase):

    model = None
    
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate
        
    def kNN(self, X, y):

        model = KNeighborsClassifier(n_neighbors = 15)
        return model
    
    def prediction(self, y):
        return self.model.predict(y)

    def get_proba(self, y):
        return metrics.accuracy_score((y), self.prediction(y))

    """ Use this probability for the expected utility""" 
    
    def expected_utility(self, X):
        p = self.get_proba(X)
        gain = self.calculate_gain(X)
        return gain.values*p.flatten()-X['amount'].values*(1-p.flatten())
    
    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)
        
    """def parse_y(self, y):
        y[np.where(y == 2)] = 0
        return y"""

    def fit(self, X, y):
        #y = self.parse_y(y.values.reshape(-1,1))
        self.model = self.kNN(X, y)
        self.model.fit(X, y)


    def get_best_action(self, X):
        if (self.expected_utility(X) < 0).all:
            actions = 1
        else:
            actions = 2
        """actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions = 0)] = 2"""
        print(actions)
        return actions
    


if __name__ == '__main__':
    run()
