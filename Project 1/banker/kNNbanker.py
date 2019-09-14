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
    
    def get_proba(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probability = metrics.accuracy_score(y, self.get_proba)
        print(probability)

    """ Use this probability for the expected utility""" 
    
    def expected_utility(self, X):
        p = self.predict_proba(X)
        gain = self.calculate_gain(X)
        return gain.values*p.flatten()-X['amount'].values*(1-p.flatten())

        return gain*p - X['amount']*(1-p) 
    
    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)
        
    def fit(self, X, y):
        self.model = self.kNN(X, y)
        self.model.fit(X, y)
        
    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int).flatten()
        actions[np.where(actions == 0)] = 2
        print(actions)
        return actions
    


if __name__ == '__main__':
    run()
