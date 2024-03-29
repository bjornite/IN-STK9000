from banker import BankerBase, run
from random import choice
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier


class kNNbanker(BankerBase):

    model = None
    norm = None

    def __init__(self, interest_rate):
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
        return self.model.predict_proba(X)[:,1]

    def expected_utility(self, X):
        import numbers
        if isinstance(X.values[0], numbers.Number):
            X_vals = X.values.reshape(1,-1)
        else:
            X_vals = X.values
        p = self.get_proba(X_vals)
        gain = self.calculate_gain(X)
        expected_utility = gain*p.flatten()-X['amount']*(1-p.flatten())
        return expected_utility

    def calculate_gain(self, X):
        return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

    def fit(self, X, y):
        y = self.parse_y(y.values.reshape(-1,1).ravel())
        self.model = self.kNN(X, y)
        self.model.fit(X,y)

    def get_best_action(self, X):
        actions = (self.expected_utility(X) > 0).astype(int)
        actions[actions == 0] = 2
        return actions

    def predict_proba(self, Xtest):
        return self.model.predict_proba(Xtest)

    def predict(self,Xtest):
        return self.model.predict(Xtest)


if __name__ == '__main__':
    run()
