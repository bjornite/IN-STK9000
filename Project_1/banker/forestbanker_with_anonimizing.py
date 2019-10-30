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
        #model = RandomForestClassifier(n_estimators=130)
        param_grid = {
            'n_estimators': np.linspace(10, 200).astype(int),
            'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }

        # Estimator for use in random search
        estimator = RandomForestClassifier(random_state = RSEED)

        # Create the random search model
        model = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                                scoring = 'roc_auc', cv = 3,
                                n_iter = 10, verbose = 1, random_state=RSEED)

        # Fit
        rs.fit(train, train_labels)
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
        pandas.options.mode.chained_assignment = None # This avoid the warn beacause, this function will write into the original frame.
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
        pandas.options.mode.chained_assignment = None # avoid warning
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
     
    ##############
    # TO TEST MY 3 FUNCTIONS
    ###############
    ## test for privacy_step
    # print(privacy_step(X['age']))
    #print(X['age'])
    #print(privacy_epsilon(X['age'],0.0001))


    # I anonymize the data with laplace mechanism for centralised DP
    ## test for privacy_epsilon
    #X['duration'] = privacy_epsilon(X['duration'],0.0001)
    #X['amount'] = privacy_epsilon(X['amount'],0.1)
    #X['residence time'] = privacy_epsilon(X['residence time'],0.0001)
    #print(X['duration'])
    #print(X[encoded_features])

    ## test for privacy_step_coin
    # X['phone_A192']=privacy_step_coin'(§è!  ,(X['phone_A192'],0.5)
    # print(X['phone_A192'])

if __name__ == '__main__':
    run()
