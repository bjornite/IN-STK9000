import pandas

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../data/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

#norm = 1/X.std()

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1)
    return utility
## Main code

### Setup model
#import random_banker # this is a random banker

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from randombanker import NeuralBankerGridSearch, RandomBanker
from forestbanker import RandomForestClassifierBanker
from kNNbanker import kNNbanker

interest_rate = 0.005
decision_makers = []
decision_makers.append(RandomBanker(interest_rate))
decision_makers.append(kNNbanker(interest_rate))
decision_makers.append(RandomForestClassifierBanker(interest_rate))
decision_makers.append(NeuralBankerGridSearch(interest_rate))
### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 5
log = []
for decision_maker in decision_makers:
    utility = 0
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        #decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)

    log.append("{}: {}".format(type(decision_maker), utility / n_tests))

for l in log:
    print(l)
