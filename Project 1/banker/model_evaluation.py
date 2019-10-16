import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from TestLendingFunctions import test_decision_maker, privacy_step, privacy_epsilon

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score


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

n_tests = 10
log = []
for decision_maker in decision_makers:
    utility_ntests = []
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        #decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        utility = (test_decision_maker(X_test, y_test, interest_rate, decision_maker))
        utility_ntests.append(utility)

    print(utility_ntests)

    log.append("{}: {}, {}".format(type(decision_maker), np.mean(utility_ntests), np.std(utility_ntests, ddof = 1)))

for l in log:
    print("Utility", l)


#Accuracy score
log1 = []
for decision_maker in decision_makers:
    ypred = decision_maker.predict(X_test)
    accuracy = accuracy_score(y_test, ypred)

    scores = cross_val_score(decision_maker.model, X[encoded_features], X[target], cv = 10)

    log1.append("{}: {}, {}".format(type(decision_maker), scores.mean(), scores.std()*2))
    #print("Accuracy score:", accuracy_score(y_test, ypred))

for l in log1:
    print("Accuracy", l)


#Confusion matrix
log2 = []
for decision_maker in decision_makers:
    ypred = decision_maker.predict(X_test)
    y_test = list(y_test)
    for i in range(0, len(y_test)):
        if y_test[i] == 2:
            y_test[i] = 0
    confusion = confusion_matrix(y_true=list(y_test), y_pred=list(ypred))
    log2.append("{}: {}".format(type(decision_maker), confusion))

    #print('Confusion matrix (p_threshold = 0.5):\n', confusion)
    #labels = ['Class 0', 'Class 1']
for l in log2:
    print("Confusion", l)



#print('ROC/AUC score:', metrics.roc_auc_score(y_test, y_pred_prob))
log3 = []
for decision_maker in decision_makers:
    y_pred_prob = decision_maker.predict_proba(X_test)[:,1] #probabilities for class 1
    AUC_score = metrics.roc_auc_score(y_test, y_pred_prob)
    log3.append("{}: {}".format(type(decision_maker), AUC_score))

    #print('Confusion matrix (p_threshold = 0.5):\n', confusion)
    #labels = ['Class 0', 'Class 1']
for l in log3:
    print("AUC-score", l)


N_test, _ = X_test.shape
p_max = np.zeros(N_test)

for decision_maker in decision_makers:
    for t in range(N_test):
        p_max[t] = max(decision_maker.predict_proba(X_test)[t])
    sns.distplot(p_max)
    plt.show()
