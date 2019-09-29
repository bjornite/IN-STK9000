import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

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


##############
# TO TEST MY 2 FUNCTIONS
# print(privacy_step(X['age']))
# print(privacy_epsilon(X['age'],0.1))
###############


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
#decision_makers.append(RandomBanker(interest_rate))
#decision_makers.append(kNNbanker(interest_rate))
decision_makers.append(RandomForestClassifierBanker(interest_rate))
#decision_makers.append(NeuralBankerGridSearch(interest_rate))
### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 1
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


ypred = decision_maker.predict(X_test)
print(ypred)

print("Accuracy score")
print(accuracy_score(y_test, ypred))


#Confusion matrix
y_test = list(y_test)
for i in range(0, len(y_test)):
    if y_test[i] == 2:
        y_test[i] = 0
confusion = confusion_matrix(y_true=list(y_test), y_pred=list(ypred))
print('Confusion matrix:\n', confusion)
labels = ['Class 0', 'Class 1']
"""
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
"""
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# Metrics from classification matrix
print('Classification accuracy:', (TP + TN) / float(TP + TN + FP + FN))  #should be the same as the accuracy score
print('Classification error:', (FP + FN) / float(TP + TN + FP + FN))
print('Sensitivity:', (TP / float(FN + TP)))
print('Specificity:', (TN / (TN + FP)))
print('False positive rate:', (FP / float(TN + FP)))
print('Precision:', TP / float(TP + FP))

# ROC curve

y_pred_prob = decision_maker.predict_proba(X_test)[:,1] #probabilities for class 1
print(y_pred_prob)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for credit loan classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# AUC
print(metrics.roc_auc_score(y_test, y_pred_prob))



"""
# Plot to show feature importance
feature_imp = decision_maker.get_importances(X_train)
print(feature_imp)

labels, ys = zip(*feature_imp)
xs = np.arange(len(labels))
width = 0.5
plt.bar(xs, ys, width, align='center')
plt.xticks(xs, labels, rotation=90)
plt.yticks(ys)
plt.title('Feature importance')
plt.show()
"""
