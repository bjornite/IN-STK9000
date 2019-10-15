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

#from randombanker import NeuralBankerGridSearch, RandomBanker
from randombanker import NeuralBankerGridSearch
from forestbanker import RandomForestClassifierBanker
from kNNbanker import kNNbanker
#from forestbanker_optimized import RandomForestClassifier


interest_rate = 0.005
decision_makers = []
#decision_makers.append(RandomBanker(interest_rate))
decision_makers.append(kNNbanker(interest_rate))
decision_makers.append(RandomForestClassifierBanker(interest_rate))
#decision_makers.append(RandomForestClassifier(interest_rate))
#decision_makers.append(NeuralBankerGridSearch(interest_rate))
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


"""
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
"""

N_test, _ = X_test.shape
p_max = np.zeros(N_test)

for decision_maker in decision_makers:
    for t in range(N_test):
        p_max[t] = max(decision_maker.predict_proba(X_test)[t])
    sns.distplot(p_max)
    plt.show()





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

y_pred_prob = decision_maker.predict_proba(X_test) #[:,1] #probabilities for class 1
print(y_pred_prob.shape)"""

# histogram of predicted probabilities
"""
plt.hist(y_pred_prob, bins=8)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of repaid')
plt.ylabel('Frequency')
plt.show()
"""
"""
# predict diabetes if the predicted probability is greater than 0.7
from sklearn.preprocessing import binarize
# it will return 1 for all values above 0.3 and 0 otherwise
# results are 2D so we slice out the first column
y_pred_class = binarize(y_pred_prob.reshape(1,-1), 0.7)[0]
print(y_pred_class.shape)
confusion_new = confusion_matrix(y_true=list(y_test), y_pred = y_pred_class)
print('Confusion matrix (p_threshold = 0.5):\n', confusion_new)

TP_new = confusion_new[1, 1]
TN_new = confusion_new[0, 0]
FP_new = confusion_new[0, 1]
FN_new = confusion_new[1, 0]

print('Specificity_new:', (TN_new / (TN_new + FP_new)))
print('Precision_new:', TP_new / float(TP_new + FP_new))
print('Sensitivity_new:', (TP_new / float(FN_new + TP_new)))
"""

"""
# ROC curve
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
#plt.savefig('./Images/ROC_curve.png')


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# AUC
print('ROC/AUC score:', metrics.roc_auc_score(y_test, y_pred_prob))
"""


"""
# Plot to show feature importance
feature_imp = decision_maker.get_importances(X_train)
#print(feature_imp)

labels, ys = zip(*feature_imp)
xs = np.arange(len(labels))
width = 0.5
plt.bar(xs, ys, width, align='center')
plt.xticks(xs, labels, rotation=90)
plt.yticks(ys)
plt.title('Feature importance')
plt.show()
#plt.savefig('../Images/feature_importance.png')
"""
