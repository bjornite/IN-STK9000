import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pd.read_csv('../data/german.data', sep=' ',
                     names=features+[target])

#sns.countplot(x = df[target], data = df)

numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pd.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))


# scale data
norm = 1/X.std()

tests = 10

utility = {}

# # The kNN model, makes 10 runs over the whole model

for i in range(tests):
    Xtrain, Xtest, Xtrain_normal, Xtest_normal, ytrain, ytest = train_test_split(X*norm, X, df[target],
                test_size=0.2) # split in training and test data for normalized data and not normalized data

    k_range = range(5,30)
    scores = {}
    scores_list = []
    for k in k_range:
        model1 = KNeighborsClassifier(n_neighbors=k)
        model1.fit(Xtrain, ytrain)
        ypred1 = model1.predict(Xtest)
        scores[k] = metrics.accuracy_score(ytest, ypred1)
        scores_list.append(metrics.accuracy_score(ytest, ypred1))

    def keywithmaxval(d):
         """ a) create a list of the dict's keys and values;
             b) return the key with the max value"""
         v=list(d.values())
         k=list(d.keys())
         return k[v.index(max(v))]

    kbest = keywithmaxval(scores)
    print("kbest", kbest)


    model = KNeighborsClassifier(n_neighbors = kbest).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

    print("Accuracy:", metrics.accuracy_score(ytest, ypred))


    # Confusion matrix

    conf_mat = confusion_matrix(y_true=ytest, y_pred=ypred)
    print('Confusion matrix:\n', conf_mat)
    """
    labels = ['Class 0', 'Class 1']
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

    predicted_prob = model.predict_proba(Xtest)[:, 0] #probability of belonging to class 1, repaing the loan

    interest_rate = 0.005
    gain = Xtest_normal['amount']*((1 + interest_rate)**(Xtest_normal['duration']) - 1)

    expected_utility = ((gain*predicted_prob)-(Xtest_normal['amount']*(1-predicted_prob))).astype(int)

    expected_utility[expected_utility > 0] = 1
    expected_utility[expected_utility < 0] = 2

    actions = expected_utility

    n_test_examples = len(Xtest)
    utility = 0
    for t in range(n_test_examples):
            action = actions.iloc[t]
            good_loan = ytest.iloc[t] # assume the labels are correct
            duration = Xtest_normal['duration'].iloc[t]
            amount = Xtest_normal['amount'].iloc[t]
            # If we don't grant the loan then nothing happens
            if (action==1):
                if (good_loan != 1):
                    utility -= amount
                else:
                    utility += amount*(pow(1 + interest_rate, duration) - 1)
    print("Utility", utility)


# Want to plot the utility over the 10 test runs, don't know yet how to do this
#plt.plot(utility)
#plt.show()
