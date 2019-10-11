import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from forestbanker_jolynde import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../data/german.data', sep=' ',
                     names=features+[target])

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
        #print("action")
        #print(action)
        good_loan = y_test.iloc[t] # assume the labels are correct
        ##print("good_loan")
        ##print(good_loan)
        duration = X_test['duration'].iloc[t]
        #print("duration")
        #print(duration)
        amount = X_test['amount'].iloc[t]
        #print("amount")
        #print(amount)
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1)
        ##print("@@@@@@@@@@@@@@@@@@@@@@utility")
        ##print(utility)
    return utility


## Main code


### Setup model
#import random_banker # this is a random banker
#decision_maker = random_banker.RandomBanker()

interest_rate = 0.005
decision_maker = RandomForestClassifier()

### Do a number of preliminary tests by splitting the data in parts

n_tests = 10
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    #print(utility)
print("Mean utility")
print(utility / n_tests)

ypred = decision_maker.predict(X_test)
print("Accuracy score")
print(accuracy_score(y_test, ypred))

#Confusion matrix

y_test = list(y_test)
for i in range(0, len(y_test)):
    if y_test[i] == 2:
        y_test[i] = 0
conf_mat = confusion_matrix(y_true=list(y_test), y_pred=list(ypred))
print('Confusion matrix:\n', conf_mat)
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
