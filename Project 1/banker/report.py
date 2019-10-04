"""
# IN-STK5000 Project 1, report 1
"""
import numpy as np # handout: exclude
import pandas as pd # handout: exclude
import matplotlib.pyplot as plt # handout: exclude
import seaborn as sns # handout: exclude
import handout # handout: exclude
from banker import BankerBase # handout: exclude
doc = handout.Handout('./handout') # handout: exclude
features = ['checking account balance', 'duration', 'credit history', # handout: exclude
            'purpose', 'amount', 'savings', 'employment', 'installment', # handout: exclude
            'marital status', 'other debtors', 'residence time', # handout: exclude
            'property', 'age', 'other installments', 'housing', 'credits', # handout: exclude
            'job', 'persons', 'phone', 'foreign'] # handout: exclude
target = 'repaid' # handout: exclude
df = pd.read_csv('../data/german.data', sep=' ', # handout: exclude
                     names=features+[target]) # handout: exclude
"""
First: Exploring and preprocessing the data.
We start by looking at the german credit data file.
When we make a countplot of the outcome variable 
(which is whether a given loan was repaid, where 0 is no and 1 is yes), we see that the data is imbalanced.
"""
df.loc[df['repaid'] == 2, 'repaid'] = 0
sns.countplot(x = df[target], data = df)
doc.add_figure(plt.gcf()) # handout: exclude
doc.show() # handout: exclude
"""
The percentage of the outcome being 0, or not repaid, is 30%, while the percentage for the outcome being 1, or repaid, is 70%.
"""

"""
Using one-hot encoding the categorical attributes are converted in order to be able to use them in the machine learning algorithm. 
Normalising is done in the models, and therefore not in the preprocessing.
"""

"""
## Task 2.1.1:
Designing a policy for giving or denying credit to individuals 
The choice for giving or denying credit to individuals is based on their probability for being credit-worthy. 
Given this probability, and taking into account the length of the loan, we can calculate the expected utility of giving a loan, using the formula 
> E(U) = gain * p - amount * (1-p)

Where amount is the loaned amount, and gain is the total amount of interest on the loan. p is the predicted probability of the loan being paid back.
The interest is calculated using the following formula:
>   amount * ((1 + interest_rate)**duration - 1)

where duration is loan duration in months, and interest_rate is return per month in %/100.
"""
def calculate_gain(self, X):
    return X['amount']*((1 + self.interest_rate)**(X['duration']) - 1)

def expected_utility(self, X):
    p = self.predict_proba(self.parse_X(X))
    gain = self.calculate_gain(X)
    expected_utilitiy = (gain.values*p.flatten()
                        -X['amount'].values*(1-p.flatten()))
    return expected_utilitiy
"""
## Task 2.1.2:
The probability is calculated using a neural network. 

We assume that the labels represent the actual outcome of each loan, i.e. either loans are fully paid back or defaulted.
We also assume the labeling process is accurate, i.e there is no noise in the labeling process.

We've chosen to implement three different models so we can compare them. The models are
kNN, random forest and a fully connected neural network.

### kNN:
A kNN classifier with k=15 is used, pipelined with a standardscaler which subtracts the mean and scales input features to unit variance.
The fit() function learns the means and standard deviations of each feature for the standardscaler, and then fits the kNN function to the training set.
predict_proba() uses the in-build function in kNN from scikit learn.

### Random forest:
Random forest does not scale the data. We use n=130 classifiers.
predict_proba() uses the in-build function in random forest from scikit learn.

### Neural network:
fit():
As a first layer for the model we use batch normalization. This centers and normalizes the input values. 
The main model is a simple fully connected artificial neural network with elu activations.
We use L2 regularization.
The final layer consists of a single neuron with a sigmoid activation.
The network is trained using binary cross-entropy loss.

We do a cross-validation (on the training data only) grid search over these parameters:
"""
def fit(self, X, y):
    param_grid = {'layer_sizes': [[32, 16], [64, 16], [64,32,16,8]], 
    'batch_size': [8],
    'epochs': [3],
    'interest_rate': [self.interest_rate],
    'optimizer': ['Adam'],
    'loss': ['binary_crossentropy'],
    'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
    self.model = GridSearchCV(NeuralBanker(), param_grid, cv=5, n_jobs=6)
    self.model.fit(X, y)
"""
The scoring function for the grid search is the utility on the holdout set of the cross-validation.

The selected model is trained using a batch size of 8 and 3 epochs of training with the Adam optimizer. layer_sizes is [64, 16], and
the l2 regularization alpha parameter is 0.01.

predict_proba():
This function merely outputs the result of running the trained network forward.
"""
"""
## Task 2.1.3
Now we have models that are well trained and working. So we will be able to combine it with our policy for giving credit.
We will retrieve the result of the function expected_utility(X). If the result is greater than 0, that is to say if we can make money with this loan, the action will take the value 1.
If the value returned is 0 or negative, the loan must not be granted. Since we made a change of value at the beginning, we change the 0 to 2.
"""
def get_best_action(self, X):
    actions = (self.expected_utility(X) > 0).astype(int).flatten()
    actions[np.where(actions == 0)] = 2
    return actions
"""
## Task 2.1.4
Running all models through the TestLending procedure produced the following results:

Average over 5 runs with 0.5% interest each month:
>RandomBanker: -79560\
>kNNbanker: 1591\
>RandomForestClassifierBanker: 8837\
>NeuralBankerGridSearch: 4816

Average over 5 runs with 5% interest each month:
>RandomBanker: 841195\
>kNNbanker: 1256564\
>RandomForestClassifierBanker: 1034688\
>NeuralBankerGridSearch: 1102298
"""