from sklearn import linear_model
import numpy as np
import pandas as pd

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


class HistoricalRecommender:

# fit on actions instead of outcome
    model = None
    pca = None

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def pca(self, data):
        pca = PCA(.70)
        data_red = pca.fit_transform(data)
        return data_red

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        param_grid = {'layer_sizes': [[32, 16], [64, 16]],
        'batch_size': [5, 10],
        'epochs': [1, 5],
        'optimizer': ['Adam', 'sgd'],
        'loss': ['mse'],
        'alpha': [0.001, 0.0001]}
        #self.model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)
        self.pca = PCA(.70)
        data_red = self.pca.fit_transform(data)

        self.model = NNDoctor(n_actions=self.n_actions, n_outcomes=self.n_outcomes)
        self.model.fit(data_red, actions)
        #print(self.model.best_params_)
        return self.model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            predicted_outcomes = self.model.predict(np.concatenate((data, policy_actions.reshape(-1,1)), axis=1))
            return self.reward(policy_actions, predicted_outcomes.reshape(1,-1)).mean()

    def predict_proba(self, data, treatment):
        #predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        pred = self.model.predict(data)
        return pred

    def predict_classes(self, data, treatment):
        #predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        predictions_classes = self.model.predict_classes(data)
        return predictions_classes

    def get_action_probabilities(self, user_data):
        #print("Recommending")
        predictions = []
        for a in range(self.n_actions):
            #estimated_outcome = self.model.predict(np.concatenate((user_data, [a])).reshape(1,-1))[0][0]
            estimated_outcome = self.model.predict(user_data.reshape(1,-1))[0][0]
            estimated_reward = self.reward(a, estimated_outcome)
            predictions.append(estimated_reward)
        return np.exp(predictions)/np.sum(np.exp(predictions))

#here the 'predict_classes' predicts an action, since the model is fitted on the actions.
    def recommend(self, user_data):
        user_data_red = self.pca.transform(user_data)
        return np.asscalar(self.model.predict_classes(user_data_red.reshape(1,-1)))
        #return np.argmax(self.get_action_probabilities(user_data))

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None


class ImprovedRecommender:

    model = None

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def fit_data(self, data):
        print("Preprocessing data")
        return None

    def train_model(self, X, a, y):
        param_grid = {'layer_sizes': [[32, 16], [64, 16]],
        'batch_size': [5, 10],
        'epochs': [1, 5],
        'optimizer': ['Adam', 'sgd'],
        'loss': ['mse'],
        'alpha': [0.001, 0.0001]}
        #self.model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)
        pca = PCA(.70)
        data_red = pca.fit_transform(data)

        self.model = NNDoctor()
        self.model.fit(np.concatenate((X, a), axis=1), y)
        #print(self.model.best_params_)

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.train_model(data, actions, outcome)
        return self.model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            predicted_outcomes = self.model.predict(np.concatenate((data, policy_actions.reshape(-1,1)), axis=1))
            return self.reward(policy_actions, predicted_outcomes.reshape(1,-1)).mean()

    def predict_proba(self, data, treatment):
        predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        return predictions

    def get_action_probabilities(self, user_data):
        #print("Recommending")
        predictions = []
        for a in range(self.n_actions):
            estimated_outcome = self.model.predict(np.concatenate((user_data, [a])).reshape(1,-1))[0][0]
            estimated_reward = self.reward(a, estimated_outcome)
            predictions.append(estimated_reward)
        return np.exp(predictions)/np.sum(np.exp(predictions))

    def estimate_historic_utility(self, data, actions, outcome):
        estimated_outcome = self.model.predict(np.concatenate((data, actions), axis=1))
        #outcome_prob = 1/(1 + np.exp(0.5-estimated_outcome))
        return self.reward(actions, estimated_outcome).mean()

    def recommend(self, user_data):
        return np.argmax(self.get_action_probabilities(user_data))

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None



#Support class for the neural network
class NNDoctor:
    def __init__(self,
                 n_actions=1,
                 n_outcomes=1,
                 layer_sizes=[32, 16],
                 batch_size=10,
                 epochs=1,
                 optimizer="sgd",
                 loss="binary_crossentropy",
                 alpha = 0.001):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not callable(v)}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict_classes(self, X):
        return self.model.predict_classes(X)

    def fit(self, X, y):
        self.model = self.build_network(X, y)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def build_network(self, X, y):
        model = Sequential()
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation='elu',kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def score(self, X, y):
        return (self.model.predict(X)**2 - y**2).mean()
