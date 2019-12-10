from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


class HistoricalRecommender:

    model = None
    pca = None

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward

    def fit_data(self, data):
        print("Preprocessing data")
        return None

    def train_model(self, X, a, y):
        print("Fitting treatment outcomes")
        param_grid = {'layer_sizes': [[32, 16], [64, 16]],
        'batch_size': [5, 10],
        'epochs': [1, 5],
        'optimizer': ['Adam', 'sgd'],
        'loss': ['mse'],
        'alpha': [0.001, 0.0001]}
        self.pca = PCA(.70)
        data_red = self.pca.fit_transform(X)
        self.model = NNDoctor(n_actions=self.n_actions, n_outcomes=self.n_outcomes)
        self.model.fit(data_red, a)

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.train_model(data, actions, outcome)
        return self.model


    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            #predicted_outcomes = self.model.predict(np.concatenate((data, policy_actions.reshape(-1,1)), axis=1))
            return self.reward(policy_actions, outcome).mean()

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        #predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        #return predictions
        pred = self.model.predict(data)
        return pred

    def get_action_probabilities(self, user_data):
        #print("Recommending")
        #the action probabilities here is just the .predict_proba from the model, which predicts the probability for
        #class 0 when the input is a 1d-vector, and the probility for each action when there are multiple action
        #because we use the .predict_classes functionality this piece of code becomes obsolete.

        return None

    def estimate_historic_utility(self, data, actions, outcome):
        estimated_outcome = self.model.predict(np.concatenate((data, actions), axis=1))
        #outcome_prob = 1/(1 + np.exp(0.5-estimated_outcome))
        return self.reward(actions, estimated_outcome).mean()

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        user_data_red = self.pca.transform(user_data.reshape(1,-1))
        #print(np.asscalar(self.model.predict_classes(user_data_red.reshape(1,-1))))
        return np.asscalar(self.model.predict_classes(user_data_red.reshape(1,-1)))
        #return np.argmax(self.get_action_probabilities(user_data_red))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None







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
