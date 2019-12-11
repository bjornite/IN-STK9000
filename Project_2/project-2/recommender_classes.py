import numpy as np
import pandas as pd

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from kneed import KneeLocator


class HistoricalRecommender:

# fit on actions instead of outcome
    model = None
    bootstrapped_models = None
    ohe = None
    action_list = None
    X = None
    y = None
    actions = None
    i = 0
    pca = None

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def n_components(self, data):
        i=0
        percento_range = np.arange(0.1,1,0.01)
        n_comp = np.zeros(len(percento_range))
        for percento in percento_range:
            pca = PCA(percento)
            pca.fit(data)
            n_comp[i] = pca.n_components_
            i = i+1
        kn = KneeLocator(n_comp, percento_range, curve='concave', direction='increasing')
        n_components_ = int(kn.knee)
        return n_components_

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        param_grid = {'layer_sizes': [[32, 16], [64, 16]],
        'batch_size': [5, 10],
        'epochs': [1, 5],
        'optimizer': ['Adam', 'sgd'],
        'alpha': [0.001, 0.0001]}
        #self.model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)

        n_compo = self.n_components(data)
        self.pca = PCA(n_components = n_compo)
        data_red = self.pca.fit_transform(data)
        print(self.pca.n_components_)
        self.ohe = OneHotEncoder(sparse=False, categories=[range(self.n_actions)])
        one_hot_a = self.ohe.fit_transform(actions)
        #bootstrap the data here

        self.model = NNDoctor(n_actions=0, n_outcomes=self.n_actions)
        self.model.fit(data_red, one_hot_a)
        #print(self.model.best_params_)
        return self.model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            #predicted_outcomes = self.model.predict(data)
            return self.reward(policy_actions, outcome).mean() #behind outcome .reshape(1,-1)

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

        #the action probabilities here is just the .predict_proba from the model, which predicts the probability for
        #class 0 when the input is a 1d-vector, and the probility for each action when there are multiple action
        #because we use the .predict_classes functionality this piece of code becomes obsolete.

        return None

#here the 'predict_classes' predicts an action, since the model is fitted on the actions.
    def recommend(self, user_data):
        user_data_red = self.pca.transform(user_data.reshape(1,-1))
        return np.asscalar(np.argmax(self.model.predict_classes(user_data_red.reshape(1,-1))))
        #return np.argmax(self.get_action_probabilities(user_data_red))

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        n_pcs = self.pca.components_.shape[0]

        most_important = [np.abs(self.pca.components_[i]).argsort()[-5:][::-1] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important[i] for i in range(n_pcs)}
        df = pd.DataFrame(dic.items(), columns = ['PC', 'Important genes'])
        df['Explained variance'] = self.pca.explained_variance_ratio_

        print(df)

        return None


class ImprovedRecommender:

    model = None
    pca = None
    ohe = None

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def n_components(self, data):
        i=0
        percento_range = np.arange(0.1,1,0.01)
        n_comp = np.zeros(len(percento_range))
        for percento in percento_range:
            pca = PCA(percento)
            pca.fit(data)
            n_comp[i] = pca.n_components_
            i = i+1
        kn = KneeLocator(n_comp, percento_range, curve='concave', direction='increasing')
        n_components_ = int(kn.knee)
        return n_components_

    def train_model(self, X, a, y):
        param_grid = {'layer_sizes': [[32, 16], [64, 16]],
        'batch_size': [5, 10],
        'epochs': [1, 5],
        'optimizer': ['Adam', 'sgd'],
        'alpha': [0.001, 0.0001]}
        #model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)
        #model = NNDoctor()
        self.ohe = OneHotEncoder(sparse=False, categories=[range(self.n_actions)])
        self.action_list = np.array(range(self.n_actions))
        self.actions = a
        one_hot_a = self.ohe.fit_transform(a)
        self.y = y
        n_compo = self.n_components(X)
        self.pca = PCA(n_components = n_compo)
        X_red = self.pca.fit_transform(X)
        X_a = np.concatenate((X_red, one_hot_a), axis=1)
        self.X = X_a
        model = MultiHeadNNDoctor(self.n_actions, self.n_outcomes)
        model.build_network(X_a, y)
        model.fit(self.X, self.y)
        #print(self.model.best_params_)
        return model
        #self.model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)

        #bootstrap the data here

        #print(self.model.best_params_)

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.model = self.train_model(data, actions, outcome)
        return self.model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            model = self.train_model(data, actions, outcome)
            predicted_outcomes = model.predict(np.concatenate((data, policy_actions.reshape(-1,1)), axis=1))
            return self.reward(policy_actions, predicted_outcomes.reshape(1,-1)).mean()

    def predict_proba(self, data, treatment):
        predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        return predictions

    def get_action_probabilities(self, user_data):
        #print("Recommending")
        predictions = []
        for a in self.action_list:
            a_vector = self.ohe.transform([[a]])
            estimated_outcome = self.model.predict(np.concatenate((user_data, a_vector.reshape(-1))).reshape(1,-1))
            estimated_reward = self.reward(a, estimated_outcome).reshape(-1)
            predictions.append(estimated_reward)
        predictions = np.array(predictions).T
        return predictions#np.exp(predictions)/np.sum(np.exp(predictions))

    def estimate_historic_utility(self, data, actions, outcome):
        estimated_outcome = self.model.predict(np.concatenate((data, actions), axis=1))
        #outcome_prob = 1/(1 + np.exp(0.5-estimated_outcome))
        return self.reward(actions, estimated_outcome).mean()

    def recommend(self, user_data):
        # Use bootstrapped Thompson sampling to recommend actions
        user_data_red = self.pca.transform(user_data.reshape(1,-1)).ravel()
        a_probs = self.get_action_probabilities(user_data_red)
        use_head = np.random.randint(len(a_probs))
        return np.argmax(a_probs[use_head])
        #print(np.argmax(self.get_action_probabilities(user_data_red)))

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None

class AdaptiveRecommender:

    model = None
    bootstrapped_models = None
    ohe = None
    action_list = None
    X = []
    data = []
    y = []
    actions = []
    i = 1

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def n_components(self, data):
        i=0
        percento_range = np.arange(0.1,1,0.01)
        n_comp = np.zeros(len(percento_range))
        for percento in percento_range:
            pca = PCA(percento)
            pca.fit(data)
            n_comp[i] = pca.n_components_
            i = i+1
        kn = KneeLocator(n_comp, percento_range, curve='concave', direction='increasing')
        n_components_ = int(kn.knee)
        return n_components_

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
        #model = GridSearchCV(NNDoctor(), param_grid, cv=10, n_jobs=4)
        #model = NNDoctor()
        self.ohe = OneHotEncoder(sparse=False, categories=[range(self.n_actions)])
        self.action_list = np.array(range(self.n_actions))
        one_hot_a = self.ohe.fit_transform(a)
        n_compo = self.n_components(X)
        self.pca = PCA(n_components = n_compo)
        X_red = self.pca.fit_transform(X)
        X_a = np.concatenate((X_red, one_hot_a), axis=1)
        model = MultiHeadNNDoctor(self.n_actions, self.n_outcomes)
        model.build_network(X_a, y)
        model.fit(X_a, y)
        self.actions = a.tolist()
        self.data = X.tolist()
        self.X = X_a.tolist()
        self.y = y.tolist()
        #print(self.model.best_params_)
        return model

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.model = self.train_model(data, actions, outcome)
        return self.model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy is None:
            return self.reward(actions, outcome).mean()
        else:
            policy_actions = np.array([policy.recommend(x) for x in data])
            model = self.train_model(data, actions, outcome)
            predicted_outcomes = model.predict(np.concatenate((self.pca.transform(data), self.ohe.transform(policy_actions.reshape(-1,1))), axis=1))
            return self.reward(policy_actions, predicted_outcomes).mean()

    def predict_proba(self, data, treatment):
        predictions = self.model.predict(np.concatenate((data, [treatment])).reshape(1,-1)).ravel()
        return predictions

    def get_action_probabilities(self, user_data):
        #print("Recommending")
        predictions = []
        for a in self.action_list:
            a_vector = self.ohe.transform([[a]])
            estimated_outcome = self.model.predict(np.concatenate((user_data, a_vector.reshape(-1))).reshape(1,-1))
            estimated_reward = self.reward(a, estimated_outcome).reshape(-1)
            predictions.append(estimated_reward)
        predictions = np.array(predictions).T
        return predictions#np.exp(predictions)/np.sum(np.exp(predictions))

    def estimate_historic_utility(self, data, actions, outcome):
        estimated_outcome = self.model.predict(np.concatenate((data, actions), axis=1))
        #outcome_prob = 1/(1 + np.exp(0.5-estimated_outcome))
        return self.reward(actions, estimated_outcome).mean()

    def recommend(self, user_data):
        # Use bootstrapped Thompson sampling to estimate
        user_data_red = self.pca.transform(user_data.reshape(1,-1)).ravel()
        a_probs = self.get_action_probabilities(user_data_red)
        #print(a_probs)
        use_head = np.random.randint(len(a_probs))
        return np.argmax(a_probs[use_head])

    def observe(self, user, action, outcome):
        #Update the model with new observation
        self.i += 1
        one_hot_a = self.ohe.transform([[action]])
        X_red = self.pca.transform(user.reshape(1,-1))
        X_a = np.concatenate((X_red.reshape(-1), one_hot_a.reshape(-1))).reshape(1,-1)
        self.X.append(X_a.tolist()[0])
        self.actions.append(action.reshape(1,-1))
        self.data.append(user.reshape(1,-1))
        self.y.append(outcome.reshape(1,-1))
        if (self.i % 100 == 0): #Retrain model every 100 observations
            self.model.fit(np.array(self.X), np.array(self.y).reshape(-1,1))
        return None

    def final_analysis(self):
        #print("Estimated utility of final policy: ")
        #print(self.estimate_utility(self.data, self.actions, self.y, self))
        return None

#Support class for the neural network
class NNDoctor:
    def __init__(self,
                 n_actions=1,
                 n_outcomes=1,
                 layer_sizes=[64, 32, 16],
                 batch_size=32,
                 epochs=10,
                 optimizer="adam",
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

    def predict(self, X):
        return self.model.predict(X)

    def build_network(self, X, y):
        model = Sequential()
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation='elu',kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(self.n_outcomes, activation='sigmoid'))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def score(self, X, y):
        return (self.model.predict(X)**2 - y**2).mean()


class MultiHeadNNDoctor(NNDoctor):
    single_head_models = None
    

    def __init__(self,
                 n_actions=1,
                 n_outcomes=1,
                 layer_sizes=[64, 32, 32, 16],
                 batch_size=10,
                 epochs=1,
                 optimizer="sgd",
                 loss="binary_crossentropy",
                 alpha = 0.0001):
        super().__init__(n_actions = n_actions,
                         n_outcomes = n_outcomes,
                         layer_sizes = layer_sizes,
                         batch_size = batch_size,
                         epochs = epochs,
                         optimizer = optimizer,
                         loss = loss,
                         alpha = alpha)
        self.n_heads = 10

    def build_network(self, X, y):
        input = Input(shape=X.shape[1:])
        x = input
        for layer_size in self.layer_sizes[:-1]:
            x = Dense(layer_size, activation='elu',kernel_regularizer=regularizers.l2(self.alpha))(x)
        heads = []
        for n in range(self.n_heads):
            heads.append(Dense(y.shape[1], activation='sigmoid')(x))
        model = Model(inputs=input,
			          outputs=heads)
        single_head_models = []
        for n in range(self.n_heads):
            sh_model = Model(inputs=input, outputs=heads[n])
            sh_model.compile(loss=self.loss,
                             optimizer=self.optimizer,
                             metrics=['accuracy'])
            single_head_models.append(sh_model)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.single_head_models = single_head_models

    def fit(self, X, y):
        # Bootstrap n_heads samples from X,y and train one head on each sample
        train_order = np.random.randint(self.n_heads, size=self.n_heads)
        for n in train_order:
            randlist = np.random.randint(X.shape[0], size=X.shape[0])
            self.single_head_models[n].fit(X[randlist], y[randlist], epochs=1, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)
