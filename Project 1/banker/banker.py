import pandas
from sklearn.model_selection import cross_val_score


class BankerBase(object):

    def fit(self, X, y):
        raise NotImplementedError()

    def get_best_action(self, X):
        raise NotImplementedError()

    def get_params(self, deep):
        return {k: v for k, v in self.__dict__.items() if not callable(v)}


def get_data():
    features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
    target = 'repaid'
    df = pandas.read_csv('../data/german.data', sep=' ', names=features+[target])
    numeric_colums = df.columns[df.dtypes == 'int64']
    categorical_columns = df.columns[df.dtypes == 'object']
    dummies = pandas.get_dummies(df[categorical_columns], drop_first=True)
    data = pandas.concat([dummies, df[numeric_colums]], axis=1)
    features = [i for i in data.columns if i != target]
    return data[features], data[target]


def calculate_gain(X, interest_rate):
    return X['amount']*((1 + interest_rate)**(X['duration']) - 1)


class UtilityCalculator(object):
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def __call__(self, banker, X, y):
        action = banker.get_best_action(X)
        loss = X['amount']
        gain = calculate_gain(X, self.interest_rate)
        utility = sum(gain[((action == 1) & (y == 1))]) - sum(loss[((action == 1) & (y == 2))])
        return utility


class DataHolder(object):
    X, y = get_data()


def get_average_utility(banker,  interest_rate, n_folds=10):
    utils = cross_val_score(banker, DataHolder.X, DataHolder.y,
                            scoring=UtilityCalculator(interest_rate),
                            cv=n_folds)
    return utils.mean(), utils.std()

def run(interest_rate=0.005):
    for cls in BankerBase.__subclasses__():
        print(cls.__name__, get_average_utility(
            cls(interest_rate=interest_rate),
            interest_rate))
