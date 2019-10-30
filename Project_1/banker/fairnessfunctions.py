def FCalibration(classifier, X, y):
    # Calculate action selection a
    utility = 0
    actions = decision_maker.get_best_action(X)
    sum = 0
    # For each action a
    X = X.assign(a=actions)
    X = X.assign(y=y)
    agroups = X.groupby(X["a"])
    # For each agegroup z
    azgroups = X.groupby(["a", pd.qcut(X["age"], 3)])
    for key, group in azgroups:
        # For each actual outcome y
        for outcome in [1, 2]:
            #sum += P(y | a, z) - P(y | a)
            sum += ((group[group["y"] == outcome].size / group.size) - (X[(X["a"] == key[0]) & (X["y"] == outcome)].size/X[X["a"] == key[0]].size))**2
    return sum

def FBalance(classifier, X, y):
    # Calculate action selection a
    utility = 0
    actions = decision_maker.get_best_action(X)
    sum = 0
    # For each action a
    X = X.assign(a=actions)
    X = X.assign(y=y)
    agroups = X.groupby(X["y"])
    # For each agegroup z
    azgroups = X.groupby(["y", pd.qcut(X["age"], 3)])
    for key, group in azgroups:
        # For each actual outcome y
        for outcome in [1, 2]:
            #sum += P(y | a, z) - P(y | a)
            sum += ((group[group["a"] == outcome].size / group.size) - (X[(X["y"] == key[0]) & (X["a"] == outcome)].size/X[X["y"] == key[0]].size))**2
    return sum


def FCalibrationAmount(classifier, X, y):
        # Calculate action selection a
    utility = 0
    actions = decision_maker.get_best_action(X)
    sum = 0
    # For each action a
    X = X.assign(a=actions)
    X = X.assign(y=y)
    agroups = X.groupby(X["a"])
    # For each agegroup z
    azgroups = X.groupby(["a", pd.qcut(X["age"], 3), pd.qcut(X["amount"], 3)])
    for key, group in azgroups:
        # For each actual outcome y
        for outcome in [1, 2]:
            #sum += P(y | a, z) - P(y | a)
            sum += ((group[group["y"] == outcome].size / group.size) - (X[(X["a"] == key[0]) & (X["y"] == outcome)].size/X[X["a"] == key[0]].size))**2
    return sum

def FBalanceAmount(classifier, X, y):
    # Calculate action selection a
    utility = 0
    actions = decision_maker.get_best_action(X)
    sum = 0
    # For each action a
    X = X.assign(a=actions)
    X = X.assign(y=y)
    agroups = X.groupby(X["y"])
    # For each agegroup z
    azgroups = X.groupby(["y", pd.qcut(X["age"], 3), pd.qcut(X["amount"], 3)])
    for key, group in azgroups:
        # For each actual outcome y
        for outcome in [1, 2]:
            #sum += P(y | a, z) - P(y | a)
            sum += ((group[group["a"] == outcome].size / group.size) - (X[(X["y"] == key[0]) & (X["a"] == outcome)].size/X[X["y"] == key[0]].size))**2
    return sum