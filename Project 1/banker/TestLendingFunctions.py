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