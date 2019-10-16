# This function take as parameters an array X_one_column with the corresponding column that we want to anonymize.
# For instance X['age']. It wiil return the new array with interval of value and not numérical value.
def privacy_step(X_one_column):
    #pandas.options.mode.chained_assignment = None # This avoid the warn beacause, this function will write into the original frame.
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

# This function is for randomising responses. The function return an array with anonymized data.
# The principe is to flip a coin and if it comes heads, respond truthfully.
# Otherwise, change the data randomly
def privacy_step_coin(X_one_column,p):
    #pandas.options.mode.chained_assignment = None # avoid warning
    New_X_one_column = X_one_column
    for i in range(0,len(New_X_one_column)) :
        n = 1
        coin = numpy.random.binomial(n,p)
        # if coin = 1 we do nothing because we say the truth
        if coin ==0:
            #we chose aleatory in the list of type of data.
            class_of_X = list(set(New_X_one_column))
            high_value_class = len(class_of_X)-1
            random_i = numpy.random.randint(low=0,high=high_value_class)
            New_X_one_column[i] =  class_of_X[random_i]
    return New_X_one_column
