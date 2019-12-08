import numpy as np
import pandas

def default_reward_function(action, outcome):
    return -0.1 * (action!= 0) + outcome

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u

features = pandas.read_csv('../medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('../medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('../medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
import random_recommender
policy_factory = random_recommender.RandomRecommender
RR = random_recommender.RandomRecommender
import recommender_classes
HR = recommender_classes.HistoricalRecommender
import mat_recommender
HR_m = mat_recommender.HistoricalRecommender
policies = [random_recommender.RandomRecommender, 
            HR,
            HR_m,
            recommender_classes.ImprovedRecommender, 
            recommender_classes.AdaptiveRecommender]
for policy_factory in policies:
    print("-------------{}-------------".format(policy_factory.__name__))
    ## First test with the same number of treatments
    print("---- Testing with only two treatments ----")

    print("Setting up simulator")
    generator = data_generation.DataGenerator(matrices="./generating_matrices.mat")
    print("Setting up policy")
    policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    ## Run an online test with a small number of actions
    print("Running an online test")
    n_tests = 1000
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", result)
    print("Final analysis of results")
    policy.final_analysis()

    ## First test with the same number of treatments
    print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
    print("Setting up simulator")
    generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
    print("Setting up policy")
    policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    ## Run an online test with a small number of actions
    print("Running an online test")
    n_tests = 1000
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", result)
    print("Final analysis of results")
    policy.final_analysis()
    print('-----------------------------------------')
HR_ = HR(generator.get_n_actions(), generator.get_n_outcomes())
HR_ = HR_.fit_treatment_outcome(features, actions, outcome)
util = HR_.estimate_utility(features, actions, outcome, HR_)
print('Expected utility historic policy', util)
>>>>>>> 88e91417457925c3d8f3b23ad1166972446ba412
