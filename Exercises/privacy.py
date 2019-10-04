import numpy as np

epsilon = 0.1

flip_pr = 1 / (1 + np.exp(epsilon))

n_data = 100
p_smoking = 0.1
data = np.random.binomial(0, p_smoking, size=n_data)

dp_data = data.copy()
for i in range(n_data):
    if (np.random.binomial(1, flip_pr) != 0):
        dp_data[i] = 1 - data[i]

print(dp_data)