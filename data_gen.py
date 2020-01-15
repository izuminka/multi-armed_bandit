import numpy as np
import os

def create_gauss_mat(num_actions, num_points, low_mu,high_mu,std):
    mu_array = np.random.uniform(low=low_mu, high=high_mu, size=(num_actions,))
    gauss_matrix = np.zeros((num_actions, num_points))
    for i in range(num_actions):
        mu = mu_array[i]
        gauss_matrix[i] = np.random.normal(mu, std, num_points)
    return gauss_matrix, mu_array


# default
num_actions = 10
num_points = 10**3
low_mu, high_mu = -3,3
std = 1

gauss_mat, mu_arr = create_gauss_mat(num_actions, num_points, low_mu,high_mu,std)
data_set_name = f'data/{num_actions}_{num_points}_{low_mu}_{high_mu}_{std}'
if not os.path.exists(data_set_name): os.makedirs(data_set_name)
np.savetxt(f'{data_set_name}/gauss-mat.csv',gauss_mat, delimiter=',')
np.savetxt(f'{data_set_name}/mu-arr.csv',mu_arr, delimiter=',')

# # look at dist
# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(gm[1], 70, normed=True)
