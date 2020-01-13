import numpy as np

def create_gauss_mat(k_samples, num_points, low_mu,high_mu,std):
    mu_array = np.random.uniform(low=low_mu, high=high_mu, size=(k_samples,))
    gauss_matrix = np.zeros((k_samples, num_points))
    for i in range(k_samples):
        mu = mu_array[i]
        gauss_matrix[i] = np.random.normal(mu, std, num_points)
    return gauss_matrix, mu_array


k_samples = 10
low_mu, high_mu = -3,3
std = 1
num_points = 10**3

gauss_mat, mu_arr = create_gauss_mat(k_samples, num_points, low_mu,high_mu,std)
np.savetxt('data/gauss_mat.csv',gauss_mat, delimiter=',')
np.savetxt('data/mu_arr.csv',mu_arr, delimiter=',')

# # look at dist
# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(gm[1], 70, normed=True)
