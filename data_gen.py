import numpy as np  # to generate the data
import os  # to create the dir for the data
# import matplotlib.pyplot as plt # to look a the resulting destrubutions


def create_gauss_mat(num_gauss, num_points, low_mu, upper_mu, std):
    """Generate a random matrix of Gaussians centered at various points

    Args:
        num_gauss (int): Numer of Gaussians
        num_points (int): Number of points in the distribution
        low_mu (float): Lower bound for the mean
        upper_mu (float): Upper bound for the mean
        std (float): Standard Deviation for all Gaussians

    Returns:
        np.array(num_gauss x num_points), np.array(1 x num_points):
            Gaussian matrix and array of centers/means for those Gaussians
    """
    mu_array = np.random.uniform(low=low_mu, high=upper_mu, size=(num_gauss,))
    gauss_matrix = np.zeros((num_gauss, num_points))
    for i in range(num_gauss):
        mu = mu_array[i]
        gauss_matrix[i] = np.random.normal(mu, std, num_points)
    return gauss_matrix, mu_array


def generate_save_data(num_gauss, num_points, low_mu, upper_mu, std):
    """Save the generated data

    Args:
        num_gauss (int): Numer of Gaussians
        num_points (int): Number of points in the distribution
        low_mu (float): Lower bound for the mean
        upper_mu (float): Upper bound for the mean
        std (float): Standard Deviation for all Gaussians

    Returns:
        None
    """
    gauss_mat, mu_arr = create_gauss_mat(
        num_gauss, num_points, low_mu, upper_mu, std)
    data_set_name = f'data/{num_gauss}_{num_points}_{low_mu}_{upper_mu}_{std}'
    if not os.path.exists(data_set_name):
        os.makedirs(data_set_name)
    np.savetxt(f'{data_set_name}/gauss-mat.csv', gauss_mat, delimiter=',')
    np.savetxt(f'{data_set_name}/mu-arr.csv', mu_arr, delimiter=',')
    print('saved at', data_set_name)
    # # look at the dist
    # count, bins, ignored = plt.hist(gauss_mat[1], 70, normed=True)



if __name__ == '__main__':
    num_gauss, num_points = 10, 10**3
    low_mu, upper_mu, std = -3, 3, 1
    generate_save_data(num_gauss, num_points, low_mu, upper_mu, std)
