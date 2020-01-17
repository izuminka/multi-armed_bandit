import numpy as np
import os

class StationaryGauss:
    """Simple Environment with stationary Gaussians

    Args:
        data_params (str): name of the dataset, composed of the params

    Attributes:
        _gauss_mat (np.array): matrix where rows are distributions
        _mu_arr (np.array): array of means for the distributions
        num_actions (int): Num of actions available, (num reward restributions)
        time_counter (type): Action/time counter
    """

    def __init__(self, data_params='10_1000_-2_2_1', save_data=False):
        """Load the data for the env, set the env variables

        Args:
            data_params (str): name of the dataset, composed of the params
        """
        self.data_params = data_params
        self.save_data = save_data
        self._gauss_mat, self._mu_arr = self.__create_gauss_mat()
        self.num_actions = self._gauss_mat.shape[0]
        self.time_counter = 0

    def reward(self, action_id):
        """Reward for the action, sampled from the distribution

        Args:
            action_id (int): action corresponding to the reward dist

        Returns:
            float: reward value (neg or positive)
        """
        self.time_counter += 1
        return np.random.choice(self._gauss_mat[action_id])


    def __create_gauss_mat(self):
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
        #TODO random seed
        num_gauss, num_points, low_mu, upper_mu, std = self.data_params.split('_')
        num_gauss, num_points = int(num_gauss), int(num_points)
        low_mu, upper_mu, std = float(low_mu), float(upper_mu), float(std)
        mu_array = np.random.uniform(low=low_mu, high=upper_mu, size=(num_gauss,))
        gauss_matrix = np.zeros((num_gauss, num_points))
        for i in range(num_gauss):
            mu = mu_array[i]
            gauss_matrix[i] = np.random.normal(mu, std, num_points)
        if self.save_data:
            data_set_name = f'data/{num_gauss}_{num_points}_{low_mu}_{upper_mu}_{std}'
            if not os.path.exists(data_set_name): os.makedirs(data_set_name)
            np.savetxt(f'{data_set_name}/gauss-mat.csv', gauss_mat, delimiter=',')
            np.savetxt(f'{data_set_name}/mu-arr.csv', mu_array, delimiter=',')
            print('saved at', data_set_name)
        return gauss_matrix , mu_array
