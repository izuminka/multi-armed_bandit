import numpy as np

class Env:
    def __init__(self):
        # TODO: generalize data generation
        # TEMP: test values
        self.num_actions = 10
        self.low_mu = -3 # lower mean for rewards
        self.high_mu = 3 # upper mean for rewards
        self.std = 1
        self.num_points = 10**3

        # load data
        self._gauss_mat_rewards = np.loadtxt('data/gauss_mat.csv', delimiter=',')
        self._mean_reward = np.loadtxt('data/mu_arr.csv', delimiter=',')

        self.time_counter = 0

    def reward(self, action_id):
        self.time_counter+=1
        return np.random.choice(self._gauss_mat_rewards[action_id])

# class
#
# reward_action_ls = [0]*num_actions
# np_actions_taken_ls = [0]*num_actions
#
# def est_mean_reward(action_id):
#     # converges to mean_reward[action_id] in the limit
#     if np_actions_taken_ls[action_id]:
#         return reward_action_ls[action_id]/np_actions_taken_ls[action_id]
#     return 0
#
#
# get_reward(0,num_points)
