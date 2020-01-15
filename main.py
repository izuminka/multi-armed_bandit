import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, data_set_name='10_1000_-3_3_1'):

        actions, points, low_m, high_m, std = data_set_name.split('_')
        self.num_actions = int(actions)

        self._num_points = int(points)
        self._low_mu = float(low_m) # lower mean for rewards
        self._high_mu = float(high_m) # upper mean for rewards
        self._std = float(std)

        # load data
        self._gauss_mat = np.loadtxt(f'data/{data_set_name}/gauss-mat.csv', delimiter=',')
        self._mu_arr = np.loadtxt(f'data/{data_set_name}/mu-arr.csv', delimiter=',')

        self.time_counter = 0

    def reward(self, action_id):
        self.time_counter+=1
        return np.random.choice(self._gauss_mat[action_id])

class EpsGreedyAgent:
    def __init__(self, num_actions, epsilon=0.01):
        self.num_actions = num_actions
        self.epsilon = epsilon

        self.num_actions_taken = [0]*self.num_actions
        self.sum_reward_actions = [0]*self.num_actions
        self.reward_total = 0
        self.step_total = 0

        # for analysis
        self._average_reward = []
        self._action_sequence = []

    def est_mean_reward(self, action_id):
        # converges to mean_reward[action_id] in the limit
        if self.num_actions_taken[action_id]:
            return self.sum_reward_actions[action_id]/self.num_actions_taken[action_id]
        return 0

    def greedy_action(self):
        action_rewards = [(a, self.est_mean_reward(a)) for a in range(self.num_actions)]
        return max(action_rewards, key=lambda x:x[1])[0]

    def epsilon_greedy_action(self):
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(self.num_actions)
        return self.greedy_action()

    def take_action(self, action_id, reward):
        self.num_actions_taken[action_id]+=1
        self.sum_reward_actions[action_id]+=reward
        self.reward_total += reward
        self.step_total+=1

    def average_reward(self):
        return self.reward_total/self.step_total

    def train(self, env, num_interactions):
        for _ in range(num_interactions):
            action_id = agent.epsilon_greedy_action()
            reward = env.reward(action_id)
            self.take_action(action_id, reward)

            # for analysis
            self._action_sequence.append(action_id)
            self._average_reward.append(self.average_reward())



if __name__ == '__main__':
    env = Environment()

    num_interactions = 20000
    x = range(num_interactions)
    for eps in [1, 0.1, 0.01, 0.001, 0]:
        agent = EpsGreedyAgent(env.num_actions, epsilon=eps)
        agent.train(env, num_interactions)
        y = [abs(env._mu_arr.max() - v) for v in agent._average_reward]
        plt.plot(x, y, label=str(eps))

    plt.xlabel('Number of Interactions')
    plt.ylabel('| E[Reward] - E[Max Possible Reward] |')
    plt.legend()
    plt.show()
