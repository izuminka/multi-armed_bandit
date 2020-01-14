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

class Agent:
    def __init__(self, eps, num_actions):
        self.num_actions = num_actions
        self.eps = eps

        self.num_actions_taken = [0]*self.num_actions
        self.sum_reward_actions = [0]*self.num_actions
        self.reward_total = 0
        self.steps_counter = 0

    def est_mean_reward(self, action_id):
        # converges to mean_reward[action_id] in the limit
        if self.num_actions_taken[action_id]:
            return self.sum_reward_actions[action_id]/self.num_actions_taken[action_id]
        return 0

    def take_action(self, action_id, reward):
        self.num_actions_taken[action_id]+=1
        self.sum_reward_actions[action_id]+=reward
        self.reward_total += reward
        self.steps_counter+=1

    def greedy_action(self):
        return max([(a, self.est_mean_reward(a)) for a in range(self.num_actions)], key=lambda x:x[1])[0]

    def eps_greedy_action(self):
        if np.random.random_sample() < self.eps:
            return np.random.choice(self.num_actions)
        return self.greedy_action()

    def average_reward(self):
        return self.reward_total/self.steps_counter




env_test = Env()
eps = 0.01
num_actions = env_test.num_actions
agent_test = Agent(0.01, num_actions)

number_steps = 10000
average_reward = []
action_sequence = []
for _ in range(number_steps):
    action_id = agent_test.eps_greedy_action()
    reward = env_test.reward(action_id)
    agent_test.take_action(action_id, reward)

    # analysis
    action_sequence.append(action_id)
    average_reward.append(agent_test.average_reward())


import matplotlib.pyplot as plt
plt.plot(range(number_steps), average_reward)

average_reward[-1]
env_test._mean_reward.max()
