import numpy as np

class Env:
    def __init__(self, data_set_name = '10_1000_-3_3_1'):
        actions, points, low_m, high_m, std = data_set_name.split('_')
        self.num_actions = int(actions)
        self.num_points = int(points)
        self.low_mu = float(low_m) # lower mean for rewards
        self.high_mu = float(high_m) # upper mean for rewards
        self.std = float(std)

        # load data
        self._gauss_mat_rewards = np.loadtxt(f'data/{data_set_name}/gauss-mat.csv', delimiter=',')
        self._mean_reward = np.loadtxt(f'data/{data_set_name}/mu-arr.csv', delimiter=',')

        self.time_counter = 0

    def reward(self, action_id):
        self.time_counter+=1
        return np.random.choice(self._gauss_mat_rewards[action_id])

class Agent:
    def __init__(self, epsilon, num_actions):
        self.num_actions = num_actions
        self.epsilon = epsilon

        self.num_actions_taken = [0]*self.num_actions
        self.sum_reward_actions = [0]*self.num_actions
        self.reward_total = 0
        self.stepsilon_total = 0

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
        self.stepsilon_total+=1

    def average_reward(self):
        return self.reward_total/self.stepsilon_total




env_test = Env()
num_actions = env_test.num_actions
epsilon = 0.01
agent_test = Agent(0.01, num_actions)

number_steps = 10000
average_reward = []
action_sequence = []
for _ in range(number_steps):
    action_id = agent_test.epsilon_greedy_action()
    reward = env_test.reward(action_id)
    agent_test.take_action(action_id, reward)
    # analysis
    action_sequence.append(action_id)
    average_reward.append(agent_test.average_reward())


# import matplotlib.pyplot as plt
# plt.plot(range(number_steps), average_reward)

average_reward[-1]
env_test._mean_reward.max()
