import numpy as np
import matplotlib.pyplot as plt  # plot the performace graph


class Environment:
    """Simple Environment for the agent

    Args:
        dataset_name (str): name of the dataset, composed of the params

    Attributes:
        _gauss_mat (np.array): matrix where rows are distributions
        _mu_arr (np.array): array of means for the distributions
        num_actions (int): Num of actions available, (num reward restributions)
        time_counter (type): Action/time counter
    """

    def __init__(self, dataset_name='10_1000_-3_3_1'):
        """Load the data for the env, set the env variables

        Args:
            dataset_name (str): name of the dataset, composed of the params
        """
        self._gauss_mat = np.loadtxt(
            f'data/{dataset_name}/gauss-mat.csv', delimiter=',')
        self._mu_arr = np.loadtxt(
            f'data/{dataset_name}/mu-arr.csv', delimiter=',')
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


class EpsGreedyAgent:
    """Agent that acts greedy with 1-ε prob, takes random action with ε prob

    Args:
        num_actions (int): Num actions available to the agent
        epsilon (float): Probability with which takes random action

    Attributes:
        actions_count (list): Counter of actions
        actions_sum_reward (list): Tot rewards for each action
        total_rewards (float): Tot reward for all actions
        total_actions (type): Tot num of actions taken
    """

    def __init__(self, num_actions, epsilon=0.01):
        """Set up the variables

        Args:
            num_actions (int): Num actions available to the agent
            epsilon (float): Probability with which takes random action
        """
        self.num_actions = num_actions
        self.epsilon = epsilon

        self.actions_count = [0] * self.num_actions
        self.actions_sum_reward = [0] * self.num_actions
        self.total_rewards = 0
        self.total_actions = 0

    def est_mean_reward(self, action_id):
        """Estimate mean reward for a given action over previous time steps

        Args:
            action_id (int): action taken now

        Returns:
            0: if no previous such action was taken
            float: expected reward for this action based on the previous obs.
        """
        # converges to mean_reward[action_id] in the limit
        if self.actions_count[action_id]:
            return self.actions_sum_reward[action_id] / self.actions_count[action_id]
        return 0

    def greedy_action(self):
        """Chose a greedy action based on highest expected reward across actions

        Returns:
            int: action_id
        """
        action_rewards = [(a, self.est_mean_reward(a))
                          for a in range(self.num_actions)]
        return max(action_rewards, key=lambda x: x[1])[0]

    def epsilon_greedy_action(self):
        """Chose a greedy action with prob 1-ε, or rand act with prob ε

        Returns:
            int: action_id
        """
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(self.num_actions)
        return self.greedy_action()

    def chose_action(self):
        """Chose an action to act on the env"""
        return self.epsilon_greedy_action()

    def update(self, action_id, reward):
        """Update the agent after an interaction with the env

        Args:
            action_id (int): action taken
            reward (float): reward got from env

        Returns:
            None
        """
        self.actions_count[action_id] += 1
        self.actions_sum_reward[action_id] += reward
        self.total_rewards += reward
        self.total_actions += 1

    def average_reward(self):
        """Expected total reward at time t
        """
        return self.total_rewards / self.total_actions


def agent_env_inter(env, agent, num_inter):
    """Agent Env interaction/training

    Args:
        env (class Environment): given env
        agent (class Agent): given agent
        num_inter (int): num interactions with env

    Returns:
        list: average reward at each time step.
    """
    average_reward = []  # for analysis
    for _ in range(num_inter):
        action_id = agent.chose_action()
        reward = env.reward(action_id)
        agent.update(action_id, reward)
        average_reward.append(agent.average_reward())
    return average_reward


if __name__ == '__main__':
    env = Environment()

    num_inter = 20000
    x = range(num_inter)
    eps_range = [1, 0.1, 0.01, 0.001, 0]
    for eps in eps_range:
        agent = EpsGreedyAgent(env.num_actions, epsilon=eps)
        average_reward_ls = agent_env_inter(env, agent, num_inter)
        y = [abs(env._mu_arr.max() - v) for v in average_reward_ls]
        plt.plot(x, y, label=str(eps))
    plt.title('Picking the Best ε')
    plt.xlabel('Number of Interactions')
    plt.ylabel('| E[Reward] - E[Max Possible Reward] |')
    plt.legend()
    plt.savefig('results.png')
    # plt.show()
