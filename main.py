import numpy as np
import matplotlib.pyplot as plt  # plot the performace graph


class Environment:
    """Simple Environment for the agent

    Args:
        dataset_name (str): name of the dataset, composed of the params

    Attributes:
        num_actions (int): Num of actions available, (num reward restributions)
        _num_points (int): Number of points in one reward restribution
        _low_mu (float): Lower bound for the mean across distributions
        _high_mu (float): Upper bound for the mean across distributions
        _std (float): Standard Deviation for all distributions
        _gauss_mat (np.array): matrix where rows are distributions
        _mu_arr (np.array): array of means for the distributions
        time_counter (type): Action/time counter
    """

    def __init__(self, dataset_name='10_1000_-3_3_1'):
        """Load the data for the env, set the env variables

        Args:
            dataset_name (str): name of the dataset, composed of the params
        """

        actions, points, low_m, high_m, std = dataset_name.split('_')
        self.num_actions = int(actions)

        self._num_points = int(points)
        self._low_mu = float(low_m)  # lower mean for rewards
        self._high_mu = float(high_m)  # upper mean for rewards
        self._std = float(std)

        # load data
        self._gauss_mat = np.loadtxt(
            f'data/{dataset_name}/gauss-mat.csv', delimiter=',')
        self._mu_arr = np.loadtxt(
            f'data/{dataset_name}/mu-arr.csv', delimiter=',')

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
        _average_reward (list): average reward at each time step
        _action_sequence (list): sequence of actions at each time step
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

        # for analysis
        self._average_reward = []
        self._action_sequence = []

    def est_mean_reward(self, action_id):
        """Estimate mean reward for a given action over previous time steps

        Args:
            action_id (type): action taken now

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

    def take_action_update(self, env, action_id):
        """Take action, interact with env, rescive reward, update model

        Args:
            env (type): Description of parameter `env`.
            action_id (type): Description of parameter `action_id`.

        Returns:
            type: Description of returned object.

        """
        # act on env
        reward = env.reward(action_id)
        # update agent
        self.actions_count[action_id] += 1
        self.actions_sum_reward[action_id] += reward
        self.total_rewards += reward
        self.total_actions += 1
        # return env

    def average_reward(self):
        """Expected total reward at time t
        """
        return self.total_rewards / self.total_actions

    def train(self, env, num_interactions):
        """Train the agent in the given env.

        Args:
            env (class Environment): given env
            num_interactions (int): num interactions with env

        Returns:
            None
        """
        for _ in range(num_interactions):
            action_id = agent.epsilon_greedy_action()
            self.take_action_update(env, action_id)
            # for analysis
            self._action_sequence.append(action_id)
            self._average_reward.append(self.average_reward())


if __name__ == '__main__':
    env = Environment()

    num_interactions = 20000
    x = range(num_interactions)
    eps_range = [1, 0.1, 0.01, 0.001, 0]
    for eps in eps_range:
        agent = EpsGreedyAgent(env.num_actions, epsilon=eps)
        agent.train(env, num_interactions)
        y = [abs(env._mu_arr.max() - v) for v in agent._average_reward]
        plt.plot(x, y, label=str(eps))

    plt.title('Picking the Best ε')
    plt.xlabel('Number of Interactions')
    plt.ylabel('| E[Reward] - E[Max Possible Reward] |')
    plt.legend()
    plt.savefig('results.png')
    # plt.show()
