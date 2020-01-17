import numpy as np

class Agent:
    """Base class for the bandit agents

    Args:
        avail_actions (int): Num actions available to the agent

    Attributes:
        total_rewards (float): Tot reward for all actions
        total_actions (int): Tot num of actions taken
        estimates (list): Reward estimates for each action
        actions_count (list): Counter of actions
    """

    def __init__(self, avail_actions):
        """Set up the variables

        Args:
            avail_actions (int): Num actions available to the agent
        """
        self.avail_actions = avail_actions
        self.total_actions = 0
        self.total_rewards = 0
        self.estimates = [0] * self.avail_actions
        self.actions_count = [0] * self.avail_actions

    def new_estimate(self, old_estimate, target, step_size):
        """New estimate of the reward for a given action"""
        return old_estimate + step_size * (target-old_estimate)

    def average_reward(self):
        """Expected total reward at time t
        """
        if not self.total_actions:
            return 0
        return self.total_rewards / self.total_actions


class EpsGreedyAgent(Agent):
    """Agent that acts greedy with 1-ε prob, takes random action with ε prob

    Args:
        avail_actions (int): Num actions available to the agent
        epsilon (float): Probability with which takes random action
    """
    def __init__(self, avail_actions, epsilon):
        super().__init__(avail_actions)
        self.epsilon = epsilon

    def action(self):
        """Chose a greedy action with prob 1-ε, or rand act with prob ε

        Returns:
            int: action_id
        """
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(self.avail_actions)
        action_rewards = [(a, self.estimates(a)) for a in range(self.avail_actions)]
        return max(action_rewards, key=lambda x: x[1])[0]


    def update(self, action_id, reward):
        """Update the agent after an interaction with the env

        Args:
            action_id (int): action taken
            reward (float): reward got from env

        Returns:
            None
        """
        step_size = 1/self.actions_count[action_id]
        self.estimates[action_id] = self.new_estimate(estimates[action_id], reward, step_size)
        self.total_rewards += reward
        self.actions_count[action_id] += 1
        self.total_actions += 1


# greedy_test = EspGreedyAgent(10,0.01)
# greedy_test.average_reward()
