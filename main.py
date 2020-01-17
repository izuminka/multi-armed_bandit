import matplotlib.pyplot as plt  # plot the performace graph
from bandits import EpsGreedyAgent
from environments import StationaryGauss
import numpy as np
import os

def agent_env_inter(env, agent, num_act_taken):
    """Agent Env interaction/training

    Args:
        env (class Environment): given env
        agent (class Agent): given agent
        num_act_taken (int): num interactions with env

    Returns:
        list: average reward at each time step.
    """
    average_reward = np.zeros(num_act_taken)  # for analysis
    for i in range(num_act_taken):
        action_id = agent.action()
        reward = env.reward(action_id)
        agent.update(action_id, reward)
        average_reward[i] = agent.average_reward()
    return average_reward


if __name__ == '__main__':
    # Setup
    num_inter = 1000
    num_trials = 2000
    eps_range = [1, 0.1, 0.01, 0.001, 0]
    save_path = f'results/stationary/EpsGreedyAgent/{num_inter}-{num_trials}'
    if not os.path.exists(save_path): os.makedirs(save_path)

    # Generate Results
    for eps in eps_range:
        trials_mat = np.zeros((num_trials, num_inter))
        for trial_i in range(num_trials):
            env = StationaryGauss()
            agent = EpsGreedyAgent(env.num_actions, epsilon=eps)
            trials_mat[trial_i][:] = agent_env_inter(env, agent, num_inter)
        np.savetxt(f'{save_path}/{eps}.csv', np.mean(trials_mat, axis=0), delimiter=',')

    # Plot Results
    x = range(num_inter)
    for file in os.listdir(f'{save_path}'):
        if file.endswith('.csv'):
            average_reward = np.loadtxt(f'{save_path}/{file}', delimiter=',')
            plt.plot(x, average_reward, label=file.strip('.csv'))
    plt.title('Picking the Best ε')
    plt.xlabel('Number of Interactions')
    plt.ylabel('E[Reward]')
    plt.legend()
    plt.savefig(f'{save_path}/eps_plot.svg')
    plt.show()
