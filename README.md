## Overview
Exploration of [Multi-armed Bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)
problem with stationary rewards distribution using ε-greedy agents.

**Assumptions**
 - environment with k actions
 - float positive or negative reward after the action is taken
 - reward sampled from a stationary rewards distribution (time independent)
 - agent is ε-greedy (picks random action with ε probability)

**Objective**
  - to maximize the expected total reward. Since the rewards distributions for each action
    is stationary, figure out the action that gives the largest expected total reward as fast
    as possible and take it every time.
  - pick the best ε

## Prerequisites
- **[Python 3](https://www.python.org/downloads/)**
- **[SciPy](https://www.scipy.org)**. The particular SciPy packages needed are:
    - **[numpy](http://www.numpy.org)**

## Data
I've generated a default dummy data set consisting of 10 random Gaussian distributions with 1000 points,
mean in range of low_mu=-3, upper_mu=3 and std=1. Distribution matrix corresponds
to a reward from taking one of the 10 actions. The set is located at data/10_1000_-3_3_1/gauss-mat.csv.
Additionally I've included data/10_1000_-3_3_1/mu-arr.csv (10 x 1 array of means at which
the Gaussians are centered) for analysis.

In order to generate custom dataset modify the constants and run data_gen.py.
Script generates a K x M numpy array, where K is the number of actions available
to the agent, M is the number of points in the Gaussian distribution and 1 X K array
with generated means of the Gaussians. It would be saved in a newly gen dir:
data/num gauss _ num points _ low mu _ high mu _ std

       python data_gen.py

## Running
Run the script to generate the graph of the best epsilon to pic for your agent.

       python main.py


![Results](results.png)

## Analysis
When ε = 1 we have a purely random agent. When ε = 0 we have a purely greedy
agent that does not take random actions. It appears that ε = 0.01 has a better
convergence than other values and gives great estimate of E[Max Possible Reward]
which took ~1000 interactions with the environment.
