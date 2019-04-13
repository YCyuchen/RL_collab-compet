#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/4/13 10:10 PM 
# @Author : Yuchen 
# @File : debuging.py 
# @Software: PyCharm

from collections import deque
import torch
import numpy as np
from unityagents import UnityEnvironment

from ddpg_agent import Agent

env = UnityEnvironment(file_name="./Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# traing paramenter
n_episodes = 4000
print_every = 100
scores_deque = deque(maxlen=print_every)
scores_final = []
agent = Agent(state_size, action_size, num_agents, random_seed=2)
# ----------------------- training the agents ----------------------- #
for i_episode in range(n_episodes):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions = agent.act(states)  # select an action (for each agent)
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent) next_states shape:(2,24)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        agent.step(states, actions, rewards, next_states, dones)
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
        scores_deque.append(max(scores))
        scores_final.append(scores)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)), end="")
    if i_episode % 100 == 0:
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores_deque)))
    if np.mean(scores_deque)> 0.5:
        torch.save(agent.actor_local.state_dict(), './checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), './checkpoint_critic.pth')
        break

