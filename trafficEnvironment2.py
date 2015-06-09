# -*- coding: utf-8 -*-
"""
@author: EliFriedman
"""
from qlearningAgents import QLearningAgent
from envprep import ALfunc,AMfunc,BLfunc,BMfunc, cost_mat, costs
import numpy as np


agentList = []
term_states = []
alpha=0.5; epsilon=1; gamma=0.8; decay_rate = 0.99
for i in range(600):
  agentList.append(QLearningAgent(ALfunc, alpha, epsilon, gamma ))
  agentList[-1].startEpisode(0,11)
  term_states.append((0,11))
for i in range(400):
  agentList.append(QLearningAgent(AMfunc, alpha, epsilon, gamma))
  agentList[-1].startEpisode(0,12)
  term_states.append((0,12))
for i in range(300):
  agentList.append(QLearningAgent(BLfunc, alpha, epsilon, gamma))
  agentList[-1].startEpisode(1,11)
  term_states.append((1,11))
for i in range(400):
  agentList.append(QLearningAgent(BMfunc, alpha, epsilon, gamma))
  agentList[-1].startEpisode(1,12)
  term_states.append((1,12))

cost_mat *= 0.02
print("Starting Loop")
actions = range(len(agentList)) # initialize array...will be overwritten
numtravellers = np.zeros_like(costs)
L = len(agentList)
for n in range(1001):
#  actions.fill(-1)
  numtravellers.fill(0)
  for i in range(len(agentList)):
    actions[i] = agentList[i].getAction(term_states[i][0])
    numtravellers[actions[i]] += 1
  rewards = -(np.dot(cost_mat,numtravellers)+costs)
  avgreward = 0
  for i in range(len(agentList)):
    a = actions[i]
    agentList[i].observeTransition(term_states[i][0],a,term_states[i][1],rewards[a])
    agentList[i].setEpsilon(agentList[i].epsilon*decay_rate)
    avgreward += rewards[a]
  if n%200 == 0:
    print n,': ',avgreward/L