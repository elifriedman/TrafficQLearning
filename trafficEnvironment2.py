# -*- coding: utf-8 -*-
"""
@author: EliFriedman
"""
from qlearningAgents import QLearningAgent
from envprep import ALfunc,AMfunc,BLfunc,BMfunc, cost_mat, costs, pp
import numpy as np

data = np.zeros((3,6)) # a small datastructure for storing data to be written to file
decay_rate = 0.995
g=-1;
for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
  g += 1
  al = -1
  for alpha in [0.3, 0.5, 0.7]:
    al += 1
    epsilon=1;
    agentList = []
    term_states = []
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
    
    print "Starting Loop: gamma=",gamma,' alpha=',alpha
    actions = range(len(agentList)) # initialize array...will be overwritten
    numtravellers = np.zeros_like(costs)
    L = len(agentList)
    last_reward = 0
    Nmax = 500
    for n in range(Nmax):
      numtravellers.fill(0)
      for i in range(L):
        actions[i] = agentList[i].getAction(term_states[i][0])
        numtravellers[actions[i]] += 1
      rewards = -(np.dot(cost_mat,numtravellers)+costs) # calculate the (negative) cost for each agent
      avgreward = 0
      for i in range(L):
        a = actions[i]
        agentList[i].observeTransition(term_states[i][0],a,term_states[i][1],rewards[a])
        agentList[i].setEpsilon(agentList[i].epsilon*decay_rate)
        avgreward += rewards[a]
      avg = avgreward/L
      if n%100==0:
        if abs(last_reward-avg) < 5E-2 or n==Nmax-1:
          print 'Ending: ',n,': ',avg
          print ''
          last_reward = avg
          break
        else:
          last_reward = avg
    data[al,g] = last_reward;


# just write out our data for future use
s = ''
gamma = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
alpha = [0.3,0.5,0.7]
f=open('data_modified','w')
for g in range(len(gamma)):
    s+='\t'+str(gamma[g])
s += '\n'
for a in range(len(alpha)):
    s += str(alpha[a])
    for g in range(len(gamma)):
        s += '\t'+str(round(data[a,g],3))
    s += '\n'
f.write(s);f.close()


f=open('QL_modified.csv','w')
s = 'paths,numdrivers,costs\n'
for i in range(len(rewards)):
    s += pp(paths[i])+','+str(numtravellers[i])+','+str(rewards[i])+'\n'
f.write(s);f.close()
