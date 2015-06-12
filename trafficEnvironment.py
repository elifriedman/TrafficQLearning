# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:31:08 2015

@author: EliFriedman
"""
"""
1) init: initialize agent with acitonFn
2) startEpisode: called at beginning of episode
3) observeTransition: gives reward and transfers state
4) stopEpisode: ends episode
"""
from qlearningAgents import QLearningAgent
import util

class TrafficGame:

  def __init__(self, states, costs, flow_cost=0.02, num_cars=[600, 400, 300, 400], start_states=[0, 0, 1, 1], end_states=[11, 12, 11, 12]):
    """
      Initializes the traffic simulation with the graph represented by states,
      which contains a list of arrays representing a node's edges.
      The costs map represents the cost of each edge. It's indexed by
      (start, end)
      num_cars contains a list of the number of cars to be initialized in
      different states.
      start_states contains the actual states in which to initialize the cars.
    """
    self.states = states
    self.costs = costs
    self.flow_cost = flow_cost
    self.num_cars = num_cars
    self.start_states = start_states
    self.end_states = end_states
    state2action = lambda state: self.states[state]
    self.agentList = []
    for i in range(sum(num_cars)):
      q=QLearningAgent(actionFn=state2action, numTraining=1000, epsilon=0.5, alpha=0.5, gamma=.99, index=i)
      self.agentList.append(q)
    self.trained = False

  def test(self,maxSteps=1000):
    print "Testing..."
    self.runEpisode(maxSteps=maxSteps, percentFinished=1, epsilon=0,alpha=0)

    totalAvg = 0
    pathSet = util.Counter()
    baList = []
    for agent in self.agentList:
      p=tracepath(agent,agent.startState)
      if p == -1:
        baList.append(agent)
      else:
        pathSet[str(p)] += 1
        totalAvg += -agent.getTestAvg()
    return (float(totalAvg) / (len(self.agentList)-len(baList)),pathSet,baList)

  def train(self,n,updateTime):
    """
    n: The number of episodes to run
    updateTime: prints an update every updateTime number of times
    """
    print "Training for %d episodes" % (n)
    print "iteration #: avg # steps"
    for agent in self.agentList:
      agent.setNumTraining(n)
    epsilon = 1
    count=0
    avgSteps=0
    for i in range(n):
      count += 1
      avgSteps += self.runEpisode(maxSteps=1000, percentFinished=1, epsilon=epsilon)
      epsilon *= 0.995
      if updateTime != 0 and i % updateTime == 0:
        print "%d: %f" % (i, float(avgSteps)/count )
        avgSteps = 0
        count = 0
    self.trained = True

  def initializeEpisode(self,epsilon,alpha):
    n = 0
    car_grp = 0
    for agent in self.agentList:
      agent.setEpsilon(epsilon)
      agent.setLearningRate(alpha)
      agent.startEpisode(self.start_states[car_grp], self.end_states[car_grp])
      n += 1
      if n >= self.num_cars[car_grp]:
        n = 0
        car_grp += 1

  def runEpisode(self,maxSteps=200, percentFinished=1, epsilon=.5, alpha=0.5):
    """
      . initializes the agents to their start statest
      . step a certain number of times or until all/% cars arrive at destination
      .
    """
    self.initializeEpisode(epsilon,alpha)

    for i in range(maxSteps):
      num_finished = self.step()
      if num_finished >= percentFinished*len(self.agentList):
        break

    for agent in self.agentList:
      agent.stopEpisode()

    return i

  def step(self):
    """
      . get each agent's action
      . update costs based on agent's action
      . give each agent its reward (= -cost)
    """
    # deep copy our cost dictionary
    self.new_costs = dict((k,v) for (k, v) in self.costs.items())
    num_reached_goals = 0
    for agent in self.agentList:
      state = agent.lastState
      if state == agent.goalState:
        num_reached_goals += 1
        continue
      action = agent.getAction(state)
      nextState = action
      if nextState == agent.goalState:
        num_reached_goals += 1
      # cost = base_cost + flow_cost * #drivers
      self.new_costs[(state,nextState)] += self.flow_cost

    for agent in self.agentList:
      state = agent.lastState
      if state == agent.goalState: continue
      action = agent.getAction(state)
      nextState = action
      reward = -self.new_costs[(state,nextState)]
      agent.observeTransition(state,action,nextState,reward)

    return num_reached_goals

def getAction(s):
    r = [];
    for state in states[ord(s)-ord('A')]:
        r.append(chr(state+ord('A')))
    return r

def getCost(x,s1,s2):
    i1 = ord(s1)-ord('A')
    i2 = ord(s2)-ord('A')
    try:
        ind = states[i1].index(i2)
        return x.new_costs[i1][ind]
    except ValueError:
        return -1;

states = [
            [1, 2, 3],          # A
            [0, 3, 4],          # B
            [0, 3, 5, 6],       # C
            [0, 1, 2, 4, 6, 7], # D
            [1, 3, 7],          # E
            [2, 6, 8],          # F
            [2, 3, 5, 7, 9, 10],# G
            [3, 4, 6, 10],      # H
            [5, 9, 11],         # I
            [6, 8, 10, 11, 12], # J
            [6, 7, 9, 12],      # K
            [8, 9],             # L
            [9, 10]             # M
        ]
cost_lists = [
            [7, 5, 15],
            [7, 11, 11],
            [5, 7, 11, 9],
            [15, 11, 7, 7, 7, 9],
            [11, 7, 7],
            [11, 9, 13],
            [9, 7, 9, 9, 3, 13],
            [9, 7, 9, 3],
            [13, 9, 2],
            [3, 9, 9, 12, 12],
            [13, 3, 9, 2],
            [2, 12],
            [12, 2]
        ]
costs = {}
for i in range(len(states)):
  for j in range(len(states[i])):
    costs[(i,states[i][j])] = cost_lists[i][j]

def t(state): return chr(state+ord('A'))
def tracepath2(agent,start_state):
  n = 0
  s = start_state
  l = [t(s)]
  while n < 15 and s != agent.goalState:
    s = agent.getAction(s)
    l.append(t(s))
    n += 1
  return l

def tracepath(agent,start_state):
  n = 0
  s = start_state
  l = t(s)
  while n < 15 and s != agent.goalState:
    s = agent.getAction(s)
    l = l + t(s)
    n += 1

  if n < 15:
    return l
  return -1

x = TrafficGame(states,costs)#,flow_cost=0.2, num_cars=[6], start_states=[0], end_states=[11])
x.train(1000,100)
v=x.test()