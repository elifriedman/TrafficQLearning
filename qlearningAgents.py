from learningAgents import ReinforcementAgent

import random,util,sys

class QLearningAgent:
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, actionFn, alpha=0.5 , epsilon=0.05, gamma=0.8,**args):
    """
    actionFn should return a list of legal actions for a given state
             or None if there are no legal actions
    """
    self.getLegalActions = actionFn;
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.gamma = float(gamma)
    self.Q = util.Counter()

  def setEpsilon(self, epsilon):
    self.epsilon = epsilon

  def setLearningRate(self, alpha):
    self.alpha = alpha

  def setDiscount(self, gamma):
    self.gamma = gamma

  def startEpisode(self,startState, endState):
    self.episode_reward = 0
    self.state = startState
    self.goal = endState

  def stopEpisode(self):
    pass

  def reachedGoal(self):
    return self.state == self.goal
    
  def observeTransition(self,state,action,nextState,reward):
    if action in self.getLegalActions(state):
      self.update(state,action,nextState,reward)
      self.state = nextState
      self.episode_reward += reward

  ######## Q Learning Stuff ###########

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return self.Q[(state, action)]

  def getMaxActionValue(self, state):
    """
      Returns a tuple containing the action and value that
      maximizes Q transitioning from this state.
      returns (action,Q(state,action))
    """
    if self.getLegalActions(state)==None:
      return (None,0.0)
    value = -sys.maxint-1
    alist = []
    for action in self.getLegalActions(state):
        if self.Q[(state, action)] > value:
            value = self.Q[(state, action)]
            alist = [action]
        elif self.Q[(state, action)] == value:
            alist.append(action)
    return (random.choice(alist), value)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    return self.getMaxActionValue(state)[0] # returns the action

  def getValue(self, state):
    """
     Returns max_action Q(state,action)
     where the max is over legal actions.  Note that if
     there are no legal actions, which is the case at the
     terminal state, you should return a value of 0.0.
    """
    return self.getMaxActionValue(state)[1] # returns the value


  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    if legalActions==None: return None
    action = None
    if util.flipCoin(self.epsilon):
        action = random.choice(legalActions)
    else:
        action = self.getPolicy(state)
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # find max_a{Q(s',a)} of next state
    maxQsp = self.getValue(nextState);

    # learn the alpha % of the (discounted) new action and reward
    # Q(s,a) = (1-alpha)Q(s,a) + alpha(reward+gamma*maxQ(s',a))
    self.Q[(state,action)] = (1-self.alpha)*self.Q[(state,action)] + self.alpha*(reward + self.gamma*maxQsp)
