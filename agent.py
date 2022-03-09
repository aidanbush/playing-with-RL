from abc import abstractmethod
from typing import Any

import numpy as np
import math
import random
import representation

class BaseAgent:
    @abstractmethod
    def start(self, observation: Any) -> int:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, reward: float, observation: Any) -> int:
        raise NotImplementedError('Expected `step` to be implemented')

    @abstractmethod
    def end(self, reward: float) -> None:
        raise NotImplementedError('Expected `end` to be implemented')

class RandomAgent(BaseAgent):
    numActions = None

    def __init__(self, parameters):
        self.numActions = parameters["numActions"]

    def start(self, observation):
        return np.random.randint(self.numActions)

    def step(self, reward, observation):
        return np.random.randint(self.numActions)

    def end(self, reward):
        pass

class AccessControlSol(BaseAgent):
    nullAction = -1

    def __init__(self, parameters):
        pass

    def start(self, observation):
        return self.selectAction(observation)

    def step(self, reward, observation):
        return self.selectAction(observation)

    def greedyAction(self, observation, itterations):
        return self.selectAction(observation)

    def selectAction(self, observation):
        freeServers = observation[0]
        priority = observation[1]

        if priority <= 1 and freeServers <= 5:
            return 0
        if priority <= 2 and freeServers <= 3:
            return 0
        if priority <= 4 and freeServers <= 1:
            return 0

        return 1

    def end(self, reward):
        pass

class SpecificAction(BaseAgent):
    action = None

    def __init__(self, parameters):
        self.action = parameters["action"]

    def start(self, observation):
        return self.action

    def step(self, reward, observation):
        return self.action

    def end(self, reward):
        pass

class TabularSARSA(BaseAgent):

    # state = s[0] + s[1]*stateLen[0] ...
    numActions = None
    actionValues = [] # index by [state][action]
    stateFormat = []
    gamma = 1
    epsilon = 0.1
    alpha = 0.1
    lastAction = None
    lastState = None
    initialValue = 0
    nullAction = -1

    def __init__(self, parameters):
        self.numActions = parameters["numActions"]
        self.gamma = parameters["gamma"]
        self.epsilon = parameters["epsilon"]
        self.alpha = parameters["alpha"]
        self.stateFormat = parameters["stateFormat"] # [x max, y max, z max ...] # assume min = 0
        
        numStates = 1
        # calculate the number of states
        for i in range(len(self.stateFormat)): # number of parameters
            # number of states in that parameter - add 1 to account for state 0
            numStates *= self.stateFormat[i] + 1
            

        # initialize action-value array [state][action]
        self.actionValues = [self.initialValue for j in range(numStates*self.numActions)]

    def start(self, observation):
        # since tabular dont need to do anything to the observation

        # select action
        action = self.selectAction(observation)

        # store state and action
        self.lastState = observation
        self.lastAction = action

        # return action
        return action

    def step(self, reward, observation, forcedAction):
        if forcedAction != None:
            self.lastAction = forcedAction
        # select action
        action = self.selectAction(observation)

        # update Q
        self.updateActionValue(observation, action, reward)

        # store state and action
        self.lastState = observation
        self.lastAction = action

        return action

    def end(self, reward, forcedAction):
        if forcedAction != None:
            self.lastAction = forcedAction
        self.terminalUpdateActionValue(reward)

    def selectAction(self, state):
        # e greedy
        if np.random.choice([0,1], p=[1 - self.epsilon, self.epsilon]) == 1:
            return np.random.randint(self.numActions)

        # use Q to determine
        bestActions = [0]
        bestValue = self.Q(state, bestActions[0])
        for a in range(1, self.numActions):
            value = self.Q(state, a)
            if value > bestValue:
                bestActions = [a]
                bestValue = value
            elif value == bestValue:
                bestActions.append(a)

        return np.random.choice(bestActions)

    def updateActionValue(self, state, action, reward):
        lastIndex = self.QIndex(self.lastState, self.lastAction)

        # Q(S,A) = Q(S,A) + a(R + yQ(S',A')-Q(S,A))
        self.actionValues[lastIndex] += self.alpha * \
                (reward + self.gamma * self.Q(state, action) - self.Q(self.lastState, self.lastAction))

    def terminalUpdateActionValue(self, reward):
        # because terminal: Q(S',A') = 0
        lastIndex = self.QIndex(self.lastState, self.lastAction)

        # Q(S,A) = Q(S,A) + a(R + yQ(S',A')-Q(S,A))
        self.actionValues[lastIndex] += self.alpha * (reward - self.Q(self.lastState, self.lastAction))

    def Q(self, state, action):
        index = self.QIndex(state, action)

        return self.actionValues[index]

    def QIndex(self, state, action):
        # convert action and state into an integer
        offset = 1
        index = action
        offset *= self.numActions
        index += state[0] * offset
        for i in range(1, len(state)):
            offset *= self.stateFormat[i-1]
            index += state[i] * offset

        return index

    # works only for 2 dimensional state spaces
    def printValues(self):
        print(self.stateFormat)
        # for every state
        for i in range(self.stateFormat[1]): # y
            for j in range(self.stateFormat[0]): # x
                #stateAvg = 0
                print(end="[")
                for k in range(self.numActions):
                    #index = self.QIndex([j,i], k)
                    Q = self.Q([j,i], k)
                    #stateAvg += self.actionValues[index]
                    print("{:4.0f}".format(Q), end=",")
                print(end="],")
                #print("{:8.2f}".format(stateAvg), end=",")
            print()

    def greedyAction(self, state, iterations):
        bestAction = []
        actionValue = -math.inf

        for action in range(numActions):
            avgVal = sampleStateAction(state, action, iterations)
            if avgVal == self.initialValue:
                continue
            if avgVal == actionValue:
                bestAction.append(action)
            if avgVal > actionValue:
                bestAction = [action]

        return np.random.choice(bestAction)

    def sampleStateAction(self, state, action, iterations):
        # check if any of the states are set to None
        if not None in state:
            return self.Q(state, action)

        val = 0

        for _ in range(interations):
            s = state.copy()
            for i in range(s):
                if s[i] == None:
                    #gen value
                    s[i] = np.random.randint(self.stateFormat[i]+1)
                val += self.Q(s, action)

        return val / iterations

class NStepSARSA(TabularSARSA):

    nSteps = None # n
    time = None # t
    terminal = None # T
    tau = None # \tau

    actionBuffer = None
    stateBuffer = None
    rewardBuffer = None

    def __init__(self, parameters):
        self.nSteps = parameters["n_steps"]
        super().__init__(parameters)

    def storeActionAndState(self, action, state):
        self.actionBuffer.append(action)
        self.stateBuffer.append(state)

        if len(self.actionBuffer) > self.nSteps + 1:
            self.actionBuffer.pop(0) # might not want this here

        if len(self.stateBuffer) > self.nSteps + 1:
            self.stateBuffer.pop(0) # might not want this here

    def storeReward(self, reward):
        self.rewardBuffer.append(reward)

        # track one less reward than actions and states
        if len(self.rewardBuffer) > self.nSteps:
            self.rewardBuffer.pop(0) # might not want this here

    def start(self, observation):
        self.actionBuffer = []
        self.stateBuffer = []
        self.rewardBuffer = []

        self.time = 0
        self.terminal = math.inf

        # since tabular dont need to do anything to the observation

        # select action
        action = self.selectAction(observation)
        self.storeActionAndState(action, observation)

        # store state and action
        self.lastState = observation
        self.lastAction = action

        # return action
        return action

    def step(self, reward, observation):
        self.storeReward(reward)

        self.time += 1

        # select and store action
        action = self.selectAction(observation)
        self.storeActionAndState(action, observation)

        # tau is the time that is being updated
        self.tau = self.time - self.nSteps + 1

        if (self.tau >= 0):
            # update Q
            self.updateActionValues()

        return action

    def end(self, reward):
        self.storeReward(reward)

        self.terminal = self.time + 1
        self.terminalUpdateActionValue(reward)

    def updateActionValues(self):
        # G = sum^min(tau+n,T)_i=tau+1 (y^(i-tau-1)) R_i
        G = 0
        for i in range(len(self.rewardBuffer)):
            G += self.gamma**i * self.rewardBuffer[i]

        # if tau + n < T then G = G + y^n Q(S_{tau + n}, A_{tau + n})
        if self.tau + self.nSteps < self.terminal:
            G += self.gamma**self.nSteps * self.Q(self.stateBuffer[-1], self.actionBuffer[-1])

        #print("update G", G)
        #print(self.rewardBuffer)
        # Q(S_tau, A_tau) = Q(S_tau, A_tau) + a (G - Q(S_tau, A_tau))
        updateIndex = self.QIndex(self.stateBuffer[0], self.actionBuffer[0])
        self.actionValues[updateIndex] += self.alpha * (G - self.actionValues[updateIndex])

    def terminalUpdateActionValue(self, reward):
        # keep updating until at the end with final
        while len(self.rewardBuffer) > 0:
            # update
            self.updateActionValues()

            # pop from all buffers
            self.rewardBuffer.pop()
            self.actionBuffer.pop()
            self.stateBuffer.pop()


class ExpectedSARSA(TabularSARSA):

    def updateActionValue(self, state, action, reward):
        lastIndex = self.QIndex(self.lastState, self.lastAction)

        # expectedQ = SUM a pi(a|S')Q(S',a)
        expectedQ = self.expectedActionValue(state)

        # Q(S,A) = Q(S,A) + alpha(R + y[SUM a pi(a|S')Q(S',a)]-Q(S,A))
        self.actionValues[lastIndex] += self.alpha * \
                (reward + self.gamma * expectedQ - self.Q(self.lastState, self.lastAction))

    def terminalUpdateActionValue(self, reward):
        raise NotImplementedError('Expected `terminalUpdateActionValue` to be implemented')


    def expectedActionValue(self, state):
        maxActions = [0]
        maxValue = self.Q(state, 0)

        otherValues = [] # action values of non max actions

        for a in range(1, self.numActions):
            value = self.Q(state, a)
            if value > maxValue:
                # copy to otherValues
                for _ in range(len(maxActions)):
                    otherValues.append(maxValue)
                # set maxActions
                maxAction = a
                maxValue = value
            elif value == maxValue:
                maxActions.append(a)
            else:
                otherValues.append(value)

        # since all the max actions have the same value I treat them as a single action
        expected = (1 - self.epsilon) * maxValue
        for q in otherValues:
            expected += self.epsilon/len(otherValues) * q

        return expected

class QLearning(TabularSARSA):

    def updateActionValue(self, state, action, reward):
        lastIndex = self.QIndex(self.lastState, self.lastAction)

        # expectedQ = max a Q(S',a)
        maxQ = self.maxActionValue(state)

        # Q(S,A) = Q(S,A) + alpha(R + y[max a Q(S',a)]-Q(S,A))
        self.actionValues[lastIndex] += self.alpha * \
                (reward + self.gamma * maxQ - self.Q(self.lastState, self.lastAction))

    def maxActionValue(self, state):
        maxValue = self.Q(state, 0)

        for a in range(1, self.numActions):
            value = self.Q(state, a)
            if value > maxValue:
                maxValue = value

        return maxValue

class DifferentialSemiGradientSARSA(BaseAgent):
    numActions = None
    stateRanges = None

    tc = None
    nTiles = None

    alpha = None
    beta = None
    epsilon = None

    averageR = None # R bar
    weights = None

    lastAction = None
    lastState = None

    initialValue = 0
    nullAction = -1

    def __init__(self, parameters):
        self.numActions = parameters["numActions"]
        self.stateRanges = parameters["stateFormat"]

        tilecoderStateRanges = [[s[0] for s in self.stateRanges], [s[1] for s in self.stateRanges]]

        tilings = parameters["tilings"]
        numTiles = parameters["numTiles"]
        self.nTiles = numTiles

        self.tc = representation.TileCoding(
                    input_indices = [np.arange(len(self.stateRanges))],
                    ntiles = [numTiles],
                    ntilings = [tilings],
                    hashing = None,
                    state_range = tilecoderStateRanges,
                    rnd_stream = np.random.RandomState(),
                    bias_term=False)

        self.alpha = parameters["alpha"] / tilings
        self.beta = parameters["beta"]
        self.epsilon = parameters["epsilon"]

        self.averageR = 0
        self.weights = [np.zeros(self.tc.size) for _ in range(self.numActions)]

        self.resetR = parameters["resetR"]

    def getFeatures(self, observation):
        state = np.zeros(self.tc.size)
        indices = self.tc(observation)

        for i in indices:
            state[i] = 1

        return state

    def Q(self, state, action):
        return state.dot(self.weights[action])

    def gradQ(self, state, action):
        # tilecoding gradient is the features
        return state

    def forceAction(self, observation):
        # TODO use env
        if observation[0] == 0:
            return 0
        return None

    def selectAction(self, state, observation):
        forcedAction = self.forceAction(observation)
        if forcedAction != None:
            return forcedAction

        # e greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.numActions)

        # best Q value
        bestActions = [0]
        bestValue = self.Q(state, bestActions[0])
        for a in range(1, self.numActions):
            value = self.Q(state, a)
            if value > bestValue:
                bestActions = [a]
                bestValue = value
            elif value == bestValue:
                bestActions.append(a)

        return np.random.choice(bestActions)

    def updateWeights(self, reward, state, action):
        # delta = R - Rbar + qhat(S',A',w) - qhat(S,A,w)
        delta = reward - self.averageR + self.Q(state, action) - self.Q(self.lastState, self.lastAction)
        #print(np.array_equal(self.lastState, state), self.lastAction)

        # Rbar = Rbar + beta * delta
        self.averageR += self.beta * delta

        # w = w + alpha * delta gradient(qhat(S,A,w))
        self.weights[self.lastAction] += self.alpha * delta * self.gradQ(self.lastState, self.lastAction)

    def terminalUpdateWeights(self, reward):
        # because terminal: qhat(S',A',w) = 0
        delta = reward - self.averageR - self.Q(self.lastState, self.lastAction)

        # Rbar = Rbar + beta * delta
        self.averageR += self.beta * delta

        # w = w + alpha * delta gradient(qhat(S,A,w))
        self.weights[self.lastAction] += self.alpha * delta * self.gradQ(self.lastState, self.lastAction)

    def start(self, observation):
        #print("averageR", self.averageR)

        # TODO do I set averageR = 0 here? ask around
        print("averageR", self.averageR)
        if self.resetR:
            self.averageR = 0

        state = self.getFeatures(observation)

        action = self.selectAction(state, observation)

        self.lastState = state
        self.lastAction = action

        return action

    def step(self, reward, observation):
        state = self.getFeatures(observation)

        # select action
        action = self.selectAction(state, observation)

        # update
        self.updateWeights(reward, state, action)

        self.lastState = state
        self.lastAction = action

        return action

    def end(self, reward):
        self.terminalUpdateWeights(reward)

    def printWeights(self):
        print("weights")
        tilingSize = self.nTiles**len(self.stateRanges)
        print(tilingSize)
        for a in range(len(self.weights)):
            print("action", a)
            for i in range(len(self.weights[a])):
                if i % tilingSize == 0:
                    print()
                w = self.weights[a][i]
                print("{:.3f}".format(w), end=",")
            print()

    def greedyAction(self, state, iterations):
        bestAction = []
        actionValue = -math.inf

        for action in range(self.numActions):
            avgVal = self.sampleStateAction(state, action, iterations)
            if avgVal == self.initialValue:
                continue
            if avgVal == actionValue:
                bestAction.append(action)
            if avgVal > actionValue:
                bestAction = [action]
                actionValue = avgVal

        if len(bestAction) == 0:
            return self.nullAction

        return np.random.choice(bestAction)

    def sampleStateAction(self, state, action, iterations):
        # check if any of the states are set to None
        if not None in state:
            s = self.getFeatures(np.array(state))
            return self.Q(s, action)

        val = 0

        for _ in range(iterations):
            s = np.array(state.copy())
            for i in range(len(s)):
                if s[i] == None:
                    #gen value
                    minR = self.stateRanges[i][0]
                    s[i] = minR + np.random.random()*(self.stateRanges[i][1]-minR)

            s = self.getFeatures(s)
            val += self.Q(s, action)

        return val / iterations

# -1 reverse
# 0 = neutral
# 1 = forward
# weights and alpha not correct
class EpisodicSemiGradSARSA(BaseAgent):

    def __init__(self, parameters, RBF=False):
        self.gamma = 1
        self.epsilon = 0.1
        self.RBF = RBF

        if RBF:
            if "RBFAlpha" not in parameters:
                print("no alpha!")
                parameters["RBFAlpha"] = 0.1/16
            if "RBFSigma" not in parameters:
                print("no sigma!")
                parameters["RBFSigma"] = 1/8
            # representation.RBFCoding(np.array([[.5],[.5],[.5],[.5]]), np.array([[0,0],[0,1],[1,0],[1,1]]), bias_term=False)
            # create weights for 0-1 by 0-1 state space
            #weight = 1/8
            centers_per_dim = 8
            weights = np.array([[parameters["RBFSigma"]] for i in range(centers_per_dim**2)])
            centers = np.array([[i, j] for i in np.linspace(0,1,8) for j in np.linspace(0,1,8)])
            self.rbf = representation.RBFCoding(
                    #np.array([[.5],[.5],[.5],[.5]]),
                    weights,
                    #np.array([[0,0],[0,1],[1,0],[1,1]]),
                    centers,
                    bias_term=False)
            self.weights = [np.zeros(self.rbf.size) for i in range(3)]
            self.alpha = parameters["RBFAlpha"] #0.1 / (12) # not sure if correct may want to find expected value
        else:
            if "Tilings" not in parameters:
                print("no tilings!")
                parameters["Tilings"] = 8
            if "TileAlpha" not in parameters:
                print("no alpha!")
                parameters["TileAlpha"] = 0.1 / parameters["Tilings"] # alpha = 0.1 / E[xTx]
            state_range = [[-1.2, -0.07], [0.5, 0.07]]
            self.tc = representation.TileCoding(
                    input_indices = [np.arange(2)],
                    ntiles = [8],
                    ntilings = [parameters["Tilings"]],
                    hashing = None,
                    state_range = state_range,
                    rnd_stream = np.random.RandomState(),
                    bias_term=False)
            self.weights = [np.zeros(self.tc.size) for i in range(3)]

            self.alpha = parameters["TileAlpha"]

    def start(self, observation):
        # tilecode
        state = self.getFeatures(observation)

        action = self.selectAction(state)

        self.lastState = state
        self.lastAction = action

        return action

    def step(self, reward, observation):
        # tilecode
        state = self.getFeatures(observation)

        # select action
        action = self.selectAction(state)
        #print(self.q(state, action))

        # update
        self.updateWeights(reward, state, action)

        # update last
        self.lastState = state
        self.lastAction = action

        return action

    def end(self, reward):
        # dont need to tilecode

        self.terminalUpdateWeights(reward)

    def getFeatures(self, observation):
        if self.RBF:
            # normalize between 0 and 1
            pos = (observation[0] + 1.2) / (0.5 + 1.2)
            vel = (observation[1] + 0.07) / (0.07 + 0.07)
            state = self.rbf(np.array([[pos, vel]]))
        else:
            indices = self.tc(observation)
            state = np.zeros(self.tc.size)

            for i in indices:
                state[i] = 1

        #print(state)
        return state

    def selectAction(self, state):
        # e greedy
        if np.random.choice([0, 1], p=[1 - self.epsilon, self.epsilon]) == 1:
            return np.random.choice([0,1])

        # best q value
        bestValue = self.q(state, -1)
        bestActions = [-1]
        for i in range(0, 2):
            value = self.q(state,i)
            if value > bestValue:
                bestValue = value
                bestActions = [i]
            elif value == bestValue:
                bestActions.append(i)

        return np.random.choice(bestActions)

    def terminalUpdateWeights(self, reward):
        self.weights[self.lastAction + 1] = self.weights[self.lastAction + 1] \
            + self.alpha * (reward - self.q(self.lastState, self.lastAction)) \
                * self.gradientQ(self.lastState, self.lastAction)

    def updateWeights(self, reward, state, action):
        #code.interact(local=locals())
        self.weights[self.lastAction + 1] = self.weights[self.lastAction + 1] \
            + self.alpha * (reward + self.gamma * self.q(state, action) \
                    - self.q(self.lastState, self.lastAction)) \
                * self.gradientQ(self.lastState, self.lastAction)

    def q(self, state, action):
        if action not in [-1, 0, 1]:
            raise Exception('Action does not exist')
        return state.dot(self.weights[action + 1])

    def gradientQ(self, state, action):
        # same for RBF and tilecoding gradient
        return state
