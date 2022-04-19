from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import math
import gym

class BaseEnvironment:
    @abstractmethod
    def init(self, params: Dict[Any, Any] = {}) -> None:
        raise NotImplementedError('Expected `init` to be implemented')

    @abstractmethod
    def start(self) -> Any:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, action: int) -> Tuple[float, Any, bool, int]: # last is forced action None if
        raise NotImplementedError('Expected `step` to be implemented')

class CliffWalk(BaseEnvironment):
    # 0,0 is in the bottom left
    worldSize = [12,4] # x, y
    state = [] # x, y
    initialState = [0,0] # x, y
    goal = [11,0] # x, y

    rewardMapping = [ # index using 3-y, x
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1]
            ]

    """ actions [x,y]
        0 up [0,1]
        1 right [1,0]
        2 down [0,-1]
        3 left [-1,0]
    """

    numActions = 4
    actionMapping = [(0,1),(1,0),(0,-1),(-1,0)]

    def init(self, params):
        return

    def start(self):
        self.state = self.initialState.copy()

        return self.state

    def step(self, action):
        # move and get new state
        state = [self.state[0] + self.actionMapping[action][0], self.state[1] + self.actionMapping[action][1]]
        # clamp action
        if state[0] < 0:
            state[0] = 0
        elif state[0] > self.worldSize[0] - 1:
            state[0] = self.worldSize[0] - 1
        if state[1] < 0:
            state[1] = 0
        elif state[1] > self.worldSize[1] - 1:
            state[1] = self.worldSize[1] - 1

        # reward of new state
        reward = self.rewardMapping[3-state[1]][state[0]]

        self.state = state

        # check if off cliff
        #if (state[1] == 0 and state[0] >= 1 and state[0] <= 10):
            #print("off cliff", state, reward)

        # if fell off cliff move to start
        if reward == -100:
            self.state = self.initialState

        return (reward, self.state, self.state == self.goal)

    def agentParams(self):
        return {"numActions": self.numActions, "stateFormat": self.worldSize}

class RandomWalk(BaseEnvironment):

    worldSize = [6]
    state = None
    initialState = [3]
    goal = [6]
    failure = [0]

    numActions = 2

    def init(self, params):
        return

    def start(self):
        self.state = self.initialState.copy()
        return self.state

    def step(self, action):
        if (action == 0):
            self.state[0] -= 1
        else:
            self.state[0] += 1

        reward = 0
        end = False

        if self.state == self.goal:
            reward = 1
            end = True

        if self.state == self.failure:
            end = True

        return (reward, self.state, end)

    def agentParams(self):
        return {"numActions": self.numActions, "stateFormat": self.worldSize}

class CartAndPoleEnvironment(BaseEnvironment):

    numDims = 4
    numActions = 2

    state = None # cart pos(x), cart velocity(x dot), pole angle(theta), pole velocity(theta dot)
    #stateRange = [(-4.8,4.8),(-math.inf,math.inf),(-0.418, 0.418),(-math.inf,math.inf)]
    stateRange = [(-4.8,4.8),(-4.8,4.8),(-0.418, 0.418),(-0.418,0.418)]
    initialState = [(-0.005,0.005),(-0.005,0.005),(-0.005,0.005),(-0.005,0.005)]
    termState = [(-2.4, 2.4), None, (-.2095, .2095), None]

    # pole and cart physics information
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    # Neuronlike adaptive elements that can solve difficult learning control problems
    gravity = 9.8 # m/s^2
    cartMass = 1.0 # Kg
    poleMass = 0.1 # Kg
    poleHalfLength = 0.5 # m
    pushForce = 10 # N

    stepLength = 0.2 # s

    totalMass = cartMass + poleMass

    def init(self, params):
        pass

    def start(self):
        self.state = np.zeros(4)

        for i in range(self.numDims):
            self.state[0] = np.random.uniform(self.initialState[i][0], self.initialState[i][1])

        return self.state

    # updates are calculated without force
    def step(self, action):
        trueAction = action * 2 - 1;

        forceApplied = trueAction * self.pushForce

        cosTheta = math.cos(self.state[2])
        sinTheta = math.sin(self.state[2])

        # calculate angle
        angleAcc = (self.gravity * sinTheta + cosTheta * (
                (-forceApplied - self.poleMass * self.state[3]**2 * sinTheta) / self.totalMass
                )) / (self.poleHalfLength * (4/3 - self.poleMass * cosTheta**2 / self.totalMass))
        xAcc = ((forceApplied + self.poleMass * self.poleHalfLength * (self.state[3]**2 - angleAcc * cosTheta))
            / self.totalMass)

        # update x then x velocity
        self.state[0] += self.stepLength * self.state[1]
        self.state[1] += self.stepLength * xAcc
        # update theta then theta velocity
        self.state[2] += self.stepLength * self.state[3]
        self.state[3] += self.stepLength * angleAcc

        self.boundStates()

        done = self.endState()

        reward = 1

        return reward, self.state, done

    def endState(self):
        # check cart position
        if self.state[0] < self.termState[0][0] or self.state[0] > self.termState[0][1]:
            return True

        # check pole angle
        if self.state[2] < self.termState[2][0] or self.state[2] > self.termState[2][1]:
            return True

        return False

    def agentParams(self):
        return {"numActions": self.numActions, "stateFormat": self.stateRange}

    def bound(self, val, bounds): # bounds = [lower, upper]
        if val > bounds[1]:
            return bounds[1]
        if val < bounds[0]:
            return bounds[0]
        return val

    def boundStates(self):
        for i in range(len(self.state)):
            self.state[i] = self.bound(self.state[i], self.stateRange[i])

class AccessControlQueuing(BaseEnvironment):
    # TODO introduce forced drop actions
    possibleJobs = [1,2,4,8]
    state = None # free server #, customer priority
    numServers = 10
    freeServers = None
    currentQueue = None
    completeChance = 0.06

    stateRange = [(0,10),(1,8)]

    # action 0 = drop job, action 1 = run job
    numActions = 2

    def init(self, params):
        pass

    def start(self):
        self.state = np.zeros(2)

        self.freeServers = self.numServers
        self.currentQueue = self.genCustomer()

        self.setState()

        #print("curQueue", self.currentQueue, "state", self.state)

        return self.state

    def step(self, action):
        # address action and calc reward
        reward = None

        self.completeJobs()

        # if no room or drop room
        if self.freeServers <= 0 and action == 1:
            action = 0
            #TODO this is an error

        if action == 0:
            reward = 0
        else:
            self.freeServers -= 1
            reward = self.currentQueue

        self.currentQueue = self.genCustomer()

        # loop until there is room in the queue
        #self.waitForRoom()

        self.setState()

        #print("action", action, "reward", reward, "curQueue", self.currentQueue, "state", self.state)

        return reward, self.state, False

    def setState(self):
        self.state[0] = self.freeServers
        self.state[1] = self.currentQueue

    def completeJobs(self):
        completed = 0
        for _ in range(self.numServers - self.freeServers):
            if np.random.random() < self.completeChance:
                completed += 1

        self.freeServers += completed

    def waitForRoom(self):
        while self.freeServers <= 0:
            self.completeJobs()

    def genCustomer(self):
        return np.random.choice(self.possibleJobs)

    def agentParams(self):
        return {"numActions": self.numActions, "stateFormat": self.stateRange}

class MountainCarEnvironment(BaseEnvironment):
    #position = 0
    #velocity = 0

    numActions = 3

    state = None # position, velocity
    sparseReward = False
    stateRanges = [(-1.2, 5), (-0.07, 0.07)]

    def init(self, params={}):
        self.sparseReward = params["sparseReward"]

    def start(self):
        self.state = np.zeros(2)
        # random start [-0.6, -0.4)
        #self.position = np.random.uniform(-0.6, -0.4)
        self.state[0] = np.random.uniform(-0.6, -0.4)

        #self.velocity = 0

        #state = np.array([self.position, self.velocity])

        return self.state

    def step(self, action):
        self.takeAction(action)

        end = False

        reward = -1
        if self.sparseReward:
            reward = 0

        #if self.position >= 0.5:
        if self.state[0] >= 0.5:
            if self.sparseReward:
                reward = 1
            else:
                reward = 0

            end = True
            #return (0, self.state, True)

        #return (-1, self.state, False)

        return (reward, self.state, end)

    def takeAction(self, action):
        action -= 1

        #self.velocity = self.boundVelocity(self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position))
        self.state[1] = self.boundVelocity(self.state[1] + 0.001 * action - 0.0025 * np.cos(3 * self.state[0]))

        #self.position = self.boundPosition(self.position + self.velocity)
        self.state[0] = self.boundPosition(self.state[0] + self.state[1])

    def bound(self, val, lower, upper):
        if val > upper:
            return upper
        if val < lower:
            return lower
        return val

    def boundVelocity(self, velocity):
        return self.bound(velocity, -0.07, 0.07)

    def boundPosition(self, position):
        return self.bound(position, -1.2, 0.5)

    def agentParams(self):
        return {"numActions": self.numActions, "stateFormat": self.stateRanges}


class GymMountainCarEnvironment(BaseEnvironment):
    envName = "MountainCarCustom-v0"
    env = None

    def __init__(self):
        gym.envs.register(
            id=self.envName,
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=400,      # MountainCar-v0 uses 200
        )

        self.env = gym.make(self.envName)

    def init(self, params):
        pass

    def start(self):
        return self.env.reset()

    def step(self, action): # last is forced action None if
        s = self.env.step(action)
        return s[1], s[0], s[2]

    def agentParams(self):
        stateFormat = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        return {"numActions": self.env.action_space.n, "stateFormat": stateFormat}


class MountainCarEnvironmentCA(MountainCarEnvironment):
    actionRange = (-1, 1)

    def takeAction(self, action):
        # bound Action
        if action < -1:
            action = -1

        if action > 1:
            action = 1

        #self.velocity = self.boundVelocity(self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position))
        self.state[1] = self.boundVelocity(self.state[1] + 0.001 * action - 0.0025 * np.cos(3 * self.state[0]))

        #self.position = self.boundPosition(self.position + self.velocity)
        self.state[0] = self.boundPosition(self.state[0] + self.state[1])

    def agentParams(self):
        # TODO refactor to support both, continuous and descrete action
        return {"actionRange": self.actionRange, "stateFormat": self.stateRanges}

class GymMountainCarEnvironmentCA(GymMountainCarEnvironment):
    envName = "MountainCarContinuousCustom-v0"

    def __init__(self):
        gym.envs.register(
            id=self.envName,
            entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
            max_episode_steps=400,      # MountainCar-v0 uses 200
        )

        self.env = gym.make(self.envName)

    def step(self, action):
        return super().step(np.array([action]))

    def agentParams(self):
        stateFormat = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        actionRange = [self.env.action_space.low, self.env.action_space.high]
        return {"actionRange": actionRange, "stateFormat": stateFormat}

class MeanChasing(BaseEnvironment):
    # state (-1,1)
    # mean is always 1
    # reward is distance from the
    pass
