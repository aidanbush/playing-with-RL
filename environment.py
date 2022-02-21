from abc import abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

class BaseEnvironment:
    @abstractmethod
    def init(self, params: Dict[Any, Any] = {}) -> None:
        raise NotImplementedError('Expected `init` to be implemented')

    @abstractmethod
    def start(self) -> Any:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, action: int) -> Tuple[float, Any, bool]:
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

    def init(self):
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

class MountainCarEnvironment(BaseEnvironment):
    position = 0
    velocity = 0

    def init(self, params={}):
        pass

    def start(self):
        # random start [-0.6, -0.4)
        self.position = np.random.uniform(-0.6, -0.4)

        self.velocity = 0

        state = np.array([self.position, self.velocity])

        return state

    def step(self, action):
        self.velocity = self.boundVelocity(self.velocity + 0.001 * action - 0.0025 * np.cos(3 * self.position))

        self.position = self.boundPosition(self.position + self.velocity)

        state = np.array([self.position, self.velocity])

        if self.position >= 0.5:
            return (0, state, True)

        return (-1, state, False)

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
