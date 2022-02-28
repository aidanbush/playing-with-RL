import environment as e
import agent as a
from rl_glue import RlGlue

import numpy as np
import statistics
import time
import code

import matplotlib.pyplot as plt
import matplotlib.style as style

MAX_STEPS = 10000
# types
EPISODIC="episodic"
CONTINUOUS="continuous"

def runTest(numRuns, numEpisodes, agent, env, parameters, testParams, report=False):
    stepsData = []
    stepsMeans = np.zeros(numEpisodes)
    stepsStdev = np.zeros(numEpisodes)

    returnsData = []
    returnsMeans = np.zeros(numEpisodes)
    returnsStdev = np.zeros(numEpisodes)

    for i in range(numRuns):
        steps, returns = runAgent(numEpisodes, agent, env, parameters, testParams, report=report)
        if report:
            print("Run", i)
        stepsData.append(steps)
        returnsData.append(returns)

    # for each episode
    for i in range(numEpisodes):
        steps = []
        returns = []
        # for each run
        for j in range(numRuns):
            steps.append(stepsData[j][i])
            returns.append(returnsData[j][i])
        stepsMeans[i] = statistics.mean(steps)
        stepsStdev[i] = statistics.stdev(steps)
        returnsMeans[i] = statistics.mean(returns)
        returnsStdev[i] = statistics.stdev(returns)

    return stepsMeans, stepsStdev, returnsMeans, returnsStdev

def runAgent(numEpisodes, agentClass, envClass, parameters, testParams={"maxSteps":MAX_STEPS}, report=False):
    # seed np
    np.random.seed(int(time.time()))

    # create env
    env = envClass()
    parameters.update(env.agentParams())

    # create agent
    agent = agentClass(parameters)

    # create glue object
    glue = RlGlue(agent, env)

    steps = []
    returns = []

    # run test
    for i in range(numEpisodes):
        glue.runEpisode(max_steps=testParams["maxSteps"])
        if report:
            print("Episode", i, glue.num_steps, "steps")
        steps.append(glue.num_steps)
        returns.append(glue.total_reward)
    #agent.printValues() # for testing value and action value values
    #agent.printWeights()
    #plotActions(agent, env, 21)


    return steps, returns

def plotActions(agent, environment, resolution):
    stateRanges = environment.agentParams()["stateFormat"]
    numStates = len(stateRanges)
    numActions = environment.agentParams()["numActions"]
    actionColours = plt.cm.get_cmap("hsv", numActions)
    nonActionColour = (0.0, 0.0, 0.0, 1.0)
    # assume stateRange >= 2
    if numStates < 2:
        # plot line
        return

    # create figure with len(stateRange) * len(stateRange) plots
    fig, ax = plt.subplots(numStates, numStates)

    # for each pair of environment states
    stateSampleIterations = 200
    for s1 in range(numStates):
        print("s1",s1)
        for s2 in range(numStates):
            print("s2",s2)
            imshowList = np.zeros((resolution+1, resolution+1, 4))
            # loop through values
            for i in range(resolution+1):
                for j in range(resolution+1):
                    # if i == j only loop through j
                    state = [None for _ in range(numStates)]

                    s1Min = stateRanges[s1][0]
                    s1Max = stateRanges[s1][1]
                    s2Min = stateRanges[s2][0]
                    s2Max = stateRanges[s2][1]

                    s1Val = s1Min + (i/resolution)*(s1Max-s1Min)
                    s2Val = s2Min + (j/resolution)*(s1Max-s1Min)

                    state[s1] = s1Val
                    state[s2] = s2Val

                    action = agent.greedyAction(state, stateSampleIterations)
                    colour = nonActionColour

                    if action != agent.nullAction:# unupdated
                        colour = actionColours(action)
                    for imI in range(4):
                        imshowList[i][j][imI] = colour[imI]
            ax[s1,s2].imshow(imshowList)
    #
    #ax[0][0].legend()
    plt.show()

def main():
    # numRuns = 30
    # numEpisodes = 250
    # agent = a.TabularSARSA
    # agent = a.ExpectedSARSA
    # agents = [a.TabularSARSA, a.QLearning, a.ExpectedSARSA]
    # parameters = {"gamma": 1, "alpha": 0.1, "epsilon": 0.1}
    # agents = [a.TabularSARSA, a.NStepSARSA]
    # env = e.CliffWalk
    # parameters = {"gamma": 1, "alpha": 0.1, "epsilon": 0.1, "n_steps": 5}
    # testParams = {"algType": EPISODIC, "maxSteps":MAX_STEPS}

    # numRuns = 10
    # numEpisodes = 250
    agents = [a.DifferentialSemiGradientSARSA, a.RandomAgent]
    # env = e.CartAndPoleEnvironment
    # parameters = {"alpha": 0.1, "beta":0.05, "epsilon": 0.1, "tilings":10, "numTiles":20}
    numRuns = 5
    numEpisodes = 2
    env = e.AccessControlQueuing
    parameters = {"alpha": 0.1, "beta":0.2, "epsilon": 0.1, "tilings":2, "numTiles":11}
    testParams = {"algType": CONTINUOUS, "maxSteps":2000}

    data = []
    for agent in agents:
        data.append(runTest(numRuns, numEpisodes, agent, env, parameters, testParams, report=True))

    # plot steps
    for i in range(len(agents)):
        plt.plot(data[i][0], label=agents[i].__name__)
        plt.fill_between(np.arange(len(data[i][0])), data[i][0]-data[i][1], data[i][0]+data[i][1], alpha=0.5)

    plt.title("steps")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()

    # plot returns
    for i in range(len(agents)):
        plt.plot(data[i][2], label=agents[i].__name__)
        plt.fill_between(np.arange(len(data[i][2])), data[i][2]-data[i][3], data[i][2]+data[i][3], alpha=0.5)

    plt.title("return")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()

main()
