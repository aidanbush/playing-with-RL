import environment as e
import agent as a
from rl_glue import RlGlue

import numpy as np
import torch
import statistics
import time
import code
import random

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.lines import Line2D

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
        returnsMeans[i] = statistics.mean(returns)
        if numRuns > 1:
            stepsStdev[i] = statistics.stdev(steps)
            returnsStdev[i] = statistics.stdev(returns)
        else:
            stepsStdev[i] = 0
            returnsStdev[i] = 0

    return stepsMeans, stepsStdev, returnsMeans, returnsStdev

def runAgent(numEpisodes, agentClass, envClass, parameters, testParams={"maxSteps":MAX_STEPS}, report=False):
    # seed np
    np.random.seed(int(time.time() % 10000 * 1000))

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
    #plotActions(agent, env, 40)


    return steps, returns

def plotActions(agent, environment, resolution):
    stateRanges = environment.agentParams()["stateFormat"]
    # if not continuous min is 0
    if type(stateRanges[0]) == int:
        stateRanges = [(0, s) for s in stateRanges]

    numStates = len(stateRanges)
    numActions = environment.agentParams()["numActions"]

    actionColours = plt.cm.get_cmap("hsv", numActions+1)
    nonActionColour = (0.0, 0.0, 0.0, 1.0)

    print(stateRanges)

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
            s1Min = stateRanges[s1][0]
            s1Max = stateRanges[s1][1]
            s2Min = stateRanges[s2][0]
            s2Max = stateRanges[s2][1]

            imshowList = np.zeros((resolution+1, resolution+1, 4))
            # loop through values
            for i in range(resolution+1):
                if s1 == s2:
                    break
                for j in range(resolution+1):
                    state = [None for _ in range(numStates)]

                    s1Val = s1Min + (i/resolution)*(s1Max-s1Min)
                    s2Val = s2Min + (j/resolution)*(s2Max-s2Min)

                    state[s1] = s1Val
                    state[s2] = s2Val

                    action = agent.greedyAction(state, stateSampleIterations)
                    colour = nonActionColour

                    if action != agent.nullAction:# unupdated
                        colour = actionColours(action)

                    # copy colour
                    for imI in range(4):
                        imshowList[i][j][imI] = colour[imI]
                    #print("[{:.2f},{:.2f}]".format(state[0],state[1]), end=",")
            #print()
            ax[s1][s2].imshow(imshowList)
            ax[s1][s2].xaxis.tick_top()
            ax[s1][s2].xaxis.set_label_position('top')

            ax[s1][s2].set_xlabel(s1)
            ax[s1][s2].set_ylabel(s2)

    lines = [Line2D([0], [0], color=colour, lw=4) for colour in [nonActionColour]+[actionColours(i) for i in range(numActions)]]
    ax[0][0].legend(lines, ["none"] + list(range(numActions)))
    plt.show()

def plotData(data, labels):
    # plot steps
    plt.figure(figsize=(16,16))
    for i in range(len(data)):
        label = labels[i]
        plt.plot(data[i][0], label=label)
        plt.fill_between(np.arange(len(data[i][0])), data[i][0]-data[i][1], data[i][0]+data[i][1], alpha=0.5)

    plt.title("steps")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
    #plt.savefig("step_plot.pdf")

    plt.figure(figsize=(16,16))

    # plot returns
    for i in range(len(data)):
        label = labels[i]
        plt.plot(data[i][2], label=label)
        plt.fill_between(np.arange(len(data[i][2])), data[i][2]-data[i][3], data[i][2]+data[i][3], alpha=0.5)

    plt.title("return")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
    #plt.savefig("return_plot.pdf")

def parameterSweep(numRuns, numEpisodes, agent, env, params, testParams):
    # only parameters at a time
    sweepParam = None
    for key in params.keys():
        if isinstance(params[key], list):
            if sweepParam != None:
                print("only test one param at a time")
                return
            sweepParam = key

    sweepVals = params[sweepParam]
    data = []
    labels = []
    
    for pVal in sweepVals:
        labels.append(agent.__name__ + " " + sweepParam + " " + str(pVal))
        sweepParams = params.copy()
        sweepParams[sweepParam] = pVal
        print("running sweep", pVal, len(labels), "/", len(sweepVals))
        data.append(runTest(numRuns, numEpisodes, agent, env, sweepParams, testParams, report=True))

    # plot
    plotData(data, labels)

    # params plots

def basicTest():
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

    #numRuns = 5
    #numEpisodes = 100
    #agents = [a.DifferentialSemiGradientSARSA, a.RandomAgent]
    #env = e.CartAndPoleEnvironment
    #parameters = [{"alpha": 0.1, "beta":0.1, "epsilon": 0.1, "tilings":10, "numTiles":20, "resetR":True}, {}]
    #testParams = {"algType": CONTINUOUS, "maxSteps":1000}

    #numRuns = 2
    #numEpisodes = 3#75
    #agents = [a.DifferentialSemiGradientSARSA, a.RandomAgent]
    #agents = [a.DifferentialSemiGradientSARSA, a.RandomAgent, a.SpecificAction, a.SpecificAction, a.AccessControlSol]
    #env = e.AccessControlQueuing
    #parameters = [{"alpha": 0.01, "beta":0.01, "epsilon": 0.1, "tilings":2, "numTiles":21, "resetR":False}, {}, {"action":1}, {"action":0}, {}]
    #testParams = {"algType": CONTINUOUS, "maxSteps":1000000}

    #numRuns = 5
    #numEpisodes = 200
    #env = e.MountainCarEnvironment
    #agents = [a.EpisodicSemiGradSARSA, a.EpisodicSemiGradSARSA, a.EpisodicSemiGradSARSA]
    #parameters = [{"alpha": 0.05, "gamma":1, "epsilon": 0.01, "tilings":24, "numTiles":16}, {"alpha": 0.05, "gamma":1, "epsilon": 0.05, "tilings":24, "numTiles":16}, {"alpha": 0.05, "gamma":1, "epsilon": 0.1, "tilings":24, "numTiles":16}]
    #agents = [a.EpisodicSemiGradSARSA]
    #parameters = [{"alpha": 0.05, "gamma":1, "epsilon": 0.1, "tilings":24, "numTiles":8}]
    #testParams = {"algType": EPISODIC, "maxSteps":MAX_STEPS, "sparseReward": True}

    #numRuns = 2
    #numEpisodes = 200#1000
    #agents = [a.EpisodicActorCritic]
    #env = e.MountainCarEnvironmentCA
    #env = e.GymMountainCarEnvironmentCA
    #parameters = [{"alphaW": 0.01, "alphaTheta": 0.001, "gamma":1, "tilings":8, "numTiles":8, "tau":0.01, "softplus":True, "softplusBeta":1}]
    #parameters = [{"alphaW": 0.01, "alphaTheta": 0.001, "gamma":1, "tilings":8, "numTiles":8, "tau":0, "softplus":True, "softplusBeta":1}]
    #testParams = {"algType": EPISODIC, "maxSteps":2500}

    numRuns = 2
    numEpisodes = 500#1000
    agents = [a.DQN]
    #env = e.MountainCarEnvironment
    env = e.GymMountainCarEnvironment
    parameters = [{"alpha": 0.001, "gamma":0.99, "epsilon": 0.1, "targetUpdate": 1, "batchSize": 32, "bufferCap":4096}] # mountain car - copy from shivams works
    #parameters = [{"alpha": 2**-11, "gamma":0.99, "epsilon": 0.1, "targetUpdate": 1, "batchSize": 32, "bufferCap":100000}] # mountain car - the larger buffer helps a lot
    #testParams = {"algType": EPISODIC, "maxSteps":1000}
    #env = e.CartAndPoleEnvironment
    #parameters = [{"alpha": 0.001, "gamma":0.99, "epsilon": 0.1, "targetUpdate": 1, "batchSize": 32, "bufferCap":4096}] # cart and pole - works
    testParams = {"algType": EPISODIC, "maxSteps":400, "sparseReward":False}

    #numRuns = 2
    #numEpisodes = 100
    #agents = [a.DQNTabular]
    #parameters = [{"alpha": 0.001, "gamma":0.9, "epsilon": 0.05, "targetUpdate": 1, "batchSize": 128, "bufferCap":10000}]
    #agents = [a.TabularSARSA, a.DQNTabular]
    #parameters = [{"gamma": 1, "alpha": 0.1, "epsilon": 0.1},{"alpha": 0.001, "gamma":0.9, "epsilon": 0.05, "targetUpdate": 1, "batchSize": 128, "bufferCap":10000}]
    #env = e.CliffWalk
    #testParams = {"algType": EPISODIC, "maxSteps":200}

    data = []
    labels = []

    for i in range(len(agents)):
        labels.append(agents[i].__name__ + " " + str(i))
        agent = agents[i]
        params = parameters
        if isinstance(parameters, list):
            params = parameters[i]
        data.append(runTest(numRuns, numEpisodes, agent, env, params, testParams, report=True))

    plotData(data, labels)

def sweepTest():
    numRuns = 5
    numEpisodes = 400#1000
    agent = a.DQN
    env = e.MountainCarEnvironment
    parameters = {"alpha": [2**-11, 2**-10, 2**-9, 2**-8], "gamma":0.999, "epsilon": 0.1, "targetUpdate": 1, "batchSize": 32, "bufferCap":100000}
    testParams = {"algType": EPISODIC, "maxSteps":1000}

    parameterSweep(numRuns, numEpisodes, agent, env, parameters, testParams)

def main():
    torch.set_num_threads(torch.get_num_threads()*2-2)
    basicTest()
    #sweepTest()
    return

main()
