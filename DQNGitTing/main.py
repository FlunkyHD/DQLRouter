"""
Script containing the training and testing loop for DQNAgent
"""
import sys
import numpy as np
from environment import *
from Graph import Node, Road



from agent import DQNAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# While loop should probably be changed to a counting loop where done is not the condition. 
def fill_memory(env, dqn_agent, num_memory_fill_eps, randomReset, maxGoals, alwaysCapGoals, resetState):
    """
    Function that performs a certain number of episodes of random interactions
    with the environment to populate the replay buffer

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_memory_fill_eps: int
        Number of episodes of interaction to be performed

    Returns
    ---
    none
    """

    for _ in range(num_memory_fill_eps):
        done = False
        state = []
        if(randomReset == True) :
            state = env.randomReset(maxGoals=maxGoals, alwaysCapGoals=alwaysCapGoals)
        else : 
            state = env.reset()
            
        while not done:
            action = env.sampleAction()
            next_state, reward, done = env.step(state,action)
            dqn_agent.memory.store(state=state, 
                                action=action, 
                                next_state=next_state, 
                                reward=reward, 
                                done=done)
            state = next_state


def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, 
          filePath, fileName, randomReset, maxGoals, alwaysCapGoals):
    """
    Function to train the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_train_eps: int
        Number of episodes of training to be performed
    num_memory_fill_eps: int
        Number of episodes of random interaction to be performed
    update_frequency: int
        Number of steps after which the target models must be updated
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform a learning step
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment
    
    Returns
    ---
    none
    """
#def fill_memory(env, dqn_agent, num_memory_fill_eps, randomReset, maxGoals, alwaysCapGoals, resetState):
    
    fill_memory(env, dqn_agent, num_memory_fill_eps, randomReset, maxGoals, alwaysCapGoals, randomReset)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    
    reward_history = []
    epsilon_history = []

    step_cnt = 0
    best_score = np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.epsilon)

        done = False
        state = []
        if(randomReset == True) :
            state = env.randomReset(maxGoals=maxGoals, alwaysCapGoals=alwaysCapGoals)
        else : 
            state = env.reset()

        ep_score = 0

        while not done:
            #print(state)
            action = dqn_agent.select_action(state)
            #print(action)
            next_state, reward, done = env.step(state, action)
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)

            dqn_agent.learn(batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1
            

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:]) # moving average of last 100 episodes
        if doPrint != 0 :
            if ep_cnt % doPrint == 0:  
                print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'.format(ep_cnt, step_cnt, ep_score, current_avg_score, epsilon_history[-1]))

        if current_avg_score <= best_score:
            # dqn_agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score
        if( ep_cnt > 0 ) : #and ep_cnt % 100 == 0
            dqn_agent.save_model(fileName=fileName, filePath=filePath)
            


def test(env, dqn_agent, num_test_eps, randomReset, maxGoals, alwaysCapGoal):
    """
    Function to test the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_test_eps: int
        Number of episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment

    Returns
    ---
    none
    """

    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        if randomReset is True:
            state = env.randomReset(maxGoals, alwaysCapGoal)
        else:
            state = env.reset()
        
        while not done:

            action = dqn_agent.select_action(state)
            next_state, reward, done = env.step(state, action)
            #print(state, action)
            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

   # with open('{}/test_reward_history_{}.pkl'.format(results_basepath, seed), 'wb') as f:
        #pickle.dump(reward_history, f)
        
def makeLeftRightGraph() :
    graph = []
    node0 = Node(False, 0)
    node1 = Node(False, 1)
    node2 = Node(False, 2)
    node3 = Node(False, 3)
    node4 = Node(False, 4)
    
    node0.addConnection(Road(1000,node0))
    node0.addConnection(Road(2,node1))
    node1.addConnection(Road(2,node0))
    node1.addConnection(Road(2,node2))
    node2.addConnection(Road(2,node1))
    node2.addConnection(Road(2,node3))
    node3.addConnection(Road(2,node2))
    node3.addConnection(Road(2,node4))
    node4.addConnection(Road(2,node3))
    node4.addConnection(Road(1000,node4))

    graph.append(node0)
    graph.append(node1)
    graph.append(node2)
    graph.append(node3)
    graph.append(node4)
    return graph

def makeGridGraphWithlength(length) :
    completeGraph = []
    nodeArr = []
    for i in range (length*length) :
        nodeArr.append(Node(name=i))
    
    for k in range (4) : 
        for i in range (length) : 
            for j in range (length) : 
                if k == 0 : 
                    if j % length == 0 : #Her får vi de mest venstre noder. 
                        nodeArr[j + i*length].addConnection(Road(1000, nodeArr[j + i*length], str(j + i*length) +"-"+ str(j + i*length)  ))
                    else :
                        nodeArr[j+i*length].addConnection(Road(2, nodeArr[j+(i*length)-1], str(j+i*length) +"-"+ str(j+(i*length)-1)))
                elif k == 1 :
                    if i == 0 : 
                         nodeArr[j].addConnection(Road(1000, nodeArr[j], str(j) + "-"+str(j)))
                    else :
                        nodeArr[j+i*length].addConnection(Road(2, nodeArr[j+(i-1)*length], str(j+i*length) +"-"+ str(j+(i-1)*length))) 
                elif k == 2 :
                    if j +1 == length :  
                        nodeArr[j + i*length].addConnection(Road(1000, nodeArr[j + i*length], str(j + i*length) +"-"+ str(j + i*length)))
                    else :
                        nodeArr[j+i*length].addConnection(Road(2, nodeArr[j+(i*length)+1], str(j + i*length) +"-"+ str(j + i*length +1)))
                elif k == 3 :
                    if (i + 1) == length : 
                         nodeArr[j + i*length].addConnection(Road(1000, nodeArr[j + i*length], str(j + i*length) + "-"+str(j + i*length)))
                    else :
                        nodeArr[j+i*length].addConnection(Road(2, nodeArr[j+(i+1)*length], str(j + i*length) + "-"+str(j + (i+1)*length))) 
                               
    for i in range (length*length) :
        completeGraph.append(nodeArr[i])  
    return completeGraph

def makeGridGraph() :
    graph = []
    node0 = Node(False, 0)
    node1 = Node(False, 1)
    node2 = Node(False, 2)
    node3 = Node(False, 3)
    node4 = Node(False, 4)
    node5 = Node(False, 5)
    node6 = Node(False, 6)
    node7 = Node(False, 7)
    node8 = Node(False, 8)
    
    node0.addConnection(Road(1000,node0))
    node0.addConnection(Road(1000,node0))
    node0.addConnection(Road(2,node1))
    node0.addConnection(Road(2,node3))
    
    node1.addConnection(Road(2,node0))
    node1.addConnection(Road(1000,node1))
    node1.addConnection(Road(2,node2))
    node1.addConnection(Road(2,node4))
    
    node2.addConnection(Road(2,node1))
    node2.addConnection(Road(1000,node2))
    node2.addConnection(Road(1000,node2))
    node2.addConnection(Road(2,node5))
    
    node3.addConnection(Road(1000,node3))
    node3.addConnection(Road(2,node0))
    node3.addConnection(Road(2,node4))
    node3.addConnection(Road(2,node6))
    
    node4.addConnection(Road(2,node3))
    node4.addConnection(Road(2,node1))
    node4.addConnection(Road(2,node5))
    node4.addConnection(Road(2,node7))
    
    node5.addConnection(Road(2,node4))
    node5.addConnection(Road(2,node2))
    node5.addConnection(Road(1000,node5))
    node5.addConnection(Road(2,node8))
    
    node6.addConnection(Road(1000,node6))
    node6.addConnection(Road(2,node3))
    node6.addConnection(Road(2,node7))
    node6.addConnection(Road(1000,node6))
    
    node7.addConnection(Road(2,node6))
    node7.addConnection(Road(2,node4))
    node7.addConnection(Road(2,node8))
    node7.addConnection(Road(1000,node7))

    node8.addConnection(Road(2,node7))
    node8.addConnection(Road(2,node5))
    node8.addConnection(Road(1000,node8))
    node8.addConnection(Road(1000,node8))
    
    graph.append(node0)
    graph.append(node1)
    graph.append(node2)
    graph.append(node3)
    graph.append(node4)
    graph.append(node5)
    graph.append(node6)
    graph.append(node7)
    graph.append(node8)
    
    return graph
    
def pruneGridGraph(state,nodeGraph ) :
    offset = len(state)*len(state)
    graphLength = len(state)/2
    goalLocgraph = []
    for i in range (graphLength) : 
        if(state[i] == 1) :     #Adds the goals
            goalLocgraph.append(1)
        elif(state[i+offset] == 1) : #Adds the location
            goalLocgraph.append(1)
        else :
            goalLocgraph.append(0)
    
    for i in range (len(goalLocgraph)) :
        if(goalLocgraph[i] == 0) : #We have a node that should be pruned. 
            removeNode = nodeGraph[i]
            connections = removeNode.connections
            for removeRoad in connections : 
                for road in connections : 
                    #nodeAddNewCoonection = 
                    print("")
            
    
def procesInput(input) : 
    
    if(len(input) != 6) :
        print("Wrong amount of inputs: THe format is:\n",
              "Print: which is how often it prints(0-ininity)\n",
              "FileName, Which says which file it should be saved as(model.txt)\n",
              "LoadModel, wither it should load a given model, or start from a new(no, filename.txt)\n",
              "DiscountFactor(0.2-1)\n",
              "EpisodeCount, which is amount of episodes(100000)\n",
              "A default call is: 1 4by44Loc.txt no 0.7 100000")
        exit()
    print(input, "Test")
    
    doPrint = int(input[1])
    fileName = input[2]
    loadModel = input[3]
    discountFactor = float(input[4])
    episodeCount = int(input[5])
        
    whichGraph = "Grid"        #USED FOR SPECIFING WHICH GRAPH.  "Grid"  "leftRight"
    gridSize = 4
    maxGoals = 3                #Used for speciging the max amount of goals in the graph.     
    randomReset = True          #Used to set if the reset should be random, so the loactino and goals are randomly placed
    alwaysCapGoals = False      #Specifies if there should always be goals equal to the maxgoals, or if it should be fewer aswell
    shouldTrain = True
    
    if(whichGraph == "Grid") :
        graph = makeGridGraphWithlength(gridSize)
        actionsSpace = 4
    elif whichGraph == "leftRight" :
        graph = makeLeftRightGraph()
        actionsSpace = 2

        
    return doPrint, graph, maxGoals, actionsSpace, fileName, randomReset, alwaysCapGoals, loadModel, discountFactor, episodeCount, shouldTrain

def printGridWithNodes(graph, length) : 
    for i in range (length *length) : 
        print(graph[i].name)
        for j in range (4) : 
            print(graph[i].connections[j].name)
        print("\n")

def printGridAsGrid(state, length) :
    print("Goals and Locations. goal = 2, loc = 1")
    offset = length*length
    for i in range (length) : 
        for j in range (length) :
            if(state[j + i*length] == 1) : #Goals
                print("2", end = '')
            elif(state[j+i*length+offset] == 1) :
                print("1", end = '')
            else : 
                print("-", end = '')
        print()        

if __name__ ==  '__main__':
    filePath = "./DQNGitTing/saveModel/" #Virker nødevendigt. 
    doPrint, graph, maxGoals, actionsSpace, fileName, randomReset, alwaysCapGoals, loadModel, discountFactor, episodeCount, shouldTrain = procesInput(sys.argv)
    np.random.seed()
    torch.manual_seed(1)
    env = environment(graph, actionsSpace)
    
    dqn_agent = DQNAgent(device, 
                            inputSize=len(graph) * 2, 
                            actionSize=actionsSpace, 
                            discount= discountFactor, 
                            train_mode=False, 
                            eps_min = 0.01,
                            eps_max= 1)
    if(loadModel != 'no') :
        dqn_agent.load_model(fileName=loadModel, filePath = filePath)
    
    if(shouldTrain) : 
        train(env=env, 
                dqn_agent=dqn_agent, 
                # results_basepath=args.results_folder, 
                num_train_eps=episodeCount, 
                num_memory_fill_eps=20, 
                update_frequency=1000,
                batchsize=64, 
                fileName=fileName,
                filePath=filePath, 
                randomReset=randomReset, 
                maxGoals=maxGoals, 
                alwaysCapGoals=alwaysCapGoals)
    else : 
        test(env,dqn_agent,
             episodeCount,
             randomReset=randomReset,
             maxGoals=maxGoals,
             alwaysCapGoal=alwaysCapGoals)
    
#def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, 
#          filePath, fileName, randomReset, maxGoals, resetState):

  #  testRandomTestsReturnAverage(env, dqn_agent, 1, alwaysCapGoals = False, numberOfTests = 100)



#def testRandomTestsReturnAverage(env, dqnAgent, maxGoal, alwaysCapGoals = False, numberOfTests = 100) :
    #     dqnAgent.epsilon = 0
#     reward_history = []
#     stepHistory = []

#     for i in range(numberOfTests):

#         done = False
#         #state = env.randomReset(maxGoal, alwaysCapGoals)
#         state = env.reset()
#         step_cnt = 0

#         ep_score = 0

#         while not done:
#             action = dqnAgent.select_action(state)
#             next_state, reward, done = env.step(state, action)

#             state = next_state
#             ep_score += reward
#             step_cnt += 1
#             if(step_cnt > len(env.graph) *3 + 1) :
#                 print("Svær opgave")
#                 print(state)
#                 break 

#         reward_history.append(ep_score)
#         stepHistory.append(step_cnt)

#     averageReward = sum(reward_history) / numberOfTests
#     print(averageReward)
