# -*- coding: utf-8 -*-
from typing import ForwardRef
import numpy as np
import random
from Graph import *
from aStarGreedy import*
import ast
import torch as py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import operator
import enum

random.seed(1)

class DQN(nn.Module) :
    def __init__(self, learningRate, inputDim, fc1Dim, fc2Dim, nActions):
        super(DQN, self).__init__()
        self.inputDim = inputDim
        self.fc1Dim = fc1Dim
        self.fc2Dim = fc2Dim
        self.nActions = nActions
        self.fc1 = nn.Linear(*self.inputDim, self.fc1Dim)
        self.fc2 = nn.Linear(*self.fc1Dim, self.fc2Dim)
        self.fc3 = nn.Linear(*self.fc2Dim, self.nActions)

        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.device = py.device('cuda:0' if py.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forwardPass(self, state) :
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
        

class agent() : 
    def __init__(self, discountFactor, explorationRate, learningRate, inputDimensions, batchSize, nActions, 
                maxMemory = 100000, explorationRateLow = 0.01, explorationRateDec = 5e-4) : 
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        self.learningRate = learningRate
        self.inputDim = inputDimensions
        # self.batchSize = batchSize
        self.actionSpace = range(nActions)
        # Bør nok have fc1Dim og fcDim2 som inputs til constructor så vi kan variere lageret.
        self.onlineDqn = DQN(self.learningRate, nActions = nActions, inputDim=inputDimensions, fc1Dim=5, fc2Dim=5)        
        self.targetDqn = DQN(self.learningRate, nActions = nActions, inputDim=inputDimensions, fc1Dim=5, fc2Dim=5)        
        
        
        # self.memoryCounter = 0
        # self.memorySize = maxMemory

        # self.stateMemory = np.zeros((self.memorySize, *inputDimensions),dType=np.float32)
        # self.newStateMemory = np.zeros((self.memorySize, *inputDimensions),dType=np.float32)
        # self.actionMemory = np.zeros(self.memorySize, dType=np.int32)
        # self.rewardMemory = np.zeros(self.memorySize, dType=np.float32)
        # self.terminalMemory = np.zeros(self.memorySize, dType=np.bool)

    # def storeTransition(self, state, action, reward, nextState, done):
    #     index = self.memoryCounter % self.memorySize
    #     self.stateMemory[index] = state
    #     self.newStateMemory[index] = nextState
    #     self.actionMemory[index] = action
    #     self.rewardMemory[index] = reward
    #     self.terminalMemory = done

    #     self.memoryCounter += 1

        

    def chooseAction(self, observation):
        if np.random.random() > self.explorationRate:
            state = py.tensor([observation]).to(self.dqn.device)
            outputActions = self.dqn.forwardPass(state)
            # Choose best action
            action = py.argmax(outputActions).item()
        else: 
            action = np.random.choice(self.actionSpace)
        return action
    
    
    def learn(self):
        # if self.memoryCounter < self.batchSize:
        #     return
        
        self.onlineDqn.optimizer.zero_grad()
        self.targetDqn.optimizer.zero_grad()

        qPredict = self.onlineDqn.forwardPass()


        # maxMem = min(self.memoryCounter, self.memorySize)
        # batch = np.random.choice(maxMem, self.batchSize, replace = False)

        # batchIndex = np.arrange(self.batchSize, dType = np.int32)

        # stateBatch = py.tensor(self.stateMemory[batch]).to(self.dqn.device)
        # newStateBatch = py.tensor(self.newStateMemory[batch]).to(self.dqn.device)
        # rewardBatch = py.tensor(self.rewardMemory[batch]).to(self.dqn.device)
        # terminalBatch = py.tensor(self.terminalMemory[batch]).to(self.dqn.device)

        # actionBatch = self.actionMemory[batch]

        # qEval = self.dqn.forwardPass(stateBatch)[batchIndex, actionBatch]
        # qNext = self.dqn.forwardPass(newStateBatch)
        # qNext[terminalBatch] = 0.0

        # qTarget = rewardBatch + self.discountFactor * py.max(qNext, dim=1)[0]
        
        loss = self.dqn.loss(qTarget, qEval).to(self.dqn.device)
        loss.backward
        self.dqn.optimizer.step()
        
        self.epsilon = self.explorationRate - self.explorationRateDec if self.explorationRate > self.explorationRateLow else self.explorationRateLow
            

class nodeType(enum.Enum):
    normal = 1
    goal = 2
    location = 3
    locationAndGoal = 4

class environment():
    
    def  __init__(self, graph):
        self.graph = graph
       
    def step(self, state, action):
        stateId = -1
        for i in range(len(state)):
            if state[i] == nodeType.location or state[i] == nodeType.locationAndGoal:
                stateId = i
        
        if stateId == -1:
            raise Exception("No location found in input.")

        currentNode = self.graph[stateId]

        if action > len(currentNode.connections):
            return state, 0

        road = currentNode.connections[action]
        reward = road.weight
        nextNode = road.destination

        state[stateId] = nodeType.normal
        
        if nextNode.isGoal():
            state[nextNode.id] = nodeType.locationAndGoal
        else: 
            state[nextNode.id] = nodeType.location

        return state, reward
        


















# environment = [
#     [0, 2, 1, 0, 0],
#     [0, 0, 0, 3, 0],
#     [0, 0, 0, 2, 4],
#     [0, 0, 1, 0, 2],
#     [0, 0, 0, 1, 0]
# ]


# #agent = agent(discountFactor=0.8, explorationRate=0.1, learningRate=0.03, inputDimensions=[10], batchSize=50, nActions=2,explorationRateLow=0.01)
# #costs, episodes = [], []
# #episodes = 500


# def initializeLayer(numRows, numColoumns):
#     layer = []
#     for i in range(numRows):
#         layer.append([])
#         for j in range(numColoumns):
#             layer[i].append(py.tensor(random.random()))
#     return layer
        

# #7x5 matrix of input
# input1 = [[]]
# #input converted to 35x1 array
# input1 = np.array(input1).flatten()
# input2 = input1
# input2[30] = 0
# input2[31] = 10
# input3 = input1
# input3[30] = 0
# input3[32] = 10
# input4 = input1
# input4[30] = 0
# input4[33] = 10
# input5 = input1
# input5[30] = 0
# input5[34] = 10
# #inputTensor = py.tensor(input1)

# inputRows = len(input1)

# #input: 35x1
# #hidden1: 10x1
# #hidden2: 10x1
# #output: 3x1

# #weights from layer 1(input) to 2:  l2 = w1*l1 + b1: 10x1
# weights1 = initializeLayer(10,inputRows)
# biases1 = initializeLayer(10,1)

# #weights from layer 2 to 3:         l3 = w2*l2 + b2: 10x1
# weights2 = initializeLayer(10,10)
# biases2 = initializeLayer(10,1)

# #weights from layer 3 to 4(output): l4 = w3*l3 + b3: 3x1
# weights3 = initializeLayer(3,10)
# biases3 = initializeLayer(3,1)


# learningRate = 0.01

# for i in range(1000):
#     print("hellll--------------------------------------------------------")
#     layer2 = np.dot(weights1,input1) + biases1
#     layer3 = np.dot(weights2,layer2) + biases2
#     output = np.dot(weights3,layer3) + biases3

#     allParameters = []
#     #np.array(input1).flatten()
#     #allParameters.append(np.array(weights1).flatten()).append(np.array(weights2).flatten()).
#     #    append()
#     #append(np.array(weights3).flatten()).append(np.array(biases1).flatten()).append(
#     #    np.array(biases2).flatten().append(np.array(biases3).flatten()))

        
#     for parameter in allParameters:
#         parameter.grad.zero_()   

#     yExpected = [1,2,3]

#     #insert loss function
#     loss = output - yExpected

#     loss.backward()

#     for parameter in allParameters:
#         parameter.data -= parameter.grad*learningRate
#         parameter.grad.zero_()   

    





