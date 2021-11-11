# -*- coding: utf-8 -*-
import numpy as np
import random
from Graph import *
from aStarGreedy import*
import ast
import operator
random.seed(100)

def qLearningOnGraph(qTabel, visitTabel, episode, step, discountFactor=None, epsilon=None, nodes=None):
    for i in range(episode):
        if((i % 100)== 0) :
            print(i/episode * 100, "%")
        nodeIndex = random.randrange(0,len(nodes))
        currentState = nodes[nodeIndex]
        for j in range(step):
            closedList = []
            currentState = nodes[nodeIndex]
            lastState = currentState

            while (currentState.isGoal is not True):
                action = epsilonGreedy(epsilon, qTabel, currentState, lastState)
                visitTabel[currentState.name][action] += 1
                learningRate = 10 / (9 + visitTabel[currentState.name][action])

                nextState, reward = environment(currentState, action)
                
                penalty = 0
                if nextState not in closedList:
                    closedList.append(nextState)
                else:
                    penalty = qTabel[nextState.name][0]
                    for k in range(len(qTabel[nextState.name])):
                        penalty = min(qTabel[nextState.name][k], penalty)
                
                currentQValue = qTabel[currentState.name][action]

                qTabel[currentState.name][action] = currentQValue + learningRate *(reward+penalty + discountFactor * min(qTabel[nextState.name]) - currentQValue)
                lastState = currentState
                currentState = nextState  
    return qTabel, visitTabel

def environment(currentState, action):
    nextState = currentState.connections[action].destination
    reward = currentState.connections[action].weight
    return nextState, reward


def epsilonGreedy(epsilon, qTabel, currentState, lastState):
    randomNumber = random.random()
    if randomNumber < epsilon:
        return random.randint(0, len(qTabel[currentState.name])-1)
    else:
        minQValue = qTabel[currentState.name][0]
        bestAction = 0
        for i in range(len(qTabel[currentState.name])):
            nextQValue = qTabel[currentState.name][i]
            if  nextQValue < minQValue and lastState.name != currentState.connections[i].name:
                minQValue = nextQValue
                bestAction = i
        return bestAction

def initializeQTabel(nodes):
    qTabel = []
    for i in range(len(nodes)):
        qTabel.append([])
        for j in range(len(nodes[i].connections)):
            if nodes[i].isGoal:
                qTabel[i].append(0)
            else:
                qTabel[i].append(random.random()+0.01)
    print("RandomTable")
    printTabel(qTabel)
    return qTabel

def printBestQPath(qTabel, root, nodes):
    numberOfActions = 0
    currentState = root
    path = [currentState]
    closeList = []

    totalCost = 0

    while numberOfActions < len(nodes) and currentState.isGoal is not True:
        minQValue = qTabel[currentState.name][0]
        bestAction = 0
        for i in range(len(qTabel[currentState.name])):
            nextQValue = qTabel[currentState.name][i]
            if  nextQValue < minQValue and currentState.connections[i].destination not in closeList:
                minQValue = nextQValue
                bestAction = i
        closeList.append(currentState.connections[bestAction].destination)
        totalCost += currentState.connections[bestAction].weight
        
        currentState = currentState.connections[bestAction].destination
        path.append(currentState)
        numberOfActions += 1
    for node in path:
        print(node.name)
    print("Cost according to QTabel: ", totalCost) 

def initializeVisitTabel(nodes):
    VisitTabel = []
    for i in range(len(nodes)):
        VisitTabel.append([])
        for j in range(len(nodes[i].connections)):
            VisitTabel[i].append(0)
    return VisitTabel

def printTabel(QTabel):
    for i in range(len(QTabel)):
        print("state: ", i)
        for j in range(len(QTabel[i])):
            print(QTabel[i][j])

def loadTabel(filePath = "Algorithms\QTableLogs\qtable.txt", line = 1) :
    #open and read the file
    f = open(filePath, "r")
    Lines = f.readlines()
    newQtable = ast.literal_eval(Lines[line])
    return newQtable

def writeQTabelToLog(QTable, filePath = "Algorithms\QTableLogs\qtable.txt") :
    f = open(filePath, "a") 
    f.write("\n" + str(QTable))
    f.close()

def findDifferenceBetweenLists(List1, List2) :
    list3 = list(np.array(List1) - np.array(List2))
    difference = sum(sum(list3))
    return abs(difference)


#Creates the graph
nodes = createNodes(15)
createRoadConnections(nodes, 4)
nodes[0].isGoal = True
giveGoalNode(nodes, 0)
#printNodesAndConnections(nodes)


#Performs qlearning
print("Creating Q tabel")
qTabel = initializeQTabel(nodes)
visitTabel = initializeVisitTabel(nodes)
episode = 25
step = 25
count = 0
while count < 50: 
    qTabel, visitTable = qLearningOnGraph(qTabel, visitTabel, episode, step, nodes = nodes, epsilon = 0.1, discountFactor = 0.9)
    print("Best path q-learning")
    printBestQPath(qTabel = qTabel, root = nodes[14], nodes = nodes)
    count += 1
    writeQTabelToLog(qTabel)


path = "Algorithms\QTableLogs\qtable.txt"
newQtable49 = loadTabel(path, 3)
newQtable50 = loadTabel(path, 49)

print("Difference between the 3 and last Qtable", findDifferenceBetweenLists(newQtable49, newQtable50))

# print("____________________OBJECTTEST")
# printBestQPath(qTabel = newQtable, root = nodes[14], nodes = nodes)


print("xxxxxxxxxxxxxxxxxxxxxBEST PATH FOR Q KEARNIGN goiejogrjorejgioejriogjreg--------------!!!!!!!!!€!€!€!€!€!€!€!€!€")
printBestQPath(qTabel = qTabel, root = nodes[6], nodes = nodes)
print("--------------------------AStar:")
CheapestPathNode = AStar(nodes[6])
printBestPath(CheapestPathNode)
print("\n\n--------------------------Table:")
#printTabel(qTabel)

