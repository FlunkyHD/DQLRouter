from aStarGreedy import *
import math
import random
from Graph import *
from qLearn import *

def printNodes(Nodes):
    for node in Nodes:
        print(str(node.name) + " " + str(node.heuristicGuess))

def printAllNodesAll(Nodes):
    for node in Nodes: 
        print(node.__str__())


Nodes = createNodes(1000)
Nodes[0].isGoal = True
#print(str(Nodes[0].isGoal))
giveGoalNode(Nodes, 0)
createRoadConnections(Nodes, 4)
#printAllNodesAll(Nodes)
CheapestPathNode = AStar(Nodes[4])
if CheapestPathNode :
    print("gr")

    printBestPath(CheapestPathNode)
else : 
    print("Den er falsk")





