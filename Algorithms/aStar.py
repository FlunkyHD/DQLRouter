import math
import random
random.seed(3)
from Graph import Road, Node, NodeWithCoordinate

from aStarGreedy import AStar, Greedy


def createNodes(amountOfNodes) :
    random.seed(5)
    areaLength = math.sqrt(amountOfNodes)
    Nodes = []
    count = 0
    while count < amountOfNodes :
        Nodes.append(NodeWithCoordinate(str(count), random.random*areaLength, random.random*areaLength))
    return Nodes

def giveGoalNode(NodesArray, NodeNumber) :
    GoalNode = Node
    for node in NodesArray : 
        if node.name == str(GoalNode) : 
            GoalNode = node
            node.isGoal = True
    
    for node in NodesArray : 
        node.giveGoalNode(GoalNode)


def createRoadConnections(NodesArray, connectionsPerNode):
    Roads = []
    for node in NodesArray:
        #We find the closest nodes, which the the nodes we make a road to
        closestNodes = node.findClosestNodes(NodesArray.copy(), connectionsPerNode)
        count = 0

        #We create a road for each of the closest nodes. 
        random.seed(3)
        while count < connectionsPerNode :
            weight = node.findDistanceToPoint(closestNodes[count]) + random.random()*0  # Fejl indsæt rigtigt værdi
            newRoad = Road(weight, closestNodes[count], node.name + closestNodes[count].name )
            Roads.append(newRoad)
            node.addConnection(newRoad)
            count += 1

def printBestPath(BestPath):
    parentList = [BestPath]
    CurrentNode = BestPath
    while CurrentNode.parent != None :
        parentList.append(CurrentNode.parent)
        CurrentNode = CurrentNode.parent
    parentList.reverse()
    for elements in parentList :
        print(elements.name)



