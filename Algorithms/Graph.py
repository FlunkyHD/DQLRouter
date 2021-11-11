# -*- coding: utf-8 -*-
import math
import random
random.seed(5)
class Node:
    def __init__(self, goal, name):
        self.connections = []
        self.currentCost = 0
        self.expectedCost = 0
        self.isGoal = goal
        self.parent = None
        self.visited = False
        self.name = name
        self.heuristicGuess = 0

    def addConnection(self, newConnection):
        self.connections.append(newConnection)

class NodeWithCoordinate:
    def findDistanceToPoint(self, destination):
        return math.sqrt( (self.x - destination.x)**2 + (self.y - destination.y)**2 )

    def __init__(self, name, x, y, goalNode = None, isGoal = False):
        self.connections = []
        self.currentCost = 0
        self.expectedCost = 0
        self.isGoal = isGoal
        self.parent = None
        self.visited = False
        self.name = name
        self.x = x
        self.y = y
        self.goalNode = goalNode
        if goalNode != None:
            self.heuristicGuess = self.findDistanceToPoint(goalNode)
        else :
            self.heuristicGuess = None

    def addConnection(self, newConnection):
        self.connections.append(newConnection)

    def giveGoalNode(self, goalNode) :
        self.goalNode = goalNode
        self.heuristicGuess = self.findDistanceToPoint(goalNode)
    def __str__(self):
        connectionsStr = " "
        for connections in self.connections : 
            connectionsStr = connectionsStr + str(connections.name)
        String1 = "name: " + str(self.name) + "\n connections" + connectionsStr + "\n x: "
        String2 = str(self.x) + "\n y: " + str(self.y) 
        String3 =  "\n Heuristic: " + str(self.heuristicGuess) 
        String4 = "\nWhat i think is goal: "
        if self.goalNode != None :
            String4 = String4 + str(self.goalNode.name)
        String5 = "Am i goal? " + str(self.isGoal)
        return String1 + String2 + String3 + String4 + String5

    def makeGoalNode(self) : 
        self.isGoal == True
    def findClosestNodes(self, NodeList, amount):
        #First we remove ourself as the closest node
        for node in NodeList:
            if node.name == self.name :
                NodeList.remove(node)

        closestNodes = []
        #We add the first nodes, and expect them to be the closest
        count = 0
        while count < amount:
            closestNodes.append(NodeList.pop())
            count += 1
        closestNodes.sort(key = lambda x : x.findDistanceToPoint(self), reverse = True)

        for node in NodeList:
            if self.findDistanceToPoint(node) < self.findDistanceToPoint(closestNodes[0]):
                closestNodes.remove(closestNodes[0])
                closestNodes.append(node)
            closestNodes.sort(key = lambda x : x.findDistanceToPoint(self), reverse = True)

        return closestNodes



class Road:
    def __init__(self, weight, destination, name):
        self.weight = weight
        self.destination = destination
        self.name = name
    def getWeigth(self):
        return self.weight


def createNodes(amountOfNodes) :
    random.seed(6)
    areaLength = math.sqrt(amountOfNodes)
    Nodes = []
    count = 0
    while count < amountOfNodes :
        Nodes.append(NodeWithCoordinate(count, random.random()*areaLength, random.random()*areaLength))
        count += 1
    return Nodes

def giveGoalNode(NodesArray, NodeNumber) :
    # for node in NodesArray :
    #     if node != NodesArray[NodeNumber] :
    #         node.giveGoalNode(NodesArray[NodeNumber])
    #     else :
    #         node.heuristicGuess = 0
    for node in NodesArray :
        node.giveGoalNode(NodesArray[NodeNumber])




def createRoadConnections(NodesArray, connectionsPerNode):
    Roads = []
    for node in NodesArray:
        #We find the closest nodes, which the the nodes we make a road to
        closestNodes = node.findClosestNodes(NodesArray.copy(), connectionsPerNode)
        count = 0

        #We create a road for each of the closest nodes. 
        random.seed(3)
        while count < connectionsPerNode :
            weight = node.findDistanceToPoint(closestNodes[count]) #+ random.random()*0  
            #weight = 1 
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

def printNodesAndConnections(nodes):
    for node in nodes : 
        print("Node name: ", node.name)
        for connections in node.connections : 
            print("destination: ", connections.destination.name)



