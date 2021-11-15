import random
import numpy as np
from torch._C import Graph
import math
random.seed()

class environment():
    
    def  __init__(self, graph, actionSpace):
        self.graph = graph
        self.actionSpace = actionSpace
    

    def isDone(self,state):
        for i in range(int(len(state) / 2)):
            if state[i] == 1:
                return False
        return True

    def reset(self): #Dette retunere en specifik state. Skal ændres senere.
       # thing = np.array([0,0,0,0,0,0,0,0,1,
        #                  1,0,0,0,0,0,0,0,0])
        graph = []
        for i in range (2*2*2) :
            if(i == 3 or i == 4) :
                graph.append(1)
            else :
                graph.append(0)
        return graph
    
    
    # [1,1,0,1,0 ,
    #                       0,0,1,0,0])

    def randomReset(self, maxGoals, alwaysCapGoals = False):
        
        totalNodes = len(self.graph)
        agentLoc = random.randint(0,totalNodes - 1)
        
        if maxGoals >= totalNodes :
            return

        if alwaysCapGoals :
            numberOfGoals = maxGoals
        else : 
            numberOfGoals = random.randint(1, maxGoals)
        count = 0
        state = []
        freeLoc = []
        for i in range (totalNodes) : 
            if(count < numberOfGoals) :
                state.append(1)
            else : 
                state.append(0)
            count += 1
        random.shuffle(state)
        
        for i in range (totalNodes) : 
            if state[i] == 0 :
                freeLoc.append(i)
            
        getLocIndex = random.randint(0,len(freeLoc) - 1)
        agentLoc = freeLoc[getLocIndex]        
        for i in range (totalNodes) : 
            if i == (agentLoc) :
                state.append(1)
            else : 
                state.append(0)
        print("state: \n", state)
        return state
        
    
    def printGridAsGrid(self, state, length) :
        #print("Goals and Locations. goal = 2, loc = 1")
        offset = length*length
        for i in range (length) : 
            for j in range (length) :
                if(state[j + i*length] == 1) : #Goals
                    print("1 ", end = '')
                elif(state[j+i*length+offset] == 1) :
                    print("2 ", end = '')
                else : 
                    print("- ", end = '')
            print()         
        

    def sampleAction(self):
        return random.randint(0, self.actionSpace-1)    

    
    def step(self, state, action):
        locOffset = int(len(state) / 2) # 10
        locId = -1
        for i in range(locOffset):
            if state[i + locOffset ] == 1 :
                locId = i + locOffset
        #print("   ", action)
        #self.printGridAsGrid(state, int(math.sqrt(len(state) / 2)))
        if locId == -1:
            raise Exception("No location found in input.")

        currentNode = self.graph[locId - locOffset]
        reward = 1
        nextNode = self.graph[action]
        
        road = currentNode.connections[action]
        reward = road.weight
        nextNode = road.destination 
        
        #Here the location is updated
        state[locId] = 0
        state[nextNode.name + locOffset] = 1
        if state[nextNode.name] == 1:
            reward -= 3
            state[nextNode.name] = 0

        return state, reward, self.isDone(state)
    
    
    
