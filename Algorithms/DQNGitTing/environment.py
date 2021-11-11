import random
import enum
import numpy as np

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
        thing = np.array([0,0,0,0,0,0,0,0,1, 
                          1,0,0,0,0,0,0,0,0])
        return thing

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
               # print(count, "if")
                state.append(1)
            else : 
               # print(count, "else")
                state.append(0)
                #freeLoc.append(count)
            count += 1
        random.shuffle(state)
        
        for i in range (totalNodes) : 
            if state[i] == 0 :
                freeLoc.append(i)
            
        getLocIndex = random.randint(0,len(freeLoc) - 1)
        agentLoc = freeLoc[getLocIndex]        
        #print(agentLoc)
        for i in range (totalNodes) : 
            if i == (agentLoc) :
                state.append(1)
            else : 
                state.append(0)
                
        return state
        
        
        

    def sampleAction(self):
        return random.randint(0, self.actionSpace-1)    

    
    def step(self, state, action):
        
        locOffset = int(len(state) / 2) # 5
        locId = -1
        for i in range(locOffset):
            if state[i + locOffset ] == 1 :
                locId = i + locOffset
        
        if locId == -1:
            raise Exception("No location found in input.")

        currentNode = self.graph[locId - locOffset]
        reward = 1
        # if action >= len(self.graph):  #Hvis en action som ikke kan foretages vælges, retuneres samme stae. 
        #     return state, reward, False #self.isDone(state)

        nextNode = self.graph[action]
        
        #roadNew = currentNode.isConnected(nextNode)
        #if roadNew is False : 
            #print(currentNode.name, nextNode.name)
            #return  state, reward, False #self.isDone(state)            
        
        road = currentNode.connections[action]
        reward = road.weight
        nextNode = road.destination 
        
        #Here the location is updated
        state[locId] = 0
        state[nextNode.name + locOffset] = 1
        
        if state[nextNode.name] == 1:
            #reward -= 10
            state[nextNode.name] = 0
        #print(currentNode.name, nextNode.name)
        #print(state)
        return state, reward, self.isDone(state)
    
    
    
    
        # nextNode = self.graph[action]
        
        # #roadNew = currentNode.isConnected(nextNode)
        # #if roadNew is False : 
        #     #print(currentNode.name, nextNode.name)
        #     #return  state, reward, False #self.isDone(state)            
        
        # #road = currentNode.connections[action]
        # reward = roadNew.weight
        # nextNode = roadNew.destination 
        
        # #Here the location is updated
        # state[locId] = 0
        # state[nextNode.name + locOffset] = 1
        
        # if state[nextNode.name] == 1:
        #     reward -= 10
        #     state[nextNode.name] = 0
        # #print(currentNode.name, nextNode.name)
        
        # return state, reward, self.isDone(state)