def Greedy(root):
    openList = []
    closeList = []
    openList.append(root)
    while openList.count != 0 :
        openList.sort(key = lambda x : x.currentCost)

        cheapestNode = openList.pop(0)

        closeList.append(cheapestNode)
        if cheapestNode.isGoal:
            root.parent = None
            return cheapestNode
        #Add the nodes reachable from bestNode to the list, and set their currentcost
        for roads in cheapestNode.connections :
            newDest = roads.destination
            #In here is the cheapest way to a given node. 
            if newDest.visited == False or newDest.currentCost > roads.weight + cheapestNode.currentCost :
                newDest.currentCost = roads.weight + cheapestNode.currentCost
                newDest.visited = True
                newDest.parent = cheapestNode
                openList.append(roads.destination)
    return False


#AStar


def AStar(root):
    openList = []
    closeList = []
    openList.append(root)
    while openList :
        openList.sort(key = lambda x : x.currentCost + x.heuristicGuess)

        cheapestNode = openList.pop(0)

        closeList.insert(0,cheapestNode)
        #print("\nHej Hej \n" + cheapestNode.__str__())
        if cheapestNode.isGoal:
            root.parent = None
            return cheapestNode
        #Add the nodes reachable from bestNode to the list, and set their currentcost
        for roads in cheapestNode.connections :
            newDest = roads.destination
            #In here is the cheapest way to a given node. 
            if newDest.visited == False or newDest.currentCost > roads.weight + cheapestNode.currentCost :
                newDest.currentCost = roads.weight + cheapestNode.currentCost
                newDest.visited = True
                newDest.parent = cheapestNode
                openList.append(roads.destination)


    return False


