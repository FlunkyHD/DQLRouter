from aStarGreedy import *
import math
import random
from Graph import *

NodesArray = []

a = NodeWithCoordinate("a", 3, 17)
b = NodeWithCoordinate("b", 2, 14)
c = NodeWithCoordinate("c", 5, 19)
d = NodeWithCoordinate("d", 6, 16)
e = NodeWithCoordinate("e", 5, 13)
f = NodeWithCoordinate("f", 3, 10)
g = NodeWithCoordinate("g", 5, 7)
h = NodeWithCoordinate("h", 10, 17)
i = NodeWithCoordinate("i", 9, 12)
j = NodeWithCoordinate("j", 9, 7)
k = NodeWithCoordinate("k", 6, 3)
l = NodeWithCoordinate("l", 1, 3)
m = NodeWithCoordinate("m", 11, 3)
n = NodeWithCoordinate("n", 12, 9)
o = NodeWithCoordinate("o", 15, 17)
p = NodeWithCoordinate("p", 15, 13)
q = NodeWithCoordinate("q", 15, 6, None, True)

NodesArray.append(a)
NodesArray.append(b)
NodesArray.append(c)
NodesArray.append(d)
NodesArray.append(e)
NodesArray.append(f)
NodesArray.append(g)
NodesArray.append(h)
NodesArray.append(i)
NodesArray.append(j)
NodesArray.append(k)
NodesArray.append(l)
NodesArray.append(m)
NodesArray.append(n)
NodesArray.append(o)
NodesArray.append(p)
NodesArray.append(q)

for node in NodesArray : 
    node.giveGoalNode(q)



def simpleNetwork() : 
    a = Node(False, 'a')
    b = Node(False, "b")
    c = Node(False, "c")
    d = Node(False, "d")
    e = Node(False, "e")
    f = Node(False, "f")
    g = Node(True, "g")

    AB = Road(3, b, "AB")
    AC = Road(4, c, "AC")
    BE = Road(2, e, "BE")
    BF = Road(3, f, "BF")
    FG = Road(2, g, "FG")
    EG = Road(5, g, "EG")
    CD = Road(6, d, "CD")
    CE = Road(3, e, "CE")
    DE = Road(1, e, "DE")

    a.addConnection(AB)
    a.addConnection(AC)
    b.addConnection(BE)
    b.addConnection(BF)
    c.addConnection(CE)
    c.addConnection(CD)
    d.addConnection(DE)
    e.addConnection(EG)
    f.addConnection(FG)

