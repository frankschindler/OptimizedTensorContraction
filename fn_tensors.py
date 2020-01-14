import numpy as np
import itertools

from copy import deepcopy
from scipy import special as spsp

def flatten(list):
    return [elel for el in list for elel in el]

'''
We store a tensor network as a collection of nodes and bonds.
A bond is a collection of one or two nodes and a unique ID.
A node is a collection of one or more bonds and a unique ID.
'''
def reformat(g):
    nodes = {i:v for i,v in enumerate(g)} # Node ID == position in list g
    bonds = {b:list(i for i,v in enumerate(g) if b in v) for b in set(flatten(g))} # Node ID == position in list g
    max_ID = len(g) - 1
    g = (nodes, bonds, max_ID)
    return g

# square tensor network of linear extent L
def squaretn(L):
    g = [['a1','a'+str(L)],['a'+str((L-1)*(L-1)+(L-2)*L+1),'a'+str((L-1)*(L-1)+(L-2)*L+1+L)],['a'+str((L-1)*(L-1)+(L-2)*L+L),'a'+str((L-1)*(L-1)+(L-2)*L+L+L-1)],['a'+str(L-1),'a'+str(2*L-1)]]
    for i in np.arange(1,L-1):
        g.append(['a'+str(1+i*(L-1)+(i-1)*L),'a'+str(1+(i+1)*(L-1)+i*L),'a'+str(1+i*(L-1)+i*L)])
        for j in (np.arange(1,L-1)):
            g.append(['a'+str(j+1+i*(L-1)+(i-1)*L),'a'+str(j+1+(i+1)*(L-1)+i*L),'a'+str(j+1+i*(L-1)+i*L),'a'+str(j+i*(L-1)+i*L)])
        g.append(['a'+str(i+1),'a'+str(i),'a'+str(i+L)])
        g.append(['a'+str(1+i*(L-1)+(i-1)*L+L-1),'a'+str(1+(i+1)*(L-1)+i*L+L-1),'a'+str(1+i*(L-1)+i*L+L-2)])
        g.append(['a'+str(i+1+(L-1)*(L-1)+L*(L-1)),'a'+str(i+(L-1)*(L-1)+L*(L-1)),'a'+str(i+1-L+(L-1)*(L-1)+L*(L-1))])
    g = reformat(g)
    return g, (2*L*(L-1))

# random tensor network of L vertices with connectivity con
def randomtn(L, con):
    g = [[] for i in range(L)]
    bnum = 0
    for i in range(L):
        for j in range(i):
            if np.random.choice([0,1],p=[con,1-con]) == 0:
                bnum += 1
                g[i].append('a'+str(bnum))
                g[j].append('a'+str(bnum))
    g = reformat(g)
    return g, bnum

# evaluate the cost of a single contraction step
def step_and_weight(graph, bond, bonddims):
    nodes, bonds, max_ID = graph
    bondlab = 'a'+str(bond)
    mergedNodes = bonds[bondlab]
    # Construct merged node and delete old nodes and bond
    newNode = []
    for m in mergedNodes:
        newNode = newNode + nodes[m]
        del nodes[m]
    del bonds[bondlab]
    # Remove dubplicates
    newNode = list(set(newNode))
    newNode.remove(bondlab)
    newNode = list(newNode)
    # Store new node
    nodes[max_ID+1] = newNode
    max_ID += 1
    # Update bonds to point to the new node
    for b in newNode:
        bonds[b].append(max_ID)
        for m in mergedNodes:
            if m in bonds[b]:
                bonds[b].remove(m)
    weight = np.log(bonddims[bond-1]) + np.sum([np.log(bonddims[int(newbond[1:])-1]) for newbond in newNode])
    return (nodes, bonds, max_ID), weight

# contract a graph with given order and return the log of the total cost
def logcontract(graph, bonddims, bondsequence, resolved = False):
    sumlist = []
    for bond in bondsequence:
        graph, weight = step_and_weight(graph, bond, bonddims)
        sumlist.append(weight)
    if resolved:
        return sumlist
    return spsp.logsumexp(sumlist)

# the greedy algorithm
def greedy(graph, bonddims, level):
    bondset = list(np.arange(1,len(bonddims)+1))
    sum = 0
    numvals = 0
    idealperm = []
    for i in range(len(bonddims)):
        sequenceset = list(itertools.permutations(bondset, int(np.min([level,len(bondset)]))))
        wset = [logcontract(deepcopy(graph), bonddims, sequence) for sequence in sequenceset]
        numvals += len(sequenceset)
        if (len(wset)>len(set(wset))):
            wsetloc = np.array(wset)
            index_set = np.where(wsetloc == wsetloc.min())[0]
            idealbond = sequenceset[np.random.choice(index_set)][0]
        else:
            idealbond = sequenceset[np.argmin(wset)][0]
        idealperm.append(idealbond)
        graph, weight = step_and_weight(graph, idealbond, bonddims)
        sum = spsp.logsumexp([sum,weight])
        bondset.remove(idealbond)
    return sum, numvals, idealperm
