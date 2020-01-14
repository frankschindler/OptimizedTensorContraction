import sys
import numpy as np
import fn_tensors as fnt
from copy import deepcopy
from joblib import Parallel, delayed

L = 10 #sys.argv[1]
datroot = 'dataRandom' + str(L)
ntrials = 40

graph, num_bonds = fnt.randomtn(L,0.8)
#graph, num_bonds = fnt.squaretn(L)
bdims = [10 for i in range(num_bonds)]

np.save(datroot+'/graph',graph)
np.save(datroot+'/bdims',bdims)

def trials():
    funlist = []
    numlist = []

    greedcost, numvals, bperm = fnt.greedy(deepcopy(graph), bdims, 2)

    return (greedcost, numvals, bperm)

data = Parallel(n_jobs=ntrials)(delayed(trials)() for i in range(ntrials))
funlist = [el[0] for el in data]
numlist = [el[1] for el in data]
bpermlist = [el[2] for el in data]

np.save(datroot+'/funlist_greedy',funlist)
np.save(datroot+'/numlist_greedy',numlist)
np.save(datroot+'/bpermlist_greedy',bpermlist)
