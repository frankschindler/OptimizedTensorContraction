import sys
import numpy as np
import fn_tensors as fnt
import fn_genetic as fng
from copy import deepcopy
from joblib import Parallel, delayed

L = 10 #sys.argv[1]
datroot = 'dataRandom' + str(L)
ntrials = 40

graph = np.load(datroot+'/graph.npy',allow_pickle = True)
bdims = np.load(datroot+'/bdims.npy',allow_pickle = True)
num_bonds = len(bdims)

num_greedy = np.load(datroot+'/numlist_greedy.npy').astype(int)[0]

def trials():
    num_evaluate = 0
    population = np.array([np.random.permutation(num_bonds)+1 for i in range(20)])
    history = []
    best_genotype = []
    while (num_evaluate < 1*num_greedy):
        population, minrew, best_genotype = fng.step(graph, bdims, population, mut_rate=0.6)
        num_evaluate += len(population)
        history.append([num_evaluate, minrew])
    return history, best_genotype

data = Parallel(n_jobs=ntrials)(delayed(trials)() for i in range(ntrials))
historylist = [el[0] for el in data]
bpermlist = [el[1] for el in data]

np.save(datroot+'/historylist_genetics',historylist)
np.save(datroot+'/bpermlist_genetics',bpermlist)
