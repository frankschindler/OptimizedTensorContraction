import sys
import numpy as np
import fn_tensors as fnt
import fn_annealing as fna
from scipy import optimize as opt
from copy import deepcopy
from joblib import Parallel, delayed

L = 10 #sys.argv[1]
datroot = 'dataRandom' + str(L)
ntrials = 40

graph = np.load(datroot+'/graph.npy',allow_pickle = True)
bdims = np.load(datroot+'/bdims.npy',allow_pickle = True)
num_bonds = len(bdims)

num_greedy = np.load(datroot+'/numlist_greedy.npy').astype(int)[0]

def to_be_annealed(perm):
    perm = (1+np.argsort(perm))
    return fnt.logcontract(deepcopy(graph), bdims, perm)

def trials():
    res = fna.dual_annealing(to_be_annealed, [[0,1] for i in range(num_bonds)], maxiter=1000000000000, maxfun = 1*num_greedy)
    return res.history, 1+np.argsort(res.x)

data = Parallel(n_jobs=ntrials)(delayed(trials)() for i in range(ntrials))
historylist = [el[0] for el in data]
bpermlist = [el[1] for el in data]

np.save(datroot+'/historylist_annealing',historylist)
np.save(datroot+'/bpermlist_annealing',bpermlist)
