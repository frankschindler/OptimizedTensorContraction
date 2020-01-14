import numpy as np
import fn_tensors as fnt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
from copy import deepcopy

L = 10 #sys.argv[1]
datroot = 'dataRandom' + str(L)

def reexp(x):
    return x/(np.log(10))

def argmedian(data):
    #return np.argsort(data)[(len(data)//2)]
    return np.argmin(data)

graph = np.load(datroot+'/graph.npy',allow_pickle = True)
bdims = np.load(datroot+'/bdims.npy',allow_pickle = True)
num_bonds = len(bdims)

funlist_greedy = np.load(datroot+'/funlist_greedy.npy',allow_pickle=True)
bpermlist_greedy = np.load(datroot+'/bpermlist_greedy.npy',allow_pickle=True)
historylist_annealing = np.load(datroot+'/historylist_annealing.npy',allow_pickle=True)
bpermlist_annealing = np.load(datroot+'/bpermlist_annealing.npy',allow_pickle=True)
historylist_genetics = np.load(datroot+'/historylist_genetics.npy',allow_pickle=True)
bpermlist_genetics = np.load(datroot+'/bpermlist_genetics.npy',allow_pickle=True)

medgred = argmedian(funlist_greedy)
bpermgred = bpermlist_greedy[medgred]
medann = argmedian([np.array(el)[-1,1] for el in historylist_annealing])
bpermann = bpermlist_annealing[medann]
medgen = argmedian([np.array(el)[-1,1] for el in historylist_genetics])
bpermgen = bpermlist_genetics[medgen]

# consistency check and output contraction sequences
print(reexp(fnt.logcontract(deepcopy(graph),bdims,bpermgred)))
print(reexp(np.min(funlist_greedy)))
print(reexp(fnt.logcontract(deepcopy(graph),bdims,bpermann)))
print(reexp(historylist_annealing[medann][-1][1]))
print(reexp(fnt.logcontract(deepcopy(graph),bdims,bpermgen)))
print(reexp(historylist_genetics[medgen][-1][1]))
print('greedy ', bpermgred)
print('annealing ', bpermann)
print('genetics ', bpermgen)

# baseline contraction sequence for L=6 square TN with chi=10
#bespokeperm = [6,7,8,9,10,11,17,18,19,20,21,22,28,29,30,31,32,33,39,40,41,42,43,44,50,51,52,53,54,55,1,12,23,34,45,56,2,13,24,35,46,57,3,14,25,36,47,58,4,15,26,37,48,59,5,16,27,38,49,60]

#pdatbespoke = [np.exp(el) for el in fnt.logcontract(deepcopy(graph),bdims,bespokeperm,resolved=True)]
pdatgred = [np.exp(el) for el in fnt.logcontract(deepcopy(graph),bdims,bpermgred,resolved=True)]
pdatann = [np.exp(el) for el in fnt.logcontract(deepcopy(graph),bdims,bpermann,resolved=True)]
pdatgen = [np.exp(el) for el in fnt.logcontract(deepcopy(graph),bdims,bpermgen,resolved=True)]

# plot the cumulative cost
plt.figure(figsize=(3,3))

#plt.plot(np.cumsum(pdatbespoke),'o-',label='Baseline',lw=1,color='black')
plt.plot(np.cumsum(pdatgred),'o-',label='Greedy',lw=1,color='orange')
plt.plot(np.cumsum(pdatann),'o-',label='Annealing',lw=1,color='blue')
#plt.plot(np.cumsum(pdatgen),'o-',label='Genetic',lw=1,color='green')

plt.ylabel('integrated contraction cost')
plt.yscale('log')
plt.xlabel('contraction step')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(datroot+'/integratedcontractioncost.pdf')

# plot the cost per contraction step
plt.figure(figsize=(3,3))

#plt.plot(pdatbespoke,'o-',label='Baseline',lw=1,color='black')
plt.plot(pdatgred,'o-',label='Greedy',lw=1,color='orange')
plt.plot(pdatann,'o-',label='Annealing',lw=1,color='blue')
#plt.plot(pdatgen,'o-',label='Genetic',lw=1,color='green')

plt.ylabel('contraction cost per step')
plt.yscale('log')
plt.xlabel('contraction step')
plt.legend()
plt.tight_layout()
plt.savefig(datroot+'/contractioncost.pdf')

plt.show()
