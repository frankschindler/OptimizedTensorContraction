import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

L = 10 #sys.argv[1]
datroot = 'dataRandom' + str(L)

funlist_greedy = np.load(datroot+'/funlist_greedy.npy',allow_pickle=True)
num_greedy = np.load(datroot+'/numlist_greedy.npy').astype(int)[0]
historylist_annealing = np.load(datroot+'/historylist_annealing.npy',allow_pickle=True)
historylist_genetics = np.load(datroot+'/historylist_genetics.npy',allow_pickle=True)

def argmedian(data):
    #return np.argsort(data)[(len(data)//2)]
    return np.argmin(data)

medgred = argmedian(funlist_greedy)
medann = argmedian([np.array(el)[-1,1] for el in historylist_annealing])
medgen = argmedian([np.array(el)[-1,1] for el in historylist_genetics])

hlistann = np.array(historylist_annealing[medann])
hlistgen = np.array(historylist_genetics[medgen])

# plot the optimization history
plt.figure(figsize=(3,3))

plt.gca().axvline(num_greedy,ls='--',color='orange')
plt.gca().axhline(np.exp(funlist_greedy[medgred]),ls='--',color='orange',label='Greedy')
plt.plot(hlistann[:,0],np.exp(hlistann[:,1]),'o-',label='Annealing',lw=1,color='blue')
plt.plot(hlistgen[:,0],np.exp(hlistgen[:,1]),'o-',label='Genetic',lw=1,color='green')

plt.ylabel('minimal cost achieved')
plt.xlabel('# of cost function evaluations')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(datroot+'/plothist.pdf')

plt.show()
