import numpy as np
import fn_tensors as fnt
from copy import deepcopy

def test(graph, bdims, perm):
    return fnt.logcontract(deepcopy(graph), bdims, perm)

def fitness(graph, bdims, population):
    reverse_fit = [test(graph, bdims, population[i]) for i in range(len(population))]
    maximum = np.max(reverse_fit)
    minimum = np.min(reverse_fit)
    spread = maximum-minimum
    if spread == 0:
        spread += 1
    return np.exp(1-(reverse_fit-minimum)/spread)-0.99, minimum

def mutate(num_bonds,population):
    for i in range(len(population)):
        swap = np.random.randint(0,num_bonds,size=2)
        population[i][swap[0]], population[i][swap[1]] = population[i][swap[1]], population[i][swap[0]]
    return population

def step(graph, bdims, population, mut_rate=.2):
    fit, minrew = fitness(graph, bdims, population)
    probabilities = fit/fit.sum()
    best_genotype = population[np.argmax(probabilities)].reshape(1,-1)
    which_genes = np.random.choice(len(population), size=len(population)-1, replace=True, p=probabilities).astype(int)
    population = np.array([population[i] for i in which_genes])
    num_mutate = int(mut_rate*len(population))
    population = np.concatenate((mutate(len(bdims),population[0:num_mutate]), population[num_mutate:], best_genotype), axis=0)
    return population, minrew, best_genotype[0]
