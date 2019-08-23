import numpy as np
import csv
import networkx as nx
from numpy.random import choice
from copy import deepcopy

# Create a weighted undirected graph G from a file or a variable
def CreateGraph(n,file=True, fname=None, adj_matrix=None):
	"""
	Accepts either a file name or a list of args
	Arguements:
		file (Bool): whether to read from a file or from a variable
		num_vertices(int): number of vertices
		if True:
			fname(str): Path to file
		else:
			adj_matrix(list/array): list of lists or a np array of shape(n,n)
	Returns:
		G: A weighted undirected graph
	"""

	if file:
		with open(fname) as f:
			wtMatrix = []
			reader = csv.reader(f)
			for row in reader:
				list1 = list(map(float, row))
				wtMatrix.append(list1)
		wtMatrix = np.array(wtMatrix)
	else:
			wtMatrix = np.array(adj_matrix)

	if wtMatrix.shape != (n,n):
		raise Exception(f'Incorrect Shape: Expected ({n},{n}) but got {wtMatrix.shape} instead')

	#Adds egdes along with their weights to the graph
	G = nx.Graph()
	for i in range(n) :
		for j in range(i,n):
			G.add_edge(i, j, length = wtMatrix[i][j])

    # Add individual node probabilites
	on_off = get_node_vals("./dataset/node_vals.csv")
	on_off_dict = {
        x: {"prob_in": on_off[x][0], "prob_out": on_off[x][1]} for x in range(len(on_off))
    }
	add_weights(G, on_off_dict)
	return G


def get_node_vals(fname):
    with open(fname) as f:
        data = list(csv.reader(f))
    data = [list(map(float, d)) for d in data]
    return np.array(data)

def get_nbrs(G, node, first=None, last=None):
    nbrs = sorted(list(G.neighbors(0)), key=lambda n: G[0][n]["length"], reverse=True)

    if first:
        return nbrs[:first]
    elif last:
        return nbrs[-last:]
    else:
        return nbrs

def random_walk(G, s, d, l):
    walk = [s]
    while len(walk) != l - 1:
        node = walk[-1]
        first = np.array(get_nbrs(G, node, first=len(walk)+1))
        last = np.array(get_nbrs(G, node, first=10))

        if d in first:
            first = np.delete(first, np.argwhere(first == d))
        elif d in last:
            last = np.delete(last, np.argwhere(last == d))

        probs = []
        ind_f = []
        ind_l = []
        for i in range(first.shape[0]):
            try:
                probs.append(G[first[i]][d]["length"])
                ind_f.append(i)
            except:
                pass
        for i in range(last.shape[0]):
            try:
                probs.append(G[last[i]][d]["length"] * 4)
                ind_l.append(i)
            except:
                pass

        probs = np.array(probs)
        probs = 1 / (1 + probs)
        probs = probs / np.sum(probs)

        node = choice(np.concatenate([first[ind_f], last[ind_l]]), p=probs)
        if node != walk[-1]:
            walk.append(node)
    return walk + [d]


def add_weights(grph, weights):
    for k, w in weights.items():
        grph.add_node(k, **w)

def fitness(routes, sim_dg, c1, c2, c3, opt_bus, components=False):
    # Prevent this method from having side-effects
    # Make a copy of the param
    prev_dg = sim_dg
    sim_dg = deepcopy(sim_dg)

    deboard_dict = dict()
    for route in routes.routes:
        current_capacity = route.num * route.cap
        for i in range(len(route.v) - 1):
            current_capacity += deboard_dict.get(i, 0)
            deboard_dict[i] = 0
            for k in set(sim_dg[route.v[i]]).intersection(set(route.v[i + 1 :])):
                people_boarding = min(sim_dg[route.v[i]][k]["weight"], current_capacity)
                sim_dg[route.v[i]][k]["weight"] -= people_boarding
                deboard_dict[k] = deboard_dict.get(k, 0) + people_boarding
                current_capacity -= people_boarding
    num_ppl = (prev_dg.size(weight="weight") - sim_dg.size(weight="weight")) / (1+routes.cap)
    if components:
        return num_ppl, routes.num_buses, routes.cum_len
    return c1*num_ppl + c2/(1+np.abs(routes.num_buses - opt_bus)) + c3/(1+routes.cum_len)

def simulate_people(G, num_of_people):
    arr_out = [x for y, x in nx.get_node_attributes(G, "prob_out").items()]
    arr_out = [x / sum(arr_out) for x in arr_out]
    arr_in = [x for y, x in nx.get_node_attributes(G, "prob_in").items()]
    arr_in = [x / sum(arr_in) for x in arr_in]
    counts_out = choice(G, num_of_people, p=arr_out)
    counts_in = choice(G, num_of_people, p=arr_in)
    edges = [k for k in zip(counts_in, counts_out)]
    counts = dict()
    for i in edges:
        if i[0] != i[1]:
            counts[i] = counts.get(i, 0) + 1
    DG = nx.DiGraph()
    for i, j in counts.items():
        x, y = i
        DG.add_edge(x, y, weight=j)
    return DG

def GA(iter, pop, pop_size, ppl, c1, c2, c3, opt_bus, elite, mutation_prob, crossover_perc):
    print('\nTraining the Genetic Algorithm ...')
    new_pop = deepcopy(pop)
    for i in range(iter):
        print(f'Iteration {i+1} / {iter}')
        curr_pop = deepcopy(new_pop)
        new_pop = []

        # get fitness of everyone
    #    print('-- Fitness')
        fit = [(p,fitness(p, ppl, c1, c2, c3, opt_bus)) for p in curr_pop]

        #fit = map(fitness, pop, ppl, c1, c2, c3, opt_bus)
        fit.sort(reverse=True,key=lambda x: x[1])

        # Transfer elite directly to the next generation
        elite_num = int(elite*pop_size)
        elite_pop = []
        for j in range(elite_num):
            elite_pop.append(deepcopy(fit[j][0]))
    #    print('-- Selection')

        # Select the rest according to the fitness function (Selection)
        new_pop = choice(curr_pop, size=pop_size-elite_num, p=[f[1] for f in fit]/np.sum([f[1] for f in fit]))

    #    print('-- Crossover')
        # Crossover the rest
        cross_size = int(crossover_perc*len(new_pop))
        cross_size += cross_size%2
        cross_routes = choice(new_pop, size=cross_size)
        for i in range(0,cross_size,2):
            curr_pop[i].crossover(curr_pop[i+1])

    #    print('-- Mutation')
        # Mutatie every elemnt (may not take place actually)
        for p in new_pop:
            p.mutate(mutation_prob)

        new_pop = np.concatenate([new_pop, elite_pop])
        best = fit[0][0]
        print(f'-- Average: {np.mean([f[1] for f in fit])} Best: {fit[0][1]} Worst: {fit[-1][1]}')
    return best, new_pop
