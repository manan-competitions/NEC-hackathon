import numpy as np
import csv
import networkx as nx
from helpers.k_centers_problem import CreateGraph
from helpers.route import Route

def get_node_vals(fname):
    with open(fname) as f:
        data = list(csv.reader(f))
    data = [list(map(int, d)) for d in data]
    return np.array(data)

def get_nbrs(node, first=None, last=None):
    nbrs = sorted(list(G.neighbors(0)), key=lambda n: G[0][n]['length'], reverse=True)

    if first:
        return nbrs[:first]
    elif last:
        return nbrs[-last:]
    else:
        return nbrs

def random_walk(s, d, l):
    walk = [s]
    while len(walk) != l-1:
        node = walk[-1]
        first = np.array(get_nbrs(node, first=10))
        last = np.array(get_nbrs(node, first=5))

        if d in first:
            first = np.delete(first, np.argwhere(first==d))
        elif d in last:
            last = np.delete(last, np.argwhere(last==d))

        probs = []
        ind_f = []
        ind_l = []
        for i in range(first.shape[0]):
            try:
                probs.append(G[first[i]][d]['length'])
                ind_f.append(i)
            except:
                pass
        for i in range(last.shape[0]):
            try:
                probs.append(G[last[i]][d]['length']*4)
                ind_l.append(i)
            except:
                pass

        probs = np.array(probs)
        probs = 1/(1+probs)
        probs = probs/np.sum(probs)
        walk.append(np.random.choice(np.concatenate([first[ind_f], last[ind_l]]), p=probs))
    return walk+[d]

## -- MAIN -- ##

G = CreateGraph(75, fname='final_full_graph.csv')
on_off = get_node_vals('node_vals.csv')

pop_size = 100
walk_length = [5,12]

pop = []
for i in range(pop_size):
    source, destination = np.random.choice(list(G.nodes()), size=2)
    length = np.random.randint(walk_length[0], walk_length[1]+1)
    pop.append(Route(random_walk(source, destination, length)))
"""
for i in range(5):
    print(pop[i])
"""
