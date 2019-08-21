import numpy as np
import csv
import networkx as nx
from helpers.k_centers_problem import CreateGraph
from helpers.route import Route
from numpy.random import choice
from pprint import pprint


def get_node_vals(fname):
    with open(fname) as f:
        data = list(csv.reader(f))
    data = [list(map(float, d)) for d in data]
    return np.array(data)


def get_nbrs(node, first=None, last=None):
    nbrs = sorted(list(G.neighbors(0)), key=lambda n: G[0][n]["length"], reverse=True)

    if first:
        return nbrs[:first]
    elif last:
        return nbrs[-last:]
    else:
        return nbrs


def random_walk(s, d, l):
    walk = [s]
    while len(walk) != l - 1:
        node = walk[-1]
        first = np.array(get_nbrs(node, first=10))
        last = np.array(get_nbrs(node, first=5))

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
        walk.append(
            np.random.choice(np.concatenate([first[ind_f], last[ind_l]]), p=probs)
        )
    return walk + [d]


def add_weights(grph, weights):
    for k, w in weights.items():
        grph.add_node(k, **w)


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
    return counts
    # DG = nx.DiGraph()
    # for i, j in counts.items():
    #     x, y = i
    #     DG.add_edge(x, y, weight=j)
    # return DG


## -- MAIN -- ##

G = CreateGraph(75, fname="./dataset/final_full_graph.csv")

on_off = get_node_vals("./dataset/node_vals.csv")
on_off_dict = {
    x: {"prob_in": on_off[x][0], "prob_out": on_off[x][1]} for x in range(len(on_off))
}
add_weights(G, on_off_dict)
choice

pop_size = 100
walk_length = [5, 12]

pop = []
for i in range(pop_size):
    source, destination = np.random.choice(list(G.nodes()), size=2)
    length = np.random.randint(walk_length[0], walk_length[1] + 1)
    pop.append(Route(random_walk(source, destination, length)))
"""
for i in range(5):
    print(pop[i])
"""
