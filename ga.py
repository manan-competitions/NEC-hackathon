import numpy as np
import csv
import networkx as nx
from helpers.k_centers_problem import CreateGraph
from helpers.route import Route, Routes
from numpy.random import choice
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm


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
        walk.append(choice(np.concatenate([first[ind_f], last[ind_l]]), p=probs))
    return walk + [d]


def add_weights(grph, weights):
    for k, w in weights.items():
        grph.add_node(k, **w)


def fitness(routes, sim_dg):
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
    return (prev_dg.size(weight="weight") - sim_dg.size(weight="weight")) / routes.cap


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


## -- MAIN -- ##

G = CreateGraph(75, fname="./dataset/final_full_graph.csv")

on_off = get_node_vals("./dataset/node_vals.csv")

on_off_dict = {
    x: {"prob_in": on_off[x][0], "prob_out": on_off[x][1]} for x in range(len(on_off))
}
add_weights(G, on_off_dict)


pop_size = 100
walk_length = [7, 15]
num_routes = [6, 12]
num_ppl = 50000
cap = 60
ppl = simulate_people(G, num_ppl)

pop = []
print("Generating initial routes...")
Route.initialize_class(G)
for i in tqdm(range(pop_size)):
    source, destination = choice(list(G.nodes()), size=2)
    routenum = np.random.randint(num_routes[0], num_routes[1] + 1)
    rts = []
    for i in range(routenum):
        length = np.random.randint(walk_length[0], walk_length[1] + 1)
        rts.append(Route(cap, random_walk(source, destination, length)))
    pop.append(Routes(rts))
