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
        first = np.array(get_nbrs(node, first=len(walk)+1))
        last = np.array(get_nbrs(node, first=10))

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
    num_ppl = (prev_dg.size(weight="weight") - sim_dg.size(weight="weight")) / routes.cap
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
# c1: [0,1]*50, c2: [0,1]*10 c3: avg_len = 10.98
c1,c2,c3 = (50,20,100)
opt_bus = 20
elite = 0.1
iter = 10
crossover_perc = 0.9
mutation_prob = 0.15

pop = []
print("Generating initial routes ...")
Route.initialize_class(G)
for i in tqdm(range(pop_size)):
    source, destination = choice(list(G.nodes()), size=2)
    routenum = np.random.randint(num_routes[0], num_routes[1] + 1)
    rts = []
    for i in range(routenum):
        length = np.random.randint(walk_length[0], walk_length[1] + 1)
        rts.append(Route(cap, random_walk(source, destination, length)))
    pop.append(Routes(rts))

# Add graph to the routes
Route.initialize_class(G)
Routes.initialize_class(G)

# Use a GA to solve the problem
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

print('\nFinal Solution:')
print(best)
opt_seats_taken, opt_num_bus, opt_cum_len = fitness(best, ppl, c1, c2, c3, opt_bus, components=True)
print(f'\nThis route has {opt_num_bus} buses, On average, {opt_seats_taken}% of seats are occupied, Total length of all routes combined is {round(opt_cum_len,2)} km')
