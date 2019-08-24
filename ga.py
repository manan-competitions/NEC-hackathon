import numpy as np
import csv
import networkx as nx
from numpy.random import choice
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
import json
import sys
from helpers.route import Route, Routes
from helpers.utils import get_node_vals, get_nbrs, random_walk, add_weights, fitness, simulate_people, GA, CreateGraph
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 ga.py [prefix for in/out file]")
    exit(0)

pre = sys.argv[1]

# Get the graph
G = CreateGraph(75, fname=f'./data/{pre}_final_full_graph.csv', pre=pre, node_prob=True)
#G_ = CreateGraph(75, fname=f'./data/{pre}_final_full_graph.csv', pre=pre)

#print(np.max(nx.get_edge_attributes(G,'length')))

# Hyper parameters
pop_size = 100
walk_length = [7, 15]
num_routes = [12, 25]
num_ppl = 50000
cap = 150
consts = {
'optimal': (30, -1000, -70),
'people': (1, 0, 0),
'money': (0, 1000, 70),
}
opt_bus = 20
max_trips = 5 # no of times a bus can run on a specific route
elite = 0.1
iter = 5
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
        rts.append(Route(cap, random_walk(G, source, destination, length)))
    pop.append(Routes(rts))

# Add graph to the routes
Route.initialize_class(G)
Routes.initialize_class(G)

# Use a GA to solve the problem
best, ppl, final_pop = GA(iter, pop, pop_size, G, num_ppl, consts, \
                            opt_bus, max_trips, elite, mutation_prob, \
                            crossover_perc, mode='people')

print("\nFinal Solution:")
print(best)
best_fit = fitness(best, ppl, G, consts, opt_bus, max_trips, components=True)
print(f'\nThis route serves {100*best_fit[0]/best.cap}% ({best_fit[0]}) of people , About {best_fit[1]} buses run  with a total capacity of {best.cap} and the average length of a bus route is {best_fit[2]} km')

"""
data_out = { 'data': [(int(route.num), [int(r) for r in route.v_disabled]) for route in best.routes] }
with open('data/{pre}_optimal_routes.json','w') as f:
    json.dump(data_out, f, indent=2)
"""
