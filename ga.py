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
from helpers.utils import (
    get_node_vals,
    get_nbrs,
    random_walk,
    add_weights,
    fitness,
    simulate_people,
    GA,
    CreateGraph,
)

if len(sys.argv) < 2:
    print("Usage: python3 ga.py [prefix for in/out file]")
    exit(0)

pre = sys.argv[1]

# Get the graph
G = CreateGraph(75, fname=f"./data/{pre}_final_full_graph.csv", pre=pre, node_prob=True)

# Hyper parameters
pop_size = 100
walk_length = [3, 15]
num_routes = [6, 12]
num_ppl = 50000
cap = 60
# c1: [0,1]*50, c2: [0,1]*10 c3: avg_len = 10.98
c1, c2, c3 = (50, 20, 100)
opt_bus = 20
elite = 0.1
iter = 20
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
best, final_pop = GA(
    iter,
    pop,
    pop_size,
    G,
    num_ppl,
    c1,
    c2,
    c3,
    opt_bus,
    elite,
    mutation_prob,
    crossover_perc,
)

print("\nFinal Solution:")
print(best)
opt_seats_taken, opt_num_bus, opt_cum_len = fitness(
    best, ppl, c1, c2, c3, opt_bus, components=True
)
print(
    f"\nThis route has {opt_num_bus} buses, On average, {opt_seats_taken}% of seats are occupied, Total length of all routes combined is {round(opt_cum_len,2)} km"
)

data_out = {
    "data": [(int(route.num), [int(r) for r in route.v]) for route in best.routes]
}

with open("data/{pre}_optimal_routes.json", "w") as f:
    json.dump(data_out, f, indent=2)
