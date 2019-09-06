import numpy as np
import csv
import json
import networkx as nx
from numpy.random import choice
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from helpers.route import Route, Routes


def dump(best, pre, mode):
    data_out = {
        "data": [
            (int(route.num), [int(r) for r in route.v_disabled])
            for route in best.routes
        ]
    }
    with open(f"final_data/{pre}_optimal_routes_{mode}.json", "w") as f:
        json.dump(data_out, f, indent=2)
        print("-- cached --")


def get_diff(route1, route2, G, ppl, show_vertex_color=True, vertex_weight="prob_in"):
    G_1 = fitness_trunc(route1, ppl, G)
    G_2 = fitness_trunc(route2, ppl, G)
    G_diff = routes_diff(G_1, G_2, "weight")

    count_pos = 0
    count_neg = 0
    for k in nx.get_edge_attributes(G_diff, name="weight").keys():
        if G_diff[k[0]][k[1]]["weight"] == 0:
            G_diff.remove_edge(*k)
        elif G_diff[k[0]][k[1]]["weight"] > 0:
            count_pos += G_diff[k[0]][k[1]]["weight"]
        elif G_diff[k[0]][k[1]]["weight"] < 0:
            count_neg -= G_diff[k[0]][k[1]]["weight"]

    return count_pos, count_neg, G_diff


def get_routes_csv(fname, cap=150):
    rs = []
    with open(fname) as f:
        for row in csv.reader(f):
            rs.append(Route(cap, [int(r) for r in row]))
        return Routes(rs)


# Create a weighted undirected graph G from a file or a variable
def CreateGraph(n, pre, file=True, fname=None, adj_matrix=None, node_prob=False):
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

    if wtMatrix.shape != (n, n):
        raise Exception(
            f"Incorrect Shape: Expected ({n},{n}) but got {wtMatrix.shape} instead"
        )

    # Adds egdes along with their weights to the graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i, n):
            G.add_edge(i, j, length=wtMatrix[i][j])

    # Add individual node probabilites
    if node_prob:
        on_off = get_node_vals(f"./data/{pre}_node_probs.csv")
        on_off_dict = {
            x: {"prob_in": on_off[x][0], "prob_out": on_off[x][1]}
            for x in range(len(on_off))
        }
        add_weights(G, on_off_dict)
    return G


def dist_km(lat1, lat2, lon1, lon2):
    # approximate radius of earth in km
    R = 6373.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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
        first = np.array(get_nbrs(G, node, first=len(walk) + 1))
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


def fitness(
    routes,
    sim_dg,
    G,
    consts,
    opt_bus,
    max_trips,
    components=False,
    mode="optimal",
    ret_graph=False,
    ret_miles_traveled=False,
):
    c1, c2, c3 = consts[mode]
    # Prevent this method from having side-effects
    # Make a copy of the param
    prev_dg = sim_dg
    sim_dg = deepcopy(sim_dg)
    miles_traveled = dict()
    deboard_dict = dict()
    for route in routes.routes:
        current_capacity = route.num * route.cap
        for i in range(len(route.v_disabled) - 1):
            current_capacity += deboard_dict.get(i, 0)
            deboard_dict[i] = 0
            for k in set(sim_dg[route.v_disabled[i]]).intersection(
                set(route.v_disabled[i + 1 :])
            ):
                p = route.v.index(k)
                people_boarding = min(
                    sim_dg[route.v_disabled[i]][k]["weight"], current_capacity
                )
                miles_traveled[(i, k)] = (
                    miles_traveled.get((i, k), 0)
                    + sum(
                        [
                            G[x[0]][x[1]]["length"]
                            for x in zip(route.v[i:p], route.v[i + 1 : p + 1])
                        ]
                    )
                    * people_boarding
                )
                sim_dg[route.v_disabled[i]][k]["weight"] -= people_boarding
                deboard_dict[k] = deboard_dict.get(k, 0) + people_boarding
                current_capacity -= people_boarding
    num_ppl = prev_dg.size(weight="weight") - sim_dg.size(weight="weight")

    num_buses_per_route = np.sum(
        [np.ceil(route.num / max_trips) for route in routes.routes]
    )
    if ret_miles_traveled:
        return miles_traveled

    if ret_graph:
        return sim_dg

    if components:
        return num_ppl, num_buses_per_route, routes.cum_len / routes.num_buses

    return max(
        0,
        c1 * num_ppl
        + c2 * num_buses_per_route
        + c3 * routes.cum_len / routes.num_buses,
    )


def fitness_trunc(routes, sim_dg, G):
    prev_dg = sim_dg
    sim_dg = deepcopy(sim_dg)
    miles_traveled = dict()
    deboard_dict = dict()
    for route in routes.routes:
        current_capacity = route.num * route.cap
        for i in range(len(route.v_disabled) - 1):
            current_capacity += deboard_dict.get(i, 0)
            deboard_dict[i] = 0
            for k in set(sim_dg[route.v_disabled[i]]).intersection(
                set(route.v_disabled[i + 1 :])
            ):
                p = route.v.index(k)
                people_boarding = min(
                    sim_dg[route.v_disabled[i]][k]["weight"], current_capacity
                )
                miles_traveled[(i, k)] = (
                    miles_traveled.get((i, k), 0)
                    + sum(
                        [
                            G[x[0]][x[1]]["length"]
                            for x in zip(route.v[i:p], route.v[i + 1 : p + 1])
                        ]
                    )
                    * people_boarding
                )
                sim_dg[route.v_disabled[i]][k]["weight"] -= people_boarding
                deboard_dict[k] = deboard_dict.get(k, 0) + people_boarding
                current_capacity -= people_boarding
    return sim_dg


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


def GA(
    iter,
    pop,
    pop_size,
    G,
    num_ppl,
    consts,
    opt_bus,
    max_trips,
    elite,
    mutation_prob,
    crossover_perc,
    pre,
    mode="optimal",
    plot=False,
    every=10,
):
    print(f"\nTraining the Genetic Algorithm in mode: {mode} ...")
    new_pop = deepcopy(pop)
    ppl = simulate_people(G, num_ppl)
    avg_fit = []
    best_fit = []
    worst_fit = []

    for i in tqdm(range(iter)):
        # 		print(f'Iteration {i+1} / {iter}')
        curr_pop = deepcopy(new_pop)
        new_pop = []

        if (i + 1) % every == 0:
            dump(best, pre, mode)

        fit = [
            (p, fitness(p, ppl, G, consts, opt_bus, max_trips, mode=mode))
            for p in curr_pop
        ]

        fit.sort(reverse=True, key=lambda x: x[1])
        fit_vals = [f[1] for f in fit]
        if not (np.any(np.greater(fit_vals, 0))):
            print("Scaling negative weights")
            min = np.min(fit_vals)
            fit = [(fit_vals[i][0], fit[i][1]) for i in range(len(fit))]
        elif np.any(np.less_equal(fit_vals, 0)):
            print("Correcting negative vals")
            fit = np.array(fit)
            orig_shape = fit.shape
            positives = fit[np.greater(fit_vals, 0)]
            # print(positives)
            fit = np.ndarray.tolist(
                np.tile(positives, (int(np.ceil(fit.shape[0] / positives.shape[0])), 1))
            )[: len(fit_vals)]

            # print(fit)
            # print(len(fit))
            # print(len(fit_vals))
            assert len(fit) == len(fit_vals)

        # Transfer elite directly to the next generation
        elite_num = int(elite * pop_size)
        elite_pop = []
        for j in range(elite_num):
            elite_pop.append(deepcopy(fit[j][0]))
        #    print('-- Selection')

        # Select the rest according to the fitness function (Selection)
        new_pop = choice(
            curr_pop,
            size=pop_size - elite_num,
            p=[f[1] for f in fit] / np.sum([f[1] for f in fit]),
        )

        #    print('-- Crossover')
        # Crossover the rest
        cross_size = int(crossover_perc * len(new_pop))
        cross_size += cross_size % 2
        cross_routes = choice(new_pop, size=cross_size)
        for i in range(0, cross_size, 2):
            curr_pop[i].crossover(curr_pop[i + 1])

        #    print('-- Mutation')
        # Mutatie every elemnt (may not take place actually)
        for p in new_pop:
            p.mutate(mutation_prob)

        new_pop = np.concatenate([new_pop, elite_pop])
        best = fit[0][0]
        # print(f'-- Average: {np.mean([f[1] for f in fit])} Best: {fit[0][1]} Worst: {fit[-1][1]}')
        avg_fit.append(np.mean([f[1] for f in fit]))
        best_fit.append(fit[0][1])
        worst_fit.append(fit[-1][1])

    plt.plot(avg_fit, color="b")
    plt.plot(best_fit, color="g")
    plt.plot(worst_fit, color="r")
    plt.savefig(f"final_data/{pre}_{mode}.jpg")

    if plot:
        plt.plot(avg_fit, color="b")
        plt.plot(best_fit, color="g")
        plt.plot(worst_fit, color="r")
        plt.show()

    return best, ppl, new_pop, avg_fit, best_fit, worst_fit


def routes_diff(G1, G2, weight):
    G3 = nx.DiGraph()
    G1 = deepcopy(G1)
    G2 = deepcopy(G2)
    for node in G1:
        for key in G2:
            G1Val = 0
            G2Val = 0
            try:
                G1Val = G1[node][key][weight]
            except:
                G1Val = 0
            finally:
                try:
                    G2Val = G2[node][key][weight]
                except:
                    G2val = 0
                finally:
                    G3.add_edge(node, key, weight=G1Val - G2Val)
    return G3
