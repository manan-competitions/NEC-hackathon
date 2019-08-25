import numpy as np
import os
import networkx as nx
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# from tqdm import tqdm
from helpers.utils import add_weights, get_diff, fitness, simulate_people
from helpers.route import Route, Routes

img_dir = "imgs/"


def clear_imgs():
    os.rmdir("imgs")
    os.makedirs("imgs")


def get_route(pre, mode):
    with open(f"final_data/{pre}_optimal_routes_{mode}.json") as f:
        data = json.load(f)["data"]
    rs = []
    for row in data:
        rs.append(Route(row[0], row[1]))
    return Routes(rs)


def plot_graph(G, fname=None, edge_weight="length", vertex_weight="both", show=False):
    pos = nx.spring_layout(G)
    # color = range(G_diff.size())
    color = [G[u][v][edge_weight] for u, v in G.edges()]
    node_color = "Y"
    if vertex_weight == "both":
        _node_color = [
            nx.get_node_attributes(G, "prob_in")[x]
            + nx.get_node_attributes(G, "prob_out")[x]
            for x in list(G.nodes())
        ]
    else:
        _node_color = [
            nx.get_node_attributes(G, vertex_weight)[x] for x in list(G.nodes())
        ]

    nx.draw(
        G,
        pos,
        node_color=_node_color,
        edge_color=color,
        width=1,
        cmap=cm.get_cmap("Reds"),
        edge_cmap=cm.get_cmap("rainbow"),
        with_labels=True,
    )

    if fname:
        plt.savefig(img_dir + fname)

    if show:
        plt.show()


def plot_route(G, list_route, fname=None, show=False):
    if not isinstance(list_route, list):
        list_route = [list_route]

    color = cm.get_cmap("rainbow")(np.linspace(0, 1, len(list_route)))
    G_ = nx.DiGraph()
    for route in list_route:
        for i in range(len(route.v_disabled) - 1):
            v_curr = route.v_disabled[i]

            G_.add_edge(
                v_curr,
                v_next,
                length=G[v_curr][v_next]["length"],
                route=list_route.index(route),
            )

        on_off_dict = {
            v: {
                "prob_in": nx.get_node_attributes(G, "prob_in")[v],
                "prob_out": nx.get_node_attributes(G, "prob_out")[v],
            }
            for v in route.v_disabled
        }
        add_weights(G_, on_off_dict)
    plot_graph(G_, edge_weight="route", fname=fname, show=show)


def get_stats(G, route1, route2, num_of_people=5000):
    ppl = simulate_people(G, num_of_people)
    cp, cn, G_diff = get_diff(route1, route2, G, ppl, vertex_weight="both")
    diffs = nx.get_edge_attributes(G_diff, "weight")
    # Were better in route1
    affected = []
    # Were benefitted in route2
    benefited = []
    for k, v in diffs.items():
        if v < 0:
            benefited.append((k, -v))
        elif v > 0:
            affected.append((k, v))

        affected.sort(key=lambda x: x[1], reverse=True)
        benefited.sort(key=lambda x: x[1], reverse=True)
    miles_traveled_1 = fitness(
        route1, ppl, G, {"optimal": (0, 0, 0)}, 0, 5, ret_miles_traveled=True
    )
    miles_traveled_2 = fitness(
        route2, ppl, G, {"optimal": (0, 0, 0)}, 0, 5, ret_miles_traveled=True
    )
    inter = set(miles_traveled_1.keys()).intersection(set(miles_traveled_2.keys()))
    inconvinience = []
    for k in inter:
        inconvinience.append(
            (
                k,
                miles_traveled_1[k],
                miles_traveled_2[k],
                miles_traveled_1[k] - miles_traveled_2[k],
            )
        )
    inconvinience.sort(key=lambda x: x[3])
    return cp, cn, affected, benefited, inconvinience
