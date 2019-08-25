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
from routePath import optimal_route

img_dir = "imgs/"


def clear_imgs():
    os.rmdir(img_dir)
    os.makedirs(img_dir)


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
    print(color)
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

    fig = plt.figure(frameon=False)
    fig.set_size_inches(13,6.25)
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
        plt.savefig(img_dir + fname, quality=100)

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
            v_next = route.v_disabled[i+1]
            G_.add_edge(v_curr, v_next,
                        length=G[v_curr][v_next]['length'], route=list_route.index(route))

    for route in list_route:
        on_off_dict = { v: {'prob_in': nx.get_node_attributes(G,'prob_in')[v], 'prob_out': nx.get_node_attributes(G,'prob_out')[v] } for v in route.v_disabled}
        add_weights(G_, on_off_dict)

    plot_graph(G_, edge_weight="route", fname=fname, show=show)

def plot_route_sd(G, list_route, src, dest, fname=None, show=False):
    opt_route = optimal_route(list_route, src, dest)
    plot_route(G, [r[0] for r in opt_route], fname=fname, show=show)
"""
def plot_route_sd(G, list_route, src, dest, fname=None, show=False):
    G_ = nx.Graph()
    opt_route = optimal_route(list_route, src, dest)
    if len(opt_route) == 1:
        rt = opt_route[0][0]
        if min(rt.v_disabled.index(src), rt.v_disabled.index(dest))+1 == max(rt.v_disabled.index(src), rt.v_disabled.index(dest)):
            i = min(rt.v_disabled.index(src), rt.v_disabled.index(dest))
            G_.add_edge(rt.v_disabled[i], rt.v_disabled[i+1],
                        length=G[rt.v_disabled[i]][rt.v_disabled[i+1]]['length'], route=0)
            v_lst = (rt.v_disabled[i],rt.v_disabled[i+1])
            on_off_dict = { v: {'prob_in': nx.get_node_attributes(G,'prob_in')[v], 'prob_out': nx.get_node_attributes(G,'prob_out')[v] } for v in v_lst }
            add_weights(G_, on_off_dict)
        else:
            for i in range(min(rt.v_disabled.index(src), rt.v_disabled.index(dest)), max(rt.v_disabled.index(src), rt.v_disabled.index(dest))):
                G_.add_edge(rt.v_disabled[i], rt.v_disabled[i+1],
                            length=G[rt.v_disabled[i]][rt.v_disabled[i+1]]['length'], route=0)

            on_off_dict = { rt.v_disabled[i]: {'prob_in': nx.get_node_attributes(G,'prob_in')[rt.v_disabled],
                            'prob_out': nx.get_node_attributes(G,'prob_out')[rt.v_disabled] }
                            for i in range(min(rt.v_disabled.index(src), rt.v_disabled.index(dest)),
                            max(rt.v_disabled.index(src), rt.v_disabled.index(dest))+1) }
            add_weights(G_, on_off_dict)
    else:
        for j in range(len(opt_route)):
            print(j)
            src_l = opt_route[j][1]
            if j!=len(opt_route)-1:
                rt_possible = [opt_route[j][0], opt_route[j+1][0]]
                dest_l = opt_route[j+1][1]
            else:
                rt_possible = [opt_route[j][0], opt_route[j-1][0]]
                dest_l = dest

            if src_l in rt_possible[0].v_disabled and dest_l in rt_possible[0].v_disabled:
                rt = rt_possible[0]

            elif src_l in rt_possible[1].v_disabled and dest_l in rt_possible[1].v_disabled:
                rt = rt_possible[1]

            if min(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l))+1 == max(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l)):
                i = min(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l))
                G_.add_edge(rt.v_disabled[i], rt.v_disabled[i+1],
                            length=G[rt.v_disabled[i]][rt.v_disabled[i+1]]['length'], route=0)
                v_lst = (rt.v_disabled[i],rt.v_disabled[i+1])
                on_off_dict = { v: {'prob_in': nx.get_node_attributes(G,'prob_in')[v], 'prob_out': nx.get_node_attributes(G,'prob_out')[v] } for v in v_lst }
                add_weights(G_, on_off_dict)
            else:
                for i in range(min(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l)), max(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l))):
                    G_.add_edge(rt.v_disabled[i], rt.v_disabled[i+1],
                                length=G[rt.v_disabled[i]][rt.v_disabled[i+1]]['length'], route=0)

                on_off_dict = { rt.v_disabled[i]: {'prob_in': nx.get_node_attributes(G,'prob_in')[rt.v_disabled],
                                'prob_out': nx.get_node_attributes(G,'prob_out')[rt.v_disabled] }
                                for i in range(min(rt.v_disabled.index(src_l), rt.v_disabled.index(dest_l)),
                                max(rt.v_disabled.index(src), rt.v_disabled.index(dest))+1) }
                add_weights(G_, on_off_dict)

    plot_graph(G_, edge_weight='route', fname=fname, show=show)
"""
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
