import networkx as nx
from helpers.k_centers_problem import CreateGraph

def get_closest_nbrs(node, num=None):
    nbrs = sorted(list(G.neighbors(0)), key=lambda n: G[0][n]['length'], reverse=True)
    if not num:
        return nbrs
    return nbrs[:num]

G = CreateGraph(75, fname='final_full_graph.csv')
