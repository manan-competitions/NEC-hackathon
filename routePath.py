import networkx as nx
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt

"""
Routes.routes = (list of 'Route')
Route.v = (ordered set of vert)
"""


def optimal_route(routels, src, dest):

    print(src, dest)
    # Check if direct Path - O(n)
    direct_possible = []
    for rot in routels.routes:
        if (src in rot.v_disabled) and (dest in rot.v_disabled):
            direct_possible.append(rot)
    if direct_possible:
        min_dist = -1
        for ix in range(len(direct_possible)):
            rot = direct_possible[ix].v_disabled
            st_i, end_i = 0, 0
            for i in range(len(rot)):
                if src == rot[i]:
                    st_i = i
                elif dest == rot[i]:
                    end_i = i
            dist = abs(st_i - end_i)
            if dist < min_dist or min_dist == -1:
                min_dist = dist
                low = min(st_i, end_i)
                min_path = [(direct_possible[ix], rot[j]) for j in range(low, low+dist)]
        return min_path

    # Generate nx graph and HashTableLists
    rats = defaultdict(list)
    G = nx.Graph()
    for rotno, rot in enumerate(routels.routes):
        vrot = rot.v_disabled
        for i in range(len(vrot) - 1):
            G.add_edges_from([(vrot[i], vrot[i+1])])
            rats[vrot[i]].append(rotno)
            rats[vrot[i+1]].append(rotno)
    # Find all shortest path
    print(G.edges())
    print(rats)
    shortpaths = nx.all_shortest_paths(G, src, dest)
    #Search through product of routes of all shortest paths
    switchlist = []
    routelist = []
    pather = []
    for stpath in shortpaths:
        routcombi = [rats[stop] for stop in stpath]
        routeset = list(product(*routcombi))
        for rot in routeset:
            switch = 0
            for i in range(len(rot)-1):
                if rot[i+1] != rot[i]:
                    switch += 1
            switchlist.append(switch)
            pather.append(stpath)
        routelist.extend(routeset)
    # Return all shortest path with least switches
    minsw = min(switchlist)
    for i in range(len(switchlist)):
        if switchlist[i] == minsw:
            break
    best_path = [(routels.routes[routelist[i][j]], pather[i][j]) for j in range(len(pather[i]))]
    return best_path
