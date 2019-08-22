import numpy as np
from pprint import pprint,pformat

get_bin = lambda x, n: format(x, 'b').zfill(n)

def get_nbrs(node, G, first=None, last=None):
    nbrs = sorted(list(G.neighbors(node)), key=lambda n: G[node][n]['length'], reverse=True)

    if first:
        return nbrs[:first]
    elif last:
        return nbrs[-last:]
    else:
        return nbrs

class Route(object):
    """
    Specifies a route for buses to run on
    Attributes:
        num(int): Number of buses running on this routes
        num_bits(int): number of bits to use in the binary description of num
        v(list): (ordered) list of vertices covered in this route
    Methods:
        mutate(G,mut_prob): Mutate the given route
        crossover(route): Crossover current route with the specified route
    """
    def __init__(self, vertices, num=None):
        self.num_bits = 3
        if not num:
            self.num = np.random.randint(1,2**self.num_bits)
        else:
            self.num = num
        self.v = vertices

    def __str__(self):
        return f'{self.num} | {len(self.v)} | {self.v}'
    # To enable better printing
    __repr__ = __str__

    def mutate(self, G, mut_prob=0.05):
        # Mutate the number of buses
        bin_num = list(str(get_bin(self.num, self.num_bits)))
        for i in range(len(bin_num)):
            if np.random.rand() < mut_prob:
                bin_num[i] = str(abs(int(bin_num[i])-1))
        self.num = 4*int(bin_num[0]) + 2*int(bin_num[1]) + 1*int(bin_num[2])

        # Mutate the route
        print(self.v)
        for i in range(len(self.v)):
            if np.random.rand() < mut_prob:
                nbrs = get_nbrs(self.v[i], G, first=len(self.v)+1)
                for n in nbrs[:]:
                    if n in self.v:
                        nbrs.remove(n)
                probs = np.array([G[self.v[i]][n]['length'] for n in nbrs])
                probs = probs / np.sum(probs)
                self.v[i] = np.random.choice(nbrs, p=probs)
        print(self.v)

    def crossover(self, other_route):
        v1 = set(self.v[1:-1])
        v2 = set(other_route.v[1:-1])
        common = list(v1.intersection(v2))
        print(common)

        if len(common) == 0:
            return

        if len(common) == 1:
            ind_1 = self.v.index(common[0])
            ind_2 = other_route.v.index(common[0])
            temp_v = self.v
            self.v = self.v[:ind_1] + other_route.v[ind_2:]
            route.v = other_route.v[:ind_2] + temp_v[ind_1:]

        else:
            elem1, elem2 = np.random.choice(common, size=2, replace=False)
            ind_1_l = min(self.v.index(elem1),self.v.index(elem2))
            ind_1_u = max(self.v.index(elem1),self.v.index(elem2))
            ind_2_l = min(other_route.v.index(elem1),other_route.v.index(elem2))
            ind_2_u = max(other_route.v.index(elem1),other_route.v.index(elem2))
            temp_v = self.v
            self.v[ind_1_l+1:ind_1_u] = other_route.v[ind_2_l+1:ind_2_u]
            other_route.v[ind_2_l+1:ind_2_u] = temp_v[ind_1_l+1:ind_1_u]

class Routes(object):
    """
    Collection of routes to be used as a population for the final Genetic Algorithm
    Attributes:
    """
    def __init__(self, num_routes, list_routes):
        self.num_routes = num_routes
        self.routes = list_routes

    def __str__(self):
        return pformat([self.num_routes] + [r for r in self.routes])
    __repr__ = __str__

    def mutate(self, G, mut_prob=0.05, cross_perc=0.3):
        # Mutate individual routes
        for route in self.routes:
            if np.random.rand() < mut_prob:
                route.mutate(G, mut_prob)

        # internally Crossover some routes
        num_cross = int(cross_perc*self.num_routes)
        num_cross = num_cross if not num_cross%2 else num_cross+1
        cross_routes = np.random.choice(self.routes, replace=False, size=num_cross)
        for i in range(0,num_cross,2):
            cross_routes[i].crossover(cross_routes[i+1])

    def crossover(self, other_routes, cross_transfer=0.1, cross_perc=0.3):
        num_cross = int(cross_perc*max(self.num_routes,other_routes.num_routes))
        num_transfer = int(cross_transfer*max(self.num_routes,other_routes.num_routes))

        # Transfer some routes
        ind_1 = np.random.choice(range(self.num_routes), replace=False, size=num_transfer)
        ind_2 = np.random.choice(range(other_routes.num_routes), replace=False, size=num_transfer)
        for i in range(num_transfer):
            temp = self.routes[ind1]
            self.routes[ind1] = other_routes[ind2]
            other_routes[ind2] = temp

        # Crossover some routes
        cross_1 = np.random.choice(self.routes, replace=False, size=num_cross)
        cross_2 = np.random.choice(other_routes.routes, replace=False, size=num_cross)
        for i in range(num_cross):
            cross_1[i].crossover(cross_2[i])
