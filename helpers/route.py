import numpy as np

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
        v(list): (ordered) list of vertices covered in this route
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

    def crossover(self, route):
        v1 = set(self.v[1:-1])
        v2 = set(route.v[1:-1])
        common = list(v1.intersection(v2))
        print(common)

        if len(common) == 0:
            return

        if len(common) == 1:
            ind_1 = self.v.index(common[0])
            ind_2 = route.v.index(common[0])
            temp_v = self.v
            self.v = self.v[:ind_1] + route.v[ind_2:]
            route.v = route.v[:ind_2] + temp_v[ind_1:]

        else:
            elem1, elem2 = np.random.choice(common, size=2, replace=False)
            ind_1_l = min(self.v.index(elem1),self.v.index(elem2))
            ind_1_u = max(self.v.index(elem1),self.v.index(elem2))
            ind_2_l = min(route.v.index(elem1),route.v.index(elem2))
            ind_2_u = max(route.v.index(elem1),route.v.index(elem2))
            temp_v = self.v
            self.v[ind_1_l+1:ind_1_u] = route.v[ind_2_l+1:ind_2_u]
            route.v[ind_2_l+1:ind_2_u] = temp_v[ind_1_l+1:ind_1_u]
