import numpy as np

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
