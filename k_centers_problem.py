"""
Code taken from:
https://github.com/MUSoC/Visualization-of-popular-algorithms-in-Python
"""

import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import csv

def k_centers(G, n):
	centers = []
	cities = G.copy()
	#add an arbitrary node, here, the first node,to the centers list
	centers.append(list(G.nodes)[np.random.randint(low=0, high=n)])
	cities.remove_node(centers[0])
	n = n-1 #since we have already added one center
	#choose n-1 centers
	while n!= 0:
		city_dict = {}
		for cty in cities:
			min_dist = float("inf")
			for c in centers:
				min_dist = min(min_dist,G[cty][c]['length'])
			city_dict[cty] = min_dist
		#print city_dict
		new_center = max(city_dict, key = lambda i: city_dict[i])
		#print new_center
		centers.append(new_center)
		cities.remove_node(new_center)
		n = n-1
	return centers

# Create a weighted undirected graph G from a file or a variable
def CreateGraph(n, k, file=True, fname=None, adj_matrix=None):
	"""
	Accepts either a file name or a list of args
	Arguements:
		file (Bool): whether to read from a file or from a variable
		num_vertices(int): number of vertices
		num_centers(int): number of centers (K)
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

	if wtMatrix.shape != (n,n):
		raise Exception(f'Incorrect Shape: Expected ({n},{n}) but got {wtMatrix.shape} instead')

	#Adds egdes along with their weights to the graph
	G = nx.Graph()
	for i in range(n) :
		for j in range(i,n):
			G.add_edge(i, j, length = wtMatrix[i][j])

	return G

#draws the graph and displays the weights on the edges
def DrawGraph(G, centers=[]):
	pos = nx.spring_layout(G)
	color_map = ['blue'] * len(G.nodes())
	#all the center nodes are marked with 'red'
	for c in centers:
		color_map[c] = 'red'
	nx.draw(G, pos, node_color = color_map, with_labels = True)  #with_labels=true is to show the node number in the output graph
	#edge_labels = nx.get_edge_attributes(G, 'length')
	#nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 11) #prints weight on all the edges

#main function
if __name__ == "__main__":
	n = 150
	k = 50
	G = CreateGraph(n=n, k=k, fname='final_out.csv')
	print('Graph created')
	centers = k_centers(G, k)
	print(sorted(centers))
	DrawGraph(G, centers)
	plt.show()
