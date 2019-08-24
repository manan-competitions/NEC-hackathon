import sys
import json
import csv
import numpy as np
import pandas as pd
from helpers.k_centers_problem import k_centers, DrawGraph
from helpers.utils import CreateGraph, dist_km, remove_duplicates

def get_reverse_dict(fname):
	with open(fname) as f:
		data = json.load(f)
	return { int(v['unique_name']): int(k) for k,v in data.items()}

def f(x, G, centers):
	min_len = np.min([G[c][x]['length'] for c in centers])
	K =  np.where(centers==min_len)[0][0]
	return K

if len(sys.argv) < 4:
	print('Usage: python3 get_orig.py [path/to/data].csv [num_stops] [num_centers] [prefix for out_files]')
	exit(0)

num_stops = int(sys.argv[2])
infile = sys.argv[1]
max_radius = 2.5
k = int(sys.argv[3])
pre = sys.argv[4]

orig_data = pd.read_csv(infile)
routes = list(set(orig_data['COTA_ROUTE']))

route_stops = []
for route in routes:
	route_df = orig_data.loc[orig_data['COTA_ROUTE']==route]
	vec_sort = route_df[['UNIQUE_STOP_NUMBER', 'RANK']].sort_values(by='RANK')
	route_stops.append(np.array(vec_sort['UNIQUE_STOP_NUMBER']))

rev_dict = get_reverse_dict(f'data/{pre}_labels.json')
G = CreateGraph(n=num_stops, fname=f'data/{pre}_initial_graph.csv', pre=pre)
centers = k_centers(G, k)
#print(sorted(centers))

all_routes = []
for route in route_stops:
	temp_route = [int(rev_dict[r]) for r in route if r in rev_dict]
	final_route = remove_duplicates([centers.index(v) if v in centers else f(v, G, centers) for v in temp_route])
	all_routes.append([1, [int(route) for route in final_route]])

with open(f'final_data/{pre}_optimal_routes_original.json', 'w') as fl:
	json.dump({'data': all_routes}, fl, indent=2)
