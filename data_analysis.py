import pandas as pd
import numpy as np
from pprint import pprint
import csv
import sys
from helpers.k_centers_problem import k_centers, DrawGraph
from helpers.utils import CreateGraph
import matplotlib.pyplot as plt

def dist_km(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R*c

if len(sys.argv) < 4:
    print('Usage: python3 data_analysis.py [path/to/data].csv [num_stops] [num_centers]')
    exit(0)

num_stops = int(sys.argv[2])
#infile = './dataset/may-trimester-2017-stop-ridership-ranking-saturday-csv-9.csv'
infile = sys.argv[1]
max_radius = 1
k = int(sys.argv[3])

data = pd.read_csv(infile)
data = data.drop(['DAY_OF_WEEK', 'STOP_NAME', 'RANK'], axis=1)
cols = list(data.columns)
data_arr = np.array(data)
#print(cols)

stops = set()
full_data = data_arr[:,1][np.argsort(data_arr[:,4])]
num_stops = min(num_stops, full_data.shape[0])
i = num_stops
while len(stops) < num_stops:
    stops = set(full_data[-i:])
    i += 1

final_data = []
final_cols = ['STOP_NUM', 'LAT', 'LON', 'ON', 'OFF']
for stop in stops:
    ind = np.where(data_arr[:,1]==stop)
    on = np.sum(data_arr[:,2][ind])
    off = np.sum(data_arr[:,3][ind])
    lat = data_arr[:,5][ind][0]
    lon = data_arr[:,6][ind][0]
    final_data.append((stop, lat, lon, on, off))

final_data = np.array(final_data)

stop_dict = dict()
for i in range(final_data[:,0].shape[0]):
    stop_dict[i] = final_data[:,0][i]

#Create initial data
dists = np.zeros((num_stops, num_stops))
for i in range(final_data.shape[0]):
    stop = final_data[i,0]
    lat = final_data[i,1]
    lon = final_data[i,2]
    inds = np.where(dist_km(final_data[:,1], lat, final_data[:,2], lon) <= max_radius)[0]
    #dists[i,inds] = np.sqrt((np.square(final_data[inds,1]-lat) + np.square(final_data[inds,2]-lon)))/max_radius
    dists[i,inds] = dist_km(final_data[inds,1], lat, final_data[inds,2], lon) / max_radius
print(f'{np.min(dists)} < r = {max_radius} < {np.max(dists)}')
print(100*np.sum((dists<=max_radius).astype(int))/dists.shape[0]**2, '% of values are included in the graph')
G = CreateGraph(n=num_stops, adj_matrix=dists, file=False)
centers = k_centers(G, k)
#DrawGraph(G, centers)
#plt.show()

# Create data for a fully connected graph with just the bus stops
full_dists = np.zeros((k, k))
for i in range(len(centers)):
    lat = final_data[centers[i],1]
    lon = final_data[centers[i],2]
    full_dists[i,:] = np.sqrt((np.square(final_data[centers][:,1]-lat) + np.square(final_data[centers][:,2]-lon)))

# Save info for later debugging
with open('dataset/labels.txt','w') as f:
    pprint(final_cols, f)
    pprint(stop_dict, f)

np.savetxt('dataset/final_full_graph.csv', full_dists, delimiter=',', fmt='%1.3f')
np.savetxt('dataset/node_vals.csv', final_data[centers,1:], delimiter=',', fmt='%i')
np.savetxt('dataset/node_probs.csv', final_data[centers,3:]/np.sum(final_data[centers,3:], axis=0), delimiter=',', fmt='%3.3f')
