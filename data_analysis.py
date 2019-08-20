import pandas as pd
import numpy as np
from pprint import pprint
import csv
import sys
from helpers.k_centers_problem import CreateGraph, k_centers, DrawGraph
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print('Usage: python3 data_analysis.py [path/to/data].csv [num_stops] [num_centers]')
    exit(0)

num_stops = sys.argv[2]
#infile = './dataset/may-trimester-2017-stop-ridership-ranking-saturday-csv-9.csv'
infile = sys.argv[1]
max_radius = 0.05
k = sys.argv[3]

data = pd.read_csv(infile)
data = data.drop(['DAY_OF_WEEK', 'STOP_NAME', 'RANK'], axis=1)
cols = list(data.columns)
data_arr = np.array(data)
#print(cols)

stops = set()
i = num_stops
full_data = data_arr[:,1][np.argsort(data_arr[:,4])]
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

# Save info for later debugging
with open('helpers/labels.txt','w') as f:
    pprint(final_cols, f)
    pprint(stop_dict, f)

#Create initial data
dists = np.zeros((num_stops, num_stops))
for i in range(final_data.shape[0]):
    stop = final_data[i,0]
    lat = final_data[i,1]
    lon = final_data[i,2]
    inds = np.where(np.square(final_data[:,1]-lat) + np.square(final_data[:,2]-lon) <= max_radius**2)[0]
    dists[i,inds] = np.sqrt((np.square(final_data[inds,1]-lat) + np.square(final_data[inds,2]-lon)))/max_radius

#np.savetxt('initial_graph.csv', dists, delimiter=',', fmt='%1.3f')

G = CreateGraph(n=num_stops, adj_matrix=dists, file=False)
centers = k_centers(G, k)
DrawGraph(G, centers)
plt.show()

# Create data for a fully connected graph with just the bus stops
full_dists = np.zeros((k, k))
for i in range(len(centers)):
    lat = final_data[centers[i],1]
    lon = final_data[centers[i],2]
    full_dists[i,:] = np.sqrt((np.square(final_data[centers][:,1]-lat) + np.square(final_data[centers][:,2]-lon)))

np.savetxt('final_full_graph.csv', full_dists, delimiter=',', fmt='%1.3f')
