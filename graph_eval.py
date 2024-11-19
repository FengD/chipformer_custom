import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import time
import pickle
# from input_data import load_data
from preprocessing import *
import graph_args as args
import graph_model
from mingpt.place_db import PlaceDB
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

benchmark_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4']
# benchmark_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
# 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', 
# 'ibm01', 'ibm02', 'ibm03', 'ibm04']

model_path = "save_graph_models/2024-11-19-11-38-39-0.9874-0.9865.pkl"
state_dict = torch.load(model_path)

result = {}

for benchmark in benchmark_list:
    place_db = PlaceDB(benchmark)
    features = place_db.features
    adj = place_db.adj
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_norm = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    features = torch.tensor(features, dtype=torch.float32)
    
    model = getattr(graph_model, args.model)(adj_norm)
    model.load_state_dict(state_dict, strict = True)
    model.eval()

    z_emb = model.encode(features)

    print("z_emb", z_emb)
    print("z_emb shape", z_emb.shape)
    z_emb_avg = torch.mean(z_emb, axis=0)
    result[benchmark] = z_emb_avg

strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
pickle.dump(result, open("circuit_g_token-{}.pkl".format(strftime),'wb'))

# the graph_eval.py is used to transform the circuit feature of the node graph to a high level feature

# output of pickle one of the eval result use the training parameters in the graph_args.py
# {'adaptec1': tensor([ 0.4371, -0.2983, -0.1665, -0.0065, -0.0415, -0.5930, -0.0202, -0.3140,
# -0.1697,  0.1594, -0.0920,  0.2450, -0.0451,  0.0189, -0.1841,  0.1438,
#     0.3394, -0.3279, -0.1858,  0.1149, -0.0847,  0.1246, -0.0881,  0.1055,
# -0.1648,  0.0790, -0.1099, -0.1600, -0.1187,  0.1262, -0.1529,  0.1334],requires_grad=True),
# 'adaptec2': tensor([-0.2558,  0.0857,  0.2109,  0.2362, -0.1428,  0.2356, -0.2333,  0.2450,
# -0.2171, -0.1795,  0.2155, -0.2072,  0.2477, -0.2824, -0.1798,  0.1520,
# -0.2732, -0.1305, -0.1799, -0.3029, -0.2565,  0.1893, -0.2494,  0.2318,
# -0.1872, -0.2480, -0.2151,  0.1175, -0.2430,  0.2216, -0.1786,  0.1599],requires_grad=True), 
# 'adaptec3': tensor([-0.1791,  0.1601, -0.2042, -0.1555,  0.0947,  0.2801,  0.0779,  0.1429,
#     0.1001,  0.2362,  0.0620,  0.2586, -0.0130, -0.0350,  0.0213, -0.2116,
#     0.2183,  0.1594, -0.0471,  0.0075,  0.0699, -0.2358,  0.1304, -0.1015,
#     0.1711,  0.0746,  0.0694, -0.2944,  0.1887, -0.0944,  0.1753, -0.0799],requires_grad=True), 
# 'adaptec4': tensor([ 0.1360, -0.0013,  0.1035, -0.0150,  0.0958, -0.1544,  0.1298, -0.1248,
#     0.1548, -0.1538, -0.1588, -0.1717, -0.1439,  0.2502,  0.2426,  0.0229,
# -0.1534,  0.1188,  0.3040,  0.1645,  0.1593,  0.0451,  0.1009, -0.1236,
#     0.0557,  0.0705,  0.1521,  0.2445,  0.0463, -0.1341,  0.0310, -0.1293],requires_grad=True)}

