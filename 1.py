from string import digits
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

path = ".//new/"

data = HeteroData()

class_A = {}
class_B = {}
relation = {}
relation_feat = {}
A_feat = []
B_feat = []
metapath_dict = {}

with open('./color.txt', "r") as f:
    for index, color in enumerate(set(f.read().split('\n'))):
        class_B[color.strip()] = index

with open('./keyword.txt', "r") as f:
    for _, keyword in enumerate(set(f.read().split('\n'))):
        if len(keyword.strip()) > 1:
            relation[keyword.strip()] = []
            relation_feat[keyword.strip()] = []

for index, i in enumerate(os.listdir(path)):
    table = str.maketrans('', '', digits)
    class_label = i.translate(table).split('.')[-2].replace('_', ' ')
    class_A[class_label] = index

    with open(os.path.join(path, i), "r") as f:
        txt = f.read()
    for j in txt.split('\n'):
        if len(j.split(',')) == 2:
            src, dst = j.split(',')
            relation[src.strip()].append([index, class_B[dst.strip()]])

data['A'].x = torch.rand(200, 768)
data['B'].x = torch.rand(328, 768)
for i in relation:
    data['A', i, 'B'].edge_index = torch.tensor(np.array(relation[i]), dtype=torch.long).t().contiguous().view(2, -1)
    data['B', i, 'A'].edge_index = data['A', i, 'B'].edge_index.flip([0])
    # data['A', i, 'B'].edge_attr = torch.rand(np.array(relation[i]).shape[0], 768)
    # data['B', i, 'A'].edge_attr = torch.rand(np.array(relation[i]).shape[0], 768)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN(hidden_channels=64, out_channels=16)
model = to_hetero(model, data.metadata(), aggr='sum')
out = model(data.x_dict, data.edge_index_dict)
print(out)
