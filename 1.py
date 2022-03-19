from string import digits
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero

# example:
# dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
# data = dataset[0]
# paper:736389
# author:1134649
# cites_edge_data = data['paper', 'cites', 'paper'].edge_index
# writes_edge = data['author', 'writes', 'paper'].edge_index

path = "./CUB/"
data = HeteroData()
class_A = {}
A_feat = []
B_feat = []
metapath_dict = {}

B = pd.read_excel(os.path.join(path, 'color.xlsx'), sheet_name="df")
class_B = set([j.replace('-', ' ') for index, j in enumerate(B['color'])])
class_B = {color: index for index, color in enumerate(class_B)}

edge = pd.read_excel(os.path.join(path, 'keyword.xlsx'), sheet_name="df")
relation = {i: [] for i in edge['keyword']}
relation2 = {i: [] for i in edge['keyword']}
relation_feat = {i: [] for i in edge['keyword']}

for index, i in enumerate(os.listdir(os.path.join(path, "extract"))):
    table = str.maketrans('', '', digits)
    class_label = i.translate(table).split('.')[-2].replace('_', ' ')
    class_A[class_label] = index

    with open(os.path.join(path, "extract", i), "r") as f:
        txt = f.read()
    for j in txt.split('\n'):
        if len(j.split(',')) == 2:
            src, dst = j.split(',')

            relation[src.strip()].append([class_A[class_label], class_B[dst.strip().replace('-', ' ')]])
            relation2[src.strip()].append([class_B[dst.strip().replace('-', ' ')], class_A[class_label]])


data['A'].x = torch.rand(200,768)
data['B'].x = torch.rand(430,768)
for i in relation:
    data['A', i, 'B'].edge_index = torch.tensor(np.array(relation[i]), dtype=torch.long).t().contiguous()
    data['B', i, 'A'].edge_index = torch.tensor(np.array(relation2[i]), dtype=torch.long).t().contiguous()
    data['A', i, 'B'].edge_attr = torch.rand(data['A', i, 'B'].edge_index.size(1), 768)
    data['B', i, 'A'].edge_attr = torch.rand(data['A', i, 'B'].edge_index.size(1), 768)


print(data)

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
