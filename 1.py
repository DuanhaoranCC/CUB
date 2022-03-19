from transformers import AutoTokenizer, AutoModel
from string import digits
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero

'''
HeteroData(
  A={ x=[200, 768] },
  B={ x=[450, 768] },
  (A, relation, B)={
    edge_index=[2, 3189],
    edge_attr=[3189, 768]
  }
)
'''
# example:
# dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
# data = dataset[0]
# paper:736389
# author:1134649
# cites_edge_data = data['paper', 'cites', 'paper'].edge_index
# writes_edge = data['author', 'writes', 'paper'].edge_index

path = "F:/GRAND/CUB/"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
data = HeteroData()
# graph = Graph("http://localhost:7474", auth=("neo4j", "test"))
class_A = {}
A_feat = []
B_feat = []
metapath_dict = {}

B = pd.read_excel(os.path.join(path, 'color.xlsx'), sheet_name="df")
class_B = set([j.replace('-', ' ') for index, j in enumerate(B['color'])])
# node_B = {name.replace('-', ' '): Node(label="B", name=name.replace('-', ' ')) for name in class_B}
class_B = {color: index for index, color in enumerate(class_B)}

# [graph.create(i) for i in node_B.values()]  # 430
# node_A = {}

edge = pd.read_excel(os.path.join(path, 'keyword.xlsx'), sheet_name="df")
relation = {i: [] for i in edge['keyword']}
relation2 = {i: [] for i in edge['keyword']}
relation_feat = {i: [] for i in edge['keyword']}

for index, i in enumerate(os.listdir(os.path.join(path, "extract"))):
    table = str.maketrans('', '', digits)
    class_label = i.translate(table).split('.')[-2].replace('_', ' ')
    class_A[class_label] = index
    # node_A[index] = Node(label="A", name=class_label)
    # graph.create(node_A[index])

    middle = tokenizer(class_label, return_tensors="pt")
    A_feat.append(model(**middle)[1].detach().numpy())

    with open(os.path.join(path, "extract", i), "r") as f:
        txt = f.read()
    for j in txt.split('\n'):
        if len(j.split(',')) == 2:
            src, dst = j.split(',')
            # rela = Relationship(node_A[index], src.strip(), node_B[dst.strip().replace('-', ' ')])
            # graph.create(rela)
            relation[src.strip()].append([class_A[class_label], class_B[dst.strip().replace('-', ' ')]])
            relation2[src.strip()].append([class_B[dst.strip().replace('-', ' ')], class_A[class_label]])
            middle = tokenizer(src.strip(), return_tensors="pt")
            relation_feat[src.strip()].append(model(**middle)[1].detach().numpy())

for i in class_B:
    middle = tokenizer(i, return_tensors='pt')
    B_feat.append(model(**middle)[1].detach().numpy())

data['A'].x = torch.FloatTensor(np.array(A_feat)).squeeze(1)
data['B'].x = torch.FloatTensor(np.array((B_feat))).squeeze(1)
for i in relation:
    data['A', i, 'B'].edge_index = torch.tensor(np.array(relation[i]), dtype=torch.long).t().contiguous()
    data['B', i, 'A'].edge_index = torch.tensor(np.array(relation2[i]), dtype=torch.long).t().contiguous()
    data['A', i, 'B'].edge_attr = torch.FloatTensor(np.array(relation_feat[i])).squeeze(1)
    data['B', i, 'A'].edge_attr = torch.FloatTensor(np.array(relation_feat[i])).squeeze(1)
    # metapath_dict[('A', i, 'B')] = None


print(data)
#
# schema_dict = {
#     ('B', 'A'): None
# }
#
# data['metapath_dict'] = metapath_dict
# data['schema_dict'] = schema_dict
# data['main_node'] = 'A'
# data['use_nodes'] = ('B', 'A')

# torch.save(data, 'data.pt')
data = torch.load('data.pt')
print(data.x_dict.keys())
print(data.edge_index_dict.keys())
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
