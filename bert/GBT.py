'''
This is the implementation of GAT from dgl official website
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from dgl import DGLGraph
from dgl.data import citation_graph as citegraph


class GATLayer(nn.Module):  # GAT层相关
    def __init__(self, g, in_feat, out_feat):
        super(GATLayer, self).__init__()
        self.g = g  # 第二个参数是graph相关的:g
        self.W = nn.Linear(in_feat, out_feat, bias=False)  # Linear层的努力
        self.alpha = nn.Linear(2 * out_feat, 1, bias=False)  # alpha参数

    def edge_attention(self, edges):  # 边注意力
        zz = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  #
        e = self.alpha(zz)  # alpha结构
        return {'e': F.leaky_relu(e)}  #

    def msg_function(self, edges):  #
        return {'z': edges.src['z'], 'e': edges.data['e']}  #

    def reduce_function(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, input):  #
        # print(self.W.weight)
        z = self.W(input)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.msg_function, self.reduce_function)
        return self.g.ndata.pop('h')


class MultiHeadLayer(nn.Module):
    def __init__(self, g, in_feat, out_feat, head_num, merge='cat'):  # 建了一个多层的头
        super(MultiHeadLayer, self).__init__()  # 这一切主要还
        self.merge = merge
        self.heads = nn.ModuleList()
        for i in range(head_num):
            self.heads.append(GATLayer(g, in_feat, out_feat))

    def forward(self, input):
        head_outs = [head(input) for head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)


class GAT(nn.Module):  # 多层的东西
    def __init__(self, g, in_feat, hidden_feat, out_feat, head_num):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadLayer(g, in_feat, out_feat, head_num, merge="avg")

    def forward(self, input):
        h = self.layer1(input)
        h = F.relu(h)
        return h


class GBT(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, head_num, device):
        super(GBT, self).__init__()
        self.device = device
        self.GAT = GAT(Get_DGL(), in_feat, hidden_feat, out_feat, head_num)
        features = np.memmap('/home/student/raw_data/entity/entity2vec4.bin', dtype='float32', mode='r')
        self.features = torch.Tensor(features.reshape(-1, 50)).cuda()   # everywhere cuda!

    def forward(self, input):
        lower_layer = self.GAT(self.features)
        # lower_layer = self.features
        bs = input.shape[0]
        len = input.shape[1]
        forward = torch.zeros(bs, len, 50)
        for i, query in enumerate(input):
            for j, rank in enumerate(query):
                if rank != 0:
                    tensor = lower_layer[rank-1]
                    forward[i][j] = tensor
        forward = forward.cuda()
        return forward


def Get_DGL():
    key1 = []
    key2 = []
    maximum = 0
    with open('/home/student/raw_data/entity/triple2id4.txt', 'r') as ft:
        readlines = ft.readlines()
        for line in readlines:
            lines = line.split('\t')
            nod1 = int(lines[0])
            nod2 = int(lines[2])
            if(nod1 > maximum):
                maximum = nod1
            if(nod2 > maximum):
                maximum = nod2
            key1.append(nod1)
            key2.append(nod2)
        ft.close()
    #print(len(key1))
    c = DGLGraph()
    c.add_nodes(maximum+1)
    c.add_edges(key1,key2)
    return c
