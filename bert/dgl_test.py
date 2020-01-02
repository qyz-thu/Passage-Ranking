'''
This is the implementation of GAT from dgl official website
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegraph


class GATLayer(nn.Module):
    def __init__(self, g, in_feat, out_feat):
        super(GATLayer, self).__init__()
        self.graph = g
        self.W = nn.Linear(in_feat, out_feat, bias=False)
        self.alpha = nn.Linear(2 * out_feat, 1, bias=False)

    def edge_attention(self, edges):
        zz = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        e = self.alpha(zz)
        return {'e': nn.LeakyReLU(e)}

    @staticmethod
    def msg_function(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    @staticmethod
    def reduce_function(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, input):
        z = self.W(input)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.msg_function, self.reduce_function)
        return self.g.ndata.pop('h')


class MultiHeadLayer(nn.Module):
    def __init__(self, g, in_feat, out_feat, head_num, merge='cat'):
        super(MultiHeadLayer, self).__init__()
        self.merge = merge
        self.heads = nn.ModuleList()
        for i in range(head_num):
            self.heads.append(GATLayer(g, in_feat, out_feat))

    def forward(self, input):
        head_outs = [head(input) for head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_feat, hidden_feat, out_feat, head_num):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadLayer(g, in_feat, out_feat, head_num, merge="avg")

    def forward(self, input):
        h = self.layer1(input)
        h = nn.ReLU(h)
        return h


def load_cora_data():
    data = citegraph.load_cora()
    features = torch.Tensor(data.features).float()
    labels = torch.Tensor(data.labels).long()
    mask = torch.Tensor(data.train_mask).byte()
    g = data.graph
    g.remove_edges_from(g.selfloog_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

