import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from GBT import GBT
from local_model import local_model
from KNRM import KNRM
from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder

from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder
import dgl.function as fn
from torch.autograd import Variable
import numpy as np


class inference_model(nn.Module):
    def __init__(self, bert_model, args, device, get_full_output=False):
        super(inference_model, self).__init__()
        self.device = device
        self.bert_hidden_dim = args.bert_hidden_dim
        self.batch_size = args.train_batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        self.get_full_output = get_full_output
        self.kernal_num = args.kernal_num
        self.proj_hidden = nn.Linear(self.bert_hidden_dim, args.entity_dim, bias=False)
        self.GAT = GBT(args.entity_dim, args.GAT_hidden_dim, args.entity_dim, args.head_num, device).cuda()
        self.local_model = local_model(args.passage_len, args.query_len)
        self.weight = nn.Linear(2, 1, bias=False)
        self.KNRM = KNRM(args.train_batch_size, args.kernal_num)

    def forward(self, inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor, msk_tensor, seg_tensor, inp_ent, inp_matrix):
        query, _q = self.pred_model(inp_tensor_qry, msk_tensor_qry, seg_tensor_qry)
        passage, _p = self.pred_model(inp_tensor, msk_tensor, seg_tensor)

        psg_wrd = self.proj_hidden(passage)
        qry_wrd = self.proj_hidden(query)
        qry_ent = self.GAT(inp_ent[0])
        psg_ent = self.GAT(inp_ent[1])
        score1 = self.KNRM(qry_wrd, psg_wrd, qry_ent, psg_ent)
        score1 = score1.unsqueeze(dim=1)

        score2 = self.local_model(inp_matrix)
        scores = torch.cat([score1, score2], dim=1)
        score = self.weight(scores)

        return score.squeeze()
