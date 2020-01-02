import torch
import torch.nn as nn


class local_model(nn.Module):
    def __init__(self, passage_len, query_len):
        super(local_model, self).__init__()
        self.conv = nn.Conv1d(in_channels=passage_len, out_channels=100, kernel_size=1, stride=1, bias=False)
        self.linear1 = nn.Linear(query_len, 1, bias=True)
        self.linear2 = nn.Linear(100, 1, bias=True)

    def forward(self, input):
        conv_out = torch.tanh(self.conv(input))
        l1_out = torch.tanh(self.linear1(conv_out)).squeeze()
        l2_out = torch.tanh(self.linear2(l1_out))

        return l2_out
