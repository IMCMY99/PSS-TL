import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch_geometric.nn import MLP
from torch.nn.modules.module import Module

# from args import args
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.mm(input.float().cpu(), self.weight.float().cpu())
        output = torch.spmm(adj.float().cpu(), support.cpu())

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Teacher_S(nn.Module):
    def __init__(self, num_nodes, in_size,  mlp_width,hidden_size, out_size, dropout, device):
        super(Teacher_S, self).__init__()
        self.adj_transform = MLP([3, mlp_width, 2]).to('cuda')
        self.tgc1 = GraphConvolution(in_size, hidden_size)
        self.tgc2 = nn.Linear(hidden_size, 64)
        self.tgc3 = GraphConvolution(64, out_size)
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(num_nodes, in_size, bias=True)
        self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)
        self.bn1 = nn.BatchNorm1d(hidden_size).to(device)
        self.bn2 = nn.BatchNorm1d(out_size).to(device)
        self.device=device

    def forward(self, data):
        indices = torch.nonzero(data.adj.view(-1, 1), as_tuple=True)[
            0].cpu().detach().numpy()  # 把adj变为列数为一行数自动变换的张量中非零元素的索引

        adj_flattened = torch.zeros(data.adj.view(-1, 1).shape[0]).to(self.device)  # 和indices同样大小的0张量

        x, adj = data.x, torch.cat(
            (data.adj.view(-1, 1)[indices].to(self.device), data.xdeg.view(-1, 1)[indices].to(self.device), data.ydeg.view(-1, 1)[indices].to(self.device)), 1)
        adj_mask = self.softmax(self.adj_transform(adj.to(self.device)))[:, 1]  # 计算Social Degree Correction矩阵M
        # 以下是本文的结构方法：重构邻接矩阵
        adj_flattened[indices] = adj_mask.to(self.device)
        adj_mask = adj_flattened.reshape(data.adj.shape[0], data.adj.shape[1])  # 再将adj_mask变为和adj同样形状的张量

        adj = data.adj.to(self.device) * adj_mask

        adj = adj + torch.eye(*adj.shape).to(self.device)
        rowsum = torch.sum(adj, dim=1)

        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim=0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)

        adj = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)

        middle_representations = []

        pe = self.linear(self.pe_feat)
        pe = F.dropout(pe, self.dropout, training=self.training)
        pe = pe.to(self.device)

        h = self.tgc1(pe, adj)
        h = self.bn1(h.to(self.device))
        h=self.tgc2(h)

        middle_representations=h
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.tgc3(h, adj)
        output=self.bn2(h.to(self.device))


        return output, middle_representations