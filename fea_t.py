
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GCN2Conv
class Teacher_F(nn.Module):
    def __init__(self, num_nodes, in_dim, hid_dim, n_classes, num_layers, dropout,device,
                 shared_weights=True):
        super(Teacher_F, self).__init__()
        if num_layers == 1:
            hid_dim = n_classes

        # self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_dim))).to(args.device)
        self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_dim)))
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)

        # self.fm1 = nn.Linear(in_dim, hid_dim, bias=True).to(args.device)
        self.fm1 = nn.Linear(in_dim, hid_dim, bias=True)

        # self.linear=nn.Linear(hid_dim,64).to(args.device)


        # self.fm2 = nn.Linear(hid_dim, n_classes, bias=True).to(args.device)
        self.fm2 = nn.Linear(hid_dim, n_classes, bias=True)
        # self.convs = torch.nn.ModuleList()
        # for layer in range(num_layers):
        #     self.convs.append(
        #         GCN2Conv(hid_dim, alpha, theta, layer + 1,
        #                  shared_weights, normalize=False))
        self.dropout = dropout
        self.weights_init()
        self.device=device
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        feature = data.x.to(self.device)
        # feature = data.x
        imp = torch.zeros([data.x.shape[0], data.x.shape[1]]).cpu()
        #
        # feature = torch.where(torch.isnan(feature.to(self.device)), self.imp_feat.to(self.device),
        #                       feature.to(self.device)).to(self.device)#参数化特征矩阵
        feature = torch.where(torch.isnan(feature.to(self.device)), imp.to(self.device),
                                                    feature.to(self.device)).to(self.device)#0插补特征矩阵
        # feature = torch.where(torch.isnan(feature), self.imp_feat,
        #                       feature.to(args.device))
        # print(feature)
        h = self.fm1(feature)

        h = F.dropout(h, self.dropout, training=self.training)


        fea_rep = F.dropout(h, self.dropout, training=self.training)
        # fea_rep = self.linear(h)

        h = self.fm2(fea_rep)

        return h, fea_rep