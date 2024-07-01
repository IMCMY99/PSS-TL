from stru_t import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP,GCNConv, SAGEConv, GATConv
import torch.nn.init as init



import numpy as np




def row_normalize(x):
    """
    对二维张量的每行进行归一化。
    参数：
    x -- 一个二维张量
    返回：
    归一化后的二维张量
    """
    # 计算每行的和
    row_sums = torch.sum(x, dim=1, keepdim=True)
    # 防止除以零
    row_sums[row_sums == 0] = 1
    # 每行元素除以该行的和
    x_normalized = x / row_sums
    return x_normalized






class GCNDECOR(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, n_classes, mlp_width,device,model_name):
        super().__init__()

        self.adj_transform = MLP([3, mlp_width, 2])
        # self.adj_transform = torch.nn.DataParallel(MLP([3, mlp_width, 2]), device_ids=[0, 1, 2, 3])
        self.softmax = nn.Softmax(dim = -1)
        self.weight1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.weight0 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.weight2 = nn.Linear(hid_dim, n_classes, bias=False)
        self.dropout = nn.Dropout()
        if model_name == 'gcn':
            self.conv1 = GCNConv(in_dim, in_dim)
            self.conv2 = GCNConv(in_dim, in_dim)
        elif model_name == 'sage':
            self.conv1 = SAGEConv(in_dim, in_dim)
            self.conv2 = SAGEConv(in_dim, in_dim)
        elif model_name == 'gat':
            self.conv1 = GATConv(in_dim, in_dim)
            self.conv2 = GATConv(in_dim, in_dim)




        self.feat2stu = torch.nn.Linear(512,64)
        self.stru2stu = torch.nn.Linear(64,64)
        self.tau = 0.5
        self.device=device



    def forward(self, data):

        adj=data.adj.to(self.device)
        adj=row_normalize(adj)
        # adj=torch.eye(data.x.shape[0]).to(self.device)


        #局部传播树
        ADJ=data.A.to(self.device)

        news_list=data.news_list
        #将特征被掩码的部分替换为0
        nodes=data.nodes_features.to(self.device)
        imp = torch.zeros([nodes.shape[0], nodes.shape[1]]).cpu()
        # noise_mean = 0.0  # 噪声的均值
        # noise_std = 1.0  # 噪声的 standard deviation
        # 首先，创建一个与data.x形状相同的张量，并用噪声填充
        # 这里使用高斯噪声
        # imp = torch.randn([nodes.shape[0], nodes.shape[1]]).cpu() * noise_std + noise_mean
        nodes = torch.where(torch.isnan(nodes).cpu(), imp, nodes.cpu()).to(self.device)

        #得到新闻的局部传播特征
        nodes=F.relu(self.conv1(nodes,ADJ))
        x_loc=F.relu(self.conv2(nodes,ADJ))[news_list]





        #使用新闻的局部传播特征在全局图上进行传播

        support = self.weight1(x_loc)
        output = torch.mm(adj, support)
        hid1 = self.dropout(output)
        hid1 = self.weight0(hid1)



        #取得中间的表示特征用于中间对齐
        middle_representations = hid1

        hid2 = torch.mm(adj, hid1)

        output = F.relu(self.weight2(hid2))


        # support = self.weight1(x)
        # output = torch.mm(adj, support)
        # hid = self.dropout(output)
        # middle_representations=hid
        # support = self.weight2(hid)
        # output = torch.mm(adj, support)
        # output = torch.mm(adj, x)
        # output = self.resid_weight * x + output
        # hid = self.nn1(output)
        # hid = self.dropout(hid)
        # middle_representations = hid
        # output = torch.mm(adj, hid)
        # output = self.resid_weight * hid + output
        # output = self.nn2(output)


        return output,middle_representations
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
             mean: bool = True):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=self.device
        R_stu_1 = z1.to(device)
        R_fea_1 = self.feat2stu(z2.to(device))
        R_str_1 = self.stru2stu(z3.to(device))
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        str_stu_1 = self.semi_loss(R_stu_1, R_str_1)
        fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
        str_stu_1 = str_stu_1.mean() if mean else str_stu_1.sum()



        return  fea_stu_1,str_stu_1