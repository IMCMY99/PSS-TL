import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from load_data import load_data
from tqdm import tqdm
from torch_geometric.nn import MLP
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from args import args
import time


## if multi-gpu is needed for training on large social graphs, uncomment the commented codes and run the following command
## CUDA_VISIBLE_DEVICES=0,1,2,3 python src/gcn_decor.py --dataset_name [dataset_name]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = args.device

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


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


def mid_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
         mean: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    R_stu_1 = z1[0].to(device)
    R_fea_1 = self.feat2stu(z2[0].to(device))
    R_str_1 = self.stru2stu(z3[0].to(device))
    fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
    str_stu_1 = self.semi_loss(R_stu_1, R_str_1)
    fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
    str_stu_1 = str_stu_1.mean() if mean else str_stu_1.sum()

    R_stu_2 = z1[1].to(device)
    R_fea_2 = z2[1].to(device)
    R_str_2 = z3[1].to(device)
    fea_stu_2 = self.semi_loss(R_stu_2, R_fea_2)
    str_stu_2 = self.semi_loss(R_stu_2, R_str_2)
    fea_stu_2 = fea_stu_2.mean() if mean else fea_stu_2.sum()
    str_stu_2 = str_stu_2.mean() if mean else str_stu_2.sum()

    loss_mid_fea = fea_stu_1 + fea_stu_2
    loss_mid_str = str_stu_1 + str_stu_2

    return loss_mid_fea, loss_mid_str

class Teacher_F(nn.Module):
    def __init__(self, num_nodes, in_dim,hid_dim, n_classes, num_layers, dropout):
        super(Teacher_F, self).__init__()
        if num_layers == 1:
            hid_dim = n_classes

        self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_dim))).to(args.device)
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)

        self.fm1 = nn.Linear(in_dim, hid_dim, bias=True).to(args.device)
        self.fm2 = nn.Linear(hid_dim, n_classes, bias=True).to(args.device)
        self.dropout = dropout
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        feature=data.x
        feature = torch.where(torch.isnan(feature), self.imp_feat, feature).to(args.device)

        middle_representations=[]
        h = self.fm1(feature)
        middle_representations.append(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        h = self.fm2(h)
        middle_representations.append(h)

        return h,middle_representations
class Teacher_S(nn.Module):
    def __init__(self, num_nodes, in_dim,  mlp_width,hid_dim, n_classes,dropout, device):
        super(Teacher_S, self).__init__()
        self.adj_transform = MLP([3, mlp_width, 2]).to(device)
        self.weight1 = nn.Linear(in_dim, hid_dim, bias=False).to(device)
        self.weight2 = nn.Linear(64, n_classes, bias=False).to(device)
        self.dropout = nn.Dropout()
        self.Dropout=dropout
        self.softmax = nn.Softmax(dim=-1)
        self.linear1 = nn.Linear(num_nodes, in_dim, bias=True)
        self.linear2=nn.Linear(hid_dim,64).to(device)
        self.pe_feat = torch.FloatTensor(torch.eye(num_nodes))
        self.bn1 = nn.BatchNorm1d(hid_dim).to(device)
        self.bn2 = nn.BatchNorm1d(n_classes).to(device)
        self.device=device

    def forward(self, data):
        indices = torch.nonzero(data.adj.view(-1, 1), as_tuple=True)[0].cpu().detach().numpy()  # 把adj变为列数为一行数自动变换的张量中非零元素的索引

        adj_flattened = torch.zeros(data.adj.view(-1, 1).shape[0]).to(data.adj.device)  # 和indices同样大小的0张量

        x, adj = data.x, torch.cat(
            (data.adj.view(-1, 1)[indices], data.xdeg.view(-1, 1)[indices], data.ydeg.view(-1, 1)[indices]), 1)
        adj_mask = self.softmax(self.adj_transform(adj.to(device)))[:, 1]  # 计算Social Degree Correction矩阵M
        # 以下是本文的结构方法：重构邻接矩阵
        adj_flattened[indices] = adj_mask.to(data.adj.device)
        adj_mask = adj_flattened.reshape(data.adj.shape[0], data.adj.shape[1])  # 再将adj_mask变为和adj同样形状的张量

        adj = data.adj * adj_mask

        adj = adj + torch.eye(*adj.shape).to(data.adj.device)
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

        # pe = self.linear1(self.pe_feat)
        # pe = F.dropout(pe, self.Dropout, training=self.training)
        # pe = pe.to(device)

        support = self.weight1(data.x)
        output = torch.mm(adj, support)
        hid = self.dropout(output).to(device)
        hid=self.linear2(hid)
        hid = self.dropout(hid).to(device)
        middle_representations.append(hid)
        support = self.weight2(hid)
        output = torch.mm(adj, support)
        output = self.dropout(output).to(device)
        middle_representations.append(output)



        return output, middle_representations
class GCNDecor(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, n_classes, mlp_width):
        super().__init__()

        self.adj_transform = MLP([3, mlp_width, 2])
        # self.adj_transform = torch.nn.DataParallel(MLP([3, mlp_width, 2]), device_ids=[0, 1, 2, 3])
        self.softmax = nn.Softmax(dim=-1)
        self.weight1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.weight2 = nn.Linear(hid_dim, n_classes, bias=False)
        self.dropout = nn.Dropout()

    def forward(self, data):
        indices = torch.nonzero(data.adj.view(-1, 1), as_tuple=True)[
            0].cpu().detach().numpy()  # 把adj变为列数为一行数自动变换的张量中非零元素的索引

        adj_flattened = torch.zeros(data.adj.view(-1, 1).shape[0]).to(data.adj.device)  # 和indices同样大小的0张量

        x, adj = data.x, torch.cat(
            (data.adj.view(-1, 1)[indices], data.xdeg.view(-1, 1)[indices], data.ydeg.view(-1, 1)[indices]), 1)
        adj_mask = self.softmax(self.adj_transform(adj))[:, 1]  # 计算Social Degree Correction矩阵M
        # 以下是本文的结构方法：重构邻接矩阵
        adj_flattened[indices] = adj_mask
        adj_mask = adj_flattened.reshape(data.adj.shape[0], data.adj.shape[1])  # 再将adj_mask变为和adj同样形状的张量

        adj = data.adj * adj_mask

        adj = adj + torch.eye(*adj.shape).to(data.adj.device)
        rowsum = torch.sum(adj, dim=1)

        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim=0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)

        adj = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)
        middle_representations=[]
        support = self.weight1(data.x)
        output = torch.mm(adj, support)
        hid = self.dropout(output)
        middle_representations.append(hid)
        support = self.weight2(hid)
        output = torch.mm(data.adj, support)
        middle_representations.append(output)


        return output,middle_representations


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def pre_train_teacher_str(model,data, iter):
     t = time.time()
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_str, weight_decay=args.weight_decay_str)
     model.train()
     best_teacher_str_val = 0
     for epoch in tqdm(range(800)):
         optimizer.zero_grad()
         out,_ = model(data)
         criterion = nn.CrossEntropyLoss()
         loss = criterion(out[data.train_mask], data.y_train)
         _,pred = out[data.train_mask].max(dim=-1)
         train_acc = pred.eq(data.y_train).sum().item() / len(data.y_train)
         _,testpred = out[data.test_mask].max(dim=-1)
         test_acc = testpred.eq(data.y_test).sum().item() / len(data.y_test)
         loss.backward()
         optimizer.step()
         loss_val=criterion(out[data.val_mask], data.y_val)
         _,valpred=out[data.val_mask].max(dim=-1)
         val_acc= valpred.eq(data.y_val).sum().item() / len(data.y_val)
         if val_acc > best_teacher_str_val:
             best_teacher_str_val = val_acc
             teacher_str_state = {
                 'state_dict': model.state_dict(),
                 'best_val': val_acc,
                 'best_epoch': epoch + 1,
                 'optimizer': optimizer.state_dict(),  # 保留模型和参数
             }
             # filename1='./.checkpoints/' + args.dataset_name
             # torch.save(teacher_str_state, filename1)
             # print('Successfully saved structure teacher model\n...')
         print('Epoch: {:04d}'.format(epoch + 1),
                'loss_train: {:.4f}'.format(loss.item()),
                'acc_train: {:.4f}'.format(train_acc),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(val_acc),
                'time: {:.4f}s'.format(time.time() - t))
     model.eval()
     out,_ = model(data)
     pred=out.argmax(dim=1)
     y_pred = pred[data.test_mask]
     acc = accuracy_score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
     precision, recall, fscore, _ = score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
                                          average='macro')
     print("{ts} Test set results:".format(ts='teacher_str'),

           "accuracy= {:.4f}".format(acc))
     return teacher_str_state,acc

def pre_train_teacher_fea(model, data, iter):
    t = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_str, weight_decay=args.weight_decay_str)
    model.train()
    best_teacher_fea_val = 0
    for epoch in tqdm(range(400)):
        optimizer.zero_grad()
        out, _ = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out[data.train_mask], data.y_train)
        _, pred = out[data.train_mask].max(dim=-1)
        train_acc = pred.eq(data.y_train).sum().item() / len(data.y_train)
        _, testpred = out[data.test_mask].max(dim=-1)
        test_acc = testpred.eq(data.y_test).sum().item() / len(data.y_test)
        loss.backward()
        optimizer.step()
        loss_val = criterion(out[data.val_mask], data.y_val)
        _, valpred = out[data.val_mask].max(dim=-1)
        val_acc = valpred.eq(data.y_val).sum().item() / len(data.y_val)
        if val_acc > best_teacher_fea_val:
            best_teacher_fea_val = val_acc
            teacher_fea_state = {
                'state_dict': model.state_dict(),
                'best_val': val_acc,
                'best_epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),  # 保留模型和参数
            }
            # filename2 = './.checkpoints/' + args.dataset_name
            # torch.save(teacher_fea_state, filename2)
            # print('Successfully saved fea teacher model\n...')
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(train_acc),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(val_acc),
              'time: {:.4f}s'.format(time.time() - t))
    model.eval()
    out, _ = model(data)
    pred = out.argmax(dim=1)
    y_pred = pred[data.test_mask]
    acc = accuracy_score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    precision, recall, fscore, _ = score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
                                         average='macro')
    print("{ts} Test set results:".format(ts='teacher_fea'),

          "accuracy= {:.4f}".format(acc))
    return teacher_fea_state,acc

def train(model_s,model_t,model_f,data, iter):

    optimizer = torch.optim.Adam(model_s.parameters(), lr=5e-4, weight_decay=1e-6)
    model_s.train()
    str_model=model_t
    fea_model=model_f

    for epoch in tqdm(range(800)):
        optimizer.zero_grad()
        out,mid_stu = model_s(data)
        soft_target_str, middle_emb_str = str_model(data)
        soft_target_fea, middle_emb_fea = fea_model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out[data.train_mask], data.y_train)
        kd_loss_s=KD_loss(out.to(args.device), soft_target_str)
        kd_loss_f=KD_loss(out.to(args.device), soft_target_fea)
        loss_mid_fea,loss_mid_str=mid_loss(mid_stu,middle_emb_fea,middle_emb_str)
        _, pred = out[data.train_mask].max(dim=-1)

        loss_train=loss+(1-args.lambd)*(kd_loss_f+loss_mid_fea)+args.lambd*(kd_loss_s+loss_mid_str)
        train_acc = pred.eq(data.y_train).sum().item() / len(data.y_train)
        _, testpred = out[data.test_mask].max(dim=-1)
        test_acc = testpred.eq(data.y_test).sum().item() / len(data.y_test)
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(train_acc),
              )
    model_s.eval()
    out, _ = model_s(data)
    pred = out.argmax(dim=1)
    y_pred = pred[data.test_mask]
    acc = accuracy_score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    precision, recall, fscore, _ = score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
                                         average='macro')

    print(['Global Test Accuracy:{:.4f}'.format(acc),
           'Precision:{:.4f}'.format(precision),
           'Recall:{:.4f}'.format(recall),
           'F1:{:.4f}'.format(fscore)])
    print("-----------------End of Iter {:03d}-----------------".format(iter))

    return acc, precision, recall, fscore

def save_checkpoint(state,filename='./.checkpoints/' + args.dataset, ts='teacher_fea'):
        print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            state=state
            torch.save(state, filename)
            print('Successfully saved feature teacher model\n...')
        elif ts == 'teacher_str':
            state=state
            torch.save(state, filename)
            print('Successfully saved structure teacher model\n...')


def load_checkpoint(model,filename='./.checkpoints/' + args.dataset, ts='teacher_fea'):
        print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            load_state = torch.load(filename)
            model.load_state_dict(load_state['state_dict'])
            # optimizerTeacherFea.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded feature teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'])
        elif ts == 'teacher_str':
            load_state = torch.load(filename)
            model.load_state_dict(load_state['state_dict'])
            # optimizerTeacherStr.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded structure teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'])

        return model




KD_loss=SoftTarget(args.Ts)
datasetname = args.dataset_name
u_thres = args.u_thres
iterations = args.iters
mlp_width = 16 if datasetname == 'politifact' else 8
data = load_data(datasetname,args.device,args.rate, u_thres).to(device)
stu_model= GCNDecor(768, 64, 2, mlp_width).to(device)
str_t=Teacher_S( num_nodes=497, in_dim=768,  mlp_width=mlp_width,hid_dim=128, n_classes=2,dropout=args.dropout_str, device=args.device)
fea_t=Teacher_F(num_nodes=497,in_dim=768,hid_dim=args.hidden_fea,n_classes=2,num_layers=args.num_fea_layers,dropout=args.dropout_fea)
test_accs = []
prec_all, rec_all, f1_all = [], [], []
ACC_str,ACC_fea=[],[]
for iter in range(iterations):
    set_seed(iter)
    # teacher_str_state,acc_str=pre_train_teacher_str(str_t,data,iter)
    # save_checkpoint(teacher_str_state,ts='teacher_str')
    teacher_fea_state,acc_fea=pre_train_teacher_fea(fea_t,data,iter)
    save_checkpoint(teacher_fea_state,ts='teacher_fea')
    # str_t=load_checkpoint(str_t,ts='teacher_str')
    fea_t=load_checkpoint(fea_t,ts='teacher_fea')
    print('\n--------------\n')
    ACC_fea.append(acc_fea)
    # ACC_str.append(acc_str)

    acc, prec, recall, f1 = train(stu_model,str_t,fea_t,data, iter)
    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}|fea_acc:{:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations, sum(ACC_fea)/iterations))

with open('logs/log_' + datasetname + '_train80pct' + '_' + args.model_name + '_user_t' + str(u_thres) + '.iter' + str(
        iterations), 'a+') as f:
    f.write('All Acc.s:{}\n'.format(test_accs))
    f.write('All Prec.s:{}\n'.format(prec_all))
    f.write('All Rec.s:{}\n'.format(rec_all))
    f.write('All F1.s:{}\n'.format(f1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write(
        'Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) / iterations, sum(rec_all) / iterations,
                                                                sum(f1_all) / iterations))
    f.write('\n')

