from stu_new import GCNDECOR
from stru_t import Teacher_S
from st import SoftTarget
from fea_t import Teacher_F
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
# from sklearn.metrics import accuracy_score

# from run import args
from load_data import load_data, accuracy_score,f1_score,precision_score,recall_score



class Train:

    def __init__(self, args, repeat, acc_fea, acc_str, acc_stu,f1_fea, f1_str, f1_stu, pre_fea, pre_str, pre_stu, rec_fea,rec_str,rec_stu):
        self.args = args
        self.repeat = repeat
        self.best_teacher_fea_val, self.best_teacher_str_val, self.best_student_val = 0, 0, 0
        self.teacher_fea_state, self.teacher_str_state, self.student_state = None, None, None

        self.acc_list_fea = acc_fea
        self.acc_list_str = acc_str
        self.acc_list = acc_stu

        self.f1_list_fea = f1_fea
        self.f1_list_str =f1_str
        self.f1_list = f1_stu

        self.pre_list_fea = pre_fea
        self.pre_list_str = pre_str
        self.pre_list = pre_stu

        self.rec_list_fea = rec_fea
        self.rec_list_str = rec_str
        self.rec_list = rec_stu

        self.device=args.device
    # Model Initialization

        self.features,  self.adj, self.xdeg, self.ydeg, self.train_idx,self.val_idx, self.test_idx, self.labels ,_,_,_= load_data(
            args.dataset, args.device, args.rate,args.drop_edge, u_thres=3)
        self.data=load_data(args.dataset, args.device, args.rate,args.drop_edge, u_thres=3)
        self.labels=self.labels[1]

        # mlp_width = 16 if args.datasetname == 'politifact' else 8


        self.str_model = Teacher_S(num_nodes=self.features[1].shape[0],
                                   in_size=self.features[1].shape[1],
                                   mlp_width=8,
                                   hidden_size=self.args.hidden_str,
                                   out_size=2,
                                   dropout=self.args.dropout_str,
                                   device=args.device)
        self.str_model.to(args.device)

        self.fea_model= Teacher_F(num_nodes=self.features[1].shape[0],in_dim=768,hid_dim=args.hidden_fea,n_classes=2,num_layers=args.num_fea_layers,dropout=args.dropout_fea,device=args.device)

        self.fea_model.to(args.device)

        #

        self.stu_model = GCNDECOR(in_dim=self.features[1].shape[1],
                                 hid_dim=self.args.hidden_stu,
                                 n_classes=2,
                                 mlp_width=16,
                                      device=args.device,model_name=args.model_name )

        # self.stu_model=Pre_head(in_dim=128,hid_dim=256,n_classes=2)
        self.stu_model.to(args.device)

        self.criterionTeacherStr = nn.CrossEntropyLoss()  # 结构教师训练损失
        self.criterionTeacherFea = nn.CrossEntropyLoss()  # 特征教师训练损失
        self.criterionStudent = nn.CrossEntropyLoss()  # 学生模型损失
        self.criterionStudentKD = SoftTarget(args.Ts)  # 知识蒸馏

        # self.log_var_a = torch.zeros((1,), requires_grad=True)
        # self.log_var_b = torch.zeros((1,), requires_grad=True)
        # self.awl=AutomaticWeightedLoss(4)
        self.optimizerTeacherStr = optim.Adam(self.str_model.parameters(), lr=self.args.lr_str,
                                              weight_decay=self.args.weight_decay_str)
        self.optimizerTeacherFea = optim.Adam(self.fea_model.parameters(), lr=self.args.lr_fea,
                                              weight_decay=self.args.weight_decay_fea)
        self.optimizerStudent = torch.optim.Adam(self.stu_model.parameters(), lr=self.args.lr_stu,
                                           weight_decay=self.args.weight_decay_stu)

    # def load_data(self):
    #     self.features,self.tadj, self.adj, self.xdeg, self.ydeg, self.train_idx[1], self.test_idx[1], self.labels= load_data(
    #         self.args.dataset, self.device, self.args.rate,self.args.drop_edge, u_thres = 3)
    #     self.data=load_data(self.args.dataset, self.device, self.args.rate,self.args.drop_edge, u_thres = 3)

        print('Data load init finish')
        print('Num nodes: {}  | Num classes: {}'.format(
            self.adj[1].shape[0],  2))

    def pre_train_teacher_str(self, epoch):
            t = time.time()
            self.str_model.train()
            self.optimizerTeacherStr.zero_grad()
            output,_ = self.str_model(self.data)
            loss_train = self.criterionTeacherStr(output[self.train_idx[1].cpu()].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))
            acc_train = accuracy_score(output[self.train_idx[1].cpu()].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))
            loss_train.backward()
            self.optimizerTeacherStr.step()
            if not self.args.fastmode:
                self.str_model.eval()
                output, _ = self.str_model(self.data)

            loss_val = self.criterionTeacherStr(output[self.val_idx[1].cpu()].to(self.device),
                                                self.labels[self.val_idx[1].cpu()].to(self.device))
            acc_val = accuracy_score(output[self.val_idx[1].cpu()].to(self.device),
                               self.labels[self.val_idx[1].cpu()].to(self.device))

            if acc_val > self.best_teacher_str_val:
                self.best_teacher_str_val = acc_val
                self.teacher_str_state = {
                    'state_dict': self.str_model.state_dict(),
                    'best_val': acc_val,
                    'best_epoch': epoch + 1,
                    'optimizer': self.optimizerTeacherStr.state_dict(),  # 保留模型和参数
                }
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def pre_train_teacher_fea(self, epoch):
        t = time.time()
        self.fea_model.train()
        self.optimizerTeacherFea.zero_grad()

        output, _ = self.fea_model(self.data)
        loss_train = self.criterionTeacherFea(output[self.train_idx[1].cpu()].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))
        acc_train = accuracy_score(output[self.train_idx[1].cpu()].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))
        loss_train.backward()
        self.optimizerTeacherFea.step()

        if not self.args.fastmode:
            self.fea_model.eval()
            output, _ = self.fea_model(self.data)

        loss_val = self.criterionTeacherFea(output[self.val_idx[1].cpu()].to(self.device),
                                            self.labels[self.val_idx[1].cpu()].to(self.device))
        acc_val = accuracy_score(output[self.val_idx[1].cpu()].to(self.device),
                                 self.labels[self.val_idx[1].cpu()].to(self.device))

        if acc_val > self.best_teacher_fea_val:
            self.best_teacher_fea_val = acc_val
            self.teacher_fea_state = {
                'state_dict': self.fea_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch + 1,
                'optimizer': self.optimizerTeacherFea.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))



    def train_student(self, epoch):
            t = time.time()
            self.stu_model.train()

            self.optimizerStudent.zero_grad()

            output,middle_emb_stu= self.stu_model(self.data)

            soft_target_str, middle_emb_str = self.str_model(self.data)
            soft_target_fea, middle_emb_fea = self.fea_model(self.data)



            contrast_loss_fea,contrast_loss_str = self.stu_model.loss(middle_emb_stu[self.train_idx[1]].to(self.device), middle_emb_fea[self.train_idx[1]].to(self.device),middle_emb_str[self.train_idx[1]].to(self.device))



            # 使用任务权重加权损失
            # loss_train = sum(task_weights[i] * losses[i] for i in range(len(losses)))
            # print(self.train_idx[1])
            # print(self.train_idx[1].dtype)
            _, pred = output[self.train_idx[1]].max(dim=-1)
            # print(self.labels)
            # print(self.labels.dtype)
            # labels=self.labels[self.train_idx[1].cpu()]
            # loss_train=self.criterionStudent(output[self.train_idx[1].cpu()].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))
            loss_train = self.criterionStudent(output[self.train_idx[1]].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device)) +self.args.lambd*(self.criterionStudentKD(output[self.train_idx[1]].to(self.device), soft_target_str[self.train_idx[1]].to(self.device)))+(1-self.args.lambd)*(self.criterionStudentKD(output[self.train_idx[1]].to(self.device), soft_target_fea[self.train_idx[1]].to(self.device)))+self.args.beta*contrast_loss_str+(1-self.args.beta)*contrast_loss_fea
            # loss_train = (1-self.args.lambd_s-self.args.lambd_f)*self.criterionStudent(output[self.train_idx[1]].to(self.device), self.labels_train[1].to(self.device)) +self.args.lambd_f*(self.criterionStudentKD(output.to(self.device), soft_target_fea.to(self.device)))


            acc_train = pred.eq(self.labels[self.train_idx[1].cpu()].to(self.device)).sum().item() / len(self.labels[self.train_idx[1].cpu()].to(self.device))


            loss_train.backward()
            self.optimizerStudent.step()
            # if not self.args.fastmode:
            #     self.stu_model.eval()
            #     output, _ = self.stu_model(self.data)

            loss_val = self.criterionTeacherStr(output[self.val_idx[1].cpu()].to(self.device),
                                                self.labels[self.val_idx[1].cpu()].to(self.device))
            acc_val = accuracy_score(output[self.val_idx[1].cpu()].to(self.device),
                               self.labels[self.val_idx[1].cpu()].to(self.device))

            if acc_val > self.best_student_val:
                self.best_student_val = acc_val
                self.student_state  = {
                    'state_dict': self.stu_model.state_dict(),
                    'best_val': acc_val,
                    'best_epoch': epoch + 1,
                    'optimizer': self.optimizerStudent.state_dict(),  # 保留模型和参数
                }


            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  # 'constrast_str: {:.4f}'.format(contrast_str),
                  'criterionStudent: {:.4f}'.format(self.criterionStudent(output[self.train_idx[1]].to(self.device), self.labels[self.train_idx[1].cpu()].to(self.device))),
                  # 'criterionStudentKD_Str: {:.4f}'.format(self.criterionStudentKD(output.to(self.device), soft_target_str.to(self.device))),
                  # 'criterionStudentKD_fea: {:.4f}'.format(self.criterionStudentKD(output.to(self.device), soft_target_fea.to(self.device))),
                  # 'middle_loss: {:.4f}'.format(contrast_loss),
                  'time: {:.4f}s'.format(time.time() - t))


    def test(self, ts='student'):

            if ts == 'teacher_str':
                model = self.str_model
                criterion = self.criterionTeacherStr
                model.eval()
                output,_ = model(self.data)
                loss_test = criterion(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                acc_test = accuracy_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                f1_test = f1_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                pre_test = precision_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                rec_test = recall_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                print("{ts} Test set results:".format(ts=ts),
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()),
                      "F1= {:.4f}".format(f1_test),
                      "Precision= {:.4f}".format(pre_test),
                      "Recall= {:.4f}".format(rec_test),
                      )
                self.acc_list_str.append(round(acc_test.item(), 4))
                self.f1_list_str.append(round(f1_test, 4))
                self.pre_list_str.append(round(pre_test, 4))
                self.rec_list_str.append(round(rec_test, 4))


            if ts == 'teacher_fea':
                model = self.fea_model
                criterion = self.criterionTeacherFea
                model.eval()
                output,_ = model(self.data)
                loss_test = criterion(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                acc_test = accuracy_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                f1_test = f1_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                pre_test = precision_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                rec_test = recall_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                print("{ts} Test set results:".format(ts=ts),
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()),
                      "F1= {:.4f}".format(f1_test),
                      "Precision= {:.4f}".format(pre_test),
                      "Recall= {:.4f}".format(rec_test),
                      )
                self.acc_list_fea.append(round(acc_test.item(), 4))
                self.f1_list_fea.append(round(f1_test, 4))
                self.pre_list_fea.append(round(pre_test, 4))
                self.rec_list_fea.append(round(rec_test, 4))

            if ts == 'student':
                model = self.stu_model
                criterion = self.criterionStudent
                model.eval()
                # soft_target_str, middle_emb_str = self.str_model(self.data)
                # soft_target_fea, middle_emb_fea = self.fea_model(self.data)

                # output = model(torch.cat((middle_emb_fea, middle_emb_str), dim=1))
                output, _ = model(self.data)

                loss_test = criterion(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                _, testpred = output[self.test_idx[1]].max(dim=-1)
                acc_test = accuracy_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                f1_test = f1_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                pre_test = precision_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                rec_test = recall_score(output[self.test_idx[1]].to(self.device), self.labels[self.test_idx[1].cpu()].to(self.device))
                print("{ts} Test set results:".format(ts=ts),
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()),
                      "F1= {:.4f}".format(f1_test),
                      "Precision= {:.4f}".format(pre_test),
                      "Recall= {:.4f}".format(rec_test),
                      )
                self.acc_list.append(round(acc_test.item(), 4))
                self.f1_list.append(round(f1_test, 4))
                self.pre_list.append(round(pre_test, 4))
                self.rec_list.append(round(rec_test, 4))




    def save_checkpoint(self,  ts='teacher_fea'):
        print('Save {ts} model...'.format(ts=ts))
        filename = './.checkpoints/' + self.args.dataset
        filename += '_{ts}'.format(ts=ts)


        if ts == 'teacher_str':
            torch.save(self.teacher_str_state, filename)
            print('Successfully saved structure teacher model\n...')
        if ts == 'teacher_fea':
            torch.save(self.teacher_fea_state, filename)
            print('Successfully saved feature teacher model\n...')
        elif ts == 'student':
            torch.save(self.student_state, filename)
            print('Successfully saved student model\n...')


    def load_checkpoint(self,  ts='teacher_fea'):
        print('Load {ts} model...'.format(ts=ts))
        filename = './.checkpoints/' + self.args.dataset
        filename += '_{ts}'.format(ts=ts)

        if ts == 'teacher_str':
            load_state = torch.load(filename)
            self.str_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherStr.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded structure teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        if ts == 'teacher_fea':
            load_state = torch.load(filename)
            self.fea_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherFea.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded feature teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        elif ts == 'student':
            load_state = torch.load(filename)
            self.stu_model.load_state_dict(load_state['state_dict'])
            self.optimizerStudent.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded student model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())