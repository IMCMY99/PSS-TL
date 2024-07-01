import numpy as np
import torch
import time
import os
from train_new import Train
# from args import args
from load_data import setup_seed
import time
import torch
import argparse

if __name__ == '__main__':
    # mask_rate_list=[0,0.01,0.03,0.05,0.07,0.09]
    # drop_edge_list=[0,0.01,0.1,0.3,0.5,0.7]
    # args.rate=

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', default=0, type=float, help='masking rate')
    parser.add_argument('--drop_edge', default=0, type=float, help='drop edge rate')
    # parser.add_argument('--lambd_contrast', default=10, type=float, help='contrast_loss weight')
    # Feature Teacher
    parser.add_argument('--num_fea_layers', type=int, default=2, help='Number pf layers for Feature Teacher')
    parser.add_argument('--hidden_fea', type=int, default=512, help='Number of hidden units.')
    # parser.add_argument('--lambd_f', type=float, default=0.1, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout_fea', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--epoch_fea', type=int, default=200, help='Number of epochs for Feature Teacher')
    parser.add_argument('--lr_fea', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay_fea', type=float, default=0.0005, help='loss para for fea_t')
    # parser.add_argument('--alpha', type=float, default=0.3, help='GCNII的初始残差连接权重')
    # parser.add_argument('--theta', type=float, default=0.5, help='GCNII的跳接连接权重')
    # Structure Teacher
    parser.add_argument('--hidden_str', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout_str', type=float, default=0.4, help='Dropout rate.')
    parser.add_argument('--epoch_str', type=int, default=400, help='Number of epochs for Structure Teacher')
    # parser.add_argument('--alpha_ppr', default=0.15, type=float, help='alpha_ppr')
    parser.add_argument('--epsilon', default=1e-4, type=float, help='epsilon')
    # parser.add_argument('--topk', type=int, default=3, help='topk')
    parser.add_argument('--lr_str', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay_str', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
    # parser.add_argument('--lambd_s', type=float, default=0.4, help='loss para for stru_t')

    # Student (GCN)
    parser.add_argument('--epoch_stu', type=int, default=200, help='Max number of epochs for gcn. Default is 400.')

    parser.add_argument('--num_gcn_layers', type=int, default=2, help='Number pf layers for gcn')
    parser.add_argument('--hidden_stu', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout_stu', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--Ts', type=float, default=1, help='temperature for ST')
    parser.add_argument('--lambd', type=float, default=0.1, help='trade-off parameter for kd loss')
    parser.add_argument('--beta', type=float, default=0.8, help='trade-off parameter for contrast loss')

    parser.add_argument('--lr_stu', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay_stu', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')

    # parser.add_argument('--beta', type=float, default=0, help='control contrast_loss.')


    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--repeat', type=int, default=20, help='repeat.')
    parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')
    # parser.add_argument('--dataset_name', default='weibo', type=str)
    parser.add_argument('--model_name', default='gcn', type=str)
    parser.add_argument('--u_thres', default=3, type=int)
    parser.add_argument('--iters', default=5, type=int)
    parser.add_argument('--dataset', default='gossipcop', type=str)
    parser.add_argument('--devicename', default='cuda:0', type=str)
    parser.add_argument('--name', default='test', type=str)

    args = parser.parse_args()

    args.device = torch.device(args.devicename if args.no_cuda and torch.cuda.is_available() else 'cpu')
    args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')

    print(args)
    acc_fea = []
    acc_str = []
    acc_stu = []
    f1_fea = []
    f1_str = []
    f1_stu = []
    pre_fea = []
    pre_str = []
    pre_stu = []
    rec_fea = []
    rec_str = []
    rec_stu = []

    for repeat in range(args.iters):
        print('-------------------- Repeat {} Start -------------------'.format(repeat))
        setup_seed(1, torch.cuda.is_available())
        train = Train(args,repeat,acc_fea,acc_str,acc_stu, f1_fea, f1_str, f1_stu, pre_fea, pre_str, pre_stu, rec_fea,rec_str,rec_stu)
        t_total = time.time()
        #
        for epoch in range(args.epoch_fea):
            train.pre_train_teacher_fea(epoch)
        train.save_checkpoint(ts='teacher_fea')

        # # pre-train Structure teacher model
        for epoch in range(args.epoch_str):
            train.pre_train_teacher_str(epoch)
        train.save_checkpoint(ts='teacher_str')

        # load best pre-train teahcer models
        train.load_checkpoint(ts='teacher_fea')
        train.load_checkpoint(ts='teacher_str')
        print('\n--------------\n')

        # train student model GCN
        for epoch in range(args.epoch_stu):
            train.train_student(epoch)
        train.save_checkpoint(ts='student')

        # test teahcer models
        train.test('teacher_fea')
        train.test('teacher_str')

        # test student model GCN
        train.load_checkpoint(ts='student')
        train.test('student')

        print('******************** Repeat {} Done ********************\n'.format(repeat+1))



    print('ACC_Result_str: {}'.format(acc_str))
    print('Avg acc_str: {:.6f}'.format(sum(acc_str) / args.iters))
    print('ACC_Result_fea: {}'.format(acc_fea))
    print('Avg acc_fea: {:.6f}'.format(sum(acc_fea) / args.iters))
    print('ACC_Result_STU: {}'.format(acc_stu))
    print('Avg acc_STU: {:.6f}'.format(sum(acc_stu) / args.iters))

    print('F1_Result_str: {}'.format(f1_str))
    print('Avg F1_str: {:.6f}'.format(sum(f1_str) / args.iters))
    print('F1_Result_fea: {}'.format(f1_fea))
    print('Avg F1_fea: {:.6f}'.format(sum(f1_fea) / args.iters))
    print('F1_Result_STU: {}'.format(f1_stu))
    print('Avg F1_STU: {:.6f}'.format(sum(f1_stu) / args.iters))

    print('Precision_Result_str: {}'.format(pre_str))
    print('Avg pre_str: {:.6f}'.format(sum(pre_str) / args.iters))
    print('Precision_Result_fea: {}'.format(pre_fea))
    print('Avg pre_fea: {:.6f}'.format(sum(pre_fea) / args.iters))
    print('Precision_Result_STU: {}'.format(pre_stu))
    print('Avg pr_STUe: {:.6f}'.format(sum(pre_stu) / args.iters))

    print('Recall_Result_str: {}'.format(rec_str))
    print('Avg rec_str: {:.6f}'.format(sum(rec_str) / args.iters))
    print('Recall_Result_fea: {}'.format(rec_fea))
    print('Avg rec_fea: {:.6f}'.format(sum(rec_fea) / args.iters))
    print('Recall_Result_STU: {}'.format(rec_stu))
    print('Avg rec_STU: {:.6f}'.format(sum(rec_stu) / args.iters))

    print('\nAll Done!')
