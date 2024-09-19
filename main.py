import datetime
import argparse
import random
import numpy as np
import torch
import os.path
from sklearn.model_selection import StratifiedKFold
import torch_geometric as tg
import copy
from utils import *
from models.inception import InceptionModel
from sklearn.metrics import roc_curve, auc
from models.ServerModel import UniteGCN, Server
from models.ClientModel import DenseGCN, Client
import pandas as pd
class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of FedSP')
        # parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        # parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--server_lr', default=0.001, type=float, help='initial server learning rate')
        parser.add_argument('--client_lr', default=0.001, type=float, help='initial client learning rate')
        parser.add_argument('--server_wd', default=0.0005, type=float, help='initial server weight decay')
        parser.add_argument('--client_wd', default=0.0005, type=float, help='initial client weight decay')
        parser.add_argument('--epoch', default=200, type=int, help='number of epochs for training')
        parser.add_argument('--num_client', default=4, type=int, help='number of client')

        parser.add_argument('--client_in', default=512, type=int, help='size of input client')
        parser.add_argument('--client_hgc', default=128, type=int, help='size of hidden client')
        parser.add_argument('--client_out', default=10, type=int, help='size of output client')
        parser.add_argument('--client_dropout', default=0.5, type=float, help='size of dropout client')

        # parser.add_argument('--server_in', default=512, type=int, help='size of input server')
        parser.add_argument('--server_hgc', default=128, type=int, help='size of hidden server')
        parser.add_argument('--server_out', default=2, type=int, help='size of output server')
        parser.add_argument('--server_dropout', default=0.5, type=float, help='size of dropout server')
        parser.add_argument('--server_iter', default=1, type=int, help='size of dropout server')

        parser.add_argument('--time_out', default=64, type=float, help='size of inception output')
        parser.add_argument('--time_lg', default=2, type=float, help='layers of inception blocks ')

        parser.add_argument('--seed', type=int, default=1000, help='random seed to set')
        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        # 记录本次实验的结果
        file = open(f"../output/{args.time}.txt", 'w')
        file.close()

        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.args = args

    def initialize(self):
        self.set_seed(self.args.seed)
        return self.args

    def set_seed(self, seed=1000):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def main(data, label, args):

    # 分层抽样
    skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    data_index = range(len(label))
    Fold_Best_result = []
    best_acc_list = []
    Data = torch.tensor(data).to(args.device)
    Label = torch.tensor(label).to(args.device)
    criterion = nn.CrossEntropyLoss()

    # 训练
    fold = 0
    for train_index, test_index in skf.split(data_index, label):
        fold = fold + 1

        dict_users = dataset_iid(Data[train_index], args.num_client)
        dict_users_test = dataset_iid(Data[test_index], args.num_client)

        from ServerModel import GCN
        net_glob_client = DenseGCN(args.client_in, args.client_hgc, args.client_out, args.client_dropout)


        net_glob_client.to(args.device)
        # net_glob_server = GCN(30*10+64, 128, 2, 0.5)
        # FedGST
        net_glob_server = UniteGCN(30*args.client_out+args.time_out, args.server_hgc, args.server_out, args.server_dropout)

        # 对比方法
        # net_glob_server = UniteGCN(30+args.time_out, args.server_hgc, args.server_out, args.server_dropout)
        # net_glob_server = GCN(30+args.time_out, args.server_hgc, args.server_out, args.server_dropout)
        net_glob_server.to(args.device)

        w_glob_client = net_glob_client.state_dict()
        best_acc = 0
        best_result = 0

        # 创建客服端, 每折一直存在, 初始化客户端
        client_list = []
        for idx in range(args.num_client):
            train_idx = train_index[list(dict_users[idx])]
            test_idx = test_index[list(dict_users_test[idx])]
            client = Client(copy.deepcopy(net_glob_client).to(args.device), idx, args.client_lr, args.device, client_data=Data, client_label=Label,
                   idxs=train_idx, idxs_test=test_idx)
            client_list.append(client)
        # 创建服务器端
        server = Server(copy.deepcopy(net_glob_server).to(args.device), args.server_lr, args.device, criterion, server_iter=args.server_iter)
        for iter in range(args.epoch):
            print(f"Epoch {iter}:\t", end='')
            frac = 1
            m = max(int(frac * args.num_client), 1)
            idxs_users = np.random.choice(range(args.num_client), m, replace=False)
            w_locals_client = []
            # print(scheduler.get_last_lr()[0])
            # 初始化客户端结果列表
            activation_list = []
            client_label = []
            label_index_list = []  # 保存客户端训练集标签长度，以便再server提取训练集进行传播
            for idx in idxs_users:
                # 取客户端
                cur_client = client_list[idx]
                # 客户端前向传播，返回最后一层的activation
                activation, cur_label, label_index = cur_client.train()
                # 收集所有客户端的activation输入到全局模型
                activation_list.append(activation)
                client_label.append(cur_label)
                label_index_list.append(label_index)
            # server 端训练
            client_gradient, result, loss = server.train(activation_list, client_label, label_index_list)

            # 生成每个客户端的索引来分割梯度
            start = 0
            for idx in idxs_users:
                # 计算客户端的样本长度
                client_length = len(label_index_list[idx])
                # 根据样本长度分割梯度
                idx_client_gradient = client_gradient[start:client_length + start, :]
                # 梯度发送到对应客户端进行反向传播
                w_client = client_list[idx].back(idx_client_gradient)
                # 收集每个客户端优化后的梯度
                w_locals_client.append(copy.deepcopy(w_client))

                start = client_length + start
            # 所有客户端进行联邦学习的权重聚合
            # print("------ FedServer: Federation process at Client-Side ------- ")
            w_glob_client = FedAvg(w_locals_client)
            eval_activation_list = []
            for idx in idxs_users:
                client_list[idx].update_model(w_glob_client)
                eval_activation = client_list[idx].evaluate()
                eval_activation_list.append(eval_activation)
            test_result = server.evaluate(eval_activation_list)

            if test_result[0] > best_acc:
                best_acc = test_result[0]
                best_result = test_result

                # 保存模型
                # server.save_pred()
                torch.save(w_glob_client, f'./checkpoint/ADHD_client_{fold}.pt')
                server.save_model(fold=fold)

        print(f"Fold-Result: {best_result}")
        Fold_Best_result.append(best_result)
    print("Training and Evaluation completed!")
    (ACC, AUC, SEN, SPE, F1) = print_metric(Fold_Best_result)

    # # 输出每个中心的结果
    # for idx in range(args.num_client):
    #     test_idx = test_index[list(dict_users_test[idx])]
    #     client_pred = server.pred[test_idx]
    #     test_result = metric(client_pred, self.Label[[self.Test_Label_Index]])


    # 保存参数
    args_dict = {}
    for arg, content in args.__dict__.items():
        args_dict[arg] = content
    args_dict['ACC'] = ACC
    args_dict['AUC'] = AUC
    args_dict['SEN'] = SEN
    args_dict['SPE'] = SPE
    args_dict['F1'] = F1

    filename = 'ADHD.csv'
    if os.path.exists(f'output/{filename}') == False:
        df = pd.DataFrame(columns=list(args_dict.keys()))
        col_index = df.shape[0]
        df.loc[col_index] = list(args_dict.values())
        df.to_csv(f'output/{filename}', index=False)
    else:
        metric_df = pd.read_csv(f'output/{filename}', delimiter=',')
        col_index = metric_df.shape[0]
        metric_df.loc[col_index] = list(args_dict.values())
        metric_df.to_csv(f'output/{filename}', index=False)


if __name__ == "__main__":
    #  参数初始化
    opt = OptInit()
    opt.initialize()

    # ASD 数据读取
    data = np.load("data/DynamicFc871_vec_30.npy").astype(np.float32)
    label = np.load("data/label.npy").astype(np.int64)
    t_index = np.load("data/ASD871_t_index_onefold.npy")
    data = data[:, :, t_index]

    # ADHD数据读取
    # data = np.load("data/ADHD200_DynamicFC.npy").astype(np.float32)
    # label = np.load("data/ADHD200_Label.npy")
    # t_index = np.load("data/ADHD_t_index_onefold.npy")
    # data = data[:, :, t_index]
    # label = np.where(label>0, 1, 0).astype(np.int64)

    # COBRE 数据读取
    # data = np.load("data/COBRE_DynamicFC_30.npy").astype(np.float32)
    # label = np.load("data/COBRE_label.npy").astype(np.int64)
    # t_index = np.load("data/COBRE_t_index_onefold.npy")
    # data = data[:, :, t_index]
    # c_lr = range(5, 11)
    # s_lr = range(5, 11)
    # for client_lr in c_lr:
    #     for server_lr in s_lr:
    #         opt.args.client_wd = client_lr * 0.0001
    #         opt.args.server_wd = server_lr * 0.0001
    #         main(data, label, opt.args)
    main(data, label, opt.args)
    print("finish")