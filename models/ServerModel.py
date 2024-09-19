import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from models.Attention import AttentionLayer
from utils import *
import os
class UniteGCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, is_pred=False):
        super().__init__()

        self.gc1 = tg.nn.TransformerConv(dim_in, dim_out)
        self.gc2 = tg.nn.SAGEConv(dim_in, dim_out, aggr='mean')
        self.gc3 = tg.nn.GraphConv(dim_in, dim_out, aggr='mean')
        # self.w = nn.Parameter(torch.ones([1, 1, 1], dtype=torch.float).to('cuda'), requires_grad=True)
        self.w = nn.Parameter(torch.ones(3))
        self.is_pred = is_pred
        self.Attn = AttentionLayer(dim_out, num_heads=2,qkv_bias=True,qk_scale=None)
        self.fc = nn.Linear(dim_out*3, dim_out)
    def forward(self, x, edge_index, edge_weight):
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        x_gc1 = self.gc1(x, edge_index)
        x_gc2 = self.gc2(x, edge_index)
        x_gc3 = self.gc3(x, edge_index, edge_weight)
        if self.is_pred == False:
            x_unite = torch.cat((x_gc1.unsqueeze(1), x_gc2.unsqueeze(1), x_gc3.unsqueeze(1)), dim=1)
            x_unite, attn = self.Attn(x_unite)
            x_unite = x_unite.view(x.size()[0], -1)
            x_unite = self.fc(x_unite)
        else:
            x_unite = torch.cat((x_gc1, x_gc2, x_gc3), dim=1)
            x_unite = x_unite.view(x.size()[0], 3, -1)
            x_unite = torch.mean(x_unite, dim=1)
        # if self.is_pred:
        #
        # x_unite = x_gc1 * w1 + x_gc2 * w2 + x_gc3 * w3
        return x_unite


class UniteGCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.5):
        super(UniteGCN, self).__init__()
        self.gcn1 = UniteGCNLayer(dim_in, dim_hidden)
        self.gcn2 = UniteGCNLayer(dim_hidden, dim_out, is_pred=True)

        self.bn1 = nn.BatchNorm1d(dim_hidden)
        # self.relu = nn.ReLU(dim_hidden)
        self.relu = nn.Softplus()

        self.dropout = nn.Dropout(p=dropout)
    def KNN(self, adj, K=40):
        node_size = adj.size()[0]
        sort_idx = torch.argsort(adj, dim=1)
        adj_idx = torch.zeros((node_size, node_size)).to("cuda")
        sort_idx_left = torch.arange(node_size).repeat(K).view(K, node_size).T
        adj_idx[(sort_idx_left, sort_idx[:, -K:])] = 1
        # raw_adj = adj
        KNN_adj = adj * adj_idx
        # KNN_adj = F.normalize(KNN_adj, dim=1)
        return KNN_adj

    def Adj(self, x, Type='N'):
        if Type == 'N':
            x_norm = F.normalize(x, dim=-1)
            adj = torch.matmul(x_norm, x_norm.T)
        elif Type == 'G':
            x = x.unsqueeze(dim=1) - x.unsqueeze(dim=0)
            x = torch.pow(x, 2)
            adj = torch.sum(x, dim=-1)
            sigma = adj.mean()
            adj = torch.exp(- adj / sigma)
        return self.KNN(adj, 20)
        # return adj

    def forward(self, x):
        adj = self.Adj(x)
        edge_index, edgenet_input = tg.utils.dense_to_sparse(adj)
        x = self.dropout(x)
        x = self.gcn1(x, edge_index, edgenet_input)
        # x = self.gcn1(x, edge_index)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edgenet_input)
        # x = self.gcn2(x, edge_index)
        # return F.log_softmax(x, dim=-1)
        return x


class Server(object):
    def __init__(self, net_Server_model, lr, device, criterion, server_iter):
        self.device = device
        self.lr = lr
        self.model = net_Server_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0005, eps=1e-4)
        self.loss = criterion
        self.best_metric = [0,]
        self.pred = None
        self.server_iter = server_iter
    def train(self, data, label, train_label_index):
        print("Training: ", end='')
        self.model.train()
        # 拼接数据
        self.Data = torch.cat(data).clone().detach().requires_grad_(True)
        self.Label = torch.cat(label)
        self.Train_Label_Index = torch.cat(train_label_index)
        self.Test_Label_Index = (self.Train_Label_Index==False)
        # 训练模型
        for iter in range(self.server_iter):
            self.optimizer.zero_grad()
            pred = self.model(self.Data)
            train_loss = self.loss(pred[self.Train_Label_Index], self.Label[self.Train_Label_Index])
            test_loss = self.loss(pred[self.Test_Label_Index], self.Label[self.Test_Label_Index])
            train_loss.backward()
        # 生成梯度, 发送到客户端
        gradient_client = self.Data.grad.clone().detach()
        self.optimizer.step()

        # 计算性能指标
        train_result = metric(pred[[self.Train_Label_Index]], self.Label[[self.Train_Label_Index]])
        test_result = metric(pred[[self.Test_Label_Index]], self.Label[[self.Test_Label_Index]])
        print("Train ACC: {:.4f}, loss: {:.4f}".format(train_result[0], train_loss), end=' | ')
        # print("Test ACC: {:.4f}, loss: {:.4f}".format(test_result[0], test_loss))

        return gradient_client, (train_result, test_result), (train_loss,test_loss)

    def evaluate(self, data):
        # print("\t\t Testing:  ", end='')
        self.model.eval()
        eval_data = torch.cat(data)
        with torch.no_grad():
            pred = self.model(eval_data)
        train_loss = self.loss(pred[self.Train_Label_Index], self.Label[self.Train_Label_Index])
        test_loss = self.loss(pred[self.Test_Label_Index], self.Label[self.Test_Label_Index])
        # 计算性能指标
        train_result = metric(pred[[self.Train_Label_Index]], self.Label[[self.Train_Label_Index]])
        test_result = metric(pred[[self.Test_Label_Index]], self.Label[[self.Test_Label_Index]])
        # print("Train ACC: {:.4f}, loss: {:.4f}".format(train_result[0], train_loss), end=' | ')
        if test_result[0] > self.best_metric[0]:
            self.best_metric = test_result
            self.pred = pred

            # 保存不同中心的结果
            # quarter_length = pred.size()[0] // 4
            # for i in range(4):
            #     indices = torch.arange(i*quarter_length, (i+1)*quarter_length)
            #     client_result = metric(pred[indices], self.Label[indices])
            #     print(client_result)
        print("Test ACC: {:.4f}, loss: {:.4f}".format(test_result[0], test_loss))
        return test_result
    def save_model(self, PATH='./checkpoint', fold=0):
        state = {}
        state['model_state'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, os.path.join(PATH, f'server_{fold}.pt'))
    def save_pred(self):

        y_test = self.Label[[self.Test_Label_Index]].cpu().numpy()
        y_score = self.pred[[self.Test_Label_Index]].detach().cpu().numpy()
        fpr, tpr, thread = roc_curve(y_test, y_score[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
        plt.show()


class GCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.3):
        super(GCN, self).__init__()
        # self.gcn1 = tg.nn.ChebConv(dim_in, dim_hidden, 3, normalization='sym', bias=False)
        # self.gcn2 = tg.nn.ChebConv(dim_hidden, dim_out, 3, normalization='sym', bias=False)
        self.gcn1 = tg.nn.GCNConv(dim_in, dim_hidden)
        self.gcn2 = tg.nn.GCNConv(dim_hidden, dim_out)
        # self.gcn1 = tg.nn.TransformerConv(dim_in, dim_hidden)
        # self.gcn2 = tg.nn.TransformerConv(dim_hidden, dim_out)
        # self.gcn1 = tg.nn.GraphConv(dim_in, dim_hidden, aggr='mean')
        # self.gcn2 = tg.nn.GraphConv(dim_hidden, dim_out, aggr='mean')
        # self.gcn1 = tg.nn.SAGEConv(dim_in, dim_hidden, aggr='mean')
        # self.gcn2 = tg.nn.SAGEConv(dim_hidden, dim_out, aggr='mean')

        # self.cls = tg.nn.models.MLP(in_channels=30 * dim_in,
        #              hidden_channels=dim_hidden,
        #              out_channels=dim_out,
        #              num_layers=2,
        #              dropout=0.3,
        #              act='relu')
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu = nn.ReLU(dim_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def KNN(self, adj, K=40):
        node_size = adj.size()[0]
        sort_idx = torch.argsort(adj, dim=1)
        adj_idx = torch.zeros((node_size, node_size)).to("cuda")
        sort_idx_left = torch.arange(node_size).repeat(K).view(K, node_size).T
        adj_idx[(sort_idx_left, sort_idx[:, -K:])] = 1
        # raw_adj = adj
        KNN_adj = adj * adj_idx
        # KNN_adj = F.normalize(KNN_adj, dim=1)
        return KNN_adj

    def Adj(self, x, Type='N'):
        if Type == 'N':
            x_norm = F.normalize(x, dim=-1)
            adj = torch.matmul(x_norm, x_norm.T)
        elif Type == 'G':
            x = x.unsqueeze(dim=1) - x.unsqueeze(dim=0)
            x = torch.pow(x, 2)
            adj = torch.sum(x, dim=-1)
            sigma = adj.mean()
            adj = torch.exp(- adj / sigma)
        return self.KNN(adj, 20)
        # return adj

    def forward(self, x):
        adj = self.Adj(x)
        edge_index, edgenet_input = tg.utils.dense_to_sparse(adj)
        # x = self.cls(x, edge_index=edge_index, edge_weight=edgenet_input)
        x = self.dropout(x)
        x = self.gcn1(x, edge_index, edgenet_input)
        # x = self.gcn1(x, edge_index)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edgenet_input)
        # x = self.gcn2(x, edge_index)
        # return F.log_softmax(x, dim=-1)
        return x