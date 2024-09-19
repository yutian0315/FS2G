import torch_geometric as tg
from torch import nn
from models.inception import InceptionModel
import torch
import os.path
import torch.nn.functional as F
from models.Attention import AttentionLayer

class DenseGCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.3):
        super(DenseGCN, self).__init__()
        # DenseGCNConv
        # DenseGraphConv
        # DenseSAGEConv
        # DenseGATConv
        self.gcn1 = tg.nn.DenseSAGEConv(dim_in, dim_hidden)
        self.gcn2 = tg.nn.DenseSAGEConv(dim_hidden, dim_out)
        self.relu = nn.ReLU(dim_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.EdgeDropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()
        self.attn_fc = nn.Linear(dim_out, 1)
        # self.cls = MLP(30 * dim_out, 128, 2, 0.5)
        # self.cls = tg.nn.models.MLP(in_channels=30 * dim_out + 64,
        #                             hidden_channels=128,
        #                             out_channels=2,
        #                             num_layers=2,
        #                             dropout=0.5,
        #                             act='relu')
        # self.lstm = LSTM(512, 64, 2, 64)
        self.Inception = InceptionModel(num_blocks=2,
                                       in_channels=30,
                                       out_channels=8,
                                       bottleneck_channels=8,
                                       kernel_sizes=16,
                                       use_residuals=True,
                                       num_pred_classes=64)

        self.Attn = AttentionLayer(dim_in, num_heads=2, qkv_bias=True, qk_scale=None)
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
        # return adj
        return self.KNN(adj, 10)

    def forward(self, x):

        # 时间感知机制
        x_adj = [self.Adj(s).unsqueeze(0) for s in x]
        x_adj = torch.cat(x_adj)
        x_adj = self.EdgeDropout(x_adj)
        out = self.gcn1(self.dropout(x), x_adj)
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.gcn2(out, x_adj)
        # out = self.LeakyReLU(out)
        out = self.dropout(out)
        attn_score = self.attn_fc(out)

        # 普通注意力融合
        # out, attn = self.Attn(x)
        # attn = torch.mean(torch.mean(attn, dim=1), dim=1)
        # lstm_out = self.Inception(out)
        # cls_input = torch.cat([lstm_out, attn], dim=-1)

        # lstm_out = self.lstm(x * attn_score)
        lstm_out = self.Inception(x * attn_score)
        cls_input = torch.cat([lstm_out, out.reshape(out.size()[0], -1)], dim=-1)
        # pred = self.cls(cls_input)
        return cls_input

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, client_data= None, client_label = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        #self.selected_clients = []
        self.train_data = client_data[idxs]
        self.test_data = client_data[idxs_test]
        self.train_label = client_label[idxs]
        self.test_label = client_label[idxs_test]
        self.model = net_client_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0005, eps=1e-4)
        self.data = torch.cat([self.train_data, self.test_data], dim=0).to(self.device)
        self.label = torch.cat([self.train_label, self.test_label]).to(self.device)
        train_index = torch.ones(size=(len(self.train_label),), dtype=torch.bool)
        test_index = torch.zeros(size=(len(self.test_label),), dtype=torch.bool)
        self.label_index = torch.cat([train_index, test_index], dim=0)
        self.out = None
    def train(self,):

        self.model.train()

        for iter in range(self.local_ep):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            client_fx = out.clone().detach().requires_grad_(True)
            self.out = out

        return client_fx, self.label, self.label_index

    def back(self, gradient):
        self.out.backward(gradient)
        self.optimizer.step()
        return self.model.state_dict()

    def update_model(self, weight):
        self.model.load_state_dict(weight)

    def evaluate(self,):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
        return out

    def save_model(self, PATH='./checkpoint'):
        state = {}
        state['model_state'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['data'] = self.data
        torch.save(state, os.path.join(PATH, f'state_{self.idx}'))