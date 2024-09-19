import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn

def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def metric(preds, labels):
    # preds, labels----Tensor类型

    pred_label = preds.max(1, keepdim=True)[1].cpu().numpy()
    label = labels.cpu().numpy()
    pred_proba = preds.detach().cpu().numpy()
    result = []
    acc = sklearn.metrics.accuracy_score(label, pred_label)
    AUC = sklearn.metrics.roc_auc_score(np.eye(2)[label], pred_proba)
    fpr, tpr, thresholds = roc_curve(label, pred_proba[:, 1], pos_label=1)
    # t = auc(fpr,tpr)
    f1 = sklearn.metrics.f1_score(label, pred_label)
    cm = sklearn.metrics.confusion_matrix(label, pred_label)
    # recall = sklearn.metrics.recall_score(labels, preds)
    sen = cm[1][1] / (cm[1][1] + cm[1][0])
    spe = cm[0][0] / (cm[0][0] + cm[0][1])

    result.append(acc)
    result.append(AUC)
    result.append(sen)
    result.append(spe)
    result.append(f1)

    return result

def print_metric(result_list):
    # 一次kfold实验的结果
    acc_list = []
    auc_list = []
    sen_list = []
    spe_list = []
    f1_list = []

    for fold in result_list:
        acc_list.append(fold[0])
        auc_list.append(fold[1])
        sen_list.append(fold[2])
        spe_list.append(fold[3])
        f1_list.append(fold[4])

    print("ACC: {:.2f} ± {:.2f}".format(np.mean(acc_list) * 100, np.std(acc_list) * 100))
    print("AUC: {:.2f} ± {:.2f}".format(np.mean(auc_list) * 100, np.std(auc_list) * 100))
    print("SEN: {:.2f} ± {:.2f}".format(np.mean(sen_list) * 100, np.std(sen_list) * 100))
    print("SPE: {:.2f} ± {:.2f}".format(np.mean(spe_list) * 100, np.std(spe_list) * 100))
    print("F1: {:.2f} ± {:.2f}".format(np.mean(f1_list) * 100, np.std(f1_list) * 100))

    return (np.mean(acc_list) * 100,
            np.mean(auc_list) * 100,
            np.mean(sen_list) * 100,
            np.mean(spe_list) * 100,
            np.mean(f1_list) * 100)
def integrate_metric(result_list):
    # 将多个result去评价到一个result
    acc = 0
    auc = 0
    sen = 0
    spe = 0
    f1 = 0

    for result in result_list:
        acc += result[0]
        auc += result[1]
        sen += result[2]
        spe += result[3]
        f1 += result[4]

    list_len = len(result_list)
    final_result = []
    final_result.append(acc / list_len)
    final_result.append(auc / list_len)
    final_result.append(sen / list_len)
    final_result.append(spe / list_len)
    final_result.append(f1 / list_len)

    return final_result


import torch


def add_gaussian_noise(t, mean=0, std=0.01):
    """
    在输入的张量上添加高斯噪声，返回带有噪声的张量。

    Args:
        tensor (torch.Tensor): 输入的张量。
        mean (float): 高斯分布的均值，默认为0。
        std (float): 高斯分布的标准差，默认为0.1。

    Returns:
        torch.Tensor: 带有高斯噪声的张量。
    """
    noise = torch.randn(t.size()) * std + mean
    noisy_tensor = t + noise.to('cuda')
    return noisy_tensor

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
    return adj

# Custom dataset prepration in Pytorch format
class TensorData(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]

        if self.transform:
            X = self.transform(X)

        return X, y

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float() / preds.shape[0]
    return acc

# 模型参数初始化
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

from scipy.spatial import distance

# 高斯距离构图 numpy
def Gaussi_Graph(data):
    distv = distance.pdist(data, metric='euclidean')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))

    return sparse_graph

# numpy 版本
def get_knn_graph(adj, K=10):
    node_size = adj.shape[0]
    sort_idx = np.argsort(adj, axis=1)
    adj_idx = np.zeros((node_size, node_size))
    sort_idx_left = np.arange(node_size).repeat(K).reshape(K, node_size).T
    adj_idx[(sort_idx_left, sort_idx[:, :K])] = 1
    KNN_adj = adj * adj_idx
    return KNN_adj

# torch 版本
def KNN(adj, K=40):
    node_size = adj.size()[0]
    sort_idx = torch.argsort(adj, dim=1)
    adj_idx = torch.zeros((node_size, node_size)).to("cuda")
    sort_idx_left = torch.arange(node_size).repeat(K).view(K, node_size).T
    adj_idx[(sort_idx_left, sort_idx[:,-K:])] = 1
    # raw_adj = adj
    KNN_adj = adj * adj_idx
    # KNN_adj = F.normalize(KNN_adj, dim=1)
    return KNN_adj

from torch.nn import init
#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

# t-test 特征选择
def feature_select_t(Data, label, train_index, k):
    print("特征选择...")
    from scipy import stats
    train_feature = Data[train_index]
    zeros = (label[train_index] == 0)
    ones = (label[train_index] == 1)
    pvalue_list = []
    from tqdm import tqdm
    for i in tqdm(range(train_feature.shape[1])):
        hc_data = train_feature[:,i][zeros]
        asd_data = train_feature[:,i][ones]
        pvalue = stats.ttest_ind(hc_data,asd_data)[1]
        pvalue_list.append(pvalue)
    sort_index = np.argsort(pvalue_list)
    # feat_test = Data[:, sort_index[:k]]
    return sort_index[:k]


