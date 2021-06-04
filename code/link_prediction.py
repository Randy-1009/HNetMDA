import random

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import mean

from graph_info import hetero_graph
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


# 激活层，归一化函数sigmoid
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 进行k次负采样，针对要预测的边的种类构建负采样图
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super(RGCN, self).__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super(Model, self).__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        # print(x)
        h = self.sage(g, x)
        # print(h)
        # 返回正样本得分和负样本得分
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    # print(n_edges)
    # print(pos_score.unsqueeze(1))
    # print(neg_score.view(n_edges, -1))
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


# 负采样次数
k = 4
epoch_num = 200
# print(hetero_graph.etypes)
# model = Model(16, 32, 16, hetero_graph.etypes)
# drug_feats = hetero_graph.nodes['drug'].data['feature']
# microbe_feats = hetero_graph.nodes['microbe'].data['feature']
# node_features = {'drug': drug_feats, 'microbe': microbe_feats}
# opt = torch.optim.Adam(model.parameters())

link = []
link_num = int(hetero_graph.num_edges() / 2)
for i in range(link_num):
    link.append(i)
kf = KFold(n_splits=2, shuffle=True)
kf.get_n_splits(link)
auc_list = []
aupr_list = []
accuracy_list = []
edges_num = hetero_graph.num_edges('dm')

for train_index, test_index in kf.split(link):
    # 测试集中边的出点和入点ID
    # src_dst = hetero_graph.find_edges(test_index, 'dm')
    # 复制一个异构网络用于采集子网
    sub_graph = dgl.edge_subgraph(hetero_graph, {('drug', 'dm', 'microbe'): link,
                                                 ('microbe', 'md', 'drug'): link})
    model = Model(32, 64, 32, hetero_graph.etypes)
    drug_feats = sub_graph.nodes['drug'].data['feature']
    microbe_feats = sub_graph.nodes['microbe'].data['feature']
    node_features = {'drug': drug_feats, 'microbe': microbe_feats}
    opt = torch.optim.Adam(model.parameters())
    sub_graph.remove_edges(test_index, 'dm')
    # print(sub_graph)
    # 利用子图也就是训练集训练模型
    for epoch in range(epoch_num):
        negative_graph = construct_negative_graph(sub_graph, k, ('drug', 'dm', 'microbe'))
        pos_score, neg_score = model.forward(sub_graph, negative_graph, node_features, ('drug', 'dm', 'microbe'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

    # torch.save(model, )
    pos_link = 0
    neg_link = 0
    test_len = len(test_index)  # 测试集个数
    drug_feats = hetero_graph.nodes['drug'].data['feature']
    microbe_feats = hetero_graph.nodes['microbe'].data['feature']
    node_features = {'drug': drug_feats, 'microbe': microbe_feats}
    negative_graph = construct_negative_graph(hetero_graph, k, ('drug', 'dm', 'microbe'))
    pos_score, neg_score = model.forward(hetero_graph, negative_graph, node_features, ('drug', 'dm', 'microbe'))
    edges_score = []
    edges_label = []
    neg_test_index = []
    # 生成负采样中用于结果评分计算的随机验证集，长度和正属性集长度相等
    for i in range(test_len):
        rand_num = random.randrange(0, k * edges_num)
        if rand_num not in neg_test_index:
            neg_test_index.append(rand_num)

    for eid in test_index:
        score = float(pos_score[eid][0])
        real_score = sigmoid(score)
        edges_label.append(1)
        edges_score.append(real_score)
        if real_score > 0.7:
            pos_link = pos_link + 1

    for eid in neg_test_index:
        score = float(neg_score[eid][0])
        real_score = sigmoid(score)
        edges_label.append(0)
        edges_score.append(real_score)
    '''
    for eid in pos_score:
        score = float(eid[0])
        real_score = sigmoid(score)
        edges_label.append(1)
        edges_score.append(real_score)
    for eid in neg_score:
        score = float(eid[0])
        real_score = sigmoid(score)
        edges_label.append(0)
        edges_score.append(real_score)
    '''
    # print(edges_score)
    # print(edges_label)
    fpr, tpr, thresholds_auc = roc_curve(edges_label, edges_score)
    precision, recall, thresholds_aupr = precision_recall_curve(edges_label, edges_score)
    auc_score = auc(fpr, tpr)
    auc_list.append(auc_score)
    accuracy = pos_link / len(test_index)
    accuracy_list.append(accuracy)
    aupr_score = auc(recall, precision)
    aupr_list.append(aupr_score)
    print('\n')

    plt.figure(1)  # 创建图表1
    plt.title('ROC Curve')  # give plot a title
    plt.xlabel('FPR(1-specificity)')  # make axis labels
    plt.ylabel('TPR(sensitivity)')
    # print(len(precision))
    # print(precision)
    plt.plot(fpr, tpr)
    # plt.show()

    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.figure(1)
    plt.plot(recall, precision)
    # plt.show()
    # plt.savefig('p-r.png')
print(auc_list)
print(aupr_list)
# print(accuracy_list)
print('AUC:' + str(mean(auc_list)))
print('AUPR:' + str(mean(aupr_list)))
# print('Accuracy:' + str(mean(accuracy_list)))
print('AUC_standard deviation:' + str(np.std(auc_list, ddof=1)))
print('AUPR_standard deviation:' + str(np.std(aupr_list, ddof=1)))
