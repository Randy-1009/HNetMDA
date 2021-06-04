import dgl
import torch as th
import numpy as np
import tqdm
import re


input_drug_microbe = np.loadtxt('../data/aBiofilm/adj.txt')  # 微生物-药物的关联关系
input_drug_drug = np.loadtxt('../data/aBiofilm/drugsimilarity.txt')  # 药物相似性
input_microbe_microbe = np.loadtxt('../data/aBiofilm/microbesimilarity.txt')  # 微生物相似性

drug_s_source = []  # 药物相似网络
drug_s_target = []
drug_t_source = []  # 药物-微生物关联网络
microbe_t_target = []
microbe_s_source = []  # 微生物相似网络
microbe_s_target = []

drug_microbe_weight = []
# drug_weight = input_drug_drug.tolist()
# microbe_weight = input_microbe_microbe.tolist()

i = 0
drug = []
microbe = []
edges_num = len(input_drug_microbe)
while i < edges_num:
    drug_name = 'd_' + str(int(input_drug_microbe[i][0]))
    microbe_name = 'm_' + str(int(input_drug_microbe[i][1]))
    drug_t_source.append(drug_name)
    microbe_t_target.append(microbe_name)
    # drug_microbe_weight.append(input_drug_microbe[i][2])  # 权重，有关联即为1
    if drug_name not in drug:
        drug.append(drug_name)
    if microbe_name not in microbe:
        microbe.append(microbe_name)
    # print(drug_t_source[i], microbe_t_target[i])
    i += 1

# print(drug)
# print(microbe)
drug_ids_index_map = {x: i for i, x in enumerate(drug)}
microbe_ids_index_map = {x: i for i, x in enumerate(microbe)}
drug_index_id_map = {i: x for i, x in enumerate(drug)}
microbe_index_id_map = {i: x for i, x in enumerate(microbe)}

i = 0
drug_microbe_src = []
drug_microbe_dst = []
while i < edges_num:
    drug_microbe_src.append(drug_ids_index_map.get(drug_t_source[i]))
    drug_microbe_dst.append(microbe_ids_index_map.get(microbe_t_target[i]))
    i = i + 1

# print(drug_ids_index_map)
# print(microbe_ids_index_map)
'''
i = 0
edges_num = len(input_drug_drug)
while i < edges_num:
    drug_weight.append(input_drug_drug[i])

i = 0
edges_num = len(input_microbe_microbe)
while i < edges_num:
    drug_weight.append(input_microbe_microbe[i])
'''

# 2种节点类型：微生物、药物；
# 3种边类型：药物-药物、药物-微生物、微生物-微生物
'''
graph_data = {
    ('drug', 'dd', 'drug'): (th.tensor([drug_s_source]), th.tensor(drug_s_target)),
    ('drug', 'dm', 'microbe'): (th.tensor([drug_t_source]), th.tensor(microbe_t_target)),
    ('microbe', 'mm', 'microbe'): (th.tensor([microbe_s_source]), th.tensor(microbe_s_target)),
}
'''

graph_data = {
    ('drug', 'dm', 'microbe'): (th.tensor(drug_microbe_src), th.tensor(drug_microbe_dst)),
    ('microbe', 'md', 'drug'): (th.tensor(drug_microbe_dst), th.tensor(drug_microbe_src))
}
hetero_graph = dgl.heterograph(graph_data)
tool_graph = dgl.heterograph(graph_data)

file_name = '../output/aBiofilm_embeddings.txt'
with open(file_name, 'r') as f:
    data = f.readlines()

features = []
for row in data:
    row = row.replace('\n', '')
    # print(row)
    features.append(row)

# print(len(features))
num, dim = features[0].split(' ')  # 具有特征的节点的个数和特征的维度
nodes_num = int(num)
features_dim = int(dim)
# 初始化节点特征
hetero_graph.nodes['drug'].data['feature'] = th.zeros(hetero_graph.num_nodes('drug'), features_dim)
hetero_graph.nodes['microbe'].data['feature'] = th.zeros(hetero_graph.num_nodes('microbe'), features_dim)

# print(hetero_graph.nodes['microbe'].data['feature'][0])

j = 0
for i in range(1, int(nodes_num) + 1):
    raw_features = features[i].split()
    # print(raw_features)
    node_name = raw_features[0]
    raw_features.remove(raw_features[0])
    # print(raw_features)
    for index, item in enumerate(raw_features):
        raw_features[index] = float(item)

    node_feature = th.tensor(raw_features)
    # print(node_feature)
    if re.match(r'd_', node_name):
        # print(node_name)
        drug_id = drug_ids_index_map[node_name]
        hetero_graph.nodes['drug'].data['feature'][drug_id] = node_feature
        j = j + 1
        # print(hetero_graph.nodes['drug'].data['feature'][drug_id])
        # print(drug_id)
    if re.match(r'm_', node_name):
        # print(node_name)
        microbe_id = microbe_ids_index_map[node_name]
        hetero_graph.nodes['microbe'].data['feature'][microbe_id] = node_feature
        j = j + 1
        # print(hetero_graph.nodes['microbe'].data['feature'][microbe_id])
        # print(microbe_id)
    # print(raw_features)
# print(j)
# print(hetero_graph.nodes['drug'].data['feature'])
# print(hetero_graph.nodes['microbe'].data['feature'])

# print(drug_ids_index_map)
# print(microbe_ids_index_map)
# print(hetero_graph.edge_ids(th.tensor([0]), th.tensor([0]), etype='dm'))
# print(hetero_graph.edge_ids(th.tensor([21]), th.tensor([3]), etype='dm'))
# print(hetero_graph.num_nodes('drug'))
# print(hetero_graph.num_nodes('microbe'))
# print(hetero_graph.all_edges(etype='dm'))

