import gc
import pathlib
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io

save_prefix = '../output/relationship/VI_step_data_划分/'
num_miRNA = 3153
num_circRNA = 10062
num_lncRNA = 3862
num_gene = 218
num_disease = 6184
miRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2mi_allinf.csv', encoding='utf-8', delimiter=',', names=['miRNAid','miRNAName','diseaseid','disease','database','pmid'])
circRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2circ_allinf.csv', encoding='utf-8', delimiter=',', names=['circRNAid','circRNAName','diseaseid','disease','database','pmid'])
lncRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2lnc_allinf.csv', encoding='utf-8', delimiter=',', names=['lncRNAid','lncRNAName','diseaseid','disease','database','pmid'])
gene_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2gene_allind.csv', encoding='utf-8', delimiter=',', names=['geneid','gene','diseaseid','disease','database','pmid'])
miRNA_circRNA = pd.read_csv('../output/relationship/V_step_relationship/mir2circ_allinf.csv', encoding='utf-8', delimiter=',', names=['miRNAid','miRNAName','circRNAid','circRNA','database','pmid'])
miRNA_lncRNA = pd.read_csv('../output/relationship/V_step_relationship/mir2lnc_allinf.csv', encoding='utf-8', delimiter=',', names=['miRNAid','miRNAName','lncRNAid','lncRNA','database','pmid'])
miRNA_gene = pd.read_csv('../output/relationship/V_step_relationship/mir2gene_allinf.csv', encoding='utf-8', delimiter=',', names=['miRNAid','miRNAName','geneid','gene','database','pmid'])

miRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/miRNASim.csv', encoding='utf-8', delimiter=',', names=['miRNAID', 'adjacentID'])
miRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_similarity.csv', encoding='utf-8', delimiter='\t', names=['similarity'])
circRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/circRNASim.csv', encoding='utf-8', delimiter=',', names=['circRNAID', 'adjacentID'])
circRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_similarity.csv', encoding='utf-8', delimiter='\t', names=['similarity'])
lncRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/lncRNASim.csv', encoding='utf-8', delimiter=',', names=['lncRNAID', 'adjacentID'])
lncRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_similarity.csv', encoding='utf-8', delimiter='\t', names=['similarity'])
gene_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/geneSim.csv', encoding='utf-8', delimiter=',', names=['geneID', 'adjacentID'])
gene_Sim = pd.read_csv('../output/relationship/IV_step_similarity/gene_similarity.csv', encoding='utf-8', delimiter='\t', names=['similarity'])
disease_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj.csv', encoding='utf-8', delimiter=':', names=['diseaseID', 'adjacentID'])
disease_Sim = pd.read_csv('../output/relationship/IV_step_similarity/disease_similarity.csv', encoding='utf-8', delimiter='\t', names=['similarity'])
print(1)


dis2circ_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/dis2circ_train_val_test_idx.npz')
# dis2circ_train_idx = dis2circ_train_val_test_idx['dis2circ_train_idx']
# dis2circ_val_idx = dis2circ_train_val_test_idx['dis2circ_val_idx']
# dis2circ_test_idx = dis2circ_train_val_test_idx['dis2circ_test_idx']
dis2circ_train_idx = dis2circ_train_val_test_idx['train_idx']
dis2circ_val_idx = dis2circ_train_val_test_idx['val_idx']
dis2circ_test_idx = dis2circ_train_val_test_idx['test_idx']
circRNA_disease = circRNA_disease.loc[dis2circ_train_idx].reset_index(drop=True)
print(1)
dis2lnc_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/dis2lnc_train_val_test_idx.npz')
# dis2lnc_train_idx = dis2lnc_train_val_test_idx['dis2lnc_train_idx']
# dis2lnc_val_idx = dis2lnc_train_val_test_idx['dis2lnc_val_idx']
# dis2lnc_test_idx = dis2lnc_train_val_test_idx['dis2lnc_test_idx']
dis2lnc_train_idx = dis2lnc_train_val_test_idx['train_idx']
dis2lnc_val_idx = dis2lnc_train_val_test_idx['val_idx']
dis2lnc_test_idx = dis2lnc_train_val_test_idx['test_idx']
lncRNA_disease = lncRNA_disease.loc[dis2lnc_train_idx].reset_index(drop=True)
print(1)
dis2mi_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/dis2mi_train_val_test_idx.npz')
# dis2mi_train_idx = dis2mi_train_val_test_idx['dis2mi_train_idx']
# dis2mi_val_idx = dis2mi_train_val_test_idx['dis2mi_val_idx']
# dis2mi_test_idx = dis2mi_train_val_test_idx['dis2mi_test_idx']
dis2mi_train_idx = dis2mi_train_val_test_idx['train_idx']
dis2mi_val_idx = dis2mi_train_val_test_idx['val_idx']
dis2mi_test_idx = dis2mi_train_val_test_idx['test_idx']
miRNA_disease = miRNA_disease.loc[dis2mi_train_idx].reset_index(drop=True)
print(1)
dis2gene_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/dis2gene_train_val_test_idx.npz')
# dis2gene_train_idx = dis2gene_train_val_test_idx['dis2gene_train_idx']
# dis2gene_val_idx = dis2gene_train_val_test_idx['dis2gene_val_idx']
# dis2gene_test_idx = dis2gene_train_val_test_idx['dis2gene_test_idx']

dis2gene_train_idx = dis2gene_train_val_test_idx['train_idx']
dis2gene_val_idx = dis2gene_train_val_test_idx['val_idx']
dis2gene_test_idx = dis2gene_train_val_test_idx['test_idx']
gene_disease = gene_disease.loc[dis2gene_train_idx].reset_index(drop=True)
print(1)
mi2circ_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/mi2circ_train_val_test_idx.npz')
# mi2circ_train_idx = mi2circ_train_val_test_idx['mi2circ_train_idx']
# mi2circ_val_idx = mi2circ_train_val_test_idx['mi2circ_val_idx']
# mi2circ_test_idx = mi2circ_train_val_test_idx['mi2circ_test_idx']
mi2circ_train_idx = mi2circ_train_val_test_idx['train_idx']
mi2circ_val_idx = mi2circ_train_val_test_idx['val_idx']
mi2circ_test_idx = mi2circ_train_val_test_idx['test_idx']
miRNA_circRNA = miRNA_circRNA.loc[mi2circ_train_idx].reset_index(drop=True)
print(1)
mi2lnc_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/mi2lnc_train_val_test_idx.npz')
# mi2lnc_train_idx = mi2lnc_train_val_test_idx['mi2lnc_train_idx']
# mi2lnc_val_idx = mi2lnc_train_val_test_idx['mi2lnc_val_idx']
# mi2lnc_test_idx = mi2lnc_train_val_test_idx['mi2lnc_test_idx']
mi2lnc_train_idx = mi2lnc_train_val_test_idx['train_idx']
mi2lnc_val_idx = mi2lnc_train_val_test_idx['val_idx']
mi2lnc_test_idx = mi2lnc_train_val_test_idx['test_idx']
miRNA_lncRNA = miRNA_lncRNA.loc[mi2lnc_train_idx].reset_index(drop=True)
print(1)
mi2gene_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/mi2gene_train_val_test_idx.npz')
# mi2gene_train_idx = mi2gene_train_val_test_idx['mi2gene_train_idx']
# mi2gene_val_idx = mi2gene_train_val_test_idx['mi2gene_val_idx']
# mi2gene_test_idx = mi2gene_train_val_test_idx['mi2gene_test_idx']
mi2gene_train_idx = mi2gene_train_val_test_idx['train_idx']
mi2gene_val_idx = mi2gene_train_val_test_idx['val_idx']
mi2gene_test_idx = mi2gene_train_val_test_idx['test_idx']
miRNA_gene = miRNA_gene.loc[mi2gene_train_idx].reset_index(drop=True)
# print(1)
# # build the adjacency matrix
# #    0 for miRNA, 1 for circRNA, 2 for lncRNA, 3 for gene, 4 for disease
# dim = num_miRNA + num_circRNA +  num_lncRNA +  num_gene +  num_disease
# # 构建零矩阵 一个长度为dim的一维数组。dtype=int指定了数组的数据类型为整数。
# # 从0索引到num_miRNA - 1索引的元素赋值为0。标记为类型1，表示miRNA。
# type_mask = np.zeros((dim), dtype=int)   #
# # 从num_miRNA索引到num_miRNA + num_circRNA - 1索引的元素赋值为1。标记为类型2，表示circRNA。
# type_mask[num_miRNA:num_miRNA + num_circRNA] = 1
# # 从num_miRNA + num_circRNA索引到num_miRNA + num_circRNA +  num_lncRNA - 1索引的元素赋值为2。标记为类型3，表示lncRNA。
# type_mask[num_miRNA + num_circRNA:num_miRNA + num_circRNA +  num_lncRNA] = 2
# # 从num_miRNA + num_circRNA +  num_lncRNA索引到num_miRNA + num_circRNA +  num_lncRNA +  num_gene的元素赋值为3。这些元素被标记为类型4，表示基因。
# type_mask[num_miRNA + num_circRNA +  num_lncRNA:num_miRNA + num_circRNA +  num_lncRNA +  num_gene] = 3
# # 从num_miRNA + num_circRNA +  num_lncRNA +  num_gene开始的元素赋值为4。标记为类型5，表示疾病。
# type_mask[num_miRNA + num_circRNA +  num_lncRNA +  num_gene:] = 4
#
# np.save(save_prefix + 'node_types.npy', type_mask)
# del type_mask
#
# # 构建零矩阵 一个长度为dim*dim的二维数组。dtype=int指定了数组的数据类型为整数。
# adjM = np.zeros((dim, dim), dtype=int)

f = open('matrix5.txt', 'r')
all_lines = f.readlines()
f.close()

data = []
for line in all_lines:
    line_data = line.split(',')
    data.append(line_data)

matrix_data = [[float(x) for x in row] for row in data]
adjM = np.array(matrix_data)

# print(1)
# # 有mi2dis关系的两条边都设置为100
# for _, row in miRNA_disease.iterrows():
#     mid = int(row['miRNAid'])
#     did = num_miRNA + num_circRNA +  num_lncRNA +  num_gene + int(row['diseaseid'])
#     adjM[mid, did] = 100
#     adjM[did, mid] = 100
# # 有circ2dis关系的两条边都设置为100
# for _, row in circRNA_disease.iterrows():
#     cid = num_miRNA + int(row['circRNAid'])
#     did = num_miRNA + num_circRNA +  num_lncRNA +  num_gene + int(row['diseaseid'])
#     adjM[cid, did] = 100
#     adjM[did, cid] = 100
# # 有lnc2dis关系的两条边都设置为100
# for _, row in lncRNA_disease.iterrows():
#     lid = num_miRNA + num_circRNA + int(row['lncRNAid'])
#     did = num_miRNA + num_circRNA +  num_lncRNA +  num_gene + int(row['diseaseid'])
#     adjM[lid, did] = 100
#     adjM[did, lid] = 100
# # 有gene2dis关系的两条边都设置为100
# for _, row in gene_disease.iterrows():
#     gid = num_miRNA + num_circRNA + num_lncRNA + int(row['geneid'])
#     did = num_miRNA + num_circRNA + num_lncRNA +  num_gene + int(row['diseaseid'])
#     adjM[gid, did] = 100
#     adjM[did, gid] = 100
# # 有mi2circ关系的两条边都设置为100
# for _, row in miRNA_circRNA.iterrows():
#     mid = int(row['miRNAid'])
#     cid = num_miRNA + int(row['circRNAid'])
#     adjM[mid, cid] = 100
#     adjM[cid, mid] = 100
# # 有mi2lnc关系的两条边都设置为100
# for _, row in miRNA_lncRNA.iterrows():
#     mid = int(row['miRNAid'])
#     lid = num_miRNA + num_circRNA + int(row['lncRNAid'])
#     adjM[mid, lid] = 100
#     adjM[lid, mid] = 100
# # 有mi2gene关系的两条边都设置为100
# for _, row in miRNA_gene.iterrows():
#     mid = int(row['miRNAid'])
#     gid = num_miRNA + num_circRNA + num_lncRNA + int(row['geneid'])
#     adjM[mid, gid] = 100
#     adjM[gid, mid] = 100
# print(2)
# # np.savetxt('matrix1.txt', adjM)
# # 使用pd.concat函数将两个DataFrame miRNA_adjacent和miRNA_Sim按列合并，其中axis=1表示按列合并。
# miRNA_adjacent_Sim = pd.concat([miRNA_adjacent, miRNA_Sim], axis=1)
# # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
# for _, row in miRNA_adjacent_Sim.iterrows():
#     # 根据mid和aid索引，在adjM数组中的相应位置填充值。
#     mid = int(row['miRNAID'])
#     aid = int(row['adjacentID'])
#     # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
#     adjM[mid, aid] = row['similarity'] * 100
# # np.savetxt('matrix2.txt', adjM)
# # 使用pd.concat函数将两个DataFrame circRNA_adjacent和circRNA_Sim按列合并，其中axis=1表示按列合并。
# circRNA_adjacent_Sim = pd.concat([circRNA_adjacent, circRNA_Sim], axis=1)
# # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
# for _, row in circRNA_adjacent_Sim.iterrows():
#     # 根据mid和aid索引，在adjM数组中的相应位置填充值。
#     cid = int(row['circRNAID'])
#     aid = int(row['adjacentID'])
#     # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
#     adjM[cid, aid] = row['similarity'] * 100
# # np.savetxt('matrix3.txt', adjM)
# # 使用pd.concat函数将两个DataFrame lncRNA_adjacent和lncRNA_Sim按列合并，其中axis=1表示按列合并。
# lncRNA_adjacent_Sim = pd.concat([lncRNA_adjacent, lncRNA_Sim], axis=1)
# # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
# for _, row in lncRNA_adjacent_Sim.iterrows():
#     # 根据mid和aid索引，在adjM数组中的相应位置填充值。
#     lid = int(row['lncRNAID'])
#     aid = int(row['adjacentID'])
#     # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
#     adjM[lid, aid] = row['similarity'] * 100
# # np.savetxt('matrix4.csv', adjM, delimiter=',')
# # 使用pd.concat函数将两个DataFrame gene_adjacent和gene_Sim按列合并，其中axis=1表示按列合并。
# gene_adjacent_Sim = pd.concat([gene_adjacent, gene_Sim], axis=1)
# # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
# for _, row in gene_adjacent_Sim.iterrows():
#     # 根据mid和aid索引，在adjM数组中的相应位置填充值。
#     gid = int(row['geneID'])
#     aid = int(row['adjacentID'])
#     # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
#     adjM[gid, aid] = row['similarity'] * 100
# np.savetxt('matrix5.txt', adjM)
# np.savetxt('matrix4.csv', adjM, delimiter=',')
print(3)
# 创建一个名为 miRNA_disease_list 的字典。
# 字典的键是miRNA的索引，值是一个数组，表示与该miRNA相关的疾病的索引。
# 这些索引来自 adjM 矩阵中 i 行、从 num_miRNA 列到 num_miRNA + num_disease - 1 列的非零元素的索引。
# 第i行的          ·                               第num_miRNA列到num_miRNA+num_disease-1列
miRNA_disease_list = {i: adjM[i, num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[0] for i in range(num_miRNA)}
circRNA_disease_list = {i: adjM[num_miRNA + i, num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[0] for i in range(num_circRNA)}
lncRNA_disease_list = {i: adjM[num_miRNA + num_circRNA + i, num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[0] for i in range(num_lncRNA)}
gene_disease_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i, num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[0] for i in range(num_gene)}
miRNA_circRNA_list = {i: adjM[i, num_miRNA : num_miRNA + num_circRNA].nonzero()[0] for i in range(num_circRNA)}
miRNA_lncRNA_list = {i: adjM[i, num_miRNA + num_circRNA : num_miRNA + num_circRNA + num_lncRNA].nonzero()[0] for i in range(num_lncRNA)}
miRNA_gene_list = {i: adjM[i, num_miRNA + num_circRNA + num_lncRNA : num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[0] for i in range(num_gene)}
# 创建一个名为 disease_miRNA_list 的字典。
# 字典的键是疾病的索引，值是一个数组，表示与该疾病相关的miRNA的索引。
# 这些索引来自 adjM 矩阵中 num_miRNA + i 行、从第一列到 num_miRNA - 1 列的非零元素的索引。
# 第num_miRNA + i行的                              第一列到num_miRNA - 1列
disease_miRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, : num_miRNA].nonzero()[0] for i in range(num_disease)}
disease_circRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, num_miRNA : num_miRNA + num_circRNA].nonzero()[0] for i in range(num_disease)}
disease_lncRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, num_miRNA + num_circRNA : num_miRNA + num_circRNA + num_lncRNA].nonzero()[0] for i in range(num_disease)}
disease_gene_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, num_miRNA + num_circRNA + num_lncRNA : num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[0] for i in range(num_disease)}
circRNA_miRNA_list = {i: adjM[num_miRNA + i, : num_miRNA].nonzero()[0] for i in range(num_miRNA)}
lncRNA_miRNA_list = {i: adjM[num_miRNA + num_circRNA + i, : num_miRNA].nonzero()[0] for i in range(num_miRNA)}
gene_miRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i, : num_miRNA].nonzero()[0] for i in range(num_miRNA)}
# 创建一个名为 disease_miRNA_list 的字典。
# 创建一个名为 miRNA_adjacent_list 的字典。
# 字典的键是miRNA的索引，值是一个数组，表示与该miRNA相邻的miRNA的索引。
# 这些索引来自 adjM 矩阵中 i 行、从第一列到 num_miRNA - 1 列的非零元素的索引。
# 第i行的                                          第一列到num_miRNA - 1列
miRNA_adjacent_list = {i: adjM[i, :num_miRNA].nonzero()[0] for i in range(num_miRNA)}
circRNA_adjacent_list = {i: adjM[num_miRNA + i, num_miRNA : num_miRNA + num_circRNA].nonzero()[0] for i in range(num_circRNA)}
lncRNA_adjacent_list = {i: adjM[num_miRNA + num_circRNA + i, num_miRNA + num_circRNA : num_miRNA + num_circRNA + num_lncRNA].nonzero()[0] for i in range(num_lncRNA)}
gene_adjacent_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i, num_miRNA + num_circRNA + num_lncRNA : num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[0] for i in range(num_gene)}
disease_adjacent_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, num_miRNA + num_circRNA + num_lncRNA + num_gene :num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[0] for i in range(num_disease)}

scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
del adjM
#=======================M-M C-C L-L G-G D-D==============================================================
#=======================0-0 1-1 2-2 3-3 4-4==============================================================
# 0-0
# write all things
# 包含两个数组，分别表示 miRNA circRNA lncRNA gene和疾病的索引列表
target_idx_lists = [np.arange(num_miRNA), np.arange(num_circRNA), np.arange(num_lncRNA), np.arange(num_gene), np.arange(num_disease)]
# 包含两个值，分别为 0 和 num_miRNA。这将用于在索引数组中调整索引值。
offset_list = [0, num_miRNA, num_miRNA+num_circRNA, num_miRNA+num_circRNA+num_lncRNA, num_miRNA+num_circRNA+num_lncRNA+num_gene]

print(4)
gc.collect()
# miRNA之间的相邻关系。
m_m = np.array(miRNA_adjacent)
# 使用自定义的排序键对数组 m_m 进行排序。排序键是一个匿名函数，它对每个索引 i 所对应的 miRNA 相邻关系向量进行排序。
sorted_index = sorted(list(range(len(m_m))), key=lambda i: m_m[i].tolist())
# 根据排序索引，重新排列数组 m_m，以确保 miRNA 相邻关系向量按照所期望的顺序排列。
m_m = m_m[sorted_index]
with open(save_prefix + '0/0-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m) and m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m) and m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 1-1
circRNA_adjacent[:] += num_miRNA
c_c = np.array(circRNA_adjacent)
sorted_index = sorted(list(range(len(c_c))), key=lambda i: c_c[i].tolist())
c_c = c_c[sorted_index]

with open(save_prefix + '1/1-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c) and c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c) and c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
# 2-2
lncRNA_adjacent[:] += num_miRNA + num_circRNA
l_l = np.array(lncRNA_adjacent)
sorted_index = sorted(list(range(len(l_l))), key=lambda i: l_l[i].tolist())
l_l = l_l[sorted_index]
gc.collect()
with open(save_prefix + '2/2-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l) and l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l) and l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 3-3
gene_adjacent[:] += num_miRNA + num_circRNA + num_lncRNA
g_g = np.array(gene_adjacent)
sorted_index = sorted(list(range(len(g_g))), key=lambda i: g_g[i].tolist())
g_g = g_g[sorted_index]

with open(save_prefix + '3/3-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g) and g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)
gc.collect()
# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g) and g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 4-4
disease_adjacent[:] += num_miRNA + num_circRNA + num_lncRNA + num_gene
d_d = np.array(disease_adjacent)
sorted_index = sorted(list(range(len(d_d))), key=lambda i: d_d[i].tolist())
d_d = d_d[sorted_index]
with open(save_prefix + '4/4-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d) and d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d) and d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
print(5)
#======M-C-M M-L-M M-G-M M-D-M C-M-C C-D-C L-M-L L-D-L G-M-G G-D-G D-M-D D-C-D D-L-D D-G-D=========
#======0-1-0 0-2-0 0-3-0 0-4-0 1-0-1 1-4-1 2-0-2 2-4-2 3-0-3 3-4-3 4-0-4 4-1-4 4-2-4 4-3-4=========
# 0-1-0
m_c_m = []
for c, m_list in circRNA_miRNA_list.items():
    m_c_m.extend([(m1, c, m2) for m1 in m_list for m2 in m_list])
m_c_m = np.array(m_c_m)
m_c_m[:, 1] += num_miRNA
sorted_index = sorted(list(range(len(m_c_m))), key=lambda i: m_c_m[i, [0, 2, 1]].tolist())
m_c_m = m_c_m[sorted_index]

with open(save_prefix + '0/0-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_m) and m_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_m) and m_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 0-2-0
m_l_m = []
for l, m_list in lncRNA_miRNA_list.items():
    m_l_m.extend([(m1, l, m2) for m1 in m_list for m2 in m_list])
m_l_m = np.array(m_l_m)
m_l_m[:, 1] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_l_m))), key=lambda i: m_l_m[i, [0, 2, 1]].tolist())
m_l_m = m_l_m[sorted_index]

with open(save_prefix + '0/0-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_m) and m_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_m) and m_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 0-3-0
m_g_m = []
for g, m_list in gene_miRNA_list.items():
    m_g_m.extend([(m1, g, m2) for m1 in m_list for m2 in m_list])
m_g_m = np.array(m_g_m)
m_g_m[:, 1] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(m_g_m))), key=lambda i: m_g_m[i, [0, 2, 1]].tolist())
m_g_m = m_g_m[sorted_index]

with open(save_prefix + '0/0-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_m) and m_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_m) and m_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 0-4-0
# 存储疾病与miRNA之间的三元组
m_d_m = []
# 过两个嵌套的循环，将同一个疾病的所有 miRNA 两两组合，从而构建了多个三元组。
for d, m_list in disease_miRNA_list.items():
    m_d_m.extend([(m1, d, m2) for m1 in m_list for m2 in m_list])
# 转换为一个 NumPy 数组
m_d_m = np.array(m_d_m)
# 将数组中所有行的第二列（疾病索引）加上 num_miRNA，
m_d_m[:, 1] += num_miRNA +num_circRNA + num_lncRNA + num_gene
# 使用自定义的排序键对数组 m_d_m 进行排序。排序键是一个匿名函数，它对每个索引 i 所对应的三元组的元素 [0, 2, 1] 进行排序，以确保 m1、d 和 m2 的顺序正确。
sorted_index = sorted(list(range(len(m_d_m))), key=lambda i: m_d_m[i, [0, 2, 1]].tolist())
# 根据排序索引，重新排列数组 m_d_m，以确保三元组按照所期望的顺序排列。
m_d_m = m_d_m[sorted_index]

with open(save_prefix + '0/0-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_m) and m_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_m) and m_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 1-0-1
c_m_c = []
for m ,c_list in miRNA_circRNA_list.items():
    c_m_c.extend([(c1, m, c2) for c1 in c_list for c2 in c_list])
c_m_c = np.array(c_m_c)
c_m_c[:, [0,2]] += num_miRNA
sorted_index = sorted(list(range(len(c_m_c))), key=lambda i: c_m_c[i, [0, 2, 1]].tolist())
c_m_c = c_m_c[sorted_index]

with open(save_prefix + '1/1-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_c) and c_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_c) and c_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
# 1-4-1
c_d_c = []
for d ,c_list in disease_circRNA_list.items():
    c_d_c.extend([(c1, d, c2) for c1 in c_list for c2 in c_list])
c_d_c = np.array(c_d_c)
c_d_c[:, [0,2]] += num_miRNA
c_d_c[:, 1] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_c))), key=lambda i: c_d_c[i, [0, 2, 1]].tolist())
c_d_c = c_d_c[sorted_index]

with open(save_prefix + '1/1-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_c) and c_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_c) and c_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 2-0-2
l_m_l = []
for m ,l_list in miRNA_lncRNA_list.items():
    l_m_l.extend([(l1, m, l2) for l1 in l_list for l2 in l_list])
l_m_l = np.array(l_m_l)
l_m_l[:, [0,2]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(l_m_l))), key=lambda i: l_m_l[i, [0, 2, 1]].tolist())
l_m_l = l_m_l[sorted_index]

with open(save_prefix + '2/2-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_l) and l_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_l) and l_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 2-4-2
l_d_l = []
for d ,l_list in disease_lncRNA_list.items():
    l_d_l.extend([(l1, d, l2) for l1 in l_list for l2 in l_list])
l_d_l = np.array(l_d_l)
l_d_l[:, [0,2]] += num_miRNA + num_circRNA
l_d_l[:, 1] += num_miRNA + num_circRNA +num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_l))), key=lambda i: l_d_l[i, [0, 2, 1]].tolist())
l_d_l = l_d_l[sorted_index]

with open(save_prefix + '2/2-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_l) and l_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_l) and l_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
print(6)
# 3-0-3
g_m_g = []
for m ,g_list in miRNA_gene_list.items():
    g_m_g.extend([(g1, m, g2) for g1 in g_list for g2 in g_list])
g_m_g = np.array(g_m_g)
g_m_g[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(g_m_g))), key=lambda i: g_m_g[i, [0, 2, 1]].tolist())
g_m_g = g_m_g[sorted_index]

with open(save_prefix + '3/3-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_g) and g_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_g) and g_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 3-4-3
g_d_g = []
for d ,g_list in disease_gene_list.items():
    g_d_g.extend([(g1, d, g2) for g1 in g_list for g2 in g_list])
g_d_g = np.array(g_d_g)
g_d_g[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA
g_d_g[:, 1] += num_miRNA + num_circRNA +num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_g))), key=lambda i: g_d_g[i, [0, 2, 1]].tolist())
g_d_g = g_d_g[sorted_index]

with open(save_prefix + '3/3-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_g) and g_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_g) and g_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 4-0-4
d_m_d = []
for m ,d_list in miRNA_disease_list.items():
    d_m_d.extend([(d1, m, d2) for d1 in d_list for d2 in d_list])
d_m_d = np.array(d_m_d)
d_m_d[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_d))), key=lambda i: d_m_d[i, [0, 2, 1]].tolist())
d_m_d = d_m_d[sorted_index]

with open(save_prefix + '4/4-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_d) and d_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_d) and d_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
# 4-1-4
d_c_d = []
for c ,d_list in circRNA_disease_list.items():
    d_c_d.extend([(d1, c, d2) for d1 in d_list for d2 in d_list])
d_c_d = np.array(d_c_d)
d_c_d[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
d_c_d[:, 1] += num_miRNA
sorted_index = sorted(list(range(len(d_c_d))), key=lambda i: d_c_d[i, [0, 2, 1]].tolist())
d_c_d = d_c_d[sorted_index]

with open(save_prefix + '4/4-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_d) and d_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_d) and d_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 4-2-4
d_l_d = []
for l ,d_list in lncRNA_disease_list.items():
    d_l_d.extend([(d1, l, d2) for d1 in d_list for d2 in d_list])
d_l_d = np.array(d_l_d)
d_l_d[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
d_l_d[:, 1] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_l_d))), key=lambda i: d_l_d[i, [0, 2, 1]].tolist())
d_l_d = d_l_d[sorted_index]

with open(save_prefix + '4/4-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_d) and d_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_d) and d_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# 4-3-4
d_g_d = []
for g ,d_list in gene_disease_list.items():
    d_g_d.extend([(d1, g, d2) for d1 in d_list for d2 in d_list])
d_g_d = np.array(d_g_d)
d_g_d[:, [0,2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
d_g_d[:, 1] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_g_d))), key=lambda i: d_g_d[i, [0, 2, 1]].tolist())
d_g_d = d_g_d[sorted_index]

with open(save_prefix + '4/4-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_d) and d_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_d) and d_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-C-C
c_c_c = []
for c ,c_list in circRNA_adjacent_list.items():
    c_c_c.extend([(c1, c, c2) for c1 in c_list for c2 in c_list])
c_c_c = np.array(c_c_c)
c_c_c += num_miRNA
sorted_index = sorted(list(range(len(c_c_c))), key=lambda i: c_c_c[i, [0, 2, 1]].tolist())
c_c_c = c_c_c[sorted_index]

with open(save_prefix + '1/1-1-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c_c) and c_c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_c_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-1-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c_c) and c_c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_c_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-M-M
m_m_m = []
for m ,m_list in miRNA_adjacent_list.items():
    m_m_m.extend([(m1, m, m2) for m1 in m_list for m2 in m_list])
m_m_m = np.array(m_m_m)
sorted_index = sorted(list(range(len(m_m_m))), key=lambda i: m_m_m[i, [0, 2, 1]].tolist())
m_m_m = m_m_m[sorted_index]

with open(save_prefix + '0/0-0-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m_m) and m_m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_m_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-0-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m_m) and m_m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_m_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-L-L
l_l_l = []
for l ,l_list in lncRNA_adjacent_list.items():
    l_l_l.extend([(l1, l, l2) for l1 in l_list for l2 in l_list])
l_l_l = np.array(l_l_l)
l_l_l += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(l_l_l))), key=lambda i: l_l_l[i, [0, 2, 1]].tolist())
l_l_l = l_l_l[sorted_index]

with open(save_prefix + '2/2-2-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l_l) and l_l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_l_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-2-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l_l) and l_l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_l_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-G-G
g_g_g = []
for g ,g_list in gene_adjacent_list.items():
    g_g_g.extend([(g1, g, g2) for g1 in g_list for g2 in g_list])
g_g_g = np.array(g_g_g)
g_g_g += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(g_g_g))), key=lambda i: g_g_g[i, [0, 2, 1]].tolist())
g_g_g = g_g_g[sorted_index]

with open(save_prefix + '3/3-3-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g_g) and g_g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_g_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-3-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g_g) and g_g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_g_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-D-D
d_d_d = []
for d ,d_list in disease_adjacent_list.items():
    d_d_d.extend([(d1, d, d2) for d1 in d_list for d2 in d_list])
d_d_d = np.array(d_d_d)
d_d_d += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_d_d))), key=lambda i: d_d_d[i, [0, 2, 1]].tolist())
d_d_d = d_d_d[sorted_index]

with open(save_prefix + '4/4-4-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d_d) and d_d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_d_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-4-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d_d) and d_d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_d_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
print(7)
# ========M-C-C-M M-L-L-M M-G-G-M M-D-D-M C-M-M-C C-D-D-C L-M-M-L L-D-D-L G-M-M-G G-D-D-G D-M-M-D D-C-C-D D-L-L-D D-G-G-D ##########
# M-C-C-M
m_c_c_m = []
for c1, c2 in c_c:
    m_c_c_m.extend([(m1, c1, c2, m2) for m1 in circRNA_miRNA_list[c1 - num_miRNA] for m2 in circRNA_miRNA_list[c2 - num_miRNA]])
m_c_c_m = np.array(m_c_c_m)
# m_c_c_m[:, [1, 2]] += num_miRNA
sorted_index = sorted(list(range(len(m_c_c_m))), key=lambda i: m_c_c_m[i, [0, 3, 1, 2]].tolist())
m_c_c_m = m_c_c_m[sorted_index]

with open(save_prefix + '0/0-1-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_m) and m_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_m) and m_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_c_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-L-L-M
m_l_l_m = []
for l1, l2 in l_l:
    m_l_l_m.extend([(m1, l1, l2, m2) for m1 in lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA] for m2 in lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA]])
m_l_l_m = np.array(m_l_l_m)
# m_l_l_m[:, [1, 2]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_l_l_m))), key=lambda i: m_l_l_m[i, [0, 3, 1, 2]].tolist())
m_l_l_m = m_l_l_m[sorted_index]

with open(save_prefix + '0/0-2-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_m) and m_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_m) and m_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_l_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-G-G-M
m_g_g_m = []
for g1, g2 in g_g:
    m_g_g_m.extend([(m1, g1, g2, m2) for m1 in gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA] for m2 in gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA]])
m_g_g_m = np.array(m_g_g_m)
# m_g_g_m[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(m_g_g_m))), key=lambda i: m_g_g_m[i, [0, 3, 1, 2]].tolist())
m_g_g_m = m_g_g_m[sorted_index]

with open(save_prefix + '0/0-3-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_m) and m_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_m) and m_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_g_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-D-M
m_d_d_m = []
for d1, d2 in d_d:
    m_d_d_m.extend([(m1, d1, d2, m2) for m1 in disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene] for m2 in disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]])
m_d_d_m = np.array(m_d_d_m)
# m_d_d_m[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_d_d_m))), key=lambda i: m_d_d_m[i, [0, 3, 1, 2]].tolist())
m_d_d_m = m_d_d_m[sorted_index]

with open(save_prefix + '0/0-4-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_m) and m_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_m) and m_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-M-C
c_m_m_c = []
for m1, m2 in m_m:
    c_m_m_c.extend([(c1, m1, m2, c2) for c1 in miRNA_circRNA_list[m1] for c2 in miRNA_circRNA_list[m2]])
c_m_m_c = np.array(c_m_m_c)
c_m_m_c[:, [0, 3]] += num_miRNA
sorted_index = sorted(list(range(len(c_m_m_c))), key=lambda i: c_m_m_c[i, [0, 3, 1, 2]].tolist())
c_m_m_c = c_m_m_c[sorted_index]

with open(save_prefix + '1/1-0-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_c) and c_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_c) and c_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-D-C
c_d_d_c = []
for d1, d2 in d_d:
    c_d_d_c.extend([(c1, d1, d2, c2) for c1 in disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene] for c2 in disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]])
c_d_d_c = np.array(c_d_d_c)
c_d_d_c[:, [0, 3]] += num_miRNA
# c_d_d_c[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_d_c))), key=lambda i: c_d_d_c[i, [0, 3, 1, 2]].tolist())
c_d_d_c = c_d_d_c[sorted_index]

with open(save_prefix + '1/1-4-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_c) and c_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_c) and c_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-M-L
l_m_m_l = []
for m1, m2 in m_m:
    l_m_m_l.extend([(l1, m1, m2, l2) for l1 in miRNA_lncRNA_list[m1] for l2 in miRNA_lncRNA_list[m2]])
l_m_m_l = np.array(l_m_m_l)
l_m_m_l[:, [0, 3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(l_m_m_l))), key=lambda i: l_m_m_l[i, [0, 3, 1, 2]].tolist())
l_m_m_l = l_m_m_l[sorted_index]

with open(save_prefix + '2/2-0-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_l) and l_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_l) and l_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-D-L
l_d_d_l = []
for d1, d2 in d_d:
    l_d_d_l.extend([(l1, d1, d2, l2) for l1 in disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene] for l2 in disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]])
l_d_d_l = np.array(l_d_d_l)
l_d_d_l[:, [0, 3]] += num_miRNA + num_circRNA
# l_d_d_l[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_d_l))), key=lambda i: l_d_d_l[i, [0, 3, 1, 2]].tolist())
l_d_d_l = l_d_d_l[sorted_index]

with open(save_prefix + '2/2-4-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_l) and l_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_l) and l_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-M-G
g_m_m_g = []
for m1, m2 in m_m:
    g_m_m_g.extend([(g1, m1, m2, g2) for g1 in miRNA_gene_list[m1] for g2 in miRNA_gene_list[m2]])
g_m_m_g = np.array(g_m_m_g)
g_m_m_g[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(g_m_m_g))), key=lambda i: g_m_m_g[i, [0, 3, 1, 2]].tolist())
g_m_m_g = g_m_m_g[sorted_index]

with open(save_prefix + '3/3-0-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_g) and g_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_g) and g_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-D-G
g_d_d_g = []
for d1, d2 in d_d:
    g_d_d_g.extend([(g1, d1, d2, g2) for g1 in disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene] for g2 in disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]])
g_d_d_g = np.array(g_d_d_g)
g_d_d_g[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_d_g[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_d_g))), key=lambda i: g_d_d_g[i, [0, 3, 1, 2]].tolist())
g_d_d_g = g_d_d_g[sorted_index]

with open(save_prefix + '3/3-4-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_g) and g_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_g) and g_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-M-D
d_m_m_d = []
for m1, m2 in m_m:
    d_m_m_d.extend([(d1, m1, m2, d2) for d1 in miRNA_disease_list[m1] for d2 in miRNA_disease_list[m2]])
d_m_m_d = np.array(d_m_m_d)
d_m_m_d[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_m_d))), key=lambda i: d_m_m_d[i, [0, 3, 1, 2]].tolist())
d_m_m_d = d_m_m_d[sorted_index]

with open(save_prefix + '4/4-0-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_d) and d_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_d) and d_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-C-C-D
d_c_c_d = []
for c1, c2 in c_c:
    d_c_c_d.extend([(d1, c1, c2, d2) for d1 in circRNA_disease_list[c1 - num_miRNA] for d2 in circRNA_disease_list[c2 - num_miRNA]])
d_c_c_d = np.array(d_c_c_d)
# d_c_c_d[:, [1, 2]] += num_miRNA
d_c_c_d[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_c_c_d))), key=lambda i: d_c_c_d[i, [0, 3, 1, 2]].tolist())
d_c_c_d = d_c_c_d[sorted_index]

with open(save_prefix + '4/4-1-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_d) and d_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_d) and d_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-L-L-D
d_l_l_d = []
for l1, l2 in l_l:
    d_l_l_d.extend([(d1, l1, l2, d2) for d1 in lncRNA_disease_list[l1 - num_miRNA - num_circRNA] for d2 in lncRNA_disease_list[l2 - num_miRNA - num_circRNA]])
d_l_l_d = np.array(d_l_l_d)
# d_l_l_d[:, [1, 2]] += num_miRNA + num_circRNA
d_l_l_d[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_l_l_d))), key=lambda i: d_l_l_d[i, [0, 3, 1, 2]].tolist())
d_l_l_d = d_l_l_d[sorted_index]

with open(save_prefix + '4/4-2-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_d) and d_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_d) and d_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-G-G-D
d_g_g_d = []
for g1, g2 in g_g:
    d_g_g_d.extend([(d1, g1, g2, d2) for d1 in gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA] for d2 in gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA]])
d_g_g_d = np.array(d_g_g_d)
# d_g_g_d[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA
d_g_g_d[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_g_g_d))), key=lambda i: d_g_g_d[i, [0, 3, 1, 2]].tolist())
d_g_g_d = d_g_g_d[sorted_index]

with open(save_prefix + '4/4-3-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_d) and d_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_d) and d_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(8)
# M-M-M-M
m_m_m_m = []
for m3, m4 in m_m:
    m_m_m_m.extend([(m1, m3, m4, m2) for m1 in miRNA_adjacent_list[m3] for m2 in miRNA_adjacent_list[m4]])
m_m_m_m = np.array(m_m_m_m)
sorted_index = sorted(list(range(len(m_m_m_m))), key=lambda i: m_m_m_m[i, [0, 3, 1, 2]].tolist())
m_m_m_m = m_m_m_m[sorted_index]

with open(save_prefix + '0/0-0-0-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m_m_m) and m_m_m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_m_m_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-0-0-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_m_m_m) and m_m_m_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_m_m_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-C-C-C
c_c_c_c = []
for c3, c4 in c_c:
    c_c_c_c.extend([(c1, c3, c4, c2) for c1 in circRNA_adjacent_list[c3 - num_miRNA] for c2 in circRNA_adjacent_list[c4 - num_miRNA]])
c_c_c_c = np.array(c_c_c_c)
# c_c_c_c += num_miRNA
c_c_c_c[:, [0, 3]] += num_miRNA
sorted_index = sorted(list(range(len(c_c_c_c))), key=lambda i: c_c_c_c[i, [0, 3, 1, 2]].tolist())
c_c_c_c = c_c_c_c[sorted_index]

with open(save_prefix + '1/1-1-1-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c_c_c) and c_c_c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_c_c_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-1-1-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_c_c_c) and c_c_c_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_c_c_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-L-L-L
l_l_l_l = []
for l3, l4 in l_l:
    l_l_l_l.extend([(l1, l3, l4, l2) for l1 in lncRNA_adjacent_list[l3 - num_miRNA - num_circRNA] for l2 in lncRNA_adjacent_list[l4 - num_miRNA - num_circRNA]])
l_l_l_l = np.array(l_l_l_l)
l_l_l_l[:, [0, 3]] += num_miRNA + num_circRNA
# l_l_l_l[:, [1, 2]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(l_l_l_l))), key=lambda i: l_l_l_l[i, [0, 3, 1, 2]].tolist())
l_l_l_l = l_l_l_l[sorted_index]

with open(save_prefix + '2/2-2-2-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l_l_l) and l_l_l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_l_l_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-2-2-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_l_l_l) and l_l_l_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_l_l_l[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-G-G-G
g_g_g_g = []
for g3, g4 in g_g:
    g_g_g_g.extend([(g1, g3, g4, g2) for g1 in gene_adjacent_list[g3 - num_miRNA - num_circRNA - num_lncRNA] for g2 in gene_adjacent_list[g4 - num_miRNA - num_circRNA - num_lncRNA]])
g_g_g_g = np.array(g_g_g_g)
g_g_g_g[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA
# g_g_g_g[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(g_g_g_g))), key=lambda i: g_g_g_g[i, [0, 3, 1, 2]].tolist())
g_g_g_g = g_g_g_g[sorted_index]

with open(save_prefix + '3/3-3-3-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g_g_g) and g_g_g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_g_g_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-3-3-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_g_g_g) and g_g_g_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_g_g_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-D-D-D
d_d_d_d = []
for d3, d4 in d_d:
    d_d_d_d.extend([(d1, d3, d4, d2) for d1 in disease_adjacent_list[d3 - num_miRNA - num_circRNA - num_lncRNA - num_gene] for d2 in disease_adjacent_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]])
d_d_d_d = np.array(d_d_d_d)
d_d_d_d[:, [0, 3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_d_d_d[:, [1, 2]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_d_d_d))), key=lambda i: d_d_d_d[i, [0, 3, 1, 2]].tolist())
d_d_d_d = d_d_d_d[sorted_index]

with open(save_prefix + '4/4-4-4-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d_d_d) and d_d_d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_d_d_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-4-4-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_d_d_d) and d_d_d_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_d_d_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
print(9)
# # 4-0-0-4
# # 存储 疾病-miRNA-miRNA-疾病 4⃣️元组。
# d_m_m_d = []
# # 循环遍历miRNA与miRNA的关联三元组 m_m 中的每个三元组 (m1, m2)：
# for m1, m2 in m_m:
#     # 以两个miRNA中选出与其有联系的疾病
#     d_m_m_d.extend([(d1, m1, m2, d2) for d1 in miRNA_disease_list[m1] for d2 in miRNA_disease_list[m2]])
# d_m_m_d = np.array(d_m_m_d)
# # 将疾病的序号加上前面的，区别疾病与miRNA两者
# d_m_m_d[:, [0, 3]] += num_miRNA
# sorted_index = sorted(list(range(len(d_m_m_d))), key=lambda i: d_m_m_d[i, [0, 3, 1, 2]].tolist())
# d_m_m_d = d_m_m_d[sorted_index]

# ========================================五元组==============================================#
# M-C-C-C-M
m_c_c_c_m = []
for c1, c, c2 in c_c_c:
    if len(circRNA_miRNA_list[c1 - num_miRNA]) == 0 or len(circRNA_miRNA_list[c2 - num_miRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(circRNA_miRNA_list[c1 - num_miRNA]), int(0.5 * len(circRNA_miRNA_list[c1 - num_miRNA])),replace=False)
    candidate_m1_list = circRNA_miRNA_list[c1 - num_miRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(circRNA_miRNA_list[c2 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c2 - num_miRNA])),replace=False)
    candidate_m2_list = circRNA_miRNA_list[c2 - num_miRNA][candidate_m2_list]
    m_c_c_c_m.extend([(m1, c1, c, c2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_c_c_c_m = np.array(m_c_c_c_m)
m_c_c_c_m[:, [1,2,3]] += num_miRNA
sorted_index = sorted(list(range(len(m_c_c_c_m))), key=lambda i: m_c_c_c_m[i, [0, 4, 1, 2, 3]].tolist())
m_c_c_c_m = m_c_c_c_m[sorted_index]

with open(save_prefix + '0/0-1-1-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_c_m) and m_c_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_c_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-1-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_c_m) and m_c_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_c_c_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-C-D-C-M
m_c_d_c_m = []
for c1, d, c2 in c_d_c:
    if len(circRNA_miRNA_list[c1 - num_miRNA]) == 0 or len(circRNA_miRNA_list[c2 - num_miRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(circRNA_miRNA_list[c1 - num_miRNA]), int(0.5 * len(circRNA_miRNA_list[c1 - num_miRNA])),replace=False)
    candidate_m1_list = circRNA_miRNA_list[c1 - num_miRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(circRNA_miRNA_list[c2 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c2 - num_miRNA])),replace=False)
    candidate_m2_list = circRNA_miRNA_list[c2 - num_miRNA][candidate_m2_list]
    m_c_d_c_m.extend([(m1, c1, d, c2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_c_d_c_m = np.array(m_c_d_c_m)
# m_c_d_c_m[:, [1,3]] += num_miRNA
# m_c_d_c_m[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_c_d_c_m))), key=lambda i: m_c_d_c_m[i, [0, 4, 1, 2, 3]].tolist())
m_c_d_c_m = m_c_d_c_m[sorted_index]

with open(save_prefix + '0/0-1-4-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_d_c_m) and m_c_d_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_d_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-4-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_d_c_m) and m_c_d_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_d_c_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-L-L-L-M
m_l_l_l_m = []
for l1, l, l2 in l_l_l:
    if len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]), int(0.5 * len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_m1_list = lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA])),replace=False)
    candidate_m2_list = lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA][candidate_m2_list]
    m_l_l_l_m.extend([(m1, l1, l, l2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_l_l_l_m = np.array(m_l_l_l_m)
# m_l_l_l_m[:, [1,2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_l_l_l_m))), key=lambda i: m_l_l_l_m[i, [0, 4, 1, 2, 3]].tolist())
m_l_l_l_m = m_l_l_l_m[sorted_index]

with open(save_prefix + '0/0-2-2-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_l_m) and m_l_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_l_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-2-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_l_m) and m_l_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_l_l_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-L-D-L-M
m_l_d_l_m = []
for l1, d, l2 in l_d_l:
    if len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]), int(0.5 * len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_m1_list = lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA])),replace=False)
    candidate_m2_list = lncRNA_miRNA_list[l2 - num_miRNA - num_circRNA][candidate_m2_list]
    m_l_d_l_m.extend([(m1, l1, d, l2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_l_d_l_m = np.array(m_l_d_l_m)
# m_l_d_l_m[:, [1,3]] += num_miRNA + num_circRNA
# m_l_d_l_m[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_l_d_l_m))), key=lambda i: m_l_d_l_m[i, [0, 4, 1, 2, 3]].tolist())
m_l_d_l_m = m_l_d_l_m[sorted_index]

with open(save_prefix + '0/0-2-4-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_d_l_m) and m_l_d_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_d_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-4-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_d_l_m) and m_l_d_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_d_l_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(10)
# M-G-G-G-M
m_g_g_g_m = []
for g1, g, g2 in g_g_g:
    if len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]), int(0.5 * len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m1_list = gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m2_list = gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA][candidate_m2_list]
    m_g_g_g_m.extend([(m1, g1, g, g2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_g_g_g_m = np.array(m_g_g_g_m)
# m_g_g_g_m[:, [1,2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(m_g_g_g_m))), key=lambda i: m_g_g_g_m[i, [0, 4, 1, 2, 3]].tolist())
m_g_g_g_m = m_g_g_g_m[sorted_index]

with open(save_prefix + '0/0-3-3-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_g_m) and m_g_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_g_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-3-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_g_m) and m_g_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_g_g_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-G-D-G-M
m_g_d_g_m = []
for g1, d, g2 in g_d_g:
    if len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]), int(0.5 * len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m1_list = gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m2_list = gene_miRNA_list[g2 - num_miRNA - num_circRNA - num_lncRNA][candidate_m2_list]
    m_g_d_g_m.extend([(m1, g1, d, g2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_g_d_g_m = np.array(m_g_d_g_m)
# m_g_d_g_m[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA
# m_g_d_g_m[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_g_d_g_m))), key=lambda i: m_g_d_g_m[i, [0, 4, 1, 2, 3]].tolist())
m_g_d_g_m = m_g_d_g_m[sorted_index]

with open(save_prefix + '0/0-3-4-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_d_g_m) and m_g_d_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_d_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-4-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_d_g_m) and m_g_d_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_d_g_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-C-D-M
m_d_c_d_m = []
for d1, c, d2 in d_c_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_c_d_m.extend([(m1, d1, c, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_c_d_m = np.array(m_d_c_d_m)
# m_d_c_d_m[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_c_d_m[:, 2] += num_miRNA
sorted_index = sorted(list(range(len(m_d_c_d_m))), key=lambda i: m_d_c_d_m[i, [0, 4, 1, 2, 3]].tolist())
m_d_c_d_m = m_d_c_d_m[sorted_index]

with open(save_prefix + '0/0-4-1-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_c_d_m) and m_d_c_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_c_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-1-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_c_d_m) and m_d_c_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_c_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-L-D-M
m_d_l_d_m = []
for d1, l, d2 in d_l_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_l_d_m.extend([(m1, d1, l, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_l_d_m = np.array(m_d_l_d_m)
# m_d_l_d_m[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_l_d_m[:, 2] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_d_l_d_m))), key=lambda i: m_d_l_d_m[i, [0, 4, 1, 2, 3]].tolist())
m_d_l_d_m = m_d_l_d_m[sorted_index]

with open(save_prefix + '0/0-4-2-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_l_d_m) and m_d_l_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_l_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-2-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_l_d_m) and m_d_l_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_l_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-G-D-M
# 0-1-2-1-0
# 存储 miRNA-基因-疾病-基因-miRNA 五元组。
m_d_g_d_m = []
# 循环遍历基因与疾病的关联三元组 d_g_d 中的每个三元组 (d1, g, d2)：
for d1, g, d2 in d_g_d:
    # 检查与d1疾病和d2疾病相关的miRNA是否为空列表。如果任一列表为空，就跳过当前循环迭代。
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    # 从disease_miRNA_list[d1 - num_miRNA]中随机选择一半的miRNA（candidate_m1_list）和从disease_miRNA_list[d2 - num_miRNA]
    # 中随机选择一半的miRNA（candidate_m2_list）。这里的采样率是0.5。
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])), replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])), replace=False)
    candidate_m2_list = disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    # 将两个随机选择的miRNA列表组合为一个miRNA - 基因 - 疾病 - 基因 - miRNA五元组，并将这些五元组添加到m_d_g_d_m列表中。
    m_d_g_d_m.extend([(m1, d1, g, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
# 转换为一个 NumPy 数组
m_d_g_d_m = np.array(m_d_g_d_m)
# m_d_g_d_m[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_g_d_m[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# 所对应的五元组的元素 [0, 4, 1, 2, 3] 进行排序。
sorted_index = sorted(list(range(len(m_d_g_d_m))), key=lambda i: m_d_g_d_m[i, [0, 4, 1, 2, 3]].tolist())
# 根据排序索引，重新排列数组 m_d_g_d_m，以确保五元组按照所期望的顺序排列。
m_d_g_d_m = m_d_g_d_m[sorted_index]

with open(save_prefix + '0/0-4-3-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_g_d_m) and m_d_g_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_g_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-3-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_g_d_m) and m_d_g_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_g_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(11)
# M-D-D-D-M
m_d_d_d_m = []
for d1, d, d2 in d_d_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_d_d_m.extend([(m1, d1, d, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_d_d_m = np.array(m_d_d_d_m)
# m_d_d_d_m[:, [1,2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_d_d_d_m))), key=lambda i: m_d_d_d_m[i, [0, 4, 1, 2, 3]].tolist())
m_d_d_d_m = m_d_d_d_m[sorted_index]

with open(save_prefix + '0/0-4-4-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_d_m) and m_d_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_d_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-4-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_d_m) and m_d_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_d_d_m[left:right, -1] - offset_list[0]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-M-M-C
c_m_m_m_c = []
for m1, m, m2 in m_m_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m2]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]), int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m2]),int(0.5 * len(miRNA_circRNA_list[m2])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m2][candidate_c2_list]
    c_m_m_m_c.extend([(c1, m1, m, m2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_m_m_c = np.array(c_m_m_m_c)
c_m_m_m_c[:, [0,4]] += num_miRNA
sorted_index = sorted(list(range(len(c_m_m_m_c))), key=lambda i: c_m_m_m_c[i, [0, 4, 1, 2, 3]].tolist())
c_m_m_m_c = c_m_m_m_c[sorted_index]

with open(save_prefix + '1/1-0-0-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_m_c) and c_m_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_m_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-0-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_m_c) and c_m_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_m_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-L-M-C
c_m_l_m_c = []
for m1, l, m2 in m_l_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m2]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]), int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m2]),int(0.5 * len(miRNA_circRNA_list[m2])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m2][candidate_c2_list]
    c_m_l_m_c.extend([(c1, m1, l, m2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_l_m_c = np.array(c_m_l_m_c)
c_m_l_m_c[:, [0,4]] += num_miRNA
# c_m_l_m_c[:, 2] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(c_m_l_m_c))), key=lambda i: c_m_l_m_c[i, [0, 4, 1, 2, 3]].tolist())
c_m_l_m_c = c_m_l_m_c[sorted_index]

with open(save_prefix + '1/1-0-2-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_l_m_c) and c_m_l_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_l_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-2-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_l_m_c) and c_m_l_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_l_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-G-M-C
c_m_g_m_c = []
for m1, g, m2 in m_g_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m2]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]), int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m2]),int(0.5 * len(miRNA_circRNA_list[m2])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m2][candidate_c2_list]
    c_m_g_m_c.extend([(c1, m1, g, m2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_g_m_c = np.array(c_m_g_m_c)
c_m_g_m_c[:, [0,4]] += num_miRNA
# c_m_g_m_c[:, 2] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(c_m_g_m_c))), key=lambda i: c_m_g_m_c[i, [0, 4, 1, 2, 3]].tolist())
c_m_g_m_c = c_m_g_m_c[sorted_index]

with open(save_prefix + '1/1-0-3-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_g_m_c) and c_m_g_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_g_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-3-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_g_m_c) and c_m_g_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_g_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-D-M-C
c_m_d_m_c = []
for m1, d, m2 in m_d_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m2]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]), int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m2]),int(0.5 * len(miRNA_circRNA_list[m2])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m2][candidate_c2_list]
    c_m_d_m_c.extend([(c1, m1, d, m2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_d_m_c = np.array(c_m_d_m_c)
c_m_d_m_c[:, [0,4]] += num_miRNA
# c_m_d_m_c[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_m_d_m_c))), key=lambda i: c_m_d_m_c[i, [0, 4, 1, 2, 3]].tolist())
c_m_d_m_c = c_m_d_m_c[sorted_index]

with open(save_prefix + '1/1-0-4-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_m_c) and c_m_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_d_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-4-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_m_c) and c_m_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_d_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-M-D-C
c_d_m_d_c = []
for d1, m, d2 in d_m_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_m_d_c.extend([(c1, d1, m, d2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_m_d_c = np.array(c_d_m_d_c)
c_d_m_d_c[:, [0,4]] += num_miRNA
# c_d_m_d_c[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_m_d_c))), key=lambda i: c_d_m_d_c[i, [0, 4, 1, 2, 3]].tolist())
c_d_m_d_c = c_d_m_d_c[sorted_index]

with open(save_prefix + '1/1-4-0-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_m_d_c) and c_d_m_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_m_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-0-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_m_d_c) and c_d_m_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_m_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-L-D-C
c_d_l_d_c = []
for d1, l, d2 in d_l_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_l_d_c.extend([(c1, d1, l, d2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_l_d_c = np.array(c_d_l_d_c)
c_d_l_d_c[:, [0,4]] += num_miRNA
# c_d_l_d_c[:, 2] += num_miRNA + num_circRNA
# c_d_l_d_c[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_l_d_c))), key=lambda i: c_d_l_d_c[i, [0, 4, 1, 2, 3]].tolist())
c_d_l_d_c = c_d_l_d_c[sorted_index]

with open(save_prefix + '1/1-4-2-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_l_d_c) and c_d_l_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_l_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-2-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_l_d_c) and c_d_l_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_l_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-G-D-C
c_d_g_d_c = []
for d1, g, d2 in d_l_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_g_d_c.extend([(c1, d1, g, d2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_g_d_c = np.array(c_d_g_d_c)
c_d_g_d_c[:, [0,4]] += num_miRNA
# c_d_g_d_c[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# c_d_g_d_c[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_g_d_c))), key=lambda i: c_d_g_d_c[i, [0, 4, 1, 2, 3]].tolist())
c_d_g_d_c = c_d_g_d_c[sorted_index]

with open(save_prefix + '1/1-4-3-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_g_d_c) and c_d_g_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_g_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-3-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_g_d_c) and c_d_g_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_g_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(12)
# C-D-D-D-C
c_d_d_d_c = []
for d1, d, d2 in d_d_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_d_d_c.extend([(c1, d1, d, d2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_d_d_c = np.array(c_d_d_d_c)
c_d_d_d_c[:, [0,4]] += num_miRNA
# c_d_d_d_c[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# c_d_d_d_c[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_d_d_c))), key=lambda i: c_d_d_d_c[i, [0, 4, 1, 2, 3]].tolist())
c_d_d_d_c = c_d_d_d_c[sorted_index]

with open(save_prefix + '1/1-4-4-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_d_c) and c_d_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_d_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-4-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_d_c) and c_d_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_d_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-M-M-L
l_m_m_m_l = []
for m1, m, m2 in m_m_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m2]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]), int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m2]),int(0.5 * len(miRNA_lncRNA_list[m2])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m2][candidate_l2_list]
    l_m_m_m_l.extend([(l1, m1, m, m2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_m_m_l = np.array(l_m_m_m_l)
l_m_m_m_l[:, [0,4]] += num_miRNA + num_circRNA
# l_m_m_m_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_m_m_m_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_m_m_m_l))), key=lambda i: l_m_m_m_l[i, [0, 4, 1, 2, 3]].tolist())
l_m_m_m_l = l_m_m_m_l[sorted_index]

with open(save_prefix + '2/2-0-0-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_m_l) and l_m_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_m_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-0-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_m_l) and l_m_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_m_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-C-M-L
l_m_c_m_l = []
for m1, c, m2 in m_c_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m2]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]), int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m2]),int(0.5 * len(miRNA_lncRNA_list[m2])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m2][candidate_l2_list]
    l_m_c_m_l.extend([(l1, m1, c, m2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_c_m_l = np.array(l_m_c_m_l)
l_m_c_m_l[:, [0,4]] += num_miRNA + num_circRNA
# l_m_c_m_l[:, 2] += num_miRNA
# l_m_c_m_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_m_c_m_l))), key=lambda i: l_m_c_m_l[i, [0, 4, 1, 2, 3]].tolist())
l_m_c_m_l = l_m_c_m_l[sorted_index]

with open(save_prefix + '2/2-0-1-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_c_m_l) and l_m_c_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_c_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-1-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_c_m_l) and l_m_c_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_c_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-G-M-L
l_m_g_m_l = []
for m1, g, m2 in m_g_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m2]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]), int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m2]),int(0.5 * len(miRNA_lncRNA_list[m2])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m2][candidate_l2_list]
    l_m_g_m_l.extend([(l1, m1, g, m2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_g_m_l = np.array(l_m_g_m_l)
l_m_g_m_l[:, [0,4]] += num_miRNA + num_circRNA
# l_m_g_m_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# l_m_g_m_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_m_g_m_l))), key=lambda i: l_m_g_m_l[i, [0, 4, 1, 2, 3]].tolist())
l_m_g_m_l = l_m_g_m_l[sorted_index]

with open(save_prefix + '2/2-0-3-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_g_m_l) and l_m_g_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_g_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-3-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_g_m_l) and l_m_g_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_g_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-D-M-L
l_m_d_m_l = []
for m1, d, m2 in m_d_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m2]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]), int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m2]),int(0.5 * len(miRNA_lncRNA_list[m2])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m2][candidate_l2_list]
    l_m_d_m_l.extend([(l1, m1, d, m2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_d_m_l = np.array(l_m_d_m_l)
l_m_d_m_l[:, [0,4]] += num_miRNA + num_circRNA
# l_m_d_m_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_m_d_m_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_m_d_m_l))), key=lambda i: l_m_d_m_l[i, [0, 4, 1, 2, 3]].tolist())
l_m_d_m_l = l_m_d_m_l[sorted_index]

with open(save_prefix + '2/2-0-4-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_d_m_l) and l_m_d_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_d_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-4-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_d_m_l) and l_m_d_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_d_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-M-D-L
l_d_m_d_l = []
for d1, m, d2 in d_m_d:
    if len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l2_list]
    l_d_m_d_l.extend([(l1, d1, m, d2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_m_d_l = np.array(l_d_m_d_l)
l_d_m_d_l[:, [0,4]] += num_miRNA + num_circRNA
# l_d_m_d_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_d_m_d_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_m_d_l))), key=lambda i: l_d_m_d_l[i, [0, 4, 1, 2, 3]].tolist())
l_d_m_d_l = l_d_m_d_l[sorted_index]

with open(save_prefix + '2/2-4-0-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_m_d_l) and l_d_m_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_m_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-0-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_m_d_l) and l_d_m_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_m_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-C-D-L
l_d_c_d_l = []
for d1, c, d2 in d_c_d:
    if len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l2_list]
    l_d_c_d_l.extend([(l1, d1, c, d2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_c_d_l = np.array(l_d_c_d_l)
l_d_c_d_l[:, [0,4]] += num_miRNA + num_circRNA
# l_d_c_d_l[:, 2] += num_miRNA
# l_d_c_d_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_c_d_l))), key=lambda i: l_d_c_d_l[i, [0, 4, 1, 2, 3]].tolist())
l_d_c_d_l = l_d_c_d_l[sorted_index]

with open(save_prefix + '2/2-4-1-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_c_d_l) and l_d_c_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_c_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-1-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_c_d_l) and l_d_c_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_c_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right
gc.collect()
# L-D-G-D-L
l_d_g_d_l = []
for d1, g, d2 in d_g_d:
    if len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l2_list]
    l_d_g_d_l.extend([(l1, d1, g, d2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_g_d_l = np.array(l_d_g_d_l)
l_d_g_d_l[:, [0,4]] += num_miRNA + num_circRNA
# l_d_g_d_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# l_d_g_d_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_g_d_l))), key=lambda i: l_d_g_d_l[i, [0, 4, 1, 2, 3]].tolist())
l_d_g_d_l = l_d_g_d_l[sorted_index]

with open(save_prefix + '2/2-4-3-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_g_d_l) and l_d_g_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_g_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-3-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_g_d_l) and l_d_g_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_g_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-D-D-L
l_d_d_d_l = []
for d1, d, d2 in d_d_d:
    if len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_l2_list]
    l_d_d_d_l.extend([(l1, d1, d, d2, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_d_d_l = np.array(l_d_d_d_l)
l_d_d_d_l[:, [0,4]] += num_miRNA + num_circRNA
# l_d_d_d_l[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_d_d_d_l[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_d_d_l))), key=lambda i: l_d_d_d_l[i, [0, 4, 1, 2, 3]].tolist())
l_d_d_d_l = l_d_d_d_l[sorted_index]

with open(save_prefix + '2/2-4-4-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_d_l) and l_d_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_d_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-4-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_d_l) and l_d_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_d_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-M-M-G
g_m_m_m_g = []
for m1, m, m2 in m_m_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m2]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]), int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m2]),int(0.5 * len(miRNA_gene_list[m2])),replace=False)
    candidate_g2_list = miRNA_gene_list[m2][candidate_g2_list]
    g_m_m_m_g.extend([(g1, m1, m, m2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_m_m_g = np.array(g_m_m_m_g)
g_m_m_m_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_m_m_g[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_m_m_m_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_m_m_m_g))), key=lambda i: g_m_m_m_g[i, [0, 4, 1, 2, 3]].tolist())
g_m_m_m_g = g_m_m_m_g[sorted_index]

with open(save_prefix + '3/3-0-0-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_m_g) and g_m_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_m_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-0-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_m_g) and g_m_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_m_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-C-M-G
g_m_c_m_g = []
for m1, c, m2 in m_c_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m2]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]), int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m2]),int(0.5 * len(miRNA_gene_list[m2])),replace=False)
    candidate_g2_list = miRNA_gene_list[m2][candidate_g2_list]
    g_m_c_m_g.extend([(g1, m1, c, m2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_c_m_g = np.array(g_m_c_m_g)
g_m_c_m_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_c_m_g[:, 2] += num_miRNA
# g_m_c_m_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_m_c_m_g))), key=lambda i: g_m_c_m_g[i, [0, 4, 1, 2, 3]].tolist())
g_m_c_m_g = g_m_c_m_g[sorted_index]

with open(save_prefix + '3/3-0-1-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_c_m_g) and g_m_c_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_c_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-1-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_c_m_g) and g_m_c_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_c_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


# G-M-L-M-G
g_m_l_m_g = []
for m1, l, m2 in m_l_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m2]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]), int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m2]),int(0.5 * len(miRNA_gene_list[m2])),replace=False)
    candidate_g2_list = miRNA_gene_list[m2][candidate_g2_list]
    g_m_l_m_g.extend([(g1, m1, l, m2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_l_m_g = np.array(g_m_l_m_g)
g_m_l_m_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_l_m_g[:, 2] += num_miRNA + num_circRNA
# g_m_l_m_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_m_l_m_g))), key=lambda i: g_m_l_m_g[i, [0, 4, 1, 2, 3]].tolist())
g_m_l_m_g = g_m_l_m_g[sorted_index]

with open(save_prefix + '3/3-0-2-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_l_m_g) and g_m_l_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_l_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-2-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_l_m_g) and g_m_l_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_l_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-D-M-G
g_m_d_m_g = []
for m1, d, m2 in m_d_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m2]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]), int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m2]),int(0.5 * len(miRNA_gene_list[m2])),replace=False)
    candidate_g2_list = miRNA_gene_list[m2][candidate_g2_list]
    g_m_d_m_g.extend([(g1, m1, d, m2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_d_m_g = np.array(g_m_d_m_g)
g_m_d_m_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_d_m_g[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_m_l_m_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_m_d_m_g))), key=lambda i: g_m_d_m_g[i, [0, 4, 1, 2, 3]].tolist())
g_m_d_m_g = g_m_d_m_g[sorted_index]

with open(save_prefix + '3/3-0-4-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_d_m_g) and g_m_d_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_d_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-4-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_d_m_g) and g_m_d_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_d_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-M-D-G
g_d_m_d_g = []
for d1, m, d2 in d_m_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_m_d_g.extend([(g1, d1, m, d2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_m_d_g = np.array(g_d_m_d_g)
g_d_m_d_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_m_d_g[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_d_m_d_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_m_d_g))), key=lambda i: g_d_m_d_g[i, [0, 4, 1, 2, 3]].tolist())
g_d_m_d_g = g_d_m_d_g[sorted_index]

with open(save_prefix + '3/3-4-0-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_m_d_g) and g_d_m_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_m_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-0-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_m_d_g) and g_d_m_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_m_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-C-D-G
g_d_c_d_g = []
for d1, c, d2 in d_c_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_c_d_g.extend([(g1, d1, c, d2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_c_d_g = np.array(g_d_c_d_g)
g_d_c_d_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_c_d_g[:, 2] += num_miRNA
# g_d_c_d_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_c_d_g))), key=lambda i: g_d_c_d_g[i, [0, 4, 1, 2, 3]].tolist())
g_d_c_d_g = g_d_c_d_g[sorted_index]

with open(save_prefix + '3/3-4-1-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_c_d_g) and g_d_c_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_c_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-1-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_c_d_g) and g_d_c_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_c_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-L-D-G
g_d_l_d_g = []
for d1, l, d2 in d_l_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_l_d_g.extend([(g1, d1, l, d2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_l_d_g = np.array(g_d_l_d_g)
g_d_l_d_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_l_d_g[:, 2] += num_miRNA + num_circRNA
# g_d_l_d_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_l_d_g))), key=lambda i: g_d_l_d_g[i, [0, 4, 1, 2, 3]].tolist())
g_d_l_d_g = g_d_l_d_g[sorted_index]

with open(save_prefix + '3/3-4-2-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_l_d_g) and g_d_l_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_l_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-2-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_l_d_g) and g_d_l_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_l_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(13)
# G-D-D-D-G
g_d_d_d_g = []
for d1, d, d2 in d_d_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]), int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d2 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_d_d_g.extend([(g1, d1, d, d2, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_d_d_g = np.array(g_d_d_d_g)
g_d_d_d_g[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_d_d_g[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_d_d_d_g[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_d_d_g))), key=lambda i: g_d_d_d_g[i, [0, 4, 1, 2, 3]].tolist())
g_d_d_d_g = g_d_d_d_g[sorted_index]

with open(save_prefix + '3/3-4-4-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_d_g) and g_d_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_d_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-4-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_d_g) and g_d_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_d_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-M-M-D
d_m_m_m_d = []
for m1, m, m2 in m_m_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m2]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]), int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m2]),int(0.5 * len(miRNA_disease_list[m2])),replace=False)
    candidate_d2_list = miRNA_disease_list[m2][candidate_d2_list]
    d_m_m_m_d.extend([(d1, m1, m, m2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_m_m_d = np.array(d_m_m_m_d)
d_m_m_m_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_m_m_d[:, 2] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_m_m_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_m_m_d))), key=lambda i: d_m_m_m_d[i, [0, 4, 1, 2, 3]].tolist())
d_m_m_m_d = d_m_m_m_d[sorted_index]

with open(save_prefix + '4/4-0-0-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_m_d) and d_m_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_m_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-0-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_m_d) and d_m_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_m_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-C-M-D
d_m_c_m_d = []
for m1, c, m2 in m_c_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m2]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]), int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m2]),int(0.5 * len(miRNA_disease_list[m2])),replace=False)
    candidate_d2_list = miRNA_disease_list[m2][candidate_d2_list]
    g_m_c_m_g.extend([(d1, m1, c, m2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_c_m_d = np.array(d_m_c_m_d)
d_m_c_m_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_c_m_d[:, 2] += num_miRNA
# d_m_c_m_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_c_m_d))), key=lambda i: d_m_c_m_d[i, [0, 4, 1, 2, 3]].tolist())
d_m_c_m_d = d_m_c_m_d[sorted_index]

with open(save_prefix + '4/4-0-1-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_c_m_d) and d_m_c_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_c_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-1-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_c_m_d) and d_m_c_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_c_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-L-M-D
d_m_l_m_d = []
for m1, l, m2 in m_l_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m2]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]), int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m2]),int(0.5 * len(miRNA_disease_list[m2])),replace=False)
    candidate_d2_list = miRNA_disease_list[m2][candidate_d2_list]
    d_m_l_m_d.extend([(d1, m1, l, m2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_l_m_d = np.array(d_m_l_m_d)
d_m_l_m_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_l_m_d[:, 2] += num_miRNA + num_circRNA
# d_m_c_m_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_l_m_d))), key=lambda i: d_m_l_m_d[i, [0, 4, 1, 2, 3]].tolist())
d_m_l_m_d = d_m_l_m_d[sorted_index]

with open(save_prefix + '4/4-0-2-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_l_m_d) and d_m_l_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_l_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-2-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_l_m_d) and d_m_l_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_l_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-G-M-D
d_m_g_m_d = []
for m1, g, m2 in m_g_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m2]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]), int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m2]),int(0.5 * len(miRNA_disease_list[m2])),replace=False)
    candidate_d2_list = miRNA_disease_list[m2][candidate_d2_list]
    d_m_g_m_d.extend([(d1, m1, g, m2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_g_m_d = np.array(d_m_g_m_d)
d_m_g_m_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_g_m_d[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# d_m_g_m_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_g_m_d))), key=lambda i: d_m_g_m_d[i, [0, 4, 1, 2, 3]].tolist())
d_m_g_m_d = d_m_g_m_d[sorted_index]

with open(save_prefix + '4/4-0-3-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_g_m_d) and d_m_g_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_g_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-3-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_g_m_d) and d_m_g_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_g_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-C-M-C-D
d_c_m_c_d = []
for c1, m, c2 in c_m_c:
    if len(circRNA_disease_list[c1 - num_miRNA]) == 0 or len(circRNA_disease_list[c2 - num_miRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(circRNA_disease_list[c1 - num_miRNA]), int(0.5 * len(circRNA_disease_list[c1 - num_miRNA])),replace=False)
    candidate_d1_list = circRNA_disease_list[c1 - num_miRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(circRNA_disease_list[c2 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c2 - num_miRNA])),replace=False)
    candidate_d2_list = circRNA_disease_list[c2 - num_miRNA][candidate_d2_list]
    d_c_m_c_d.extend([(d1, c1, m, c2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_c_m_c_d = np.array(d_c_m_c_d)
d_c_m_c_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_c_m_c_d[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# d_c_m_c_d[:, [1,3]] += num_miRNA
sorted_index = sorted(list(range(len(d_c_m_c_d))), key=lambda i: d_c_m_c_d[i, [0, 4, 1, 2, 3]].tolist())
d_c_m_c_d = d_c_m_c_d[sorted_index]

with open(save_prefix + '4/4-1-0-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_m_c_d) and d_c_m_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_m_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-0-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_m_c_d) and d_c_m_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_m_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(14)
# D-C-C-C-D
d_c_c_c_d = []
for c1, c, c2 in c_c_c:
    if len(circRNA_disease_list[c1 - num_miRNA]) == 0 or len(circRNA_disease_list[c2 - num_miRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(circRNA_disease_list[c1 - num_miRNA]), int(0.5 * len(circRNA_disease_list[c1 - num_miRNA])),replace=False)
    candidate_d1_list = circRNA_disease_list[c1 - num_miRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(circRNA_disease_list[c2 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c2 - num_miRNA])),replace=False)
    candidate_d2_list = circRNA_disease_list[c2 - num_miRNA][candidate_d2_list]
    d_c_c_c_d.extend([(d1, c1, c, c2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_c_c_c_d = np.array(d_c_c_c_d)
d_c_c_c_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_c_c_c_d[:, 2] += num_miRNA
# d_c_c_c_d[:, [1,3]] += num_miRNA
sorted_index = sorted(list(range(len(d_c_c_c_d))), key=lambda i: d_c_c_c_d[i, [0, 4, 1, 2, 3]].tolist())
d_c_c_c_d = d_c_c_c_d[sorted_index]

with open(save_prefix + '4/4-1-1-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_c_d) and d_c_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_c_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-1-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_c_d) and d_c_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_c_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-L-M-L-D
d_l_m_l_d = []
for l1, m, l2 in l_m_l:
    if len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]), int(0.5 * len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_d1_list = lncRNA_disease_list[l1 - num_miRNA - num_circRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA])),replace=False)
    candidate_d2_list = lncRNA_disease_list[l2 - num_miRNA - num_circRNA][candidate_d2_list]
    d_l_m_l_d.extend([(d1, l1, m, l2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_l_m_l_d = np.array(d_l_m_l_d)
d_l_m_l_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_l_m_l_d[:, 2] += num_miRNA
# d_l_m_l_d[:, [1,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_l_m_l_d))), key=lambda i: d_l_m_l_d[i, [0, 4, 1, 2, 3]].tolist())
d_l_m_l_d = d_l_m_l_d[sorted_index]

with open(save_prefix + '4/4-2-0-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_m_l_d) and d_l_m_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_m_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-0-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_m_l_d) and d_l_m_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_m_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-L-L-L-D
d_l_l_l_d = []
for l1, l, l2 in l_l_l:
    if len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]), int(0.5 * len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_d1_list = lncRNA_disease_list[l1 - num_miRNA - num_circRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA])),replace=False)
    candidate_d2_list = lncRNA_disease_list[l2 - num_miRNA - num_circRNA][candidate_d2_list]
    d_l_l_l_d.extend([(d1, l1, l, l2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_l_l_l_d = np.array(d_l_l_l_d)
d_l_l_l_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_l_l_l_d[:, 2] += num_miRNA + num_circRNA
# d_l_l_l_d[:, [1,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_l_l_l_d))), key=lambda i: d_l_l_l_d[i, [0, 4, 1, 2, 3]].tolist())
d_l_l_l_d = d_l_l_l_d[sorted_index]

with open(save_prefix + '4/4-2-2-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_l_d) and d_l_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_l_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)
gc.collect()
# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-2-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_l_d) and d_l_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_l_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-G-M-G-D
d_g_m_g_d = []
for g1, m, g2 in g_m_g:
    if len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]), int(0.5 * len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d1_list = gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d2_list = gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA][candidate_d2_list]
    d_g_m_g_d.extend([(d1, g1, m, g2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_g_m_g_d = np.array(d_g_m_g_d)
d_g_m_g_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_g_m_g_d[:, 2] += num_miRNA + num_circRNA
# d_g_m_g_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_g_m_g_d))), key=lambda i: d_g_m_g_d[i, [0, 4, 1, 2, 3]].tolist())
d_g_m_g_d = d_g_m_g_d[sorted_index]

with open(save_prefix + '4/4-3-0-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_m_g_d) and d_g_m_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_m_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-0-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_m_g_d) and d_g_m_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_m_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-G-G-G-D
d_g_g_g_d = []
for g1, g, g2 in g_g_g:
    if len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]), int(0.5 * len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d1_list = gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d2_list = gene_disease_list[g2 - num_miRNA - num_circRNA - num_lncRNA][candidate_d2_list]
    d_g_g_g_d.extend([(d1, g1, g, g2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_g_g_g_d = np.array(d_g_g_g_d)
d_g_m_g_d[:, [0,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_g_m_g_d[:, 2] += num_miRNA + num_circRNA + num_lncRNA
# d_g_m_g_d[:, [1,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_g_g_g_d))), key=lambda i: d_g_g_g_d[i, [0, 4, 1, 2, 3]].tolist())
d_g_g_g_d = d_g_g_g_d[sorted_index]

with open(save_prefix + '4/4-3-3-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_g_d) and d_g_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_g_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-3-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_g_d) and d_g_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_g_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


print(15)
# ========================================六元组====================================
# M-C-C-C-C-M
m_c_c_c_c_m = []
for c1, c2, c3, c4 in c_c_c_c:
    if len(circRNA_miRNA_list[c1 - num_miRNA]) == 0 or len(circRNA_miRNA_list[c4 - num_miRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(circRNA_miRNA_list[c1 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c1 - num_miRNA])),replace=False)
    candidate_m1_list = circRNA_miRNA_list[c1 - num_miRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(circRNA_miRNA_list[c4 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c4 - num_miRNA])),replace=False)
    candidate_m2_list = circRNA_miRNA_list[c4 - num_miRNA][candidate_m2_list]
    m_c_c_c_c_m.extend([(m1, c1, c2, c3, c4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_c_c_c_c_m = np.array(m_c_c_c_c_m)
# m_c_c_c_c_m[:, [1,2,3,4]] += num_miRNA
sorted_index = sorted(list(range(len(m_c_c_c_c_m))), key=lambda i: m_c_c_c_c_m[i, [0, 3, 4, 5, 1, 2]].tolist())
m_c_c_c_c_m = m_c_c_c_c_m[sorted_index]

with open(save_prefix + '0/0-1-1-1-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_c_c_m) and m_c_c_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_c_c_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-1-1-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_c_c_c_m) and m_c_c_c_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_c_c_c_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-C-D-D-C-M
m_c_d_d_c_m = []
for c1, d2, d3, c4 in c_d_d_c:
    if len(circRNA_miRNA_list[c1 - num_miRNA]) == 0 or len(circRNA_miRNA_list[c4 - num_miRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(circRNA_miRNA_list[c1 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c1 - num_miRNA])),replace=False)
    candidate_m1_list = circRNA_miRNA_list[c1 - num_miRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(circRNA_miRNA_list[c4 - num_miRNA]),int(0.5 * len(circRNA_miRNA_list[c4 - num_miRNA])),replace=False)
    candidate_m2_list = circRNA_miRNA_list[c4 - num_miRNA][candidate_m2_list]
    m_c_d_d_c_m.extend([(m1, c1, d2, d3, c4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_c_d_d_c_m = np.array(m_c_d_d_c_m)
# m_c_d_d_c_m[:, [1,4]] += num_miRNA
# m_c_d_d_c_m[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_c_d_d_c_m))), key=lambda i: m_c_d_d_c_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_c_d_d_c_m = m_c_d_d_c_m[sorted_index]

with open(save_prefix + '0/0-1-4-4-1-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_d_d_c_m) and m_c_d_d_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_c_d_d_c_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-1-4-4-1-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_c_d_d_c_m) and m_c_d_d_c_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_c_d_d_c_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


# M-L-L-L-L-M
m_l_l_l_l_m = []
for l1, l2, l3, l4 in l_l_l_l:
    if len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_m1_list = lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA])),replace=False)
    candidate_m2_list = lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA][candidate_m2_list]
    m_l_l_l_l_m.extend([(m1, l1, l2, l3, l4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_l_l_l_l_m = np.array(m_l_l_l_l_m)
# m_l_l_l_l_m[:, [1,2,3,4]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_l_l_l_l_m))), key=lambda i: m_l_l_l_l_m[i, [0, 3, 4, 5, 1, 2]].tolist())
m_l_l_l_l_m = m_l_l_l_l_m[sorted_index]

with open(save_prefix + '0/0-2-2-2-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_l_l_m) and m_l_l_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_l_l_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-2-2-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_l_l_l_m) and m_l_l_l_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_l_l_l_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-L-D-D-L-M
m_l_d_d_l_m = []
for l1, d2, d3, l4 in l_d_d_l:
    if len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_m1_list = lncRNA_miRNA_list[l1 - num_miRNA - num_circRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA])),replace=False)
    candidate_m2_list = lncRNA_miRNA_list[l4 - num_miRNA - num_circRNA][candidate_m2_list]
    m_l_d_d_l_m.extend([(m1, l1, d2, d3, l4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_l_d_d_l_m = np.array(m_l_d_d_l_m)
# m_l_d_d_l_m[:, [1,4]] += num_miRNA + num_circRNA
# m_l_d_d_l_m[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_l_d_d_l_m))), key=lambda i: m_l_d_d_l_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_l_d_d_l_m = m_l_d_d_l_m[sorted_index]

with open(save_prefix + '0/0-2-4-4-2-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_d_d_l_m) and m_l_d_d_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_l_d_d_l_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-2-4-4-2-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_l_d_d_l_m) and m_l_d_d_l_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_l_d_d_l_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-G-G-G-G-M
m_g_g_g_g_m = []
for g1, g2, g3, g4 in g_g_g_g:
    if len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m1_list = gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m2_list = lncRNA_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA][candidate_m2_list]
    m_g_g_g_g_m.extend([(m1, g1, g2, g3, g4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_g_g_g_g_m = np.array(m_g_g_g_g_m)
# m_g_g_g_g_m[:, [1,2,3,4]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(m_g_g_g_g_m))), key=lambda i: m_g_g_g_g_m[i, [0, 3, 4, 5, 1, 2]].tolist())
m_g_g_g_g_m = m_g_g_g_g_m[sorted_index]

with open(save_prefix + '0/0-3-3-3-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_g_g_m) and m_g_g_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_g_g_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-3-3-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_g_g_g_m) and m_g_g_g_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_g_g_g_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-G-D-D-G-M
m_g_d_d_g_m = []
for g1, d2, d3, g4 in g_d_d_g:
    if len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m1_list = gene_miRNA_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_m2_list = gene_miRNA_list[g4 - num_miRNA - num_circRNA - num_lncRNA][candidate_m2_list]
    m_g_d_d_g_m.extend([(m1, g1, d2, d3, g4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_g_d_d_g_m = np.array(m_g_d_d_g_m)
# m_g_d_d_g_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA
# m_g_d_d_g_m[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_g_d_d_g_m))), key=lambda i: m_g_d_d_g_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_g_d_d_g_m = m_g_d_d_g_m[sorted_index]

with open(save_prefix + '0/0-3-4-4-3-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_d_d_g_m) and m_g_d_d_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_g_d_d_g_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-3-4-4-3-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_g_d_d_g_m) and m_g_d_d_g_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_g_d_d_g_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-C-C-D-M
m_d_c_c_d_m = []
for d1, c2, c3, d4 in d_c_c_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_c_c_d_m.extend([(m1, d1, c2, c3, d4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_c_c_d_m = np.array(m_d_c_c_d_m)
# m_d_c_c_d_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_c_c_d_m[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(m_d_c_c_d_m))), key=lambda i: m_d_c_c_d_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_d_c_c_d_m = m_d_c_c_d_m[sorted_index]

with open(save_prefix + '0/0-4-1-1-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_c_c_d_m) and m_d_c_c_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_c_c_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-1-1-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_c_c_d_m) and m_d_c_c_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_c_c_d_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-L-L-D-M
m_d_l_l_d_m = []
for d1, l2, l3, d4 in d_l_l_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_l_l_d_m.extend([(m1, d1, l2, l3, d4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_l_l_d_m = np.array(m_d_l_l_d_m)
# m_d_l_l_d_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_l_l_d_m[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(m_d_l_l_d_m))), key=lambda i: m_d_l_l_d_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_d_l_l_d_m = m_d_l_l_d_m[sorted_index]

with open(save_prefix + '0/0-4-2-2-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_l_l_d_m) and m_d_l_l_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_l_l_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-2-2-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_l_l_d_m) and m_d_l_l_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_l_l_d_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-G-G-D-M
m_d_g_g_d_m = []
for d1, g2, g3, d4 in d_g_g_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_g_g_d_m.extend([(m1, d1, g2, g3, d4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_g_g_d_m = np.array(m_d_g_g_d_m)
# m_d_g_g_d_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_g_g_d_m[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(m_d_g_g_d_m))), key=lambda i: m_d_g_g_d_m[i, [0, 4, 5, 1, 2, 3]].tolist())
m_d_g_g_d_m = m_d_g_g_d_m[sorted_index]

with open(save_prefix + '0/0-4-3-3-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_g_g_d_m) and m_d_g_g_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_g_g_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-3-3-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_g_g_d_m) and m_d_g_g_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_g_g_d_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# M-D-D-D-D-M
m_d_d_d_d_m = []
for d1, d2, d3, d4 in d_d_d_d:
    if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_m2_list = disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    m_d_d_d_d_m.extend([(m1, d1, d2, d3, d4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_d_d_d_m = np.array(m_d_d_d_d_m)
# m_d_d_d_d_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# m_d_d_d_d_m[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(m_d_d_d_d_m))), key=lambda i: m_d_d_d_d_m[i, [0, 3, 4, 5, 1, 2]].tolist())
m_d_d_d_d_m = m_d_d_d_d_m[sorted_index]

with open(save_prefix + '0/0-4-4-4-4-0_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_d_d_m) and m_d_d_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = m_d_d_d_d_m[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '0/0-4-4-4-4-0.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[0]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(m_d_d_d_d_m) and m_d_d_d_d_m[right, 0] == target_idx + offset_list[0]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = m_d_d_d_d_m[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-M-M-M-C
c_m_m_m_m_c = []
for m1, m2, m3, m4 in m_m_m_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m4]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]),int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m4]),int(0.5 * len(miRNA_circRNA_list[m4])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m4][candidate_c2_list]
    c_m_m_m_m_c.extend([(c1, m1, m2, m3, m4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_m_m_m_c = np.array(c_m_m_m_m_c)
c_m_m_m_m_c[:, [0,5]] += num_miRNA
sorted_index = sorted(list(range(len(c_m_m_m_m_c))), key=lambda i: c_m_m_m_m_c[i, [0, 3, 4, 5, 1, 2]].tolist())
c_m_m_m_m_c = c_m_m_m_m_c[sorted_index]

with open(save_prefix + '1/1-0-0-0-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_m_m_c) and c_m_m_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_m_m_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-0-0-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_m_m_m_c) and c_m_m_m_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_m_m_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-L-L-M-C
gc.collect()
c_m_l_l_m_c = []
for m1, l2, l3, m4 in m_l_l_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m4]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]),int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m4]),int(0.5 * len(miRNA_circRNA_list[m4])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m4][candidate_c2_list]
    c_m_l_l_m_c.extend([(c1, m1, l2, l3, m4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_l_l_m_c = np.array(c_m_l_l_m_c)
c_m_l_l_m_c[:, [0,5]] += num_miRNA
# c_m_l_l_m_c[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(c_m_l_l_m_c))), key=lambda i: c_m_l_l_m_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_m_l_l_m_c = c_m_l_l_m_c[sorted_index]

with open(save_prefix + '1/1-0-2-2-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_l_l_m_c) and c_m_l_l_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_l_l_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-2-2-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_l_l_m_c) and c_m_l_l_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_l_l_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


# C-M-G-G-M-C
c_m_g_g_m_c = []
for m1, g2, g3, m4 in m_g_g_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m4]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]),int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m4]),int(0.5 * len(miRNA_circRNA_list[m4])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m4][candidate_c2_list]
    c_m_g_g_m_c.extend([(c1, m1, g2, g3, m4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_g_g_m_c = np.array(c_m_g_g_m_c)
c_m_g_g_m_c[:, [0,5]] += num_miRNA
# c_m_g_g_m_c[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(c_m_g_g_m_c))), key=lambda i: c_m_g_g_m_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_m_g_g_m_c = c_m_g_g_m_c[sorted_index]

with open(save_prefix + '1/1-0-3-3-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_g_g_m_c) and c_m_g_g_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_g_g_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-3-3-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_g_g_m_c) and c_m_g_g_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_g_g_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-M-D-D-M-C
c_m_d_d_m_c = []
for m1, d2, d3, m4 in m_d_d_m:
    if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m4]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]),int(0.5 * len(miRNA_circRNA_list[m1])),replace=False)
    candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m4]),int(0.5 * len(miRNA_circRNA_list[m4])),replace=False)
    candidate_c2_list = miRNA_circRNA_list[m4][candidate_c2_list]
    c_m_d_d_m_c.extend([(c1, m1, d2, d3, m4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_m_d_d_m_c = np.array(c_m_d_d_m_c)
c_m_d_d_m_c[:, [0,5]] += num_miRNA
# c_m_d_d_m_c[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_m_d_d_m_c))), key=lambda i: c_m_d_d_m_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_m_d_d_m_c = c_m_d_d_m_c[sorted_index]

with open(save_prefix + '1/1-0-4-4-0-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_d_m_c) and c_m_d_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_d_d_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)
gc.collect()
# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-0-4-4-0-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_d_m_c) and c_m_d_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_d_d_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


print(16)
gc.collect()
# C-D-M-M-D-C
c_d_m_m_d_c = []
for d1, m2, m3, d4 in d_m_m_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_m_m_d_c.extend([(c1, d1, m2, m3, d4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_m_m_d_c = np.array(c_d_m_m_d_c)
c_d_m_m_d_c[:, [0,5]] += num_miRNA
# c_d_m_m_d_c[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_m_m_d_c))), key=lambda i: c_d_m_m_d_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_d_m_m_d_c = c_d_m_m_d_c[sorted_index]

with open(save_prefix + '1/1-4-0-0-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_d_m_c) and c_m_d_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_m_d_d_m_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-0-0-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_m_d_d_m_c) and c_m_d_d_m_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_m_d_d_m_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-L-L-D-C
c_d_l_l_d_c = []
for d1, l2, l3, d4 in d_l_l_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_l_l_d_c.extend([(c1, d1, l2, l3, d4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_l_l_d_c = np.array(c_d_l_l_d_c)
c_d_l_l_d_c[:, [0,5]] += num_miRNA
# c_d_l_l_d_c[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# c_d_l_l_d_c[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(c_d_l_l_d_c))), key=lambda i: c_d_l_l_d_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_d_l_l_d_c = c_d_l_l_d_c[sorted_index]

with open(save_prefix + '1/1-4-2-2-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_l_l_d_c) and c_d_l_l_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_l_l_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-2-2-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_l_l_d_c) and c_d_l_l_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_l_l_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-G-G-D-C
c_d_g_g_d_c = []
for d1, g2, g3, d4 in d_g_g_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_g_g_d_c.extend([(c1, d1, g2, g3, d4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_g_g_d_c = np.array(c_d_g_g_d_c)
c_d_g_g_d_c[:, [0,5]] += num_miRNA
# c_d_g_g_d_c[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# c_d_g_g_d_c[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(c_d_g_g_d_c))), key=lambda i: c_d_g_g_d_c[i, [0, 4, 5, 1, 2, 3]].tolist())
c_d_g_g_d_c = c_d_g_g_d_c[sorted_index]

with open(save_prefix + '1/1-4-3-3-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_g_g_d_c) and c_d_g_g_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_g_g_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-3-3-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_g_g_d_c) and c_d_g_g_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_g_g_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# C-D-D-D-D-C
c_d_d_d_d_c = []
for d1, d2, d3, d4 in d_d_d_d:
    if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_c2_list = disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    c_d_d_d_d_c.extend([(c1, d1, d2, d3, d4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_d_d_d_d_c = np.array(c_d_d_d_d_c)
c_d_d_d_d_c[:, [0,5]] += num_miRNA
# c_d_d_d_d_c[:, [1,2,3,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(c_d_d_d_d_c))), key=lambda i: c_d_d_d_d_c[i, [0, 3, 4, 5, 1, 2]].tolist())
c_d_d_d_d_c = c_d_d_d_d_c[sorted_index]

with open(save_prefix + '1/1-4-4-4-4-1_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_d_d_c) and c_d_d_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = c_d_d_d_d_c[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '1/1-4-4-4-4-1.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[1]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(c_d_d_d_d_c) and c_d_d_d_d_c[right, 0] == target_idx + offset_list[1]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = c_d_d_d_d_c[left:right, -1] - offset_list[1]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-M-M-M-L
l_m_m_m_m_l = []
for m1, m2, m3, m4 in m_m_m_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m4]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]),int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m4]),int(0.5 * len(miRNA_lncRNA_list[m4])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m4][candidate_l2_list]
    l_m_m_m_m_l.extend([(l1, m1, m2, m3, m4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_m_m_m_l = np.array(l_m_m_m_m_l)
l_m_m_m_m_l[:, [0,5]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(l_m_m_m_m_l))), key=lambda i: l_m_m_m_m_l[i, [0, 3, 4, 5, 1, 2]].tolist())
l_m_m_m_m_l = l_m_m_m_m_l[sorted_index]

with open(save_prefix + '2/2-0-0-0-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_m_m_l) and l_m_m_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_m_m_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-0-0-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_m_m_m_l) and l_m_m_m_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_m_m_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-C-C-M-L
l_m_c_c_m_l = []
for m1, c2, c3, m4 in m_c_c_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m4]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]),int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m4]),int(0.5 * len(miRNA_lncRNA_list[m4])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m4][candidate_l2_list]
    l_m_c_c_m_l.extend([(l1, m1, c2, c3, m4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_c_c_m_l = np.array(l_m_c_c_m_l)
l_m_c_c_m_l[:, [0,5]] += num_miRNA + num_circRNA
# l_m_c_c_m_l[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(l_m_c_c_m_l))), key=lambda i: l_m_c_c_m_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_m_c_c_m_l = l_m_c_c_m_l[sorted_index]

with open(save_prefix + '2/2-0-1-1-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_c_c_m_l) and l_m_c_c_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_c_c_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-1-1-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_c_c_m_l) and l_m_c_c_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_c_c_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-G-G-M-L
l_m_g_g_m_l = []
for m1, g2, g3, m4 in m_g_g_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m4]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]),int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m4]),int(0.5 * len(miRNA_lncRNA_list[m4])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m4][candidate_l2_list]
    l_m_g_g_m_l.extend([(l1, m1, g2, g3, m4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_g_g_m_l = np.array(l_m_g_g_m_l)
l_m_g_g_m_l[:, [0,5]] += num_miRNA + num_circRNA
# l_m_g_g_m_l[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(l_m_g_g_m_l))), key=lambda i: l_m_g_g_m_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_m_g_g_m_l = l_m_g_g_m_l[sorted_index]

with open(save_prefix + '2/2-0-3-3-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_g_g_m_l) and l_m_g_g_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_g_g_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-3-3-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_g_g_m_l) and l_m_g_g_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_g_g_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-M-D-D-M-L
l_m_d_d_m_l = []
for m1, d2, d3, m4 in m_d_d_m:
    if len(miRNA_lncRNA_list[m1]) == 0 or len(miRNA_lncRNA_list[m4]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(miRNA_lncRNA_list[m1]),int(0.5 * len(miRNA_lncRNA_list[m1])),replace=False)
    candidate_l1_list = miRNA_lncRNA_list[m1][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(miRNA_lncRNA_list[m4]),int(0.5 * len(miRNA_lncRNA_list[m4])),replace=False)
    candidate_l2_list = miRNA_lncRNA_list[m4][candidate_l2_list]
    l_m_d_d_m_l.extend([(l1, m1, d2, d3, m4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_m_d_d_m_l = np.array(l_m_d_d_m_l)
l_m_d_d_m_l[:, [0,5]] += num_miRNA + num_circRNA
# l_m_d_d_m_l[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_m_d_d_m_l))), key=lambda i: l_m_d_d_m_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_m_d_d_m_l = l_m_d_d_m_l[sorted_index]

with open(save_prefix + '2/2-0-4-4-0-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_d_d_m_l) and l_m_d_d_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_m_d_d_m_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-0-4-4-0-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_m_d_d_m_l) and l_m_d_d_m_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_m_d_d_m_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-M-M-D-L
l_d_m_m_d_l = []
for d1, m2, m3, d4 in d_m_m_d:
    if len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0 or len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l2_list]
    l_d_m_m_d_l.extend([(l1, d1, m2, m3, d4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_m_m_d_l = np.array(l_d_m_m_d_l)
l_d_m_m_d_l[:, [0,5]] += num_miRNA + num_circRNA
# l_d_m_m_d_l[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_m_m_d_l))), key=lambda i: l_d_m_m_d_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_d_m_m_d_l = l_d_m_m_d_l[sorted_index]

with open(save_prefix + '2/2-4-0-0-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_m_m_d_l) and l_d_m_m_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_m_m_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-0-0-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_m_m_d_l) and l_d_m_m_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_m_m_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-C-C-D-L
l_d_c_c_d_l = []
for d1, c2, c3, d4 in d_c_c_d:
    if len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0 or len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l2_list]
    l_d_c_c_d_l.extend([(l1, d1, c2, c3, d4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_c_c_d_l = np.array(l_d_c_c_d_l)
l_d_c_c_d_l[:, [0,5]] += num_miRNA + num_circRNA
# l_d_c_c_d_l[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_d_c_c_d_l[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(l_d_c_c_d_l))), key=lambda i: l_d_c_c_d_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_d_c_c_d_l = l_d_c_c_d_l[sorted_index]
gc.collect()
with open(save_prefix + '2/2-4-1-1-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_c_c_d_l) and l_d_c_c_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_c_c_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-1-1-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_c_c_d_l) and l_d_c_c_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_c_c_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


print(17)
gc.collect()
# L-D-G-G-D-L
l_d_g_g_d_l = []
for d1, g2, g3, d4 in d_g_g_d:
    if len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0 or len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l2_list]
    l_d_g_g_d_l.extend([(l1, d1, g2, g3, d4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_g_g_d_l = np.array(l_d_c_c_d_l)
l_d_g_g_d_l[:, [0,5]] += num_miRNA + num_circRNA
# l_d_g_g_d_l[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# l_d_g_g_d_l[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(l_d_g_g_d_l))), key=lambda i: l_d_g_g_d_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_d_g_g_d_l = l_d_g_g_d_l[sorted_index]

with open(save_prefix + '2/2-4-3-3-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_g_g_d_l) and l_d_g_g_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_g_g_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-3-3-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_g_g_d_l) and l_d_g_g_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_g_g_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# L-D-D-D-D-L
l_d_d_d_d_l = []
for d1, d2, d3, d4 in d_d_d_d:
    if len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0 or len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]) == 0:
        continue
    candidate_l1_list = np.random.choice(len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l1_list = disease_lncRNA_list[d1 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l1_list]
    candidate_l2_list = np.random.choice(len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene]),int(0.5 * len(disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene])),replace=False)
    candidate_l2_list = disease_lncRNA_list[d4 - num_miRNA + num_circRNA + num_lncRNA + num_gene][candidate_l2_list]
    l_d_d_d_d_l.extend([(l1, d1, d2, d3, d4, l2) for l1 in candidate_l1_list for l2 in candidate_l2_list])
l_d_d_d_d_l = np.array(l_d_c_c_d_l)
l_d_d_d_d_l[:, [0,5]] += num_miRNA + num_circRNA
# l_d_d_d_d_l[:, [1,2,3,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(l_d_d_d_d_l))), key=lambda i: l_d_d_d_d_l[i, [0, 4, 5, 1, 2, 3]].tolist())
l_d_d_d_d_l = l_d_d_d_d_l[sorted_index]

with open(save_prefix + '2/2-4-4-4-4-2_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_d_d_l) and l_d_d_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = l_d_d_d_d_l[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '2/2-4-4-4-4-2.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[2]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(l_d_d_d_d_l) and l_d_d_d_d_l[right, 0] == target_idx + offset_list[2]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = l_d_d_d_d_l[left:right, -1] - offset_list[2]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-M-M-M-G
g_m_m_m_m_g = []
for m1, m2, m3, m4 in m_m_m_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m4]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]),int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m4]),int(0.5 * len(miRNA_gene_list[m4])),replace=False)
    candidate_g2_list = miRNA_gene_list[m4][candidate_g2_list]
    g_m_m_m_m_g.extend([(g1, m1, m2, m3, m4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_m_m_m_g = np.array(l_d_c_c_d_l)
g_m_m_m_m_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(g_m_m_m_m_g))), key=lambda i: g_m_m_m_m_g[i, [0, 3, 4, 5, 1, 2]].tolist())
g_m_m_m_m_g = g_m_m_m_m_g[sorted_index]

with open(save_prefix + '3/3-0-0-0-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_m_m_g) and g_m_m_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_m_m_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-0-0-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_m_m_m_g) and g_m_m_m_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_m_m_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-C-C-M-G
g_m_c_c_m_g = []
for m1, c2, c3, m4 in m_c_c_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m4]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]),int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m4]),int(0.5 * len(miRNA_gene_list[m4])),replace=False)
    candidate_g2_list = miRNA_gene_list[m4][candidate_g2_list]
    g_m_c_c_m_g.extend([(g1, m1, c2, c3, m4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_c_c_m_g = np.array(l_d_c_c_d_l)
g_m_c_c_m_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_c_c_m_g[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(g_m_c_c_m_g))), key=lambda i: g_m_c_c_m_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_m_c_c_m_g = g_m_c_c_m_g[sorted_index]

with open(save_prefix + '3/3-0-1-1-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_c_c_m_g) and g_m_c_c_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_c_c_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-1-1-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_c_c_m_g) and g_m_c_c_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_c_c_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-L-L-M-G
g_m_l_l_m_g = []
for m1, l2, l3, m4 in m_l_l_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m4]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]),int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m4]),int(0.5 * len(miRNA_gene_list[m4])),replace=False)
    candidate_g2_list = miRNA_gene_list[m4][candidate_g2_list]
    g_m_l_l_m_g.extend([(g1, m1, l2, l3, m4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_l_l_m_g = np.array(l_d_c_c_d_l)
g_m_l_l_m_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_l_l_m_g[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(g_m_l_l_m_g))), key=lambda i: g_m_l_l_m_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_m_l_l_m_g = g_m_l_l_m_g[sorted_index]

with open(save_prefix + '3/3-0-2-2-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_l_l_m_g) and g_m_l_l_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_l_l_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-2-2-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_l_l_m_g) and g_m_l_l_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_l_l_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-M-D-D-M-G
g_m_d_d_m_g = []
for m1, d2, d3, m4 in m_d_d_m:
    if len(miRNA_gene_list[m1]) == 0 or len(miRNA_gene_list[m4]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(miRNA_gene_list[m1]),int(0.5 * len(miRNA_gene_list[m1])),replace=False)
    candidate_g1_list = miRNA_gene_list[m1][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(miRNA_gene_list[m4]),int(0.5 * len(miRNA_gene_list[m4])),replace=False)
    candidate_g2_list = miRNA_gene_list[m4][candidate_g2_list]
    g_m_d_d_m_g.extend([(g1, m1, d2, d3, m4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_m_d_d_m_g = np.array(l_d_c_c_d_l)
g_m_d_d_m_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_m_d_d_m_g[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_m_d_d_m_g))), key=lambda i: g_m_d_d_m_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_m_d_d_m_g = g_m_d_d_m_g[sorted_index]

with open(save_prefix + '3/3-0-4-4-0-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_d_d_m_g) and g_m_d_d_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_m_d_d_m_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-0-4-4-0-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_m_d_d_m_g) and g_m_d_d_m_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_m_d_d_m_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(18)
gc.collect()
# G-D-M-M-D-G
g_d_m_m_d_g = []
for d1, m2, m3, d4 in d_m_m_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_m_m_d_g.extend([(g1, d1, m2, m3, d4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_m_m_d_g = np.array(l_d_c_c_d_l)
g_d_m_m_d_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_m_m_d_g[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_m_m_d_g))), key=lambda i: g_d_m_m_d_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_d_m_m_d_g = g_d_m_m_d_g[sorted_index]

with open(save_prefix + '3/3-4-0-0-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_m_m_d_g) and g_d_m_m_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_m_m_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-0-0-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_m_m_d_g) and g_d_m_m_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_m_m_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-C-C-D-G
g_d_c_c_d_g = []
for d1, c2, c3, d4 in d_c_c_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_c_c_d_g.extend([(g1, d1, c2, c3, d4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_c_c_d_g = np.array(l_d_c_c_d_l)
g_d_c_c_d_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_c_c_d_g[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_d_c_c_d_g[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(g_d_c_c_d_g))), key=lambda i: g_d_c_c_d_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_d_c_c_d_g = g_d_c_c_d_g[sorted_index]

with open(save_prefix + '3/3-4-1-1-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_c_c_d_g) and g_d_c_c_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_c_c_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)
gc.collect()
# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-1-1-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_c_c_d_g) and g_d_c_c_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_c_c_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-L-L-D-G
g_d_l_l_d_g = []
for d1, l2, l3, d4 in d_l_l_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_l_l_d_g.extend([(g1, d1, l2, l3, d4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_l_l_d_g = np.array(g_d_l_l_d_g)
g_d_l_l_d_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_l_l_d_g[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_d_l_l_d_g[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(g_d_l_l_d_g))), key=lambda i: g_d_l_l_d_g[i, [0, 4, 5, 1, 2, 3]].tolist())
g_d_l_l_d_g = g_d_l_l_d_g[sorted_index]

with open(save_prefix + '3/3-4-2-2-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_l_l_d_g) and g_d_l_l_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_l_l_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-2-2-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_l_l_d_g) and g_d_l_l_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_l_l_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# G-D-D-D-D-G
g_d_d_d_d_g = []
for d1, d2, d3, d4 in d_d_d_d:
    if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
        continue
    candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    candidate_g2_list = np.random.choice(len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    candidate_g2_list = disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    g_d_d_d_d_g.extend([(g1, d1, d2, d3, d4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
g_d_d_d_d_g = np.array(g_d_d_d_d_g)
g_d_d_d_d_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
# g_d_d_d_d_g[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# g_d_d_d_d_g[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(g_d_d_d_d_g))), key=lambda i: g_d_d_d_d_g[i, [0, 3, 4, 5, 1, 2]].tolist())
g_d_d_d_d_g = g_d_d_d_d_g[sorted_index]

with open(save_prefix + '3/3-4-4-4-4-3_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_d_d_g) and g_d_d_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = g_d_d_d_d_g[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '3/3-4-4-4-4-3.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[3]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(g_d_d_d_d_g) and g_d_d_d_d_g[right, 0] == target_idx + offset_list[3]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = g_d_d_d_d_g[left:right, -1] - offset_list[3]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-M-M-M-D
d_m_m_m_m_d = []
for m1, m2, m3, m4 in m_m_m_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m4]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]),int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m4]),int(0.5 * len(miRNA_disease_list[m4])),replace=False)
    candidate_d2_list = miRNA_disease_list[m4][candidate_d2_list]
    d_m_m_m_m_d.extend([(d1, m1, m2, m3, m4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_m_m_m_d = np.array(d_m_m_m_m_d)
d_m_m_m_m_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
sorted_index = sorted(list(range(len(d_m_m_m_m_d))), key=lambda i: d_m_m_m_m_d[i, [0, 3, 4, 5, 1, 2]].tolist())
d_m_m_m_m_d = d_m_m_m_m_d[sorted_index]

with open(save_prefix + '4/4-0-0-0-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_m_m_d) and d_m_m_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_m_m_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-0-0-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_m_m_m_d) and d_m_m_m_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_m_m_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


# D-M-C-C-M-D
d_m_c_c_m_d = []
for m1, c2, c3, m4 in m_c_c_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m4]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]),int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m4]),int(0.5 * len(miRNA_disease_list[m4])),replace=False)
    candidate_d2_list = miRNA_disease_list[m4][candidate_d2_list]
    d_m_c_c_m_d.extend([(d1, m1, c2, c3, m4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_c_c_m_d = np.array(d_m_c_c_m_d)
d_m_c_c_m_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_c_c_m_d[:, [2,3]] += num_miRNA
sorted_index = sorted(list(range(len(d_m_c_c_m_d))), key=lambda i: d_m_c_c_m_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_m_c_c_m_d = d_m_c_c_m_d[sorted_index]


with open(save_prefix + '4/4-0-1-1-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_c_c_m_d) and d_m_c_c_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_c_c_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-1-1-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_c_c_m_d) and d_m_c_c_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_c_c_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-L-L-M-D
d_m_l_l_m_d = []
for m1, l2, l3, m4 in m_l_l_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m4]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]),int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m4]),int(0.5 * len(miRNA_disease_list[m4])),replace=False)
    candidate_d2_list = miRNA_disease_list[m4][candidate_d2_list]
    d_m_l_l_m_d.extend([(d1, m1, l2, l3, m4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_l_l_m_d = np.array(d_m_l_l_m_d)
d_m_l_l_m_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_l_l_m_d[:, [2,3]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_m_l_l_m_d))), key=lambda i: d_m_l_l_m_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_m_l_l_m_d = d_m_l_l_m_d[sorted_index]

with open(save_prefix + '4/4-0-2-2-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_l_l_m_d) and d_m_l_l_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_l_l_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-2-2-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_l_l_m_d) and d_m_l_l_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_l_l_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-M-G-G-M-D
d_m_g_g_m_d = []
for m1, g2, g3, m4 in m_g_g_m:
    if len(miRNA_disease_list[m1]) == 0 or len(miRNA_disease_list[m4]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(miRNA_disease_list[m1]),int(0.5 * len(miRNA_disease_list[m1])),replace=False)
    candidate_d1_list = miRNA_disease_list[m1][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(miRNA_disease_list[m4]),int(0.5 * len(miRNA_disease_list[m4])),replace=False)
    candidate_d2_list = miRNA_disease_list[m4][candidate_d2_list]
    d_m_g_g_m_d.extend([(d1, m1, g2, g3, m4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_m_g_g_m_d = np.array(d_m_g_g_m_d)
d_m_g_g_m_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_m_g_g_m_d[:, [2,3]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_m_g_g_m_d))), key=lambda i: d_m_g_g_m_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_m_g_g_m_d = d_m_g_g_m_d[sorted_index]

with open(save_prefix + '4/4-0-3-3-0-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_g_g_m_d) and d_m_g_g_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_m_g_g_m_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-0-3-3-0-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_m_g_g_m_d) and d_m_g_g_m_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_m_g_g_m_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

print(19)
gc.collect()
# D-C-M-M-C-D
d_c_m_m_c_d = []
for c1, m2, m3, c4 in c_m_m_c:
    if len(circRNA_disease_list[c1 - num_miRNA]) == 0 or len(circRNA_disease_list[c4 - num_miRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(circRNA_disease_list[c1 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c1 - num_miRNA])),replace=False)
    candidate_d1_list = circRNA_disease_list[c1 - num_miRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(circRNA_disease_list[c4 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c4 - num_miRNA])),replace=False)
    candidate_d2_list = circRNA_disease_list[c4 - num_miRNA][candidate_d2_list]
    d_c_m_m_c_d.extend([(d1, c1, m2, m3, c4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_c_m_m_c_d = np.array(d_c_m_m_c_d)
d_c_m_m_c_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_c_m_m_c_d[:, [1,4]] += num_miRNA
sorted_index = sorted(list(range(len(d_c_m_m_c_d))), key=lambda i: d_c_m_m_c_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_c_m_m_c_d = d_c_m_m_c_d[sorted_index]

with open(save_prefix + '4/4-1-0-0-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_m_m_c_d) and d_c_m_m_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_m_m_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-0-0-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_m_m_c_d) and d_c_m_m_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_m_m_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


# D-C-C-C-C-D
d_c_c_c_c_d = []
for c1, c2, c3, c4 in c_c_c_c:
    if len(circRNA_disease_list[c1 - num_miRNA]) == 0 or len(circRNA_disease_list[c4 - num_miRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(circRNA_disease_list[c1 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c1 - num_miRNA])),replace=False)
    candidate_d1_list = circRNA_disease_list[c1 - num_miRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(circRNA_disease_list[c4 - num_miRNA]),int(0.5 * len(circRNA_disease_list[c4 - num_miRNA])),replace=False)
    candidate_d2_list = circRNA_disease_list[c4 - num_miRNA][candidate_d2_list]
    d_c_c_c_c_d.extend([(d1, c1, c2, c3, c4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_c_c_c_c_d = np.array(d_c_c_c_c_d)
d_c_c_c_c_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_c_c_c_c_d[:, [1,2,3,4]] += num_miRNA
sorted_index = sorted(list(range(len(d_c_c_c_c_d))), key=lambda i: d_c_c_c_c_d[i, [0, 3, 4, 5, 1, 2]].tolist())
d_c_c_c_c_d = d_c_c_c_c_d[sorted_index]

with open(save_prefix + '4/4-1-1-1-1-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_c_c_d) and d_c_c_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_c_c_c_c_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-1-1-1-1-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_c_c_c_c_d) and d_c_c_c_c_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_c_c_c_c_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-L-M-M-L-D
d_l_m_m_l_d = []
for l1, m2, m3, l4 in l_m_m_l:
    if len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_d1_list = lncRNA_disease_list[l1 - num_miRNA - num_circRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA])),replace=False)
    candidate_d2_list = lncRNA_disease_list[l4 - num_miRNA - num_circRNA][candidate_d2_list]
    d_l_m_m_l_d.extend([(d1, l1, m2, m3, l4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_l_m_m_l_d = np.array(d_l_m_m_l_d)
d_l_m_m_l_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_l_m_m_l_d[:, [1,4]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_l_m_m_l_d))), key=lambda i: d_l_m_m_l_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_l_m_m_l_d = d_l_m_m_l_d[sorted_index]

with open(save_prefix + '4/4-2-0-0-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_m_m_l_d) and d_l_m_m_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_m_m_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-0-0-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_m_m_l_d) and d_l_m_m_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_m_m_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-L-L-L-L-D
d_l_l_l_l_d = []
for l1, l2, l3, l4 in l_l_l_l:
    if len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA])),replace=False)
    candidate_d1_list = lncRNA_disease_list[l1 - num_miRNA - num_circRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l4 - num_miRNA - num_circRNA])),replace=False)
    candidate_d2_list = lncRNA_disease_list[l4 - num_miRNA - num_circRNA][candidate_d2_list]
    d_l_l_l_l_d.extend([(d1, l1, l2, l3, l4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_l_l_l_l_d = np.array(d_l_l_l_l_d)
d_l_l_l_l_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_l_l_l_l_d[:, [1,2,3,4]] += num_miRNA + num_circRNA
sorted_index = sorted(list(range(len(d_l_l_l_l_d))), key=lambda i: d_l_l_l_l_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_l_l_l_l_d = d_l_l_l_l_d[sorted_index]
gc.collect()
with open(save_prefix + '4/4-2-2-2-2-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_l_l_d) and d_l_l_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_l_l_l_l_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-2-2-2-2-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_l_l_l_l_d) and d_l_l_l_l_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_l_l_l_l_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-G-M-M-G-D
d_g_m_m_g_d = []
for g1, m2, m3, g4 in g_m_m_g:
    if len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d1_list = gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d2_list = gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA][candidate_d2_list]
    d_g_m_m_g_d.extend([(d1, g1, m2, m3, g4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_g_m_m_g_d = np.array(d_g_m_m_g_d)
d_g_m_m_g_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_g_m_m_g_d[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_g_m_m_g_d))), key=lambda i: d_g_m_m_g_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_g_m_m_g_d = d_g_m_m_g_d[sorted_index]

with open(save_prefix + '4/4-3-0-0-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_m_m_g_d) and d_g_m_m_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_m_m_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-0-0-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_m_m_g_d) and d_g_m_m_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_m_m_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right

# D-G-G-G-G-D
d_g_g_g_g_d = []
for g1, g2, g3, g4 in g_g_g_g:
    if len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]) == 0 or len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA]) == 0:
        continue
    candidate_d1_list = np.random.choice(len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d1_list = gene_disease_list[g1 - num_miRNA - num_circRNA - num_lncRNA][candidate_d1_list]
    candidate_d2_list = np.random.choice(len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA]),int(0.5 * len(gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA])),replace=False)
    candidate_d2_list = gene_disease_list[g4 - num_miRNA - num_circRNA - num_lncRNA][candidate_d2_list]
    d_g_g_g_g_d.extend([(d1, g1, g2, g3, g4, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
d_g_g_g_g_d = np.array(d_g_g_g_g_d)
d_g_g_g_g_d[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
# d_g_g_g_g_d[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA
sorted_index = sorted(list(range(len(d_g_g_g_g_d))), key=lambda i: d_g_g_g_g_d[i, [0, 4, 5, 1, 2, 3]].tolist())
d_g_g_g_g_d = d_g_g_g_g_d[sorted_index]

with open(save_prefix + '4/4-3-3-3-3-4_idx.pickle', 'wb') as out_file:
    # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    target_metapaths_mapping = {}
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_g_g_d) and d_g_g_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
        target_metapaths_mapping[target_idx] = d_g_g_g_g_d[left:right, ::-1]
        # 更新 left 为 right
        left = right
    # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    pickle.dump(target_metapaths_mapping, out_file)

# 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
with open(save_prefix + '4/4-3-3-3-3-4.adjlist', 'w') as out_file:
    # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    left = 0
    right = 0
    for target_idx in target_idx_lists[4]:
        # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
        while right < len(d_g_g_g_g_d) and d_g_g_g_g_d[right, 0] == target_idx + offset_list[4]:
            right += 1
        # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
        neighbors = d_g_g_g_g_d[left:right, -1] - offset_list[4]
        neighbors = list(map(str, neighbors))
        if len(neighbors) > 0:
            out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
        else:
            out_file.write('{}\n'.format(target_idx))
        left = right


print(20)
gc.collect()