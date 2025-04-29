import csv
import gc
import pathlib
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io


def data_deal():
    save_prefix = '../output/relationship/VI_step_data_division/'
    miRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2mi_allinf.csv', encoding='utf-8',
                                delimiter=',',
                                names=['miRNAid', 'miRNAName', 'diseaseid', 'disease', 'database', 'pmid'])
    circRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2circ_allinf.csv', encoding='utf-8',
                                  delimiter=',',
                                  names=['circRNAid', 'circRNAName', 'diseaseid', 'disease', 'database', 'pmid'])
    lncRNA_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2lnc_allinf.csv', encoding='utf-8',
                                 delimiter=',',
                                 names=['lncRNAid', 'lncRNAName', 'diseaseid', 'disease', 'database', 'pmid'])
    gene_disease = pd.read_csv('../output/relationship/V_step_relationship/dis2gene_allind.csv', encoding='utf-8',
                               delimiter=',', names=['geneid', 'gene', 'diseaseid', 'disease', 'database', 'pmid'])
    miRNA_circRNA = pd.read_csv('../output/relationship/V_step_relationship/mir2circ_allinf.csv', encoding='utf-8',
                                delimiter=',',
                                names=['miRNAid', 'miRNAName', 'circRNAid', 'circRNA', 'database', 'pmid'])
    miRNA_lncRNA = pd.read_csv('../output/relationship/V_step_relationship/mir2lnc_allinf.csv', encoding='utf-8',
                               delimiter=',', names=['miRNAid', 'miRNAName', 'lncRNAid', 'lncRNA', 'database', 'pmid'])
    miRNA_gene = pd.read_csv('../output/relationship/V_step_relationship/mir2gene_allinf.csv', encoding='utf-8',
                             delimiter=',', names=['miRNAid', 'miRNAName', 'geneid', 'gene', 'database', 'pmid'])

    miRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/miRNASim.csv', encoding='utf-8',
                                 delimiter=',',
                                 names=['miRNAID', 'adjacentID'])
    miRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_similarity.csv', encoding='utf-8',
                            delimiter='\t', names=['similarity'])
    circRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/circRNASim.csv', encoding='utf-8',
                                   delimiter=',', names=['circRNAID', 'adjacentID'])
    circRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_similarity.csv', encoding='utf-8',
                              delimiter='\t', names=['similarity'])
    lncRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/lncRNASim.csv', encoding='utf-8',
                                  delimiter=',', names=['lncRNAID', 'adjacentID'])
    lncRNA_Sim = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_similarity.csv', encoding='utf-8',
                             delimiter='\t', names=['similarity'])
    gene_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/geneSim.csv', encoding='utf-8',
                                delimiter=',',
                                names=['geneID', 'adjacentID'])
    gene_Sim = pd.read_csv('../output/relationship/IV_step_similarity/gene_similarity.csv', encoding='utf-8',
                           delimiter='\t', names=['similarity'])
    disease_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj.csv', encoding='utf-8',
                                   delimiter=':', names=['diseaseID', 'adjacentID'])
    disease_Sim = pd.read_csv('../output/relationship/IV_step_similarity/disease_similarity.csv', encoding='utf-8',
                              delimiter='\t', names=['similarity'])
    print(1)

    num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
    num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
    num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
    num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
    num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv', sep=':').shape[0] + 1
    dis2circ_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2circ_train_val_test_idx.npz')
    # dis2circ_train_idx = dis2circ_train_val_test_idx['dis2circ_train_idx']
    # dis2circ_val_idx = dis2circ_train_val_test_idx['dis2circ_val_idx']
    # dis2circ_test_idx = dis2circ_train_val_test_idx['dis2circ_test_idx']
    dis2circ_train_idx = dis2circ_train_val_test_idx['train_idx']
    dis2circ_val_idx = dis2circ_train_val_test_idx['val_idx']
    dis2circ_test_idx = dis2circ_train_val_test_idx['test_idx']
    circRNA_disease = circRNA_disease.loc[dis2circ_train_idx].reset_index(drop=True)
    print(1)
    dis2lnc_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2lnc_train_val_test_idx.npz')
    # dis2lnc_train_idx = dis2lnc_train_val_test_idx['dis2lnc_train_idx']
    # dis2lnc_val_idx = dis2lnc_train_val_test_idx['dis2lnc_val_idx']
    # dis2lnc_test_idx = dis2lnc_train_val_test_idx['dis2lnc_test_idx']
    dis2lnc_train_idx = dis2lnc_train_val_test_idx['train_idx']
    dis2lnc_val_idx = dis2lnc_train_val_test_idx['val_idx']
    dis2lnc_test_idx = dis2lnc_train_val_test_idx['test_idx']
    lncRNA_disease = lncRNA_disease.loc[dis2lnc_train_idx].reset_index(drop=True)
    print(1)
    dis2mi_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2mi_train_val_test_idx.npz')
    # dis2mi_train_idx = dis2mi_train_val_test_idx['dis2mi_train_idx']
    # dis2mi_val_idx = dis2mi_train_val_test_idx['dis2mi_val_idx']
    # dis2mi_test_idx = dis2mi_train_val_test_idx['dis2mi_test_idx']
    dis2mi_train_idx = dis2mi_train_val_test_idx['train_idx']
    dis2mi_val_idx = dis2mi_train_val_test_idx['val_idx']
    dis2mi_test_idx = dis2mi_train_val_test_idx['test_idx']
    miRNA_disease = miRNA_disease.loc[dis2mi_train_idx].reset_index(drop=True)
    print(1)
    dis2gene_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2gene_train_val_test_idx.npz')
    # dis2gene_train_idx = dis2gene_train_val_test_idx['dis2gene_train_idx']
    # dis2gene_val_idx = dis2gene_train_val_test_idx['dis2gene_val_idx']
    # dis2gene_test_idx = dis2gene_train_val_test_idx['dis2gene_test_idx']

    dis2gene_train_idx = dis2gene_train_val_test_idx['train_idx']
    dis2gene_val_idx = dis2gene_train_val_test_idx['val_idx']
    dis2gene_test_idx = dis2gene_train_val_test_idx['test_idx']
    gene_disease = gene_disease.loc[dis2gene_train_idx].reset_index(drop=True)
    print(1)
    mi2circ_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/mi2circ_train_val_test_idx.npz')
    # mi2circ_train_idx = mi2circ_train_val_test_idx['mi2circ_train_idx']
    # mi2circ_val_idx = mi2circ_train_val_test_idx['mi2circ_val_idx']
    # mi2circ_test_idx = mi2circ_train_val_test_idx['mi2circ_test_idx']
    mi2circ_train_idx = mi2circ_train_val_test_idx['train_idx']
    mi2circ_val_idx = mi2circ_train_val_test_idx['val_idx']
    mi2circ_test_idx = mi2circ_train_val_test_idx['test_idx']
    miRNA_circRNA = miRNA_circRNA.loc[mi2circ_train_idx].reset_index(drop=True)
    print(1)
    mi2lnc_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/mi2lnc_train_val_test_idx.npz')
    # mi2lnc_train_idx = mi2lnc_train_val_test_idx['mi2lnc_train_idx']
    # mi2lnc_val_idx = mi2lnc_train_val_test_idx['mi2lnc_val_idx']
    # mi2lnc_test_idx = mi2lnc_train_val_test_idx['mi2lnc_test_idx']
    mi2lnc_train_idx = mi2lnc_train_val_test_idx['train_idx']
    mi2lnc_val_idx = mi2lnc_train_val_test_idx['val_idx']
    mi2lnc_test_idx = mi2lnc_train_val_test_idx['test_idx']
    miRNA_lncRNA = miRNA_lncRNA.loc[mi2lnc_train_idx].reset_index(drop=True)
    print(1)
    mi2gene_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/mi2gene_train_val_test_idx.npz')
    # mi2gene_train_idx = mi2gene_train_val_test_idx['mi2gene_train_idx']
    # mi2gene_val_idx = mi2gene_train_val_test_idx['mi2gene_val_idx']
    # mi2gene_test_idx = mi2gene_train_val_test_idx['mi2gene_test_idx']
    mi2gene_train_idx = mi2gene_train_val_test_idx['train_idx']
    mi2gene_val_idx = mi2gene_train_val_test_idx['val_idx']
    mi2gene_test_idx = mi2gene_train_val_test_idx['test_idx']
    miRNA_gene = miRNA_gene.loc[mi2gene_train_idx].reset_index(drop=True)
    print(1)
    # build the adjacency matrix
    #    0 for miRNA, 1 for circRNA, 2 for lncRNA, 3 for gene, 4 for disease
    dim = num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease
    # 构建零矩阵 一个长度为dim的一维数组。dtype=int指定了数组的数据类型为整数。
    # 从0索引到num_miRNA - 1索引的元素赋值为0。标记为类型1，表示miRNA。
    type_mask = np.zeros(dim, dtype=int)  #
    # 从num_miRNA索引到num_miRNA + num_circRNA - 1索引的元素赋值为1。标记为类型2，表示circRNA。
    type_mask[num_miRNA:num_miRNA + num_circRNA] = 1
    # 从num_miRNA + num_circRNA索引到num_miRNA + num_circRNA +  num_lncRNA - 1索引的元素赋值为2。标记为类型3，表示lncRNA。
    type_mask[num_miRNA + num_circRNA:num_miRNA + num_circRNA + num_lncRNA] = 2
    # 从num_miRNA + num_circRNA +  num_lncRNA索引到num_miRNA + num_circRNA +  num_lncRNA +  num_gene的元素赋值为3。这些元素被标记为类型4，表示基因。
    type_mask[num_miRNA + num_circRNA + num_lncRNA:num_miRNA + num_circRNA + num_lncRNA + num_gene] = 3
    # 从num_miRNA + num_circRNA +  num_lncRNA +  num_gene开始的元素赋值为4。标记为类型5，表示疾病。
    type_mask[num_miRNA + num_circRNA + num_lncRNA + num_gene:] = 4

    np.save(save_prefix + 'node_types.npy', type_mask)
    del type_mask

    # 构建零矩阵 一个长度为dim*dim的二维数组。dtype=int指定了数组的数据类型为整数。
    adjM = np.zeros((dim, dim), dtype=int)
    print(1)
    # 有mi2dis关系的两条边都设置为100
    for _, row in miRNA_disease.iterrows():
        mid = int(row['miRNAid'])
        did = num_miRNA + num_circRNA + num_lncRNA + num_gene + int(row['diseaseid'])
        adjM[mid, did] = 100
        adjM[did, mid] = 100
    # 有circ2dis关系的两条边都设置为100
    for _, row in circRNA_disease.iterrows():
        cid = num_miRNA + int(row['circRNAid'])
        did = num_miRNA + num_circRNA + num_lncRNA + num_gene + int(row['diseaseid'])
        adjM[cid, did] = 100
        adjM[did, cid] = 100
    # 有lnc2dis关系的两条边都设置为100
    for _, row in lncRNA_disease.iterrows():
        lid = num_miRNA + num_circRNA + int(row['lncRNAid'])
        did = num_miRNA + num_circRNA + num_lncRNA + num_gene + int(row['diseaseid'])
        adjM[lid, did] = 100
        adjM[did, lid] = 100
    # 有gene2dis关系的两条边都设置为100
    for _, row in gene_disease.iterrows():
        gid = num_miRNA + num_circRNA + num_lncRNA + int(row['geneid'])
        did = num_miRNA + num_circRNA + num_lncRNA + num_gene + int(row['diseaseid'])
        adjM[gid, did] = 100
        adjM[did, gid] = 100
    # 有mi2circ关系的两条边都设置为100
    for _, row in miRNA_circRNA.iterrows():
        mid = int(row['miRNAid'])
        cid = num_miRNA + int(row['circRNAid'])
        adjM[mid, cid] = 100
        adjM[cid, mid] = 100
    # 有mi2lnc关系的两条边都设置为100
    for _, row in miRNA_lncRNA.iterrows():
        mid = int(row['miRNAid'])
        lid = num_miRNA + num_circRNA + int(row['lncRNAid'])
        adjM[mid, lid] = 100
        adjM[lid, mid] = 100
    # 有mi2gene关系的两条边都设置为100
    for _, row in miRNA_gene.iterrows():
        mid = int(row['miRNAid'])
        gid = num_miRNA + num_circRNA + num_lncRNA + int(row['geneid'])
        adjM[mid, gid] = 100
        adjM[gid, mid] = 100
    print(2)
    # np.savetxt('matrix1.csv', adjM, delimiter=',')
    # 使用pd.concat函数将两个DataFrame miRNA_adjacent和miRNA_Sim按列合并，其中axis=1表示按列合并。
    miRNA_adjacent_Sim = pd.concat([miRNA_adjacent, miRNA_Sim], axis=1)
    # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
    for _, row in miRNA_adjacent_Sim.iterrows():
        # 根据mid和aid索引，在adjM数组中的相应位置填充值。
        mid = int(row['miRNAID'])
        aid = int(row['adjacentID'])
        # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
        if row['similarity'] > 0.9:
            adjM[mid, aid] = row['similarity'] * 100
    # np.savetxt('matrix2.csv', adjM, delimiter=',')
    # 使用pd.concat函数将两个DataFrame circRNA_adjacent和circRNA_Sim按列合并，其中axis=1表示按列合并。
    circRNA_adjacent_Sim = pd.concat([circRNA_adjacent, circRNA_Sim], axis=1)
    # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
    for _, row in circRNA_adjacent_Sim.iterrows():
        # 根据mid和aid索引，在adjM数组中的相应位置填充值。
        cid = int(row['circRNAID']) + num_miRNA
        aid = int(row['adjacentID']) + num_miRNA
        # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
        if row['similarity'] > 0.8:
            adjM[cid, aid] = row['similarity'] * 100
    # np.savetxt('matrix3.csv', adjM, delimiter=',')
    # 使用pd.concat函数将两个DataFrame lncRNA_adjacent和lncRNA_Sim按列合并，其中axis=1表示按列合并。
    lncRNA_adjacent_Sim = pd.concat([lncRNA_adjacent, lncRNA_Sim], axis=1)
    # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
    for _, row in lncRNA_adjacent_Sim.iterrows():
        # 根据mid和aid索引，在adjM数组中的相应位置填充值。
        lid = int(row['lncRNAID']) + num_miRNA + num_circRNA
        aid = int(row['adjacentID']) + num_miRNA + num_circRNA
        # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
        if row['similarity'] > 0.8:
         adjM[lid, aid] = row['similarity'] * 100
    # np.savetxt('matrix4.csv', adjM, delimiter=',')
    # 使用pd.concat函数将两个DataFrame gene_adjacent和gene_Sim按列合并，其中axis=1表示按列合并。
    gene_adjacent_Sim = pd.concat([gene_adjacent, gene_Sim], axis=1)
    # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
    for _, row in gene_adjacent_Sim.iterrows():
        # 根据mid和aid索引，在adjM数组中的相应位置填充值。
        gid = int(row['geneID']) + num_miRNA + num_circRNA + num_lncRNA
        aid = int(row['adjacentID']) + num_miRNA + num_circRNA + num_lncRNA
        # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
        if row['similarity'] > 0.8:
            adjM[gid, aid] = row['similarity'] * 100
        # 使用pd.concat函数将两个DataFrame gene_adjacent和gene_Sim按列合并，其中axis=1表示按列合并。
    disease_adjacent_Sim = pd.concat([disease_adjacent, disease_Sim], axis=1)
    # 在 adjM 矩阵中添加miRNA相邻性和相似性的矩阵。
    for _, row in disease_adjacent_Sim.iterrows():
        # 根据mid和aid索引，在adjM数组中的相应位置填充值。
        gid = int(row['diseaseID']) + num_miRNA + num_circRNA + num_lncRNA + num_gene
        aid = int(row['adjacentID']) + num_miRNA + num_circRNA + num_lncRNA + num_gene
        # row['similarity']获取当前行的'similarity'列的值，然后乘以100，并将结果填充到adjM数组的指定位置
        if row['similarity'] > 0.8:
            adjM[gid, aid] = row['similarity'] * 100
    # np.savetxt('matrix.txt', adjM)
    # # print(3)
    # # 创建一个名为 miRNA_disease_list 的字典。
    # # 字典的键是miRNA的索引，值是一个数组，表示与该miRNA相关的疾病的索引。
    # # 这些索引来自 adjM 矩阵中 i 行、从 num_miRNA 列到 num_miRNA + num_disease - 1 列的非零元素的索引。
    # # 第i行的          ·                               第num_miRNA列到num_miRNA+num_disease-1列
    miRNA_disease_list = {i: adjM[i,
                             num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[
        0] for i in range(num_miRNA)}
    with open('miRNA_disease_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in miRNA_disease_list.items():
            writer.writerow([key] + value.tolist())
    circRNA_disease_list = {i: adjM[num_miRNA + i,
                               num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[
        0] for i in range(num_circRNA)}
    with open('circRNA_disease_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in circRNA_disease_list.items():
            writer.writerow([key] + value.tolist())
    lncRNA_disease_list = {i: adjM[num_miRNA + num_circRNA + i,
                              num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[
        0] for i in range(num_lncRNA)}
    with open('lncRNA_disease_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in lncRNA_disease_list.items():
            writer.writerow([key] + value.tolist())
    gene_disease_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i,
                            num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[
        0] for i in range(num_gene)}
    with open('gene_disease_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in gene_disease_list.items():
            writer.writerow([key] + value.tolist())
    miRNA_circRNA_list = {i: adjM[i, num_miRNA: num_miRNA + num_circRNA].nonzero()[0] for i in range(num_miRNA)}
    with open('miRNA_circRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in miRNA_circRNA_list.items():
            writer.writerow([key] + value.tolist())
    miRNA_lncRNA_list = {i: adjM[i, num_miRNA + num_circRNA: num_miRNA + num_circRNA + num_lncRNA].nonzero()[0] for i in
                         range(num_miRNA)}
    with open('miRNA_lncRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in miRNA_lncRNA_list.items():
            writer.writerow([key] + value.tolist())

    miRNA_gene_list = {
        i: adjM[i, num_miRNA + num_circRNA + num_lncRNA: num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[0]
        for i
        in range(num_miRNA)}
    with open('miRNA_gene_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in miRNA_gene_list.items():
            writer.writerow([key] + value.tolist())
    # # 创建一个名为 disease_miRNA_list 的字典。
    # # 字典的键是疾病的索引，值是一个数组，表示与该疾病相关的miRNA的索引。
    # # 这些索引来自 adjM 矩阵中 num_miRNA + i 行、从第一列到 num_miRNA - 1 列的非零元素的索引。
    # # 第num_miRNA + i行的                              第一列到num_miRNA - 1列
    disease_miRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, : num_miRNA].nonzero()[0] for i
                          in
                          range(num_disease)}

    with open('disease_miRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in disease_miRNA_list.items():
            writer.writerow([key] + value.tolist())
    disease_circRNA_list = {
        i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i, num_miRNA: num_miRNA + num_circRNA].nonzero()[0]
        for i
        in range(num_disease)}
    with open('disease_circRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in disease_circRNA_list.items():
            writer.writerow([key] + value.tolist())
    disease_lncRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i,
                              num_miRNA + num_circRNA: num_miRNA + num_circRNA + num_lncRNA].nonzero()[0] for i in
                           range(num_disease)}
    with open('disease_lncRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in disease_lncRNA_list.items():
            writer.writerow([key] + value.tolist())
    disease_gene_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i,
                            num_miRNA + num_circRNA + num_lncRNA: num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[
        0] for i in range(num_disease)}
    with open('disease_gene_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in disease_gene_list.items():
            writer.writerow([key] + value.tolist())
    circRNA_miRNA_list = {i: adjM[num_miRNA + i, : num_miRNA].nonzero()[0] for i in range(num_circRNA)}
    with open('circRNA_miRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in circRNA_miRNA_list.items():
            writer.writerow([key] + value.tolist())
    lncRNA_miRNA_list = {i: adjM[num_miRNA + num_circRNA + i, : num_miRNA].nonzero()[0] for i in range(num_lncRNA)}
    with open('lncRNA_miRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in lncRNA_miRNA_list.items():
            writer.writerow([key] + value.tolist())
    gene_miRNA_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i, : num_miRNA].nonzero()[0] for i in
                       range(num_gene)}
    with open('gene_miRNA_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in gene_miRNA_list.items():
            writer.writerow([key] + value.tolist())
    # # # 创建一个名为 disease_miRNA_list 的字典。
    # # 创建一个名为 miRNA_adjacent_list 的字典。
    # # 字典的键是miRNA的索引，值是一个数组，表示与该miRNA相邻的miRNA的索引。
    # # 这些索引来自 adjM 矩阵中 i 行、从第一列到 num_miRNA - 1 列的非零元素的索引。
    # # 第i行的                                          第一列到num_miRNA - 1列
    miRNA_adjacent_list = {i: adjM[i, :num_miRNA].nonzero()[0] for i in range(num_miRNA)}
    with open('miRNA_adjacent_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in miRNA_adjacent_list.items():
            writer.writerow([key] + value.tolist())
    circRNA_adjacent_list = {i: adjM[num_miRNA + i, num_miRNA: num_miRNA + num_circRNA].nonzero()[0] for i in
                             range(num_circRNA)}
    with open('circRNA_adjacent_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in circRNA_adjacent_list.items():
            writer.writerow([key] + value.tolist())
    lncRNA_adjacent_list = {
        i: adjM[num_miRNA + num_circRNA + i, num_miRNA + num_circRNA: num_miRNA + num_circRNA + num_lncRNA].nonzero()[0]
        for
        i in range(num_lncRNA)}
    with open('lncRNA_adjacent_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in lncRNA_adjacent_list.items():
            writer.writerow([key] + value.tolist())
    gene_adjacent_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + i,
                             num_miRNA + num_circRNA + num_lncRNA: num_miRNA + num_circRNA + num_lncRNA + num_gene].nonzero()[
        0] for i in range(num_gene)}
    with open('gene_adjacent_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in gene_adjacent_list.items():
            writer.writerow([key] + value.tolist())
    disease_adjacent_list = {i: adjM[num_miRNA + num_circRNA + num_lncRNA + num_gene + i,
                                num_miRNA + num_circRNA + num_lncRNA + num_gene:num_miRNA + num_circRNA + num_lncRNA + num_gene + num_disease].nonzero()[
        0] for i in range(num_disease)}
    with open('disease_adjacent_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in disease_adjacent_list.items():
            writer.writerow([key] + value.tolist())
    scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
