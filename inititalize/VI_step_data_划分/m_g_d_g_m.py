import csv

import numpy as np
import pandas as pd
import pickle

def m_g_d_g_m():
    save_prefix = 'VI_step_data_划分/alldatetest/'

    num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
    num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
    num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
    num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
    num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv', sep=':').shape[0] + 1

    # write all things
    # 包含两个数组，分别表示 miRNA circRNA lncRNA gene和疾病的索引列表
    target_idx_lists = [np.arange(num_miRNA), np.arange(num_circRNA), np.arange(num_lncRNA), np.arange(num_gene), np.arange(num_disease)]
    # 包含两个值，分别为 0 和 num_miRNA。这将用于在索引数组中调整索引值。
    offset_list = [0, num_miRNA, num_miRNA+num_circRNA, num_miRNA+num_circRNA+num_lncRNA, num_miRNA+num_circRNA+num_lncRNA+num_gene]


    disease_gene_list = {}
    with open('disease_gene_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_gene_list[key] = value
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

    gene_miRNA_list = {}
    with open('gene_miRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            gene_miRNA_list[key] = value
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