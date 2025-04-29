# miRNA之间的相邻关系。
import csv

import numpy as np
import pandas as pd
import pickle

def d_c_m_c_d():
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

    miRNA_circRNA_list = {}
    with open('miRNA_circRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            miRNA_circRNA_list[key] = value
     # 1-0-1
    c_m_c = []
    for m ,c_list in miRNA_circRNA_list.items():
        c_m_c.extend([(c1, m, c2) for c1 in c_list for c2 in c_list])
    c_m_c = np.array(c_m_c)
    c_m_c[:, [0,2]] += num_miRNA
    sorted_index = sorted(list(range(len(c_m_c))), key=lambda i: c_m_c[i, [0, 2, 1]].tolist())
    c_m_c = c_m_c[sorted_index]
    print("c_m_c")
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

    circRNA_disease_list = {}
    with open('circRNA_disease_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            circRNA_disease_list[key] = value
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
    print("d_c_m_c_d")
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
