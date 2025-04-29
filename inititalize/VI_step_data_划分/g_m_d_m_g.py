import csv

import numpy as np
import pandas as pd
import pickle

def g_m_d_m_g():
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


    disease_miRNA_list = {}
    with open('disease_miRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_miRNA_list[key] = value
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

    miRNA_circRNA_list = {}
    with open('miRNA_circRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            miRNA_circRNA_list[key] = value
    # C-M-D-M-C
    c_m_d_m_c = []
    for m1, d, m2 in m_d_m:
        if len(miRNA_circRNA_list[m1]) == 0 or len(miRNA_circRNA_list[m2]) == 0:
            continue
        candidate_c1_list = np.random.choice(len(miRNA_circRNA_list[m1]), int(0.1 * len(miRNA_circRNA_list[m1])),replace=False)
        candidate_c1_list = miRNA_circRNA_list[m1][candidate_c1_list]
        candidate_c2_list = np.random.choice(len(miRNA_circRNA_list[m2]),int(0.1 * len(miRNA_circRNA_list[m2])),replace=False)
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

    miRNA_lncRNA_list = {}
    with open('miRNA_lncRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            miRNA_lncRNA_list[key] = value

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

    miRNA_gene_list = {}
    with open('miRNA_gene_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            miRNA_gene_list[key] = value
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
