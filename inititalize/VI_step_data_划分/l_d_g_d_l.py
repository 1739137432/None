import csv

import numpy as np
import pandas as pd
import pickle

def l_d_g_d_l():
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

    gene_disease_list = {}
    with open('gene_disease_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            gene_disease_list[key] = value
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

    disease_miRNA_list = {}
    with open('disease_miRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_miRNA_list[key] = value
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

    disease_circRNA_list = {}
    with open('disease_circRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_circRNA_list[key] = value
    # C-D-G-D-C
    c_d_g_d_c = []
    for d1, g, d2 in d_g_d:
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

    disease_lncRNA_list = {}
    with open('disease_lncRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_lncRNA_list[key] = value

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
