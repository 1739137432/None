# miRNA之间的相邻关系。
import csv

import numpy as np
import pandas as pd
import pickle

def g_d_l_l_d_g():
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

    lncRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/lncRNASim.csv', encoding='utf-8',delimiter=',', names=['lncRNAID', 'adjacentID'])

    lncRNA_disease_list = {}
    with open('lncRNA_disease_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            lncRNA_disease_list[key] = value
    disease_miRNA_list = {}
    with open('disease_miRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_miRNA_list[key] = value
    disease_circRNA_list = {}
    with open('disease_circRNA_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_circRNA_list[key] = value
    disease_gene_list = {}
    with open('disease_gene_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = int(row[0])
            value = np.array(row[1:], dtype=int)
            disease_gene_list[key] = value


     # 2-2
    lncRNA_adjacent[:] += num_miRNA + num_circRNA
    l_l = np.array(lncRNA_adjacent)
    sorted_index = sorted(list(range(len(l_l))), key=lambda i: l_l[i].tolist())
    l_l = l_l[sorted_index]

    # D-L-L-D
    d_l_l_d = []
    for l1, l2 in l_l:
        if len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]) == 0 or len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]) == 0:
            continue
        candidate_d1_list = np.random.choice(len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l1 - num_miRNA - num_circRNA])),replace=False)
        candidate_d1_list = lncRNA_disease_list[l1 - num_miRNA - num_circRNA][candidate_d1_list]
        candidate_d2_list = np.random.choice(len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA]),int(0.5 * len(lncRNA_disease_list[l2 - num_miRNA - num_circRNA])),replace=False)
        candidate_d2_list = lncRNA_disease_list[l2 - num_miRNA - num_circRNA][candidate_d2_list]
        d_l_l_d.extend([(d1, l1, l2, d2) for d1 in candidate_d1_list for d2 in candidate_d2_list])
        # d_l_l_d.extend([(d1, l1, l2, d2) for d1 in lncRNA_disease_list[l1 - num_miRNA - num_circRNA] for d2 in lncRNA_disease_list[l2 - num_miRNA - num_circRNA]])
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

    # # M-D-L-L-D-M
    # m_d_l_l_d_m = []
    # for d1, l2, l3, d4 in d_l_l_d:
    #     if len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
    #         continue
    #     candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_m1_list = disease_miRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m1_list]
    #     candidate_m2_list = np.random.choice(len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_m2_list = disease_miRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_m2_list]
    #     m_d_l_l_d_m.extend([(m1, d1, l2, l3, d4, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
    # m_d_l_l_d_m = np.array(m_d_l_l_d_m)
    # # m_d_l_l_d_m[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
    # # m_d_l_l_d_m[:, [2,3]] += num_miRNA + num_circRNA
    # sorted_index = sorted(list(range(len(m_d_l_l_d_m))), key=lambda i: m_d_l_l_d_m[i, [0, 4, 5, 1, 2, 3]].tolist())
    # m_d_l_l_d_m = m_d_l_l_d_m[sorted_index]
    #
    # with open(save_prefix + '0/0-4-2-2-4-0_idx.pickle', 'wb') as out_file:
    #     # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    #     target_metapaths_mapping = {}
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[0]:
    #         # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(m_d_l_l_d_m) and m_d_l_l_d_m[right, 0] == target_idx + offset_list[0]:
    #             right += 1
    #         # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
    #         target_metapaths_mapping[target_idx] = m_d_l_l_d_m[left:right, ::-1]
    #         # 更新 left 为 right
    #         left = right
    #     # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    #     pickle.dump(target_metapaths_mapping, out_file)
    #
    # # 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
    # with open(save_prefix + '0/0-4-2-2-4-0.adjlist', 'w') as out_file:
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[0]:
    #         # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(m_d_l_l_d_m) and m_d_l_l_d_m[right, 0] == target_idx + offset_list[0]:
    #             right += 1
    #         # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
    #         neighbors = m_d_l_l_d_m[left:right, -1] - offset_list[4]
    #         neighbors = list(map(str, neighbors))
    #         if len(neighbors) > 0:
    #             out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
    #         else:
    #             out_file.write('{}\n'.format(target_idx))
    #         left = right
    # # C-D-L-L-D-C
    # c_d_l_l_d_c = []
    # for d1, l2, l3, d4 in d_l_l_d:
    #     if len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
    #         continue
    #     candidate_c1_list = np.random.choice(len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_c1_list = disease_circRNA_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c1_list]
    #     candidate_c2_list = np.random.choice(len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_c2_list = disease_circRNA_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_c2_list]
    #     c_d_l_l_d_c.extend([(c1, d1, l2, l3, d4, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
    # c_d_l_l_d_c = np.array(c_d_l_l_d_c)
    # c_d_l_l_d_c[:, [0,5]] += num_miRNA
    # # c_d_l_l_d_c[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
    # # c_d_l_l_d_c[:, [2,3]] += num_miRNA + num_circRNA
    # sorted_index = sorted(list(range(len(c_d_l_l_d_c))), key=lambda i: c_d_l_l_d_c[i, [0, 4, 5, 1, 2, 3]].tolist())
    # c_d_l_l_d_c = c_d_l_l_d_c[sorted_index]
    #
    # with open(save_prefix + '1/1-4-2-2-4-1_idx.pickle', 'wb') as out_file:
    #     # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    #     target_metapaths_mapping = {}
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[1]:
    #         # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(c_d_l_l_d_c) and c_d_l_l_d_c[right, 0] == target_idx + offset_list[1]:
    #             right += 1
    #         # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
    #         target_metapaths_mapping[target_idx] = c_d_l_l_d_c[left:right, ::-1]
    #         # 更新 left 为 right
    #         left = right
    #     # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    #     pickle.dump(target_metapaths_mapping, out_file)
    #
    # # 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
    # with open(save_prefix + '1/1-4-2-2-4-1.adjlist', 'w') as out_file:
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[1]:
    #         # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(c_d_l_l_d_c) and c_d_l_l_d_c[right, 0] == target_idx + offset_list[1]:
    #             right += 1
    #         # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
    #         neighbors = c_d_l_l_d_c[left:right, -1] - offset_list[1]
    #         neighbors = list(map(str, neighbors))
    #         if len(neighbors) > 0:
    #             out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
    #         else:
    #             out_file.write('{}\n'.format(target_idx))
    #         left = right
    #
    #
    #
    #
    #
    # # G-D-L-L-D-G
    # g_d_l_l_d_g = []
    # for d1, l2, l3, d4 in d_l_l_d:
    #     if len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0 or len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]) == 0:
    #         continue
    #     candidate_g1_list = np.random.choice(len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_g1_list = disease_gene_list[d1 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g1_list]
    #     candidate_g2_list = np.random.choice(len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene]),int(0.5 * len(disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene])),replace=False)
    #     candidate_g2_list = disease_gene_list[d4 - num_miRNA - num_circRNA - num_lncRNA - num_gene][candidate_g2_list]
    #     g_d_l_l_d_g.extend([(g1, d1, l2, l3, d4, g2) for g1 in candidate_g1_list for g2 in candidate_g2_list])
    # g_d_l_l_d_g = np.array(g_d_l_l_d_g)
    # g_d_l_l_d_g[:, [0,5]] += num_miRNA + num_circRNA + num_lncRNA
    # # g_d_l_l_d_g[:, [1,4]] += num_miRNA + num_circRNA + num_lncRNA + num_gene
    # # g_d_l_l_d_g[:, [2,3]] += num_miRNA + num_circRNA
    # sorted_index = sorted(list(range(len(g_d_l_l_d_g))), key=lambda i: g_d_l_l_d_g[i, [0, 4, 5, 1, 2, 3]].tolist())
    # g_d_l_l_d_g = g_d_l_l_d_g[sorted_index]
    #
    # with open(save_prefix + '3/3-4-2-2-4-3_idx.pickle', 'wb') as out_file:
    #     # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
    #     target_metapaths_mapping = {}
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[3]:
    #         # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(g_d_l_l_d_g) and g_d_l_l_d_g[right, 0] == target_idx + offset_list[3]:
    #             right += 1
    #         # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
    #         target_metapaths_mapping[target_idx] = g_d_l_l_d_g[left:right, ::-1]
    #         # 更新 left 为 right
    #         left = right
    #     # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
    #     pickle.dump(target_metapaths_mapping, out_file)
    #
    # # 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
    # with open(save_prefix + '3/3-4-2-2-4-3.adjlist', 'w') as out_file:
    #     # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
    #     left = 0
    #     right = 0
    #     for target_idx in target_idx_lists[3]:
    #         # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
    #         while right < len(g_d_l_l_d_g) and g_d_l_l_d_g[right, 0] == target_idx + offset_list[3]:
    #             right += 1
    #         # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
    #         neighbors = g_d_l_l_d_g[left:right, -1] - offset_list[3]
    #         neighbors = list(map(str, neighbors))
    #         if len(neighbors) > 0:
    #             out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
    #         else:
    #             out_file.write('{}\n'.format(target_idx))
    #         left = right
