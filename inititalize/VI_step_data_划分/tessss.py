import csv
import os
import pickle
import random

import numpy as np
import pandas as pd



miRNA_adjacent_list = {}
with open('miRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_adjacent_list[key] = value

miRNA_circRNA_list = {}
with open('miRNA_circRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_circRNA_list[key] = value
miRNA_lncRNA_list = {}
with open('miRNA_lncRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_lncRNA_list[key] = value
miRNA_gene_list = {}
with open('miRNA_gene_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_gene_list[key] = value
miRNA_disease_list = {}
with open('miRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_disease_list[key] = value
circRNA_adjacent_list = {}
with open('circRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        circRNA_adjacent_list[key] = value

circRNA_miRNA_list = {}
with open('circRNA_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        circRNA_miRNA_list[key] = value
circRNA_disease_list = {}
with open('circRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        circRNA_disease_list[key] = value
lncRNA_adjacent_list = {}
with open('lncRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_adjacent_list[key] = value

lncRNA_miRNA_list = {}
with open('lncRNA_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_miRNA_list[key] = value
lncRNA_disease_list = {}
with open('lncRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_disease_list[key] = value
gene_adjacent_list = {}
with open('gene_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_adjacent_list[key] = value

gene_miRNA_list = {}
with open('gene_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_miRNA_list[key] = value
gene_disease_list = {}
with open('gene_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_disease_list[key] = value

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
disease_lncRNA_list = {}
with open('disease_lncRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_lncRNA_list[key] = value
disease_gene_list = {}
with open('disease_gene_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_gene_list[key] = value
disease_adjacent_list = {}
with open('disease_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_adjacent_list[key] = value

dis2mi = []
dis2circ = []
dis2lnc = []
dis2gene = []
mi2circ = []
mi2lnc = []
mi2gene = []

def creat_metapath(metapath,maxnumber,allmaxrepeat):
    save_prefix = '../output/relationship/VI_step_data_划分/'
    # save_prefix = ''
    num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
    num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
    num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
    num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
    num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv', sep=":").shape[0] + 1

    # write all things
    # 包含两个数组，分别表示 miRNA circRNA lncRNA gene和疾病的索引列表
    target_idx_lists = [np.arange(num_miRNA), np.arange(num_circRNA), np.arange(num_lncRNA), np.arange(num_gene),
                        np.arange(num_disease)]
    # 包含两个值，分别为 0 和 num_miRNA。这将用于在索引数组中调整索引值。
    offset_list = [0, num_miRNA, num_miRNA + num_circRNA, num_miRNA + num_circRNA + num_lncRNA,
                   num_miRNA + num_circRNA + num_lncRNA + num_gene]

    # metapath = "m_d_m"
    # # if len(metapath) ==5:
    # #     one = metapath[0]
    # #     two = metapath[2]
    # #     three = metapath[4]
    # #     print(one + two + three)



    print("start......")
    paths = []
    repeat = 0
    qs = []
    for m in metapath.split("_"):
        qs.append(m)
    while len(paths) < maxnumber:
        path = []
        one = 0
        # print(len(qs))
        if qs[0] == "m":
            if qs[1] == "m":
                one = random.choice(list(miRNA_adjacent_list.keys()))
            elif qs[1] == "c":
                one = random.choice(list(miRNA_circRNA_list.keys()))
            elif qs[1] == "l":
                one = random.choice(list(miRNA_lncRNA_list.keys()))
            elif qs[1] == "g":
                one = random.choice(list(miRNA_gene_list.keys()))
            else:
                one = random.choice(list(miRNA_disease_list.keys()))
        elif qs[0] == "c":
            if qs[1] == "m":
                one = random.choice(list(circRNA_miRNA_list.keys()))
            elif qs[1] == "c":
                one = random.choice(list(circRNA_adjacent_list.keys()))
            elif qs[1] == "d":
                one = random.choice(list(circRNA_disease_list.keys()))
        elif qs[0] == "l":
            if qs[1] == "m":
                one = random.choice(list(lncRNA_miRNA_list.keys()))
            elif qs[1] == "l":
                one = random.choice(list(lncRNA_adjacent_list.keys()))
            elif qs[1] == "d":
                one = random.choice(list(lncRNA_disease_list.keys()))
        elif qs[0] == "g":
            if qs[1] == "m":
                one = random.choice(list(gene_miRNA_list.keys()))
            elif qs[1] == "g":
                one = random.choice(list(gene_adjacent_list.keys()))
            elif qs[1] == "d":
                one = random.choice(list(gene_disease_list.keys()))
        else:
            if qs[1] == "m":
                one = random.choice(list(disease_miRNA_list.keys()))
            elif qs[1] == "c":
                one = random.choice(list(disease_circRNA_list.keys()))
            elif qs[1] == "l":
                one = random.choice(list(disease_lncRNA_list.keys()))
            elif qs[1] == "g":
                one = random.choice(list(disease_gene_list.keys()))
            else:
                one = random.choice(list(disease_adjacent_list.keys()))
        path.append(one)
        status  = True
        for i in range(1, len(qs)):
            if qs[i-1] == "m":
                if qs[i] == "m":
                    if len(miRNA_adjacent_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(miRNA_adjacent_list[one])
                elif qs[i] == "c":
                    if len(miRNA_circRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(miRNA_circRNA_list[one])
                elif qs[i] == "l":
                    if len(miRNA_lncRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(miRNA_lncRNA_list[one])
                elif qs[i] == "g":
                    if len(miRNA_gene_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(miRNA_gene_list[one])
                else:
                    if len(miRNA_disease_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(miRNA_disease_list[one])
            elif qs[i-1] == "c":
                if qs[i] == "m":
                    if len(circRNA_miRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(circRNA_miRNA_list[one])
                elif qs[i] == "c":
                    if len(circRNA_adjacent_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(circRNA_adjacent_list[one])
                elif qs[i] == "d":
                    if len(circRNA_disease_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(circRNA_disease_list[one])
            elif qs[i-1] == "l":
                if qs[i] == "m":
                    if len(lncRNA_miRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(lncRNA_miRNA_list[one])
                elif qs[i] == "l":
                    if len(lncRNA_adjacent_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(lncRNA_adjacent_list[one])
                elif qs[i] == "d":
                    if len(lncRNA_disease_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(lncRNA_disease_list[one])
            elif qs[i-1] == "g":
                if qs[i] == "m":
                    if len(gene_miRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(gene_miRNA_list[one])
                elif qs[i] == "g":
                    if len(gene_adjacent_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(gene_adjacent_list[one])
                elif qs[i] == "d":
                    if len(gene_disease_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(gene_disease_list[one])
            else:
                if qs[i] == "m":
                    if len(disease_miRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(disease_miRNA_list[one])
                elif qs[i] == "c":
                    if len(disease_circRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(disease_circRNA_list[one])
                elif qs[i] == "l":
                    if len(disease_lncRNA_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(disease_lncRNA_list[one])
                elif qs[i] == "g":
                    if len(disease_gene_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(disease_gene_list[one])
                else:
                    if len(disease_adjacent_list[one]) == 0:
                        status = False
                        break
                    else:
                        one = random.choice(disease_adjacent_list[one])
            path.append(one)
        if status ==False:
            continue
        for i in range(0, len(qs)):
            # print(i)
            if qs[i] == "m":
                continue
            elif qs[i] == "c":
                path[i] = path[i] + num_miRNA
            elif qs[i] == "l":
                path[i] = path[i] + num_miRNA + num_circRNA
            elif qs[i] == "g":
                path[i] = path[i] + num_miRNA + num_circRNA + num_lncRNA
            else:
                path[i] = path[i] + num_miRNA + num_circRNA + num_lncRNA + num_gene
        if path not in paths and path[::-1] not in paths:
            paths.append(path)
            repeat = 0
        else:
            if repeat > allmaxrepeat:
                break
            repeat += 1
    paths = np.array(paths)
    sorted_index = sorted(list(range(len(paths))), key=lambda i: paths[i, [0]].tolist())
    paths = paths[sorted_index]
    print(paths)
    path_one = 0
    if qs[0] == "m":
        path_one = 0
    elif qs[0] == "c":
        path_one = 1
    elif qs[0] == "l":
        path_one = 2
    elif qs[0] == "g":
        path_one = 3
    else:
        path_one = 4
    path_two = ""
    for i in range(0, len(qs)-1):
        if qs[i] == "m":
            path_two = path_two + "0-"
        elif qs[i] == "c":
            path_two = path_two + "1-"
        elif qs[i] == "l":
            path_two = path_two + "2-"
        elif qs[i] == "g":
            path_two = path_two + "3-"
        else:
            path_two = path_two + "4-"
    if qs[len(qs)-1] == "m":
        path_two = path_two + "0"
    elif qs[len(qs)-1] == "c":
        path_two = path_two + "1"
    elif qs[len(qs)-1] == "l":
        path_two = path_two + "2"
    elif qs[len(qs)-1] == "g":
        path_two = path_two + "3"
    else:
        path_two = path_two + "4"



    # edge_metapath_idx_array = metapath_indices_mapping[metapath]
    # print(paths)
    # print(paths[0, 0])
    if not os.path.exists(save_prefix + str(path_one)):
        os.makedirs(save_prefix + str(path_one))
    with open(save_prefix + str(path_one) + '/' + path_two + '_idx.pickle', 'wb') as out_file:
        # 创建一个名为 target_metapaths_mapping 的字典，将在接下来的步骤中用于存储目标索引与元路径索引的映射。
        target_metapaths_mapping = {}
        # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界。
        left = 0
        right = 0
        for target_idx in target_idx_lists[path_one]:
            # 在循环中，right 增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
            while right < len(paths) and paths[right, 0] == target_idx + offset_list[path_one]:
                right += 1
            # print(right)
            # 将目标索引 target_idx 与对应的元路径索引数组添加到 target_metapaths_mapping 字典中
            target_metapaths_mapping[target_idx] = paths[left:right, ::-1]
            # 更新 left 为 right
            left = right
        # 使用 pickle 模块将 target_metapaths_mapping 字典保存到刚刚打开的文件中。
        pickle.dump(target_metapaths_mapping, out_file)

    # 打开一个.adjlist 格式的文件，文件名使用 save_prefix、索引 i、元路径字符串和 '.adjlist' 构建，用于保存邻接列表。
    with open(save_prefix + str(path_one) + '/' + path_two +  '.adjlist', 'w') as out_file:
        # 初始化 left 和 right，表示在 edge_metapath_idx_array 中的左右边界
        left = 0
        right = 0
        for target_idx in target_idx_lists[path_one]:
            # 在循环中，right增加，直到 edge_metapath_idx_array[right, 0] 不等于 target_idx + offset_list[i]。
            while right < len(paths) and paths[right, 0] == target_idx + offset_list[path_one]:
                right += 1
            # 获取邻居索引，并将它们写入邻接列表文件中。就要最后一列的
            neighbors = paths[left:right, -1] - offset_list[path_one]
            # print(neighbors)
            neighbors = list(map(str, neighbors))
            if len(neighbors) > 0:
                out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
            else:
                out_file.write('{}\n'.format(target_idx))
            left = right
    print("end.......")

    file_path = save_prefix + str(path_one) + '/' + path_two + '_idx.pickle'
    in_file = open(file_path, 'rb')
    pathss = pickle.load(in_file)
    print(pathss)
metapaths = [["m_m"
    , "m_c_m"
    ,
             "m_l_m", "m_g_m", "m_d_m",
                            "m_c_c_m", "m_l_l_m", "m_g_g_m", "m_d_d_m",
                            "m_c_d_c_m",
                            "m_l_d_l_m", "m_g_d_g_m", "m_d_c_d_m",
                            "m_d_l_d_m", "m_d_g_d_m",
                            "m_c_d_d_c_m", "m_l_d_d_l_m",
                            "m_g_d_d_g_m", "m_d_c_c_d_m",
                            "m_d_l_l_d_m", "m_d_g_g_d_m"],

                            ["c_c", "c_m_c", "c_d_c", "c_m_m_c",
                            "c_d_d_c",
                            "c_m_l_m_c", "c_m_g_m_c", "c_m_d_m_c",
                            "c_d_m_d_c", "c_d_l_d_c",
                            "c_d_g_d_c", "c_m_l_l_m_c",
                            "c_m_g_g_m_c", "c_m_d_d_m_c",
                            "c_d_m_m_d_c", "c_d_l_l_d_c",
                            "c_d_g_g_d_c"],

                            ["l_l", "l_m_l", "l_d_l", "l_m_m_l",
                            "l_d_d_l",
                            "l_m_c_m_l", "l_m_g_m_l", "l_m_d_m_l",
                            "l_d_m_d_l", "l_d_c_d_l",
                            "l_d_g_d_l", "l_m_c_c_m_l",
                            "l_m_g_g_m_l", "l_m_d_d_m_l",
                            "l_d_m_m_d_l", "l_d_c_c_d_l",
                            "l_d_g_g_d_l"],

                            ["g_g", "g_m_g", "g_d_g", "g_m_m_g",
                            "g_d_d_g",
                            "g_m_c_m_g", "g_m_l_m_g", "g_m_d_m_g",
                            "g_d_m_d_g", 'g_d_c_d_g',
                            "g_d_l_d_g", "g_m_c_c_m_g",
                            "g_m_l_l_m_g", "g_m_d_d_m_g",
                            "g_d_m_m_d_g", "g_d_c_c_d_g",
                            "g_d_l_l_d_g"],

                            ["d_d", "d_m_d", "d_c_d", "d_l_d", "d_g_d",
                            'd_m_m_d', "d_c_c_d", "d_l_l_d", "d_g_g_d",
                            'd_m_c_m_d',
                            "d_m_l_m_d", "d_m_g_m_d", "d_c_m_c_d",
                            "d_l_m_l_d", "d_g_m_g_d",
                            "d_m_c_c_m_d", "d_m_l_l_m_d",
                            'd_m_g_g_m_d', "d_c_m_m_c_d",
                            "d_l_m_m_l_d", "d_g_m_m_g_d"]
                            ]
def birmetapath_m(number,repeat):
    print("开始")
    for metapath in metapaths[0]:
        print(metapath)
        creat_metapath(metapath,number,repeat)
def birmetapath_c(number,repeat):
    print("开始")
    for metapath in metapaths[1]:
        print(metapath)
        creat_metapath(metapath,number,repeat)
def birmetapath_l(number,repeat):
    print("开始")
    for metapath in metapaths[2]:
        print(metapath)
        creat_metapath(metapath,number,repeat)
def birmetapath_g(number,repeat):
    print("开始")
    for metapath in metapaths[3]:
        print(metapath)
        creat_metapath(metapath,number,repeat)
def birmetapath_d(number,repeat):
    print("开始")
    for metapath in metapaths[4]:
        print(metapath)
        creat_metapath(metapath,number,repeat)
# birmetapath(10,10)