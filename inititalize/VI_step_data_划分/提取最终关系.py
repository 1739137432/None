import csv
import pickle

import numpy as np
import pandas as pd

num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv', sep=":").shape[0] + 1

# write all things
# 包含两个数组，分别表示 miRNA circRNA lncRNA gene和疾病的索引列表
# target_idx_lists = [np.arange(num_miRNA), np.arange(num_circRNA), np.arange(num_lncRNA), np.arange(num_gene),
#                     np.arange(num_disease)]
# 包含两个值，分别为 0 和 num_miRNA。这将用于在索引数组中调整索引值。
offset_list = [0, num_miRNA, num_miRNA + num_circRNA, num_miRNA + num_circRNA + num_lncRNA,
               num_miRNA + num_circRNA + num_lncRNA + num_gene]
metapaths = [
    # "m_m"
    # ,
             "m_c_m"
    ,
             "m_l_m", "m_g_m", "m_d_m",
                            "m_c_c_m", "m_l_l_m", "m_g_g_m", "m_d_d_m",
                            "m_c_d_c_m",
                            "m_l_d_l_m", "m_g_d_g_m", "m_d_c_d_m",
                            "m_d_l_d_m", "m_d_g_d_m",
                            "m_c_d_d_c_m", "m_l_d_d_l_m",
                            "m_g_d_d_g_m", "m_d_c_c_d_m",
                            "m_d_l_l_d_m", "m_d_g_g_d_m",

                            "c_c", "c_m_c", "c_d_c", "c_m_m_c",
                            "c_d_d_c",
                            "c_m_l_m_c", "c_m_g_m_c", "c_m_d_m_c",
                            "c_d_m_d_c", "c_d_l_d_c",
                            "c_d_g_d_c", "c_m_l_l_m_c",
                            "c_m_g_g_m_c", "c_m_d_d_m_c",
                            "c_d_m_m_d_c", "c_d_l_l_d_c",
                            "c_d_g_g_d_c",

                            "l_l", "l_m_l", "l_d_l", "l_m_m_l",
                            "l_d_d_l",
                            "l_m_c_m_l", "l_m_g_m_l", "l_m_d_m_l",
                            "l_d_m_d_l", "l_d_c_d_l",
                            "l_d_g_d_l", "l_m_c_c_m_l",
                            "l_m_g_g_m_l", "l_m_d_d_m_l",
                            "l_d_m_m_d_l", "l_d_c_c_d_l",
                            "l_d_g_g_d_l",

                            "g_g", "g_m_g", "g_d_g", "g_m_m_g",
                            "g_d_d_g",
                            "g_m_c_m_g", "g_m_l_m_g", "g_m_d_m_g",
                            "g_d_m_d_g", 'g_d_c_d_g',
                            "g_d_l_d_g", "g_m_c_c_m_g",
                            "g_m_l_l_m_g", "g_m_d_d_m_g",
                            "g_d_m_m_d_g", "g_d_c_c_d_g",
                            "g_d_l_l_d_g",

                            "d_d", "d_m_d", "d_c_d", "d_l_d", "d_g_d",
                            'd_m_m_d', "d_c_c_d", "d_l_l_d", "d_g_g_d",
                            'd_m_c_m_d',
                            "d_m_l_m_d", "d_m_g_m_d", "d_c_m_c_d",
                            "d_l_m_l_d", "d_g_m_g_d",
                            "d_m_c_c_m_d", "d_m_l_l_m_d",
                            'd_m_g_g_m_d', "d_c_m_m_c_d",
                            "d_l_m_m_l_d", "d_g_m_m_g_d"
                            ]

dis2mi = []
dis2circ = []
dis2lnc = []
dis2gene = []
mi2circ = []
mi2lnc = []
mi2gene = []
def ti():
    for metapath in metapaths:
        qs = []
        for m in metapath.split("_"):
            qs.append(m)
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
        for i in range(0, len(qs) - 1):
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
        if qs[len(qs) - 1] == "m":
            path_two = path_two + "0"
        elif qs[len(qs) - 1] == "c":
            path_two = path_two + "1"
        elif qs[len(qs) - 1] == "l":
            path_two = path_two + "2"
        elif qs[len(qs) - 1] == "g":
            path_two = path_two + "3"
        else:
            path_two = path_two + "4"
        save_prefix = '../output/relationship/VI_step_data_划分/'
        file_path = save_prefix + str(path_one) + '/' + path_two + '_idx.pickle'

        in_file = open(file_path, 'rb')
        paths = pickle.load(in_file)
        # print(paths)
        for i in range(1, len(qs)):
            if qs[i - 1] == "m":
                if qs[i] == "d":
                    for key,patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1],path[i] - offset_list[4], ]
                            if tem not in dis2mi:
                                dis2mi.append(tem)
                elif qs[i] == "c":
                    # print(paths)
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1], path[i] - offset_list[1]]
                            if tem not in mi2circ:
                                mi2circ.append(tem)
                elif qs[i] == "l":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1], path[i] - offset_list[2]]
                            if tem not in mi2lnc:
                                mi2lnc.append(tem)
                elif qs[i] == "g":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1], path[i] - offset_list[3]]
                            if tem not in mi2gene:
                                mi2gene.append(tem)
            elif qs[i - 1] == "c":
                if qs[i] == "m":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i], path[i - 1] - offset_list[1]]
                            if tem not in mi2circ:
                                mi2circ.append(tem)
                elif qs[i] == "d":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1] - offset_list[1], path[i] - offset_list[4]]
                            if tem not in dis2circ:
                                dis2circ.append(tem)
            elif qs[i - 1] == "l":
                if qs[i] == "m":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i], path[i - 1] - offset_list[2]]
                            if tem not in mi2lnc:
                                mi2lnc.append(tem)
                elif qs[i] == "d":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1] - offset_list[2], path[i] - offset_list[4]]
                            if tem not in dis2lnc:
                                dis2lnc.append(tem)
            elif qs[i - 1] == "g":
                if qs[i] == "m":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i], path[i - 1] - offset_list[3]]
                            if tem not in mi2gene:
                                mi2gene.append(tem)
                elif qs[i] == "d":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i - 1] - offset_list[3],path[i] - offset_list[4]]
                            if tem not in dis2gene:
                                dis2gene.append(tem)
            else:
                if qs[i] == "m":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i],path[i - 1] - offset_list[4]]
                            if tem not in dis2mi:
                                dis2mi.append(tem)
                elif qs[i] == "c":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i] - offset_list[1],path[i - 1] - offset_list[4]]
                            if tem not in dis2circ:
                                dis2circ.append(tem)
                elif qs[i] == "l":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i] - offset_list[2],path[i - 1] - offset_list[4]]
                            if tem not in dis2lnc:
                                dis2lnc.append(tem)
                elif qs[i] == "g":
                    for key, patha in paths.items():
                        for path in patha:
                            tem = [path[i] - offset_list[3],path[i - 1] - offset_list[4]]
                            if tem not in dis2gene:
                                dis2gene.append(tem)
        in_file.close()
        # print(paths)
        # print(mi2circ)
    pre= "../output/relationship/VI_step_data_划分/"
    # 分别提取两列数据
    dis = [row[1] for row in dis2mi]
    mi = [row[0] for row in dis2mi]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"dis2mi_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dis)):
            writer.writerow([mi[i],dis[i]])

    np.save(pre + "disease_miRNA.npy", dis2mi)


    dis = [row[1] for row in dis2circ]
    circ = [row[0] for row in dis2circ]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"dis2circ_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dis)):
            writer.writerow([circ[i],dis[i]])
    np.save(pre + "disease_circRNA.npy", dis2circ)

    dis = [row[1] for row in dis2lnc]
    lnc = [row[0] for row in dis2lnc]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"dis2lnc_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dis)):
            writer.writerow([lnc[i],dis[i]])
    np.save(pre + "disease_lncRNA.npy", dis2lnc)


    dis = [row[1] for row in dis2gene]
    gene = [row[0] for row in dis2gene]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"dis2gene_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dis)):
            writer.writerow([gene[i],dis[i]])
    np.save(pre + "disease_gene.npy", dis2gene)

    mi = [row[0] for row in mi2circ]
    circ = [row[1] for row in mi2circ]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"mi2circ_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(mi)):
            writer.writerow([mi[i], circ[i]])
    np.save(pre + "miRNA_circRNA.npy", mi2circ)

    mi = [row[0] for row in mi2lnc]
    lnc = [row[1] for row in mi2lnc]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"mi2lnc_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(mi)):
            writer.writerow([mi[i], lnc[i]])
    np.save(pre + "miRNA_lncRNA.npy", mi2lnc)

    mi = [row[0] for row in mi2gene]
    gene = [row[1] for row in mi2gene]
    # 设置要保存的 CSV 文件名
    csv_file = pre+"mi2gene_id.csv"
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(mi)):
            writer.writerow([mi[i], gene[i]])
    np.save(pre + "miRNA_gene.npy", mi2gene)









