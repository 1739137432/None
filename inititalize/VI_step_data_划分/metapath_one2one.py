import csv
import gc
import pathlib
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io

save_prefix = '../../output/relationship/6_step_data_划分/'

num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv', sep=':').shape[0] + 1

lncRNA_miRNA_list = {}
with open('lncRNA_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_miRNA_list[key] = value

gene_miRNA_list = {}
with open('gene_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_miRNA_list[key] = value
disease_miRNA_list = {}
with open('disease_miRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_miRNA_list[key] = value
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

disease_lncRNA_list = {}
with open('disease_lncRNA_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_lncRNA_list[key] = value

miRNA_gene_list = {}
with open('miRNA_gene_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_gene_list[key] = value

disease_gene_list = {}
with open('disease_gene_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_gene_list[key] = value

miRNA_disease_list = {}
with open('miRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_disease_list[key] = value

circRNA_disease_list = {}
with open('circRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        circRNA_disease_list[key] = value

lncRNA_disease_list = {}
with open('lncRNA_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_disease_list[key] = value

gene_disease_list = {}
with open('gene_disease_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_disease_list[key] = value

miRNA_adjacent_list ={}
with open('miRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        miRNA_adjacent_list[key] = value

circRNA_adjacent_list ={}
with open('circRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        circRNA_adjacent_list[key] = value

lncRNA_adjacent_list ={}
with open('lncRNA_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        lncRNA_adjacent_list[key] = value

gene_adjacent_list ={}
with open('gene_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        gene_adjacent_list[key] = value

disease_adjacent_list ={}
with open('disease_adjacent_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        key = int(row[0])
        value = np.array(row[1:], dtype=int)
        disease_adjacent_list[key] = value
miRNA_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/miRNASim.csv', encoding='utf-8',
                             delimiter=',',
                             names=['miRNAID', 'adjacentID'])

gene_adjacent = pd.read_csv('../output/relationship/IV_step_similarity/geneSim.csv', encoding='utf-8',  delimiter=',',
                            names=['geneID', 'adjacentID'])

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


#

gc.collect()
print(5)
#======M-C-M M-L-M M-G-M M-D-M C-M-C C-D-C L-M-L L-D-L G-M-G G-D-G D-M-D D-C-D D-L-D D-G-D=========
#======0-1-0 0-2-0 0-3-0 0-4-0 1-0-1 1-4-1 2-0-2 2-4-2 3-0-3 3-4-3 4-0-4 4-1-4 4-2-4 4-3-4=========


#
gc.collect()

#
# ========M-C-C-M M-L-L-M M-G-G-M M-D-D-M C-M-M-C C-D-D-C L-M-M-L L-D-D-L G-M-M-G G-D-D-G D-M-M-D D-C-C-D D-L-L-D D-G-G-D ##########

# ========================================五元组==============================================#
