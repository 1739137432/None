import os
import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd

def train_val_test():
    # output positive and negative samples for training, validation and testing
    # 检查保存路径中的目录是否存在，如果不存在则创建

    save_prefix = "../output/relationship/未知关系/"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    np.random.seed(453289)

    num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0]+1
    num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0]+1
    num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0]+1
    num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0]+1
    num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv',sep=':').shape[0]+1


    # id adjid
    miRNA_disease = np.load('../output/relationship/V_step_relationship/miRNA_disease.npy')
    circRNA_disease = np.load('../output/relationship/V_step_relationship/circRNA_disease.npy')
    lncRNA_disease = np.load('../output/relationship/V_step_relationship/lncRNA_disease.npy')
    gene_disease = np.load('../output/relationship/V_step_relationship/gene_disease.npy')
    miRNA_circRNA = np.load('../output/relationship/V_step_relationship/miRNA_circRNA.npy')
    miRNA_lncRNA = np.load('../output/relationship/V_step_relationship/miRNA_lncRNA.npy')
    miRNA_gene = np.load('../output/relationship/V_step_relationship/miRNA_gene.npy')


    dis2mi_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_disease):
            status = True
            for inde in range(len(miRNA_disease)):
                if i == miRNA_disease[inde, 0] and j == miRNA_disease[inde, 1]:
                    status = False
                    break
            if status:
                dis2mi_neg_candidates.append([i, j])
    dis2mi_neg_candidates = np.array(dis2mi_neg_candidates)
    np.savez(save_prefix + 'dis2mi_neg.npz',
             dis2mi_neg=dis2mi_neg_candidates)

    #=======================================================
    dis2circ_neg_candidates = []
    counter = 0
    for i in range(num_circRNA):
        for j in range(num_disease):
            status = True
            for inde in range(len(circRNA_disease)):
                if i == circRNA_disease[inde, 0] and j == circRNA_disease[inde, 1]:
                    status = False
                    break
            if status:
                dis2circ_neg_candidates.append([i, j])
    dis2circ_neg_candidates = np.array(dis2circ_neg_candidates)
    np.savez(save_prefix + 'dis2circ_neg.npz',
             dis2circ_neg=dis2circ_neg_candidates)


    dis2lnc_neg_candidates = []
    counter = 0
    for i in range(num_lncRNA):
        for j in range(num_disease):
            status = True
            for inde in range(len(lncRNA_disease)):
                if i == lncRNA_disease[inde, 0] and j == lncRNA_disease[inde, 1]:
                    status = False
                    break
            if status:
                dis2lnc_neg_candidates.append([i, j])
    dis2lnc_neg_candidates = np.array(dis2lnc_neg_candidates)
    np.savez(save_prefix + 'dis2lnc_neg.npz',
             dis2lnc_neg=dis2lnc_neg_candidates)


    dis2gene_neg_candidates = []
    counter = 0
    for i in range(num_gene):
        for j in range(num_disease):
            status = True
            for inde in range(len(gene_disease)):
                if i == gene_disease[inde, 0] and j == gene_disease[inde, 1]:
                    status = False
                    break
            if status:
                dis2gene_neg_candidates.append([i, j])
    dis2gene_neg_candidates = np.array(dis2gene_neg_candidates)
    np.savez(save_prefix + 'dis2gene_neg.npz',
             dis2gene_neg=dis2gene_neg_candidates)

    mi2gene_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_gene):
            status = True
            for inde in range(len(miRNA_gene)):
                if i == miRNA_gene[inde, 0] and j == miRNA_gene[inde, 1]:
                    status = False
                    break
            if status:
                mi2gene_neg_candidates.append([i, j])
    mi2gene_neg_candidates = np.array(mi2gene_neg_candidates)
    np.savez(save_prefix + 'mi2gene_neg.npz',
             mi2gene_neg=mi2gene_neg_candidates)

    mi2circ_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_circRNA):
            status = True
            for inde in range(len(miRNA_circRNA)):
                if i == miRNA_circRNA[inde, 0] and j == miRNA_circRNA[inde, 1]:
                    status = False
                    break
            if status:
                mi2circ_neg_candidates.append([i, j])
    mi2circ_neg_candidates = np.array(mi2circ_neg_candidates)
    np.savez(save_prefix + 'mi2circ_neg.npz',
             mi2circ_neg=mi2circ_neg_candidates)

    mi2lnc_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_lncRNA):
            status = True
            for inde in range(len(miRNA_lncRNA)):
                if i == miRNA_lncRNA[inde, 0] and j == miRNA_lncRNA[inde, 1]:
                    status = False
                    break
            if status:
                mi2lnc_neg_candidates.append([i, j])
    mi2lnc_neg_candidates = np.array(mi2lnc_neg_candidates)
    np.savez(save_prefix + 'mi2lnc_neg.npz',
             mi2lnc_neg=mi2lnc_neg_candidates)

# train_val_test()

