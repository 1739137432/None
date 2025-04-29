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

    save_prefix = "new/"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    np.random.seed(453289)

    # example: disease To ALL
    # The disease [14,2813,3371,3596,3700,3701,4168,4323,5740]   your disease indices
    # other: ALL nodes
    num_miRNA = pd.read_csv('../../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0]+1
    num_circRNA = pd.read_csv('../../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0]+1
    num_lncRNA = pd.read_csv('../../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0]+1
    num_gene = pd.read_csv('../../output/relationship/IV_step_similarity/gene_id.csv').shape[0]+1
    # num_disease = pd.read_csv('../../output/relationship/IV_step_similarity/disease_id.csv',sep=':').shape[0]+1
    disease = [14,2813,3371,3596,3700,3701,4168,4323,5740]
    dis2mi_neg_candidates = []
    dis2mi_pos_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in disease:
            if counter%2==0:
                dis2mi_neg_candidates.append([i, j])
            else:
                dis2mi_pos_candidates.append([i, j])
            counter+=1
    dis2mi_pos_candidates = np.array(dis2mi_pos_candidates[:len(dis2mi_neg_candidates)])
    dis2mi_neg_candidates = np.array(dis2mi_neg_candidates)


    np.savez(save_prefix + 'dis2mi_train_val_test_neg.npz',
             dis2mi_test_neg=dis2mi_neg_candidates)
    np.savez(save_prefix + 'dis2mi_train_val_test_pos.npz',
             dis2mi_test_pos=dis2mi_pos_candidates)

    #=======================================================
    mi2circ_neg_candidates = []
    mi2circ_pos_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_circRNA):
            if counter % 2 == 0:
                mi2circ_neg_candidates.append([i, j])
            else:
                mi2circ_pos_candidates.append([i, j])
            counter += 1

    mi2circ_pos_candidates = np.array(mi2circ_pos_candidates[:len(mi2circ_neg_candidates)])
    mi2circ_neg_candidates = np.array(mi2circ_neg_candidates)

    np.savez(save_prefix + 'mi2circ_train_val_test_neg.npz',
             mi2circ_test_neg=mi2circ_neg_candidates)
    np.savez(save_prefix + 'mi2circ_train_val_test_pos.npz',
             mi2circ_test_pos=mi2circ_pos_candidates)

    #=======================================================
    mi2lnc_neg_candidates = []
    mi2lnc_pos_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_lncRNA):
            if counter % 2 == 0:
                mi2lnc_neg_candidates.append([i, j])
            else:
                mi2lnc_pos_candidates.append([i, j])
            counter += 1

    mi2lnc_pos_candidates = np.array(mi2lnc_pos_candidates[:len(mi2lnc_neg_candidates)])
    mi2lnc_neg_candidates = np.array(mi2lnc_neg_candidates)

    np.savez(save_prefix + 'mi2lnc_train_val_test_neg.npz',
             mi2lnc_test_neg=mi2lnc_neg_candidates)
    np.savez(save_prefix + 'mi2lnc_train_val_test_pos.npz',
             mi2lnc_test_pos=mi2lnc_pos_candidates)

    #=======================================================
    mi2gene_neg_candidates = []
    mi2gene_pos_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_gene):
            if counter % 2 == 0:
                mi2gene_neg_candidates.append([i, j])
            else:
                mi2gene_pos_candidates.append([i, j])
            counter += 1
    mi2gene_pos_candidates = np.array(mi2gene_pos_candidates[:len(mi2gene_neg_candidates)])
    mi2gene_neg_candidates = np.array(mi2gene_neg_candidates)

    np.savez(save_prefix + 'mi2gene_train_val_test_neg.npz',
             mi2gene_test_neg=mi2gene_neg_candidates)
    np.savez(save_prefix + 'mi2gene_train_val_test_pos.npz',
             mi2gene_test_pos=mi2gene_pos_candidates)

    #=======================================================
    dis2circ_neg_candidates = []
    dis2circ_pos_candidates = []
    counter = 0
    for i in range(num_circRNA):
        for j in disease:
            if counter % 2 == 0:
                dis2circ_neg_candidates.append([i, j])
            else:
                dis2circ_pos_candidates.append([i, j])
            counter += 1
    dis2circ_pos_candidates = np.array(dis2circ_pos_candidates[:len(dis2circ_neg_candidates)])
    dis2circ_neg_candidates = np.array(dis2circ_neg_candidates)

    np.savez(save_prefix + 'dis2circ_train_val_test_neg.npz',
             dis2circ_test_neg=dis2circ_neg_candidates)
    np.savez(save_prefix + 'dis2circ_train_val_test_pos.npz',
             dis2circ_test_pos=dis2circ_pos_candidates)

    #=======================================================
    dis2lnc_neg_candidates = []
    dis2lnc_pos_candidates = []
    counter = 0
    for i in range(num_lncRNA):
        for j in disease:
            if counter % 2 == 0:
                dis2lnc_neg_candidates.append([i, j])
            else:
                dis2lnc_pos_candidates.append([i, j])
            counter += 1
    dis2lnc_pos_candidates = np.array(dis2lnc_pos_candidates[:len(dis2lnc_neg_candidates)])
    dis2lnc_neg_candidates = np.array(dis2lnc_neg_candidates)

    np.savez(save_prefix + 'dis2lnc_train_val_test_neg.npz',
             dis2lnc_test_neg=dis2lnc_neg_candidates)
    np.savez(save_prefix + 'dis2lnc_train_val_test_pos.npz',
             dis2lnc_test_pos=dis2lnc_pos_candidates)

    #=======================================================
    dis2gene_neg_candidates = []
    dis2gene_pos_candidates = []
    counter = 0
    for i in range(num_gene):
        for j in disease:
            if counter % 2 == 0:
                dis2gene_neg_candidates.append([i, j])
            else:
                dis2gene_pos_candidates.append([i, j])
            counter += 1
    dis2gene_pos_candidates = np.array(dis2gene_pos_candidates[:len(dis2gene_neg_candidates)])
    dis2gene_neg_candidates = np.array(dis2gene_neg_candidates)

    np.savez(save_prefix + 'dis2gene_train_val_test_neg.npz',
             dis2gene_test_neg=dis2gene_neg_candidates)
    np.savez(save_prefix + 'dis2gene_train_val_test_pos.npz',
             dis2gene_test_pos=dis2gene_pos_candidates)