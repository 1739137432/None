import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd


# output positive and negative samples for training, validation and testing
save_prefix = "../output/relationship/VII_step_train_val_test/"
np.random.seed(453289)

num_miRNA = 5174
num_circRNA = 38757
num_lncRNA = 3928
num_gene = 218
num_disease = 4246


# id adjid
miRNA_disease = np.load('../output/relationship/V_step_relationship/miRNA_disease.npy')

dis2mi_train_val_test_idx = np.load('../output/relationship/VI_step_data_划分/dis2mi_train_val_test_idx.npz')
dis2mi_train_idx = dis2mi_train_val_test_idx['train_idx']
dis2mi_val_idx = dis2mi_train_val_test_idx['val_idx']
dis2mi_test_idx = dis2mi_train_val_test_idx['test_idx']

dis2mi_neg_candidates = []
counter = 0
for i in range(num_miRNA):
    for j in range(num_disease):
        if counter < len(miRNA_disease):
            if i == miRNA_disease[counter, 0] and j == miRNA_disease[counter, 1]:
                counter += 1
            else:
                dis2mi_neg_candidates.append([i, j])
        else:
            dis2mi_neg_candidates.append([i, j])
dis2mi_neg_candidates = np.array(dis2mi_neg_candidates)

dis2mi_idx = np.random.choice(len(dis2mi_neg_candidates), len(dis2mi_val_idx) + len(dis2mi_test_idx), replace=False)
dis2mi_val_neg_candidates = dis2mi_neg_candidates[sorted(dis2mi_idx[:len(dis2mi_val_idx)])]
dis2mi_test_neg_candidates = dis2mi_neg_candidates[sorted(dis2mi_idx[len(dis2mi_val_idx):])]

dis2mi_train = miRNA_disease[dis2mi_train_idx]
dis2mi_train_neg_candidates = []
counter = 0
for i in range(num_miRNA):
    for j in range(num_disease):
        if counter < len(dis2mi_train):
            if i == dis2mi_train[counter, 0] and j == dis2mi_train[counter, 1]:
                counter += 1
            else:
                dis2mi_train_neg_candidates.append([i, j])
        else:
            dis2mi_train_neg_candidates.append([i, j])
dis2mi_train_neg_candidates = np.array(dis2mi_train_neg_candidates)

np.savez(save_prefix + 'dis2mi_train_val_test_neg.npz',
         dis2mi_train_neg=dis2mi_train_neg_candidates,
         dis2mi_val_neg=dis2mi_val_neg_candidates,
         dis2mi_test_neg=dis2mi_test_neg_candidates)
np.savez(save_prefix + 'dis2mi_train_val_test_pos.npz',
         dis2mi_train_pos=miRNA_disease[dis2mi_train_idx],
         dis2mi_val_pos=miRNA_disease[dis2mi_val_idx],
         dis2mi_test_pos=miRNA_disease[dis2mi_test_idx])

