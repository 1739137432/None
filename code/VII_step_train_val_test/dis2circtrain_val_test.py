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
circRNA_disease = np.load('../output/relationship/V_step_relationship/circRNA_disease.npy')


dis2circ_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2circ_train_val_test_idx.npz')
dis2circ_train_idx = dis2circ_train_val_test_idx['train_idx']
dis2circ_val_idx = dis2circ_train_val_test_idx['val_idx']
dis2circ_test_idx = dis2circ_train_val_test_idx['test_idx']

dis2circ_neg_candidates = []
counter = 0
for i in range(num_circRNA):
    for j in range(num_disease):
        if counter < len(circRNA_disease):
            if i == circRNA_disease[counter, 0] and j == circRNA_disease[counter, 1]:
                counter += 1
            else:
                dis2circ_neg_candidates.append([i, j])
        else:
            dis2circ_neg_candidates.append([i, j])
dis2circ_neg_candidates = np.array(dis2circ_neg_candidates)

dis2circ_idx = np.random.choice(len(dis2circ_neg_candidates), len(dis2circ_val_idx) + len(dis2circ_test_idx), replace=False)
dis2circ_val_neg_candidates = dis2circ_neg_candidates[sorted(dis2circ_idx[:len(dis2circ_val_idx)])]
dis2circ_test_neg_candidates = dis2circ_neg_candidates[sorted(dis2circ_idx[len(dis2circ_val_idx):])]

dis2circ_train = circRNA_disease[dis2circ_train_idx]
dis2circ_train_neg_candidates = []
counter = 0
for i in range(num_circRNA):
    for j in range(num_disease):
        if counter < len(dis2circ_train):
            if i == dis2circ_train[counter, 0] and j == dis2circ_train[counter, 1]:
                counter += 1
            else:
                dis2circ_train_neg_candidates.append([i, j])
        else:
            dis2circ_train_neg_candidates.append([i, j])
dis2circ_train_neg_candidates = np.array(dis2circ_train_neg_candidates)

np.savez(save_prefix + 'dis2circ_train_val_test_neg.npz',
         dis2circ_train_neg=dis2circ_train_neg_candidates,
         dis2circ_val_neg=dis2circ_val_neg_candidates,
         dis2circ_test_neg=dis2circ_test_neg_candidates)
np.savez(save_prefix + 'dis2circ_train_val_test_pos.npz',
         dis2circ_train_pos=circRNA_disease[dis2circ_train_idx],
         dis2circ_val_pos=circRNA_disease[dis2circ_val_idx],
         dis2circ_test_pos=circRNA_disease[dis2circ_test_idx])
