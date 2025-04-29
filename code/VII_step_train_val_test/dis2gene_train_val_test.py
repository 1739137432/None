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
gene_disease = np.load('../output/relationship/V_step_relationship/gene_disease.npy')
print(len(gene_disease))

dis2gene_train_val_test_idx = np.load('../output/relationship/VI_step_data_division/dis2gene_train_val_test_idx.npz')
dis2gene_train_idx = dis2gene_train_val_test_idx['train_idx']
dis2gene_val_idx = dis2gene_train_val_test_idx['val_idx']
dis2gene_test_idx = dis2gene_train_val_test_idx['test_idx']
for i in dis2gene_train_idx:
    print(i)

dis2gene_neg_candidates = []
counter = 0
for i in range(num_gene):
    for j in range(num_disease):
        if counter < len(gene_disease):
            if i == gene_disease[counter, 0] and j == gene_disease[counter, 1]:
                counter += 1
            else:
                dis2gene_neg_candidates.append([i, j])
        else:
            dis2gene_neg_candidates.append([i, j])
dis2gene_neg_candidates = np.array(dis2gene_neg_candidates)

dis2gene_idx = np.random.choice(len(dis2gene_neg_candidates), len(dis2gene_val_idx) + len(dis2gene_test_idx), replace=False)
dis2gene_val_neg_candidates = dis2gene_neg_candidates[sorted(dis2gene_idx[:len(dis2gene_val_idx)])]
dis2gene_test_neg_candidates = dis2gene_neg_candidates[sorted(dis2gene_idx[len(dis2gene_val_idx):])]

for i in dis2gene_train_idx:
    print(i)


dis2gene_train = gene_disease[dis2gene_train_idx]
dis2gene_train_neg_candidates = []
counter = 0
for i in range(num_gene):
    for j in range(num_disease):
        if counter < len(dis2gene_train):
            if i == dis2gene_train[counter, 0] and j == dis2gene_train[counter, 1]:
                counter += 1
            else:
                dis2gene_train_neg_candidates.append([i, j])
        else:
            dis2gene_train_neg_candidates.append([i, j])
dis2gene_train_neg_candidates = np.array(dis2gene_train_neg_candidates)

np.savez(save_prefix + 'dis2gene_train_val_test_neg.npz',
         dis2gene_train_neg=dis2gene_train_neg_candidates,
         dis2gene_val_neg=dis2gene_val_neg_candidates,
         dis2gene_test_neg=dis2gene_test_neg_candidates)
np.savez(save_prefix + 'dis2gene_train_val_test_pos.npz',
         dis2gene_train_pos=gene_disease[dis2gene_train_idx],
         dis2gene_val_pos=gene_disease[dis2gene_val_idx],
         dis2gene_test_pos=gene_disease[dis2gene_test_idx])
