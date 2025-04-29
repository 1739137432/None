import os
import pathlib
import pickle
import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd

def train_val_test():
    # output positive and negative samples for training, validation and testing

    save_prefix = "../output/relationship/VII_step_train_val_test/"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    np.random.seed(453289)

    num_miRNA = pd.read_csv('../output/relationship/IV_step_similarity/miRNA_id.csv').shape[0]+1
    num_circRNA = pd.read_csv('../output/relationship/IV_step_similarity/circRNA_id.csv').shape[0]+1
    num_lncRNA = pd.read_csv('../output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0]+1
    num_gene = pd.read_csv('../output/relationship/IV_step_similarity/gene_id.csv').shape[0]+1
    num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv',sep=':').shape[0]+1

    arr = np.array([[5, 1], [3, 2], [3, 1], [5, 0]]).tolist()
    arr.sort(key=lambda x: (x[0], x[1]))
    arr = np.array(arr)
    # id adjid
    miRNA_disease = np.load('../output/relationship/VI_step_data_division/disease_miRNA.npy')
    miRNA_disease = miRNA_disease.tolist()
    miRNA_disease.sort(key=lambda x: (x[0], x[1]))
    miRNA_disease = np.array(miRNA_disease)
    circRNA_disease = np.load('../output/relationship/VI_step_data_division/disease_circRNA.npy')
    circRNA_disease = circRNA_disease.tolist()
    circRNA_disease.sort(key=lambda x: (x[0], x[1]))
    circRNA_disease = np.array(circRNA_disease)
    lncRNA_disease = np.load('../output/relationship/VI_step_data_division/disease_lncRNA.npy')
    lncRNA_disease = lncRNA_disease.tolist()
    lncRNA_disease.sort(key=lambda x: (x[0], x[1]))
    lncRNA_disease = np.array(lncRNA_disease)
    gene_disease = np.load('../output/relationship/VI_step_data_division/disease_gene.npy')
    gene_disease = gene_disease.tolist()
    gene_disease.sort(key=lambda x: (x[0], x[1]))
    gene_disease = np.array(gene_disease)
    miRNA_circRNA = np.load('../output/relationship/VI_step_data_division/miRNA_circRNA.npy')
    miRNA_circRNA = miRNA_circRNA.tolist()
    miRNA_circRNA.sort(key=lambda x: (x[0], x[1]))
    miRNA_circRNA = np.array(miRNA_circRNA)
    miRNA_lncRNA = np.load('../output/relationship/VI_step_data_division/miRNA_lncRNA.npy')
    miRNA_lncRNA = miRNA_lncRNA.tolist()
    miRNA_lncRNA.sort(key=lambda x: (x[0], x[1]))
    miRNA_lncRNA = np.array(miRNA_lncRNA)
    miRNA_gene = np.load('../output/relationship/VI_step_data_division/miRNA_gene.npy')
    miRNA_gene = miRNA_gene.tolist()
    miRNA_gene.sort(key=lambda x: (x[0], x[1]))
    miRNA_gene = np.array(miRNA_gene)

    dis2mi_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/dis2mi_train_val_test_idx.npz')
    dis2mi_train_idx = dis2mi_train_val_test_idx['train_idx']
    dis2mi_val_idx = dis2mi_train_val_test_idx['val_idx']
    dis2mi_test_idx = dis2mi_train_val_test_idx['test_idx']

    dis2circ_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/dis2circ_train_val_test_idx.npz')
    dis2circ_train_idx = dis2circ_train_val_test_idx['train_idx']
    dis2circ_val_idx = dis2circ_train_val_test_idx['val_idx']
    dis2circ_test_idx = dis2circ_train_val_test_idx['test_idx']

    dis2lnc_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/dis2lnc_train_val_test_idx.npz')
    dis2lnc_train_idx = dis2lnc_train_val_test_idx['train_idx']
    dis2lnc_val_idx = dis2lnc_train_val_test_idx['val_idx']
    dis2lnc_test_idx = dis2lnc_train_val_test_idx['test_idx']

    dis2gene_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/dis2gene_train_val_test_idx.npz')
    dis2gene_train_idx = dis2gene_train_val_test_idx['train_idx']
    dis2gene_val_idx = dis2gene_train_val_test_idx['val_idx']
    dis2gene_test_idx = dis2gene_train_val_test_idx['test_idx']

    mi2circ_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/mi2circ_train_val_test_idx.npz')
    mi2circ_train_idx = mi2circ_train_val_test_idx['train_idx']
    mi2circ_val_idx = mi2circ_train_val_test_idx['val_idx']
    mi2circ_test_idx = mi2circ_train_val_test_idx['test_idx']

    mi2lnc_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/mi2lnc_train_val_test_idx.npz')
    mi2lnc_train_idx = mi2lnc_train_val_test_idx['train_idx']
    mi2lnc_val_idx = mi2lnc_train_val_test_idx['val_idx']
    mi2lnc_test_idx = mi2lnc_train_val_test_idx['test_idx']

    mi2gene_train_val_test_idx = np.load('../output/relationship/VII_step_train_val_test/mi2gene_train_val_test_idx.npz')
    mi2gene_train_idx = mi2gene_train_val_test_idx['train_idx']
    mi2gene_val_idx = mi2gene_train_val_test_idx['val_idx']
    mi2gene_test_idx = mi2gene_train_val_test_idx['test_idx']


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

    #=======================================================

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

    #=======================================================

    dis2lnc_neg_candidates = []
    counter = 0
    for i in range(num_lncRNA):
        for j in range(num_disease):
            if counter < len(lncRNA_disease):
                if i == lncRNA_disease[counter, 0] and j == lncRNA_disease[counter, 1]:
                    counter += 1
                else:
                    dis2lnc_neg_candidates.append([i, j])
            else:
                dis2lnc_neg_candidates.append([i, j])
    dis2lnc_neg_candidates = np.array(dis2lnc_neg_candidates)

    dis2lnc_idx = np.random.choice(len(dis2lnc_neg_candidates), len(dis2lnc_val_idx) + len(dis2lnc_test_idx), replace=False)
    dis2lnc_val_neg_candidates = dis2lnc_neg_candidates[sorted(dis2lnc_idx[:len(dis2lnc_val_idx)])]
    dis2lnc_test_neg_candidates = dis2lnc_neg_candidates[sorted(dis2lnc_idx[len(dis2lnc_val_idx):])]

    dis2lnc_train = lncRNA_disease[dis2lnc_train_idx]
    dis2lnc_train_neg_candidates = []
    counter = 0
    for i in range(num_lncRNA):
        for j in range(num_disease):
            if counter < len(dis2lnc_train):
                if i == dis2lnc_train[counter, 0] and j == dis2lnc_train[counter, 1]:
                    counter += 1
                else:
                    dis2lnc_train_neg_candidates.append([i, j])
            else:
                dis2lnc_train_neg_candidates.append([i, j])
    dis2lnc_train_neg_candidates = np.array(dis2lnc_train_neg_candidates)

    np.savez(save_prefix + 'dis2lnc_train_val_test_neg.npz',
             dis2lnc_train_neg=dis2lnc_train_neg_candidates,
             dis2lnc_val_neg=dis2lnc_val_neg_candidates,
             dis2lnc_test_neg=dis2lnc_test_neg_candidates)
    np.savez(save_prefix + 'dis2lnc_train_val_test_pos.npz',
             dis2lnc_train_pos=lncRNA_disease[dis2lnc_train_idx],
             dis2lnc_val_pos=lncRNA_disease[dis2lnc_val_idx],
             dis2lnc_test_pos=lncRNA_disease[dis2lnc_test_idx])

    #=======================================================

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


    #=======================================================

    mi2circ_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_circRNA):
            if counter < len(miRNA_circRNA):
                if i == miRNA_circRNA[counter, 0] and j == miRNA_circRNA[counter, 1]:
                    counter += 1
                else:
                    mi2circ_neg_candidates.append([i, j])
            else:
                mi2circ_neg_candidates.append([i, j])
    mi2circ_neg_candidates = np.array(mi2circ_neg_candidates)

    mi2circ_idx = np.random.choice(len(mi2circ_neg_candidates), len(mi2circ_val_idx) + len(mi2circ_test_idx), replace=False)
    mi2circ_val_neg_candidates = mi2circ_neg_candidates[sorted(mi2circ_idx[:len(mi2circ_val_idx)])]
    mi2circ_test_neg_candidates = mi2circ_neg_candidates[sorted(mi2circ_idx[len(mi2circ_val_idx):])]

    mi2circ_train = miRNA_circRNA[mi2circ_train_idx]
    mi2circ_train_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_circRNA):
            if counter < len(mi2circ_train):
                if i == mi2circ_train[counter, 0] and j == mi2circ_train[counter, 1]:
                    counter += 1
                else:
                    mi2circ_train_neg_candidates.append([i, j])
            else:
                mi2circ_train_neg_candidates.append([i, j])
    mi2circ_train_neg_candidates = np.array(mi2circ_train_neg_candidates)

    np.savez(save_prefix + 'mi2circ_train_val_test_neg.npz',
             mi2circ_train_neg=mi2circ_train_neg_candidates,
             mi2circ_val_neg=mi2circ_val_neg_candidates,
             mi2circ_test_neg=mi2circ_test_neg_candidates)
    np.savez(save_prefix + 'mi2circ_train_val_test_pos.npz',
             mi2circ_train_pos=miRNA_circRNA[mi2circ_train_idx],
             mi2circ_val_pos=miRNA_circRNA[mi2circ_val_idx],
             mi2circ_test_pos=miRNA_circRNA[mi2circ_test_idx])


    #=======================================================

    mi2lnc_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_lncRNA):
            if counter < len(miRNA_lncRNA):
                if i == miRNA_lncRNA[counter, 0] and j == miRNA_lncRNA[counter, 1]:
                    counter += 1
                else:
                    mi2lnc_neg_candidates.append([i, j])
            else:
                mi2lnc_neg_candidates.append([i, j])
    mi2lnc_neg_candidates = np.array(mi2lnc_neg_candidates)

    mi2lnc_idx = np.random.choice(len(mi2lnc_neg_candidates), len(mi2lnc_val_idx) + len(mi2lnc_test_idx), replace=False)
    mi2lnc_val_neg_candidates = mi2lnc_neg_candidates[sorted(mi2lnc_idx[:len(mi2lnc_val_idx)])]
    mi2lnc_test_neg_candidates = mi2lnc_neg_candidates[sorted(mi2lnc_idx[len(mi2lnc_val_idx):])]

    mi2lnc_train = miRNA_lncRNA[mi2lnc_train_idx]
    mi2lnc_train_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_lncRNA):
            if counter < len(mi2lnc_train):
                if i == mi2lnc_train[counter, 0] and j == mi2lnc_train[counter, 1]:
                    counter += 1
                else:
                    mi2lnc_train_neg_candidates.append([i, j])
            else:
                mi2lnc_train_neg_candidates.append([i, j])
    mi2lnc_train_neg_candidates = np.array(mi2lnc_train_neg_candidates)

    np.savez(save_prefix + 'mi2lnc_train_val_test_neg.npz',
             mi2lnc_train_neg=mi2lnc_train_neg_candidates,
             mi2lnc_val_neg=mi2lnc_val_neg_candidates,
             mi2lnc_test_neg=mi2lnc_test_neg_candidates)
    np.savez(save_prefix + 'mi2lnc_train_val_test_pos.npz',
             mi2lnc_train_pos=miRNA_lncRNA[mi2lnc_train_idx],
             mi2lnc_val_pos=miRNA_lncRNA[mi2lnc_val_idx],
             mi2lnc_test_pos=miRNA_lncRNA[mi2lnc_test_idx])


    #=======================================================

    mi2gene_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_gene):
            if counter < len(miRNA_gene):
                if i == miRNA_gene[counter, 0] and j == miRNA_gene[counter, 1]:
                    counter += 1
                else:
                    mi2gene_neg_candidates.append([i, j])
            else:
                mi2gene_neg_candidates.append([i, j])
    mi2gene_neg_candidates = np.array(mi2gene_neg_candidates)

    mi2gene_idx = np.random.choice(len(mi2gene_neg_candidates), len(mi2gene_val_idx) + len(mi2gene_test_idx), replace=False)
    mi2gene_val_neg_candidates = mi2gene_neg_candidates[sorted(mi2gene_idx[:len(mi2gene_val_idx)])]
    mi2gene_test_neg_candidates = mi2gene_neg_candidates[sorted(mi2gene_idx[len(mi2gene_val_idx):])]

    mi2gene_train = miRNA_gene[mi2gene_train_idx]
    mi2gene_train_neg_candidates = []
    counter = 0
    for i in range(num_miRNA):
        for j in range(num_gene):
            if counter < len(mi2gene_train):
                if i == mi2gene_train[counter, 0] and j == mi2gene_train[counter, 1]:
                    counter += 1
                else:
                    mi2gene_train_neg_candidates.append([i, j])
            else:
                mi2gene_train_neg_candidates.append([i, j])
    mi2gene_train_neg_candidates = np.array(mi2gene_train_neg_candidates)

    np.savez(save_prefix + 'mi2gene_train_val_test_neg.npz',
             mi2gene_train_neg=mi2gene_train_neg_candidates,
             mi2gene_val_neg=mi2gene_val_neg_candidates,
             mi2gene_test_neg=mi2gene_test_neg_candidates)
    np.savez(save_prefix + 'mi2gene_train_val_test_pos.npz',
             mi2gene_train_pos=miRNA_gene[mi2gene_train_idx],
             mi2gene_val_pos=miRNA_gene[mi2gene_val_idx],
             mi2gene_test_pos=miRNA_gene[mi2gene_test_idx])