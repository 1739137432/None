# 1. 读取CSV文件为Pandas DataFrame
import os

import numpy as np
import pandas as pd
def data_划分():

    # dis2mi_data = pd.read_csv('../output/relationship/V_step_relationship/dis2mi_id.csv')
    # dis2gene_data = pd.read_csv('../output/relationship/V_step_relationship/dis2gene_id.csv')
    # dis2circ_data = pd.read_csv('../output/relationship/V_step_relationship/dis2circ_id.csv')
    # dis2lnc_data = pd.read_csv('../output/relationship/V_step_relationship/dis2lnc_id.csv')
    # mi2circ_data = pd.read_csv('../output/relationship/V_step_relationship/mir2circ_id.csv')
    # mi2lnc_data = pd.read_csv('../output/relationship/V_step_relationship/mir2lnc_id.csv')
    # mi2gene_data = pd.read_csv('../output/relationship/V_step_relationship/mir2gene_id.csv')

    dis2mi_data = np.load('../output/relationship/VI_step_data_划分/disease_miRNA.npy').tolist()
    dis2gene_data = np.load('../output/relationship/VI_step_data_划分/disease_gene.npy').tolist()
    dis2circ_data = np.load('../output/relationship/VI_step_data_划分/disease_circRNA.npy').tolist()
    dis2lnc_data = np.load('../output/relationship/VI_step_data_划分/disease_lncRNA.npy').tolist()
    mi2circ_data = np.load('../output/relationship/VI_step_data_划分/miRNA_circRNA.npy').tolist()
    mi2lnc_data = np.load('../output/relationship/VI_step_data_划分/miRNA_lncRNA.npy').tolist()
    mi2gene_data = np.load('../output/relationship/VI_step_data_划分/miRNA_gene.npy').tolist()
    train_ratio = 0.5  # 50% 用于训练集
    val_ratio = 0.3    # 30% 用于验证集
    test_ratio = 0.2   # 20% 用于测试集
    # 检查保存路径中的目录是否存在，如果不存在则创建
    if not os.path.exists('../output/relationship/VII_step_train_val_test/'):
        os.makedirs('../output/relationship/VII_step_train_val_test/')

    dis2mi_total_rows = len(dis2mi_data)
    dis2mi_train_size = int(train_ratio * dis2mi_total_rows)
    dis2mi_val_size = int(val_ratio * dis2mi_total_rows)
    dis2mi_test_size = dis2mi_total_rows - dis2mi_train_size - dis2mi_val_size
    dis2mi_train_idx = np.random.choice(dis2mi_total_rows, dis2mi_train_size, replace=False)
    dis2mi_remaining_idx = np.setdiff1d(np.arange(dis2mi_total_rows), dis2mi_train_idx)
    dis2mi_val_idx = np.random.choice(dis2mi_remaining_idx, dis2mi_val_size, replace=False)
    dis2mi_test_idx = np.setdiff1d(dis2mi_remaining_idx, dis2mi_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/dis2mi_train_val_test_idx.npz', train_idx=dis2mi_train_idx, val_idx=dis2mi_val_idx, test_idx=dis2mi_test_idx)

    gace_data = np.load('../output/relationship/VII_step_train_val_test/dis2mi_train_val_test_idx.npz')
    train_idx = gace_data['train_idx']
    print(train_idx)





    print(1)
    dis2gene_total_rows = len(dis2gene_data)
    print(dis2gene_total_rows)
    dis2gene_train_size = int(train_ratio * dis2gene_total_rows)
    dis2gene_val_size = int(val_ratio * dis2gene_total_rows)
    dis2gene_test_size = dis2gene_total_rows - dis2gene_train_size - dis2gene_val_size
    dis2gene_train_idx = np.random.choice(dis2gene_total_rows, dis2gene_train_size, replace=False)
    for i in dis2gene_train_idx:
        print(i)
    dis2gene_remaining_idx = np.setdiff1d(np.arange(dis2gene_total_rows), dis2gene_train_idx)
    dis2gene_val_idx = np.random.choice(dis2gene_remaining_idx, dis2gene_val_size, replace=False)
    dis2gene_test_idx = np.setdiff1d(dis2gene_remaining_idx, dis2gene_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/dis2gene_train_val_test_idx.npz', train_idx=dis2gene_train_idx, val_idx=dis2gene_val_idx, test_idx=dis2gene_test_idx)


    print(1)
    dis2circ_total_rows = len(dis2circ_data)
    dis2circ_train_size = int(train_ratio * dis2circ_total_rows)
    dis2circ_val_size = int(val_ratio * dis2circ_total_rows)
    dis2circ_test_size = dis2circ_total_rows - dis2circ_train_size - dis2circ_val_size
    dis2circ_train_idx = np.random.choice(dis2circ_total_rows, dis2circ_train_size, replace=False)
    dis2circ_remaining_idx = np.setdiff1d(np.arange(dis2circ_total_rows), dis2circ_train_idx)
    dis2circ_val_idx = np.random.choice(dis2circ_remaining_idx, dis2circ_val_size, replace=False)
    dis2circ_test_idx = np.setdiff1d(dis2circ_remaining_idx, dis2circ_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/dis2circ_train_val_test_idx.npz', train_idx=dis2circ_train_idx, val_idx=dis2circ_val_idx, test_idx=dis2circ_test_idx)
    print(1)
    dis2lnc_total_rows = len(dis2lnc_data)
    dis2lnc_train_size = int(train_ratio * dis2lnc_total_rows)
    dis2lnc_val_size = int(val_ratio * dis2lnc_total_rows)
    dis2lnc_test_size = dis2lnc_total_rows - dis2lnc_train_size - dis2lnc_val_size
    dis2lnc_train_idx = np.random.choice(dis2lnc_total_rows, dis2lnc_train_size, replace=False)
    dis2lnc_remaining_idx = np.setdiff1d(np.arange(dis2lnc_total_rows), dis2lnc_train_idx)
    dis2lnc_val_idx = np.random.choice(dis2lnc_remaining_idx, dis2lnc_val_size, replace=False)
    dis2lnc_test_idx = np.setdiff1d(dis2lnc_remaining_idx, dis2lnc_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/dis2lnc_train_val_test_idx.npz', train_idx=dis2lnc_train_idx, val_idx=dis2lnc_val_idx, test_idx=dis2lnc_test_idx)
    print(1)
    mi2circ_total_rows = len(mi2circ_data)
    mi2circ_train_size = int(train_ratio * mi2circ_total_rows)
    mi2circ_val_size = int(val_ratio * mi2circ_total_rows)
    mi2circ_test_size = mi2circ_total_rows - mi2circ_train_size - mi2circ_val_size
    mi2circ_train_idx = np.random.choice(mi2circ_total_rows, mi2circ_train_size, replace=False)
    mi2circ_remaining_idx = np.setdiff1d(np.arange(mi2circ_total_rows), mi2circ_train_idx)
    mi2circ_val_idx = np.random.choice(mi2circ_remaining_idx, mi2circ_val_size, replace=False)
    mi2circ_test_idx = np.setdiff1d(mi2circ_remaining_idx, mi2circ_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/mi2circ_train_val_test_idx.npz', train_idx=mi2circ_train_idx, val_idx=mi2circ_val_idx, test_idx=mi2circ_test_idx)
    print(1)
    mi2lnc_total_rows = len(mi2lnc_data)
    mi2lnc_train_size = int(train_ratio * mi2lnc_total_rows)
    mi2lnc_val_size = int(val_ratio * mi2lnc_total_rows)
    mi2lnc_test_size = mi2lnc_total_rows - mi2lnc_train_size - mi2lnc_val_size
    mi2lnc_train_idx = np.random.choice(mi2lnc_total_rows, mi2lnc_train_size, replace=False)
    mi2lnc_remaining_idx = np.setdiff1d(np.arange(mi2lnc_total_rows), mi2lnc_train_idx)
    mi2lnc_val_idx = np.random.choice(mi2lnc_remaining_idx, mi2lnc_val_size, replace=False)
    mi2lnc_test_idx = np.setdiff1d(mi2lnc_remaining_idx, mi2lnc_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/mi2lnc_train_val_test_idx.npz', train_idx=mi2lnc_train_idx, val_idx=mi2lnc_val_idx, test_idx=mi2lnc_test_idx)
    print(1)
    mi2gene_total_rows = len(mi2gene_data)
    mi2gene_train_size = int(train_ratio * mi2gene_total_rows)
    mi2gene_val_size = int(val_ratio * mi2gene_total_rows)
    mi2gene_test_size = mi2gene_total_rows - mi2gene_train_size - mi2gene_val_size
    mi2gene_train_idx = np.random.choice(mi2gene_total_rows, mi2gene_train_size, replace=False)
    mi2gene_remaining_idx = np.setdiff1d(np.arange(mi2gene_total_rows), mi2gene_train_idx)
    mi2gene_val_idx = np.random.choice(mi2gene_remaining_idx, mi2gene_val_size, replace=False)
    mi2gene_test_idx = np.setdiff1d(mi2gene_remaining_idx, mi2gene_val_idx)
    np.savez('../output/relationship/VII_step_train_val_test/mi2gene_train_val_test_idx.npz', train_idx=mi2gene_train_idx, val_idx=mi2gene_val_idx, test_idx=mi2gene_test_idx)
    #
