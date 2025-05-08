import csv
import time
import argparse
from sklearn.svm import SVC
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_PABDMH_data
from utils.toolss import IndexGenerator, parse_minibatch_PABDMH
from models.PABDMH_lp import PABDMH_lp

# Params
# 表示图中不同节点类型的数量
num_ntype = 5
# 表示在模型训练过程中的随机失活（dropout）比例。dropout 是一种正则化技术，有助于防止过拟合。这里设置为0.5，表示在训练时，每个神经元以0.5的概率被随机丢弃。
dropout_rate = 0.5
# 学习率，表示在梯度下降中每次更新模型参数时的步长
lr = 0.0001
# 权重衰减，是正则化项的一种形式，用于防止模型的参数过大
weight_decay = 0.0001
#29

etypes_lists = [
    [
        [None], [0, 1], [2, 3], [4, 5], [6, 7], [0, None, 1], [2, None, 3], [4, None, 5], [6, None, 7],
        [0, 8, 9, 1],  [2, 10, 11, 3],  [4, 12, 13, 5],
        [6, 9, 8, 7], [6, 11, 10, 7], [6, 13, 12, 7],
        # [0, 8, None, 9, 1], [2, 10, None, 11, 3],
        # [4, 12, None, 13, 5], [6, 9, None, 8, 7], [6, 11, None, 10, 7], [6, 13, None, 12, 7]
    ],
    [
        [None], [1, 0], [8, 9], [1, None, 0], [8, None, 9],
        [1, 2, 3, 0], [1, 4, 5, 0], [1, 6, 7, 0], [8, 7, 6, 9], [8, 11, 10, 9], [8, 13, 12, 9],
        # [1, 2, None, 3, 0], [1, 4, None, 5, 0], [1, 6, None, 7, 0], [8, 7, None, 6, 9],
        # [8, 11, None, 10, 9], [8, 13, None, 12, 9]
    ],
    [
        [None], [3, 2], [10, 11], [3, None, 2], [10, None, 11],
        [3, 0, 1, 2], [3, 4, 5, 2], [3, 6, 7, 2], [10, 6, 7, 11], [10, 9, 8, 11], [10, 13, 12, 11],
        # [3, 0, None, 1, 2], [3, 4, None, 5, 2], [3, 6, None, 7, 2], [10, 6, None, 7, 11],
        # [10, 9, None, 8, 11], [10, 13, None, 12, 11]
    ],
    [
        [None], [5, 4], [12, 13], [5, None, 4], [12, None, 13],
        [5, 0, 1, 4], [5, 3, 2, 4], [5, 7, 6, 4], [12, 7, 6, 13], [12, 9, 8, 13], [12, 11, 10, 13],
        # [5, 0, None, 1, 4], [5, 3, None, 2, 4], [5, 7, None, 6, 4], [12, 7, None, 6, 13],
        # [12, 9, None, 8, 13], [12, 11, None, 10, 13]
    ],
    [
        [None], [7, 6], [9, 8], [11, 10], [13, 12], [7, None, 6], [9, None, 8], [11, None, 10], [13, None, 12],
        [7, 0, 1, 6], [7, 2, 3, 6], [7, 4, 5, 6], [9, 1, 0, 8], [11, 2, 3, 10],
        [13, 5, 4, 12],
        # [7, 0, None, 1, 6], [7, 2, None, 3, 6], [7, 4, None, 5, 6], [9, 1, None, 0, 8],
        # [11, 2, None, 3, 10], [13, 5, None, 4, 12]
    ]
]  # 关系种类

masks = []
mi2disease_masks = [
    [False, False, False, False, True,
     False, False, False, True, False,
     False, False, True, True,True],
    [False, False, False, False, False,
     False, False, False, True, True,
     False],
    [False, False, False, False, False,
     False, False, False, True, True,
     False],
    [False, False, False, False, False,
     False, False, False, True, True,
     False],
    [False, True, False, False, False,
     True, False, False, False, True,
     True, True, False, False, False]
]  # 验证集：是否包含miRNA_disease链接
circ2disease_masks = [
    [False, False, False, False, False,
     False, False, False, False, True,
     False, False, True, False, False],
    [False, False, True, False, True,
     False, False, False, True, True,
     True],
    [False, False, False, False, False,
     False, False, False, False, True,
     False],
    [False, False, False, False, False,
     False, False, False, False, True,
     False],
    [False, False, True, False, False,
     False, True, False, False, False,
     False, False, True, False, True]
]  # 验证集：是否包含miRNA_disease链接
lnc2disease_masks = [
    [False, False, False, False, False,
     False, False, False, False, False,
     True, False, False, True, False],
    [False, False, False, False, False,
     False, False, False, False, True,
     False],
    [False, False, True, False, True,
     False, False, False, True, True,
     True],
    [False, False, False, False, False,
     False, False, False, False, False,
     True],
    [False, False, False, True, False,
     False, False, True, False, False,
     False, False, False, True, False]
]
gene2disease_masks = [
    [False, False, False, False, False,
     False, False, False, False, False,
     False, True, False, False, True],
    [False, False, False, False, False,
     False, False, False, False, False,
     True],
    [False, False, False, False, False,
     False, False, False, False, False,
     True],
    [False, False, True, False, True,
     False, False, False, True, True,
     True],
    [False, False, False, False, True,
     False, False, False, True, False,
     False, False, False, False, True]
]
mi2circ_masks = [
    [False, True, False, False, False,
     True, False, False, False, True,
     False, False, False, False, False],
    [False, True, False, True, False,
     True, True, True, False, False,
     False],
    [False, False, False, False, False,
     True, False, False, False, False,
     False],
    [False, False, False, False, False,
     True, False, False, False, False,
     False],
    [False, False, False, False, False,
     False, False, False, False, True,
     False, False, True, False, False]
]
mi2lnc_masks = [
    [False, False, True, False, False,
     False, True, False, False, False,
     True, False, False, False, False],
    [False, False, False, False, False,
     True, False, False, False, False,
     False],
    [False, True, False, True, False,
     True, True, True, False, False,
     False],
    [False, False, False, False, False,
     False, True, False, False, False,
     False],
    [False, False, False, False, False,
     False, False, False, False, False,
     True, False, False, True, False]
]
mi2gene_masks = [
    [False, False, False, True, False,
     False, False, True, False, False,
     False, True, False, False, False],
    [False, False, False, False, False,
     False, True, False, False, False,
     False],
    [False, False, False, False, False,
     False, True, False, False, False,
     False],
    [False, True, False, True, False,
     True, True, True, False, False,
     False],
    [False, False, False, False, False,
     False, False, False, False, False,
     False, True, False, False, True]
]
masks.append(mi2disease_masks)
masks.append(circ2disease_masks)
masks.append(lnc2disease_masks)
masks.append(gene2disease_masks)
masks.append(mi2circ_masks)
masks.append(mi2lnc_masks)
masks.append(mi2gene_masks)

no_masks = [[False] * 15, [False] * 11,[False]*11,[False]*11,[False]*15]   #测试集

num_miRNA = pd.read_csv('data/output/relationship/IV_step_similarity/miRNA_id.csv').shape[0] + 1
num_circRNA = pd.read_csv('data/output/relationship/IV_step_similarity/circRNA_id.csv').shape[0] + 1
num_lncRNA = pd.read_csv('data/output/relationship/IV_step_similarity/lncRNA_id.csv').shape[0] + 1
num_gene = pd.read_csv('data/output/relationship/IV_step_similarity/gene_id.csv').shape[0] + 1
num_disease = pd.read_csv('data/output/relationship/IV_step_similarity/disease_id.csv',sep=':').shape[0] + 1


num = []
num.append(num_miRNA)
num.append(num_circRNA)
num.append(num_lncRNA)
num.append(num_gene)
num.append(num_disease)

expected_metapaths = [
        [(0, 0), (0, 1, 0), (0, 2, 0),(0, 3, 0),(0, 4, 0),
         (0,1,1,0),(0,2,2,0),(0,3,3,0),(0,4,4,0),
         (0,1,4,1,0),(0,2,4,2,0),(0,3,4,3,0),(0,4,1,4,0),(0,4,2,4,0),
         (0,4,3,4,0),
         # (0,1,4,4,1,0),(0,2,4,4,2,0),(0,3,4,4,3,0),(0,4,1,1,4,0),(0,4,2,2,4,0),(0,4,3,3,4,0),
         ],
        [(1, 1), (1, 0, 1), (1, 4, 1),
         (1, 0, 0, 1),(1,4,4,1),
         (1,0,2,0,1),(1,0,3,0,1),(1,0,4,0,1),(1,4,0,4,1),(1,4,2,4,1),(1,4,3,4,1),
         # (1,0,2,2,0,1),(1,0,3,3,0,1),(1,0,4,4,0,1),(1,4,0,0,4,1),(1,4,2,2,4,1),(1,4,3,3,4,1)
         ],
        [(2, 2), (2,0,2),(2,4,2),
         (2,0,0,2),(2,4,4,2),
         (2,0,1,0,2),(2,0,3,0,2),(2,0,4,0,2),(2,4,0,4,2),(2,4,1,4,2),(2,4,3,4,2),
         # (2,0,1,1,0,2),(2,0,3,3,0,2),(2,0,4,4,0,2),(2,4,0,0,4,2),(2,4,1,1,4,2),(2,4,3,3,4,2)
         ],
        [(3, 3), (3,0,3),(3,4,3),
         (3,0,0,3),(3,4,4,3),
         (3,0,1,0,3),(3,0,2,0,3),(3,0,4,0,3),(3,4,0,4,3),(3,4,1,4,3),(3,4,2,4,3),
         # (3,0,1,1,0,3),(3,0,2,2,0,3),(3,0,4,4,0,3),(3,4,0,0,4,3),(3,4,1,1,4,3),(3,4,2,2,4,3)
         ],
        [(4, 4), (4,0,4),(4,1,4),(4,2,4),(4,3,4),
         (4,0,0,4),(4,1,1,4),(4,2,2,4),(4,3,3,4),
         (4,0,1,0,4),(4,0,2,0,4),(4,0,3,0,4),(4,1,0,1,4),(4,2,0,2,4),(4,3,0,3,4),
         # (4,0,1,1,0,4),(4,0,2,2,0,4),(4,0,3,3,0,4),(4,1,0,0,1,4),(4,2,0,0,2,4),(4,3,0,0,3,4),
         ]
    ]


def run_model_PABDMH(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix,num_embeding,checkpoint):
    adjlists_ua, edge_metapath_indices_list_ua, _, type_mask,dis2mi_train_val_test_pos, dis2mi_train_val_test_neg, dis2circ_train_val_test_neg, dis2circ_train_val_test_pos, dis2lnc_train_val_test_neg, dis2lnc_train_val_test_pos, dis2gene_train_val_test_neg, dis2gene_train_val_test_pos, mi2circ_train_val_test_neg, mi2circ_train_val_test_pos, mi2lnc_train_val_test_neg, mi2lnc_train_val_test_pos, mi2gene_train_val_test_neg, mi2gene_train_val_test_pos = load_PABDMH_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    features_list = []
    in_dims = []
    if feats_type == 0:   #all id vectors / Default
        for i in range(num_ntype):    
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:   #all zero vector
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    # dis2mi_train_pos = dis2mi_train_val_test_pos['dis2mi_train_pos']
    # dis2mi_val_pos = dis2mi_train_val_test_pos['dis2mi_val_pos']
    dis2mi_test_pos = dis2mi_train_val_test_pos['dis2mi_test_pos']
    # dis2mi_test_pos = np.concatenate((dis2mi_train_pos, dis2mi_val_pos, dis2mi_test_pos1))

    # dis2mi_train_neg = dis2mi_train_val_test_neg['dis2mi_train_neg']
    # dis2mi_val_neg = dis2mi_train_val_test_neg['dis2mi_val_neg']
    dis2mi_test_neg = dis2mi_train_val_test_neg['dis2mi_test_neg']
    # dis2mi_test_neg = np.concatenate((dis2mi_train_neg, dis2mi_val_neg, dis2mi_test_neg1))
    print(len(dis2mi_test_pos))
    # dis2circ_train_neg = dis2circ_train_val_test_neg['dis2circ_train_neg']
    # dis2circ_val_neg = dis2circ_train_val_test_neg['dis2circ_val_neg']
    dis2circ_test_neg = dis2circ_train_val_test_neg['dis2circ_test_neg']
    # dis2circ_test_neg = np.concatenate((dis2circ_train_neg, dis2circ_val_neg, dis2circ_test_neg1))

    # dis2circ_train_pos = dis2circ_train_val_test_pos['dis2circ_train_pos']
    # dis2circ_val_pos = dis2circ_train_val_test_pos['dis2circ_val_pos']
    dis2circ_test_pos = dis2circ_train_val_test_pos['dis2circ_test_pos']
    # dis2circ_test_pos = np.concatenate((dis2circ_train_pos, dis2circ_val_pos, dis2circ_test_pos1))
    print(len(dis2circ_test_pos))
    # dis2lnc_train_neg = dis2lnc_train_val_test_neg['dis2lnc_train_neg']
    # dis2lnc_val_neg = dis2lnc_train_val_test_neg['dis2lnc_val_neg']
    dis2lnc_test_neg = dis2lnc_train_val_test_neg['dis2lnc_test_neg']
    # dis2lnc_test_neg = np.concatenate((dis2lnc_train_neg, dis2lnc_val_neg, dis2lnc_test_neg1))

    # dis2lnc_train_pos = dis2lnc_train_val_test_pos['dis2lnc_train_pos']
    # dis2lnc_val_pos = dis2lnc_train_val_test_pos['dis2lnc_val_pos']
    dis2lnc_test_pos = dis2lnc_train_val_test_pos['dis2lnc_test_pos']
    # dis2lnc_test_pos = np.concatenate((dis2lnc_train_pos, dis2lnc_val_pos, dis2lnc_test_pos1))
    print(len(dis2lnc_test_pos))
    # dis2gene_train_val_test_neg
    # dis2gene_train_val_test_pos
    # dis2gene_train_neg = dis2gene_train_val_test_neg['dis2gene_train_neg']
    # dis2gene_val_neg = dis2gene_train_val_test_neg['dis2gene_val_neg']
    dis2gene_test_neg = dis2gene_train_val_test_neg['dis2gene_test_neg']
    # dis2gene_test_neg = np.concatenate((dis2gene_train_neg, dis2gene_val_neg, dis2gene_test_neg1))

    # dis2gene_train_pos = dis2gene_train_val_test_pos['dis2gene_train_pos']
    # dis2gene_val_pos = dis2gene_train_val_test_pos['dis2gene_val_pos']
    dis2gene_test_pos = dis2gene_train_val_test_pos['dis2gene_test_pos']
    # dis2gene_test_pos = np.concatenate((dis2gene_train_pos, dis2gene_val_pos, dis2gene_test_pos1))
    print(len(dis2gene_test_pos))
    # mi2circ_train_val_test_neg
    # mi2circ_train_val_test_pos
    # mi2circ_train_neg = mi2circ_train_val_test_neg['mi2circ_train_neg']
    # mi2circ_val_neg = mi2circ_train_val_test_neg['mi2circ_val_neg']
    mi2circ_test_neg = mi2circ_train_val_test_neg['mi2circ_test_neg']
    # mi2circ_test_neg = np.concatenate((mi2circ_train_neg, mi2circ_val_neg, mi2circ_test_neg1))

    # mi2circ_train_pos = mi2circ_train_val_test_pos['mi2circ_train_pos']
    # mi2circ_val_pos = mi2circ_train_val_test_pos['mi2circ_val_pos']
    mi2circ_test_pos = mi2circ_train_val_test_pos['mi2circ_test_pos']
    # mi2circ_test_pos = np.concatenate((mi2circ_train_pos, mi2circ_val_pos, mi2circ_test_pos1))
    print(len(mi2circ_test_pos))
    # mi2lnc_train_val_test_neg
    # mi2lnc_train_val_test_pos
    # mi2lnc_train_neg = mi2lnc_train_val_test_neg['mi2lnc_train_neg']
    # mi2lnc_val_neg = mi2lnc_train_val_test_neg['mi2lnc_val_neg']
    mi2lnc_test_neg = mi2lnc_train_val_test_neg['mi2lnc_test_neg']
    # mi2lnc_test_neg = np.concatenate((mi2lnc_train_neg, mi2lnc_val_neg, mi2lnc_test_neg1))

    # mi2lnc_train_pos = mi2lnc_train_val_test_pos['mi2lnc_train_pos']
    # mi2lnc_val_pos = mi2lnc_train_val_test_pos['mi2lnc_val_pos']
    mi2lnc_test_pos = mi2lnc_train_val_test_pos['mi2lnc_test_pos']
    # mi2lnc_test_pos = np.concatenate((mi2lnc_train_pos, mi2lnc_val_pos, mi2lnc_test_pos1))
    print(len(mi2lnc_test_pos))
    # mi2gene_train_val_test_neg
    # mi2gene_train_val_test_pos
    # mi2gene_train_neg = mi2gene_train_val_test_neg['mi2gene_train_neg']
    # mi2gene_val_neg = mi2gene_train_val_test_neg['mi2gene_val_neg']
    mi2gene_test_neg = mi2gene_train_val_test_neg['mi2gene_test_neg']
    # mi2gene_test_neg = np.concatenate((mi2gene_train_neg, mi2gene_val_neg, mi2gene_test_neg1))

    # mi2gene_train_pos = mi2gene_train_val_test_pos['mi2gene_train_pos']
    # mi2gene_val_pos = mi2gene_train_val_test_pos['mi2gene_val_pos']
    mi2gene_test_pos = mi2gene_train_val_test_pos['mi2gene_test_pos']
    # mi2gene_test_pos = np.concatenate((mi2gene_train_pos, mi2gene_val_pos, mi2gene_test_pos1))
    # train_pos_miRNA_disease = train_val_test_pos_miRNA_disease['train_pos_miRNA_disease']
    # val_pos_miRNA_disease = train_val_test_pos_miRNA_disease['val_pos_miRNA_disease']
    # test_pos_miRNA_disease = train_val_test_pos_miRNA_disease['test_pos_miRNA_disease']
    # train_neg_miRNA_disease = train_val_test_neg_miRNA_disease['train_neg_miRNA_disease']
    # val_neg_miRNA_disease = train_val_test_neg_miRNA_disease['val_neg_miRNA_disease']
    # test_neg_miRNA_disease = train_val_test_neg_miRNA_disease['test_neg_miRNA_disease']
    # y_true_test = np.array([1] * len(test_pos_miRNA_disease) + [0] * len(test_neg_miRNA_disease))
    print(len(mi2gene_test_pos))
    auc_list = []
    ap_list = []
    for _ in range(repeat):
        print("repeat: "+str(_))
        # [3, 3]：指定每个元路径类型在图神经网络中的权重矩阵数量
        # 4：
        # etypes_lists：关系种类
        # in_dims：存储每个节点类型的维度。
        # hidden_dim：节点隐藏状态的维度，默认为 64
        # hidden_dim：节点隐藏状态的维度，默认为 64
        # num_heads：注意力头的数量，默认为 8
        # attn_vec_dim：注意力向量的维度，默认为 128
        # rnn_type：聚合器（aggregator）的类型，默认为 'max-pooling'
        # dropout_rate：随机失活（dropout）比例
        a = [15,11,11,11,15]
        net = PABDMH_lp(a, 5, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        # svm = SVC(kernel='rbf', C=50, gamma='auto', probability=True, cache_size=1000).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()

        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=checkpoint+'/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []

        index_generator = IndexGenerator([list(range(len(dis2mi_test_pos))), list(range(len(dis2circ_test_pos))),
                                        list(range(len(dis2lnc_test_pos))), list(range(len(dis2gene_test_pos))),
                                          list(range(len(mi2circ_test_pos))), list(range(len(mi2lnc_test_pos))),
                                          list(range(len(mi2gene_test_pos)))], batch_size)

        net.load_state_dict(torch.load(checkpoint+'/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        y_true_test_pos = []
        y_true_test_neg = []


        with torch.no_grad():
            pos_trapob = []
            neg_trapob = []
            indexs = 0
            for batch in index_generator:
                mi2dis_mi = []
                mi2dis_dis = []
                mi2dis_pro = []
                mi2circ_mi = []
                mi2circ_circ = []
                mi2circ_pro = []
                mi2lnc_mi = []
                mi2lnc_lnc = []
                mi2lnc_pro = []
                mi2gene_mi = []
                mi2gene_gene = []
                mi2gene_pro = []
                circ2dis_circ = []
                circ2dis_dis = []
                circ2dis_pro = []
                lnc2dis_lnc = []
                lnc2dis_dis = []
                lnc2dis_pro = []
                gene2dis_gene = []
                gene2dis_dis = []
                gene2dis_pro = []
                # print(batch.__len__())
            # for iteration in range(batch_size):
                # forward
                # y_true_test = np.array([1] * (
                #             len(dis2mi_test_pos) + len(dis2circ_test_pos) + len(dis2lnc_test_pos) + len(
                #         dis2gene_test_pos) + len(mi2circ_test_pos) + len(mi2lnc_test_pos) + len(mi2gene_test_pos))
                #                        + [0] * (len(dis2mi_test_neg) + len(dis2circ_test_neg) + len(
                #     dis2lnc_test_neg) + len(dis2gene_test_neg) + len(mi2circ_test_neg) + len(mi2lnc_test_neg) + len(
                #     mi2gene_test_neg)))
                if indexs % 10 == 0:
                    print(indexs)
                indexs = indexs + 1
                y_true_test_pos.extend([1] * batch_size*7)
                y_true_test_neg.extend([0] * batch_size * 7)
                pos_test_batch = []
                neg_test_batch = []
                # dis2mi_test_idx_batch = dis2mi_test_idx_generator.next()
                # print(len(dis2mi_test_pos))
                # print(batch[0])
                dis2mi_test_pos_batch = dis2mi_test_pos[batch[0]].tolist()
                dis2mi_test_neg_batch = dis2mi_test_neg[batch[0]].tolist()
                pos_test_batch.append(dis2mi_test_pos_batch)
                neg_test_batch.append(dis2mi_test_neg_batch)


                # dis2circ_test_idx_batch = dis2circ_test_idx_generator.next()
                dis2circ_test_pos_batch = dis2circ_test_pos[batch[1]].tolist()
                dis2circ_test_neg_batch = dis2circ_test_neg[batch[1]].tolist()
                pos_test_batch.append(dis2circ_test_pos_batch)
                neg_test_batch.append(dis2circ_test_neg_batch)

                # dis2lnc_test_idx_batch = dis2lnc_test_idx_generator.next()
                dis2lnc_test_pos_batch = dis2lnc_test_pos[batch[2]].tolist()
                dis2lnc_test_neg_batch = dis2lnc_test_neg[batch[2]].tolist()
                pos_test_batch.append(dis2lnc_test_pos_batch)
                neg_test_batch.append(dis2lnc_test_neg_batch)

                # dis2gene_test_idx_batch = dis2gene_test_idx_generator.next()
                dis2gene_test_pos_batch = dis2gene_test_pos[batch[3]].tolist()
                dis2gene_test_neg_batch = dis2gene_test_neg[batch[3]].tolist()
                pos_test_batch.append(dis2gene_test_pos_batch)
                neg_test_batch.append(dis2gene_test_neg_batch)

                # mi2circ_test_idx_batch = mi2circ_test_idx_generator.next()
                mi2circ_test_pos_batch = mi2circ_test_pos[batch[4]].tolist()
                mi2circ_test_neg_batch = mi2circ_test_neg[batch[4]].tolist()
                pos_test_batch.append(mi2circ_test_pos_batch)
                neg_test_batch.append(mi2circ_test_neg_batch)

                # mi2lnc_test_idx_batch = mi2lnc_test_idx_generator.next()
                mi2lnc_test_pos_batch = mi2lnc_test_pos[batch[5]].tolist()
                mi2lnc_test_neg_batch = mi2lnc_test_neg[batch[5]].tolist()
                pos_test_batch.append(mi2lnc_test_pos_batch)
                neg_test_batch.append(mi2lnc_test_neg_batch)

                # mi2gene_test_idx_batch = mi2gene_test_idx_generator.next()
                mi2gene_test_pos_batch = mi2gene_test_pos[batch[6]].tolist()
                mi2gene_test_neg_batch = mi2gene_test_neg[batch[6]].tolist()
                pos_test_batch.append(mi2gene_test_pos_batch)
                neg_test_batch.append(mi2gene_test_neg_batch)

                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists,data = parse_minibatch_PABDMH(
                    adjlists_ua, edge_metapath_indices_list_ua, pos_test_batch, neg_test_batch, device,
                    neighbor_samples, num)

                t1 = time.time()
                # dur1.append(t1 - t0)

                [embedding_miRNA, embedding_circRNA, embedding_lncRNA, embedding_gene, embedding_disease], [h_miRNA,
                                                                                                            h_circRNA,
                                                                                                            h_lncRNA,
                                                                                                            h_gene,
                                                                                                            h_disease] = net(
                    (test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists,data,num,num_embeding))

                # print(embedding_miRNA.shape)
                # print(embedding_circRNA.shape)
                # print(embedding_lncRNA.shape)
                # print(embedding_gene.shape)
                # print(embedding_disease.shape)
                # print("=========================")
                pos_embedding_miRNA, neg_embedding_miRNA = embedding_miRNA.chunk(2, dim=0)  # 每10个一组
                pos_embedding_circRNA, neg_embedding_circRNA = embedding_circRNA.chunk(2, dim=0)
                pos_embedding_lncRNA, neg_embedding_lncRNA = embedding_lncRNA.chunk(2, dim=0)
                pos_embedding_gene, neg_embedding_gene = embedding_gene.chunk(2, dim=0)
                pos_embedding_disease, neg_embedding_disease = embedding_disease.chunk(2, dim=0)

                pos_embedding_miRNA_1 = pos_embedding_miRNA.view(-1, 1, pos_embedding_miRNA.shape[1])
                neg_embedding_miRNA_1 = neg_embedding_miRNA.view(-1, 1, neg_embedding_miRNA.shape[1])
                pos_embedding_circRNA_1 = pos_embedding_circRNA.view(-1, 1, pos_embedding_circRNA.shape[1])
                pos_embedding_circRNA_2 = pos_embedding_circRNA.view(-1, pos_embedding_circRNA.shape[1], 1)
                neg_embedding_circRNA_1 = neg_embedding_circRNA.view(-1, 1, neg_embedding_circRNA.shape[1])
                neg_embedding_circRNA_2 = neg_embedding_circRNA.view(-1, neg_embedding_circRNA.shape[1], 1)
                pos_embedding_lncRNA_1 = pos_embedding_lncRNA.view(-1, 1, pos_embedding_lncRNA.shape[1])
                pos_embedding_lncRNA_2 = pos_embedding_lncRNA.view(-1, pos_embedding_lncRNA.shape[1], 1)
                neg_embedding_lncRNA_1 = neg_embedding_lncRNA.view(-1, 1, neg_embedding_lncRNA.shape[1])
                neg_embedding_lncRNA_2 = neg_embedding_lncRNA.view(-1, neg_embedding_lncRNA.shape[1], 1)
                pos_embedding_gene_1 = pos_embedding_gene.view(-1, 1, pos_embedding_gene.shape[1])
                pos_embedding_gene_2 = pos_embedding_gene.view(-1, pos_embedding_gene.shape[1], 1)
                neg_embedding_gene_1 = neg_embedding_gene.view(-1, 1, neg_embedding_gene.shape[1])
                neg_embedding_gene_2 = neg_embedding_gene.view(-1, neg_embedding_gene.shape[1], 1)
                pos_embedding_disease_2 = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                neg_embedding_disease_2 = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)
                # print(pos_embedding_miRNA_1.shape)
                # print(pos_embedding_miRNA_1[:,:, :batch_size].shape)
                # print(pos_embedding_disease_2.shape)
                # print(pos_embedding_disease_2[:,:batch_size,:].shape)
                mi2dis_pos_out = torch.bmm(pos_embedding_miRNA_1[:batch_size, :, :],
                                           pos_embedding_disease_2[:batch_size, :, :])
                mi2dis_neg_out = torch.bmm(neg_embedding_miRNA_1[:batch_size, :, :],
                                            neg_embedding_disease_2[:batch_size, :, :])
                circ2dis_pos_out = torch.bmm(pos_embedding_circRNA_1[:batch_size, :, :],
                                             pos_embedding_disease_2[batch_size:2 * batch_size, :, :])
                circ2dis_neg_out = torch.bmm(neg_embedding_circRNA_1[:batch_size, :, :],
                                              neg_embedding_disease_2[batch_size:2 * batch_size, :, :])
                lnc2dis_pos_out = torch.bmm(pos_embedding_lncRNA_1[:batch_size, :, :],
                                            pos_embedding_disease_2[2 * batch_size:3 * batch_size, :, :])
                lnc2dis_neg_out = torch.bmm(neg_embedding_lncRNA_1[:batch_size, :, :],
                                             neg_embedding_disease_2[2 * batch_size:3 * batch_size, :, :])
                gene2dis_pos_out = torch.bmm(pos_embedding_gene_1[:batch_size, :, :],
                                             pos_embedding_disease_2[3 * batch_size:4 * batch_size, :, :])
                gene2dis_neg_out = torch.bmm(neg_embedding_gene_1[:batch_size, :, :],
                                              neg_embedding_disease_2[3 * batch_size:4 * batch_size, :, :])
                mi2circ_pos_out = torch.bmm(pos_embedding_miRNA_1[batch_size:2 * batch_size, :, :],
                                            pos_embedding_circRNA_2[batch_size:2 * batch_size, :, :])
                mi2circ_neg_out = torch.bmm(neg_embedding_miRNA_1[batch_size:2 * batch_size, :, :],
                                             neg_embedding_circRNA_2[batch_size:2 * batch_size, :, :])
                mi2lnc_pos_out = torch.bmm(pos_embedding_miRNA_1[2 * batch_size:3 * batch_size, :, :],
                                           pos_embedding_lncRNA_2[batch_size:2 * batch_size, :, :])
                mi2lnc_neg_out = torch.bmm(neg_embedding_miRNA_1[2 * batch_size:3 * batch_size, :, :],
                                            neg_embedding_lncRNA_2[batch_size:2 * batch_size, :, :])
                mi2gene_pos_out = torch.bmm(pos_embedding_miRNA_1[3 * batch_size:4 * batch_size, :, :],
                                            pos_embedding_gene_2[batch_size:2 * batch_size, :, :])
                mi2gene_neg_out = torch.bmm(neg_embedding_miRNA_1[3 * batch_size:4 * batch_size, :, :],
                                             neg_embedding_gene_2[batch_size:2 * batch_size, :, :])
                # print(pos_embedding_miRNA_1.shape)
                # print(pos_embedding_miRNA_1[:,:, :batch_size].shape)
                # print(pos_embedding_disease_2.shape)
                # print(pos_embedding_disease_2[:,:batch_size,:].shape)
                pos_proba_list.append(torch.sigmoid(mi2dis_pos_out))

                neg_proba_list.append(torch.sigmoid(mi2dis_neg_out))

                pos_proba_list.append(torch.sigmoid(circ2dis_pos_out))

                neg_proba_list.append(torch.sigmoid(circ2dis_neg_out))

                pos_proba_list.append(torch.sigmoid(lnc2dis_pos_out))

                neg_proba_list.append(torch.sigmoid(lnc2dis_neg_out))

                pos_proba_list.append(torch.sigmoid(gene2dis_pos_out))

                neg_proba_list.append(torch.sigmoid(gene2dis_neg_out))

                pos_proba_list.append(torch.sigmoid(mi2circ_pos_out))

                neg_proba_list.append(torch.sigmoid(mi2circ_neg_out))

                pos_proba_list.append(torch.sigmoid(mi2lnc_pos_out))
                # print(torch.sigmoid(mi2lnc_neg_out).shape)
                neg_proba_list.append(torch.sigmoid(mi2lnc_neg_out))

                pos_proba_list.append(torch.sigmoid(mi2gene_pos_out))

                neg_proba_list.append(torch.sigmoid(mi2gene_neg_out))

                for mi2dis in dis2mi_test_pos_batch:
                    mi2dis_mi.append(mi2dis[0])
                    mi2dis_dis.append(mi2dis[1])
                for mi2dis in dis2mi_test_neg_batch:
                    mi2dis_mi.append(mi2dis[0])
                    mi2dis_dis.append(mi2dis[1])
                for circ2dis in dis2circ_test_pos_batch:
                    circ2dis_circ.append(circ2dis[0])
                    circ2dis_dis.append(circ2dis[1])
                for circ2dis in dis2circ_test_neg_batch:
                    circ2dis_circ.append(circ2dis[0])
                    circ2dis_dis.append(circ2dis[1])
                for lnc2dis in dis2lnc_test_pos_batch:
                    lnc2dis_lnc.append(lnc2dis[0])
                    lnc2dis_dis.append(lnc2dis[1])
                for lnc2dis in dis2lnc_test_neg_batch:
                    lnc2dis_lnc.append(lnc2dis[0])
                    lnc2dis_dis.append(lnc2dis[1])
                for gene2dis in dis2gene_test_pos_batch:
                    gene2dis_gene.append(gene2dis[0])
                    gene2dis_dis.append(gene2dis[1])
                for gene2dis in dis2gene_test_neg_batch:
                    gene2dis_gene.append(gene2dis[0])
                    gene2dis_dis.append(gene2dis[1])
                for mi2circ in mi2circ_test_pos_batch:
                    mi2circ_mi.append(mi2circ[0])
                    mi2circ_circ.append(mi2circ[1])
                for mi2circ in mi2circ_test_neg_batch:
                    mi2circ_mi.append(mi2circ[0])
                    mi2circ_circ.append(mi2circ[1])
                for mi2lnc in mi2lnc_test_pos_batch:
                    mi2lnc_mi.append(mi2lnc[0])
                    mi2lnc_lnc.append(mi2lnc[1])
                for mi2lnc in mi2lnc_test_neg_batch:
                    mi2lnc_mi.append(mi2lnc[0])
                    mi2lnc_lnc.append(mi2lnc[1])
                for mi2gene in mi2gene_test_pos_batch:
                    mi2gene_mi.append(mi2gene[0])
                    mi2gene_gene.append(mi2gene[1])
                for mi2gene in mi2gene_test_neg_batch:
                    mi2gene_mi.append(mi2gene[0])
                    mi2gene_gene.append(mi2gene[1])
                for pro in [item[0][0] for item in
                            torch.cat([torch.sigmoid(mi2dis_pos_out), torch.sigmoid(mi2dis_neg_out)]).cpu().numpy()]:
                    mi2dis_pro.append(pro)
                for pro in [item[0][0] for item in torch.cat(
                        [torch.sigmoid(circ2dis_pos_out), torch.sigmoid(circ2dis_neg_out)]).cpu().numpy()]:
                    circ2dis_pro.append(pro)
                for pro in [item[0][0] for item in
                            torch.cat([torch.sigmoid(lnc2dis_pos_out), torch.sigmoid(lnc2dis_neg_out)]).cpu().numpy()]:
                    lnc2dis_pro.append(pro)
                for pro in [item[0][0] for item in torch.cat(
                        [torch.sigmoid(gene2dis_pos_out), torch.sigmoid(gene2dis_neg_out)]).cpu().numpy()]:
                    gene2dis_pro.append(pro)
                for pro in [item[0][0] for item in
                            torch.cat([torch.sigmoid(mi2circ_pos_out), torch.sigmoid(mi2circ_neg_out)]).cpu().numpy()]:
                    mi2circ_pro.append(pro)
                for pro in [item[0][0] for item in
                            torch.cat([torch.sigmoid(mi2lnc_pos_out), torch.sigmoid(mi2lnc_neg_out)]).cpu().numpy()]:
                    mi2lnc_pro.append(pro)
                for pro in [item[0][0] for item in
                            torch.cat([torch.sigmoid(mi2gene_pos_out), torch.sigmoid(mi2gene_neg_out)]).cpu().numpy()]:
                    mi2gene_pro.append(pro)
                with open(r'./miRNAdisease_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(mi2dis_mi)):
                        wf.writerow([mi2dis_mi[i], mi2dis_dis[i], mi2dis_pro[i]])
                with open(r'./miRNAcircRNA_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(mi2circ_mi)):
                        wf.writerow([mi2circ_mi[i], mi2circ_circ[i], mi2circ_pro[i]])
                with open(r'./miRNAlncRNA_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(mi2lnc_mi)):
                        wf.writerow([mi2lnc_mi[i], mi2lnc_lnc[i], mi2lnc_pro[i]])
                with open(r'./miRNAgene_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(mi2gene_mi)):
                        wf.writerow([mi2gene_mi[i], mi2gene_gene[i], mi2gene_pro[i]])
                with open(r'./circRNAdisease_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(circ2dis_circ)):
                        wf.writerow([circ2dis_circ[i], circ2dis_dis[i], circ2dis_pro[i]])
                with open(r'./lncRNAdisease_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(lnc2dis_lnc)):
                        wf.writerow([lnc2dis_lnc[i], lnc2dis_dis[i], lnc2dis_pro[i]])
                with open(r'./genedisease_pro.csv', mode='a+', newline='', encoding='utf8') as cfa:
                    wf = csv.writer(cfa)
                    for i in range(len(gene2dis_gene)):
                        wf.writerow([gene2dis_gene[i], gene2dis_dis[i], gene2dis_pro[i]])

            print(pos_proba_list)
            y_proba_test = torch.cat([torch.cat(pos_proba_list), torch.cat(neg_proba_list)]).cpu().numpy()
            y_proba_test = [item[0][0] for item in y_proba_test]
            print(y_proba_test)
            print(len(y_proba_test))
            y_true_test = y_true_test_pos + y_true_test_neg
            print(len(y_true_test))
            print()
            auc = roc_auc_score(y_true_test, y_proba_test)
            ap = average_precision_score(y_true_test, y_proba_test)
            np.savetxt('1y_true_test.txt', y_true_test)
            np.savetxt('1y_proba_test.txt', y_proba_test)
            print('Link Prediction Test')
            print('AUC = {}'.format(auc))
            print('AP = {}'.format(ap))
            auc_list.append(auc)
            ap_list.append(ap)

    # print('----------------------------------------------------------------')
    # print('Link Prediction Tests Summary')
    # print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    # print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


# def main():
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PABDMH testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector;' +
                         '2 - all lncRNA vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='mix_pool', help='Type of the aggregator. Default is max-pooling. gru:lstm:bi-gru:bi-lstm:linear:max-pooling:neighbor-linear')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=10, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='PABDMH', help='Postfix for the saved model and result. Default is PABDMH.')
    ap.add_argument('--num_embeding', type=int, default=1, help='Batch size. Default is 8.')
    # ap.add_argument('--save-postfix', default='PABDMH')
    ap.add_argument('--checkpoint', default='checkpoint')
    args = ap.parse_args()

    run_model_PABDMH(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix,args.num_embeding,args.checkpoint)