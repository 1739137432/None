import random

import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

# def idx_to_one_hot(idx_arr):
#     one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
#     one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
#     return one_hot

def parse_adjlist_PABDMH(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    # print("我到这！！------2")
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:

                    # if mode == 0:
                    #     mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    # else:
                    #     mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    # print(mask)
                    mask = [False if [u1, a1] in exclude or [u2, a2] in exclude else True for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    mask = [False if [u1, a1] in exclude or [u2, a2] in exclude else True for
                            u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    # if mode == 0: # mi dis(X)  mi  dis(X)
                    #     mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    # else:        # dis mi dis mi
                    #     mask = [False if [a1, u1 - offset] in exclude or [a2, u2 - offset] in exclude else True for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    #     # mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]

                    # print(indices[sampled_idx])
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])


            # if mode == 1:
            #     indices += offset



            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping

# 解析一个批量的数据，将它们转化为DGL图对象以及其他信息，用于模型输入
# adjlists_ua:各个元路径的邻接
# edge_metapath_indices_list_ua:各个元路径的下标
# user_artist_batch:数据batch，是一个列表，包含7中关系
# device：使用的设备，GPU/CPU
# samples=None  邻居节点采样数量  默认100
# use_masks=None：各种路径的标签，7种
# offset=None  各种节点的数量，是一个列表

                # adjlists_ua:原[[],[]]    现[[],[],[],[],[]]    2:5
                # edge_metapath_indices_list_ua:原[[],[]]    现[[],[],[],[],[]]    2:5
                # train_batch原：X     现：[X,X,X,X,X,X,X]     1:7
                # masks   [[],[]]      [[[],[],[],[],[]],
                #                       [[],[],[],[],[]],
                #                       [[],[],[],[],[]],
                #                       [[],[],[],[],[]],
                #                       [[],[],[],[],[]],
                #                       [[],[],[],[],[]],
                #                       [[],[],[],[],[]]]   2:7*5
                # nums   X    [X,X,X,X,X]
def parse_minibatch_PABDMH(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch_pos,user_artist_batch_neg, device, samples=None, offset=None):
    # g_lists： 存储生成的图数据的列表
    # result_indices_lists: 存储结果索引的列表。
    # idx_batch_mapped_lists: 存储映射的索引的列表。
    g_lists = [[], [], [], [], []]
    result_indices_lists = [[], [], [], [], []]
    idx_batch_mapped_lists = [[], [], [], [], []]
    # g_lists = [[], [] ]
    # result_indices_lists = [[], []]
    # idx_batch_mapped_lists = [[], []]
    # print("我到这！！------1")
    data = []
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        tem = []

        for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
            # print(mode)
            if mode == 0: #miRNA
                user_artist_batch1 = user_artist_batch_pos[0] + user_artist_batch_pos[4] +user_artist_batch_pos[5] + user_artist_batch_pos[6]+\
                                     user_artist_batch_neg[0] + user_artist_batch_neg[4] +user_artist_batch_neg[5] + user_artist_batch_neg[6]
            elif mode == 1: #circRNA
                user_artist_batch1 = user_artist_batch_pos[1] + [row[::-1] for row in user_artist_batch_pos[4]] + \
                                     user_artist_batch_neg[1] + [row[::-1] for row in user_artist_batch_neg[4]]
            elif mode == 2: #lncRNA
                user_artist_batch1 = user_artist_batch_pos[2] + [row[::-1] for row in user_artist_batch_pos[5]] + \
                                     user_artist_batch_neg[2] + [row[::-1] for row in user_artist_batch_neg[5]]
            elif mode == 3: #gene
                user_artist_batch1 = user_artist_batch_pos[3] + [row[::-1] for row in user_artist_batch_pos[6]] + \
                                     user_artist_batch_neg[3] + [row[::-1] for row in user_artist_batch_neg[6]]
            else:  # disease
                user_artist_batch1 = [row[::-1] for row in user_artist_batch_pos[0]] + [row[::-1] for row in user_artist_batch_pos[1]] +\
                                     [row[::-1] for row in user_artist_batch_pos[2]]+[row[::-1] for row in user_artist_batch_pos[3]] + \
                                     [row[::-1] for row in user_artist_batch_neg[0]] + [row[::-1] for row in user_artist_batch_neg[1]] +\
                                     [row[::-1] for row in user_artist_batch_neg[2]]+[row[::-1] for row in user_artist_batch_neg[3]]
            # print(user_artist_batch1)




            if len(tem) == 0:
                for row in user_artist_batch1:
                    # if row[0] not in tem:
                    tem.append(row[0])




            edges, result_indices, num_nodes, mapping = parse_adjlist_PABDMH(
                [adjlist[int(row[0])] for row in user_artist_batch1],
                [indices[int(row[0])] for row in user_artist_batch1], samples, user_artist_batch1, offset, mode)

            # 使用 dgl.DGLGraph 创建一个多图对象 g，添加相应数量的节点
            g = dgl.DGLGraph(multigraph=True).to(device)
            g.add_nodes(num_nodes)


            # 如果有边需要添加，它根据边的排序索引，将边添加到多图 g 中
            # 将结果索引转换为PyTorch的LongTensor，并添加到result_indices_lists 列表中
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)

            # print(len(mapping))
            # print(mapping)
            # 将映射的索引添加到 idx_batch_mapped_lists 列表中
            idx_batch_mapped_lists[mode].append(np.array([mapping[int(row[0])] for row in user_artist_batch1]))
        data.append(tem)
    return g_lists, result_indices_lists, idx_batch_mapped_lists,data

class IndexGenerator:
    def __init__(self, datasets,size):
        self.datasets = datasets
        self.size = size
        self.current_indices = [set() for _ in range(len(datasets))]
        self.max_length = max(len(data) for data in datasets)
        self.current_indice = 0
        # print(self.max_length)
    def __iter__(self):
        return self

    def __next__(self):

        if self.current_indice >= self.max_length:
            # print(self.current_indice)
            raise StopIteration

        results = []

        for i, data in enumerate(self.datasets):
            if len(self.current_indices[i]) >= len(data):
                # 如果该数据集的所有元素都已经被抽取过，则重新开始抽取
                self.current_indices[i] = set()

            # 从数据集中抽取未抽取过的元素
            available_indices = set(range(len(data))) - self.current_indices[i]
            if self.size > len(available_indices):
                selected_indices = random.sample(available_indices, len(available_indices))
                shengxiade = self.size - len(available_indices)
                # print(selected_indices)
                self.current_indices[i] = set()

                available_indices = set(range(len(data))) - self.current_indices[i]
                selected_indices1 = random.sample(available_indices, shengxiade)
                selected_indices = selected_indices + selected_indices1
                self.current_indices[i].update(selected_indices)
                # print(selected_indices1)
                # print(selected_indices)
            else:
                selected_indices = random.sample(available_indices, self.size)
                self.current_indices[i].update(selected_indices)
            # sample_size = min(self.size, len(available_indices))



            # 将抽取的元素添加到结果中
            results.append([data[index] for index in selected_indices])
        # if len(data) == self.max_length:
        self.current_indice = self.current_indice + self.size
        return tuple(results)

# class index_generator:
#     # 初始化函数，可以指定批量大小 batch_size、数据总数 num_data、自定义索引数组 indices 和是否进行洗牌 shuffle。
#     # 如果提供了 num_data，则将数据总数设置为该值，并生成从 0 到 num_data - 1 的索引。
#     # 如果提供了 indices，则使用该索引作为数据索引。
#     # 如果 shuffle 为 True，则在初始化时对索引进行洗牌。
#     def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
#         if num_data is not None:
#             self.num_data = num_data
#             self.indices = np.arange(num_data)
#         if indices is not None:
#             self.num_data = len(indices)
#             self.indices = np.copy(indices)
#         self.batch_size = batch_size
#         self.iter_counter = 0
#         self.shuffle = shuffle
#         if shuffle:
#             np.random.shuffle(self.indices)
#
#     # 返回下一个批量的索引数组。它会递增迭代计数器，根据当前迭代计数器的值计算批量的起始索引和结束索引，并返回这个范围内的索引数组
#     def next(self):
#         if self.num_iterations_left() <= 0:
#             self.reset()
#         self.iter_counter += 1
#         return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])
#
#     # 返回数据集在给定批量大小下所需的迭代次数（总批量数）
#     def num_iterations(self):
#         return int(np.ceil(self.num_data / self.batch_size))
#
#     # 返回剩余迭代次数，即总迭代次数减去当前迭代次数
#     def num_iterations_left(self):
#         return self.num_iterations() - self.iter_counter
#
#     # 重置迭代计数器，并根据是否需要洗牌来重新洗牌索引数组
#     def reset(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
#         self.iter_counter = 0