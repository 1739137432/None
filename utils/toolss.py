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


def parse_minibatch_PABDMH(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch_pos,user_artist_batch_neg, device, samples=None, offset=None):

    g_lists = [[], [], [], [], []]
    result_indices_lists = [[], [], [], [], []]
    idx_batch_mapped_lists = [[], [], [], [], []]

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

            g = dgl.DGLGraph(multigraph=True).to(device)
            g.add_nodes(num_nodes)


            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)

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
        print(self.max_length)
    def __iter__(self):
        return self

    def __next__(self):

        if self.current_indice >= self.max_length:
            # print(self.current_indice)
            raise StopIteration

        results = []

        for i, data in enumerate(self.datasets):
            if len(self.current_indices[i]) >= len(data):
                self.current_indices[i] = set()

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

            results.append([data[index] for index in selected_indices])
        # if len(data) == self.max_length:
        self.current_indice = self.current_indice + self.size
        return tuple(results)
