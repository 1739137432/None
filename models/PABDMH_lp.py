import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from sklearn.svm import SVC


class MixPool2d(nn.Module):
    def __init__(self, out_dim, num_heads,p=0.5):
        super(MixPool2d, self).__init__()
        # self.kernel_size = kernel_size
        # self.stride = stride if stride is not None else kernel_size
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, edata):
        # 计算最大池化结果
        max_pool, _ = torch.max(self.rnn(edata), dim=1)
        max_pool = max_pool.unsqueeze(dim=0)
        # 计算平均池化结果
        avg_pool = torch.mean(edata, dim=1)
        avg_pool = torch.cat([avg_pool] * self.num_heads, dim=1)
        avg_pool = avg_pool.unsqueeze(dim=0)
        # 将最大池化结果和平均池化结果按一定比例相加
        mix_pool = self.p*max_pool + (1-self.p)*avg_pool
        # mix_pool = (avg_pool ** self.p + max_pool ** self.p) ** (1 / self.p)
        return mix_pool

class PABDMH_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.001,
                 minibatch=False,
                 attn_switch=False):
        super(PABDMH_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.minibatch = minibatch
        self.attn_switch = attn_switch

        # rnn-like metapath instance aggregator
        # consider multiple attention heads

        # self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif self.rnn_type == 'mix_pool':
            self.mix_pool = MixPool2d(out_dim, num_heads)


        # node-level attention
        # attention considers the center node embedding or not
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # miRNA torch.nn.functional.embedding or torch.embedding here
        # do not miRNA torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence

        # hidden, _ = torch.max(self.rnn(edata), dim=1)
        # hidden = hidden.unsqueeze(dim=0)
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'mix_pool':
            hidden = self.mix_pool(edata)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        # g = g.to('cuda:0')
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.minibatch:
            return ret[target_idx]
        else:
            return ret
class PABDMH_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 minibatch=False):
        super(PABDMH_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.minibatch = minibatch

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(PABDMH_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                minibatch=minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
# for link prediction task
class PABDMH_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(PABDMH_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.miRNA_layer = PABDMH_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   minibatch=True)
        self.circRNA_layer = PABDMH_ctr_ntype_specific(num_metapaths_list[1],
                                                     etypes_lists[1],
                                                     in_dim,
                                                     num_heads,
                                                     attn_vec_dim,
                                                     rnn_type,
                                                     r_vec,
                                                     attn_drop,
                                                     minibatch=True)
        self.lncRNA_layer = PABDMH_ctr_ntype_specific(num_metapaths_list[2],
                                                     etypes_lists[2],
                                                     in_dim,
                                                     num_heads,
                                                     attn_vec_dim,
                                                     rnn_type,
                                                     r_vec,
                                                     attn_drop,
                                                     minibatch=True)
        self.gene_layer = PABDMH_ctr_ntype_specific(num_metapaths_list[3],
                                                     etypes_lists[3],
                                                     in_dim,
                                                     num_heads,
                                                     attn_vec_dim,
                                                     rnn_type,
                                                     r_vec,
                                                     attn_drop,
                                                     minibatch=True)
        self.disease_layer = PABDMH_ctr_ntype_specific(num_metapaths_list[4],
                                                   etypes_lists[4],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_miRNA = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_circRNA = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_lncRNA = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_gene = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_disease = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_miRNA.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_circRNA.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_lncRNA.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_gene.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_disease.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_miRNA = self.miRNA_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_circRNA = self.circRNA_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))
        h_lncRNA = self.lncRNA_layer(
            (g_lists[2], features, type_mask, edge_metapath_indices_lists[2], target_idx_lists[2]))
        h_gene = self.gene_layer(
            (g_lists[3], features, type_mask, edge_metapath_indices_lists[3], target_idx_lists[3]))
        h_disease = self.disease_layer(
            (g_lists[4], features, type_mask, edge_metapath_indices_lists[4], target_idx_lists[4]))

        # print("h_miRNA:")
        # print(h_miRNA)
        logits_miRNA = self.fc_miRNA(h_miRNA)
        logits_circRNA = self.fc_circRNA(h_circRNA)
        logits_lncRNA = self.fc_lncRNA(h_lncRNA)
        logits_gene = self.fc_gene(h_gene)
        logits_disease = self.fc_disease(h_disease)
        return [logits_miRNA, logits_circRNA, logits_lncRNA, logits_gene, logits_disease],\
               [h_miRNA, h_circRNA, h_lncRNA, h_gene, h_disease]
class PABDMH_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(PABDMH_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # PABDMH_lp layers
        self.layer1 = PABDMH_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists, data, num ,num_embeding= inputs
        # print(features_list)
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        # print(transformed_features.shape)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)
        # print(transformed_features.shape)

        l_miRNA = []
        l_circRNA = []
        l_lncRNA = []
        l_gene = []
        l_disease = []
        for i,node_type in enumerate(data):
            for index in node_type:
                if i == 0:
                    l_miRNA.append(transformed_features[index])
                elif i == 1:
                    l_circRNA.append(transformed_features[index+num[0]])
                elif i == 2:
                    l_lncRNA.append(transformed_features[index + num[0] + num[1]])
                elif i == 3:
                    l_gene.append(transformed_features[index + num[0] + num[1] + num[2]])
                else:
                    l_disease.append(transformed_features[index + num[0] + num[3] + num[1] + num[2]])


        for _ in range(num_embeding):
            [logits_miRNA, logits_circRNA,logits_lncRNA,logits_gene,logits_disease], \
            [h_miRNA, h_circRNA,h_lncRNA,h_gene,h_disease] = \
                self.layer1((g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
            # print(logits_miRNA.shape)

            # 找出最初的五种节点的特征，与进过模型训练之后的特征与相加，然后去relu
            # 问题是做怎么找到这几个节点对应的初始特征
            # 目前想到的就是将下标传过来，然后去transformed_features去找出最初始的特征向量
            logits_miRNA =  F.relu(logits_miRNA+torch.stack(l_miRNA))
            logits_circRNA = F.relu(logits_circRNA+torch.stack(l_circRNA))
            logits_lncRNA = F.relu(logits_lncRNA+torch.stack(l_lncRNA))
            logits_gene = F.relu(logits_gene+torch.stack(l_gene))
            logits_disease= F.relu(logits_disease+torch.stack(l_disease))

            for i, node_type in enumerate(data):
                ba = 0
                for index in node_type:
                    if i == 0:
                        # print(logits_miRNA[ba])
                        # print(transformed_features[ba])
                        transformed_features[index] = logits_miRNA[ba]
                    elif i == 1:
                        transformed_features[index + num[0]] = logits_circRNA[ba]
                    elif i == 2:
                        transformed_features[index + num[0] + num[1]] = logits_lncRNA[ba]
                    elif i == 3:
                        transformed_features[index + num[0] + num[1] + num[2]] = logits_gene[ba]
                    else:
                        transformed_features[index + num[0] + num[3] + num[1] + num[2]] = logits_disease[ba]
                    ba=ba+1
            #
            [logits_miRNA, logits_circRNA, logits_lncRNA, logits_gene, logits_disease], \
                [h_miRNA, h_circRNA, h_lncRNA, h_gene, h_disease] = \
                self.layer1((g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        return [logits_miRNA, logits_circRNA,logits_lncRNA,logits_gene,logits_disease], \
        [h_miRNA, h_circRNA,h_lncRNA,h_gene,h_disease]