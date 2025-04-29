import csv
import os

import pandas as pd
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# k-mer
def k_mers(k, seq):
    if k > len(seq):
        return []
    num = len(seq)-k+1
    split = []
    for i in range(num):
        split.append(seq[i:i+k])
    return split

def train_doc2vec_model(mers, name):
    # Define a function to train a Doc2Vec model on a list of strings
    tagged_docs = create_tagged_documents(mers, name)
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
def get_vector_embeddings(all_mers, all_name, model):
    # Define a function to get vector embeddings of strings using a Doc2Vec model
    tagged_docs = create_tagged_documents(all_mers, all_name)
    vectors = np.array([])
    vectors = {}
    for doc in tagged_docs:
        vectors[doc.tags] = model.infer_vector(doc.words)
    return vectors
def create_tagged_documents(mers, name):
    # Define a function to create tagged documents from a list of strings
    tagged_docs = [TaggedDocument(mers[i], name[i])
                   for i in range(len(name))]

    return tagged_docs


def miRNA_similarity():
    mirnas = pd.read_csv("../output/relationship/III_step_idaddseq/miRNA_id_seq.csv")

    longName = []
    shortName = []
    longSeq = []
    with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
            longName.append(Symbol)
            shortName.append(Symbol1)
            longSeq.append(Sequence)
    mirnas_name = mirnas["mirna"].tolist()
    for name in mirnas_name:
        if name in shortName:
            mirnas = mirnas.drop(mirnas_name.index(name))
    unique_mi = list(set(mirnas['mirna']))

    mi_seq = []

    for i in unique_mi:
        seq = mirnas[mirnas['mirna'] == i]["mirna_seq"]
        seq = list(seq)
        seq = seq[0]
        seq = seq.replace('.', '')
        if ',' in seq:
            seq = seq.split(',')
            seq = seq[0]
        mi_seq.append(seq)

    mi_seq_mers = []

    for i in mi_seq:
        mi_seq_mers.append(k_mers(3, i))

    all_mers = mi_seq_mers
    all_name = unique_mi

    if not os.path.exists('../output/relationship/IV_step_similarity/'):
        os.makedirs('../output/relationship/IV_step_similarity/')
    with open("../output/relationship/IV_step_similarity/miRNASim.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in range(len(all_name)):
            for na in range(len(all_name)):
                writer.writerow([name] + [na])

    with open("../output/relationship/IV_step_similarity/miRNA_id.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in all_name:
            writer.writerow([name] + [all_name.index(name)])

    # miRNAid = []
    # adjid = []
    # for name in all_name:
    #     for na in all_name:
    #         miRNAid.append(all_name.index(name))
    #         adjid.append(all_name.index(na))
    #
    # path_df = pd.DataFrame()
    # path_df['miRNAid'] = miRNAid
    # path_df['adjid'] = adjid
    # path_df.to_csv("../output/relationship/IV_step_similarity/miRNASim.csv", index=False, header=False)
    # miRNA = []
    # for name in all_name:
    #         miRNA.append(name + "," + str(all_name.index(name)))
    # path_df1 = pd.DataFrame()
    # path_df1['miRNA'] = miRNA
    # path_df1.to_csv("../output/relationship/IV_step_similarity/miRNA_id.csv", index=False, header=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pretrain_model = train_doc2vec_model(all_mers, all_name)
    vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)

    graph_embedding = np.zeros((len(all_name), 100))
    for node, vec in vectors.items():
        position = all_name.index(node)
        graph_embedding[position] = vec
    x_embedding = torch.tensor(graph_embedding).float()
    print(x_embedding)


    euclidean_distances = np.linalg.norm(x_embedding[:, np.newaxis] - x_embedding, axis=2)
    # manhattan_distances = np.sum(np.abs(x_embedding[:, np.newaxis] - x_embedding), axis=2)


    scaler = MinMaxScaler()
    normalized_euclidean_distances = scaler.fit_transform(euclidean_distances)
    # normalized_manhattan_distances = scaler.fit_transform(manhattan_distances)


    similarity_euclidean = 1 - normalized_euclidean_distances
    print(type(similarity_euclidean))

    with open('../output/relationship/IV_step_similarity/miRNA_similarity.csv', 'w') as f:
        for row in similarity_euclidean:
            for value in row:
                f.write(str(value) + '\n')


def lncRNA_similarity():
    lncrnas = pd.read_csv("../output/relationship/III_step_idaddseq/LncRNA_id_seq.csv")

    unique_lnc = list(set(lncrnas['lncrna']))

    lnc_seq = []

    for i in unique_lnc:
        seq = lncrnas[lncrnas['lncrna'] == i]["lncrna_seq"]
        seq = list(seq)
        seq = seq[0]
        seq = seq.translate(str.maketrans('', '', string.punctuation))
        lnc_seq.append(seq)

    # k-mers切分
    lnc_seq_mers = []

    for i in lnc_seq:
        lnc_seq_mers.append(k_mers(3, i))

        # [[],[]..284]  [[],[]..520]
    all_mers = lnc_seq_mers
    all_name = unique_lnc
    with open("../output/relationship/IV_step_similarity/lncRNASim.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in range(len(all_name)):
            for na in range(len(all_name)):
                writer.writerow([name] + [na])

    with open("../output/relationship/IV_step_similarity/lncRNA_id.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in all_name:
            writer.writerow([name] + [all_name.index(name)])
    # lncRNAid = []
    #
    # adjid = []
    # for name in all_name:
    #     for na in all_name:
    #         lncRNAid.append(all_name.index(name))
    #         adjid.append(all_name.index(na))
    #
    #
    # path_df = pd.DataFrame()
    # path_df['lncRNAid'] = lncRNAid
    # path_df['adjid'] = adjid
    # path_df.to_csv("../output/relationship/IV_step_similarity/lncRNASim.csv", index=False, header=False)
    # lncRNA = []
    # for name in all_name:
    #         lncRNA.append(name + "," + str(all_name.index(name)))
    # path_df1 = pd.DataFrame()
    # path_df1['lncRNA'] = lncRNA
    # path_df1.to_csv("../output/relationship/IV_step_similarity/lncRNA_id.csv", index=False, header=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pretrain_model = train_doc2vec_model(all_mers, all_name)
    vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)

    graph_embedding = np.zeros((len(all_name), 100))
    for node, vec in vectors.items():
        position = all_name.index(node)
        graph_embedding[position] = vec
    x_embedding = torch.tensor(graph_embedding).float()
    print(x_embedding)


    euclidean_distances = np.linalg.norm(x_embedding[:, np.newaxis] - x_embedding, axis=2)


    scaler = MinMaxScaler()
    normalized_euclidean_distances = scaler.fit_transform(euclidean_distances)


    similarity_euclidean = 1 - normalized_euclidean_distances
    with open('../output/relationship/IV_step_similarity/lncRNA_similarity.csv', 'w') as f:
        for row in similarity_euclidean:
            for value in row:
                f.write(str(value) + '\n')


def gene_similarity():
    genes = pd.read_csv("../output/relationship/III_step_idaddseq/gene_id_seq.csv")

    unique_gene = list(set(genes['gene']))

    gene_seq = []

    for i in unique_gene:
        seq = genes[genes['gene'] == i]["gene_seq"]
        seq = list(seq)
        seq = seq[0]
        seq = seq.translate(str.maketrans('', '', string.punctuation))
        gene_seq.append(seq)

    # k-mers切分
    gene_seq_mers = []

    for i in gene_seq:
        gene_seq_mers.append(k_mers(3, i))

        # [[],[]..284]  [[],[]..520]
    all_mers = gene_seq_mers
    all_name = unique_gene

    with open("../output/relationship/IV_step_similarity/geneSim.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in range(len(all_name)):
            for na in range(len(all_name)):
                writer.writerow([name] + [na])

    with open("../output/relationship/IV_step_similarity/gene_id.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in all_name:
            writer.writerow([name] + [all_name.index(name)])

    # geneid = []
    # adjid = []
    # for name in all_name:
    #     for na in all_name:
    #         geneid.append(all_name.index(name))
    #         adjid.append(all_name.index(na))
    #
    # path_df = pd.DataFrame()
    # path_df['geneid'] = geneid
    # path_df['adjid'] = adjid
    # path_df.to_csv("../output/relationship/IV_step_similarity/geneSim.csv", index=False, header=False)
    # gene = []
    # for name in all_name:
    #         gene.append(str(all_name.index(name))+","+name  )
    # path_df1 = pd.DataFrame()
    # path_df1['gene'] = gene
    # path_df1.to_csv("../output/relationship/IV_step_similarity/gene_id.csv", index=False, header=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pretrain_model = train_doc2vec_model(all_mers, all_name)
    vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)

    graph_embedding = np.zeros((len(all_name), 100))
    for node, vec in vectors.items():
        position = all_name.index(node)
        graph_embedding[position] = vec
    x_embedding = torch.tensor(graph_embedding).float()
    print(x_embedding)


    euclidean_distances = np.linalg.norm(x_embedding[:, np.newaxis] - x_embedding, axis=2)


    scaler = MinMaxScaler()
    normalized_euclidean_distances = scaler.fit_transform(euclidean_distances)


    similarity_euclidean = 1 - normalized_euclidean_distances
    with open('../output/relationship/IV_step_similarity/gene_similarity.csv', 'w') as f:
        for row in similarity_euclidean:
            for value in row:
                f.write(str(value) + '\n')


def circRNA_similarity():
    circrnas = pd.read_csv("../output/relationship/II_step_ncRNA_disease_id/circRNA_id.csv")
    unique_circ = list(set(circrnas['circRNA']))

    circ_seq = []
    print(unique_circ)
    for i in unique_circ:
        seq = circrnas[circrnas['circRNA'] == i]["circRNA_seq"]
        seq = list(seq)
        seq = seq[0]
        # print(seq)
        seq = seq.translate(str.maketrans('', '', string.punctuation))
        circ_seq.append(seq)
    print(1)
    circ_seq_mers = []

    for i in circ_seq:
        circ_seq_mers.append(k_mers(3, i))
    print(1)
    all_mers = circ_seq_mers  # k-mers list[['AGC',GCG...],[],...804]
    all_name = unique_circ  # RNAname list['','',...804]

    with open("../output/relationship/IV_step_similarity/circRNASim.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in range(len(all_name)):
            for na in range(len(all_name)):
                writer.writerow([name] + [na])

    with open("../output/relationship/IV_step_similarity/circRNA_id.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for name in all_name:
            writer.writerow([name] + [all_name.index(name)])

    # circRNA = []
    # print(1)
    # for name in all_name:
    #         circRNA.append([name] + [all_name.index(name)])
    # path_df1 = pd.DataFrame()
    # path_df1['circRNA'] = circRNA
    # path_df1.to_csv("../output/relationship/IV_step_similarity/circRNA_id.csv", index=False, header=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # print(1)
    pretrain_model = train_doc2vec_model(all_mers, all_name)
    print(1)
    vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)

    print(1)
    graph_embedding = np.zeros((len(all_name), 100))
    print(1)
    for node, vec in vectors.items():
        position = all_name.index(node)
        graph_embedding[position] = vec
    print(1)
    x_embedding = torch.tensor(graph_embedding).float()
    print(x_embedding)


    euclidean_distances = np.linalg.norm(x_embedding[:, np.newaxis] - x_embedding, axis=2)
    print(1)

    scaler = MinMaxScaler()
    print(1)
    normalized_euclidean_distances = scaler.fit_transform(euclidean_distances)
    print(1)


    similarity_euclidean = (1 - normalized_euclidean_distances)
    print(1)
    with open('../output/relationship/IV_step_similarity/circRNA_similarity.csv', 'w') as f:
        for row in similarity_euclidean:
            for value in row:
                f.write(str(value) + '\n')