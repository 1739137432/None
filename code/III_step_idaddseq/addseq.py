import csv
import os

import pandas as pd


def gene_add_seq():
    seq_genes = []
    seq_gene_seq = []
    with open('../output/relationship/I_step_sumSeq/gene_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id, gene, gene_seq = row
            seq_genes.append(gene)
            seq_gene_seq.append(gene_seq)
    print(1)
    gene_id_genes = []
    gene_id_geneids = []
    gene_id_geneseq = []
    index = 0
    with open('../output/relationship/II_step_ncRNA_disease_id/gene_id.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:

            id, gene = row

            if gene in seq_genes:
                gene_id_geneids.append(index)
                gene_id_genes.append(gene)
                gene_id_geneseq.append(seq_gene_seq[seq_genes.index(gene)])
                index += 1
            # else:
            #     miRNA_id_miRNAseq.append("N/A")
    print(1)
    path_df = pd.DataFrame()
    path_df['id'] = gene_id_geneids
    path_df['gene'] = gene_id_genes
    path_df['gene_seq'] = gene_id_geneseq
    path_df.to_csv("../output/relationship/III_step_idaddseq/gene_id_seq.csv", index=False, header=True)


def lncRNA_add_seq():
    seq_lncRNAs = []
    seq_lncRNA_seq = []

    with open('../output/relationship/I_step_sumSeq/lncrna_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id, lncrna, lncrna_seq = row
            seq_lncRNAs.append(lncrna)
            seq_lncRNA_seq.append(lncrna_seq)

    print(1)
    lncRNA_id_lncRNAs = []
    lncRNA_id_lncRNAids = []
    lncRNA_id_lncRNAseq = []
    index = 0
    with open('../output/relationship/II_step_ncRNA_disease_id/lncRNA_id.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id, lncrna = row

            if lncrna in seq_lncRNAs:
                lncRNA_id_lncRNAids.append(index)
                lncRNA_id_lncRNAs.append(lncrna)
                lncRNA_id_lncRNAseq.append(seq_lncRNA_seq[seq_lncRNAs.index(lncrna)])
                index += 1
            # else:
            #     lncRNA_id_lncRNAseq.append("N/A")
    print(1)
    path_df2 = pd.DataFrame()
    path_df2['id'] = lncRNA_id_lncRNAids
    path_df2['lncrna'] = lncRNA_id_lncRNAs
    path_df2['lncrna_seq'] = lncRNA_id_lncRNAseq
    path_df2.to_csv("../output/relationship/III_step_idaddseq/LncRNA_id_seq.csv", index=False, header=True)


def miRNA_add_seq():
    seq_miRNAs = []
    seq_miRNA_seq = []
    with open('../output/relationship/I_step_sumSeq/mirna_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id, mirna, mirna_seq = row
            seq_miRNAs.append(mirna)
            seq_miRNA_seq.append(mirna_seq)
    print(1)
    miRNA_id_miRNAs = []
    miRNA_id_miRNAids = []
    miRNA_id_miRNAseq = []
    index = 0
    with open('../output/relationship/II_step_ncRNA_disease_id/miRNA_id.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:

            id, mirna = row

            if mirna in seq_miRNAs:
                miRNA_id_miRNAids.append(index)
                miRNA_id_miRNAs.append(mirna)
                miRNA_id_miRNAseq.append(seq_miRNA_seq[seq_miRNAs.index(mirna)])
                index += 1
            # else:
            #     miRNA_id_miRNAseq.append("N/A")
    print(1)
    # 检查保存路径中的目录是否存在，如果不存在则创建
    if not os.path.exists('../output/relationship/III_step_idaddseq'):
        os.makedirs('../output/relationship/III_step_idaddseq')
    path_df = pd.DataFrame()
    path_df['id'] = miRNA_id_miRNAids
    path_df['mirna'] = miRNA_id_miRNAs
    path_df['mirna_seq'] = miRNA_id_miRNAseq
    path_df.to_csv("../output/relationship/III_step_idaddseq/miRNA_id_seq.csv", index=False, header=True)