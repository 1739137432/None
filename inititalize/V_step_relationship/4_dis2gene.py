import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(50000 * 1024 * 1024)
# diseases = []
# genes = []
# pmids = []
# databases = []
# with open('../input/relationship/DisGeNET_20230406/newcircRNA-Disease.csv', newline='', encoding='utf-8') as csvfile1:
#     reader1 = csv.reader(csvfile1, delimiter=',', quotechar='"')
#     next(reader1)  # Skip header row
#     for row in reader1:
#         diseaseName, geneName, pmid = row
#         # for circRNA in circRNAs:
#         #     if circRNA == geneName &&
#         diseases.append(diseaseName)
#         # aliass.append(alias)
#         genes.append(geneName)
#         pmids.append(pmid)
#         databases.append("DisGeNET")
#
#
# path_df = pd.DataFrame()
# path_df['gene'] = genes
# path_df['disease'] = diseases
# path_df['database'] = databases
# path_df['pmid'] = pmids
# path_df.to_csv("../input/relationship/DisGeNET_20230406/Disease_gene.csv", index=False, header=True)
#
# inFile = open('../input/relationship/DisGeNET_20230406/Disease_gene.csv', 'r')  #
# outFile = open('../output/relationship/DisGeNET_20230406/Disease_gene.csv', 'w')  # 最后保存的.csv文件
# listLines = []
# for line in inFile:
#     if line in listLines:
#         continue
#     else:
#         outFile.write(line)
#         listLines.append(line)
# outFile.close()
# inFile.close()

def dis2gene():
    DisGeNETdiseases = []
    DisGeNETdiseasesids = []

    DisGeNETRNAs = []
    DisGeNETids = []

    DisGeNETpmids = []
    DisGeNETdatabases = []
    # #读取文件
    with open('../input/relationship/DisGeNET_20230406/Disease_gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row

        alldiseases = []
        alldisids = []
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                id,disease = row
                alldisids.append(id)
                alldiseases.append(disease)

        print(1)
        allgeneNames = []
        allgeneids = []
        with open('../output/relationship/IV_step_similarity/gene_id.csv', newline='', encoding='utf-8') as csvfileg:
            readerg = csv.reader(csvfileg, delimiter=',', quotechar='"')
            for row in readerg:
                gene,id= row
                allgeneids.append(id)
                allgeneNames.append(gene)
        print(1)
        for row in reader:
            gene,disease,database,pmid = row
            diseaseName = disease.lower()
            geneName = gene.lower()
            if diseaseName in alldiseases and geneName in allgeneNames:
                DisGeNETdiseasesids.append(alldisids[alldiseases.index(diseaseName)])
                DisGeNETdiseases.append(diseaseName)
                DisGeNETids.append(allgeneids[allgeneNames.index(geneName)])
                DisGeNETRNAs.append(geneName)
                DisGeNETpmids.append(pmid)
                DisGeNETdatabases.append("DisGeNET")
    print(1)


    gene2dis = []
    for i in range(len(DisGeNETids)):
        pari = []
        pari.append(DisGeNETids[i])
        pari.append(DisGeNETdiseasesids[i])
        gene2dis.append(pari)

    np.save("../output/relationship/V_step_relationship/gene_disease.npy",gene2dis)

    path_df1 = pd.DataFrame()
    path_df1['geneid'] = DisGeNETids
    path_df1['gene'] = DisGeNETRNAs
    path_df1['diseaseid'] = DisGeNETdiseasesids
    path_df1['disease'] = DisGeNETdiseases
    path_df1['database'] = DisGeNETdatabases
    path_df1['pmid'] = DisGeNETpmids
    path_df1.to_csv("../output/relationship/V_step_relationship/dis2gene_allind.csv", index=False, header=False)
    path_df = pd.DataFrame()
    path_df['miRNAid'] = DisGeNETids
    # path_df1['gene'] = DisGeNETRNAs
    path_df['diseaseid'] = DisGeNETdiseasesids
    # path_df1['disease'] = DisGeNETdiseases
    # path_df1['database'] = DisGeNETdatabases
    # path_df1['pmid'] = DisGeNETpmids
    path_df.to_csv("../output/relationship/V_step_relationship/dis2gene_id.csv", index=False, header=False)
