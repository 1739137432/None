
import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(500 * 1024 * 1024)
def dis2circ():
    # #读取文件
    circRNAs = []
    circRNAids =[]
    diseases = []
    diseaseids = []
    databases =[]
    pmids =[]


    # gene_sequence =[]
    with open('../input/relationship/Circ2Disease_20230406/Circ2Disease_Association.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alldiseases = []
        disids = []
        print(1)
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                print(row)
                id, disease = row
                disids.append(id)
                alldiseases.append(disease)
        print(2)
        cirRNANames = []
        circids = []
        with open('../output/relationship/IV_step_similarity/circRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                circrna,id = row
                circids.append(id)
                cirRNANames.append(circrna)
        print(3)
        for row in reader:
            gene,alias,C3,C4,C5,C6,C7,gene_sequence,C9,disease,C11,C12,C13,up_down,C15,C16,C17,C18,C19,pmid,C21,C22,C23,C24 = row
            disease = disease.lower()
            gene = gene.lower()
            if gene in cirRNANames:
                diseaseids.append(disids[alldiseases.index(disease)])
                diseases.append(disease)
                circRNAids.append(circids[cirRNANames.index(gene)])
                circRNAs.append(gene)
                pmids.append(pmid)
                databases.append("Circ2Disease")
    print(4)
    circ2dis = []
    for i in range(len(circRNAids)):
        pari = []
        pari.append(circRNAids[i])
        pari.append(diseaseids[i])
        circ2dis.append(pari)
    # 检查保存路径中的目录是否存在，如果不存在则创建
    if not os.path.exists('../output/relationship/V_step_relationship'):
        os.makedirs('../output/relationship/V_step_relationship')
    np.save("../output/relationship/V_step_relationship/circRNA_disease.npy",circ2dis)

    path_df = pd.DataFrame()
    path_df['circRNAid'] = circRNAids
    path_df['circRNAName'] = circRNAs
    path_df['diseaseid'] = diseaseids
    path_df['disease'] = diseases
    path_df['databases'] = databases
    path_df['pmids'] = pmids
    path_df.to_csv("../output/relationship/V_step_relationship/dis2circ_allinf.csv", index=False, header=False)
    path_df1 = pd.DataFrame()
    path_df1['circRNAid'] = circRNAids
    # path_df['circRNAName'] = circRNAs
    path_df1['diseaseid'] = diseaseids
    # path_df['disease'] = diseases
    # path_df['databases'] = databases
    # path_df['pmids'] = pmids
    path_df1.to_csv("../output/relationship/V_step_relationship/dis2circ_id.csv", index=False, header=False)

