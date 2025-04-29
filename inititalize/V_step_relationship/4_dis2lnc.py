
import csv

import numpy as np
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# csv.field_size_limit(50000 * 1024 * 1024)
def dis2lnc():
    # #读取文件
    lncRNAs = []
    lncRNAids =[]
    diseases = []
    diseaseids = []
    databases =[]
    pmids =[]
    # gene_sequence =[]
    with open('../input/relationship/LncRNADisease_v2.0/all ncRNA-disease information.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alldiseases = []
        alldisids = []
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                id, disease = row
                alldisids.append(id)
                alldiseases.append(disease)
        print(1)
        alllncRNANames = []
        alllncRNAIds = []
        with open('../output/relationship/IV_step_similarity/lncRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                lncRNA,id= row
                alllncRNAIds.append(id)
                alllncRNANames.append(lncRNA)
        print(1)
        for row in reader:
            ncRNA_Symbol,ncRNA_Category,Species,Disease_Name,Sample,Dysfunction_Pattern,Validated_MethodPrediction_Method,Description,PubMed_ID = row
            Disease_Name = Disease_Name.lower()
            ncRNA_Symbol = ncRNA_Symbol.lower()
            if Disease_Name in alldiseases and ncRNA_Symbol in alllncRNANames:
                diseaseids.append(alldisids[alldiseases.index(Disease_Name)])
                diseases.append(Disease_Name)
                lncRNAids.append(alllncRNAIds[alllncRNANames.index(ncRNA_Symbol)])
                lncRNAs.append(ncRNA_Symbol)
                pmids.append(PubMed_ID)
                databases.append("LncRNADisease")
    print(1)
    lnc2dis = []
    for i in range(len(lncRNAids)):
        pari = []
        pari.append(lncRNAids[i])
        pari.append(diseaseids[i])
        lnc2dis.append(pari)

    np.save("../output/relationship/V_step_relationship/lncRNA_disease.npy",lnc2dis)
    path_df = pd.DataFrame()
    path_df['lncRNAid'] = lncRNAids
    path_df['lncRNAName'] = lncRNAs
    path_df['diseaseid'] = diseaseids
    path_df['disease'] = diseases
    path_df['database'] = databases
    path_df['pmid'] = pmids
    path_df.to_csv("../output/relationship/V_step_relationship/dis2lnc_allinf.csv", index=False, header=False)

    path_df1 = pd.DataFrame()
    path_df1['lncRNAid'] = lncRNAids
    # path_df['lncRNAName'] = lncRNAs
    path_df1['diseaseid'] = diseaseids
    # path_df['disease'] = diseases
    # path_df['database'] = databases
    # path_df['pmid'] = pmids
    path_df1.to_csv("../output/relationship/V_step_relationship/dis2lnc_id.csv", index=False, header=False)