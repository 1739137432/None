
import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(50000 * 1024 * 1024)
def dis2mi():
    # #读取文件
    miRNAs = []
    miRNAids =[]
    diseases = []
    diseaseids = []
    databases =[]
    pmids =[]
    # gene_sequence =[]
    with open('../input/relationship/HMDD_202307/alldata_v43.csv', newline='', encoding='utf-8') as csvfile:
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
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA,id = row
                allmiRNAIds.append(id)
                allmiRNANames.append(miRNA)

        longName = []
        shortName = []
        # longSeq = []
        with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
                longName.append(Symbol)
                shortName.append(Symbol1)
                # longSeq.append(Sequence)
        print(1)
        for row in reader:
            miRNAid,miRNAName,diseaseid,disease,database,pmid = row
            disease = disease.lower()
            miRNA = miRNAName
            if disease in alldiseases and miRNA in allmiRNANames:
                if miRNA in shortName:
                    miRNA = longName[shortName.index(miRNA)]
                diseaseids.append(alldisids[alldiseases.index(disease)])
                diseases.append(disease)
                miRNAids.append(allmiRNAIds[allmiRNANames.index(miRNA)])
                miRNAs.append(miRNA)
                pmids.append(pmid)
                databases.append("HMDD")
        print(miRNAids)
    print(1)
    mi2dis = []
    for i in range(len(miRNAids)):
        pari = []
        pari.append(miRNAids[i])
        pari.append(diseaseids[i])
        mi2dis.append(pari)
    np.save("../output/relationship/V_step_relationship/miRNA_disease.npy",mi2dis)

    path_df = pd.DataFrame()
    path_df['miRNAid'] = miRNAids
    path_df['miRNAName'] = miRNAs
    path_df['diseaseid'] = diseaseids
    path_df['disease'] = diseases
    path_df['database'] = databases
    path_df['pmid'] = pmids
    path_df.to_csv("../output/relationship/V_step_relationship/dis2mi_allinf.csv", index=False, header=False)
    path_df1 = pd.DataFrame()
    path_df1['miRNAid'] = miRNAids
    # path_df['miRNAName'] = miRNAs
    path_df1['diseaseid'] = diseaseids
    # path_df['disease'] = diseases
    # path_df['database'] = databases
    # path_df['pmid'] = pmids
    path_df1.to_csv("../output/relationship/V_step_relationship/dis2mi_id.csv", index=False, header=False)