import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(50000 * 1024 * 1024)
# inFile = open('../input/relationship/miRNEt_20230406/mir2lnc.csv', 'r')  #
#
# outFile = open('../output/relationship/miRNEt_20230406/mir2lnc.csv', 'w')  # 最后保存的.csv文件
# listLines = []
#
# for line in inFile:
#
#     if line in listLines:
#         continue
#     else:
#         outFile.write(line)
#         listLines.append(line)
#
# outFile.close()
#
# inFile.close()



def mi2gene():
    pmids =[]
    miRNAs =[]
    miRNAids =[]
    lncRNAs =[]
    lncRNAids = []
    database =[]
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2lnc.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alllncRNANames = []
        alllncRNAids = []
        with open('../output/relationship/IV_step_similarity/lncRNA_id.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
            for row in readerd:
                lncRNA ,id= row
                alllncRNAids.append(id)
                alllncRNANames.append(lncRNA)
        print(1)
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/mirna_id.csv', newline='', encoding='utf-8') as csvfilec:
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
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            ID = ID
            Target = Target.lower()
            if ID in allmiRNANames and Target in alllncRNANames:
                if ID in shortName:
                    ID = longName[shortName.index(ID)]
                pmids.append(Literature)
                miRNAids.append(allmiRNAIds[allmiRNANames.index(ID)])
                miRNAs.append(ID)
                lncRNAs.append(Target)
                lncRNAids.append(alllncRNAids[alllncRNANames.index(Target)])
                database.append("miRNEt")
        print(1)
        mi2lnc = []
        for i in range(len(miRNAids)):
            pari = []
            pari.append(miRNAids[i])
            pari.append(lncRNAids[i])
            mi2lnc.append(pari)

        np.save("../output/relationship/V_step_relationship/miRNA_lncRNA.npy", mi2lnc)
        path_df = pd.DataFrame()
        path_df['miRNAid'] = miRNAids
        path_df['miRNA'] = miRNAs
        path_df['lncRNAid'] = lncRNAids
        path_df['lncRNA'] = lncRNAs
        path_df['pmid'] = pmids
        path_df['database'] = database
        path_df.to_csv("../output/relationship/V_step_relationship/mir2lnc_allinf.csv", index=False, header=False)

        path_df1 = pd.DataFrame()
        path_df1['miRNAid'] = miRNAids
        # path_df['miRMA'] = miRNAs
        path_df1['lncRNAid'] = lncRNAids
        # path_df['lncRNA'] = lncRNAs
        # path_df['pmid'] = pmids
        # path_df['database'] = database
        path_df1.to_csv("../output/relationship/V_step_relationship/mir2lnc_id.csv", index=False, header=False)