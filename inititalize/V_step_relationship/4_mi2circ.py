import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(500 * 1024 * 1024)
def mi2circ():
    pmids =[]
    miRNAids =[]
    miRNAs =[]
    circRNAs =[]
    circRNAids =[]
    circRNAseqs =[]
    database =[]
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2circ.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row

        allcircRNAids = []
        allcircRNANames = []
        allcircRNAseqs = []
        with open('../output/relationship/IV_step_similarity/circRNA_id.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
            for row in readerd:
                circRNA ,id= row
                allcircRNAids.append(id)
                allcircRNANames.append(circRNA)
        print(1)
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA ,id= row
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
            ID = ID.lower()
            Target = Target
            if ID in allmiRNANames and Target in allcircRNANames:
                if ID in shortName:
                    ID = longName[shortName.index(ID)]
                pmids.append(Literature)
                miRNAids.append(allmiRNAIds[allmiRNANames.index(ID)])
                miRNAs.append(ID)
                circRNAids.append(allcircRNAids[allcircRNANames.index(Target)])
                circRNAs.append(Target)
                # circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(Target)])
                database.append("miRNEt")
        print(1)
        with open('../input/relationship/mirtarbase/hsa_MTI.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                miRTarBaseID, miRNA, Species, TargetGene, TargetGene1, Species1, Experiments, SupportType, References = row
                miRNA = miRNA
                TargetGene = TargetGene.lower()
                if miRNA in allmiRNANames and TargetGene in allcircRNANames:
                    if miRNA in shortName:
                        miRNA = longName[shortName.index(miRNA)]
                    pmids.append(References)
                    miRNAids.append(allmiRNAIds[allmiRNANames.index(miRNA)])
                    miRNAs.append(miRNA)
                    circRNAids.append(allcircRNAids[allcircRNANames.index(TargetGene)])
                    circRNAs.append(TargetGene)
                    # circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(TargetGene)])
                    database.append("mirtarbase")
        print(1)

        #
        mi2circ = []
        for i in range(len(miRNAids)):
            pari = []
            pari.append(miRNAids[i])
            pari.append(circRNAids[i])
            mi2circ.append(pari)

        np.save("../output/relationship/V_step_relationship/miRNA_circRNA.npy", mi2circ)

        path_df = pd.DataFrame()
        path_df['miRNAid'] = miRNAids
        path_df['miRNA'] = miRNAs
        path_df['circRNAid'] = circRNAids
        path_df['circRNA'] = circRNAs
        path_df['pmid'] = pmids
        path_df['database'] = database
        path_df.to_csv("../output/relationship/V_step_relationship/mir2circ_allinf.csv", index=False, header=False)

        path_df1 = pd.DataFrame()
        path_df1['miRNAid'] = miRNAids
        # path_df['miRNA'] = miRNAs
        path_df1['circRNAid'] = circRNAids
        # path_df['circRNA'] = circRNAs
        # path_df['pmid'] = pmids
        # path_df['database'] = database
        path_df1.to_csv("../output/relationship/V_step_relationship/mir2circ_id.csv", index=False, header=False)
    # print(1)
    # inFile = open("../output/relationship/2_step_relationship/mir2circ.csv", 'r')  #
    #
    # outFile = open("../output/relationship/2_step_relationship/mir2circ_merge.csv", 'w')  # 最后保存的.csv文件
    # print(1)
    # listLines = []
    # for line in inFile:
    #     if line in listLines:
    #         continue
    #     else:
    #         outFile.write(line)
    #         listLines.append(line)
    # outFile.close()
    # inFile.close()