import csv

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(500 * 1024 * 1024)
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



def mi2lnc():
    pmids =[]
    miRMAs =[]
    miRMAids = []
    genes =[]
    geneids =[]
    database =[]

    allgeneNames = []
    allgeneids = []
    with open('../output/relationship/IV_step_similarity/gene_id.csv', newline='', encoding='utf-8') as csvfiled:
        readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
        for row in readerd:
            gene, id = row
            allgeneids.append(id)
            allgeneNames.append(gene)
    print(1)
    allmiRNANames = []
    allmiRNAIds = []
    with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
        readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
        for row in readerc:
            miRNA, id = row
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
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row


        print(1)
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            split_list = Literature.split('|')
            for pmid in split_list:
                ID = ID
                Target = Target.lower()
                if ID in allmiRNANames and Target in allgeneNames:
                    if ID in shortName:
                        ID = longName[shortName.index(ID)]
                    pmids.append(pmid)
                    miRMAs.append(ID)
                    miRMAids.append(allmiRNAIds[allmiRNANames.index(ID)])
                    genes.append(Target)
                    geneids.append(allgeneids[allgeneNames.index(Target)])
                    database.append("miRNEt")
        with open('../input/relationship/mirtarbase/hsa_MTI.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                miRTarBaseID, ID, Species, Target, TargetGene1, Species1, Experiments, SupportType, pmid = row
                split_list = pmid.split("|")
                for pmid in split_list:
                    ID = ID
                    Target = Target.lower()
                    if ID in allmiRNANames and Target in allgeneNames:
                        if ID in shortName:
                            ID = longName[shortName.index(ID)]
                        pmids.append(pmid)
                        miRMAs.append(ID)
                        miRMAids.append(allmiRNAIds[allmiRNANames.index(ID)])
                        genes.append(Target)
                        geneids.append(allgeneids[allgeneNames.index(Target)])
                        database.append("mirtarbase")
        with open('../input/relationship/Homo_sapiens_TarBase-v9/Homo_sapiens.tsv', newline='',
                  encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='	', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                species, ID, mirna_id, Target, gene_id, gene_location, transcript_name, transcript_id, chromosome, start, end, strand, experimental_method, regulation, tissue, cell_line, pmid, confidence, interaction_group, cell_type, microt_score, comment = row
                split_list = pmid.split("|")
                for pmid in split_list:
                    ID = ID
                    Target = Target.lower()
                    if ID in allmiRNANames and Target in allgeneNames:
                        if ID in shortName:
                            ID = longName[shortName.index(ID)]
                        pmids.append(pmid)
                        miRMAs.append(ID)
                        miRMAids.append(allmiRNAIds[allmiRNANames.index(ID)])
                        genes.append(Target)
                        geneids.append(allgeneids[allgeneNames.index(Target)])
                        database.append("TarBase")
                    print("TarBase")
        print(1)
        mi2gene = []
        for i in range(len(miRMAids)):
            pari = []
            pari.append(miRMAids[i])
            pari.append(geneids[i])
            mi2gene.append(pari)

        np.save("../output/relationship/V_step_relationship/miRNA_gene.npy", mi2gene)
        path_df = pd.DataFrame()
        path_df['miRNAid'] = miRMAids
        path_df['miRMA'] = miRMAs
        path_df['geneid'] = geneids
        path_df['gene'] = genes
        path_df['pmid'] = pmids
        path_df['database'] = database
        path_df.to_csv("../output/relationship/V_step_relationship/mir2gene_allinf.csv", index=False, header=False)
        path_df1 = pd.DataFrame()
        path_df1['miRNAid'] = miRMAids
        # path_df['miRMA'] = miRMAs
        path_df1['geneid'] = geneids
        # path_df['gene'] = genes
        # path_df['pmid'] = pmids
        # path_df['database'] = database
        path_df1.to_csv("../output/relationship/V_step_relationship/mir2gene_id.csv", index=False, header=False)

