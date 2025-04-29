import csv
import os

import numpy as np
import pandas as pd


def dis2circ():
    # #读取文件
    circRNAs = []
    circRNAids = []
    diseases = []
    diseaseids = []
    databases = []
    pmids = []

    # gene_sequence =[]
    with open('../input/relationship/Circ2Disease_20230406/The circRNA-disease entries.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alldiseases = []
        disids = []
        print(1)
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                print(row)
                id, disease = row
                disids.append(id)
                alldiseases.append(disease)
        print(2)
        cirRNANames = []
        circids = []
        with open('../output/relationship/IV_step_similarity/circRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                circrna, id = row
                circids.append(id)
                cirRNANames.append(circrna)
        print(3)
        for row in reader:
            CRD_ID, circRNA_Name, Synonyms, Gene_Symbol, Disease_Name, Expression_pattern, PubMed_ID, Region, Strand, Species, Experimental_techniques, Brief_description, Title = row
            gene = circRNA_Name.lower()
            Synonyms = Synonyms.lower()
            Gene_Symbol = Gene_Symbol.lower()
            disease = Disease_Name.lower()
            if gene in cirRNANames:
                diseaseids.append(disids[alldiseases.index(disease)])
                diseases.append(disease)
                circRNAids.append(circids[cirRNANames.index(gene)])
                circRNAs.append(gene)
                pmids.append(PubMed_ID)
                databases.append("Circ2Disease")
            if Synonyms in cirRNANames:
                diseaseids.append(disids[alldiseases.index(disease)])
                diseases.append(disease)
                circRNAids.append(circids[cirRNANames.index(Synonyms)])
                circRNAs.append(Synonyms)
                pmids.append(PubMed_ID)
                databases.append("Circ2Disease")
            if Gene_Symbol in cirRNANames:
                diseaseids.append(disids[alldiseases.index(disease)])
                diseases.append(disease)
                circRNAids.append(circids[cirRNANames.index(Gene_Symbol)])
                circRNAs.append(Gene_Symbol)
                pmids.append(PubMed_ID)
                databases.append("Circ2Disease")
    print(4)
    circ2dis = []
    for i in range(len(circRNAids)):
        pari = []
        pari.append(circRNAids[i])
        pari.append(diseaseids[i])
        circ2dis.append(pari)

    np.save("../output/relationship/V_step_relationship/circRNA_disease.npy", circ2dis)

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
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                id, disease = row
                alldisids.append(id)
                alldiseases.append(disease)

        print(1)
        allgeneNames = []
        allgeneids = []
        with open('../output/relationship/IV_step_similarity/gene_id.csv', newline='', encoding='utf-8') as csvfileg:
            readerg = csv.reader(csvfileg, delimiter=',', quotechar='"')
            for row in readerg:
                gene, id = row
                allgeneids.append(id)
                allgeneNames.append(gene)
        print(1)
        for row in reader:
            gene, disease, database, pmid = row
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

    np.save("../output/relationship/V_step_relationship/gene_disease.npy", gene2dis)

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


def dis2lnc():
    # #读取文件
    lncRNAs = []
    lncRNAids = []
    diseases = []
    diseaseids = []
    databases = []
    pmids = []
    # gene_sequence =[]
    with open('../input/relationship/LncRNADisease_v2.0/all ncRNA-disease information.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alldiseases = []
        alldisids = []
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                id, disease = row
                alldisids.append(id)
                alldiseases.append(disease)
        print(1)
        alllncRNANames = []
        alllncRNAIds = []
        with open('../output/relationship/IV_step_similarity/lncRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                lncRNA, id = row
                alllncRNAIds.append(id)
                alllncRNANames.append(lncRNA)
        print(1)
        for row in reader:
            ncRNA_Symbol, ncRNA_Category, Species, Disease_Name, Sample, Dysfunction_Pattern, Validated_MethodPrediction_Method, Description, PubMed_ID = row
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

    np.save("../output/relationship/V_step_relationship/lncRNA_disease.npy", lnc2dis)
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


def dis2mi():
    # #读取文件
    miRNAs = []
    miRNAids = []
    diseases = []
    diseaseids = []
    databases = []
    pmids = []
    # gene_sequence =[]
    with open('../input/relationship/HMDD_202307/alldata_v43.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alldiseases = []
        alldisids = []
        with open('../output/relationship/IV_step_similarity/disease_adj_name.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=':', quotechar='"')
            for row in readerd:
                id, disease = row
                alldisids.append(id)
                alldiseases.append(disease)
        print(1)
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA, id = row
                allmiRNAIds.append(id)
                allmiRNANames.append(miRNA)
        longName = []
        shortName = []
        longSeq = []
        with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
            readerz = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(readerz)  # Skip header row
            for row in readerz:
                Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
                longName.append(Symbol)
                shortName.append(Symbol1)
                longSeq.append(Sequence)
        print(1)
        for row in reader:
            miRNAid, miRNAName, diseaseid, disease, database, pmid = row
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
    np.save("../output/relationship/V_step_relationship/miRNA_disease.npy", mi2dis)

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


def mi2circ():
    pmids = []
    miRNAids = []
    miRNAs = []
    circRNAs = []
    circRNAids = []
    circRNAseqs = []
    database = []
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2circ.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row

        allcircRNAids = []
        allcircRNANames = []
        allcircRNAseqs = []
        with open('../output/relationship/IV_step_similarity/circRNA_id.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
            for row in readerd:
                circRNA, id = row
                allcircRNAids.append(id)
                allcircRNANames.append(circRNA)
        print(1)
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA, id = row
                allmiRNAIds.append(id)
                allmiRNANames.append(miRNA)
        longName = []
        shortName = []
        longSeq = []
        with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
            readerz = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(readerz)  # Skip header row
            for row in readerz:
                Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
                longName.append(Symbol)
                shortName.append(Symbol1)
                longSeq.append(Sequence)
        print(1)
        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
            ID = ID
            Target = Target.lower()
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

        if not os.path.exists('../output/relationship/V_step_relationship'):
            os.makedirs('../output/relationship/V_step_relationship')


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


def mi2gene():
    pmids = []
    miRMAs = []
    miRMAids = []
    genes = []
    geneids = []
    database = []
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row

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
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA, id = row
                allmiRNAIds.append(id)
                allmiRNANames.append(miRNA)
        longName = []
        shortName = []
        longSeq = []
        with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
            readerz = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(readerz)  # Skip header row
            for row in readerz:
                Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
                longName.append(Symbol)
                shortName.append(Symbol1)
                longSeq.append(Sequence)
        print(1)
        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
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
        with open('../input/relationship/Homo_sapiens_TarBase-v9/Homo_sapiens.tsv', newline='',
                      encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='	', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                species, mirna_name, mirna_id, gene_name, gene_id, gene_location, transcript_name, transcript_id, chromosome, start, end, strand, experimental_method, regulation, tissue, cell_line, article_pubmed_id, confidence, interaction_group, cell_type, microt_score, comment = row
                if mirna_name in allmiRNANames and gene_name in allgeneNames:
                    if mirna_name in shortName:
                        mirna_name = longName[shortName.index(mirna_name)]
                    pmids.append(article_pubmed_id)
                    miRMAs.append(mirna_name)
                    miRMAids.append(allmiRNAIds[allmiRNANames.index(mirna_name)])
                    genes.append(gene_name)
                    geneids.append(allgeneids[allgeneNames.index(gene_name)])
                    database.append("tarbase")
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


def mi2lnc():
    pmids = []
    miRNAs = []
    miRNAids = []
    lncRNAs = []
    lncRNAids = []
    database = []
    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2lnc.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alllncRNANames = []
        alllncRNAids = []
        with open('../output/relationship/IV_step_similarity/lncRNA_id.csv', newline='',
                  encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
            for row in readerd:
                lncRNA, id = row
                alllncRNAids.append(id)
                alllncRNANames.append(lncRNA)
        print(1)
        allmiRNANames = []
        allmiRNAIds = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='',
                  encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA, id = row
                allmiRNAIds.append(id)
                allmiRNANames.append(miRNA)
        longName = []
        shortName = []
        longSeq = []
        with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
            readerz = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(readerz)  # Skip header row
            for row in readerz:
                Accession, Symbol, Sequence, Accession1, Symbol1, Sequence1 = row
                longName.append(Symbol)
                shortName.append(Symbol1)
                longSeq.append(Sequence)
        print(1)
        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
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