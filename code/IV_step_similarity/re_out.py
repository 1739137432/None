
import csv

import numpy as np
import pandas as pd
import os
def re_out():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    csv.field_size_limit(500 * 1024 * 1024)
    # #读取文件
    diseases = []
    # gene_sequence =[]
    with open('../input/relationship/Circ2Disease_20230406/The circRNA-disease entries.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        cirRNANames = []
        with open('../output/relationship/IV_step_similarity/circRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                circrna,id = row
                cirRNANames.append(circrna)
        print(3)
        for row in reader:
            CRD_ID, circRNA_Name, Synonyms, Gene_Symbol, Disease_Name, Expression_pattern, PubMed_ID, Region, Strand, Species, Experimental_techniques, Brief_description, Title = row
            gene = circRNA_Name.lower()
            Synonyms = Synonyms.lower()
            Gene_Symbol = Gene_Symbol.lower()
            disease = Disease_Name.lower()
            if gene in cirRNANames and disease not in diseases:
                diseases.append(disease)
            if Synonyms in cirRNANames and disease not in diseases:
                diseases.append(disease)
            if Gene_Symbol in cirRNANames and disease not in diseases:
                diseases.append(disease)

    with open('../input/relationship/DisGeNET_20230406/Disease_gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        allgeneNames = []
        with open('../output/relationship/IV_step_similarity/gene_id.csv', newline='', encoding='utf-8') as csvfileg:
            readerg = csv.reader(csvfileg, delimiter=',', quotechar='"')
            for row in readerg:
                gene,id= row
                allgeneNames.append(gene)
        print(1)
        for row in reader:
            gene,disease,database,pmid = row
            diseaseName = disease.lower()
            geneName = gene.lower()
            if diseaseName not in diseases and geneName in allgeneNames:
                diseases.append(disease)

    with open('../input/relationship/LncRNADisease_v2.0/all ncRNA-disease information.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        alllncRNANames = []
        with open('../output/relationship/IV_step_similarity/lncRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                lncRNA,id= row
                alllncRNANames.append(lncRNA)
        print(1)
        for row in reader:
            ncRNA_Symbol,ncRNA_Category,Species,Disease_Name,Sample,Dysfunction_Pattern,Validated_MethodPrediction_Method,Description,PubMed_ID = row
            Disease_Name = Disease_Name.lower()
            ncRNA_Symbol = ncRNA_Symbol.lower()
            if Disease_Name not in diseases and ncRNA_Symbol in alllncRNANames:
                diseases.append(Disease_Name)


    with open('../input/relationship/HMDD_202307/alldata_v43.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        allmiRNANames = []
        with open('../output/relationship/IV_step_similarity/miRNA_id.csv', newline='', encoding='utf-8') as csvfilec:
            readerc = csv.reader(csvfilec, delimiter=',', quotechar='"')
            for row in readerc:
                miRNA,id = row
                allmiRNANames.append(miRNA)
        print(1)
        for row in reader:
            miRNAid,miRNAName,diseaseid,disease,database,pmid = row
            disease = disease.lower()
            miRNA = miRNAName.lower()
            if disease not in diseases and miRNA in allmiRNANames:
                diseases.append(disease)
    path_df = pd.DataFrame()
    path_df['disease'] = diseases
    path_df.to_csv("diseaseName.csv", index=False, header=False)


