# gene_sequence =[]
import csv

import pandas as pd
def ini_relationship():
    circ_diseases = []
    circ_cirRNAs = []

    had_seqcirc = []
    with open('../output/relationship/II_step_ncRNA_disease_id/circRNA_id.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id,circrna,circrna_seq = row
            had_seqcirc.append(circrna)

    with open('../input/relationship/Circ2Disease_20230406/The circRNA-disease entries.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            CRD_ID, circRNA_Name, Synonyms, Gene_Symbol, Disease_Name, Expression_pattern, PubMed_ID, Region, Strand, Species, Experimental_techniques, Brief_description, Title = row
            gene = circRNA_Name.lower()
            Synonyms = Synonyms.lower()
            Gene_Symbol = Gene_Symbol.lower()
            disease = Disease_Name.lower()
            if gene in had_seqcirc:
                circ_diseases.append(disease)
                circ_cirRNAs.append(gene)
            if Synonyms != "N/A" and Synonyms in had_seqcirc:
                circ_diseases.append(disease)
                circ_cirRNAs.append(gene)
            if Gene_Symbol != "N/A" and Gene_Symbol in had_seqcirc:
                circ_diseases.append(disease)
                circ_cirRNAs.append(gene)


    path_df = pd.DataFrame()
    path_df['circrna'] = circ_cirRNAs
    path_df['disease'] = circ_diseases

    path_df.to_csv("../output/relationship/IV_step_similarity/dis2circ.csv", index=False, header=False)

    lnc_diseases = []
    lnc_lncRNAs = []

    had_seqlnc = []
    with open('../output/relationship/III_step_idaddseq/LncRNA_id_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id,lncrna,lncrna_seq = row
            had_seqlnc.append(lncrna)


    with open('../input/relationship/LncRNADisease_v2.0/all ncRNA-disease information.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ncRNA_Symbol, ncRNA_Category, Species, Disease_Name, Sample, Dysfunction_Pattern, Validated_MethodPrediction_Method, Description, PubMed_ID = row
            Disease_Name = Disease_Name.lower()
            ncRNA_Symbol = ncRNA_Symbol.lower()
            if ncRNA_Symbol in had_seqlnc:
                lnc_diseases.append(Disease_Name)
                lnc_lncRNAs.append(ncRNA_Symbol)
    path_df1 = pd.DataFrame()
    path_df1['lncrna'] = lnc_lncRNAs
    path_df1['disease'] = lnc_diseases
    path_df1.to_csv("../output/relationship/IV_step_similarity/dis2lnc.csv", index=False, header=False)


    mi_diseases = []
    mi_miRNAs = []

    had_seqmi = []
    with open('../output/relationship/III_step_idaddseq/miRNA_id_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id,mirna,mirna_seq = row
            had_seqmi.append(mirna)

    longName = []
    shortName = []
    longSeq = []
    with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            Accession,Symbol,Sequence,Accession1,Symbol1,Sequence1 = row
            longName.append(Symbol)
            shortName.append(Symbol1)
            longSeq.append(Sequence)

    with open('../input/relationship/HMDD_202307/alldata_v43.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            miRNAid,miRNAName,diseaseid,disease,database,pmid = row
            disease = disease.lower()
            miRNA = miRNAName
            if miRNA in had_seqmi:
                if miRNA in shortName:
                    miRNA = longName[shortName.index(miRNA)]
                mi_diseases.append(disease)
                mi_miRNAs.append(miRNA)


    with open('../input/relationship/miRNEt_20230406/mir2dis.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            disease = Target.lower()
            miRNA = ID
            mi_diseases.append(disease)
            mi_miRNAs.append(miRNA)


    path_df2 = pd.DataFrame()
    path_df2['mirna'] = mi_miRNAs
    path_df2['disease'] = mi_diseases

    path_df2.to_csv("../output/relationship/IV_step_similarity/dis2mi.csv", index=False, header=False)


    gene_diseases = []
    gene_genes = []



    had_seqgene = []
    with open('../output/relationship/III_step_idaddseq/gene_id_seq.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            id,gene,gene_seq = row
            had_seqgene.append(gene)
    with open('../input/relationship/DisGeNET_20230406/Disease_gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            gene,disease,database,pmid = row
            disease = disease.lower()
            geneName = gene.lower()
            if geneName in had_seqgene:
                gene_diseases.append(disease)
                gene_genes.append(geneName)

    path_df3 = pd.DataFrame()
    path_df3['gene'] = gene_genes
    path_df3['disease'] = gene_diseases

    path_df3.to_csv("../output/relationship/IV_step_similarity/dis2gene.csv", index=False, header=False)