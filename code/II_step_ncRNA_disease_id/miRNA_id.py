import csv
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(500 * 1024 * 1024)


def miRNAs():
    miRNAs = []
    miRNAids = []
    mid = 0
    with open('../input/relationship/HMDD_202307/alldata_v43.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            miRNAid,miRNAName,diseaseid,disease,database,pmid = row

            miRNAName = miRNAName
            if miRNAName not in miRNAs:
                miRNAs.append(miRNAName)
                miRNAids.append(mid)
                mid += 1
    with open('../input/relationship/miRNEt_20230406/mir2dis.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            miRNA = ID
            if miRNA not in miRNAs:
                miRNAs.append(miRNA)
                miRNAids.append(mid)
                mid += 1
    with open('../input/relationship/miRNEt_20230406/mir2circ.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            miRNA = ID
            if miRNA not in miRNAs:
                miRNAs.append(miRNA)
                miRNAids.append(mid)
                mid += 1

    with open('../input/relationship/miRNEt_20230406/mir2lnc.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            miRNA = ID
            if miRNA not in miRNAs:
                miRNAs.append(miRNA)
                miRNAids.append(mid)
                mid += 1

    with open('../input/relationship/miRNEt_20230406/mir2gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID,Accession,Target,TargetID,Experiment,Literature,Tissue = row
            miRNA = ID
            if miRNA not in miRNAs:
                miRNAs.append(miRNA)
                miRNAids.append(mid)
                mid += 1

    with open('../input/relationship/mirtarbase/hsa_MTI.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            miRTarBaseID,miRNA,Species,TargetGene,TargetGene1,Species1,Experiments,SupportType,References = row
            miRNA = miRNA
            if miRNA not in miRNAs:
                miRNAs.append(miRNA)
                miRNAids.append(mid)
                mid += 1
    with open('../input/relationship/StarBase/circRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            add = False
            attributes = line.strip().split()  # Split by whitespace
            attributes[1] = attributes[1]
            if attributes[1] not in miRNAs:
                miRNAs.append(attributes[1])
                miRNAids.append(mid)
                mid += 1

    with open('../input/relationship/StarBase/lncRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            attributes = line.strip().split()  # Split by whitespace
            attributes[1] = attributes[1]
            if attributes[1] not in miRNAs:
                miRNAs.append(attributes[1])
                miRNAids.append(mid)
                mid += 1
    with open('../input/relationship/Homo_sapiens_TarBase-v9/Homo_sapiens.tsv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='	', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            species, mirna_name, mirna_id, gene_name, gene_id, gene_location, transcript_name, transcript_id, chromosome, start, end, strand, experimental_method, regulation, tissue, cell_line, article_pubmed_id, confidence, interaction_group, cell_type, microt_score, comment = row
            if mirna_name not in miRNAs:
                miRNAs.append(mirna_name)
                miRNAids.append(mid)
                mid += 1
    # 检查保存路径中的目录是否存在，如果不存在则创建
    if not os.path.exists('../output/relationship/II_step_ncRNA_disease_id'):
        os.makedirs('../output/relationship/II_step_ncRNA_disease_id')
    path_df = pd.DataFrame()
    path_df['id'] = miRNAids
    path_df['miRNA'] = miRNAs
    path_df.to_csv("../output/relationship/II_step_ncRNA_disease_id/miRNA_id.csv", index=False, header=True)


def lncRNAs():
    lncRNAs = []
    lncRNAids = []
    lid = 0

    with open('../input/relationship/LncRNADisease_v2.0/all ncRNA-disease information.csv', newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ncRNA_Symbol, ncRNA_Category, Species, Disease_Name, Sample, Dysfunction_Pattern, Validated_MethodPrediction_Method, Description, PubMed_ID = row
            disease = Disease_Name.lower()
            lncRNA = ncRNA_Symbol.lower()
            if lncRNA not in lncRNAs:
                lncRNAs.append(lncRNA)
                lncRNAids.append(lid)
                lid += 1
    with open('../input/relationship/miRNEt_20230406/mir2lnc.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
            lncRNA = Target.lower()
            miRNA = ID.lower()
            if lncRNA not in lncRNAs:
                lncRNAs.append(lncRNA)
                lncRNAids.append(lid)
                lid += 1
    with open('../input/relationship/StarBase/lncRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            attributes = line.strip().split()  # Split by whitespace
            attributes[3] = attributes[3].lower()
            if attributes[3] not in lncRNAs:
                lncRNAs.append(attributes[3])
                lncRNAids.append(lid)
                lid += 1
    path_df2 = pd.DataFrame()
    path_df2['id'] = lncRNAids
    path_df2['lncRNA'] = lncRNAs
    path_df2.to_csv("../output/relationship/II_step_ncRNA_disease_id/lncRNA_id.csv", index=False, header=True)


def genes():
    genes = []
    geneids = []
    gid = 0

    with open('../input/relationship/DisGeNET_20230406/Disease_gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            gene, disease, database, pmid = row
            # disease = disease.lower()
            gene = gene.lower()
            if gene not in genes:
                genes.append(gene)
                geneids.append(gid)
                gid += 1

    with open('../input/relationship/miRNEt_20230406/mir2gene.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
            gene = Target.lower()
            # miRNA = ID.lower()
            if gene not in genes:
                genes.append(gene)
                geneids.append(gid)
                gid += 1

    with open('../input/relationship/mirtarbase/hsa_MTI.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            miRTarBaseID, miRNA, Species, TargetGene, TargetGene1, Species1, Experiments, SupportType, References = row
            gene = TargetGene.lower()
            miRNA = miRNA.lower()
            if gene not in genes:
                genes.append(gene)
                geneids.append(gid)
                gid += 1
    with open('../input/relationship/Homo_sapiens_TarBase-v9/Homo_sapiens.tsv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='	', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            species, mirna_name, mirna_id, gene_name, gene_id, gene_location, transcript_name, transcript_id, chromosome, start, end, strand, experimental_method, regulation, tissue, cell_line, article_pubmed_id, confidence, interaction_group, cell_type, microt_score, comment = row
            if gene_name not in genes:
                genes.append(gene_name)
                geneids.append(gid)
                gid += 1
    path_df3 = pd.DataFrame()
    path_df3['id'] = geneids
    path_df3['gene'] = genes
    path_df3.to_csv("../output/relationship/II_step_ncRNA_disease_id/gene_id.csv", index=False, header=True)


def circRNAs():
    circRNAs = []
    circRNAids = []
    circRNAseqs = []

    # #读取文件
    with open('../input/relationship/miRNEt_20230406/mir2circ.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row

        allcircRNAids = []
        allcircRNANames = []
        allcircRNAseqs = []
        with open('../output/relationship/I_step_sumSeq/circrna_seq.csv', newline='', encoding='utf-8') as csvfiled:
            readerd = csv.reader(csvfiled, delimiter=',', quotechar='"')
            for row in readerd:
                id, circRNA, circRNA_seq = row
                allcircRNAids.append(id)
                allcircRNANames.append(circRNA)
                allcircRNAseqs.append(circRNA_seq)
            # print(allcircRNANames)
        print(1)

        for row in reader:
            ID, Accession, Target, TargetID, Experiment, Literature, Tissue = row
            Target = Target.lower()
            if Target not in circRNAs and Target in allcircRNANames:
                circRNAids.append(allcircRNAids[allcircRNANames.index(Target)])
                circRNAs.append(Target)
                circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(Target)])

        print(len(circRNAids))
        with open('../input/relationship/mirtarbase/hsa_MTI.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                miRTarBaseID, miRNA, Species, TargetGene, TargetGene1, Species1, Experiments, SupportType, References = row
                TargetGene = TargetGene.lower()
                if TargetGene not in circRNAs and TargetGene in allcircRNANames:
                    circRNAids.append(allcircRNAids[allcircRNANames.index(TargetGene)])
                    circRNAs.append(TargetGene)
                    circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(TargetGene)])
        print(len(circRNAids))
        with open('../input/relationship/Circ2Disease_20230406/The circRNA-disease entries.csv', newline='',
                  encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header row
            for row in reader:
                CRD_ID,circRNA_Name,Synonyms,Gene_Symbol,Disease_Name,Expression_pattern,PubMed_ID,Region,Strand,Species,Experimental_techniques,Brief_description,Title = row
                gene = circRNA_Name.lower()
                Synonyms = Synonyms.lower()
                Gene_Symbol = Gene_Symbol.lower()
                if gene not in circRNAs and gene in allcircRNANames:
                    circRNAids.append(allcircRNAids[allcircRNANames.index(gene)])
                    circRNAs.append(gene)
                    circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(gene)])
                if Synonyms != "N/A" and Synonyms not in circRNAs and Synonyms in allcircRNANames:
                    circRNAids.append(allcircRNAids[allcircRNANames.index(Synonyms)])
                    circRNAs.append(Synonyms)
                    circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(Synonyms)])
                if Gene_Symbol != "N/A" and Gene_Symbol not in circRNAs and Gene_Symbol in allcircRNANames:
                    circRNAids.append(allcircRNAids[allcircRNANames.index(Gene_Symbol)])
                    circRNAs.append(Gene_Symbol)
                    circRNAseqs.append(allcircRNAseqs[allcircRNANames.index(Gene_Symbol)])

        print(len(circRNAids))

        path_df1 = pd.DataFrame()
        path_df1['circRNAid'] = circRNAids
        path_df1['circRNA'] = circRNAs
        path_df1['circRNA_seq'] = circRNAseqs
        path_df1.to_csv("../output/relationship/II_step_ncRNA_disease_id/circRNA_id_seq_notsum.csv", index=False,
                        header=True)

        cid = 0
        w_circrnaids = []
        w_circrnas = []
        w_circrna_seqs = []
        from collections import Counter

        def get_duplicate_positions(list):
            counter = Counter(list)
            duplicates = [item for item, count in counter.items() if count >= 1]
            positions = {item: [i for i, x in enumerate(list) if x == item] for item in duplicates}
            return positions

        duplicate_positions = get_duplicate_positions(circRNAids)

        for item, positions in duplicate_positions.items():
            for i in positions:
                w_circrnaids.append(cid)
                w_circrnas.append(circRNAs[i])
                w_circrna_seqs.append(circRNAseqs[i])
            cid += 1

        path_df = pd.DataFrame()

        path_df['circRNAid'] = w_circrnaids
        path_df['circRNA'] = w_circrnas
        path_df['circRNA_seq'] = w_circrna_seqs
        path_df.to_csv("../output/relationship/II_step_ncRNA_disease_id/circRNA_id.csv", index=False, header=True)

        # path_df1 = pd.DataFrame()
        # path_df1['miRNAid'] = miRNAids
        # path_df1['circRNAid'] = circRNAids
        # path_df1.to_csv("../output/relationship/V_step_relationship/mir2circ_id.csv", index=False, header=True)