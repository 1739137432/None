import csv
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
csv.field_size_limit(500 * 1024 * 1024)


def sumSequence():
    #==============================circrna=================================
    circrnaids = []
    circrnas = []
    circrna_seqs = []
    cid = 0
    with open('../input/sequence/Circ2Disease/Circ2Disease_Association.csv', newline='', encoding='utf-8') as csvfile1:
        reader1 = csv.reader(csvfile1, delimiter=',', quotechar='"')
        next(reader1)  # Skip header row
        for row in reader1:
            gene,alias,C3,C4,C5,C6,C7,gene_sequence,C9,disease,C11,C12,C13,up_down,C15,C16,C17,C18,C19,pmid,C21,C22,C23,C24 = row
            status = False
            if gene_sequence == "N/A":
                continue
            gene = gene.lower()
            if gene not in circrnas:
                circrnas.append(gene)
                circrna_seqs.append(gene_sequence)
                circrnaids.append(cid)
                status = True
            if(alias == "N/A"):
                continue
            else:
                alias = alias.split('; ')
                for alia in alias:
                    if alia not in circrnas:
                        circrnas.append(alia)
                        circrna_seqs.append(gene_sequence)
                        circrnaids.append(cid)
                        status = True
            if status:
                cid+=1
    print(len(circrnas))
    # miRNAid	miRNAname	geneID	geneName	circID	geneType	chromosome	start	end	strand	clipExpNum	degraExpNum	RBP	merClass	14miRseq	align	16targetSeq
    with open('../input/sequence/StarBase/circRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            status = False
            attributes = line.strip().split('	')  # Split by whitespace
            circRNA = attributes[2].lower()
            circRNAsequence = attributes[16].upper().replace('-', '')
            if circRNA not in circrnas:
                circrnas.append(circRNA)
                circrna_seqs.append(circRNAsequence)
                circrnaids.append(cid)
                status = True
            circRNA = attributes[3].lower()
            if circRNA not in circrnas:
                circrnas.append(circRNA)
                circrna_seqs.append(circRNAsequence)
                circrnaids.append(cid)
                status = True
            atts = attributes[4].split(",")  # Split by whitespace
            for att in atts:
                circRNA = att.lower()
                if circRNA not in circrnas:
                    circrnas.append(circRNA)
                    circrna_seqs.append(circRNAsequence)
                    circrnaids.append(cid)
                    status = True
            if status:
                cid+=1
    with open('../input/sequence/circBase/hg19_circbase_seq.csv', newline='', encoding='utf-8') as csvfile2:
        reader2 = csv.reader(csvfile2, delimiter=',', quotechar='"')
        next(reader2)  # Skip header row
        for row in reader2:
            Header,Sequence = row
            Header = Header.lower()
            if Header not in circrnas:
                circrnas.append(Header)
                circrna_seqs.append(Sequence)
                circrnaids.append(cid)
                cid+=1
    with open('../input/sequence/circBank/circBaseSequence.csv', newline='', encoding='utf-8') as csvfile3:
        reader3 = csv.reader(csvfile3, delimiter=',', quotechar='"')
        next(reader3)  # Skip header row
        for row in reader3:
            name,sequence = row
            name = name.lower()
            if name not in circrnas:
                circrnas.append(name)
                circrna_seqs.append(sequence)
                circrnaids.append(cid)
                cid += 1
    with open('../input/sequence/rnacentral/circRNA_list_name.csv', newline='', encoding='utf-8') as csvfile3:
        reader3 = csv.reader(csvfile3, delimiter=',', quotechar='"')
        next(reader3)  # Skip header row
        for row in reader3:
            id,species_info,circName,Sequence = row
            circName = circName.lower()
            if circName not in circrnas:
                circrnas.append(circName)
                circrna_seqs.append(Sequence)
                circrnaids.append(cid)
                cid += 1
    #==============================mirna=================================



    mirnaids = []
    mirnas = []
    mirna_seqs = []
    mid = 0
    with open('../input/sequence/miRBase/miRNA.csv', newline='', encoding='utf-8') as csvfile4:
        reader4= csv.reader(csvfile4, delimiter=',', quotechar='"')
        next(reader4)  # Skip header row
        for row in reader4:
            Name,ID,Sequence = row
            # Name = Name.lower()
            if Name not in mirnas:
                mirnas.append(Name)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid+=1
    with open('../input/sequence/miRBase/miRBaseSequence.csv', newline='', encoding='utf-8') as csvfile5:
        reader5 = csv.reader(csvfile5, delimiter=',', quotechar='"')
        next(reader5)  # Skip header row
        for row in reader5:
            Name,Sequence = row
            # Name = Name.lower()
            if Name not in mirnas:
                mirnas.append(Name)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid += 1
    with open('../input/sequence/rnacentral/miRNA_list_name.csv', newline='', encoding='utf-8') as csvfile5:
        reader5 = csv.reader(csvfile5, delimiter=',', quotechar='"')
        next(reader5)  # Skip header row
        for row in reader5:
            id,species_info,Name,Sequence = row
            # Name = Name.lower()
            if Name not in mirnas:
                mirnas.append(Name)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid += 1
    # miRNAid	miRNAname	geneID	geneName	circID	geneType	chromosome	start	end	strand	clipExpNum	degraExpNum	RBP	merClass	14miRseq	align	16targetSeq
    with open('../input/sequence/StarBase/circRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            add = False
            attributes = line.strip().split('	')  # Split by whitespace
            miRNA = attributes[1]
            miRNAsequence = attributes[14].upper().replace('-', '')

            if miRNA not in mirnas:
                mirnas.append(miRNA)
                mirna_seqs.append(miRNAsequence)
                mirnaids.append(mid)
                mid += 1
    # miRNAid	miRNAname	geneID	geneName	circID	geneType	chromosome	start	end	strand	clipExpNum	degraExpNum	RBP	merClass	14miRseq	align	16targetSeq
    with open('../input/sequence/StarBase/lncRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            add = False
            attributes = line.strip().split('	')  # Split by whitespace
            miRNA = attributes[1]
            miRNAsequence = attributes[13].upper().replace('-', '')

            if miRNA not in mirnas:
                mirnas.append(miRNA)
                mirna_seqs.append(miRNAsequence)
                mirnaids.append(mid)
                mid += 1
    with open('../input/sequence/mirna_lncrna_interaction.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header row
        for row in reader6:
            id,lncrna,lncrna_seq,mirna,mirna_seq = row
            mirna = mirna
            if mirna not in mirnas:
                mirnas.append(mirna)
                mirna_seqs.append(mirna_seq)
                mirnaids.append(mid)
                mid += 1
    with open('../input/relationship/miRNA/miRNA.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header row
        for row in reader6:
            Accession,Symbol,Sequence,Accession1,Symbol1,Sequence1 = row
            mirna = Symbol
            if mirna not in mirnas:
                mirnas.append(mirna)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid += 1
            mirna = Symbol1
            if mirna not in mirnas:
                mirnas.append(mirna)
                mirna_seqs.append(Sequence1)
                mirnaids.append(mid)
                mid += 1
    with open('../input/relationship/miRNA/notpre_miRNA.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header row
        for row in reader6:
            Accession,Symbol,Sequence = row
            mirna = Symbol
            if mirna not in mirnas:
                mirnas.append(mirna)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid += 1
    with open('../input/relationship/miRNA/notmi_miRNA.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header row
        for row in reader6:
            Accession,Symbol,Sequence = row
            mirna = Symbol
            if mirna not in mirnas:
                mirnas.append(mirna)
                mirna_seqs.append(Sequence)
                mirnaids.append(mid)
                mid += 1
    # #==============================lncrna=================================
    lncrnas = []
    lncrna_seqs = []
    lncrnaids = []
    lid = 0
    with open('../input/sequence/mirna_lncrna_interaction.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header rowa
        for row in reader6:
            id,lncrna,lncrna_seq,mirna,mirna_seq = row
            lncrna = lncrna.lower()
            mirna = mirna.lower()
            if lncrna not in lncrnas:
                lncrnas.append(lncrna)
                lncrna_seqs.append(lncrna_seq)
                lncrnaids.append(lid)
                lid+=1
    # miRNAid	miRNAname	geneID	geneName	circID	geneType	chro6mosome	start	end	strand	cl10ipExpNum	degraExpNum	RBP	merClass	14miRseq	align	16targetSeq
    with open('../input/sequence/StarBase/lncRNA_miRNA_interaction.txt', 'r') as txt_file:
        for line in txt_file:
            add = False
            attributes = line.strip().split('	')  # Split by whitespace
            lncRNAsequence = attributes[15].upper().replace('-', '')
            lncRNA = attributes[3].lower()
            if lncRNA not in lncrnas:
                lncrnas.append(lncRNA)
                lncrna_seqs.append(lncRNAsequence)
                lncrnaids.append(lid)
                lid += 1

    genes = []
    gene_seqs = []
    geneids = []
    gid = 0

    with open('../input/sequence/gene_seq.csv', newline='', encoding='utf-8') as csvfile6:
        reader6 = csv.reader(csvfile6, delimiter=',', quotechar='"')
        next(reader6)  # Skip header rowa
        for row in reader6:
            gene,gene_seq = row
            gene = gene.lower()
            if gene not in genes:
                genes.append(gene)
                gene_seqs.append(gene_seq)
                geneids.append(gid)
                gid+=1

    # 检查保存路径中的目录是否存在，如果不存在则创建
    if not os.path.exists('../output/relationship/I_step_sumSeq'):
        os.makedirs('../output/relationship/I_step_sumSeq')


    path_df = pd.DataFrame()
    path_df['id'] = lncrnaids
    path_df['lncrna'] = lncrnas
    path_df['lncrna_seq'] = lncrna_seqs
    path_df.to_csv("../output/relationship/I_step_sumSeq/lncrna_seq.csv", index=False, header=True)

    path_df1 = pd.DataFrame()
    path_df1['id'] = mirnaids
    path_df1['mirna'] = mirnas
    path_df1['mirna_seq'] = mirna_seqs
    path_df1.to_csv("../output/relationship/I_step_sumSeq/mirna_seq.csv", index=False, header=True)

    path_df2 = pd.DataFrame()
    path_df2['id'] = circrnaids
    path_df2['circrna'] = circrnas
    path_df2['circrna_seq'] = circrna_seqs
    path_df2.to_csv("../output/relationship/I_step_sumSeq/circrna_seq.csv", index=False, header=True)

    path_df3 = pd.DataFrame()
    path_df3['id'] = geneids
    path_df3['gene'] = genes
    path_df3['gene_seq'] = gene_seqs
    path_df3.to_csv("../output/relationship/I_step_sumSeq/gene_seq.csv", index=False, header=True)