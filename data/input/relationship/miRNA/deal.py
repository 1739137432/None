# Short	Symbol	Accession	Family	Species	length	Sequence	CDNA
import pandas as pd

mirna_Symbol = []
mirna_Accession = []
mirna_Sequence = []
mirna_CDNA = []
with open('mirna.hsa.info.txt', 'r') as file:
    lines = file.readlines()[1:]
for line in lines:
    ss = line.split('	')
    mirna_Symbol.append(ss[1])
    mirna_Accession.append(ss[2])
    mirna_Sequence.append(ss[6])
    mirna_CDNA.append(ss[7])
# print(mirna_Symbol)
premirna_Symbol = []
premirna_Accession = []
premirna_Sequence = []
premirna_CDNA = []

with open('premirna.hsa.info.txt', 'r') as file:
    lines = file.readlines()[1:]
for line in lines:
    ss = line.split('	')
    premirna_Symbol.append(ss[1])
    premirna_Accession.append(ss[2])
    premirna_Sequence.append(ss[6])
    premirna_CDNA.append(ss[7])

pre_miRNA_NO = []
pre_miRNA_Name = []
pre_miRNA_seq = []
acc_miRNA_NO = []
acc_miRNA_Name = []
acc_miRNA_seq = []

with open('premirnaID2armsID.txt', 'r') as file:
    lines = file.readlines()[1:]
for line in lines:
    ss = line.split('	')
    if ss[0] in premirna_Accession and ss[1] in mirna_Accession:
        pre_miRNA_NO.append(ss[0])
        pre_miRNA_Name.append(premirna_Symbol[premirna_Accession.index(ss[0])])
        pre_miRNA_seq.append(premirna_Sequence[premirna_Accession.index(ss[0])])
        acc_miRNA_NO.append(ss[1])
        acc_miRNA_Name.append(ss[2])
        acc_miRNA_seq.append(mirna_Sequence[mirna_Accession.index(ss[1])])

    if ss[0] in premirna_Accession and ss[3] in mirna_Accession:
        pre_miRNA_NO.append(ss[0])
        pre_miRNA_Name.append(premirna_Symbol[premirna_Accession.index(ss[0])])
        pre_miRNA_seq.append(premirna_Sequence[premirna_Accession.index(ss[0])])
        acc_miRNA_NO.append(ss[3])
        acc_miRNA_Name.append(ss[4])
        acc_miRNA_seq.append(mirna_Sequence[mirna_Accession.index(ss[3])])

notpre_NO = []
notpre_Name = []
notpre_seq = []
for m in mirna_Accession:
    if m not in acc_miRNA_NO:
        notpre_NO.append(m)
        notpre_Name.append(mirna_Symbol[mirna_Accession.index(m)])
        notpre_seq.append(mirna_Sequence[mirna_Accession.index(m)])

notmi_NO = []
notmi_Name = []
notmi_seq = []
for m in premirna_Accession:
    if m not in pre_miRNA_NO:
        notmi_NO.append(m)
        notmi_Name.append(premirna_Symbol[premirna_Accession.index(m)])
        notmi_seq.append(premirna_Sequence[premirna_Accession.index(m)])


for pr_miRNA in notpre_Name:
    miRNA = pr_miRNA.lower()
    if miRNA in notmi_Name:
        acc_miRNA_NO.append(notpre_NO[notpre_Name.index(pr_miRNA)])
        acc_miRNA_Name.append(pr_miRNA)
        acc_miRNA_seq.append(notpre_seq[notpre_Name.index(pr_miRNA)])
        pre_miRNA_NO.append(notmi_NO[notmi_Name.index(miRNA)])
        pre_miRNA_Name.append(miRNA)
        pre_miRNA_seq.append(notmi_seq[notmi_Name.index(miRNA)])

        # notpre_NO.remove(notpre_NO[notpre_Name.index(pr_miRNA)])
        #
        # notpre_seq.remove(notpre_seq[notpre_Name.index(pr_miRNA)])
        # notpre_Name.remove(pr_miRNA)
        # notmi_NO.remove(notmi_NO[notmi_Name.index(miRNA)])
        #
        # notmi_seq.remove(notmi_seq[notmi_Name.index(miRNA)])
        # notmi_Name.remove(miRNA)

path_df = pd.DataFrame()
path_df['Accession'] = pre_miRNA_NO
path_df['Symbol'] = pre_miRNA_Name
path_df['Sequence'] = pre_miRNA_seq
path_df['Accession1'] = acc_miRNA_NO
path_df['Symbol1'] = acc_miRNA_Name
path_df['Sequence1'] = acc_miRNA_seq
path_df.to_csv("miRNA.csv", index=False, header=True)

path_df1 = pd.DataFrame()
path_df1['Accession'] = notpre_NO
path_df1['Symbol'] = notpre_Name
path_df1['Sequence'] = notpre_seq
path_df1.to_csv("notpre_miRNA.csv", index=False, header=True)

path_df2 = pd.DataFrame()
path_df2['Accession'] = notmi_NO
path_df2['Symbol'] = notmi_Name
path_df2['Sequence'] = notmi_seq
path_df2.to_csv("notmi_miRNA.csv", index=False, header=True)