import csv

import pandas as pd

gene = []
with open('Homo_sapiens.tsv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter='	', quotechar='"')
    next(reader)  # Skip header row
    for row in reader:
        specie,Tmirna_name,mirna_id,gene_name,gene_id,gene_location,transcript_name,transcript_id,chromosome,start,end,strand,experimental_method,regulation,tissue,cell_line,article_pubmed_id,confidence,interaction_group,cell_type,microt_score,comment = row
        if gene_id not in gene:
            gene.append(gene_id)

path_df = pd.DataFrame()
path_df['id'] = gene
path_df.to_csv("gene_id.csv", index=False, header=False)

