# from inititalize
from I_step_sumSeq.sumSequence import sumSequence
import II_step_ncRNA_disease_id.miRNA_id as ncRNAs
import III_step_idaddseq.addseq as addseq
from V_step_relationship import sum_relation
from IV_step_similarity import nRNA_similarity,ini_relationship_deal,mesh_deal,disease_similarity,re_out
from VI_step_data_division import data_division,metapath,datadeal
from VII_step_train_val_test import train_val_test
print("收集全部的基因序列。。。。。")
sumSequence()
ncRNAs.miRNAs()
ncRNAs.circRNAs()
ncRNAs.lncRNAs()
ncRNAs.genes()
addseq.miRNA_add_seq()
addseq.lncRNA_add_seq()
addseq.gene_add_seq()

nRNA_similarity.miRNA_similarity()
nRNA_similarity.circRNA_similarity()
nRNA_similarity.lncRNA_similarity()
nRNA_similarity.gene_similarity()

ini_relationship_deal.ini_relationship()
mesh_deal.deal_mesh()
re_out.re_out()
disease_similarity.diease_similarity()
sum_relation.mi2circ()
sum_relation.mi2lnc()
sum_relation.mi2gene()
sum_relation.dis2mi()
sum_relation.dis2circ()
sum_relation.dis2lnc()
sum_relation.dis2gene()
data_division.data_division()
metapath.metapath()
print("train_val_test")
train_val_test.train_val_test()

#