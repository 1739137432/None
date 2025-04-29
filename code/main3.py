# from inititalize
from I_step_sumSeq.sumSequence import sumSequence
import II_step_ncRNA_disease_id.miRNA_id as ncRNAs
import III_step_idaddseq.addseq as addseq
from V_step_relationship import sum_relation
from IV_step_similarity import nRNA_similarity,ini_relationship_deal,mesh_deal,disease_similarity,re_out
from VI_step_data_划分 import data_划分,metapath,datadeal
# from VI_step_划分1 import d_c_m_c_d,d_c_m_m_c_d,d_g_m_g_d,d_g_m_m_g_d,d_l_m_l_d,d_l_m_m_l_d,d_m_c_c_m_d,d_m_c_m_d,d_m_g_g_m_d,d_m_g_m_d,d_m_l_l_m_d,d_m_l_m_d,g_d_c_c_d_g,g_d_c_d_g
# from VI_step_划分1 import g_d_l_d_g,g_d_l_l_d_g,g_d_m_d_g,g_d_m_m_d_g,g_m_d_d_m_g,g_m_d_m_g,l_d_g_d_l,l_d_g_g_d_l,m_c_d_c_m,m_c_d_d_c_m,m_g_d_d_g_m,m_g_d_g_m,m_l_d_d_l_m,m_l_d_l_m
from VII_step_train_val_test import train_val_test
from VI_step_data_划分2 import graph_metapath2
# from run import main
print("收集全部的基因序列。。。。。")
#sumSequence()
print("收集全部的miRNA。。。。。")
#ncRNAs.miRNAs()
print("收集全部的circRNA。。。。。")
#ncRNAs.circRNAs()
print("收集全部的lncRNA。。。。。")
#ncRNAs.lncRNAs()
print("收集全部的gene。。。。。")
#ncRNAs.genes()
print("匹配基因与基因序列。。。。。")
#addseq.miRNA_add_seq()
#addseq.lncRNA_add_seq()
#addseq.gene_add_seq()
print("计算circRNA的相似性。。。。。")
#nRNA_similarity.miRNA_similarity()
#nRNA_similarity.circRNA_similarity()
#nRNA_similarity.lncRNA_similarity()
#nRNA_similarity.gene_similarity()

#ini_relationship_deal.ini_relationship()
#mesh_deal.deal_mesh()
#re_out.re_out()
#disease_similarity.diease_similarity()

print("提取关系中。。。。")
#sum_relation.mi2circ()
#sum_relation.mi2lnc()
#sum_relation.mi2gene()
#sum_relation.dis2mi()
#sum_relation.dis2circ()
#sum_relation.dis2lnc()
#sum_relation.dis2gene()

print("数据划分中。。。")
#data_划分.data_划分()
# #
# # print("创建元路径中。。。。")
# # metapath.metapath()
#datadeal.data_deal()

graph_metapath2.create_graph_metapath(10,10)

# train_val_test.train_val_test()
# from run import main
# print("train.............")
# main("../output/relationship/VI_step_data_划分3/")
