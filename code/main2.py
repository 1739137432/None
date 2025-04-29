# # from inititalize
# from I_step_sumSeq.sumSequence import sumSequence
# from run import main
# print("收集全部的基因序列。。。。。")
# sumSequence()
# print("收集全部的miRNA。。。。。")
# import II_step_ncRNA_disease_id.miRNA_id as ncRNAs
# ncRNAs.miRNAs()
# print("收集全部的circRNA。。。。。")
# ncRNAs.circRNAs()
# print("收集全部的lncRNA。。。。。")
# ncRNAs.lncRNAs()
# print("收集全部的gene。。。。。")
# ncRNAs.genes()
# print("匹配基因与基因序列。。。。。")
# import III_step_idaddseq.addseq as addseq
# addseq.miRNA_add_seq()
# addseq.lncRNA_add_seq()
# addseq.gene_add_seq()
# print("计算circRNA的相似性。。。。。")
#
# from IV_step_similarity import nRNA_similarity
# #
# # nRNA_similarity.miRNA_similarity()
# nRNA_similarity.circRNA_similarity()
# nRNA_similarity.lncRNA_similarity()
#
# nRNA_similarity.gene_similarity()
#
# from IV_step_similarity import re_out
# re_out.re_out()
# from IV_step_similarity import ini_relationship_deal
# ini_relationship_deal.ini_relationship()
# from IV_step_similarity import mesh_deal
# mesh_deal.deal_mesh()
# from IV_step_similarity import disease_similarity
# disease_similarity.diease_similarity()
#
# from V_step_relationship import sum_relation
# print("提取关系中。。。。")
# sum_relation.mi2circ()
# sum_relation.mi2lnc()
# sum_relation.mi2gene()
# sum_relation.dis2mi()
# sum_relation.dis2circ()
# sum_relation.dis2lnc()
# sum_relation.dis2gene()
#
#
#
# # print("创建元路径中。。。。")
# # # metapath.metapath()
# from VI_step_data_划分 import data_划分
# #
# #
# data_划分.data_划分()
# from VI_step_data_划分 import datadeal
# datadeal.data_deal()
# from VI_step_data_划分 import tessss
# number = 10000
# repeat = 1000
# tessss.birmetapath_m(number,repeat)
# tessss.birmetapath_c(number,repeat)
# tessss.birmetapath_l(number,repeat)
# tessss.birmetapath_g(number,repeat)
tessss.birmetapath_d(number,repeat)

# print("数据划分中。。。")
# from VI_step_data_划分 import 提取最终关系
# 提取最终关系.ti()

from VII_step_train_val_test import train_val_test,data_划分
data_划分.data_划分()
train_val_test.train_val_test()

from train import main
print("train.............")
main()
