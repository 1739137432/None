import pandas as pd

mirnas = pd.read_csv("deal.csv")
# print(mirnas)
# print(mirnas["mirna"])
mirna_name = mirnas["mirna"].tolist()
print(mirna_name.index("q"))
# mirnas = mirnas.drop(mirna_name)

# for name in mirnas["mirna"]:
#
#     print(name)
    # mirnas["mi"]