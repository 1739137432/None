import pandas as pd

num_disease = pd.read_csv('../output/relationship/IV_step_similarity/disease_adj_name.csv',sep=':').shape[0]
print(num_disease)