# gene_sequence =[]
import csv

import numpy as np
import pandas as pd
def deal_mesh():
    diseases = []
    UIs = []
    with open('../input/Annotation/mesh_20230723/mesh1.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)  # Skip header row
        for row in reader:
            DescriptorUI,DescriptorName,TreeNumber = row
            diseases.append(DescriptorName.lower())
            UIs.append(TreeNumber.lower())



    # print(diseases[UIs.index('c26.200.156')])
    print(len(diseases))
    diseases_diseases = []
    diseases_UIs = []

    with open('diseaseName.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            disease, = row
            # print(disease)
            if disease in diseases:
                diseases_diseases.append(disease)
                diseases_UIs.append(UIs[diseases.index(disease)])

    print(diseases_diseases)
    print(len(diseases_diseases))



    diseases = []
    UIs = []

    mesh = np.loadtxt("../input/Annotation/mesh_20230723/mesh.txt", dtype=str, delimiter=";")  # 读取源文件
    for mes in mesh:
        diseases.append(mes[1].lower())
        UIs.append(mes[2].lower())

    newdiseases =[]
    newdiseaseids =[]
    for TreeNumber in diseases_UIs:
        sss = TreeNumber.split('-')
        # print(sss)
        for ss in sss:
            s = ss.split('.')
            temp = s[0]
            if temp not in newdiseaseids:
                newdiseases.append(diseases[UIs.index(temp)])
                newdiseaseids.append(temp)
            for a in s[1:]:
                temp = temp+"." + a
                if temp not in newdiseaseids:
                    newdiseases.append(diseases[UIs.index(temp)])
                    newdiseaseids.append(temp)

    # path_df = pd.DataFrame()
    #
    # path_df['disease'] = newdiseases
    # path_df['id'] = newdiseaseids
    # path_df.to_csv("mesh_id.csv",sep=';',index=False, header=False)
    file = open('../input/Annotation/mesh_20230723/mesh_id.txt', 'w')
    for i in range(len(newdiseases)):
        data = newdiseases[i] + ";" + newdiseaseids[i] +'\n'
        file.write(data)

    file.close()