import csv

with open('dis2mi_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,database,pmid=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('dis2mi: ' +' database:' + dataset + ' miRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'disease: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')

with open('dis2circ_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,database,pmid=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('dis2circ: ' +' database:' + dataset + ' circRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'disease: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')

with open('dis2lnc_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,database,pmid=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('dis2lnc: ' +' database:' + dataset + ' lncRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'disease: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')

with open('dis2gene_allind.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,database,pmid=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('dis2gene: ' +' database:' + dataset + ' gene:' + str(len(datasets_ci[datasets.index(database)][0])) + 'disease: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')

with open('mir2circ_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,pmid,database=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('mir2circ: ' +' database:' + dataset + ' miRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'circRNA: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')

with open('mir2lnc_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,database,pmid=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('mir2lnc: ' +' database:' + dataset + ' miRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'lncRNA: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')


with open('mir2gene_allinf.csv', newline='',
          encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # next(reader)  # Skip header row
    datasets = []
    datasets_ci = []
    for row in reader:
        mi_id,mic_name,dis_id,dis_name,pmid,database=row
        if database not in datasets:
            datasets.append(database)
            datasets_ci.append([[mic_name],[dis_name]])
        else:
            if mic_name not in datasets_ci[datasets.index(database)][0]:
                datasets_ci[datasets.index(database)][0].append(mic_name)
            if dis_name not in datasets_ci[datasets.index(database)][1]:
                datasets_ci[datasets.index(database)][1].append(dis_name)
    with open('nums.txt', 'a+') as f:
        for dataset in datasets:
            f.write('mir2gene: ' +' database:' + dataset + ' miRNA:' + str(len(datasets_ci[datasets.index(database)][0])) + 'gene: ' +str(len(datasets_ci[datasets.index(database)][1])) + '\n')




