# Predicting the association between biological molecules and disease based on meta-path and heterogeneous graph model

## Requestions
- pandas~=1.5.3
- numpy~=1.23.5
- scipy~=1.11.4
- torch~=2.0.0
- dgl~=1.1.2+cu118
- scikit-learn~=1.4.0
- gensim~=4.3.2
- matplotlib~=3.8.0
- networkx~=3.1

## Data
### Relationship databases(path: data/relationship/)
- Circ2Disease(http://bioinformatics.zju.edu.cn/Circ2Disease/download.html)
- DisGeNET(http://www.disgenet.org/)
- HMDD(http://www.cuilab.cn/hmdd)
- LncRNADisease(http://www.rnanut.net/lncrnadisease/index.php/home/info/download)
- StarBase(https://rnasysu.com/encori/)
- miRNET(https://www.mirnet.ca/)
- TarBase(https://dianalab.e-ce.uth.gr/tarbasev9/downloads)
- miRTarBase(https://mirtarbase.cuhk.edu.cn/)

### Sequence databses(path: data/requence/)
- miRBase(https://www.mirbase.org/)
- RNAcentral(https://rnacentral.org/downloads)
- circBase(https://www.circbase.org/cgi-bin/downloads.cgi)
- Circ2Disease(http://bioinformatics.zju.edu.cn/Circ2Disease/download.html)
- Ensembl(https://www.ensembl.org/info/data/index.html)

### Disease data(path: data/Annotation/)
- MeSH(http://www.ncbi.nlm.nih.gov/)


## Data file
### Output files (path: /data/output/)
#### VI_step_data_division
- adjM.npz:Training set adjacency matrix
- XX-XX_id.csv:XX-XX association relationship indices
- XX-XX.npy:XX-XX association relationship
- XX-XX_train_val_test_idx.npz:XX-XX association relationship training, testing, and validation set position indices
#### VII_step_train_val_test
- XX-XX_train_val_test_pos.npz:Training, testing, and validation dataset for XX-XX association pos relationships
- XX-XX_train_val_test_neg.npz:Training, testing, and validation dataset for XX-XX association neg relationships
- XX-XX_train_val_test_idx.npz:XX-XX association relationship training, testing, and validation set position indices
### Code files (path: /code/inititalize/)
#### I_step_sumSeq
Collect sequence data
#### II_step_ncRNA_disease_id
Collect node information
#### III_step_idaddseq
Match the collected nodes with the sequence
#### IV_step_similarity
Calculate the similarity between nodes of the same type
#### V_step_relationship
Collect the final association relationship
#### VI_step_data_division
Divide the data set and extract the meta-path data
#### VII_step_train_val_test
Extract the training set, test set and validation set
#### checkpoint
Save the training model file
#### model
- PABDMH_lp.py:Model file
#### utils
- data.py:Training data loading file
- pytorchtools.py:Early stop file
- tools.py:Model tool function file

## Train model
step 1 : mian.py

step 2 : train.py

## Prediction
step 1 : VII_step_train_val_test/train_val_test_1.py

step 2 : run.py