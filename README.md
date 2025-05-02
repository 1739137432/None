# Biologically active RNA and disease association prediction based upon a meta-path and heterogeneous graph model
This repository is the implementation of our Paper under review.

# Abstract
This study proposed a novel model based on meta-path and heterogeneous graph to identify potential associations in a complex network comprising five types of nodes: miRNA, circRNA, lncRNA, protein-coding gene, and disease. First, we constructed a heterogeneous graph using collected association data and initialized the feature vectors of the nodes. Next, we employed hybrid pooling to aggregate feature information from nodes along meta-path instances and extracted neighbor node information to update the feature vectors of the starting nodes. Then, we used a graph attention network to compute the importance of different meta-paths for association relationships. By merging the updated node embeddings with initial vectors via residual connections, we generated final node embeddings. Finally, we obtained the feature vectors of the five node types (miRNA, circRNA, lncRNA, protein-coding gene, and disease) and predicted association probabilities using tensor operations. Our model predicted seven types of associations, achieved AUC and AP scores of 0.99, and demonstrated validation performance in Alzheimer's disease.

# plot_data
This directory mainly stores the data used for drawing in the paper
* AD results.xlsx : Biologically active RNA association associated with Alzheimer's disease (including our predicted outcomes)


## Requirement
- pandas~=1.5.3
- numpy~=1.23.5
- scipy~=1.11.4
- torch~=2.0.0
- dgl~=1.1.2+cu118
- scikit-learn~=1.4.0
- gensim~=4.3.2
- matplotlib~=3.8.0
- networkx~=3.1

## Datasets
### Relationship databases
- Circ2Disease(http://bioinformatics.zju.edu.cn/Circ2Disease/download.html)
- DisGeNET(http://www.disgenet.org/)
- HMDD(http://www.cuilab.cn/hmdd)
- LncRNADisease(http://www.rnanut.net/lncrnadisease/index.php/home/info/download)
- StarBase(https://rnasysu.com/encori/)
- miRNET(https://www.mirnet.ca/)
- TarBase(https://dianalab.e-ce.uth.gr/tarbasev9/downloads)
- miRTarBase(https://mirtarbase.cuhk.edu.cn/)
* Please download relationship databases and place those datasets in the folder "data/input/relationship/<Database name>/".
### Sequence databses
- miRBase(https://www.mirbase.org/)
- RNAcentral(https://rnacentral.org/downloads)
- circBase(https://www.circbase.org/cgi-bin/downloads.cgi)
- Circ2Disease(http://bioinformatics.zju.edu.cn/Circ2Disease/download.html)
- Ensembl(https://www.ensembl.org/info/data/index.html)
* Please download sequence databases and place those datasets in the folder "data/input/sequence/<Database name>/".
### Disease data
- MeSH(http://www.ncbi.nlm.nih.gov/)
* Please download disease data and place those datasets in the folder "data/Annotation/".

## Data file
### Output files
#### VI_step_data_division
- adjM.npz:Training set adjacency matrix
- XX-XX_id.csv:XX-XX association relationship indices
- XX-XX.npy:XX-XX association relationship
- XX-XX_train_val_test_idx.npz:XX-XX association relationship training, testing, and validation set position indices
#### VII_step_train_val_test
- XX-XX_train_val_test_pos.npz:Training, testing, and validation dataset for XX-XX association pos relationships
- XX-XX_train_val_test_neg.npz:Training, testing, and validation dataset for XX-XX association neg relationships
- XX-XX_train_val_test_idx.npz:XX-XX association relationship training, testing, and validation set position indices

### Code files
#### checkpoint
Save the training model file
#### model
- PABDMH_lp.py:Model file
#### utils
- data.py:Training data loading file
- pytorchtools.py:Early stop file
- tools.py:Model tool function file




## Prediction
* Step 1: In the "data/output/relationship/V_step_similarity/XXX_id CSV" file to find the appropriate disease and biological macromolecules id (please keep each relationship is more than 10).
* Step 2: Write the ids of each type of node in the form of a list to the corresponding positions in the "utils/structure_relation.py" file where the demo is located. If there is no requirement for a certain type, the data of that type does not need to be modified, and then run the file.

* Step 3 : run "run.py" file
