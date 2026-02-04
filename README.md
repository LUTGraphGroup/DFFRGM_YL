# DFFRGM_YL
DFFRGM: A Dual-view Fusion Framework using Residual gated Graph convolutional and Mamba for metabolite-disease relationships prediction

python 3.8.10
torch 2.0.0+cu11.8
torch-cluster 1.6.3+pt20cu118
torch-geometric 2.6.1
torch-scatter   2.1.2+pt20cu118
torch-sparse    0.6.18+pt20cu118
pandas 2.0.3
numpy 1.24.1

Data:
The data files needed to run the model, which contain data1 and data2.
 asssociation_matrix: Metabolite-Disease association matrix
 disease_simi_network: Comprehensive similarity matrix of diseases
 metabolite_simi_network:Comprehensive similarity matrix of metabolite

 
Code:
utils.py: Methods of data processing
param.py: The main hyperparameters of the model
data_loader：Load model data
train： run this to train model
