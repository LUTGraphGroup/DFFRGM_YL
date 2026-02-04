# DFFRGM: A Dual-view Fusion Framework Using Residual Gated Graph Convolutional and Mamba for Metabolite-Disease Relationship Prediction

## Environment Configuration

**Python**: 3.8.10  
**Deep Learning Frameworks**:
- PyTorch: 2.0.0+cu11.8
- torch-cluster: 1.6.3+pt20cu118
- torch-geometric: 2.6.1
- torch-scatter: 2.1.2+pt20cu118
- torch-sparse: 0.6.18+pt20cu118

**Data Processing Libraries**:
- pandas: 2.0.3
- numpy: 1.24.1

---

## Data Description

The data files required to run the model contain the following two datasets:

### **data1** and **data2**
Include the following key files:

1. **association_matrix**: Metabolite-Disease association matrix
2. **disease_simi_network**: Comprehensive similarity matrix of diseases  
3. **metabolite_simi_network**: Comprehensive similarity matrix of metabolites

---

## Code Structure

### Core Modules

| File | Description |
|------|------------|
| **utils.py** | Data processing methods |
| **param.py** | Main hyperparameters configuration for the model |
| **data_loader.py** | Model data loader |
| **train.py** | **Model training entry point** (Run this file to start training) |

---

## Usage Instructions

1. **Environment Setup**: Ensure installation of the specified dependency versions
2. **Data Preparation**: Place data files in the correct directory
3. **Model Training**: Run `train.py` to start the training process
4. **Parameter Adjustment**: Modify `param.py` to adjust model hyperparameters
