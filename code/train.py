from __future__ import division, print_function
import time
import random
import torch.nn.functional as F
from sklearn import metrics
from model import GCN
from data_loader import load_data
from param import parameter_parser
from utils import *
from torch_geometric.data import Data
from pathlib import Path
import os
import numpy as np

project_root = Path(__file__).parent.parent
data_dir = project_root  # 数据目录绝对路径
result_dir = os.path.join(project_root, "result")
os.makedirs(result_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parameter_parser()

# Set a fixed random seed to ensure reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# Data Loading
Adj, Dis_data, Meta_data, ori_adj, Dis_adj, Meta_adj, feature = load_data()
print(ori_adj.shape)
# Cross-validation Preparation
association_nam = ori_adj.shape[1]
random_index = ori_adj.T.tolist()
random.shuffle(random_index)
k_folds = 5
CV_size = int(association_nam / k_folds)
temp = (
    np.array(random_index[: association_nam - association_nam % k_folds])
    .reshape(k_folds, CV_size, -1)
    .tolist()
)
temp[k_folds - 1] += random_index[association_nam - association_nam % k_folds :]
random_index = temp
for i, fold in enumerate(random_index):
    print(f"Fold {i+1}: Number of positive samples = {len(fold)}")

# Result
acc_result, pre_result, recall_result, f1_result, auc_result, prc_result = (
    [],
    [],
    [],
    [],
    [],
    [],
)
fprs, tprs, recalls, precisions = [], [], [], []

print(f"seed={args.seed}, evaluating metabolite-disease....")

for k in range(k_folds):
    print(f"------ Fold {k + 1}/{k_folds} ------")

    Or_train = np.matrix(Adj, copy=True)
    val_pos_edge_index = torch.tensor(np.array(random_index[k]).T, dtype=torch.long).to(
        device
    )

    # Negative Sampling
    val_neg_edge_index = np.asmatrix(np.where(Or_train < 1)).T.tolist()
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = torch.tensor(
        np.array(val_neg_edge_index[: val_pos_edge_index.shape[1]]).T, dtype=torch.long
    ).to(device)

    # Remove the validation set border
    Or_train[tuple(np.array(random_index[k]).T)] = 0
    train_pos_edge_index = torch.tensor(
        np.asmatrix(np.where(Or_train > 0)), dtype=torch.long
    ).to(device)
    Or_train_matrix = np.matrix(Adj, copy=True)
    Or_train_matrix[tuple(np.array(random_index[k]).T)] = 0
    # Remove the edges of the validation set, and the remaining edges form a two-dimensional matrix to construct a heterogeneous network
    or_adj = constructNet2(
        torch.tensor(Or_train_matrix),
        torch.tensor(Dis_data.Dis_simi.values, dtype=torch.float32),
        torch.tensor(Meta_data.Meta_simi.values, dtype=torch.float32),
    ).to(
        device
    )  # Remove the edges of the validation set, and the remaining edges form a two-dimensional matrix to construct a heterogeneous network
    Dis_network = torch.nonzero(
        Dis_adj, as_tuple=True
    )  # Return the coordinates of the non-zero elements in the adjacency matrix (i.e., the pairs of nodes that have an edge between them)
    Dis_network = torch.stack(Dis_network)
    train_dis_data = Data(x=Dis_data.x, edge_index=Dis_network).to(device)
    
    Meta_network = torch.nonzero(Meta_adj, as_tuple=True)
    Meta_network = torch.stack(Meta_network)
    train_meta_data = Data(x=Meta_data.x, edge_index=Meta_network).to(device)

    # Build the Model
    model = GCN(
        train_meta_data.x.shape[1],
        args.hidden_dim,
        args.out_dim,
        args.gcn_layers,
        265,  # Dataset 1 is 265, and Dataset 2 is 126.
    ).to(device)

    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )

    criterion = F.binary_cross_entropy
    best_metrics = {
        "epoch": 0,
        "acc": 0,
        "auc": 0,
        "prc": 0,
        "fpr": None,
        "tpr": None,
        "recall": None,
        "precision": None,
        "full_metrics": None,
    }

    # Training Process
    for epoch in range(args.epochs):
        start = time.time()
        start_train = time.time()  # Return the current time
        model.train()
        optimizer.zero_grad()

        # Negative Sampling in Training Set
        train_neg_edge_index = np.asmatrix(np.where(Or_train_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = torch.tensor(
            np.array(train_neg_edge_index[: train_pos_edge_index.shape[1]]).T,
            dtype=torch.long,
        ).to(device)

        output = model(train_dis_data, train_meta_data, feature, adj=or_adj)
        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1)
        train_scores = output[edge_index[0], edge_index[1]].to(device)
        train_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(
            device
        )

        loss_train = criterion(train_scores, train_labels).to(device)
        loss_train.backward(retain_graph=True)
        optimizer.step()
        end_train = time.time() # Return the current time
        # Evaluation
        model.eval()
        with torch.no_grad():
            start_val = time.time()  # Return the current time
            train_auc = metrics.roc_auc_score(train_labels.cpu(), train_scores.cpu())
            predict_y_proba = output.reshape(Adj.shape[0], Adj.shape[1]).to(device)
            score_val, label_val, metric_tmp = cv_model_evaluate(
                predict_y_proba, val_pos_edge_index, val_neg_edge_index
            )

            fpr, tpr, _ = metrics.roc_curve(label_val, score_val)
            precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
            val_auc = metrics.auc(fpr, tpr)
            val_prc = metrics.auc(recall, precision)
            end_val = time.time()  # Return the current time
            end = time.time()

            print(
                f"Epoch {epoch + 1} | Loss: {loss_train.item():.4f} | Acc: {metric_tmp[0]:.4f} | "
                f"Pre: {metric_tmp[1]:.4f} | Rec: {metric_tmp[2]:.4f} | F1: {metric_tmp[3]:.4f} | "
                f"Train AUC: {train_auc:.4f} | AUC: {val_auc:.4f} | PRC: {val_prc:.4f} "
                "Time: %.2f" % (end - start),
                "Time_train: %.2f" % (end_train - start_train),
                "Time_val: %.2f" % (end_val - start_val),
            )

            if (
                metric_tmp[0] > best_metrics["acc"]
                and val_auc > best_metrics["auc"]
                and val_prc > best_metrics["prc"]
            ):
                best_metrics.update(
                    {
                        "epoch": epoch + 1,
                        "acc": metric_tmp[0],
                        "auc": val_auc,
                        "prc": val_prc,
                        "fpr": fpr,
                        "tpr": tpr,
                        "recall": recall,
                        "precision": precision,
                        "full_metrics": metric_tmp,
                    }
                )

    # Output the current optimal result
    print(
        f"Fold {k + 1} Best @Epoch {best_metrics['epoch']} | "
        f"Acc: {best_metrics['acc']:.4f} | AUC: {best_metrics['auc']:.4f} | PRC: {best_metrics['prc']:.4f}"
    )

    acc_result.append(best_metrics["acc"])
    pre_result.append(best_metrics["full_metrics"][1])
    recall_result.append(best_metrics["full_metrics"][2])
    f1_result.append(best_metrics["full_metrics"][3])
    auc_result.append(round(best_metrics["auc"], 4))
    prc_result.append(round(best_metrics["prc"], 4))
    fprs.append(best_metrics["fpr"])
    tprs.append(best_metrics["tpr"])
    recalls.append(best_metrics["recall"])
    precisions.append(best_metrics["precision"])

# Summary Results
print("\n## Training Finished !")
print("Acc", acc_result)
print("Pre", pre_result)
print("Recall", recall_result)
print("F1", f1_result)
print("AUC", auc_result)
print("PRC", prc_result)

print(
    f"AUC mean: {np.mean(auc_result):.4f}, std: {np.std(auc_result):.4f}\n"
    f"Acc mean: {np.mean(acc_result):.4f}, std: {np.std(acc_result):.4f}\n"
    f"Pre mean: {np.mean(pre_result):.4f}, std: {np.std(pre_result):.4f}\n"
    f"Recall mean: {np.mean(recall_result):.4f}, std: {np.std(recall_result):.4f}\n"
    f"F1 mean: {np.mean(f1_result):.4f}, std: {np.std(f1_result):.4f}\n"
    f"PRC mean: {np.mean(prc_result):.4f}, std: {np.std(prc_result):.4f}"
)

# tprs_path = os.path.join(result_dir, "tprs.csv")
# pd.DataFrame(tprs).to_csv(tprs_path, index=False)


# fprs_path = os.path.join(result_dir, "fprs.csv")
# pd.DataFrame(fprs).to_csv(fprs_path, index=False)


# recalls_path = os.path.join(result_dir, "recalls.csv")
# pd.DataFrame(recalls).to_csv(recalls_path, index=False)


# precisions_path = os.path.join(result_dir, "precisions.csv")
# pd.DataFrame(precisions).to_csv(precisions_path, index=False)

#
plot_auc_curves(fprs, tprs, auc_result, directory=data_dir, name="auc")
plot_prc_curves(precisions, recalls, prc_result, directory=data_dir, name="prc")
