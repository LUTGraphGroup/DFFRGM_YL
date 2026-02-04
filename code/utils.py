from sklearn.decomposition import PCA
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def constructNet(association_matrix):  # construct association matrix
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def constructNet2(association_matrix, drug_m, meta_n):  # construct association matrix

    mat1 = torch.cat((meta_n, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), drug_m), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def constructADJNet(meta_adj, dis_adj, adj):
    if isinstance(adj, list):
        adj = np.array(adj)
    if isinstance(meta_adj, list):
        meta_adj = np.array(meta_adj)
    if isinstance(dis_adj, list):
        dis_adj = np.array(dis_adj)
    mat1 = np.hstack((meta_adj, adj))
    mat2 = np.hstack((adj.T, dis_adj))
    return np.vstack((mat1, mat2))

    return features


def constructFEANet(dis_fea, meta_fea):

    if dis_fea.shape[1] != meta_fea.shape[1]:
        raise ValueError(
            "Feature dimensions do not match between disease and metabolite."
        )

    features = torch.cat([meta_fea, dis_fea], dim=0)  # 按行拼接 [M+N, D]
    return features  # 自动保持 device 和 requires_grad


def pca_reduce(embeddings, n_components=64):

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced


def constructDecoder(association_tensor):
    met_tensor = torch.zeros(association_tensor.shape[0], association_tensor.shape[0])
    dis_tensor = torch.zeros(association_tensor.shape[1], association_tensor.shape[1])
    tensor1 = torch.cat((met_tensor, association_tensor), 1)
    tensor2 = torch.cat((association_tensor.T, dis_tensor), 1)
    adj = torch.cat((tensor1, tensor2), 0)
    return adj


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1
    return link_labels


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)
    ]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1_score, 4),
    ]


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(
            fprs[i],
            tprs[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUC: %.4f" % (i + 1, auc[i]),
        )

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="BlueViolet",
        alpha=0.9,
        label="Mean AUC: %.4f $\pm$ %.4f" % (mean_auc, auc_std),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(
            recalls[i],
            precisions[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUPR: %.4f" % (i + 1, prc[i]),
        )

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(
        mean_recall,
        mean_precision,
        color="BlueViolet",
        alpha=0.9,
        label="Mean AUPR: %.4f $\pm$ %.4f" % (mean_prc, prc_std),
    )  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle="--", color="black", alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.legend(loc="lower left")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def to_tensor(data, dtype=torch.float32, device=None):

    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        tensor = torch.tensor(data.values, dtype=dtype)
    elif isinstance(data, list):
        tensor = torch.tensor(data, dtype=dtype)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_pos_edge_index).to(device)
    return (
        val_scores.cpu().numpy(),
        val_labels.cpu().numpy(),
        get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy()),
    )


def knn_binarize(sim_matrix, k=2, symmetric=True):
    if isinstance(sim_matrix, pd.DataFrame):
        sim_matrix = sim_matrix.values
    n = sim_matrix.shape[0]
    adj = np.zeros_like(sim_matrix, dtype=int)
    for i in range(n):
        neighbors = np.argsort(sim_matrix[i])[::-1][:k]
        adj[i, neighbors] = 1
    if symmetric:
        adj = np.maximum(adj, adj.T)
    return adj
