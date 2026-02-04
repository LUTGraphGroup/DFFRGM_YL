from param import *
from pathlib import Path
from utils import *
from torch_geometric.data import Data

project_root = Path(__file__).parent.parent
data_dir = project_root / "data1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    args = parameter_parser()

    # Read the adjacency matrix (disease-metabolite association)
    Adj = pd.read_csv(data_dir / "association_matrix.csv", header=0)
    Ori_adj = np.asmatrix(np.where(Adj == 1))  # 交叉验证用
    print(f"Disease metabolite adjacency matrix:\n{Adj}")

    # Read the disease similarity matrix
    Dis_simi = pd.read_csv(data_dir / "diease_simi_network.csv", header=0)
    print(f"Disease Similarity Matrix:\n{Dis_simi}")
    Dis_adj=knn_binarize(Dis_simi, k=args.kneighbor)
    print(Dis_adj.shape)


    Ori_dis_index_matrix = np.asmatrix(np.where(Dis_adj == 1))
    Dis_index_matrix = torch.tensor(Ori_dis_index_matrix, dtype=torch.long).to(device)

    Dis_features = pd.read_csv(
         data_dir / "diease_simi_network.csv", header=0
    )
    Dis_features = pca_reduce(Dis_features, n_components=64)
    Dis_features = torch.tensor(Dis_features, dtype=torch.float).to(device)



    # Read the metabolite similarity matrix
    Meta_simi = pd.read_csv(data_dir / "metabolite_simi_network.csv", header=0)
    print(f"Metabolite similarity matrix:\n{Meta_simi}")

    Meta_adj=knn_binarize(Meta_simi,k=args.kneighbor)
    Ori_meta_index_matrix = np.asmatrix(np.where(Meta_adj == 1))
    Meta_index_matrix = torch.tensor(Ori_meta_index_matrix, dtype=torch.long).to(device)

    Meta_features = pd.read_csv(data_dir / "metabolite_simi_network.csv", header=0)
    Meta_features = pca_reduce(Meta_features, n_components=64)
    Meta_features = torch.tensor(Meta_features, dtype=torch.float).to(device)

    Meta_data = Data(
        x=Meta_features,edge_index=Meta_index_matrix,Meta_adj=Meta_adj,Meta_simi=Meta_simi,
    )
    Dis_data = Data(
        x=Dis_features, edge_index=Dis_index_matrix, Dis_adj=Dis_adj, Dis_simi=Dis_simi
    )

    feature = torch.cat((Dis_features, Meta_features))
    Dis_adj=to_tensor(Dis_adj)
    Meta_adj=to_tensor(Meta_adj)
    return Adj, Dis_data, Meta_data, Ori_adj, Dis_adj, Meta_adj, feature
