import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feat_dim", type=int, default=64, help="Input node feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--out_dim", type=int, default=128, help="Output dimension")
    parser.add_argument(
        "--fusion_head", type=int, default=8, help="The number of heads during cross-attention fusion"
    )
    parser.add_argument(
        "--fusion_dropout", type=float, default=0.1, help="Dropout during cross-attention fusion"
    )
    parser.add_argument(
        "--gcn_layers", type=int, default=4, help="Number of residual connection layers"
    )
    parser.add_argument(
        "--att_dropout", type=float, default=0.1, help="Dropout during the update of homogeneous graphs"
    )
    parser.add_argument("--hops", type=int, default=9, help="Structural layer neighbor hop count")
    parser.add_argument("--kneighbor", type=int, default=5, help="k-nearest neighbor support vector")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA"
    )
    parser.add_argument("--peak_lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)"
    )
    return parser.parse_args()
