import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from mamba_ssm import Mamba


class InnerProductDecoder(Module):

    def __init__(self, input_dim, num_dis, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(input_dim, input_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.dropout = dropout
        self.num_dis = num_dis

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        D = inputs[0 : self.num_dis, :]
        M = inputs[self.num_dis :, :]
        D = torch.mm(D, self.weight)

        D = torch.t(D)
        x = torch.mm(M, D)

        outputs = torch.sigmoid(x)

        return outputs


class ResGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        self.liner = nn.Linear(input_dim, hidden_dim)

        # first floor
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.gates.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Subsequent layer
        for _ in range(1, num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gates.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        res = self.liner(x)  # Initial residual vector

        for i, conv in enumerate(self.layers):
            x_ = conv(x, edge_index)
            x_ = F.relu(x_)

            # Gate-controlled fusion
            gate_input = torch.cat([x_, res], dim=1)
            gate = torch.sigmoid(self.gates[i](gate_input))
            x = gate * x_ + (1 - gate) * res
            x = self.norms[i](x)
        return x


class StructuralViewMambaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, K=7, use_sparse=False):
        super().__init__()
        self.K = K
        self.use_sparse = use_sparse
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.agg_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def normalize(self, x):
        return self.ln(x)

    def propagate_multi_hop(self, adj, features):
        """
        Construct the neighbor feature sequence of each node within 0 to K hops, and the output shape is: (N, K+1, d)
        """
        adj = adj.to(features.dtype)
        N, d = features.shape
        x = features.clone().float()
        node_seq = torch.empty(
            N, self.K + 1, d, dtype=torch.float32, device=features.device
        )

        x = self.normalize(x)
        node_seq[:, 0, :] = x  # The 0th jump feature

        for k in range(1, self.K + 1):
            if self.use_sparse:
                x = torch.sparse.mm(adj, x)
            else:
                x = torch.matmul(adj, x)
            x = self.normalize(x)
            node_seq[:, k, :] = x

        return node_seq.float()  # [N, K+1, d]

    def forward(self, adj, features):

        node_seq = self.propagate_multi_hop(adj, features)  # [N, K+1, d_in]
        node_seq = self.input_proj(node_seq)  # [N, K+1, hidden_dim]
        out_seq = self.mamba(node_seq)  # [N, K+1, hidden_dim]

        weights = torch.softmax(self.agg_attention(out_seq), dim=1)  # [N, K+1, 1]
        out = torch.sum(weights * out_seq, dim=1)

        return out


class UnifiedFusion(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        dropout=0.1,
    ):

        super().__init__()
        self.put_dim = input_dim
        self.proj = nn.Linear(input_dim, output_dim)
        self.cross_attn_s2a = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_a2s = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, h_s, h_a):
        # Cross Attention Fusion
        # Expanded to (N, 1, dim) to conform to the input shape of MultiheadAttention (batch, seq_len, dim)
        qs = h_s.unsqueeze(1)
        ka = h_a.unsqueeze(1)
        qa = h_a.unsqueeze(1)
        ks = h_s.unsqueeze(1)

        out_s, _ = self.cross_attn_s2a(qs, ka, ka)  # Heterogeneous view as query perspectives
        out_a, _ = self.cross_attn_a2s(
            qa, ks, ks
        )  #homogeneous view as the query perspective

        out_s = out_s.squeeze(1)
        out_a = out_a.squeeze(1)

        out = (out_s + out_a)/2
        return out
