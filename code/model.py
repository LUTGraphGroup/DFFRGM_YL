from layers import *
from param import parameter_parser

args = parameter_parser()
class GCN(nn.Module):

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        out_dim,
        gcn_num_layers,  #Number of layers of residual connection in GCN
        num_dis,
    ):
        super(GCN, self).__init__()
        self.feat_dim = (feat_dim,)
        self.hidden_dim = (hidden_dim,)
        self.out_dim = (out_dim,)
        self.num_dis = (num_dis,)
        # Homogeneous Graph Convolutional Layer: (Residual Connection)
        self.resgcnlayer = ResGCNEncoder(
            feat_dim, hidden_dim, gcn_num_layers, args.att_dropout
        )
        # Heterogeneous view encoder
        self.strencoder = StructuralViewMambaEncoder(
            feat_dim, hidden_dim, args.hops, False
        )
        self.fusion = UnifiedFusion(
            hidden_dim, out_dim
        )
        # Decoder: Link Prediction
        self.decoder = InnerProductDecoder(out_dim, num_dis)


    def forward(self, dis_data, meta_data, feature, adj):

        #Feature Update from the homogeneous view ===
        x_dis = self.resgcnlayer(dis_data.x, dis_data.edge_index)
        x_meta = self.resgcnlayer(meta_data.x, meta_data.edge_index)
        x_att = torch.cat((x_dis, x_meta), dim=0)
        #Features Update from the heterogeneous view
        x_str = self.strencoder(adj, feature)

        # Integration of Two View
        x_fusion = self.fusion(x_str, x_att)

        # Decode into Link Scores
        output = self.decoder(x_fusion)
        return output
