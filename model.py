import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATv2Conv, GATConv  
import torch_geometric
import torch_geometric
from torch.nn.modules.module import Module
from torch_geometric.utils import dense_to_sparse, add_self_loops
class GATMultiHeadEnhancer(nn.Module):
    def __init__(self, in_features, num_heads=1, dropout=0.0):
        super(GATMultiHeadEnhancer, self).__init__()
        
        self.gat_layer = GATConv(
            in_channels=in_features,
            out_channels=in_features,  
            heads=num_heads,
            concat=True, 
            dropout=dropout
        )
        self.num_heads = num_heads
        self.out_features = in_features
    
    def forward(self, features, edge_index):
        combined_output = self.gat_layer(features, edge_index)
        enhanced_views = combined_output.view(-1, self.num_heads, self.out_features)
        enhanced_views = [enhanced_views[:, i, :] for i in range(self.num_heads)]
        
        return enhanced_views
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.leaky_relu, num_heads=8):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.tau = 0.5  # Add temperature parameter
        # GATv2Conv layers
        self.gat1 = GATv2Conv(in_channels=in_features, out_channels=out_features // num_heads, heads=num_heads,
                              dropout=dropout)
        # self.gat3 = GATv2Conv(in_channels=out_features, out_channels=out_features, heads=num_heads, dropout=dropout,
        #                       concat=False)
        self.linear_proj = nn.Linear(in_features, out_features)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_features, nhead=num_heads, dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # Output layers
        self.alpha_param = nn.Parameter(torch.tensor([0.8]))
        self.beta_param = nn.Parameter(torch.tensor([0.8]))
        self.alpha_param1 = nn.Parameter(torch.tensor([0.8]))
        self.beta_param1 = nn.Parameter(torch.tensor([0.8]))

        self.linear_out = nn.Linear(out_features, in_features)
        self.disc = Discriminator(self.out_features)

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()


    def forward(self, feat, feat_a, edge_index, adj_new):
       # Graph Attention and Transformer encoding
        z_gat = F.dropout(feat, self.dropout, self.training)
        z_gat = self.gat1(z_gat, edge_index)

        ff = self.linear_proj(feat)
        z_trans = self.transformer_encoder(ff.unsqueeze(0)).squeeze(0)

        alpha = torch.sigmoid(self.alpha_param)
        z = alpha * z_gat + (1 - alpha) * z_trans

        emb = self.act(z)
        z = self.linear_out(z)
      # Operation on perturbed graph
        z_a_gat = F.dropout(feat_a, self.dropout, self.training)
        z_a_gat = self.gat1(z_a_gat, edge_index)

        fa = self.linear_proj(feat_a)
        fa_trans = self.transformer_encoder(fa.unsqueeze(0)).squeeze(0)

        beta = torch.sigmoid(self.beta_param)
        z_a_gat = beta * z_a_gat + (1 - beta) * fa_trans

        emb_a = self.act(z_a_gat)

        g = self.read(emb, adj_new)
        g = self.sigm(g)

        g_a = self.read(emb_a, adj_new)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return emb, z, ret, ret_a
class Encoder_sparse(Module):
    
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
        
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        #self.linear1 = torch.nn.Linear(self.dim_input, self.dim_output)
        #self.linear2 = torch.nn.Linear(self.dim_output, self.dim_input)
        
        #self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim_output))
        #self.weight1_de = Parameter(torch.FloatTensor(self.dim_output, self.dim_input))
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        #x = self.linear1(x)
        #x = self.linear2(x)
        
        #x = torch.mm(x, self.weight1_en)
        #x = torch.mm(x, self.weight1_de)
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x
