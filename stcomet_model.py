import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv  
from torch.nn.modules.module import Module

class stCOMETMultiHeadEnhancer(nn.Module):
    def __init__(self, in_features, num_heads=1, dropout=0.0):
        super(stCOMETMultiHeadEnhancer, self).__init__()
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

class stCOMETContrastiveScorer(nn.Module):
    def __init__(self, n_h):
        super(stCOMETContrastiveScorer, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self._initialize_bilinear_layer(m)

    def _initialize_bilinear_layer(self, layer):
        if isinstance(layer, nn.Bilinear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _score_pair(self, node_emb, context_emb):
        return self.f_k(node_emb, context_emb)

    def _apply_score_bias(self, score, bias):
        if bias is not None:
            score += bias
        return score

    def _combine_positive_negative_scores(self, positive_score, negative_score):
        return torch.cat((positive_score, negative_score), 1)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  
        sc_1 = self._score_pair(h_pl, c_x)
        sc_2 = self._score_pair(h_mi, c_x)
        sc_1 = self._apply_score_bias(sc_1, s_bias1)
        sc_2 = self._apply_score_bias(sc_2, s_bias2)
        return self._combine_positive_negative_scores(sc_1, sc_2)
    
class stCOMETContextReadout(nn.Module):
    def __init__(self):
        super(stCOMETContextReadout, self).__init__()

    def _aggregate_neighbor_context(self, emb, mask):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        return vsum / row_sum

    def _normalize_context(self, context_emb):
        return F.normalize(context_emb, p=2, dim=1)

    def forward(self, emb, mask=None):
        global_emb = self._aggregate_neighbor_context(emb, mask)
        return self._normalize_context(global_emb) 
    
    
class stCOMETEncoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.leaky_relu, num_heads=8):
        super(stCOMETEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.tau = 0.5  
        self.gat1 = GATv2Conv(in_channels=in_features, out_channels=out_features // num_heads, heads=num_heads,
                              dropout=dropout)
        self.linear_proj = nn.Linear(in_features, out_features)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_features, nhead=num_heads, dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.alpha_param = nn.Parameter(torch.tensor([0.8]))
        self.beta_param = nn.Parameter(torch.tensor([0.8]))
        self.alpha_param1 = nn.Parameter(torch.tensor([0.8]))
        self.beta_param1 = nn.Parameter(torch.tensor([0.8]))
        self.linear_out = nn.Linear(out_features, in_features)
        self.disc = stCOMETContrastiveScorer(self.out_features)
        self.read = stCOMETContextReadout()
        self.sigm = nn.Sigmoid()

    def _encode_view(self, feat, edge_index, gate_param):
        z_gat = F.dropout(feat, self.dropout, self.training)
        z_gat = self.gat1(z_gat, edge_index)
        projected_feat = self.linear_proj(feat)
        z_trans = self.transformer_encoder(projected_feat.unsqueeze(0)).squeeze(0)
        gate = torch.sigmoid(gate_param)
        return gate * z_gat + (1 - gate) * z_trans

    def _read_context(self, emb, adj_new):
        graph_context = self.read(emb, adj_new)
        return self.sigm(graph_context)

    def forward(self, feat, feat_a, edge_index, adj_new):
        z = self._encode_view(feat, edge_index, self.alpha_param)
        emb = self.act(z)
        z = self.linear_out(z)

        z_a = self._encode_view(feat_a, edge_index, self.beta_param)
        emb_a = self.act(z_a)
        g = self._read_context(emb, adj_new)
        g_a = self._read_context(emb_a, adj_new)
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)
        return emb, z, ret, ret_a


def _build_stcomet_neighbor_mask(adj):
    adj = adj - torch.diag_embed(torch.diag(adj))
    adj[adj > 0] = 1
    return adj


def _count_stcomet_positive_pairs(adj):
    return torch.sum(adj, 1) * 2 + 1


def _normalize_stcomet_latent_pair(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
    return z1, z2


def _stcomet_exponential_similarity(left, right, tau):
    return torch.exp(torch.mm(left, right.t()) / tau)


def stcomet_directional_neighborhood_loss(z1, z2, adj, tau=0.5, hidden_norm=True):
    adj = _build_stcomet_neighbor_mask(adj)
    nei_count = _count_stcomet_positive_pairs(adj)
    z1, z2 = _normalize_stcomet_latent_pair(z1, z2, hidden_norm)

    intra_view_sim = _stcomet_exponential_similarity(z1, z1, tau)
    inter_view_sim = _stcomet_exponential_similarity(z1, z2, tau)
    loss = (
        inter_view_sim.diag()
        + (intra_view_sim * adj).sum(1)
        + (inter_view_sim * adj).sum(1)
    ) / (
        intra_view_sim.sum(1)
        + inter_view_sim.sum(1)
        - intra_view_sim.diag()
    )
    loss = loss / nei_count
    return -torch.log(loss).mean()


def stcomet_symmetric_neighborhood_loss(z1, z2, adj, tau=1.0, hidden_norm=True):
    l1 = stcomet_directional_neighborhood_loss(z1, z2, adj, tau, hidden_norm)
    l2 = stcomet_directional_neighborhood_loss(z2, z1, adj, tau, hidden_norm)
    return (l1 + l2) * 0.5


def compute_stcomet_multiview_loss(embeddings, graph_neigh):
    contrastive_loss_value = 0
    if len(embeddings) > 1:
        base_embedding = embeddings[0]
        for i in range(1, len(embeddings)):
            contrastive_loss_value += stcomet_symmetric_neighborhood_loss(
                base_embedding,
                embeddings[i],
                graph_neigh,
                0.25,
            )
        contrastive_loss_value /= (len(embeddings) - 1)
    return contrastive_loss_value


def compute_stcomet_training_loss(
    embeddings,
    reconstruction,
    aggregated_features,
    graph_neigh,
    loss_owner,
):
    contrastive_loss_value = compute_stcomet_multiview_loss(embeddings, graph_neigh)
    loss_feat = F.mse_loss(aggregated_features, reconstruction)
    if hasattr(loss_owner, 'recon_weight') and hasattr(loss_owner, 'contrastive_weight'):
        return loss_owner.recon_weight * loss_feat + loss_owner.contrastive_weight * contrastive_loss_value
    return 0.5 * loss_feat + contrastive_loss_value
