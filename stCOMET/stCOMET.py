import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import tqdm

from .stcomet_model import (
    compute_stcomet_training_loss,
    stCOMETEncoder,
    stCOMETMultiHeadEnhancer,
)
from .stcomet_preprocess import (
    construct_stcomet_spatial_graph,
    fix_stcomet_seed,
    get_stcomet_features,
    permute_stcomet_features,
    preprocess_stcomet,
    preprocess_stcomet_adjacency,
)
from .stcomet_utils import stcomet_spatial_clustering


class stCOMET:
    def __init__(
        self,
        adata,
        device=torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600,
        dim_output=256,
        random_seed=41,
        alpha=10,
        beta=1,
        datatype='10X',
        gat_heads=8,
        gat_dropout=0,
        gat_concat=True,
        enhancer_heads=1,
        contrastive_tau=0.25,
        n_clusters=7,
        dataset_path=None,
        recon_weight=0.5,
        contrastive_weight=1.0,
    ):
        self.adata = adata.copy()
        self.device = device
        self.n_clusters = n_clusters
        self.dataset_path = dataset_path
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.datatype = datatype
        self.enhancer_heads = enhancer_heads
        self.contrastive_tau = contrastive_tau
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        self.gat_concat = gat_concat

        fix_stcomet_seed(self.random_seed)
        self._prepare_anndata_inputs(adata)
        self._prepare_tensor_inputs()

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        self.enhancer = stCOMETMultiHeadEnhancer(
            in_features=self.dim_input,
            num_heads=self.enhancer_heads,
            dropout=self.gat_dropout,
        ).to(self.device)

        self._prepare_training_adjacency()

    def _prepare_anndata_inputs(self, original_adata):
        if 'highly_variable' not in original_adata.var.keys():
            preprocess_stcomet(self.adata)

        if 'adj' not in original_adata.obsm.keys():
            construct_stcomet_spatial_graph(self.adata)

        if 'feat' not in original_adata.obsm.keys():
            get_stcomet_features(self.adata)

    def _prepare_tensor_inputs(self):
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])
        ).to(self.device)

    def _prepare_training_adjacency(self):
        self.adj = preprocess_stcomet_adjacency(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)

    def _stcomet_dense_adj_to_edge_index(self, adj):
        if hasattr(adj, 'is_sparse') and adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        return torch.nonzero(adj_dense, as_tuple=False).t().contiguous()

    def _prepare_training_graph_state(self):
        edge_index = torch.nonzero(self.adj.to_dense(), as_tuple=False).t().contiguous()
        num_nodes = self.adj.size(0)
        adj_sparse = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(num_nodes, num_nodes),
        )
        self.adj_t = adj_sparse.t().to(self.device)
        self._prepare_aggregated_features()

    def _prepare_aggregated_features(self):
        knn_adj = self.adj.cpu().numpy()
        np.fill_diagonal(knn_adj, 1)
        self.knn_adj = torch.FloatTensor(knn_adj).to(self.device)
        degree = self.knn_adj.sum(1, keepdim=True)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        self.aggregated_features = torch.mm(self.knn_adj, self.features) / degree

    def _build_model(self):
        return stCOMETEncoder(
            in_features=self.dim_input,
            out_features=self.dim_output,
            graph_neigh=self.graph_neigh,
            dropout=self.gat_dropout,
            num_heads=self.gat_heads,
            act=F.leaky_relu,
        ).to(self.device)

    def _initialize_training_model(self):
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _forward_dense_view(self, view, edge_index):
        emb, z, _, _ = self.model(view, view, edge_index, self.graph_neigh)
        return emb, z

    def _run_dense_training_epoch(self):
        edge_index = self._stcomet_dense_adj_to_edge_index(self.adj)
        enhanced_views = [self.features] + self.enhancer(self.features, edge_index)
        embeddings = []
        reconstructions = []

        for view in enhanced_views:
            hidden_emb, reconstruction = self._forward_dense_view(view, edge_index)
            embeddings.append(hidden_emb)
            reconstructions.append(reconstruction)

        return compute_stcomet_training_loss(
            embeddings,
            reconstructions[0],
            self.aggregated_features,
            self.graph_neigh,
            self,
        )

    def _optimization_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_current_embedding(self):
        edge_index = self._stcomet_dense_adj_to_edge_index(self.adj)
        return self.model(
            self.features,
            self.features_a,
            edge_index,
            self.graph_neigh,
        )[1]

    def _read_ground_truth(self):
        metadata_path = f'{self.dataset_path}/metadata.tsv'
        df_meta = pd.read_csv(metadata_path, sep='\t')
        return df_meta['layer_guess'].values

    def _cluster_and_score(self, adata_for_clustering):
        from sklearn import metrics

        stcomet_spatial_clustering(
            adata_for_clustering,
            n_clusters=self.n_clusters,
            method='mclust',
            refinement=True,
            radius=50,
        )
        adata_for_clustering = adata_for_clustering[
            ~pd.isnull(adata_for_clustering.obs['ground_truth'])
        ]

        if 'domain' not in adata_for_clustering.obs.columns:
            return 0.0, adata_for_clustering, False

        adata_for_clustering = adata_for_clustering[
            ~pd.isnull(adata_for_clustering.obs['domain'])
        ]
        if len(adata_for_clustering) == 0:
            return 0.0, adata_for_clustering, False

        ari = metrics.adjusted_rand_score(
            adata_for_clustering.obs['domain'],
            adata_for_clustering.obs['ground_truth'],
        )
        return ari, adata_for_clustering, True

    def _evaluate_epoch(self, epoch, callback):
        self.model.eval()
        try:
            truth = self._read_ground_truth()
            adata_for_clustering = self.adata.copy()

            with torch.no_grad():
                emb_result = self._get_current_embedding()

            adata_for_clustering.obsm['emb'] = emb_result.detach().cpu().numpy()
            adata_for_clustering.obs['ground_truth'] = truth
            ari, _, has_valid_score = self._cluster_and_score(adata_for_clustering)

            if has_valid_score:
                print(f"Epoch {epoch + 1}: ARI = {ari:.4f}")
            if callback is not None:
                callback(epoch + 1, ari, copy.deepcopy(self.model.state_dict()))
        except Exception as e:
            print(f"Epoch {epoch + 1}: Error during clustering or ARI calculation: {str(e)}")
            ari = 0.0
            if callback is not None:
                callback(epoch + 1, ari, copy.deepcopy(self.model.state_dict()))
        finally:
            self.model.train()

    def _finalize_training_output(self):
        with torch.no_grad():
            self.model.eval()
            edge_index = self._stcomet_dense_adj_to_edge_index(self.adj)
            self.adata.obsm['emb'] = self.model(
                self.features,
                self.features_a,
                edge_index,
                self.graph_neigh,
            )[1].detach().cpu().numpy()
            return self.adata

    def train_stcomet(self):
        return self._train_stcomet_with_callback()

    def train_stcomet_with_callback(self, callback=None):
        return self._train_stcomet_with_callback(callback)

    def _train_stcomet_with_callback(self, callback=None):
        self._prepare_training_graph_state()
        self._initialize_training_model()

        print('Begin to train ST data...')
        epoch_model_states = {}

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_a = permute_stcomet_features(self.features)

            loss = self._run_dense_training_epoch()
            self._optimization_step(loss)

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                epoch_model_states[epoch] = copy.deepcopy(self.model.state_dict())

            if epoch >= 199 and (epoch + 1) % 50 == 0:
                self._evaluate_epoch(epoch, callback)

        print("\n" + "=" * 50)
        print("Training completed!")
        return self._finalize_training_output()

  
