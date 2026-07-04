import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import copy
import torch
from .preprocess import compute_moranI_and_filter, preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from sklearn.decomposition import PCA 
from .model import Encoder, Encoder_sparse, Encoder_map, Encoder_sc, GATMultiHeadEnhancer
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from torch_sparse import SparseTensor 
from torch_geometric.nn import GATConv
import scanpy as sc

class GraphST():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=256,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        gat_heads=8,          
        gat_dropout=0,       
        gat_concat=True ,  
        enhancer_heads=1 ,
        contrastive_tau=0.25 ,
        n_clusters=7 ,
        dataset_path=None ,
        recon_weight=0.5,
        contrastive_weight=1.0 
        ):
        
        self.adata = adata.copy()
        self.device = device
        self.n_clusters = n_clusters 
        self.dataset_path = dataset_path
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.enhancer_heads = enhancer_heads
        self.contrastive_tau = contrastive_tau
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)
        

        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
 
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        self.gat_concat = gat_concat
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        self.enhancer = GATMultiHeadEnhancer(
                in_features=self.dim_input,  
                num_heads=self.enhancer_heads,
                dropout=self.gat_dropout
            ).to(self.device)
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
    
    def neighbor_contrastive_loss(self, z1, z2, adj, tau=0.5, hidden_norm=True):
        adj = adj - torch.diag_embed(torch.diag(adj))
        adj[adj > 0] = 1
    
        nei_count = torch.sum(adj, 1) * 2 + 1
    
        if hidden_norm:
          z1 = F.normalize(z1, p=2, dim=1)
          z2 = F.normalize(z2, p=2, dim=1)
    
        f = lambda x: torch.exp(x / tau)
    
        intra_view_sim = f(torch.mm(z1, z1.t()))  
        inter_view_sim = f(torch.mm(z1, z2.t()))  
    
        loss = (inter_view_sim.diag() + 
                (intra_view_sim * adj).sum(1) + 
                (inter_view_sim * adj).sum(1)) / (
                    intra_view_sim.sum(1) + 
                    inter_view_sim.sum(1) - 
                    intra_view_sim.diag())
    
        loss = loss / nei_count
        return -torch.log(loss).mean()

    def contrastive_loss(self, z1, z2, adj, tau=1.0, hidden_norm=True):
       l1 = self.neighbor_contrastive_loss(z1, z2, adj, tau, hidden_norm)
       l2 = self.neighbor_contrastive_loss(z2, z1, adj, tau, hidden_norm)
       return (l1 + l2) * 0.5 
    
    def dense_adj_to_edge_index(self, adj):
      if hasattr(adj, 'is_sparse') and adj.is_sparse:
        adj_dense = adj.to_dense()
      else:
        adj_dense = adj
      
      edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
      return edge_index
     
    def train(self):
        
        return self._train_with_callback()
    
    def train_with_callback(self, callback=None):
       
        return self._train_with_callback(callback)
    
    def _train_with_callback(self, callback=None):
       
        if self.datatype not in ['Stereo', 'Slide']: 
            edge_index = torch.nonzero(self.adj.to_dense(), as_tuple=False).t().contiguous()
            num_nodes = self.adj.size(0)
            adj_sparse = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                    sparse_sizes=(num_nodes, num_nodes))
            adj_t = adj_sparse.t() 
            self.adj_t = adj_t.to(self.device) 
        else:
            self.adj_t = None 
        
        # Add: Build KNN graph for feature aggregation
        if self.datatype not in ['Stereo', 'Slide']:
            # Use existing adjacency matrix to build KNN graph (k=3)
            knn_adj = self.adj.cpu().numpy() if not self.adj.is_sparse else self.adj.to_dense().cpu().numpy()
            # Ensure diagonal is 0, no self-connections
            np.fill_diagonal(knn_adj, 1)
            self.knn_adj = torch.FloatTensor(knn_adj).to(self.device)
            
            # Calculate aggregated features: average neighbor features for each node
            degree = self.knn_adj.sum(1, keepdim=True)
            degree = torch.where(degree == 0, torch.ones_like(degree), degree)  # Avoid division by zero
            self.aggregated_features = torch.mm(self.knn_adj, self.features) / degree
        
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(
                in_features=self.dim_input,
                out_features=self.dim_output,
                graph_neigh=self.graph_neigh,
                dropout=self.gat_dropout,
                num_heads=self.gat_heads,
                act=F.leaky_relu
            ).to(self.device)
        
        # Add enhancer parameters to optimizer
        params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(params, self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        
        print('Begin to train ST data...')
        
      
        epoch_model_states = {}
        
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            
            # Generate augmented features
            self.features_a = permutation(self.features)
            
            # If it's a dense graph, use adj_t
            if self.datatype not in ['Stereo', 'Slide']:
                edge_index = self.dense_adj_to_edge_index(self.adj)
                
                # Use enhancer to generate multiple views
                enhanced_views = [self.features] + self.enhancer(self.features, edge_index)
                
                # Collect embeddings from all enhanced views
                embeddings = []
                reconstructions = []
                
                for view in enhanced_views:
                    if self.datatype in ['Stereo', 'Slide']:
                        hiden_emb, emb, _, _ = self.model(view, view, self.adj)
                    else:
                        emb, z, _, _ = self.model(
                            view,           # feat
                            view,         
                            edge_index,     # new parameter: edge_index
                            self.graph_neigh  # new parameter: adj_new (using graph_neigh)
                        )
                        hiden_emb = emb  # Hidden representation, used for contrastive learning
                        emb = z          # Reconstruction output, used to calculate reconstruction loss
                    embeddings.append(hiden_emb)
                    reconstructions.append(emb)
                
                # Calculate contrastive loss
                contrastive_loss_value = 0
                if len(embeddings) > 1:
                    base_embedding = embeddings[0]
                    for i in range(1, len(embeddings)):
                        contrastive_loss_value += self.contrastive_loss(
                            base_embedding, 
                            embeddings[i], 
                            self.graph_neigh, 
                            0.25
                        )
                    contrastive_loss_value /= (len(embeddings) - 1)
                    
                # Calculate reconstruction loss (using reconstruction from first view)
                loss_feat = F.mse_loss(self.aggregated_features, reconstructions[0])
                
                # Total loss
                if hasattr(self, 'recon_weight') and hasattr(self, 'contrastive_weight'):
                  loss = (self.recon_weight * loss_feat + 
                       self.contrastive_weight * contrastive_loss_value)
                else:
                
                  loss = (0.5 * loss_feat + 1 * contrastive_loss_value)
                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                # Process for sparse graph types
                hiden_emb, emb, _, _ = self.model(self.features, self.features_a, self.adj)
                loss_feat = F.mse_loss(self.features, emb)
                if hasattr(self, 'recon_weight'):
                 loss = self.recon_weight * self.alpha * loss_feat
                else:
                 loss = self.alpha * loss_feat
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                epoch_model_states[epoch] = copy.deepcopy(self.model.state_dict())
            
           
            if epoch >= 199 and (epoch + 1) % 50 == 0:
                self.model.eval()
                try:
                    import pandas as pd
                    from sklearn import metrics
                    from GraphST.utils import clustering
                    
                   
                    metadata_path = f'{self.dataset_path}/metadata.tsv'
                    df_meta = pd.read_csv(metadata_path, sep='\t')
                    truth = df_meta['layer_guess'].values 

                    adata_for_clustering = self.adata.copy()
                    edge_index = self.dense_adj_to_edge_index(self.adj)
                    
                    
                    
                    with torch.no_grad():
                        if self.datatype in ['Stereo', 'Slide']:
                          emb_result = self.model(self.features, self.features_a, self.adj)[1]
                        else:
                            emb_result = self.model(
                                    self.features, 
                                    self.features_a, 
                                    edge_index, 
                                    self.graph_neigh
                                )[1]
                    
                    adata_for_clustering.obsm['emb'] = emb_result.detach().cpu().numpy()
                    adata_for_clustering.obs['ground_truth'] = truth  
                    
                   
                    clustering(adata_for_clustering, 
                                n_clusters=self.n_clusters, 
                                method='mclust', 
                                refinement=True, 
                                radius=50)
                    
                    
                    adata_for_clustering = adata_for_clustering[~pd.isnull(adata_for_clustering.obs['ground_truth'])]
                    if 'domain' in adata_for_clustering.obs.columns:
                        adata_for_clustering = adata_for_clustering[~pd.isnull(adata_for_clustering.obs['domain'])]
                        
                       
                        if len(adata_for_clustering) > 0:
                            ARI = metrics.adjusted_rand_score(adata_for_clustering.obs['domain'],
                                                                adata_for_clustering.obs['ground_truth'])
                            print(f"Epoch {epoch + 1}: ARI = {ARI:.4f}")
                            
                           
                            if callback is not None:
                                callback(epoch + 1, ARI, copy.deepcopy(self.model.state_dict()))
                        else:
                            ARI = 0.0
                            if callback is not None:
                                callback(epoch + 1, ARI, copy.deepcopy(self.model.state_dict()))
                    else:
                        ARI = 0.0
                        if callback is not None:
                            callback(epoch + 1, ARI, copy.deepcopy(self.model.state_dict()))
                        
                except Exception as e:
                    print(f"Epoch {epoch + 1}: Error during clustering or ARI calculation: {str(e)}")
                    ARI = 0.0
                    if callback is not None:
                        callback(epoch + 1, ARI, copy.deepcopy(self.model.state_dict()))
                finally:
                    self.model.train()
        
        print("\n" + "="*50)
        print("Training completed!")
        
        
        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                else:
                    edge_index = self.dense_adj_to_edge_index(self.adj)
                    emb, self.emb_rec, _, _ = self.model(
                        self.features, 
                        self.features_a, 
                        edge_index, 
                        self.graph_neigh
                    )
                return self.emb_rec
            else:  
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                    self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                    edge_index = self.dense_adj_to_edge_index(self.adj)
                    self.adata.obsm['emb'] =  self.model(self.features, 
                        self.features_a, 
                        edge_index, 
                        self.graph_neigh)[1].detach().cpu().numpy()
                    return self.adata
    
    def train_with_best_model(self, best_model_state, callback=None):
       
        if self.datatype not in ['Stereo', 'Slide']: 
            edge_index = torch.nonzero(self.adj.to_dense(), as_tuple=False).t().contiguous()
            num_nodes = self.adj.size(0)
            adj_sparse = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                    sparse_sizes=(num_nodes, num_nodes))
            adj_t = adj_sparse.t() 
            self.adj_t = adj_t.to(self.device) 
        else:
            self.adj_t = None 
        
        # Build KNN graph for feature aggregation
        if self.datatype not in ['Stereo', 'Slide']:
            knn_adj = self.adj.cpu().numpy() if not self.adj.is_sparse else self.adj.to_dense().cpu().numpy()
            np.fill_diagonal(knn_adj, 1)
            self.knn_adj = torch.FloatTensor(knn_adj).to(self.device)
            
            degree = self.knn_adj.sum(1, keepdim=True)
            degree = torch.where(degree == 0, torch.ones_like(degree), degree)
            self.aggregated_features = torch.mm(self.knn_adj, self.features) / degree
        
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(
                in_features=self.dim_input,
                out_features=self.dim_output,
                graph_neigh=self.graph_neigh,
                dropout=self.gat_dropout,
                num_heads=self.gat_heads,
                act=F.leaky_relu
            ).to(self.device)
        
       
        self.model.load_state_dict(best_model_state)
        
        
        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                emb_result = self.model(self.features, self.features_a, self.adj)[1]
            else:
                edge_index = self.dense_adj_to_edge_index(self.adj)
                emb_result = self.model(
                    self.features, 
                    self.features_a, 
                    edge_index, 
                    self.graph_neigh
                )[1]
            
            self.adata.obsm['emb'] = emb_result.detach().cpu().numpy()
            
           
            try:
                from sklearn import metrics
                from GraphST.utils import clustering
                import pandas as pd
                
               
                metadata_path = f'{self.dataset_path}/metadata.tsv'
                df_meta = pd.read_csv(metadata_path, sep='\t')
                truth = df_meta['layer_guess'].values
                self.adata.obs['ground_truth'] = truth
                
               
                adata_for_clustering = self.adata.copy()
                clustering(adata_for_clustering, 
                          n_clusters=self.n_clusters, 
                          method='mclust', 
                          refinement=True, 
                          radius=50)
                
                
                adata_for_clustering = adata_for_clustering[~pd.isnull(adata_for_clustering.obs['ground_truth'])]
                if 'domain' in adata_for_clustering.obs.columns:
                    adata_for_clustering = adata_for_clustering[~pd.isnull(adata_for_clustering.obs['domain'])]
                    
                   
                    if len(adata_for_clustering) > 0:
                        final_ari = metrics.adjusted_rand_score(adata_for_clustering.obs['domain'],
                                                              adata_for_clustering.obs['ground_truth'])
                        print(f"Best ARI: {final_ari:.4f}")
                        
                        
                        if callback is not None:
                            callback(final_ari, adata_for_clustering)
                    else:
                        final_ari = 0.0
                        if callback is not None:
                            callback(final_ari, self.adata)
                else:
                    final_ari = 0.0
                    if callback is not None:
                        callback(final_ari, self.adata)
                        
            except Exception as e:
                print(f"ARI Error: {str(e)}")
                final_ari = 0.0
                if callback is not None:
                    callback(final_ari, self.adata)
            
        return self.adata
         
    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
         
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc
    
    def loss(self, emb_sp, emb_sc):
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M
