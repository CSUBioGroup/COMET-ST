import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd

def permute_stcomet_features(feature): 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated 

def _build_stcomet_neighbor_interaction(distance_matrix, n_neighbors):
    n_spot = distance_matrix.shape[0]
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        distance = distance_matrix[i, :].argsort()
        for t in range(1, n_neighbors + 1):
            interaction[i, distance[t]] = 1
    return interaction


def _symmetrize_stcomet_interaction(interaction):
    adj = interaction + interaction.T
    return np.where(adj > 1, 1, adj)


def construct_stcomet_spatial_graph(adata, n_neighbors=3):
    """Construct the stCOMET spot-to-spot spatial graph."""
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    interaction = _build_stcomet_neighbor_interaction(distance_matrix, n_neighbors)

    adata.obsm['distance_matrix'] = distance_matrix
    adata.obsm['graph_neigh'] = interaction
    adata.obsm['adj'] = _symmetrize_stcomet_interaction(interaction)

def _to_stcomet_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _smooth_stcomet_expression_spatially(adata, n_neighbors=6, alpha=0.5):
    from sklearn.neighbors import NearestNeighbors
    X = _to_stcomet_dense(adata.X).astype(np.float32)
    spatial = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(spatial)
    _, idx = nbrs.kneighbors(spatial)
    neigh_idx = idx[:, 1:]
    X_neigh = X[neigh_idx].mean(axis=1)
    X_smooth = alpha * X + (1 - alpha) * X_neigh
    return X_smooth


def _standardize_stcomet_gene_matrix(X):
    G = np.asarray(X, dtype=np.float32).T
    G = G - G.mean(axis=1, keepdims=True)
    std = G.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return G / std


def _build_stcomet_gene_module_graph(adata_gene, gene_neighbors, n_pcs):
    n_pcs_use = min(n_pcs, adata_gene.n_obs - 1, adata_gene.n_vars - 1)
    if n_pcs_use >= 2:
        sc.pp.pca(adata_gene, n_comps=n_pcs_use)
        sc.pp.neighbors(
            adata_gene,
            n_neighbors=gene_neighbors,
            n_pcs=n_pcs_use,
            use_rep="X_pca",
        )
    else:
        sc.pp.neighbors(
            adata_gene,
            n_neighbors=gene_neighbors,
            use_rep="X",
        )


def _select_stcomet_module_genes(labels, gene_names, min_module_size, min_keep_genes):
    module_sizes = np.bincount(labels)
    keep_modules = np.where(module_sizes >= min_module_size)[0]
    keep_mask = np.isin(labels, keep_modules)
    keep_genes = np.asarray(gene_names)[keep_mask]

    if len(keep_genes) < min_keep_genes:
        print(
            f"[GeneModule-Leiden] only keep {len(keep_genes)} genes, "
            f"fallback to original HVGs."
        )
        keep_genes = np.asarray(gene_names)
        keep_mask = np.ones(len(gene_names), dtype=bool)

    print(
        f"[GeneModule-Leiden] modules={len(module_sizes)}, "
        f"large_modules={len(keep_modules)}, "
        f"selected_genes={len(keep_genes)}"
    )
    return keep_genes, keep_mask, module_sizes, keep_modules


def _make_stcomet_module_table(gene_names, labels, module_sizes, keep_mask):
    return pd.DataFrame({
        "gene": np.asarray(gene_names),
        "module": labels,
        "module_size": module_sizes[labels],
        "selected": keep_mask,
    })


def _prepare_stcomet_hvg_source(adata, use_spatial_smooth, smooth_neighbors, smooth_alpha):
    adata_for_hvg = adata.copy()
    if use_spatial_smooth:
        print("[Preprocess] use spatial smoothing before HVG.")
        adata_for_hvg.X = _smooth_stcomet_expression_spatially(
            adata_for_hvg,
            n_neighbors=smooth_neighbors,
            alpha=smooth_alpha,
        )
    else:
        print("[Preprocess] skip spatial smoothing.")
    return adata_for_hvg


def _select_stcomet_hvg_names(adata_for_hvg, n_top_genes):
    sc.pp.highly_variable_genes(
        adata_for_hvg,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    return adata_for_hvg.var_names[adata_for_hvg.var["highly_variable"]].tolist()


def _store_stcomet_module_metadata(adata, tmp, module_labels, module_sizes, module_df):
    adata.uns["gene_module_df"] = module_df
    adata.uns["gene_module_labels_hvg"] = {
        gene: int(label)
        for gene, label in zip(tmp.var_names, module_labels)
    }
    adata.uns["gene_module_sizes"] = module_sizes.tolist()


def _filter_stcomet_hvg_by_modules(
    adata,
    adata_for_hvg,
    hvg_genes,
    min_module_size,
    min_keep_genes,
    leiden_resolution,
    gene_neighbors,
    n_pcs,
):
    tmp = adata_for_hvg[:, hvg_genes].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_hvg = _to_stcomet_dense(tmp.X)
    keep_genes, module_labels, module_sizes, module_df = _filter_stcomet_genes_by_coexpression_modules(
        X_hvg,
        tmp.var_names,
        min_module_size=min_module_size,
        min_keep_genes=min_keep_genes,
        leiden_resolution=leiden_resolution,
        gene_neighbors=gene_neighbors,
        n_pcs=n_pcs,
    )
    _store_stcomet_module_metadata(adata, tmp, module_labels, module_sizes, module_df)
    return keep_genes


def _mark_stcomet_selected_genes(adata, keep_genes):
    adata.var["highly_variable"] = False
    adata.var.loc[keep_genes, "highly_variable"] = True


def _normalize_stcomet_expression(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def _filter_stcomet_genes_by_coexpression_modules(
    X,
    gene_names,  
    min_module_size=30,
    min_keep_genes=500,
    leiden_resolution=1.0,
    gene_neighbors=15,
    n_pcs=30,
):
    G = _standardize_stcomet_gene_matrix(X)
    adata_gene = sc.AnnData(G)
    adata_gene.obs_names = np.asarray(gene_names).astype(str)
    _build_stcomet_gene_module_graph(adata_gene, gene_neighbors, n_pcs)
    sc.tl.leiden(
        adata_gene,
        resolution=leiden_resolution,
        key_added="gene_module",
        random_state=0,
    )
    labels = adata_gene.obs["gene_module"].astype(int).values
    keep_genes, keep_mask, module_sizes, _ = _select_stcomet_module_genes(
        labels,
        gene_names,
        min_module_size,
        min_keep_genes,
    )
    module_df = _make_stcomet_module_table(gene_names, labels, module_sizes, keep_mask)
    return list(keep_genes), labels, module_sizes, module_df

def preprocess_stcomet(
    adata,
    use_spatial_smooth=True,
    smooth_neighbors=6,
    smooth_alpha=0.5,
    n_top_genes=3000,
    use_gene_module_filter=True,
    min_module_size=30,
    min_keep_genes=500,
    leiden_resolution=1.0,
    gene_neighbors=15,
    n_pcs=30,
):
    adata_for_hvg = _prepare_stcomet_hvg_source(
        adata,
        use_spatial_smooth,
        smooth_neighbors,
        smooth_alpha,
    )
    hvg_genes = _select_stcomet_hvg_names(adata_for_hvg, n_top_genes)
    if use_gene_module_filter:
        keep_genes = _filter_stcomet_hvg_by_modules(
            adata,
            adata_for_hvg,
            hvg_genes,
            min_module_size=min_module_size,
            min_keep_genes=min_keep_genes,
            leiden_resolution=leiden_resolution,
            gene_neighbors=gene_neighbors,
            n_pcs=n_pcs,
        )
    else:
        print("[GeneModule] skip gene module filtering.")
        keep_genes = hvg_genes

    _mark_stcomet_selected_genes(adata, keep_genes)
    print(f"[Preprocess] final selected genes for Encoder: {len(keep_genes)}")
    _normalize_stcomet_expression(adata)


def get_stcomet_features(adata):
    adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    feat_a = permute_stcomet_features(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def normalize_stcomet_adjacency(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_stcomet_adjacency(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_stcomet_adjacency(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def fix_stcomet_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
       torch.set_num_interop_threads(1)
    except RuntimeError:
       pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    
