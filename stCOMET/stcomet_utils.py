import pandas as pd
import scanpy as sc
import ot
from sklearn.decomposition import PCA
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

def _run_stcomet_mclust(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    np.random.seed(random_seed)
    ro.r.library("mclust")
    from rpy2.robjects import pandas2ri
    with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):

        ro.r(f'set.seed({random_seed})')
        rmclust = ro.r['Mclust']
        input_data = pd.DataFrame(
            adata.obsm[used_obsm],
            index=adata.obs_names
        )
        print(type(input_data)) 
        res = rmclust(input_data, num_cluster, modelNames)
        mclust_res = np.array(res[-2])
    
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def _write_stcomet_pca_embedding(adata, n_components=20, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    adata.obsm['emb_pca'] = pca.fit_transform(adata.obsm['emb'].copy())


def _cluster_stcomet_by_graph_method(adata, method, n_clusters, start, end, increment):
    res = _search_stcomet_resolution(
        adata,
        n_clusters,
        use_rep='emb_pca',
        method=method,
        start=start,
        end=end,
        increment=increment,
    )
    if method == 'leiden':
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain']


def _assign_stcomet_domains(adata, method, n_clusters, start, end, increment):
    if method == 'mclust':
       adata = _run_stcomet_mclust(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method in ['leiden', 'louvain']:
       _cluster_stcomet_by_graph_method(adata, method, n_clusters, start, end, increment)
    return adata


def _apply_stcomet_label_refinement(adata, radius):
    new_type = _refine_stcomet_labels(adata, radius, key='domain')
    adata.obs['domain'] = new_type


def stcomet_spatial_clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    _write_stcomet_pca_embedding(adata)
    adata = _assign_stcomet_domains(
       adata,
       method,
       n_clusters,
       start,
       end,
       increment,
    )
       
    if refinement:  
       _apply_stcomet_label_refinement(adata, radius)
       
def _refine_stcomet_labels(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    new_type = [str(i) for i in list(new_type)]    
    return new_type
    
def _search_stcomet_resolution(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    
