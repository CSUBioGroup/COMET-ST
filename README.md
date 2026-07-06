# stCOMET：Co-expression Module-Enhanced Multi-view Contrastive Learning for Spatial Domain Identification in Spatial Transcriptomics
# Overview
Spatial transcriptomics measures gene expression while preserving tissue coordinates, and spatial domain identification—partitioning tissue into regions with coherent transcriptional programs—has become a foundational task for downstream analyses such as spatially resolved differential expression and cell–cell communication inference. However, current methods face two key limitations: at the feature level, highly variable gene sets are used without co-expression-based refinement, allowing noisy or isolated genes to degrade learned representations; at the representation level, embeddings are typically learned from a single spatial graph view, offering no explicit guarantee of stability under graph perturbation. To address these gaps, we present stCOMET, a spatial representation learning framework that couples co-expression-guided gene selection—which retains only genes belonging to coherent co-expression modules—with graph-based multi-view augmentation and a neighbourhood-aware contrastive objective that enforces representation consistency across complementary spatial views. Across 12 human dorsolateral prefrontal cortex sections and five MERFISH hypothalamic sections, stCOMET achieved the best average performance among eight evaluated methods, as measured by adjusted Rand index, normalized mutual information, completeness and homogeneity. stCOMET further identified a periventricular-region-associated MERFISH domain whose marker genes were enriched for neuropeptide signalling and hormone secretion, demonstrating the biological interpretability of the inferred spatial organization. These results establish stCOMET as a robust and interpretable framework for spatial domain identification across sequencing- and imaging-based spatial transcriptomics platforms.
<p align="center">
  <img src="figures/stCOMET_fig.png" width="900">
</p>

# Requirements
The main dependencies are:
Python >= 3.10, 
PyTorch 2.5.0 + CUDA 12.4, 
torch-geometric 2.7.0,
scanpy 1.11.5, 
anndata 0.11.4,
squidpy 1.6.5, 
numpy 1.26.4,
pandas 2.3.3,
scipy 1.15.3, 
scikit-learn 1.7.2,
scikit-image 0.25.2,
matplotlib 3.10.8, 
seaborn 0.13.2, 
umap-learn 0.5.11,
faiss-gpu 1.7.2, 
spatialdata 0.5.0, 
zarr 2.18.3, 
h5py 3.16.0.
