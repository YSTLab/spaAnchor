import spaAnchor as sa
import scanpy as sc
import torch
import numpy as np
from typing import List


class custom_hvg(sa.HANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ) -> List[torch.LongTensor]:
        graph_list = []

        for adata in adata_list:

            if (self.filter_cell): 
                sc.pp.filter_cells(adata, min_genes=1)

            if (self.pp):
                # sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="seurat_v3", subset=True)
                # Such as using cell_ranger for highly variable gene selection.
                sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="cell_ranger", subset=True)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)

            graph_list.append(self.build_graph(adata.obsm[self.spatial_key], **kwargs))

        return graph_list
    

class custom_graph_construction(sa.HANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_graph(self, coords: np.ndarray, user_defined_k: int, **kwargs) -> torch.Tensor:
        from sklearn.neighbors import kneighbors_graph

        adj = kneighbors_graph(coords, n_neighbors=user_defined_k, include_self=False).tocoo()
        edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)

        return edge_index
