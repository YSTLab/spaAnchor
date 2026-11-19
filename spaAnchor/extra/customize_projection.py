import spaAnchor as sa
import scanpy as sc
import torch
import numpy as np
from typing import List

from ..utils.projection import aggr_by_list


class custom_scvi(sa.HANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def projection_cell(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ):
        import scvi
        scvi.settings.seed = 0

        adata = sc.concat(adata_list, label="batch", keys=["0", "1"])
        adata.layers["counts"] = adata.obsm["X_raw"]
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch"
        )
        model = scvi.model.SCVI(adata)
        model.train(early_stopping=True, enable_progress_bar=True)
        adata.obsm['cell_latent'] = model.get_latent_representation(adata).astype(np.float32)

        return [adata[id == adata.obs["batch"]] for id in ["0", "1"]]
    

class custom_scvi_cellcharter(custom_scvi):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def projection(
        self,
        adata_list: List[sc.AnnData],
        graphs: List[torch.LongTensor],
        **kwargs
    ) -> List[sc.AnnData]:
    
        latent_adata_list = self.projection_cell(adata_list, **kwargs)
        latent_adata_list = aggr_by_list(latent_adata_list, graphs)

        for i in range(len(latent_adata_list)):
            adata_list[i].obsm["cell_latent"] = latent_adata_list[i].obsm["cell_latent"] 
            adata_list[i].obsm["niche_latent"] = latent_adata_list[i].obsm["niche_latent"]

        return adata_list
