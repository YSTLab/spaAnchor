import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, List

from ..utils.HIPT.hipt_model_utils import get_vit256, eval_transforms, VIT256_SMALL_DINO_SHA256
from ..utils.projection import get_cell_embedding, clean_cuda_cache
from .core import HANN


class diagonal_integration(HANN):

    def __init__(
        self,
        model_path: str, 
        library_id: Optional[str] = None,
        img_basis: str = "hires",
        scale_factor: Optional[float] = None,
        batch_size: Optional[int] = 5120,
        feature_key: List[str] = ["X"],
        predict_full=True,
        model_sha256: str = VIT256_SMALL_DINO_SHA256,
        **kwargs
    ):
        super().__init__(
            feature_key=feature_key, 
            predict_full=predict_full, 
            **kwargs
        )
        self.library_id = library_id
        self.img_basis = img_basis
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_sha256 = model_sha256


    def preprocess(self, adata_list: List[sc.AnnData], **kwargs) -> List["torch.LongTensor"]:
        for i, adata in enumerate(adata_list):
            sc.pp.filter_cells(adata, min_genes=1)
            adata.obsm["X_raw"] = adata.to_df()
            adata.obsm["X"] = adata.to_df()

        # Generate cell-level and niche-level embedding
        model = get_vit256(
            pretrained_weights=self.model_path, device=self.device,
            expected_sha256=self.model_sha256,
        )

        cell_embedding_list = []
        niche_embedding_list = []
        for adata in adata_list:
            niche_embedding, cell_embedding = get_cell_embedding(
                adata, 
                model, 
                eval_transforms(), 
                self.spatial_key, 
                self.library_id, 
                self.img_basis, 
                self.scale_factor, 
                self.batch_size, 
                self.device
            ) 
            cell_embedding_list.append(cell_embedding)
            niche_embedding_list.append(niche_embedding)
        del model; clean_cuda_cache()    # Uninstall image model

        cell_embedding_list = np.vstack(cell_embedding_list)
        niche_embedding_list = np.vstack(niche_embedding_list)
        adatas = sc.concat(adata_list, label="batch")
        pca = PCA(n_components=self.reduction_dim, random_state=0)
        adatas.obsm["cell_latent"] = pca.fit_transform(cell_embedding_list)
        adatas.obsm["niche_latent"] = pca.fit_transform(np.vstack(niche_embedding_list))
        for i, adata in zip(adatas.obs["batch"].unique(), adata_list):
            adata.obsm["cell_latent"] = adatas[i == adatas.obs["batch"]].obsm["cell_latent"]
            adata.obsm["niche_latent"] = adatas[i == adatas.obs["batch"]].obsm["niche_latent"]
        latent_adata_list = adata_list

        for i in range(len(latent_adata_list)):
            adata_list[i].obsm["cell_latent"] = latent_adata_list[i].obsm["cell_latent"] 
            adata_list[i].obsm["niche_latent"] = latent_adata_list[i].obsm["niche_latent"]

        return [None]*len(adata_list)


    def projection(self, adata_list: List[sc.AnnData], **kwargs) -> List[sc.AnnData]:
        # Since cell's embedding using the shared feature (H&E image in this scenario)
        # has been generated in the `self.preproces()` stage, we directly retuen the 
        # `adata_list` back, since it already contains the "cell_latent" and 
        # the "niche_latent" matrix needed in the `self.mapping()` stage.
        return adata_list
