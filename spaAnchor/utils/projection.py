import scanpy as sc
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from sklearn.decomposition import PCA
from harmony import harmonize
from typing import List, Optional, Literal

from .utils import get_X
from .torch_scatter import scatter_mean, scatter_add


def aggr(
    x: torch.FloatTensor, 
    edge_index: torch.LongTensor,
    out: Optional[torch.FloatTensor] = None,
    mode: Literal["mean", "sum"] = "sum"
) -> torch.FloatTensor:
    if ("sum" == mode):
        return scatter_add(
            x.index_select(0, edge_index[1]), 
            edge_index[0], dim=0, out=out
        ) 
    else:
        return scatter_mean(
            x.index_select(0, edge_index[1]), 
            edge_index[0], dim=0, out=out
        )
    

def weighted_aggr(
    x: torch.FloatTensor, 
    weights: torch.FloatTensor,
    edge_index: torch.LongTensor,
    out: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    return scatter_add(
        weights * x.index_select(0, edge_index[1]), 
        edge_index[0], dim=0, out=out
    )


def aggr_by_list(
    adata_list: List[sc.AnnData], 
    graphs: List[torch.LongTensor],
    use_rep: str = "cell_latent",
    k_hop: int = 2,
    output_rep: str = "niche_latent",
    aggr_opt: Literal["mean", "concat"] = "mean"
) -> List[sc.AnnData]:
    
    for i, (adata, edge_index) in enumerate(zip(adata_list, graphs)):

        x = [torch.FloatTensor(get_X(adata, use_rep))]              # 0 hop features
        for _ in range(k_hop): x.append(aggr(x[-1], edge_index))    # k hop features

        # Aggregate information from k-hop graph
        if ("mean" == aggr_opt): adata_list[i].obsm[output_rep] = torch.mean(torch.stack(x), dim=0).numpy()
        else: adata_list[i].obsm[output_rep] = torch.hstack(x).numpy()

    return adata_list


# Reduce dimension by harmony
def pca_reduce(
    adata_list: List[sc.AnnData],
    use_n_pcs: int = 20
) -> List[sc.AnnData]:

    adatas = sc.concat(adata_list, label="batch")
    adatas.obsm["cell_latent"] = PCA(
        n_components=min(adatas.shape[1], use_n_pcs), random_state=0
    ).fit_transform(get_X(adatas))

    return [adatas[i == adatas.obs["batch"]] for i in adatas.obs["batch"].unique()]


# Reduce dimension by harmony
def harmony_reduce(
    adata_list: List[sc.AnnData],
    use_n_pcs: int = 20
) -> List[sc.AnnData]:

    adatas = sc.concat(adata_list, label="batch")
    top_pcs = PCA(n_components=use_n_pcs, random_state=0).fit_transform(get_X(adatas))
    adatas.obsm["cell_latent"] = harmonize(
        top_pcs, 
        adatas.obs, 
        batch_key="batch", 
        verbose=False,
        use_gpu=False   # Stabilize result
    )

    return [adatas[i == adatas.obs["batch"]] for i in adatas.obs["batch"].unique()]


def clean_cuda_cache():
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()


@torch.inference_mode()
def get_cell_embedding(
    adata: sc.AnnData, 
    model: torch.nn.Module,
    preprocess_func: Compose,
    spatial_key: str = "spatial",
    library_id: Optional[str] = None, 
    img_key: str = "hires",
    scale_factor: Optional[float] = None,
    batch_size: Optional[int] = 1024,
    device: str = "cuda" if (torch.cuda.is_available()) else "cpu",
) -> List[torch.FloatTensor]:

    if (library_id):
        _, spatial_data = sc.pl._tools.scatterplots._check_spatial_data(adata.uns, library_id)
    else:
        _, spatial_data = sc.pl._tools.scatterplots._check_spatial_data(adata.uns, sc._utils._empty)
    img, img_key = sc.pl._tools.scatterplots._check_img(spatial_data, img=None, img_key=img_key)
    scale_factor = sc.pl._tools.scatterplots._check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )

    img = preprocess_func(img)
    img = F.pad(img, pad=(256//2,)*4, mode="constant", value=0.)

    # Get the left-top coordinates for each cell's image
    coords = adata.obsm[spatial_key] * scale_factor
    xs = torch.from_numpy(np.floor(coords[:, 0]).astype(int))
    ys = torch.from_numpy(np.floor(coords[:, 1]).astype(int))
    
    # Generate pixel indices for each cell's image
    offset_indices = torch.arange(256).unsqueeze(0)
    x_indices = (xs.unsqueeze(1) + offset_indices).unsqueeze(1) # (num_cells, 1, W)
    y_indices = (ys.unsqueeze(1) + offset_indices).unsqueeze(2) # (num_cells, H, 1)

    # Generate image embedding
    cell_num = coords.shape[0]
    niche_embed_list = []
    cell_embed_list = []
    for i in tqdm(range(np.ceil(cell_num/batch_size).astype(int))):
        idx = list(range(i*batch_size, min(cell_num, (i+1)*batch_size)))
        niche_embed_list.append(model(img[
            :, y_indices[idx], x_indices[idx]
        ].permute(1, 0, 2, 3).to(device)).squeeze().cpu())
        cell_embed_list.append(model.patch_embed(img[
            :, y_indices[idx][:,120:136], x_indices[idx][:,:,120:136]
        ].permute(1, 0, 2, 3).to(device)).squeeze().cpu())
    clean_cuda_cache()
    
    return torch.vstack(niche_embed_list).numpy(), torch.vstack(cell_embed_list).numpy()
