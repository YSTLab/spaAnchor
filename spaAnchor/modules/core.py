import scanpy as sc
import time
import torch
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_undirected, add_self_loops, coalesce
from typing import List, Dict, Optional, Literal, Union, Callable
from numbers import Number

from ..utils import seed_everything, get_X
from ..utils.projection import pca_reduce, harmony_reduce, aggr_by_list
from ..utils.mapping import get_mapping, row_wise_intersect
from ..utils.attn_model import attn_translation


class HANN():

    def __init__(
        self,
        filter_cell: bool = True,
        preprocess: bool = True,
        n_hvg: int = 8e3,
        spatial_key: str = "spatial",
        reduction_opt: Literal["pca", "harmony", "outside"] = "pca",
        reduction_dim: int = 30,
        niche_threshold: Optional[int] = None,
        cell_threshold: Optional[int] = None,
        dis_threshold: float = 0.75,
        scale: Union[bool, Number] = False,
        imbalanced: bool = False,
        feature_key: Optional[Union[str, List[str]]] = None,
        regressor_key: Optional[Dict[str, Callable]] = None, 
        regression_feat: Literal["X_raw", "X_pp"] = "X_raw",
        emb_key: str = "cell_latent",
        predict_full: bool = False,
        device: str = "cuda" if (torch.cuda.is_available()) else "cpu"
    ):
        self.filter_cell = filter_cell
        self.pp = preprocess
        self.n_hvg = n_hvg
        self.spatial_key = spatial_key
        self.reduction_opt = reduction_opt
        self.reduction_dim = reduction_dim
        self.niche_threshold = niche_threshold
        self.cell_threshold = cell_threshold
        self.dis_threshold = dis_threshold
        self.scale = scale
        self.imbalanced = imbalanced
        self.regression_feat = regression_feat
        self.emb_key = emb_key
        self.predict_full = predict_full
        self.device = device

        # Set feature key
        if (feature_key is None):
            self.feature_key = ["X"]
        elif (not isinstance(feature_key, list)):
            self.feature_key = [feature_key]
        else:
            feature_key = list(set(feature_key))    # Filter replicated features
            self.feature_key = feature_key

        # Set regressor for each kinds of feature
        self.regressor_key = {}
        for feat in self.feature_key:
            if ((isinstance(regressor_key, dict)) and regressor_key.get(feat) is not None):
                self.regressor_key[feat] = regressor_key[feat]
            else:
                self.regressor_key[feat] = self.feature_regression


    def build_graph(self, coords: np.ndarray, **kwargs) -> torch.LongTensor:

        indptr, indices = Delaunay(coords).vertex_neighbor_vertices
        edge_index = torch.LongTensor(np.array(np.nonzero(csr_matrix(
            (np.ones_like(indices), indices, indptr), 
            shape=(coords.shape[0], coords.shape[0])
        ))))

        edge_index = coalesce(add_self_loops(to_undirected(edge_index))[0])
        print(f"INFO: Generate {int(edge_index.shape[1]/2):,} edges, average {edge_index.shape[1]/coords.shape[0]/2:.3f} edges per cell.")
        return edge_index


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
                sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="seurat_v3", subset=True)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)

            graph_list.append(self.build_graph(adata.obsm[self.spatial_key], **kwargs))

        return graph_list


    def projection_cell(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ):
        if ("pca" == self.reduction_opt): 
            latent_adata_list = pca_reduce(adata_list, self.reduction_dim)
        elif ("harmony" == self.reduction_opt): 
            latent_adata_list = harmony_reduce(adata_list, self.reduction_dim)
        elif ("outside" == self.reduction_opt): 
            latent_adata_list = adata_list
        else: 
            raise ValueError(f"ERROR: {self.reduction_opt} dimension reduction method not supported yet.")

        return latent_adata_list    


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


    def mapping(
        self,
        adata_list: List[sc.AnnData],
        **kwargs
    ) -> Dict[int, Dict[int, np.ndarray]]:

        obs_shape = adata_list[0].shape[0] + adata_list[1].shape[0]
        cell_threshold = max(50, int(obs_shape*1.5e-4)) if (None == self.cell_threshold) else self.cell_threshold
        niche_threshold = max(300, int(obs_shape*3e-4)) if (None == self.niche_threshold) else self.niche_threshold

        cell_embed_0 = adata_list[0].obsm["cell_latent"]
        cell_embed_1 = adata_list[1].obsm["cell_latent"]
        niche_embed_0 = adata_list[0].obsm["niche_latent"]
        niche_embed_1 = adata_list[1].obsm["niche_latent"]

        cell_wise_mnn = get_mapping(
            cell_embed_0, 
            cell_embed_1, 
            cell_threshold, 
            self.dis_threshold, 
            self.imbalanced
        )   # Find cell-wise MNN pairs
        niche_wise_mnn = get_mapping(
            niche_embed_0, 
            niche_embed_1, 
            niche_threshold, 
            self.dis_threshold, 
            self.imbalanced
        )   # Find niche-wise MNN pairs

        # Find final mnn pairs:
        # Return shape is (0, 1) => from slice 0 to slice 1
        # To use the pairs for slice 1 => do transpose
        # ex. `mnn_pairs[::-1]`
        mnn_pairs = row_wise_intersect(
            cell_wise_mnn, 
            niche_wise_mnn,
            (cell_embed_0.shape[0], cell_embed_1.shape[0])
        )

        return mnn_pairs


    def feature_regression(
        self,
        source: np.ndarray, 
        target: np.ndarray, 
        predict: Optional[np.ndarray] = None,
        top_pcs: int = 50, 
        n_jobs: int = -1,
        **kwargs
    ) -> np.ndarray:
        
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        predict = predict if (isinstance(predict, np.ndarray)) else source
        assert (source.shape[1] == predict.shape[1]), "ERROR: source and predict must have same feature size."

        preprocessor = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(top_pcs, source.shape[1]), random_state=0))
        ])

        pp_source = preprocessor.fit_transform(source)
        pp_predict = preprocessor.transform(predict)

        def _fit_predict(y_chunk: np.ndarray) -> np.ndarray:
            model = Ridge(random_state=0)
            res = model.fit(pp_source, y_chunk).predict(pp_predict)
            return res

        if (target.shape[1] > 1e3):
            from tqdm import tqdm
            from joblib import Parallel, delayed
            return np.hstack(Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_fit_predict)(target[:, 50*i: min(50*(i+1), target.shape[1])])
                for i in tqdm(
                    range(np.ceil(target.shape[1]/50).astype(int)),
                    desc="INFO: Large panel regression"
                )
            ))
        else:
            return _fit_predict(target)
        

    def translate(
        self,
        source_emb: sc.AnnData,
        source_feat: np.ndarray,
        target_emb: sc.AnnData,
        target_feat: np.ndarray,
        hnn_pairs: np.ndarray,
        index: List[str],
        columns: List[str],
        regression_func: Callable = "self.feature_regression",
        **kwargs
    ) -> np.ndarray:
        
        # Attention transfer feature from source -> target
        transfer_result = attn_translation(
            source_x=source_emb, 
            source_y=source_feat, 
            target_x=target_emb, 
            raw_edge_index=hnn_pairs,
            **kwargs
        )

        # Global regression
        regress_result = regression_func(
            source=target_feat[np.unique(hnn_pairs[0])], 
            target=transfer_result, 
            predict=target_feat,
            **kwargs
        )

        # relu for all positive target 
        if (np.all(source_feat >= 0)):
            regress_result = np.maximum(regress_result, 0)

        return pd.DataFrame(
            regress_result, 
            index=index, 
            columns=columns
        )


    def predict_missing_feat(
        self,
        source_adata: sc.AnnData,
        target_adata: sc.AnnData,
        hnn_pairs: np.ndarray,
        feat_key: str,
        **kwargs
    ) -> pd.DataFrame:
        
        if (feat_key not in source_adata.obsm): 
            result = None   # If nothing to predict, return None
        else:
            source_emb = get_X(source_adata, self.emb_key)
            target_emb = get_X(target_adata, self.emb_key)
            source_df = get_X(source_adata, feat_key)
            if ("X" == feat_key and not self.predict_full):
                # Only predict source slice's unique feature when predicting feature from `adata.X`.
                # (this is the legacy spaAnchor behavior)
                # If you want to predict the full panel for source slice,
                # please save expression in adata.obsm and include its key in `feature_key`
                # and set `predict_full = True`.
                feat_col = np.setdiff1d(source_adata.var_names, target_adata.var_names)
                source_df = source_df[feat_col]
            target_feat = get_X(target_adata, self.regression_feat).to_numpy()
            source_feat = source_df.to_numpy()

            # Scale data if feature scale is extremely big
            if (
                not isinstance(self.scale, bool) and 
                isinstance(self.scale, Number)
            ): 
                scale = self.scale
            else: 
                scale = 1e3 if (self.scale) else 1

            regression_func = self.regressor_key[feat_key]

            result = self.translate(
                source_emb=source_emb, 
                source_feat=source_feat/scale, 
                target_emb=target_emb, 
                target_feat=target_feat, 
                hnn_pairs=hnn_pairs, 
                index=target_adata.obs_names, 
                columns=source_df.columns,
                regression_func=regression_func,
                **kwargs
            )*scale 

        return result


    def forward(
        self,
        adata_list: List[sc.AnnData],
        tgt_id: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        seed_everything()
        kwargs.setdefault("device", self.device)

        if (tgt_id is None): 
            self.tgt_id = np.ones((len(adata_list), len(adata_list)), dtype=np.bool_)
        else: 
            self.tgt_id = tgt_id
        
        transfer_result = {i: {} for i in range(len(adata_list))}

        # Preprocess
        start_time = time.time()
        for i, adata in enumerate(adata_list):
            if (0 == adata.shape[1]): 
                raise ValueError(
                    "ERROR: Please store the shared features at `sc.AnnData.X`"
                    " or overload the `self.preprocess()` function"
                    " to fit your own dataset."
                    " Please read spaAnchor's document"
                    " (https://spaanchor.readthedocs.io/en/latest/)"
                    " if you don't know how to organize the input data."
                )
            adata.obs_names += f"-{i}"
            if ("X_raw" not in adata.obsm):
                adata.obsm["X_raw"] = adata.to_df()
            if (("X" in self.feature_key) and ("X" not in adata.obsm)): 
                adata.obsm["X"] = adata.to_df()
        graphs = self.preprocess(adata_list=adata_list, **kwargs)
        for i, adata in enumerate(adata_list):
            if ("X_pp" not in adata.obsm):
                adata.obsm["X_pp"] = adata.to_df()
        print(f"INFO: Finish preprocessing. ({time.time() - start_time:.3f}s)")

        for i in range(len(adata_list)):
            for j in range(i+1, len(adata_list)):
                if (self.tgt_id[i][j] or self.tgt_id[j][i]):
                    start_time = time.time()

                    cur_adata_list = [adata_list[i], adata_list[j]]
                    graphs_list = [graphs[i], graphs[j]]

                    # Embedding generation
                    cur_adata_list = self.projection(adata_list=cur_adata_list, graphs=graphs_list, **kwargs)

                    # Identify mnn pairs
                    hnn_pairs = self.mapping(adata_list=cur_adata_list, **kwargs)

                    print(f"INFO: Finish HNN pairs finding for Slice {i} <-> Slice {j}. ({time.time() - start_time:.3f}s)")

                    # Feature translation
                    for feat in self.feature_key:
                        for target, source, mapping in [
                            (i, j, hnn_pairs),      # Slice i get feature from slice j
                            (j, i, hnn_pairs[::-1]) # Slice j get feature from slice i
                        ]:
                            if (
                                (self.tgt_id[target][source]) and
                                (feat in adata_list[source].obsm)
                            ):

                                print(f"INFO: From Slice {source} -> Slice {target} translating feature: {feat}")
                                transfer_result[target][f"{feat}_from_{source}"] = self.predict_missing_feat(
                                    source_adata=adata_list[source].copy(), 
                                    target_adata=adata_list[target].copy(), 
                                    hnn_pairs=mapping, 
                                    feat_key=feat,
                                    **kwargs
                                )

        return transfer_result


    def __call__(self, adata_list, tgt_id=None, **kwargs) -> Dict[int, Dict[str, pd.DataFrame]]:
        return self.forward(adata_list, tgt_id, **kwargs)

