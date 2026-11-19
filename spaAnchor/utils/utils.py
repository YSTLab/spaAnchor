import scanpy as sc
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import issparse
from typing import List, Dict, Optional, Iterable, Union
import re


def seed_everything(seed: int = 0) -> None:
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_X(adata: sc.AnnData, use_rep: Optional[str] = None) -> np.ndarray:
    load_func = lambda data : data.toarray() if (issparse(data)) else data
    if (use_rep is not None):
        rep = adata.obsm[use_rep]
        return load_func(rep)
    else:
        return load_func(adata.X)


def assign_genes(
    panel_list: List[str], 
    shared_panel_num: int,
    panels_per_slice: Union[List[int], int],
    seed: Optional[int] = 0
) -> Dict[str, List[str]]:
    seed_everything(seed)

    panels_per_slice = [(len(panel_list) - shared_panel_num) // panels_per_slice] * panels_per_slice \
        if (isinstance(panels_per_slice, int)) else panels_per_slice

    panel_dict = {}
    panel_dict["shared_genes"] = np.random.choice(panel_list, shared_panel_num, replace=False).tolist()
    panel_list = np.setdiff1d(panel_list, panel_dict["shared_genes"], assume_unique=True)
    for i, gene_num in enumerate(panels_per_slice):
        panel_dict[f"slice_{i}"] = np.random.choice(panel_list, gene_num, replace=False).tolist()
        panel_list = np.setdiff1d(panel_list, panel_dict[f"slice_{i}"], assume_unique=True)

    return panel_dict


def load_genes(file_name: str) -> Dict[str, List[str]]:
    with open(file_name, "rb") as f:
        gene_dict = pickle.load(f)
    return gene_dict


# https://stackoverflow.com/a/78046815
def numeric_sort(input: Iterable[str]) -> List[str]:
    def repl(num):
        return f"{int(num[0]):010d}"
    return sorted(input, key=lambda i: re.sub(r"(\d+)", repl, i))


# https://github.com/gao-lab/GLUE/blob/master/scglue/num.py#L275
def tfidf(X: np.ndarray) -> np.ndarray:

    idf = X.shape[0] / (X.sum(axis=0)+1e-8)
    tf = X / (X.sum(axis=1, keepdims=True)+1e-8)
    return tf * idf


# https://github.com/gao-lab/GLUE/blob/master/scglue/data.py#L50
def lsi(X: np.ndarray, n_components: int = 20) -> np.ndarray:

    # get variable peaks
    adata = sc.AnnData(X)
    sc.pp.highly_variable_genes(adata, n_top_genes=30000, flavor="seurat_v3")
    X = get_X(adata[:, adata.var["highly_variable"]])
    nonzero_feat = X.sum(axis=0) != 0
    nonzero_elem = X.sum(axis=1) != 0
    out = np.zeros((X.shape[0], n_components))
    X = X[nonzero_elem][:, nonzero_feat]

    # lsi decomposition
    X_norm = normalize(tfidf(X), norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = randomized_svd(X_norm, n_components, random_state=0, n_iter=15)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    out[nonzero_elem] = X_lsi
    return out
