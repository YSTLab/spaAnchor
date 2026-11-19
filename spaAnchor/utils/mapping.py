import numpy as np
import faiss
from scipy.sparse import coo_array
from typing import List


FAISS_WARNING = """
WARNING: Failed to run faiss-gpu, change to cpu version.
This process will be intolerably slow, we highly recommand 
you to install and use the gpu version of faiss.
You can install faiss-gpu by `conda install -c pytorch faiss-gpu`
"""


def row_wise_intersect(
    m1: np.ndarray, 
    m2: np.ndarray, 
    shape: tuple[int], 
) -> np.ndarray:
    return np.array(np.nonzero(
        coo_array(([1]*len(m1[0]), m1), shape=shape, dtype=np.bool_) * 
        coo_array(([1]*len(m2[0]), m2), shape=shape, dtype=np.bool_)
    ), dtype=np.int64)


def find_similar_pairs(
    source: np.ndarray, 
    target: np.ndarray, 
    top_k: int = 1, 
    dis_threshold: float = 0.75,
) -> List[np.ndarray]:
    
    index = faiss.IndexFlatL2(source.shape[1])
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    faiss.normalize_L2(source)
    faiss.normalize_L2(target) 

    # Find similar node
    try:
        index_gpu = faiss.index_cpu_to_all_gpus(index)
        index_gpu.add(target)
        distance, indices = index_gpu.search(source, top_k) 
    except Exception as e:
        print(f"Error message: {e}")
        print(FAISS_WARNING)
        index.add(target)
        distance, indices = index.search(source, top_k) 

    # generate node pairs
    mask = distance.reshape(-1) <= distance.reshape(-1).max() * dis_threshold
    return [
        np.array([np.arange(source.shape[0])]*top_k).T.reshape(-1)[mask], 
        indices.reshape(-1)[mask]
    ]


def get_mapping(
    data_i: np.ndarray,
    data_j: np.ndarray,
    top_k: int,
    dis_threshold: float = 0.75,
    unbalanced: bool = False
) -> List[np.ndarray]:
    
    if (unbalanced):
        j2i_topk = top_k + ((max(1, int((data_i.shape[0]/data_j.shape[0])))-1)*100)
        i2j_topk = top_k + ((max(1, int((data_j.shape[0]/data_i.shape[0])))-1)*100)
    else: i2j_topk, j2i_topk = [top_k]*2


    i2j_pairs = find_similar_pairs(data_i, data_j, i2j_topk, dis_threshold)
    j2i_pairs = find_similar_pairs(data_j, data_i, j2i_topk, dis_threshold)

    final_mnn_pairs = row_wise_intersect(
        i2j_pairs, [j2i_pairs[1], j2i_pairs[0]], 
        (data_i.shape[0], data_j.shape[0])
    )

    return final_mnn_pairs
