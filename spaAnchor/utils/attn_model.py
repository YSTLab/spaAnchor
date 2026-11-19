import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.sparse import coo_array, csr_array
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import softmax
from typing import List, Union, Optional, Tuple

from .projection import weighted_aggr, aggr


class cost_model(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim+1, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Softplus()
        )


    def forward(self, x, y):
        abs_diff = torch.abs(x - y)
        cross = torch.cosine_similarity(x, y).unsqueeze(1)
        return self.net(torch.hstack([abs_diff, cross]))


class attn_model(nn.Module):

    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.act = nn.Sigmoid()
        self.dec = nn.Sequential(
            nn.Linear(in_feat, 256, bias=False),
            self.act,
            nn.Linear(256, 512, bias=False),
            self.act,
            nn.Linear(512, out_feat, bias=False),
        )
        self.attn_src = nn.Sequential(
            nn.Linear(out_feat, 256, bias=False),
            nn.Linear(256, 1, bias=False)
        )
        self.attn_tar = nn.Sequential(
            nn.Linear(out_feat, 256, bias=False), 
            nn.Linear(256, 1, bias=False)
        )

        self.mse_loss = nn.MSELoss()
        self.dynamic_cost = cost_model(out_feat)

    
    def get_weight(self, y_s, y_t_pred, edge_index):

        aggr_attn_src = self.attn_src(y_s)
        aggr_attn_tar = self.attn_tar(y_t_pred)

        weight = self.act(aggr_attn_src.index_select(0, edge_index[1]) + \
            aggr_attn_tar.index_select(0, edge_index[0]))
        weight = softmax(weight, edge_index[0])

        return weight


    def predict(self, y_s, x_t, edge_index):

        y_t_pred = self.dec(x_t)
        weight = self.get_weight(y_s, y_t_pred, edge_index)
        y_t_aggr = weighted_aggr(y_s, weight, edge_index)

        if (self.training):
            return self.dynamic_cost(y_t_pred, y_t_aggr)
        else:
            return 0.5 * y_t_aggr + 0.5 * aggr(y_s, edge_index, mode="mean")


    def forward(self, x_s, y_s, x_t, edge_index):

        y_s_pred = self.dec(x_s)
        cost = self.predict(y_s, x_t, edge_index)

        return self.mse_loss(y_s_pred, y_s) + \
            torch.mean(cost) - 0.1 * torch.var(cost)


def load_adj(edge_index: Union[np.ndarray, List], shape: Tuple[int]) -> csr_array:
    return coo_array(
        (([1]*len(edge_index[0])), edge_index), 
        shape=shape, dtype=np.bool_
    ).tocsr()


def remove_isolated_node(
    x_s: np.ndarray, 
    y_s: np.ndarray, 
    x_t: np.ndarray, 
    edge_index: np.ndarray, 
    t_s_y: Optional[np.ndarray] = None,
    device: str = "cuda"
) -> List[torch.Tensor]:

    adj = load_adj(edge_index, (x_t.shape[0], x_s.shape[0]))
    nonzero_s = adj.sum(axis=0) != 0
    nonzero_t = adj.sum(axis=1) != 0
    x_s = x_s[nonzero_s]
    y_s = y_s[nonzero_s]
    x_t = x_t[nonzero_t]
    adj = load_adj(np.nonzero(adj[nonzero_t][:, nonzero_s]), (x_t.shape[0], x_s.shape[0]))

    x_s = torch.FloatTensor(x_s).to(device)
    y_s = torch.FloatTensor(y_s).to(device)
    x_t = torch.FloatTensor(x_t).to(device)
    edge_index = torch.LongTensor(np.nonzero(adj)).to(device)

    if (not t_s_y is None):
        t_s_y = torch.FloatTensor(t_s_y[nonzero_s])
    return x_s, y_s, x_t, t_s_y, edge_index


def attn_translation(
    source_x: np.ndarray, 
    source_y: np.ndarray, 
    target_x: np.ndarray, 
    raw_edge_index: np.ndarray, 
    t_source_y: Optional[np.ndarray] = None,
    device: str = "cuda", 
    epochs: int = 500, 
    lr: float = 1e-3,
    **kwargs
) -> np.ndarray:

    x_s, y_s, x_t, t_s_y, edge_index = remove_isolated_node(
        source_x, source_y, target_x, raw_edge_index, t_source_y, device
    )

    model = attn_model(x_s.shape[1], y_s.shape[1]).to(device)
    optim = Adam(model.parameters(), lr=lr)

    model.train()
    for _ in tqdm(range(epochs), desc="INFO: Learn aggregate attention score"):

        optim.zero_grad()
        loss = model(x_s, y_s, x_t, edge_index)
        loss.backward()
        optim.step()

    if (t_source_y is None):
        model.eval()
        with torch.no_grad():
            pred_t = model.predict(y_s, x_t, edge_index)

        return pred_t.cpu().numpy()
    else:
        out = torch.zeros((target_x.shape[0], t_s_y.shape[1]))

        model.eval()
        with torch.no_grad():
            y_t_pred = model.dec(x_t)
            weight = model.get_weight(y_s, y_t_pred, edge_index)
            pred_t = 0.5 * weighted_aggr(t_s_y, weight.cpu(), edge_index.cpu()) + \
                0.5 * aggr(t_s_y, edge_index.cpu(), mode="mean")

        adj = load_adj(raw_edge_index, (target_x.shape[0], source_x.shape[0]))
        nonzero_t = adj.sum(axis=1) != 0
        out[nonzero_t] = pred_t
        return out.numpy()
