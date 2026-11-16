import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_STPGNN_ROOT = REPO_ROOT / "third_party" / "STPGNN_official"
sys.path.insert(0, str(OFFICIAL_STPGNN_ROOT))
from model import STPGNN as OfficialSTPGNN  # noqa: E402


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_npz_traffic(data_file, data_key=None, target_dim=0):
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())
    arr = None
    used_key = data_key

    if data_key is not None:
        if data_key not in npz:
            raise KeyError(f"data_key={data_key!r} not found. Available keys: {keys}")
        arr = npz[data_key]
    else:
        for key in keys:
            candidate = npz[key]
            if isinstance(candidate, np.ndarray) and candidate.ndim == 3:
                arr = candidate
                used_key = key
                break
        if arr is None:
            for key in keys:
                candidate = npz[key]
                if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                    arr = candidate
                    used_key = key
                    break
        if arr is None:
            raise ValueError(f"No usable 2D/3D ndarray found in {path}. Available keys: {keys}")

    arr = np.asarray(arr)
    if arr.ndim == 2:
        values = arr.astype(np.float32)
    elif arr.ndim == 3:
        if target_dim >= arr.shape[-1]:
            raise ValueError(f"target_dim={target_dim} out of range for data shape {arr.shape}")
        values = arr[..., target_dim].astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D traffic array, got shape {arr.shape}")

    return values, {"file": str(path), "key": used_key, "shape": list(arr.shape)}


def make_demo_traffic(num_steps=1200, num_nodes=40, seed=2026):
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)
    daily = np.sin(2 * np.pi * t / 288.0)
    weekly = np.sin(2 * np.pi * t / (288.0 * 7.0))
    base = 50 + 10 * daily[:, None] + 4 * weekly[:, None]
    node_bias = rng.normal(0, 5, size=(1, num_nodes))
    node_scale = rng.uniform(0.8, 1.3, size=(1, num_nodes))
    noise = rng.normal(0, 2, size=(num_steps, num_nodes))
    values = base * node_scale + node_bias + noise
    hard_nodes = rng.choice(num_nodes, max(1, num_nodes // 5), replace=False)
    values[:, hard_nodes] += rng.normal(0, 5, size=(num_steps, len(hard_nodes)))
    return np.maximum(values, 0).astype(np.float32)


def compute_splits(num_steps, input_len, pred_len, split_ratio):
    train_ratio, val_ratio, test_ratio = split_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("--split_ratio must sum to 1.0")

    train_end = int(num_steps * train_ratio)
    val_end = int(num_steps * (train_ratio + val_ratio))
    max_start = num_steps - input_len - pred_len
    if max_start <= 0:
        raise ValueError(f"Data too short: T={num_steps}, input_len={input_len}, pred_len={pred_len}")

    train_starts = np.arange(0, max(0, train_end - input_len - pred_len + 1))
    val_start = max(0, train_end - input_len)
    val_starts = np.arange(val_start, max(val_start, val_end - input_len - pred_len + 1))
    test_start = max(0, val_end - input_len)
    test_starts = np.arange(test_start, max(test_start, max_start + 1))

    if len(train_starts) == 0 or len(val_starts) == 0 or len(test_starts) == 0:
        raise ValueError("One split has no samples. Adjust split/input/pred lengths.")

    return {
        "train_end": train_end,
        "val_end": val_end,
        "train_starts": train_starts,
        "val_starts": val_starts,
        "test_starts": test_starts,
    }


def make_observation_mask(values, train_end, sparse_ratio, low_keep_prob, dense_keep_prob, seed):
    rng = np.random.default_rng(seed)
    num_steps, num_nodes = values.shape
    finite_mask = np.isfinite(values)
    num_sparse = max(1, int(round(num_nodes * sparse_ratio)))
    sparse_nodes = np.sort(rng.choice(num_nodes, num_sparse, replace=False))

    keep_probs = np.full((num_nodes,), dense_keep_prob, dtype=np.float32)
    keep_probs[sparse_nodes] = low_keep_prob
    random_keep = rng.random((num_steps, num_nodes)) < keep_probs[None, :]
    obs_mask = finite_mask & random_keep

    for i in range(num_nodes):
        if obs_mask[:train_end, i].sum() == 0:
            valid_train_times = np.flatnonzero(finite_mask[:train_end, i])
            if len(valid_train_times) > 0:
                obs_mask[valid_train_times[0], i] = True

    return obs_mask.astype(bool), sparse_nodes


def load_adjacency_file(adj_file, num_nodes, adj_key=None):
    path = Path(adj_file)
    if not path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {path}")

    if path.suffix.lower() == ".npy":
        adj = np.load(path)
    elif path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        if adj_key is not None:
            if adj_key not in obj:
                raise KeyError(f"adj_key={adj_key!r} not found. Available keys: {list(obj.keys())}")
            adj = obj[adj_key]
        else:
            adj = None
            for key in obj.keys():
                candidate = obj[key]
                if isinstance(candidate, np.ndarray) and candidate.ndim == 2 and candidate.shape[0] == candidate.shape[1]:
                    adj = candidate
                    adj_key = key
                    break
            if adj is None:
                raise ValueError(f"No square 2D adjacency array found in {path}. Available keys: {list(obj.keys())}")
    elif path.suffix.lower() in [".csv", ".txt"]:
        adj = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    else:
        raise ValueError(f"Unsupported adjacency file type: {path.suffix}")

    adj = np.asarray(adj, dtype=np.float64)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.maximum(adj, 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj, {"source": str(path), "key": adj_key, "kind": "file"}


def build_correlation_knn_adjacency(values, obs_mask, train_end, k=8, threshold=0.0):
    train = np.asarray(values[:train_end], dtype=np.float64)
    train_mask = obs_mask[:train_end] & np.isfinite(train)
    filled = train.copy()
    for i in range(filled.shape[1]):
        valid = train_mask[:, i]
        if valid.sum() == 0:
            filled[:, i] = 0.0
        else:
            mean_i = float(filled[valid, i].mean())
            filled[~valid, i] = mean_i
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    std = filled.std(axis=0)
    keep = std > 1e-8
    standardized = np.zeros_like(filled)
    standardized[:, keep] = (filled[:, keep] - filled[:, keep].mean(axis=0)) / std[keep]
    corr = np.corrcoef(standardized.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    n = corr.shape[0]
    k = max(1, min(int(k), n - 1))
    adj = np.zeros_like(corr)
    for i in range(n):
        order = np.argsort(corr[i], kind="mergesort")[::-1]
        chosen = [j for j in order if corr[i, j] >= threshold and j != i][:k]
        if len(chosen) == 0:
            chosen = [j for j in order if j != i][:k]
        adj[i, chosen] = corr[i, chosen]
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj, {"source": "train_correlation_knn", "graph_k": int(k), "threshold": float(threshold), "kind": "generated"}


def select_topology_stress_nodes(base_adj, obs_coverage, weak_ratio, mode, seed):
    base_adj = np.asarray(base_adj, dtype=np.float64)
    n = base_adj.shape[0]
    weak_count = max(1, int(round(n * weak_ratio))) if weak_ratio > 0 else 0
    if weak_count <= 0:
        return np.array([], dtype=np.int64), {
            "stress_mode": mode,
            "weak_count": 0,
            "base_coverage_mean": 0.0,
        }

    if mode == "coverage":
        base_coverage, base_support = compute_spatial_coverage(base_adj, obs_coverage)
        order = np.argsort(base_coverage, kind="mergesort")
        weak_nodes = np.sort(order[:weak_count])
        return weak_nodes.astype(np.int64), {
            "stress_mode": mode,
            "weak_count": int(weak_count),
            "base_coverage_mean": float(base_coverage.mean()),
            "base_coverage_weak_mean": float(base_coverage[weak_nodes].mean()),
            "base_coverage_nonweak_mean": float(base_coverage[~np.isin(np.arange(n), weak_nodes)].mean()),
            "base_support_weak_mean": float(base_support[weak_nodes].mean()),
        }

    if mode == "random":
        rng = np.random.default_rng(seed + 7301)
        weak_nodes = np.sort(rng.choice(n, weak_count, replace=False))
        base_coverage, base_support = compute_spatial_coverage(base_adj, obs_coverage)
        return weak_nodes.astype(np.int64), {
            "stress_mode": mode,
            "weak_count": int(weak_count),
            "base_coverage_mean": float(base_coverage.mean()),
            "base_coverage_weak_mean": float(base_coverage[weak_nodes].mean()),
            "base_coverage_nonweak_mean": float(base_coverage[~np.isin(np.arange(n), weak_nodes)].mean()),
            "base_support_weak_mean": float(base_support[weak_nodes].mean()),
        }

    if mode == "none":
        return np.array([], dtype=np.int64), {
            "stress_mode": mode,
            "weak_count": 0,
            "base_coverage_mean": float(compute_spatial_coverage(base_adj, obs_coverage)[0].mean()),
        }

    raise ValueError(f"Unknown topology_stress_mode: {mode}")


def induce_topology_weakness(adj, weak_nodes, weak_keep_degree, weak_edge_scale):
    adj = np.asarray(adj, dtype=np.float64).copy()
    n = adj.shape[0]
    weak_nodes = np.asarray(weak_nodes, dtype=np.int64)
    keep_degree = max(0, int(weak_keep_degree))
    for i in weak_nodes:
        row = adj[i].copy()
        order = np.argsort(row, kind="mergesort")[::-1]
        keep = [j for j in order if row[j] > 0 and j != i][:keep_degree]
        new_row = np.zeros(n, dtype=np.float64)
        if len(keep) > 0:
            new_row[keep] = row[keep] * float(weak_edge_scale)
        adj[i] = new_row
    np.fill_diagonal(adj, 0.0)
    adj = np.maximum(adj, 0.0)
    return adj, weak_nodes


def compute_spatial_coverage(adj, obs_coverage, eps=1e-8):
    adj = np.asarray(adj, dtype=np.float64)
    obs_coverage = np.asarray(obs_coverage, dtype=np.float64)
    support = adj @ obs_coverage
    denom = float(np.max(support)) + eps
    return (support / denom).astype(np.float64), support.astype(np.float64)


def prepare_topology(args, values_raw, obs_mask, train_end, seed, obs_coverage):
    num_nodes = values_raw.shape[1]
    if args.adj_file:
        base_adj, meta = load_adjacency_file(args.adj_file, num_nodes, args.adj_key)
    else:
        base_adj, meta = build_correlation_knn_adjacency(
            values_raw,
            obs_mask,
            train_end,
            k=args.graph_k,
            threshold=args.graph_threshold,
        )

    weak_nodes, stress_meta = select_topology_stress_nodes(
        base_adj,
        weak_ratio=args.topology_weak_ratio,
        mode=args.topology_stress_mode,
        seed=seed,
        obs_coverage=obs_coverage,
    )
    adj, weak_nodes = induce_topology_weakness(
        base_adj,
        weak_nodes=weak_nodes,
        weak_keep_degree=args.weak_keep_degree,
        weak_edge_scale=args.weak_edge_scale,
    )
    row_support = adj.sum(axis=1)
    max_support = float(row_support.max()) if row_support.size else 0.0
    if max_support > 1e-8:
        model_adj = adj / max_support
    else:
        model_adj = adj

    weak_mask = np.zeros(num_nodes, dtype=bool)
    weak_mask[weak_nodes] = True
    meta.update({
        "topology_stress_mode": args.topology_stress_mode,
        "topology_weak_ratio": float(args.topology_weak_ratio),
        "weak_keep_degree": int(args.weak_keep_degree),
        "weak_edge_scale": float(args.weak_edge_scale),
        "weak_node_count": int(len(weak_nodes)),
        "weak_nodes": [int(x) for x in weak_nodes.tolist()],
        "row_support_mean": float(row_support.mean()) if row_support.size else 0.0,
        "row_support_max": max_support,
    })
    meta.update(stress_meta)
    return adj, model_adj.astype(np.float32), row_support, weak_mask, meta


def compute_obs_coverage(obs_mask, train_end):
    return obs_mask[:train_end].mean(axis=0).astype(np.float64)


def quantile_groups(coverage, rho=0.2):
    coverage = np.asarray(coverage, dtype=np.float64)
    n = len(coverage)
    k = max(1, int(round(n * rho)))
    order = np.argsort(coverage, kind="mergesort")
    low_idx = order[:k]
    high_idx = order[-k:]
    low = np.zeros(n, dtype=bool)
    high = np.zeros(n, dtype=bool)
    low[low_idx] = True
    high[high_idx] = True
    return low, high, {
        "rho": float(rho),
        "low_size": int(low.sum()),
        "high_size": int(high.sum()),
        "low_threshold_quantile": float(np.quantile(coverage, rho)),
        "high_threshold_quantile": float(np.quantile(coverage, 1.0 - rho)),
    }


def fit_standard_scaler(values, train_end, obs_mask):
    train_values = values[:train_end]
    train_mask = obs_mask[:train_end] & np.isfinite(train_values)
    if train_mask.sum() == 0:
        raise ValueError("No valid training observations available for scaler.")
    mean = float(train_values[train_mask].mean())
    std = float(train_values[train_mask].std())
    if std < 1e-6:
        std = 1.0
    return mean, std


class TrafficWindowDataset(Dataset):
    def __init__(self, values_norm, obs_mask, starts, input_len, pred_len, tod_slots=288):
        self.values = values_norm.astype(np.float32)
        self.obs_mask = obs_mask.astype(bool)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.tod_slots = int(tod_slots)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = int(self.starts[index])
        x = self.values[s : s + self.input_len]
        x_mask = self.obs_mask[s : s + self.input_len]
        y = self.values[s + self.input_len : s + self.input_len + self.pred_len]
        y_mask = self.obs_mask[s + self.input_len : s + self.input_len + self.pred_len]
        x_filled = np.where(x_mask, x, 0.0).astype(np.float32)
        last_input_t = s + self.input_len - 1
        tod = last_input_t % self.tod_slots
        dow = (last_input_t // self.tod_slots) % 7
        return {
            "x": torch.from_numpy(x_filled),
            "y": torch.from_numpy(y.astype(np.float32)),
            "y_mask": torch.from_numpy(y_mask.astype(np.float32)),
            "tod": torch.tensor(tod, dtype=torch.long),
            "dow": torch.tensor(dow, dtype=torch.long),
        }


class OfficialSTPGNNBackbone(nn.Module):
    """Adapter around the official STPGNN implementation.

    The official model consumes tensors shaped as (B, F, N, T) and returns
    (B, horizon, N, 1). The CAFR framework in this package uses
    (B, T, N) input windows and (B, horizon, N) predictions, so this adapter
    only performs shape conversion and optional support-mask injection.
    """
    def __init__(
        self,
        num_nodes,
        pred_len,
        device,
        topk=35,
        nhid=32,
        dims=32,
        blocks=4,
        layers=2,
        dropout=0.1,
        order=2,
        tod_slots=288,
        support_mask=None,
    ):
        super().__init__()
        self.tod_slots = int(tod_slots)
        self.core = OfficialSTPGNN(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            topk=max(1, min(int(topk), int(num_nodes))),
            out_dim=int(pred_len),
            residual_channels=int(nhid),
            dilation_channels=int(nhid),
            end_channels=int(nhid) * 16,
            kernel_size=2,
            blocks=int(blocks),
            layers=int(layers),
            days=int(tod_slots),
            dims=int(dims),
            order=int(order),
            in_dim=1,
            normalization="batch",
        )
        if support_mask is not None and hasattr(self.core, "external_support_mask"):
            mask = torch.as_tensor(support_mask, dtype=torch.float32, device=device)
            if mask.shape != (num_nodes, num_nodes):
                raise ValueError(f"support_mask shape {tuple(mask.shape)} does not match num_nodes={num_nodes}")
            self.core.external_support_mask.data.copy_(mask)

    def forward(self, x, tod, dow=None):
        # x: (B, T, N), normalized traffic values.
        del dow
        x4 = x.permute(0, 2, 1).unsqueeze(1).contiguous()  # (B, 1, N, T)
        if x4.size(-1) < self.core.receptive_field:
            x4 = nn.functional.pad(x4, (self.core.receptive_field - x4.size(-1), 0, 0, 0))
        ind = torch.remainder(tod, self.tod_slots).long().to(x4.device)
        out = self.core(x4, ind)
        return out.squeeze(-1).contiguous()


class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FormalSTID(nn.Module):
    """STID-style backbone: history embedding + node/time identity embeddings + residual MLP.

    This is the formal backbone used in exp5. It keeps the key STID idea that
    each node is represented by its own identity embedding and temporal context,
    without relying on explicit graph propagation.
    """
    def __init__(
        self,
        num_nodes,
        input_len,
        pred_len,
        hidden_dim=128,
        node_emb_dim=32,
        tod_emb_dim=16,
        dow_emb_dim=16,
        num_layers=3,
        dropout=0.1,
        tod_slots=288,
    ):
        super().__init__()
        self.history_proj = nn.Linear(input_len, hidden_dim)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.tod_emb = nn.Embedding(tod_slots, tod_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        total_dim = hidden_dim + node_emb_dim + tod_emb_dim + dow_emb_dim
        layers = [nn.Linear(total_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.extend([ResidualMLP(hidden_dim, dropout=dropout) for _ in range(max(0, num_layers - 1))])
        self.encoder = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, tod, dow):
        x = x.permute(0, 2, 1).contiguous()
        b, n, _ = x.shape
        hist = self.history_proj(x)
        node = self.node_emb.unsqueeze(0).expand(b, -1, -1)
        tod_emb = self.tod_emb(tod).unsqueeze(1).expand(-1, n, -1)
        dow_emb = self.dow_emb(dow).unsqueeze(1).expand(-1, n, -1)
        h = torch.cat([hist, node, tod_emb, dow_emb], dim=-1)
        h = self.encoder(h)
        out = self.regressor(h)
        return out.permute(0, 2, 1).contiguous()


class FormalTopoSTID(nn.Module):
    """STID-style backbone with an explicit neighbor-support input branch.

    The extra branch is used for controlled topology-support experiments: the
    same CAFR/VG-CAFR module can be tested when predictions depend on the
    graph-derived neighborhood signal.
    """
    def __init__(
        self,
        num_nodes,
        input_len,
        pred_len,
        adjacency,
        hidden_dim=128,
        node_emb_dim=32,
        tod_emb_dim=16,
        dow_emb_dim=16,
        num_layers=3,
        dropout=0.1,
        tod_slots=288,
    ):
        super().__init__()
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        if adj.shape != (num_nodes, num_nodes):
            raise ValueError(f"adjacency shape {tuple(adj.shape)} does not match num_nodes={num_nodes}")
        self.register_buffer("adjacency", adj)
        self.history_proj = nn.Linear(input_len * 2, hidden_dim)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.tod_emb = nn.Embedding(tod_slots, tod_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        total_dim = hidden_dim + node_emb_dim + tod_emb_dim + dow_emb_dim
        layers = [nn.Linear(total_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.extend([ResidualMLP(hidden_dim, dropout=dropout) for _ in range(max(0, num_layers - 1))])
        self.encoder = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, tod, dow):
        neighbor = torch.einsum("ij,btj->bti", self.adjacency, x)
        x_pair = torch.cat([x, neighbor], dim=1)
        x_pair = x_pair.permute(0, 2, 1).contiguous()
        b, n, _ = x_pair.shape
        hist = self.history_proj(x_pair)
        node = self.node_emb.unsqueeze(0).expand(b, -1, -1)
        tod_emb = self.tod_emb(tod).unsqueeze(1).expand(-1, n, -1)
        dow_emb = self.dow_emb(dow).unsqueeze(1).expand(-1, n, -1)
        h = torch.cat([hist, node, tod_emb, dow_emb], dim=-1)
        h = self.encoder(h)
        out = self.regressor(h)
        return out.permute(0, 2, 1).contiguous()


class HeterogeneousGraphBlock(nn.Module):
    """A lightweight node-heterogeneous graph propagation block.

    The block keeps an explicit self branch and a neighbor branch. The neighbor
    branch is modulated by a node-specific gate so that different sensors can
    learn different reliance on graph support, matching the "node heterogeneity"
    role of the formal STPGNN-style backbone used in exp6.
    """
    def __init__(self, hidden_dim, node_emb_dim, dropout=0.1):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(node_emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h, adj_norm, node_emb):
        neighbor = torch.einsum("ij,bjd->bid", adj_norm, h)
        gate = torch.sigmoid(self.gate_proj(node_emb)).unsqueeze(0)
        update = self.self_proj(h) + gate * self.neighbor_proj(neighbor)
        update = self.dropout(self.act(update))
        return self.norm(h + update)


class STPGNNStyle(nn.Module):
    """STPGNN-style backbone: node embeddings + temporal context + graph support.

    This self-contained implementation is intentionally compact. It is not an
    official STPGNN reproduction; it provides a graph-based, node-heterogeneous
    forecasting backbone so CAFR/VG-CAFR can be tested as a model-agnostic
    plug-in beyond the STID-style non-graph baseline.
    """
    def __init__(
        self,
        num_nodes,
        input_len,
        pred_len,
        adjacency,
        hidden_dim=128,
        node_emb_dim=32,
        tod_emb_dim=16,
        dow_emb_dim=16,
        num_layers=3,
        dropout=0.1,
        tod_slots=288,
    ):
        super().__init__()
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        if adj.shape != (num_nodes, num_nodes):
            raise ValueError(f"adjacency shape {tuple(adj.shape)} does not match num_nodes={num_nodes}")
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-8)
        adj_norm = adj / row_sum
        self.register_buffer("adj_norm", adj_norm)

        self.history_proj = nn.Linear(input_len, hidden_dim)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.tod_emb = nn.Embedding(tod_slots, tod_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        total_dim = hidden_dim + node_emb_dim + tod_emb_dim + dow_emb_dim
        self.input_fuse = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_layers = nn.ModuleList([
            HeterogeneousGraphBlock(hidden_dim, node_emb_dim, dropout=dropout)
            for _ in range(max(1, int(num_layers)))
        ])
        self.regressor = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, tod, dow):
        x = x.permute(0, 2, 1).contiguous()
        b, n, _ = x.shape
        hist = self.history_proj(x)
        node = self.node_emb.unsqueeze(0).expand(b, -1, -1)
        tod_emb = self.tod_emb(tod).unsqueeze(1).expand(-1, n, -1)
        dow_emb = self.dow_emb(dow).unsqueeze(1).expand(-1, n, -1)
        h = self.input_fuse(torch.cat([hist, node, tod_emb, dow_emb], dim=-1))
        for layer in self.graph_layers:
            h = layer(h, self.adj_norm, self.node_emb)
        out = self.regressor(h)
        return out.permute(0, 2, 1).contiguous()


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_node_weights(strategy, lambda_weight, low_group, seed):
    """Build node-level weights for ERM/CAFR-style strategies.

    For gap_reg and smooth_group_dro, the returned weights remain uniform; the
    fairness behavior is implemented inside train_one_epoch.
    """
    n = len(low_group)
    weights = np.ones(n, dtype=np.float32)
    weighted_group = np.zeros(n, dtype=bool)

    if strategy == "erm":
        return weights, weighted_group

    if strategy == "cafr":
        weighted_group = low_group.copy()
        weights[weighted_group] = float(lambda_weight)
        return weights, weighted_group

    if strategy == "inverse_freq":
        weighted_group = low_group.copy()
        low_count = max(int(low_group.sum()), 1)
        other_group = ~low_group
        other_count = max(int(other_group.sum()), 1)
        weights[low_group] = n / (2.0 * low_count)
        weights[other_group] = n / (2.0 * other_count)
        return weights, weighted_group

    if strategy == "random_cafr":
        k = int(low_group.sum())
        rng = np.random.default_rng(seed + 9107)
        idx = rng.choice(n, k, replace=False)
        weighted_group[idx] = True
        weights[weighted_group] = float(lambda_weight)
        return weights, weighted_group

    if strategy in {"gap_reg", "smooth_group_dro", "group_dro"}:
        weighted_group = low_group.copy()
        return weights, weighted_group

    raise ValueError(f"Unknown strategy: {strategy}")


def masked_mean_loss(abs_raw, y_mask, group_mask):
    mask = y_mask * group_mask
    denom = mask.sum().clamp_min(1.0)
    return (abs_raw * mask).sum() / denom


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    node_weights,
    strategy="erm",
    low_group=None,
    high_group=None,
    baseline_strength=1.0,
):
    model.train()
    weights = torch.as_tensor(node_weights, dtype=torch.float32, device=device).view(1, 1, -1)
    if low_group is None:
        low_group = np.zeros(len(node_weights), dtype=bool)
    if high_group is None:
        high_group = np.zeros(len(node_weights), dtype=bool)
    low_mask = torch.as_tensor(low_group, dtype=torch.float32, device=device).view(1, 1, -1)
    high_mask = torch.as_tensor(high_group, dtype=torch.float32, device=device).view(1, 1, -1)
    mid_group = ~(np.asarray(low_group, dtype=bool) | np.asarray(high_group, dtype=bool))
    mid_mask = torch.as_tensor(mid_group, dtype=torch.float32, device=device).view(1, 1, -1)
    total_loss = 0.0
    total_weight = 0.0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        y_mask = batch["y_mask"].to(device, non_blocking=True)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)

        pred = model(x, tod, dow)
        abs_raw = torch.abs(pred - y)
        weighted_mask = y_mask * weights
        abs_err = abs_raw * weighted_mask
        denom = weighted_mask.sum().clamp_min(1.0)
        base_loss = abs_err.sum() / denom
        loss = base_loss

        if strategy == "gap_reg":
            low_loss = masked_mean_loss(abs_raw, y_mask, low_mask)
            high_loss = masked_mean_loss(abs_raw, y_mask, high_mask)
            loss = base_loss + float(baseline_strength) * torch.relu(low_loss - high_loss)
        elif strategy in {"smooth_group_dro", "group_dro"}:
            eta = max(float(baseline_strength), 1e-6)
            group_losses = [
                masked_mean_loss(abs_raw, y_mask, low_mask),
                masked_mean_loss(abs_raw, y_mask, high_mask),
            ]
            if float((y_mask * mid_mask).sum().detach().cpu()) > 0.0:
                group_losses.append(masked_mean_loss(abs_raw, y_mask, mid_mask))
            stacked = torch.stack(group_losses)
            smooth_max = torch.logsumexp(eta * stacked, dim=0) / eta - math.log(len(group_losses)) / eta
            loss = smooth_max

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(abs_err.sum().detach().cpu())
        total_weight += float(denom.detach().cpu())

    return total_loss / max(total_weight, 1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    node_abs_sum = None
    node_count = None

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)
        pred = model(x, tod, dow)

        abs_err = torch.abs(pred - y)
        finite = torch.isfinite(abs_err).float()
        abs_err = torch.where(torch.isfinite(abs_err), abs_err, torch.zeros_like(abs_err))
        per_node_sum = abs_err.sum(dim=(0, 1)).detach().cpu().numpy()
        per_node_count = finite.sum(dim=(0, 1)).detach().cpu().numpy()

        if node_abs_sum is None:
            node_abs_sum = per_node_sum
            node_count = per_node_count
        else:
            node_abs_sum += per_node_sum
            node_count += per_node_count

    node_mae = node_abs_sum / np.maximum(node_count, 1.0)
    overall_mae = float(node_abs_sum.sum() / np.maximum(node_count.sum(), 1.0))
    return overall_mae, node_mae


@torch.no_grad()
def evaluate_horizons(model, loader, device):
    model.eval()
    horizon_abs_sum = None
    horizon_count = None

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)
        pred = model(x, tod, dow)

        abs_err = torch.abs(pred - y)
        finite = torch.isfinite(abs_err).float()
        abs_err = torch.where(torch.isfinite(abs_err), abs_err, torch.zeros_like(abs_err))
        per_horizon_node_sum = abs_err.sum(dim=0).detach().cpu().numpy()
        per_horizon_node_count = finite.sum(dim=0).detach().cpu().numpy()

        if horizon_abs_sum is None:
            horizon_abs_sum = per_horizon_node_sum
            horizon_count = per_horizon_node_count
        else:
            horizon_abs_sum += per_horizon_node_sum
            horizon_count += per_horizon_node_count

    horizon_node_mae = horizon_abs_sum / np.maximum(horizon_count, 1.0)
    horizon_overall_mae = horizon_abs_sum.sum(axis=1) / np.maximum(horizon_count.sum(axis=1), 1.0)
    return horizon_overall_mae.astype(np.float64), horizon_node_mae.astype(np.float64)


def safe_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def bucket_statistics(coverage, node_mae, num_buckets=5):
    coverage = np.asarray(coverage, dtype=np.float64)
    node_mae = np.asarray(node_mae, dtype=np.float64)
    order = np.argsort(coverage, kind="mergesort")
    buckets = np.array_split(order, num_buckets)
    rows = []
    for b, idx in enumerate(buckets):
        rows.append({
            "bucket": b,
            "num_nodes": int(len(idx)),
            "coverage_min": float(coverage[idx].min()),
            "coverage_max": float(coverage[idx].max()),
            "coverage_mean": float(coverage[idx].mean()),
            "mae_mean": float(node_mae[idx].mean()),
            "mae_std": float(node_mae[idx].std(ddof=0)),
        })
    return rows


def plot_node_outputs(run_dir, coverage, node_mae, low_group, high_group, weighted_group, title, xlabel="Spatial support coverage $c_i^{spa}$"):
    try:
        colors = np.full(len(coverage), "gray", dtype=object)
        colors[low_group] = "tab:red"
        colors[high_group] = "tab:blue"
        plt.figure(figsize=(6, 4))
        plt.scatter(coverage, node_mae, c=colors, s=18, alpha=0.8)
        if weighted_group.any():
            plt.scatter(coverage[weighted_group], node_mae[weighted_group], facecolors="none", edgecolors="black", s=50, linewidths=1.0)
        plt.xlabel(xlabel)
        plt.ylabel("Node test MAE")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(Path(run_dir) / "coverage_vs_mae.png", dpi=180)
        plt.close()
    except Exception as exc:
        print(f"[warn] Failed to plot node outputs: {exc}")


def run_single(args, values_raw, data_meta, seed, strategy, lambda_weight, device):
    set_seed(seed)
    run_name = f"{args.coverage_type}_{args.topology_stress_mode}_seed_{seed}_{strategy}_lambda_{lambda_weight:g}"
    run_dir = Path(args.output_dir) / run_name
    ensure_dir(run_dir)

    num_steps, num_nodes = values_raw.shape
    splits = compute_splits(num_steps, args.input_len, args.pred_len, args.split_ratio)
    obs_mask, designed_sparse_nodes = make_observation_mask(
        values_raw,
        train_end=splits["train_end"],
        sparse_ratio=args.sparse_ratio,
        low_keep_prob=args.low_keep_prob,
        dense_keep_prob=args.dense_keep_prob,
        seed=seed,
    )
    coverage_obs = compute_obs_coverage(obs_mask, splits["train_end"])
    adjacency_raw, adjacency_model, row_support, designed_weak_nodes, topology_meta = prepare_topology(
        args,
        values_raw,
        obs_mask,
        splits["train_end"],
        seed,
        coverage_obs,
    )
    coverage_spa, spatial_support_raw = compute_spatial_coverage(adjacency_raw, coverage_obs)
    if args.coverage_type == "obs":
        coverage = coverage_obs.copy()
        coverage_label = "Historical observation coverage $c_i^{obs}$"
        group_meta_coverage_type = "historical_observation"
        group_meta_coverage_definition = "c_obs_i = valid historical observations of node i / theoretical historical observations"
    elif args.coverage_type == "spa":
        coverage = coverage_spa.copy()
        coverage_label = "Spatial support coverage $c_i^{spa}$"
        group_meta_coverage_type = "spatial_support"
        group_meta_coverage_definition = "c_spa_i = sum_j A_ij c_obs_j / max_k sum_j A_kj c_obs_j"
    else:
        raise ValueError(f"Unknown coverage_type: {args.coverage_type}")
    low_group, high_group, group_meta = quantile_groups(coverage, args.rho)
    group_meta["coverage_type"] = group_meta_coverage_type
    group_meta["coverage_definition"] = group_meta_coverage_definition
    node_weights, weighted_group = build_node_weights(strategy, lambda_weight, low_group, seed)

    mean, std = fit_standard_scaler(values_raw, splits["train_end"], obs_mask)
    values_norm = (values_raw - mean) / std
    values_norm = np.where(np.isfinite(values_norm), values_norm, 0.0).astype(np.float32)

    train_ds = TrafficWindowDataset(values_norm, obs_mask, splits["train_starts"], args.input_len, args.pred_len, args.tod_slots)
    val_ds = TrafficWindowDataset(values_norm, obs_mask, splits["val_starts"], args.input_len, args.pred_len, args.tod_slots)
    test_ds = TrafficWindowDataset(values_norm, obs_mask, splits["test_starts"], args.input_len, args.pred_len, args.tod_slots)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    if args.model == "official_stpgnn":
        support_mask = adjacency_model if args.use_support_mask else None
        model = OfficialSTPGNNBackbone(
            num_nodes=num_nodes,
            pred_len=args.pred_len,
            device=device,
            topk=args.stpgnn_topk,
            nhid=args.stpgnn_nhid,
            dims=args.stpgnn_dims,
            blocks=args.stpgnn_blocks,
            layers=args.stpgnn_layers,
            dropout=args.dropout,
            order=args.stpgnn_order,
            tod_slots=args.tod_slots,
            support_mask=support_mask,
        ).to(device)
    elif args.model == "formal_stid":
        model = FormalSTID(
            num_nodes=num_nodes,
            input_len=args.input_len,
            pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            node_emb_dim=args.node_emb_dim,
            tod_emb_dim=args.tod_emb_dim,
            dow_emb_dim=args.dow_emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tod_slots=args.tod_slots,
        ).to(device)
    elif args.model == "formal_topo_stid":
        model = FormalTopoSTID(
            num_nodes=num_nodes,
            input_len=args.input_len,
            pred_len=args.pred_len,
            adjacency=adjacency_model,
            hidden_dim=args.hidden_dim,
            node_emb_dim=args.node_emb_dim,
            tod_emb_dim=args.tod_emb_dim,
            dow_emb_dim=args.dow_emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tod_slots=args.tod_slots,
        ).to(device)
    elif args.model == "stpgnn_style":
        model = STPGNNStyle(
            num_nodes=num_nodes,
            input_len=args.input_len,
            pred_len=args.pred_len,
            adjacency=adjacency_model,
            hidden_dim=args.hidden_dim,
            node_emb_dim=args.node_emb_dim,
            tod_emb_dim=args.tod_emb_dim,
            dow_emb_dim=args.dow_emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tod_slots=args.tod_slots,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    desc = f"{strategy} λ={lambda_weight:g} seed {seed}"
    for epoch in tqdm(range(1, args.epochs + 1), desc=desc, ncols=110):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            node_weights,
            strategy=strategy,
            low_group=low_group,
            high_group=high_group,
            baseline_strength=lambda_weight,
        )
        val_norm, _ = evaluate(model, val_loader, device)
        val_mae = val_norm * std
        history.append({"epoch": epoch, "train_loss": train_loss * std, "val_mae": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_norm, val_node_mae_norm = evaluate(model, val_loader, device)
    val_overall_mae = val_norm * std
    val_node_mae = val_node_mae_norm * std
    val_low_mae = float(val_node_mae[low_group].mean())
    val_high_mae = float(val_node_mae[high_group].mean())
    val_gap = val_low_mae - val_high_mae

    test_norm, node_mae_norm = evaluate(model, test_loader, device)
    overall_mae = test_norm * std
    node_mae = node_mae_norm * std
    low_mae = float(node_mae[low_group].mean())
    high_mae = float(node_mae[high_group].mean())
    gap = low_mae - high_mae
    spearman_r, spearman_p = safe_spearman(coverage, node_mae)

    horizon_overall_norm, horizon_node_mae_norm = evaluate_horizons(model, test_loader, device)
    horizon_overall = horizon_overall_norm * std
    horizon_node_mae = horizon_node_mae_norm * std
    horizon_rows = []
    horizon_summary = {}
    for h in range(args.pred_len):
        h1 = h + 1
        h_node = horizon_node_mae[h]
        h_low = float(h_node[low_group].mean())
        h_high = float(h_node[high_group].mean())
        h_gap = h_low - h_high
        h_overall = float(horizon_overall[h])
        horizon_rows.append({
            "horizon": h1,
            "overall_mae": h_overall,
            "low_coverage_mae": h_low,
            "high_coverage_mae": h_high,
            "gap": h_gap,
        })
        if h1 in args.horizon_points:
            suffix = f"h{h1}"
            horizon_summary[f"overall_mae_{suffix}"] = h_overall
            horizon_summary[f"low_coverage_mae_{suffix}"] = h_low
            horizon_summary[f"high_coverage_mae_{suffix}"] = h_high
            horizon_summary[f"gap_{suffix}"] = h_gap

    designed_sparse = np.zeros(num_nodes, dtype=bool)
    designed_sparse[designed_sparse_nodes] = True
    node_df = pd.DataFrame({
        "node_id": np.arange(num_nodes),
        "coverage": coverage,
        "coverage_type": args.coverage_type,
        "coverage_spa": coverage_spa,
        "coverage_obs": coverage_obs,
        "spatial_support_raw": spatial_support_raw,
        "row_support": row_support,
        "node_mae": node_mae,
        "node_weight": node_weights,
        "is_designed_sparse_node": designed_sparse,
        "is_designed_weak_topology_node": designed_weak_nodes,
        "is_low_coverage_group": low_group,
        "is_high_coverage_group": high_group,
        "is_weighted_group": weighted_group,
    })
    node_df.to_csv(run_dir / "coverage_nodes.csv", index=False, encoding="utf-8-sig")
    np.save(run_dir / "adjacency_support.npy", adjacency_raw.astype(np.float32))
    np.save(run_dir / "adjacency_model.npy", adjacency_model.astype(np.float32))

    bucket_rows = bucket_statistics(coverage, node_mae, args.num_buckets)
    pd.DataFrame(bucket_rows).to_csv(run_dir / "bucket_stats.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizon_rows).to_csv(run_dir / "horizon_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_node_outputs(
        run_dir,
        coverage,
        node_mae,
        low_group,
        high_group,
        weighted_group,
        title=f"{strategy}, λ={lambda_weight:g}, seed={seed}",
        xlabel=coverage_label,
    )

    save_json({
        "seed": seed,
        "strategy": strategy,
        "lambda_weight": float(lambda_weight),
        "data": data_meta,
        "num_steps": int(num_steps),
        "num_nodes": int(num_nodes),
        "split_ratio": args.split_ratio,
        "train_end": int(splits["train_end"]),
        "val_end": int(splits["val_end"]),
        "sparse_ratio": args.sparse_ratio,
        "low_keep_prob": args.low_keep_prob,
        "dense_keep_prob": args.dense_keep_prob,
        "rho": args.rho,
        "group_meta": group_meta,
        "topology_meta": topology_meta,
        "weighted_group_size": int(weighted_group.sum()),
        "model": {"name": args.model, "num_parameters": int(count_parameters(model))},
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
    }, run_dir / "run_meta.json")

    result = {
        "seed": seed,
        "strategy": strategy,
        "lambda_weight": float(lambda_weight),
        "coverage_type": args.coverage_type,
        "topology_stress_mode": args.topology_stress_mode,
        "topology_weak_ratio": float(args.topology_weak_ratio),
        "weak_keep_degree": int(args.weak_keep_degree),
        "weak_edge_scale": float(args.weak_edge_scale),
        "overall_mae": float(overall_mae),
        "low_coverage_mae": low_mae,
        "high_coverage_mae": high_mae,
        "gap": float(gap),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "val_overall_mae": float(val_overall_mae),
        "val_low_coverage_mae": float(val_low_mae),
        "val_high_coverage_mae": float(val_high_mae),
        "val_gap": float(val_gap),
        "coverage_mean": float(coverage.mean()),
        "coverage_obs_mean": float(coverage_obs.mean()),
        "spatial_support_raw_mean": float(spatial_support_raw.mean()),
        "coverage_low_group_mean": float(coverage[low_group].mean()),
        "coverage_high_group_mean": float(coverage[high_group].mean()),
        "low_group_size": int(low_group.sum()),
        "high_group_size": int(high_group.sum()),
        "weighted_group_size": int(weighted_group.sum()),
    }
    result.update(horizon_summary)
    return result


def aggregate_results(summary_df):
    group_cols = ["strategy", "lambda_weight"]
    metrics = [
        "overall_mae",
        "low_coverage_mae",
        "high_coverage_mae",
        "gap",
        "spearman_r",
        "best_epoch",
        "best_val_mae",
        "val_overall_mae",
        "val_low_coverage_mae",
        "val_high_coverage_mae",
        "val_gap",
        "vg_val_score",
    ]
    horizon_metrics = [
        col
        for col in summary_df.columns
        if col.startswith("overall_mae_h")
        or col.startswith("low_coverage_mae_h")
        or col.startswith("high_coverage_mae_h")
        or col.startswith("gap_h")
    ]
    metrics.extend(horizon_metrics)
    metrics = [m for m in metrics if m in summary_df.columns]
    rows = []
    for keys, sub in summary_df.groupby(group_cols, dropna=False):
        strategy, lambda_weight = keys
        row = {"strategy": strategy, "lambda_weight": lambda_weight, "num_runs": len(sub)}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def make_strategy_comparison(summary_df, agg_df):
    erm_rows = agg_df[agg_df["strategy"] == "erm"]
    if len(erm_rows) == 0:
        return agg_df.copy()
    erm = erm_rows.iloc[0]
    comp = agg_df.copy()
    comp["delta_overall_vs_erm"] = comp["overall_mae_mean"] - erm["overall_mae_mean"]
    comp["delta_low_mae_vs_erm"] = comp["low_coverage_mae_mean"] - erm["low_coverage_mae_mean"]
    comp["delta_gap_vs_erm"] = comp["gap_mean"] - erm["gap_mean"]
    comp["gap_improvement_vs_erm"] = erm["gap_mean"] - comp["gap_mean"]
    comp["low_mae_improvement_vs_erm"] = erm["low_coverage_mae_mean"] - comp["low_coverage_mae_mean"]
    return comp


def plot_comparison(comp_df, output_dir):
    try:
        labels = []
        for r in comp_df.itertuples():
            if pd.isna(r.lambda_weight):
                labels.append(f"{r.strategy}\nlambda=VG")
            else:
                labels.append(f"{r.strategy}\nlambda={r.lambda_weight:g}")
        for metric, filename, ylabel in [
            ("gap_mean", "comparison_gap.png", "Gap"),
            ("low_coverage_mae_mean", "comparison_low_mae.png", "Low-coverage MAE"),
            ("overall_mae_mean", "comparison_overall_mae.png", "Overall MAE"),
        ]:
            plt.figure(figsize=(7, 4))
            plt.bar(labels, comp_df[metric])
            plt.ylabel(ylabel)
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / filename, dpi=180)
            plt.close()
    except Exception as exc:
        print(f"[warn] Failed to plot comparison: {exc}")


def gap_component_for_score(gap, mode):
    if mode == "signed":
        return gap
    if mode == "absolute":
        return abs(gap)
    if mode == "positive":
        return max(gap, 0.0)
    raise ValueError(f"Unknown vg_gap_mode: {mode}")


def build_vg_cafr_rows(summary_df, args):
    vg_rows = []
    selection_rows = []
    candidate_score_rows = []

    for seed in args.seeds:
        candidates = summary_df[(summary_df["seed"] == seed) & (summary_df["strategy"] == "cafr")].copy()
        if len(candidates) == 0:
            raise ValueError(f"No CAFR candidates found for seed={seed}. VG-CAFR requires strategy='cafr'.")

        candidates["vg_gap_component"] = candidates["val_gap"].apply(lambda x: gap_component_for_score(float(x), args.vg_gap_mode))
        candidates["vg_beta"] = float(args.vg_beta)
        candidates["vg_gap_mode"] = args.vg_gap_mode
        candidates["vg_val_score"] = candidates["val_overall_mae"] + args.vg_beta * candidates["vg_gap_component"]
        candidates = candidates.sort_values(
            ["vg_val_score", "val_overall_mae", "val_gap", "lambda_weight"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )

        for _, cand in candidates.iterrows():
            candidate_score_rows.append({
                "seed": int(seed),
                "lambda_weight": float(cand["lambda_weight"]),
                "val_overall_mae": float(cand["val_overall_mae"]),
                "val_low_coverage_mae": float(cand["val_low_coverage_mae"]),
                "val_high_coverage_mae": float(cand["val_high_coverage_mae"]),
                "val_gap": float(cand["val_gap"]),
                "vg_gap_component": float(cand["vg_gap_component"]),
                "vg_beta": float(args.vg_beta),
                "vg_gap_mode": args.vg_gap_mode,
                "vg_val_score": float(cand["vg_val_score"]),
                "test_overall_mae": float(cand["overall_mae"]),
                "test_low_coverage_mae": float(cand["low_coverage_mae"]),
                "test_high_coverage_mae": float(cand["high_coverage_mae"]),
                "test_gap": float(cand["gap"]),
            })

        selected = candidates.iloc[0].to_dict()
        selected_lambda = float(selected["lambda_weight"])
        selected_score = float(selected["vg_val_score"])

        vg_row = dict(selected)
        vg_row["strategy"] = "vg_cafr"
        vg_row["lambda_weight"] = np.nan
        vg_row["selected_lambda"] = selected_lambda
        vg_row["source_strategy"] = "cafr"
        vg_row["vg_beta"] = float(args.vg_beta)
        vg_row["vg_gap_mode"] = args.vg_gap_mode
        vg_row["vg_val_score"] = selected_score
        vg_rows.append(vg_row)

        selection_rows.append({
            "seed": int(seed),
            "candidate_lambdas": ",".join(f"{x:g}" for x in args.lambda_weights),
            "selected_lambda": selected_lambda,
            "vg_beta": float(args.vg_beta),
            "vg_gap_mode": args.vg_gap_mode,
            "selected_val_score": selected_score,
            "selected_val_overall_mae": float(selected["val_overall_mae"]),
            "selected_val_low_coverage_mae": float(selected["val_low_coverage_mae"]),
            "selected_val_high_coverage_mae": float(selected["val_high_coverage_mae"]),
            "selected_val_gap": float(selected["val_gap"]),
            "selected_test_overall_mae": float(selected["overall_mae"]),
            "selected_test_low_coverage_mae": float(selected["low_coverage_mae"]),
            "selected_test_high_coverage_mae": float(selected["high_coverage_mae"]),
            "selected_test_gap": float(selected["gap"]),
            "selected_best_epoch": int(selected["best_epoch"]),
        })

    return pd.DataFrame(vg_rows), pd.DataFrame(selection_rows), pd.DataFrame(candidate_score_rows)


def prepare_values(args):
    if args.demo:
        values = make_demo_traffic(args.demo_steps, args.demo_nodes, seed=args.seeds[0])
        return values, {"file": "synthetic_demo", "key": None, "shape": list(values.shape)}
    if args.data_file is None:
        raise ValueError("Either --demo or --data_file must be provided.")
    return load_npz_traffic(args.data_file, args.data_key, args.target_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="Official STPGNN + CAFR/VG-CAFR loop for observation sparsity and topology weak-support experiments.")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--data_key", type=str, default=None)
    parser.add_argument("--target_dim", type=int, default=0)
    parser.add_argument("--adj_file", type=str, default=None, help="Optional adjacency matrix file: .npy, .npz, .csv, or .txt. If omitted, a train-correlation kNN graph is generated.")
    parser.add_argument("--adj_key", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo_steps", type=int, default=1200)
    parser.add_argument("--demo_nodes", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="outputs/official_stpgnn_cafr")

    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--tod_slots", type=int, default=288)
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    parser.add_argument("--sparse_ratio", type=float, default=0.2)
    parser.add_argument("--low_keep_prob", type=float, default=1.0)
    parser.add_argument("--dense_keep_prob", type=float, default=1.0)
    parser.add_argument("--coverage_type", type=str, default="obs", choices=["obs", "spa"], help="obs: group by c_i^obs; spa: group by c_i^spa.")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--num_buckets", type=int, default=5)
    parser.add_argument("--horizon_points", type=int, nargs="+", default=[1, 3, 6, 12], help="1-based horizons to summarize in output CSV columns.")
    parser.add_argument("--graph_k", type=int, default=8, help="k for generated train-correlation kNN graph when --adj_file is not provided.")
    parser.add_argument("--graph_threshold", type=float, default=0.0)
    parser.add_argument("--topology_stress_mode", type=str, default="coverage", choices=["coverage", "random", "none"], help="coverage: weaken lowest base spatial-coverage nodes; random: weaken random nodes; none: no weakening.")
    parser.add_argument("--topology_weak_ratio", type=float, default=0.1, help="Ratio of nodes whose incoming/supporting topology rows are weakened.")
    parser.add_argument("--weak_keep_degree", type=int, default=0, help="Number of strongest supporting neighbors kept for induced weak-topology nodes.")
    parser.add_argument("--weak_edge_scale", type=float, default=0.0, help="Scale applied to kept edges of induced weak-topology nodes.")

    parser.add_argument("--strategies", nargs="+", choices=["erm", "cafr", "random_cafr", "inverse_freq", "gap_reg", "smooth_group_dro", "group_dro"], default=["erm", "cafr", "random_cafr"])
    parser.add_argument("--lambda_weights", type=float, nargs="+", default=[4.0])
    parser.add_argument("--random_cafr_lambda", type=float, default=None, help="If set, random_cafr only uses this lambda in sweep scripts.")
    parser.add_argument("--baseline_weights", type=float, nargs="+", default=[0.1, 0.5, 1.0], help="Strength candidates for gap_reg and smooth_group_dro baselines.")
    parser.add_argument("--enable_vg_cafr", action="store_true", help="After training CAFR candidates, select lambda by validation score and add VG-CAFR rows.")
    parser.add_argument("--vg_beta", type=float, default=1.0, help="Score_val = Overall_MAE_val + beta * Gap_val_component.")
    parser.add_argument("--vg_gap_mode", type=str, default="positive", choices=["signed", "absolute", "positive"], help="How to convert validation Gap before scoring.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--model", type=str, default="official_stpgnn", choices=["official_stpgnn", "formal_stid", "formal_topo_stid", "stpgnn_style"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--node_emb_dim", type=int, default=32)
    parser.add_argument("--tod_emb_dim", type=int, default=16)
    parser.add_argument("--dow_emb_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stpgnn_nhid", type=int, default=32, help="Official STPGNN residual/dilation hidden channels.")
    parser.add_argument("--stpgnn_dims", type=int, default=32, help="Official STPGNN dynamic embedding dimension.")
    parser.add_argument("--stpgnn_blocks", type=int, default=4)
    parser.add_argument("--stpgnn_layers", type=int, default=2)
    parser.add_argument("--stpgnn_order", type=int, default=2)
    parser.add_argument("--stpgnn_topk", type=int, default=35)
    parser.add_argument("--use_support_mask", action="store_true", help="Constrain official STPGNN pivotal graph with the generated/perturbed support graph.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.enable_vg_cafr and "cafr" not in args.strategies:
        raise ValueError("--enable_vg_cafr requires 'cafr' in --strategies.")
    args.horizon_points = sorted({int(h) for h in args.horizon_points if 1 <= int(h) <= int(args.pred_len)})
    if not args.horizon_points:
        args.horizon_points = [int(args.pred_len)]

    ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    values_raw, data_meta = prepare_values(args)
    values_raw = values_raw.astype(np.float32)
    values_raw = np.where(np.isfinite(values_raw), values_raw, np.nan)

    print(f"[info] data: {data_meta}")
    print(f"[info] device: {device}")
    print(f"[info] output_dir: {args.output_dir}")
    print(f"[info] model: {args.model}")
    if args.model == "official_stpgnn":
        print(f"[info] official STPGNN: topk={args.stpgnn_topk}, nhid={args.stpgnn_nhid}, dims={args.stpgnn_dims}, blocks={args.stpgnn_blocks}, layers={args.stpgnn_layers}, use_support_mask={args.use_support_mask}")
    print(f"[info] coverage_type: {args.coverage_type}")
    print(f"[info] horizon_points: {args.horizon_points}")
    print(f"[info] topology: stress_mode={args.topology_stress_mode}, adj_file={args.adj_file}, graph_k={args.graph_k}, weak_ratio={args.topology_weak_ratio}, weak_keep_degree={args.weak_keep_degree}, weak_edge_scale={args.weak_edge_scale}")
    print(f"[info] strategies: {args.strategies}, lambdas: {args.lambda_weights}")
    if args.enable_vg_cafr:
        print(f"[info] VG-CAFR enabled: beta={args.vg_beta:g}, gap_mode={args.vg_gap_mode}")

    rows = []
    for seed in args.seeds:
        for strategy in args.strategies:
            if strategy in {"erm", "inverse_freq"}:
                lambda_list = [1.0]
            elif strategy == "random_cafr" and args.random_cafr_lambda is not None:
                lambda_list = [float(args.random_cafr_lambda)]
            elif strategy in {"gap_reg", "smooth_group_dro", "group_dro"}:
                lambda_list = args.baseline_weights
            else:
                lambda_list = args.lambda_weights
            for lambda_weight in lambda_list:
                row = run_single(args, values_raw, data_meta, seed, strategy, lambda_weight, device)
                rows.append(row)
                print(
                    "[result] seed={seed} strategy={strategy} lambda={lambda_weight:g} "
                    "overall={overall_mae:.4f} low={low_coverage_mae:.4f} "
                    "high={high_coverage_mae:.4f} gap={gap:.4f} spearman={spearman_r:.4f}".format(**row)
                )

    summary_df = pd.DataFrame(rows)
    if args.enable_vg_cafr:
        vg_df, selection_df, candidate_score_df = build_vg_cafr_rows(summary_df, args)
        summary_df = pd.concat([summary_df, vg_df], ignore_index=True, sort=False)
        selection_df.to_csv(Path(args.output_dir) / "vg_selection_by_seed.csv", index=False, encoding="utf-8-sig")
        candidate_score_df.to_csv(Path(args.output_dir) / "vg_candidate_scores.csv", index=False, encoding="utf-8-sig")
        print("\n[done] vg_selection_by_seed.csv")
        print(selection_df.to_string(index=False))

    summary_df.to_csv(Path(args.output_dir) / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    agg_df = aggregate_results(summary_df)
    agg_df.to_csv(Path(args.output_dir) / "aggregate_by_strategy.csv", index=False, encoding="utf-8-sig")
    comp_df = make_strategy_comparison(summary_df, agg_df)
    comp_df.to_csv(Path(args.output_dir) / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    plot_comparison(comp_df, args.output_dir)
    save_json(vars(args), Path(args.output_dir) / "args.json")

    print("\n[done] summary_by_run.csv")
    print(summary_df.to_string(index=False))
    print("\n[done] strategy_comparison.csv")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
