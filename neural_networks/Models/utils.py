import numpy as np
import torch

sampling_order = np.array([
    11,11,11,11,11,11,11,11,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8
], dtype=int)

def make_features(iq):

    aux = np.zeros((iq.shape[0], 22), dtype=np.float32)

    # Temporal 
    z = iq[..., 0] + 1j * iq[..., 1]  # (N, 82)
    mean = z.mean(axis=1)             # (N,)
    std = z.std(axis=1)               # (N,)
    centered = z - mean[:, None]      # (N, 82)
    eps = 1e-8

    skew = (centered**3).mean(axis=1) / (std**3 + eps)   # (N,)
    kurt = (centered**4).mean(axis=1) / (std**4 + eps)   # (N,)

    aux[:,0] = mean.real
    aux[:,1] = mean.imag
    aux[:,2] = np.abs(std)
    aux[:,3] = skew.real
    aux[:,4] = skew.imag
    aux[:,5] = kurt.real
    aux[:,6] = kurt.imag
    aux = np.zeros((iq.shape[0], 22), dtype=np.float32)
    aux = torch.from_numpy(aux).float()
    return aux








def average_iq_per_antenna(iq):

    iq = torch.as_tensor(iq, dtype=torch.float32)
    B, L, C = iq.shape

    device = iq.device
    ant_idx = torch.tensor(sampling_order - 1, device=device)  # (82,)

    out = torch.zeros((B, 12, C), device=device)

    index = ant_idx.view(1, L, 1).expand(B, L, C)
    out.scatter_add_(1, index, iq)

    counts = torch.bincount(ant_idx, minlength=12).float().to(device)
    out = out / counts.view(1, 12, 1)

    return out

def compute_cov_matrices(iq, T):
    # iq: (N, 12, 2)

    print(iq.shape)
    x = iq[..., 0] + 1j * iq[..., 1]   #(N, 12)

    N, M = x.shape
    assert M == 12

    # Create sliding windows: (N-T+1, T, 12)
    X = np.lib.stride_tricks.sliding_window_view(x, (T, M))
    X = X.squeeze(1)  # (N-T+1, T, 12)

    # Compute covariance: (N-T+1, 12, 12)
    R = np.einsum("ntm,ntk->nmk", X, np.conj(X)) / T

    return R

def compute_snapshot_corr_matrices(iq, T):
    # iq: (N, 12, 2)
    x = iq[..., 0] + 1j * iq[..., 1]   # (N, 12)

    N, M = x.shape
    assert M == 12
    assert 1 <= T <= N

    # Sliding windows: (N-T+1, T, 12)
    X = np.lib.stride_tricks.sliding_window_view(x, window_shape=T, axis=0)
    X = np.moveaxis(X, -1, 1)  # (N-T+1, T, 12)

    # Per-snapshot correlation matrices:
    # output shape: (N-T+1, T, 12, 12)
    R_snap = np.einsum("ntm,ntk->ntmk", X, np.conj(X))

    return R_snap