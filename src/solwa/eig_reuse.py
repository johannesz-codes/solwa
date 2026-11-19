"""Eigendecomposition of a smooth matrix family by refining the previous member.

For a batch A[0], A[1], ... that varies smoothly (one layer over wavelength), a full
``torch.linalg.eig`` is only needed for the first member. Every further member starts
from the eigenvectors V and W ~ V^-1 of its neighbour and is refined by Newton steps
that consist of matrix products: with T = W A V,

    E_ij = T_ij / (t_jj - t_ii),  V <- V (I + E),  W <- (I - E) W,  W <- W (2I - V W).

Strongly coupled or degenerate modes are diagonalized once as small blocks. A member
that fails the acceptance checks gets a full ``torch.linalg.eig``.
"""

import numpy as np
import torch

DEG_TOL = 1e-6  # relative gap below which two modes count as exactly degenerate
COUPLING_ETA = 0.1  # group modes i, j if |T_ij| > eta |t_ii - t_jj|
MU_FLOOR = 1e-2  # lower bound of |mu| in the per-mode residual
FINAL_NS = 2  # Newton-Schulz steps before W is handed to the next member


def _components(close):
    """Connected components of a boolean adjacency [B, n, n] (min-label propagation)."""
    B, n, _ = close.shape
    label = torch.arange(n, device=close.device).expand(B, n).clone()
    none = torch.full((B, n, n), n, device=close.device, dtype=label.dtype)
    while True:
        new = torch.where(close, label[:, None, :].expand(B, n, n), none).min(-1).values
        new = torch.minimum(new, label)
        if torch.equal(new, label):
            return label
        label = new


def _eig2x2(T):
    """Closed-form eigenvectors of 2x2 blocks; identity if the block is scalar."""
    a, b, c, d = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    disc = torch.sqrt(((a - d) / 2) ** 2 + b * c)
    cols = []
    for val in ((a + d) / 2 + disc, (a + d) / 2 - disc):
        va, vb = torch.stack((b, val - a), -1), torch.stack((val - d, c), -1)
        v = torch.where((va.norm(dim=-1) >= vb.norm(dim=-1))[:, None], va, vb)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-300))
    Q = torch.stack(cols, -1)
    det = (Q[:, 0, 0] * Q[:, 1, 1] - Q[:, 0, 1] * Q[:, 1, 0]).abs()
    bad = (det < 1e-6) | ~torch.isfinite(Q).all(-1).all(-1)
    return torch.where(bad[:, None, None], torch.eye(2).to(T), Q)


def _rotate_clusters(T, V, W):
    """Diagonalize T on each group of strongly coupled modes; V may carry extra rows."""
    B, n, _ = T.shape
    dev = T.device
    d = torch.diagonal(T, dim1=-2, dim2=-1)
    gap = (d[:, :, None] - d[:, None, :]).abs()
    scale = torch.maximum(d.abs()[:, :, None], d.abs()[:, None, :])
    off = torch.maximum(T.abs(), T.abs().transpose(1, 2))
    close = (off > COUPLING_ETA * gap) | (gap <= DEG_TOL * scale)
    close.diagonal(dim1=-2, dim2=-1).fill_(False)
    label = _components(close)
    size = torch.zeros_like(label).scatter_add_(1, label, torch.ones_like(label))
    size = size.gather(1, label)
    order = (label * n + torch.arange(n, device=dev)).argsort(dim=1)
    label, size = label.gather(1, order), size.gather(1, order)
    head = torch.ones_like(label[:, :1], dtype=torch.bool)
    first = torch.cat((head, label[:, 1:] != label[:, :-1]), 1)
    rows = torch.arange(V.shape[1], device=dev)[None, :, None]
    cols = torch.arange(n, device=dev)[None, None, :]
    for s in torch.unique(size).tolist():
        if s < 2:
            continue
        bi, pos = torch.nonzero((size == s) & first, as_tuple=True)
        idx = order[bi[:, None], pos[:, None] + torch.arange(s, device=dev)[None, :]]
        b3 = bi[:, None, None]
        Tc = T[b3, idx[:, :, None], idx[:, None, :]]
        if s == 2:
            Q = _eig2x2(Tc)
        else:  # few, tiny blocks: batched GPU eig is a per-matrix loop
            Q = torch.from_numpy(np.linalg.eig(Tc.cpu().numpy())[1]).to(dev)
        V[b3, rows, idx[:, None, :]] = V[b3, rows, idx[:, None, :]] @ Q
        W[b3, idx[:, :, None], cols] = torch.linalg.solve(
            Q, W[b3, idx[:, :, None], cols]
        )
    return V, W


def _residual(V, AV, mu):
    R = AV - V * mu[:, None, :]
    return R.norm(dim=-2) / (V.norm(dim=-2) * mu.abs().clamp(min=MU_FLOOR))


def refine(A, V, W, max_iter=7, tol=1e-11):
    """Newton refinement of (V, W ~ V^-1) for A [B, n, n], using only matrix products.

    Returns (mu, V, W, ok). ok[b] requires the per-mode residual, W V = I and
    sum(mu) = trace(A) to hold for batch item b.
    """
    n = A.shape[-1]
    eye = torch.eye(n).to(A)
    AV = A @ V
    for it in range(max_iter):
        T = W @ AV
        if it == 0:
            VAV, W = _rotate_clusters(T, torch.cat((V, AV), 1), W)
            V, AV = VAV[:, :n], VAV[:, n:]
            T = W @ AV
        d = torch.diagonal(T, dim1=-2, dim2=-1)
        gap = d[:, None, :] - d[:, :, None]
        scale = torch.maximum(d.abs()[:, None, :], d.abs()[:, :, None])
        small = gap.abs() <= DEG_TOL * scale
        E = torch.where(
            small,
            torch.zeros_like(T),
            T / torch.where(small, torch.ones_like(gap), gap),
        )
        V, AV, W = V + V @ E, AV + AV @ E, W - E @ W
        norm = V.norm(dim=-2, keepdim=True)
        V, AV, W = V / norm, AV / norm, W * norm.transpose(-1, -2)
        W = W @ (2 * eye - V @ W)
        mu = (W * AV.transpose(-1, -2)).sum(-1)  # diag(W AV) in O(n^2)
        if bool((_residual(V, AV, mu).amax(-1) < tol).all()):
            break
    AV = A @ V  # verify with the exact product
    mu = (W * AV.transpose(-1, -2)).sum(-1)
    res = _residual(V, AV, mu).amax(-1)
    for _ in range(FINAL_NS):
        W = W @ (2 * eye - V @ W)
    inv_err = (W @ V - eye).abs().amax(dim=(-2, -1))
    trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    trace_err = (mu.sum(-1) - trace).abs() / mu.abs().sum(-1)
    return mu, V, W, (res < tol) & (inv_err < 1e-3) & (trace_err < 1e-10)


def reuse_eig(A, broadening=1e-10):
    """Eigenvalues and eigenvectors of a batch A [B, n, n], chained along the batch.

    The batch should be ordered so that neighbours are close; otherwise members fall
    back to ``torch.linalg.eig``, which stays correct but gains nothing. Gradients do
    not pass through the refinement: one linearized eigen-update at the final pairs
    gives the usual eigendecomposition derivative (with the Lorentzian broadening of
    ``Eig``), and normal autograd handles it.
    """
    mus, Vs, Ws = [], [], []
    with torch.no_grad():
        for k in range(A.shape[0]):
            Ak = A[k : k + 1].detach()
            ok = False
            if Vs:
                mu, V, W, ok = refine(Ak, Vs[-1].clone(), Ws[-1].clone())
                ok = bool(ok[0])
            if not ok:
                mu, V = torch.linalg.eig(Ak)
                W = torch.linalg.inv(V)
            mus.append(mu)
            Vs.append(V)
            Ws.append(W)
    mu, V, W = torch.cat(mus), torch.cat(Vs), torch.cat(Ws)
    if not A.requires_grad:
        return mu, V
    T = W @ A @ V
    d = torch.diagonal(T, dim1=-2, dim2=-1)
    s = mu[:, None, :] - mu[:, :, None]
    F = s.conj() / (s.abs() ** 2 + broadening)
    F = F - torch.diag_embed(torch.diagonal(F, dim1=-2, dim2=-1))
    dV = V @ (T * F)
    return mu + (d - d.detach()), V + (dV - dV.detach())
