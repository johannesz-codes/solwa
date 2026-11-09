"""Benchmark: reuse the eigendecomposition of a patterned RCWA layer across wavelengths.

Compares ``torch.linalg.eig`` at every wavelength with a chained refinement that only
needs matrix products. See the pull request description for the method and references.

Problem: A(lambda) = P Q of a square a-Si:H pillar grating in n=1.5 (Li factorization,
normal incidence, mu=1), matrix size n = 2(2 order + 1)^2. The setup follows
``solwa.rcwa`` with a leading wavelength batch dimension.

    python benchmarks/eig_reuse.py --order 7
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDT, RDT = torch.complex128, torch.float64
NK_FILE = Path(__file__).parents[1] / "example" / "Materials_data" / "aSiH.txt"
PERIOD, WIDTH, N_DIEL, NX = 522.1, 338.9, 1.5, 512
WINDOWS = {"short": (300.0, 320.0), "edge": (440.0, 460.0), "long": (700.0, 720.0)}

DEG_TOL = 1e-6  # relative gap below which two modes count as exactly degenerate
COUPLING_ETA = 0.1  # join modes i, j if |T_ij| > eta |t_ii - t_jj|
MU_FLOOR = 1e-2  # lower bound of |mu| in the per-mode residual
FINAL_NS = 2  # Newton-Schulz steps before W is handed to the next wavelength


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


class Problem:
    """Builds A(lambda) = P Q for a batch of wavelengths."""

    def __init__(self, order):
        nk = np.loadtxt(NK_FILE)
        self.n_i, self.k_i = (
            interp1d(nk[:, 0], nk[:, c], kind="cubic") for c in (1, 2)
        )
        self.ord = torch.arange(-order, order + 1, device=DEV)
        self.M = len(self.ord)
        self.N = self.M**2
        i = torch.arange(NX, dtype=RDT, device=DEV) + 0.5
        x = PERIOD * i / NX
        edge = torch.sigmoid((WIDTH / 2 - (x - PERIOD / 2).abs()) / (PERIOD / NX / 2))
        self.fill = (edge[:, None] * edge[None, :]).to(CDT)

    def eps_si(self, lam):
        lam = lam.cpu().numpy()
        return torch.as_tensor(
            (self.n_i(lam) + 1j * self.k_i(lam)) ** 2, dtype=CDT, device=DEV
        )

    def conv_laurent(self, eps):
        gx, gy = torch.meshgrid(self.ord, self.ord, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        coef = torch.fft.fft2(eps) / NX**2
        return coef[:, gx[:, None] - gx[None, :], gy[:, None] - gy[None, :]]

    def conv_li(self, eps):
        B, N = eps.shape[0], self.N
        inv = 1.0 / eps
        delta = self.ord[:, None] - self.ord[None, :]
        # eps_xx: inverse rule along x, Laurent along y
        a = torch.fft.fft(inv, dim=1) / NX
        a = torch.fft.fft(torch.linalg.inv(a.transpose(1, 2)[:, :, delta]), dim=1) / NX
        exx = a[:, delta].permute(0, 3, 1, 4, 2).reshape(B, N, N)
        # eps_yy: inverse rule along y, Laurent along x
        a = torch.fft.fft(inv, dim=2) / NX
        a = torch.fft.fft(torch.linalg.inv(a[:, :, delta]), dim=1) / NX
        eyy = a[:, delta].permute(0, 1, 3, 2, 4).reshape(B, N, N)
        return exx, eyy

    def matrix(self, lam):
        B, M, N = lam.shape[0], self.M, self.N
        eps = N_DIEL**2 + (self.eps_si(lam)[:, None, None] - N_DIEL**2) * self.fill
        eps_conv = self.conv_laurent(eps)
        exx, eyy = self.conv_li(eps)
        k = self.ord.to(RDT)[None, :] * (lam / PERIOD)[:, None]
        Kx = torch.diag_embed(k[:, :, None].expand(-1, M, M).reshape(-1, N).to(CDT))
        Ky = torch.diag_embed(k[:, None, :].expand(-1, M, M).reshape(-1, N).to(CDT))
        eye = torch.eye(N, dtype=CDT, device=DEV).expand(B, N, N)
        Z = torch.zeros_like(eye)
        KK = torch.cat((Kx, Ky), 1)
        P = torch.cat(
            (torch.cat((Z, eye), 2), torch.cat((-eye, Z), 2)), 1
        ) + KK @ torch.linalg.solve(eps_conv, torch.cat((Ky, -Kx), 2))
        Q = torch.cat(
            (torch.cat((Z, -eyy), 2), torch.cat((exx, Z), 2)), 1
        ) + KK @ torch.cat((-Ky, Kx), 2)
        return P @ Q


def components(close):
    """Connected components of a boolean adjacency [B, n, n] (min-label propagation)."""
    B, n, _ = close.shape
    label = torch.arange(n, device=DEV).expand(B, n).clone()
    none = torch.full((B, n, n), n, device=DEV, dtype=label.dtype)
    while True:
        new = torch.where(close, label[:, None, :].expand(B, n, n), none).min(-1).values
        new = torch.minimum(new, label)
        if torch.equal(new, label):
            return label
        label = new


def eig2x2(T):
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
    return torch.where(bad[:, None, None], torch.eye(2, dtype=T.dtype, device=DEV), Q)


def rotate_clusters(T, V, W):
    """Diagonalize T on each cluster of strongly coupled modes; V may carry extra rows."""
    B, n, _ = T.shape
    d = torch.diagonal(T, dim1=-2, dim2=-1)
    gap = (d[:, :, None] - d[:, None, :]).abs()
    scale = torch.maximum(d.abs()[:, :, None], d.abs()[:, None, :])
    off = torch.maximum(T.abs(), T.abs().transpose(1, 2))
    close = (off > COUPLING_ETA * gap) | (gap <= DEG_TOL * scale)
    close.diagonal(dim1=-2, dim2=-1).fill_(False)
    label = components(close)
    size = (
        torch.zeros_like(label).scatter_add_(1, label, torch.ones_like(label))
    ).gather(1, label)
    order = (label * n + torch.arange(n, device=DEV)).argsort(dim=1)
    label, size = label.gather(1, order), size.gather(1, order)
    head = torch.ones_like(label[:, :1], dtype=torch.bool)
    first = torch.cat((head, label[:, 1:] != label[:, :-1]), 1)
    rows = torch.arange(V.shape[1], device=DEV)[None, :, None]
    cols = torch.arange(n, device=DEV)[None, None, :]
    for s in torch.unique(size).tolist():
        if s < 2:
            continue
        bi, pos = torch.nonzero((size == s) & first, as_tuple=True)
        idx = order[bi[:, None], pos[:, None] + torch.arange(s, device=DEV)[None, :]]
        b3 = bi[:, None, None]
        Tc = T[b3, idx[:, :, None], idx[:, None, :]]
        if s == 2:
            Q = eig2x2(Tc)
        else:  # few, tiny blocks: batched GPU eig is a per-matrix loop
            Q = torch.from_numpy(np.linalg.eig(Tc.cpu().numpy())[1]).to(DEV)
        V[b3, rows, idx[:, None, :]] = V[b3, rows, idx[:, None, :]] @ Q
        W[b3, idx[:, :, None], cols] = torch.linalg.solve(
            Q, W[b3, idx[:, :, None], cols]
        )
    return V, W


def residual(V, AV, mu):
    R = AV - V * mu[:, None, :]
    return R.norm(dim=-2) / (V.norm(dim=-2) * mu.abs().clamp(min=MU_FLOOR))


def refine(A, V, W, max_iter=7, tol=1e-11):
    """Newton refinement of (V, W ~ V^-1) for A, using only matrix products.

    Returns (mu, V, W, ok, n_iter). ok[b] requires the per-mode residual, W V = I and
    sum(mu) = trace(A) to hold for batch item b.
    """
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=DEV)
    AV = A @ V
    for it in range(max_iter):
        T = W @ AV
        if it == 0:
            VAV, W = rotate_clusters(T, torch.cat((V, AV), 1), W)
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
        if bool((residual(V, AV, mu).amax(-1) < tol).all()):
            break
    AV = A @ V  # verify with the exact product
    mu = (W * AV.transpose(-1, -2)).sum(-1)
    res = residual(V, AV, mu).amax(-1)
    for _ in range(FINAL_NS):
        W = W @ (2 * eye - V @ W)
    inv_err = (W @ V - eye).abs().amax(dim=(-2, -1))
    trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    trace_err = (mu.sum(-1) - trace).abs() / mu.abs().sum(-1)
    ok = (res < tol) & (inv_err < 1e-3) & (trace_err < 1e-10)
    return mu, V, W, ok.cpu().numpy(), it + 1


def geev_sweep(prob, lam):
    mus = []
    sync()
    t0 = time.perf_counter()
    for i in range(len(lam)):
        mus.append(torch.linalg.eig(prob.matrix(lam[i : i + 1]))[0][0])
    sync()
    return time.perf_counter() - t0, [m.cpu().numpy() for m in mus]


def reuse_sweep(prob, lam):
    """Refine i+2 from i, then i+1 from i; a failed step is halved, then uses eig."""
    n = len(lam)
    mus = [None] * n
    stats = dict(n_geev=0, iterations=0, refined=0)

    def anchor(i):
        mu, V = torch.linalg.eig(prob.matrix(lam[i : i + 1]))
        stats["n_geev"] += 1
        mus[i] = mu[0]
        return V, torch.linalg.inv(V)

    def step(k, V, W):
        A = prob.matrix(lam[k : k + 1])
        mu, Vn, Wn, ok, it = refine(A, V.clone(), W.clone())
        stats["iterations"] += it
        stats["refined"] += 1
        if ok[0]:
            mus[k] = mu[0]
        return Vn, Wn, ok[0]

    sync()
    t0 = time.perf_counter()
    i = 0
    V, W = anchor(0)
    while i < n - 1:
        S = 2
        while True:
            e = min(i + S, n - 1)
            Ve, We, ok = step(e, V, W)
            if ok:
                break
            if S == 1:
                Ve, We = anchor(e)
                break
            S = 1
        if e == i + 2 and not step(i + 1, V, W)[2]:
            anchor(i + 1)
        V, W, i = Ve, We, e
    sync()
    return time.perf_counter() - t0, [m.cpu().numpy() for m in mus], stats


def max_error(a, b):
    r, c = linear_sum_assignment(np.abs(a[:, None] - b[None, :]))
    return float((np.abs(a[r] - b[c]) / np.abs(b[c]).clip(1e-12)).max())


def peak_mib():
    return torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=7)
    a = ap.parse_args()
    prob = Problem(a.order)
    n = 2 * prob.N
    name = torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu"
    print(f"{name}, torch {torch.__version__}, order {a.order}, n={n}", flush=True)
    warm = torch.arange(450.0, 455.5, 1.0, dtype=RDT, device=DEV)
    geev_sweep(prob, warm[:2])
    reuse_sweep(prob, warm)
    for win, (lo, hi) in WINDOWS.items():
        lam = torch.arange(lo, hi + 0.5, 1.0, dtype=RDT, device=DEV)
        m = len(lam)
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_geev, ref = geev_sweep(prob, lam)
        peak_geev = peak_mib()
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_reuse, mus, st = reuse_sweep(prob, lam)
        err = max(max_error(mus[k], ref[k]) for k in range(m))
        print(
            f"{win:5s} {lo:.0f}-{hi:.0f} nm: eig {t_geev / m * 1e3:8.1f} ms/lam "
            f"({peak_geev:.0f} MiB) | reuse {t_reuse / m * 1e3:8.1f} ms/lam "
            f"({peak_mib():.0f} MiB) | {t_geev / t_reuse:5.1f}x | eig calls "
            f"{st['n_geev']}/{m}, {st['iterations'] / st['refined']:.2f} iter/step, "
            f"max rel. eigenvalue diff {err:.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
