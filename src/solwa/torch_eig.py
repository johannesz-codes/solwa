import torch

"""
PyTorch eigendecomposition with numerical stability.

Complex domain eigen-decomposition with stabilized gradient computation
for use in differentiable RCWA simulations. Requires PyTorch 1.10.1 or higher.
"""


class Eig(torch.autograd.Function):
    """
    Custom PyTorch autograd function for stable eigendecomposition.

    Implements forward and backward passes for eigenvalue decomposition with
    Lorentzian broadening to stabilize gradient computation near degenerate
    eigenvalues.

    Attributes
    ----------
    broadening_parameter : float
        Lorentzian broadening parameter for gradient stabilization.
        Default is 1e-10. Set to None to use machine epsilon.
    """

    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx, x, offload_device=None):
        """
        Forward pass: compute eigenvalues and eigenvectors.

        Parameters
        ----------
        ctx : context object
            PyTorch context for saving variables for backward pass.
        x : torch.Tensor
            Input square matrix for eigendecomposition.
        offload_device : torch.device or None, optional
            Where to keep the eigenpairs until the backward pass. Default is
            None (the device of ``x``).

        Returns
        -------
        tuple of torch.Tensor
            (eigval, eigvec) eigenvalues and eigenvectors of the input matrix.
        """
        ctx.input = x
        eigval, eigvec = torch.linalg.eig(x)
        # Never store the outputs themselves: that would be a reference cycle.
        ctx.eigval, ctx.eigvec = eigval.detach(), eigvec.detach()
        if offload_device is not None:
            ctx.eigval = ctx.eigval.to(offload_device)
            ctx.eigvec = ctx.eigvec.to(offload_device)
        return eigval, eigvec

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        """
        Backward pass: compute gradient with respect to input matrix.

        Uses Lorentzian broadening to stabilize gradients near degenerate
        eigenvalues, reducing numerical instability in the gradient computation.
        The broadening biases the coupling of eigenvalues closer than about
        sqrt(broadening_parameter) and drops it for exactly degenerate ones,
        so gradients that split a degeneracy are inaccurate (#91).

        Parameters
        ----------
        ctx : context object
            PyTorch context containing saved variables from forward pass.
        grad_eigval : torch.Tensor
            Gradient of the loss with respect to eigenvalues.
        grad_eigvec : torch.Tensor
            Gradient of the loss with respect to eigenvectors.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input matrix.
        """
        eigval = ctx.eigval.to(grad_eigval)
        eigvec = ctx.eigvec.to(grad_eigvec)
        grad = _eig_vjp(eigval, eigvec, grad_eigval, grad_eigvec)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)

        return grad, None  # no gradient for offload_device


def _eig_vjp(eigval, eigvec, grad_eigval, grad_eigvec):
    """Eigenpair backward with Lorentzian broadening (see ``Eig.backward``)."""
    s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

    # Lorentzian broadening: get small error but stabilizing the gradient calculation
    if Eig.broadening_parameter is not None:
        F = torch.conj(s) / (torch.abs(s) ** 2 + Eig.broadening_parameter)
    elif s.dtype == torch.complex64:
        F = torch.conj(s) / (torch.abs(s) ** 2 + 1.4e-45)
    else:
        F = torch.conj(s) / (torch.abs(s) ** 2 + 4.9e-324)
    F.diagonal(dim1=-2, dim2=-1).zero_()

    XH = eigvec.mH
    tmp = torch.diag_embed(grad_eigval) + torch.conj(F) * torch.matmul(XH, grad_eigvec)
    return torch.matmul(torch.matmul(torch.inverse(XH), tmp), XH)


def _kz(eigval):
    """Normalized kz = sqrt(eigval) on the branch with Im(kz) >= 0."""
    kz = torch.sqrt(eigval)
    return torch.where(torch.imag(kz) < 0, -kz, kz)


class EigFunc(torch.autograd.Function):
    """
    Eigendecomposition of ``A = PQ`` that also returns the layer's basis-free
    matrix functions ``K = sqrt(A)`` and ``Phi = exp(i c K)`` (#91).

    The eigenpairs keep the ``Eig`` backward (fields, ``return_layer``). ``K`` and
    ``Phi`` do not depend on the eigenvector basis, so their backward uses the
    Daleckii-Krein formula with closed-form divided differences; it needs no
    broadening and stays exact at degenerate eigenvalues.
    """

    @staticmethod
    def forward(ctx, x, c, offload_device=None):
        eigval, eigvec = torch.linalg.eig(x)
        kz = _kz(eigval)
        c = c.to(kz.real.dtype)
        cc = c[..., None] if c.dim() else c
        ph = torch.exp(1j * cc * kz)
        eigvec_inv = torch.linalg.inv(eigvec)
        K = torch.matmul(eigvec * kz[..., None, :], eigvec_inv)
        Phi = torch.matmul(eigvec * ph[..., None, :], eigvec_inv)
        saved = [t.detach() for t in (eigval, eigvec, eigvec_inv, kz, ph)]
        if offload_device is not None:
            saved = [t.to(offload_device) for t in saved]
        ctx.saved, ctx.c = saved, c.detach()
        ctx.set_materialize_grads(False)
        return eigval, eigvec, eigvec_inv, K, Phi

    @staticmethod
    def backward(ctx, g_val, g_vec, g_inv, g_K, g_Phi):
        ref = next(g for g in (g_K, g_Phi, g_vec, g_val, g_inv) if g is not None)
        eigval, X, Xinv, kz, ph = (t.to(ref.device) for t in ctx.saved)
        c = ctx.c.to(ref.device)
        XH, XinvH = X.mH, Xinv.mH
        grad = torch.zeros_like(X)

        # Eigenpair outputs (Xinv = X^-1 feeds into the eigenvector gradient)
        if g_inv is not None:
            g_vec = (0 if g_vec is None else g_vec) - XinvH @ g_inv @ XinvH
        if g_val is not None or g_vec is not None:
            grad = grad + _eig_vjp(
                eigval,
                X,
                torch.zeros_like(eigval) if g_val is None else g_val,
                torch.zeros_like(X) if g_vec is None else g_vec,
            )

        # Matrix functions: dF = X (D o (X^-1 dA X)) X^-1, D = divided differences
        g_c = None
        if g_K is not None or g_Phi is not None:
            ki, kj = kz[..., :, None], kz[..., None, :]
            ksum = ki + kj  # lambda_i - lambda_j = (kz_i - kz_j) * ksum
            inner = 0
            if g_K is not None:
                inner = (1 / ksum).conj() * (XH @ g_K @ XinvH)
            if g_Phi is not None:
                cc = c[..., None, None] if c.dim() else c
                # (ph_i - ph_j) / (k_i - k_j); sinc form only where cancellation
                # hurts (|x| < 1), since sinc and exp over/underflow for large |x|
                x = cc * (ki - kj) / 2
                small = x.abs() < 1
                xs = torch.where(small, x, torch.zeros_like(x))
                D = torch.where(
                    small,
                    torch.exp(0.5j * cc * ksum) * 1j * cc * torch.sinc(xs / torch.pi),
                    (ph[..., :, None] - ph[..., None, :])
                    / torch.where(small, 1, ki - kj),
                )
                inner = inner + (D / ksum).conj() * (XH @ g_Phi @ XinvH)
                if ctx.needs_input_grad[1]:
                    dPhi = (X * (1j * kz * ph)[..., None, :]) @ Xinv
                    g_c = (g_Phi * dPhi.conj()).sum((-2, -1)).real
            grad = grad + XinvH @ inner @ XH
        return grad, g_c, None
