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
    def forward(ctx, x):
        """
        Forward pass: compute eigenvalues and eigenvectors.

        Parameters
        ----------
        ctx : context object
            PyTorch context for saving variables for backward pass.
        x : torch.Tensor
            Input square matrix for eigendecomposition.

        Returns
        -------
        tuple of torch.Tensor
            (eigval, eigvec) eigenvalues and eigenvectors of the input matrix.
        """
        ctx.input = x
        eigval, eigvec = torch.linalg.eig(x)
        ctx.eigval = eigval.cpu()
        ctx.eigvec = eigvec.cpu()
        return eigval, eigvec

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        """
        Backward pass: compute gradient with respect to input matrix.

        Uses Lorentzian broadening to stabilize gradients near degenerate
        eigenvalues, reducing numerical instability in the gradient computation.

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

        grad_eigval = torch.diag(grad_eigval)
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        # Lorentzian broadening: get small error but stabilizing the gradient calculation
        if Eig.broadening_parameter is not None:
            F = torch.conj(s) / (torch.abs(s) ** 2 + Eig.broadening_parameter)
        elif s.dtype == torch.complex64:
            F = torch.conj(s) / (torch.abs(s) ** 2 + 1.4e-45)
        elif s.dtype == torch.complex128:
            F = torch.conj(s) / (torch.abs(s) ** 2 + 4.9e-324)

        diag_indices = torch.linspace(
            0, F.shape[-1] - 1, F.shape[-1], dtype=torch.int64
        )
        F[diag_indices, diag_indices] = 0.0
        XH = torch.transpose(torch.conj(eigvec), -2, -1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(torch.matmul(torch.inverse(XH), grad_eigval + tmp), XH)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)

        return grad
