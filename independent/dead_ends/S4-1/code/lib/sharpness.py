"""Top Hessian eigenvalue by Lanczos on Hessian-vector products."""

from __future__ import annotations

import torch


def top_hessian_eigenvalue(hvp, dim: int, steps: int = 20, device: torch.device | None = None) -> float:
    """Largest algebraic eigenvalue of a symmetric operator given by `hvp`.

    Full reorthogonalization keeps the tridiagonal faithful for the small
    networks these experiments use.
    """
    if device is None:
        device = torch.device("cpu")
    q = torch.randn(dim, device=device)
    q = q / q.norm().clamp_min(1e-12)
    q_prev = torch.zeros_like(q)
    alphas: list[float] = []
    betas: list[float] = []
    stored: list[torch.Tensor] = []
    beta = torch.zeros((), device=device)
    for _ in range(steps):
        z = hvp(q)
        alpha = torch.dot(q, z)
        z = z - alpha * q - beta * q_prev
        for previous in stored:
            z = z - torch.dot(previous, z) * previous
        beta_next = z.norm()
        alphas.append(float(alpha))
        stored.append(q)
        if float(beta_next) < 1e-8:
            break
        betas.append(float(beta_next))
        q_prev = q
        q = z / beta_next
        beta = beta_next
    width = len(alphas)
    tri = torch.zeros(width, width)
    for index, alpha in enumerate(alphas):
        tri[index, index] = alpha
    for index, beta_value in enumerate(betas[: width - 1]):
        tri[index, index + 1] = beta_value
        tri[index + 1, index] = beta_value
    eigenvalues = torch.linalg.eigvalsh(tri)
    return float(eigenvalues.max())
