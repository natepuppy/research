"""Curl of a parameter-space vector field.

For a vector field f, the Jacobian J = Df splits into a symmetric part S and a
skew-symmetric part A = (J - J^T) / 2. A is the rotational (non-conservative)
component. With v ~ N(0, I),

    E[||(J - J^T) v||^2] = ||J - J^T||_F^2 = 4 ||A||_F^2
    E[||J v||^2] = ||J||_F^2 = ||S||_F^2 + ||A||_F^2

so ||A||_F^2 / ||J||_F^2 is the fraction of the Jacobian that is rotational.
A pure gradient has fraction 0. A pure rotation has fraction 1.
"""

from __future__ import annotations

import torch


def curl_of_field(field, theta: torch.Tensor, probes: int = 2) -> dict[str, float]:
    """Stochastic estimate of the rotational fraction of Df at theta.

    `field` maps a 1-D tensor to a 1-D tensor of the same size. It may itself
    call autograd (the TD semi-gradient does).
    """
    theta = theta.detach()
    rotational = 0.0
    jacobian = 0.0
    for _ in range(probes):
        probe = torch.randn_like(theta)
        _, tangent = torch.autograd.functional.jvp(field, theta, probe)
        _, cotangent = torch.autograd.functional.vjp(field, theta, probe)
        gap = tangent - cotangent
        rotational += float(gap.pow(2).sum())
        jacobian += float(tangent.pow(2).sum())
    rotational /= probes
    jacobian /= probes
    anti_sq = rotational / 4.0
    fraction = anti_sq / (jacobian + 1e-12)
    return {
        "curl_frob": anti_sq**0.5,
        "curl_fraction": float(fraction),
        "jacobian_frob": jacobian**0.5,
    }


def rotational_component(field, theta: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(J - J^T) v / 2, the rotational part of J applied to `vector`."""
    theta = theta.detach()
    vector = vector.detach()
    _, tangent = torch.autograd.functional.jvp(field, theta, vector)
    _, cotangent = torch.autograd.functional.vjp(field, theta, vector)
    return 0.5 * (tangent - cotangent)


def _head(net, weights: torch.Tensor, inputs: torch.Tensor, actions: torch.Tensor | None, mode: str) -> torch.Tensor:
    out = net.forward(weights, inputs)
    if mode == "gather":
        return out.gather(1, actions[:, None]).squeeze(1)
    if mode == "max":
        return out.max(dim=1).values
    if mode == "scalar":
        return out.reshape(out.shape[0], -1)[:, 0]
    raise ValueError(mode)


def _semi_grad(net, theta_leaf, omega, states, next_states, rewards, dones, gamma, actions, pred_mode, boot_mode):
    """Gradient of the squared TD loss w.r.t. theta only.

    The backup is computed from `omega`. When omega is a separate leaf that
    holds the same values as theta, differentiating the gradient w.r.t. both
    leaves is the Jacobian of the semi-gradient field theta |-> grad_theta.
    Detaching the backup inside the field would drop that second term and
    leave a symmetric Hessian, whose curl is zero by construction.
    """
    prediction = _head(net, theta_leaf, states, actions, pred_mode)
    bootstrap = _head(net, omega, next_states, None, boot_mode)
    backup = rewards + gamma * (1.0 - dones) * bootstrap
    loss = (prediction - backup).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, theta_leaf, create_graph=True)
    return grad


def _td_products(
    net,
    theta: torch.Tensor,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    probe: torch.Tensor,
    actions: torch.Tensor | None,
    pred_mode: str,
    boot_mode: str,
    online: bool,
    target: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (semi-gradient, J v, J^T v) of the TD field at theta.

    online=True differentiates the bootstrap through a leaf that tracks theta.
    online=False freezes `target`, which makes J symmetric.
    """
    theta = theta.detach()
    probe = probe.detach()
    theta_leaf = theta.requires_grad_(True)
    if online:
        omega = theta.detach().clone().requires_grad_(True)
    else:
        if target is None:
            raise ValueError("a frozen-target curl needs target weights")
        omega = target.detach()
    grad = _semi_grad(
        net, theta_leaf, omega, states, next_states, rewards, dones, gamma, actions, pred_mode, boot_mode
    )
    dotted = (grad * probe).sum()
    (h_theta,) = torch.autograd.grad(dotted, theta_leaf, retain_graph=online)
    if online:
        (h_omega,) = torch.autograd.grad(dotted, omega, retain_graph=False)

        def grad_of_omega(omega_in: torch.Tensor) -> torch.Tensor:
            theta_fixed = theta.detach().requires_grad_(True)
            return _semi_grad(
                net,
                theta_fixed,
                omega_in,
                states,
                next_states,
                rewards,
                dones,
                gamma,
                actions,
                pred_mode,
                boot_mode,
            )

        _, mixed = torch.autograd.functional.jvp(grad_of_omega, theta.detach(), probe)
        jvp = h_theta + mixed
        vjp = h_theta + h_omega
    else:
        jvp = h_theta
        vjp = h_theta
    return grad.detach(), jvp.detach(), vjp.detach()


def td_curl(
    net,
    theta: torch.Tensor,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    actions: torch.Tensor | None = None,
    pred_mode: str = "gather",
    boot_mode: str = "max",
    probes: int = 2,
    online: bool = True,
    target: torch.Tensor | None = None,
) -> dict[str, float]:
    """Rotational fraction of the semi-gradient TD field. See curl_of_field."""
    rotational = 0.0
    jacobian = 0.0
    for _ in range(probes):
        probe = torch.randn_like(theta)
        _, jvp, vjp = _td_products(
            net,
            theta,
            states,
            next_states,
            rewards,
            dones,
            gamma,
            probe,
            actions,
            pred_mode,
            boot_mode,
            online,
            target,
        )
        rotational += float((jvp - vjp).pow(2).sum())
        jacobian += float(jvp.pow(2).sum())
    rotational /= probes
    jacobian /= probes
    anti_sq = rotational / 4.0
    return {
        "curl_frob": anti_sq**0.5,
        "curl_fraction": float(anti_sq / (jacobian + 1e-12)),
        "jacobian_frob": jacobian**0.5,
    }


def td_rotational_component(
    net,
    theta: torch.Tensor,
    vector: torch.Tensor,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    actions: torch.Tensor | None = None,
    pred_mode: str = "gather",
    boot_mode: str = "max",
    online: bool = True,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """(J - J^T) v / 2 for the semi-gradient TD field."""
    _, jvp, vjp = _td_products(
        net,
        theta,
        states,
        next_states,
        rewards,
        dones,
        gamma,
        vector,
        actions,
        pred_mode,
        boot_mode,
        online,
        target,
    )
    return 0.5 * (jvp - vjp)
