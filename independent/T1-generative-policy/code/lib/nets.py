"""Small functional MLP so Jacobians see an explicit parameter vector."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FuncMLP:
    """ReLU MLP whose weights live in one flat tensor.

    forward(theta, x) is a pure function of theta. That is what the curl and
    sharpness estimators differentiate.
    """

    def __init__(self, sizes: list[int], device: torch.device, seed: int = 0):
        self.sizes = list(sizes)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        chunks: list[torch.Tensor] = []
        self.layout: list[tuple[slice, slice, int, int]] = []
        cursor = 0
        for din, dout in zip(sizes[:-1], sizes[1:]):
            weight = torch.randn(dout, din, generator=generator) * math.sqrt(2.0 / din)
            bias = torch.zeros(dout)
            chunks.append(weight.reshape(-1))
            chunks.append(bias.reshape(-1))
            weight_slice = slice(cursor, cursor + dout * din)
            cursor += dout * din
            bias_slice = slice(cursor, cursor + dout)
            cursor += dout
            self.layout.append((weight_slice, bias_slice, dout, din))
        self.init_theta = torch.cat(chunks).to(device)
        self.theta = self.init_theta.clone()
        self.dim = cursor
        self.device = device

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        last = len(self.layout) - 1
        for index, (weight_slice, bias_slice, dout, din) in enumerate(self.layout):
            weight = theta[weight_slice].view(dout, din)
            bias = theta[bias_slice]
            hidden = hidden @ weight.T + bias
            if index != last:
                hidden = torch.nn.functional.silu(hidden)
        return hidden

    def hypersphere_project(self, theta: torch.Tensor) -> torch.Tensor:
        """Rescale each weight matrix to its initialization Frobenius norm.

        A cheap stand-in for the hyperspherical normalization in Simba / SimbaV2.
        Biases are left alone.
        """
        pieces = [theta]
        # Rebuild by copying so autograd history is not required (called after the step).
        out = theta.detach().clone()
        for weight_slice, _, _, _ in self.layout:
            current = out[weight_slice]
            init = self.init_theta[weight_slice]
            scale = init.norm() / current.norm().clamp_min(1e-8)
            out[weight_slice] = current * scale
        del pieces
        return out


class MLP(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for index, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
            linear = nn.Linear(din, dout)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0) if index < len(sizes) - 2 else 1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(sizes) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
