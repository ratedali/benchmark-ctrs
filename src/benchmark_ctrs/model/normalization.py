# This file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
#
# Github Permalink: https://github.com/alinlab/smoothing-catrs/blob/d4bc576e7d373d158f087ba5744af8bb48466bb7/code/datasets.py#L257
from __future__ import annotations

import torch


def normalize(
    x: torch.Tensor,
    mean: torch.Tensor | None = None,
    sd: torch.Tensor | None = None,
):
    if mean is None:
        mean = torch.tensor([0.0])

    if sd is None:
        sd = torch.tensor([1.0])

    chls_i = x.dim() - 3  # third-to-last dimension
    shape = [1] * x.dim()
    shape[chls_i] = x.size(chls_i)

    return (x - mean.view(shape)) / sd.view(shape)


class Normalization(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized
    coordinates, we add the Gaussian noise _before_ standardizing, which is why
    we have standardization be the first layer of the classifier rather than
    as a part of preprocessing as is typical
    """

    def __init__(
        self,
        mean: list[float] | None = None,
        sd: list[float] | None = None,
    ):
        super().__init__()
        self.mean = torch.nn.Parameter(
            torch.tensor(mean, dtype=torch.float),
            requires_grad=False,
        )
        self.sd = torch.nn.Parameter(
            torch.tensor(sd, dtype=torch.float),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        return normalize(x, self.mean, self.sd)
