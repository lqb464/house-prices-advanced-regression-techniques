"""GRU regressor nhỏ cho chuỗi return dài 20 phiên."""

from __future__ import annotations

import torch
from torch import nn


class ReturnGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(features)
        return self.head(output[:, -1]).squeeze(1)
