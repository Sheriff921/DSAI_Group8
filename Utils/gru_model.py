"""
GRU model definition.

This module contains the exact GRU architecture used during
training so the saved weights (.pt) can be loaded correctly.
"""

import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """
    GRU Network used for energy forecasting.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        """

        output, _ = self.gru(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output