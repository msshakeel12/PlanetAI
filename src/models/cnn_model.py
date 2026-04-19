"""1D CNN architecture for sequence-based transit classification.

This module defines the neural network used by ``train_cnn.py``. The design is
intentionally lightweight to provide a research baseline that trains quickly on
fixed-length sequences.

Main public class:
- ``TransitCNN``
"""

from __future__ import annotations

import torch
from torch import nn


class TransitCNN(nn.Module):
    """Simple 1D CNN for binary transit classification.

    Args:
        input_length: Expected sequence length.

    Notes:
        ``input_length`` is stored for metadata/traceability; adaptive pooling
        makes the downstream dense block less sensitive to exact length.
    """

    def __init__(self, input_length: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        flattened_size = 96 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 96),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(96, 1),
        )

        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits.

        Args:
            x: Tensor shaped ``(batch, channels=1, sequence_length)``.

        Returns:
            Logit tensor shaped ``(batch, 1)``.
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
