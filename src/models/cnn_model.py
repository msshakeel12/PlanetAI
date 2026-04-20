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

    def __init__(
        self,
        input_length: int,
        model_mode: str = "legacy",
        global_input_length: int | None = None,
        local_input_length: int | None = None,
        odd_even_input_length: int | None = None,
        secondary_input_length: int | None = None,
        aux_feature_dim: int = 0,
        use_odd_even: bool = True,
        use_secondary: bool = True,
        use_aux_features: bool = True,
        fusion_hidden_dim: int = 96,
    ) -> None:
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
        self.model_mode = model_mode
        self.aux_feature_dim = max(0, int(aux_feature_dim))
        self.use_odd_even = bool(use_odd_even)
        self.use_secondary = bool(use_secondary)
        self.use_aux_features = bool(use_aux_features)

        if model_mode == "multiview":
            global_len = int(global_input_length if global_input_length is not None else input_length)
            local_len = int(local_input_length if local_input_length is not None else input_length)
            self.global_encoder = self._make_shared_encoder(in_channels=1)
            self.local_encoder = self._make_shared_encoder(in_channels=1)
            self.global_input_length = global_len
            self.local_input_length = local_len
            self.odd_even_input_length = (
                int(odd_even_input_length if odd_even_input_length is not None else local_len)
                if self.use_odd_even
                else None
            )
            self.secondary_input_length = (
                int(secondary_input_length if secondary_input_length is not None else local_len)
                if self.use_secondary
                else None
            )

            branch_embedding_dim = 48 * 8
            fusion_in_dim = branch_embedding_dim * 2
            if self.use_odd_even:
                self.odd_even_encoder = self._make_shared_encoder(in_channels=2)
                fusion_in_dim += branch_embedding_dim
            else:
                self.odd_even_encoder = None

            if self.use_secondary:
                self.secondary_encoder = self._make_shared_encoder(in_channels=1)
                fusion_in_dim += branch_embedding_dim
            else:
                self.secondary_encoder = None

            if self.use_aux_features and self.aux_feature_dim > 0:
                self.aux_mlp = nn.Sequential(
                    nn.Linear(self.aux_feature_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                fusion_in_dim += 16
            else:
                self.aux_mlp = None
                self.use_aux_features = False

            hidden_dim = max(16, int(fusion_hidden_dim))
            self.fusion_classifier = nn.Sequential(
                nn.Linear(fusion_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.global_encoder = None
            self.local_encoder = None
            self.odd_even_encoder = None
            self.secondary_encoder = None
            self.aux_mlp = None
            self.fusion_classifier = None
            self.use_odd_even = False
            self.use_secondary = False
            self.use_aux_features = False

    @staticmethod
    def _make_shared_encoder(in_channels: int = 1) -> nn.Sequential:
        """Return a compact shared 1D encoder used for global/local branches."""
        return nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )

    def forward(
        self,
        x: torch.Tensor | None = None,
        global_view: torch.Tensor | None = None,
        local_view: torch.Tensor | None = None,
        odd_even_view: torch.Tensor | None = None,
        secondary_view: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run forward pass and return logits.

        Args:
            x: Legacy tensor shaped ``(batch, channels=1, sequence_length)``.
            global_view: Optional global-view tensor for multiview mode.
            local_view: Optional local-view tensor for multiview mode.
            odd_even_view: Optional odd/even tensor for multiview mode.
            secondary_view: Optional secondary-view tensor for multiview mode.
            aux_features: Optional aux-feature tensor for multiview mode.

        Returns:
            Logit tensor shaped ``(batch, 1)``.
        """
        if self.model_mode == "multiview" and global_view is not None and local_view is not None:
            if self.global_encoder is None or self.local_encoder is None or self.fusion_classifier is None:
                raise RuntimeError("Multiview components are not initialized.")

            global_embed = self.global_encoder(global_view)
            local_embed = self.local_encoder(local_view)
            parts = [global_embed, local_embed]

            if self.use_odd_even:
                if self.odd_even_encoder is None or odd_even_view is None:
                    raise ValueError("odd_even_view is required when use_odd_even=True.")
                parts.append(self.odd_even_encoder(odd_even_view))

            if self.use_secondary:
                if self.secondary_encoder is None or secondary_view is None:
                    raise ValueError("secondary_view is required when use_secondary=True.")
                parts.append(self.secondary_encoder(secondary_view))

            if self.use_aux_features:
                if self.aux_mlp is None or aux_features is None:
                    raise ValueError("aux_features are required when use_aux_features=True.")
                parts.append(self.aux_mlp(aux_features))

            fused = torch.cat(parts, dim=1)
            return self.fusion_classifier(fused)

        if x is None:
            raise ValueError("Legacy forward path requires x when multiview inputs are absent.")

        features = self.feature_extractor(x)
        return self.classifier(features)
