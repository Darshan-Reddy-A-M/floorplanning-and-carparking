from __future__ import annotations

import torch
from torch import nn


class RoomBoxRegressor(nn.Module):
    """
    Predicts a target room's normalized bounding box using:
      1. The target room category.
      2. The categories of other rooms.
      3. The bounding boxes of other rooms.

    The target room's actual bounding box must never be supplied
    as an input.
    """

    def __init__(
        self,
        num_labels: int,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_labels = num_labels

        self.label_embedding = nn.Embedding(
            num_labels,
            embedding_dim,
        )

        # Each context room contains an embedding and four coordinates.
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean-pooled context + target room embedding.
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        target_label_ids: torch.Tensor,
        context_label_ids: torch.Tensor,
        context_boxes: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_label_ids: (batch,)
            context_label_ids: (batch, max_context_rooms)
            context_boxes: (batch, max_context_rooms, 4)
            context_mask: (batch, max_context_rooms)
                          True for real context rooms.

        Returns:
            Predicted normalized bounding boxes (batch, 4)
            in x1, y1, x2, y2 format.
        """

        target_features = self.label_embedding(
            target_label_ids
        )

        context_features = self.label_embedding(
            context_label_ids
        )

        context_features = torch.cat(
            [context_features, context_boxes],
            dim=-1,
        )

        encoded_context = self.context_encoder(
            context_features
        )

        mask = context_mask.unsqueeze(-1).to(
            encoded_context.dtype
        )

        # Ignore padded context entries.
        encoded_context = encoded_context * mask

        context_count = mask.sum(dim=1).clamp(min=1.0)

        pooled_context = (
            encoded_context.sum(dim=1) / context_count
        )

        features = torch.cat(
            [target_features, pooled_context],
            dim=-1,
        )

        raw_boxes = self.predictor(features)

        # Keep individual coordinates between 0 and 1.
        coordinates = torch.sigmoid(raw_boxes)

        # Enforce x1 <= x2 and y1 <= y2.
        x1 = torch.minimum(
            coordinates[:, 0],
            coordinates[:, 2],
        )

        y1 = torch.minimum(
            coordinates[:, 1],
            coordinates[:, 3],
        )

        x2 = torch.maximum(
            coordinates[:, 0],
            coordinates[:, 2],
        )

        y2 = torch.maximum(
            coordinates[:, 1],
            coordinates[:, 3],
        )

        return torch.stack(
            [x1, y1, x2, y2],
            dim=1,
        )