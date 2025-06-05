import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer-based model for autonomous racing using LiDAR data.
    Processes pairs of consecutive LiDAR scans to predict steering and speed.
    """

    def __init__(
        self,
        num_features=360,
        num_actions=2,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        sensing_range=10.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.sensing_range = sensing_range

        self.embedding_projection = nn.Linear(1, d_model)

        # Positional encoding for angular positions
        self.register_buffer(
            "positional_encoding", self._generate_positional_encoding()
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Output layer for predicting actions
        self.action_proj = nn.Sequential(
            nn.Linear(num_features * 2, num_actions),
        )

        self._initialize_parameters()

    def _generate_positional_encoding(self):
        """
        Generate sinusoidal positional encodings for angular positions in LiDAR scan.
        """
        encoding = torch.zeros(self.num_features * 2, self.d_model)
        position = torch.arange(self.num_features * 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding  # shape: [num_features * 2, d_model]

    def _initialize_parameters(self):
        """
        Initialize weights using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Tensor of shape [batch_size, num_features * 2]
        Returns:
            action: Tensor of shape [batch_size, num_actions]
        """

        x = x.unsqueeze(-1)  # [batch, num_features * 2 , 1]
        x = self.embedding_projection(x)  # [batch, num_features * 2, d_model]

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)  # [1, num_features * 2, d_model]

        x = self.transformer_encoder(x)  # [batch, num_features * 2, d_model]

        x = x.mean(dim=2)  # Mean pool across sequence length -> [batch, num_features]

        action = self.action_proj(x)  # [batch, num_actions]

        return action
