import torch
import torch.nn as nn


class SpatialSenseModel(nn.Module):
    def __init__(
        self,
        sensing_range: float,
    ):
        super(SpatialSenseModel, self).__init__()

        # Compute k_values via an inverse sigmoid and register them as a buffer
        self.register_buffer(
            "k", self.inverse_sigmoid(sensing_range, torch.tensor([0.01]))
        )

        self.delta_proj = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1),
        )

        # Initialize parameters
        self._initialize_parameters()

    def inverse_sigmoid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-1 / x) * torch.log(y / (2 - y))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

    # Initialize the parameters of the model
    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input):
        batch_size = 1
        feature_size = 180

        sample = input[:feature_size]
        sample_next = input[feature_size:]

        sample_sigmoid = self.sigmoid(sample)
        sample_next_sigmoid = self.sigmoid(sample_next)

        delta = sample_next_sigmoid - sample_sigmoid

        # Apply the same Linear layer to all features independently
        # Reshape to (batch_size * feature_size, 1), apply Linear, then reshape back
        delta = delta.view(-1, 1)  # Reshape to apply shared linear to each feature
        delta = self.delta_proj(delta)  # Apply the same Linear layer
        delta = delta.view(batch_size, feature_size)  # Reshape back to original shape

        spatial_sense = delta * sample_next_sigmoid

        return spatial_sense.squeeze(0)
