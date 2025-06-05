import torch
import torch.nn as nn


class SpatialSenseBlock(nn.Module):
    def __init__(self, sensing_range: float, num_features: int = 360):
        super(SpatialSenseBlock, self).__init__()

        # Create a tensor filled with the same value, but shaped to match sample_sigmoid
        k_init_value = self.inverse_sigmoid(sensing_range, torch.tensor([0.01]))

        self.k = nn.Parameter(torch.full((num_features,), k_init_value.item()))

        self.delta_proj = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1),
        )

        self._initialize_parameters()

    def inverse_sigmoid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (-1 / x) * torch.log(y / (2 - y))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        # k needs to be broadcast to match x's shape
        return (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input):
        batch_size = input.size(0)
        feature_size = input.size(1) // 2

        sample = input[:, :feature_size]
        sample_next = input[:, feature_size:]

        sample_sigmoid = self.sigmoid(sample)
        sample_next_sigmoid = self.sigmoid(sample_next)

        delta = sample_next_sigmoid - sample_sigmoid

        delta = delta.view(-1, 1)
        delta = self.delta_proj(delta)
        delta = delta.view(batch_size, feature_size)

        spatial_sense = delta * sample_next_sigmoid

        return spatial_sense


class Es2Model(nn.Module):
    def __init__(self, num_features=360, num_actions=2, sensing_range=10.0):
        super(Es2Model, self).__init__()

        self.spatial_sense_block = SpatialSenseBlock(sensing_range=sensing_range)

        self.sense_action_layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_actions),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input):
        embedding = self.spatial_sense_block(input)
        action = self.sense_action_layers(embedding)
        return action
