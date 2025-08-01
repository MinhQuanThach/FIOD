import torch
import torch.nn as nn


class FogPassFilter(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, output_dim=128):
        """
        Fog-Pass Filter MLP for extracting fog factors from Gram matrices.

        Args:
            input_dim (int): Size of flattened upper triangular Gram matrix.
            hidden_dim1 (int): Size of first hidden layer.
            hidden_dim2 (int): Size of second hidden layer.
            output_dim (int): Size of output fog factor vector.
        """
        super(FogPassFilter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        """
        Compute fog factors from input feature maps.

        Args:
            x (torch.Tensor): Feature maps of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Fog factors of shape (batch_size, output_dim).
        """
        # Compute Gram matrix
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        gram = torch.bmm(x, x.transpose(1, 2))  # Shape: (batch_size, channels, channels)

        # Get upper triangular part (excluding diagonal to reduce size)
        triu_indices = torch.triu_indices(channels, channels, offset=1)
        gram_flat = gram[:, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, triu_size)

        # Pass through MLP
        fog_factors = self.mlp(gram_flat)
        return fog_factors