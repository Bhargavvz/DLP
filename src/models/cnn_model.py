"""
DeepFinDLP - 1D-CNN Model
1D Convolutional Neural Network for network traffic feature extraction.
"""
import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """1D Convolutional Neural Network for tabular/sequential classification."""

    def __init__(self, num_features: int, num_classes: int,
                 channels: list = None, kernel_sizes: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if channels is None:
            channels = [128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]

        # Reshape input: (batch, features) -> (batch, 1, features)
        conv_layers = []
        in_channels = 1
        for i, (out_channels, k) in enumerate(zip(channels, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=k,
                          padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 2, 256),  # avg + max pool concat
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_block(x)

        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        features = torch.cat([avg_pool, max_pool], dim=1)

        return self.classifier(features)

    def get_features(self, x):
        """Extract features before classifier (for t-SNE)."""
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        return torch.cat([avg_pool, max_pool], dim=1)
