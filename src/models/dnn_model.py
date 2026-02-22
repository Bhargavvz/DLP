"""
DeepFinDLP - Basic DNN Model
3-layer MLP baseline for network traffic classification.
"""
import torch
import torch.nn as nn


class BasicDNN(nn.Module):
    """Basic Deep Neural Network (MLP) baseline."""

    def __init__(self, num_features: int, num_classes: int,
                 hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract features before the classifier (for t-SNE)."""
        return self.backbone(x)
