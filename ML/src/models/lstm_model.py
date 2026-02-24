"""
DeepFinDLP - BiLSTM Model
Bidirectional LSTM for capturing sequential dependencies in network traffic.
"""
import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for sequence-based classification."""

    def __init__(self, num_features: int, num_classes: int,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Classifier
        lstm_out_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # (batch, seq_len=1, hidden)

        # Repeat to create a short sequence for LSTM
        x = x.repeat(1, 4, 1)  # (batch, 4, hidden)

        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last layer, forward
            h_backward = h_n[-1]  # Last layer, backward
            features = torch.cat([h_forward, h_backward], dim=1)
        else:
            features = h_n[-1]

        features = self.layer_norm(features)
        return self.classifier(features)

    def get_features(self, x):
        """Extract features before classifier (for t-SNE)."""
        x = self.input_proj(x)
        x = x.unsqueeze(1).repeat(1, 4, 1)
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]
