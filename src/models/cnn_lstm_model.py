"""
DeepFinDLP - CNN-BiLSTM Hybrid Model
Combines 1D-CNN for local feature extraction with BiLSTM for sequential modeling.
"""
import torch
import torch.nn as nn


class CNNBiLSTMModel(nn.Module):
    """Hybrid CNN-BiLSTM for combined local and sequential pattern recognition."""

    def __init__(self, num_features: int, num_classes: int,
                 cnn_channels: list = None, cnn_kernel_sizes: list = None,
                 lstm_hidden: int = 256, lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [128, 256]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [5, 3]

        # 1D-CNN Feature Extractor
        conv_layers = []
        in_channels = 1
        for out_ch, k in zip(cnn_channels, cnn_kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
            ])
            in_channels = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # BiLSTM Sequence Modeler
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)

        # Attention mechanism for sequence summarization
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
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

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features) -> CNN
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)  # (batch, channels, seq_len)

        # Permute for LSTM: (batch, seq_len, channels)
        lstm_in = cnn_out.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (batch, seq_len, hidden*2)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        features = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        features = self.layer_norm(features)
        return self.classifier(features)

    def get_features(self, x):
        """Extract features before classifier (for t-SNE)."""
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        features = torch.sum(attn_weights * lstm_out, dim=1)
        return self.layer_norm(features)
