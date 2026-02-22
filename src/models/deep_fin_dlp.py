"""
DeepFinDLP - Proposed Novel Architecture
Temporal Convolutional Transformer (TCT) with Squeeze-and-Excitation.

Architecture:
  Input → BatchNorm → 1D-CNN Block (3 layers) → BiLSTM (2 layers)
  → Multi-Head Self-Attention (8 heads) → SE Block → Residual FC → Softmax

Novel contributions:
  1. Hierarchical feature extraction: CNN (local) → LSTM (sequential) → Attention (global)
  2. Squeeze-and-Excitation channel recalibration for feature importance
  3. Residual connections in the classification head for gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature recalibration."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        b, c, _ = x.size()
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1)
        return x * se


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        B, S, E = x.size()

        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.attention_weights = attn_weights.detach()  # Save for visualization
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)

        return self.W_o(attn_output)


class ResidualBlock(nn.Module):
    """Residual FC block for the classification head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class DeepFinDLPModel(nn.Module):
    """
    DeepFinDLP: Temporal Convolutional Transformer (TCT) with SE Block.

    A novel architecture combining:
    - 1D Temporal Convolutions for local pattern extraction
    - Bidirectional LSTM for sequential dependency modeling
    - Multi-Head Self-Attention for global feature interactions
    - Squeeze-and-Excitation for channel recalibration
    - Residual classification head for robust prediction
    """

    def __init__(self, num_features: int, num_classes: int,
                 cnn_channels: list = None, cnn_kernel_sizes: list = None,
                 lstm_hidden: int = 256, lstm_layers: int = 2,
                 attention_heads: int = 8, attention_dim: int = 512,
                 se_reduction: int = 16,
                 fc_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [128, 256, 512]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [7, 5, 3]
        if fc_dims is None:
            fc_dims = [512, 256]

        self.num_features = num_features
        self.num_classes = num_classes

        # ═══ Input Normalization ═══
        self.input_norm = nn.BatchNorm1d(num_features)

        # ═══ 1D Temporal Convolutional Block ═══
        conv_layers = []
        in_channels = 1
        for i, (out_ch, k) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ])
            in_channels = out_ch
        self.temporal_conv = nn.Sequential(*conv_layers)

        # ═══ Squeeze-and-Excitation Block ═══
        self.se_block = SqueezeExcitationBlock(cnn_channels[-1], se_reduction)

        # ═══ Bidirectional LSTM ═══
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)

        # ═══ Multi-Head Self-Attention ═══
        # Project LSTM output to attention dimension
        self.attn_proj = nn.Linear(lstm_hidden * 2, attention_dim)
        self.multihead_attn = MultiHeadSelfAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(attention_dim)

        # ═══ Feature Aggregation ═══
        self.feature_pool = nn.AdaptiveAvgPool1d(1)

        # ═══ Residual Classification Head ═══
        head_layers = []
        in_dim = attention_dim
        for fc_dim in fc_dims:
            head_layers.append(ResidualBlock(in_dim, fc_dim, dropout))
            in_dim = fc_dim

        self.classification_head = nn.Sequential(*head_layers)
        self.final_classifier = nn.Linear(in_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # LSTM specific initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x):
        """
        Forward pass through the complete DeepFinDLP architecture.

        Args:
            x: (batch, num_features) - Input tensor
        Returns:
            logits: (batch, num_classes) - Class logits
        """
        # ═══ Input Normalization ═══
        x = self.input_norm(x)  # (batch, features)

        # ═══ 1D Temporal Convolution ═══
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.temporal_conv(x)  # (batch, channels, seq_len)

        # ═══ Squeeze-and-Excitation ═══
        x = self.se_block(x)  # (batch, channels, seq_len)

        # ═══ BiLSTM ═══
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = self.lstm_norm(lstm_out)

        # ═══ Multi-Head Self-Attention ═══
        attn_in = self.attn_proj(lstm_out)  # (batch, seq_len, attn_dim)
        attn_out = self.multihead_attn(attn_in)  # (batch, seq_len, attn_dim)
        attn_out = self.attn_norm(attn_out + attn_in)  # Residual connection

        # ═══ Feature Aggregation ═══
        attn_out = attn_out.permute(0, 2, 1)  # (batch, attn_dim, seq_len)
        pooled = self.feature_pool(attn_out).squeeze(-1)  # (batch, attn_dim)

        # ═══ Residual Classification Head ═══
        features = self.classification_head(pooled)
        logits = self.final_classifier(features)

        return logits

    def get_features(self, x):
        """Extract features before classifier (for t-SNE visualization)."""
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.se_block(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        attn_in = self.attn_proj(lstm_out)
        attn_out = self.multihead_attn(attn_in)
        attn_out = self.attn_norm(attn_out + attn_in)
        attn_out = attn_out.permute(0, 2, 1)
        pooled = self.feature_pool(attn_out).squeeze(-1)
        return self.classification_head(pooled)

    def get_attention_weights(self):
        """Get the last computed attention weights for visualization."""
        return self.multihead_attn.attention_weights

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"DeepFinDLP Model Summary")
        print(f"{'='*60}")
        print(f"  Input Features: {self.num_features}")
        print(f"  Output Classes: {self.num_classes}")
        print(f"  Total Parameters: {self.count_parameters():,}")
        print(f"  Trainable Parameters: {self.count_parameters():,}")
        print(f"{'='*60}")
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:25s} | {params:>10,} params")
        print(f"{'='*60}")
