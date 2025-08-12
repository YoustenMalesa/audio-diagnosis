import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for log-mel spectrogram inputs.

    Input shape: (B, 1, n_mels, T)
    """

    def __init__(self, n_mels: int = 128, n_classes: int = 8, cnn_channels=(32, 64, 128), rnn_hidden=128, rnn_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in cnn_channels:
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d((2, 2)))  # halves both dims
            layers.append(nn.Dropout(p=dropout))
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        # After CNN, frequency dim reduced by 2**len(cnn_channels)
        self.freq_reduction = 2 ** len(cnn_channels)
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(
            input_size=cnn_channels[-1] * (n_mels // self.freq_reduction),
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        h = self.cnn(x)  # (B, C, n_mels', T')
        B, C, F, T = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T, C*F)
        rnn_out, _ = self.rnn(h)  # (B, T, 2*hidden)
        # Temporal average pooling
        pooled = rnn_out.mean(dim=1)
        return self.classifier(pooled)
