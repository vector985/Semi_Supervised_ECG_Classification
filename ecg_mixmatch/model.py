"""GSConv/ECA feature extractor and bidirectional GRU."""

import math

import torch
from torch import nn

from . import CLASS_NAMES


class ECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = int(abs((math.log2(channels) + 1) / 2))
        kernel += 1 - kernel % 2
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.conv(self.avg_pool(x).transpose(-1, -2))
        return x * self.sigmoid(weights).transpose(-1, -2)


class GSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv1d(in_channels, mid, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid)
        self.dwconv = nn.Conv1d(
            mid, mid, kernel_size, padding=kernel_size // 2, groups=mid, bias=False
        )
        self.bn2 = nn.BatchNorm1d(mid)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        primary = self.relu(self.bn1(self.conv1(x)))
        depthwise = self.relu(self.bn2(self.dwconv(primary)))
        # Interleave the primary and depthwise channels.
        return torch.stack((primary, depthwise), dim=2).flatten(1, 2)


class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.gs_main = GSConv(in_channels, in_channels, kernel_size=1, stride=2)
        mid = in_channels // 4
        self.a = GSConv(mid, mid)
        self.b = GSConv(mid, mid)
        self.c = GSConv(mid, mid)
        self.down = GSConv(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1, x2, x3, x4 = self.gs_main(x).chunk(4, dim=1)
        y1 = self.a(x1)
        y2 = self.b(x2 + y1)
        y3 = self.c(x3 + y2)
        return self.dropout(self.down(torch.cat((y1, y2, y3, x4), dim=1)))


class ECGCNN(nn.Module):
    """Single-lead ECG classifier with configurable channel and GRU widths."""

    def __init__(self, *, base_channels=32, gru_hidden=128, dropout=0.5):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        c1, c2, c3, c4, c5 = (base_channels * 2**stage for stage in range(5))
        self.conv1 = nn.Conv1d(1, c1, 9, padding=4)
        self.bn1 = nn.BatchNorm1d(c1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout * 0.3)
        self.ms1 = MSBlock(c1, c2, dropout * 0.3)
        self.eca1 = ECABlock(c2)
        self.ms2 = MSBlock(c2, c3, dropout * 0.4)
        self.eca2 = ECABlock(c3)
        self.ms3 = MSBlock(c3, c4, dropout * 0.5)
        self.eca3 = ECABlock(c4)
        self.ms4 = MSBlock(c4, c5, dropout * 0.6)
        self.eca4 = ECABlock(c5)
        self.gru = nn.GRU(c5, gru_hidden, batch_first=True, bidirectional=True)
        self.drop_rnn = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * gru_hidden, len(CLASS_NAMES))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] < 2:
            raise ValueError("Expected single-lead ECG batches with at least two time points")
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.eca1(self.ms1(x))
        x = self.eca2(self.ms2(x))
        x = self.eca3(self.ms3(x))
        x = self.eca4(self.ms4(x))
        # Keep the GRU in FP32 when CUDA autocast is enabled.
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, hidden = self.gru(x.transpose(1, 2).float())
        return self.fc(self.drop_rnn(torch.cat((hidden[-2], hidden[-1]), dim=1)))
