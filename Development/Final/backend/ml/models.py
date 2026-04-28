"""
LeakSense Twin — Neural Network Models
Implements:
  1. LeakSenseNet: Binary leak detection (ReLU-Sigmoid-Cosine architecture)
  2. LeakLocalizationNet: Multi-class zone classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakSenseNet(nn.Module):
    """
    Binary leak detection network.
    Architecture: H = σ(ReLU(W_relu · cos(W_cos · X) + B_cos))

    Uses cosine similarity to learned healthy-state prototypes,
    ReLU to amplify anomalous deviations, and sigmoid for probability output.
    """

    def __init__(self, input_dim: int = 31, hidden_dim: int = 64):
        super().__init__()
        # Cosine similarity layer: maps input to learned healthy-state prototype space
        self.W_cos = nn.Linear(input_dim, hidden_dim, bias=True)
        # ReLU layer: suppresses normal deviations, amplifies anomalies
        self.W_relu = nn.Linear(hidden_dim, 32, bias=True)
        # Sigmoid output: probability of leak event
        self.sigmoid_out = nn.Linear(32, 1, bias=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cosine similarity to learned healthy prototypes
        cos_features = F.normalize(self.W_cos(x), dim=1)
        # ReLU to amplify deviations from healthy state
        relu_features = F.relu(self.W_relu(cos_features))
        relu_features = self.dropout(relu_features)
        # Sigmoid: outputs P(leak)
        logit = self.sigmoid_out(relu_features)
        return torch.sigmoid(logit).squeeze(-1)


class LeakLocalizationNet(nn.Module):
    """
    Multi-class zone localization network.
    Classes: 0=No leak, 1=Zone A, 2=Zone B, 3=Zone C, 4=Zone D, 5=Zone E
    """

    def __init__(self, input_dim: int = 31, n_classes: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=1)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (healthy >> fault samples).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss
        return loss.mean()
