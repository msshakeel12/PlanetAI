"""2D folded CNN model and training helpers for transit classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DepthwiseSeparableBlock(nn.Module):
    """Xception-style depthwise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class Folded2DCNN(nn.Module):
    """CNN for 2D folded phase images with shape (1, 10, 1000)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 256, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class FoldedImageDataset(Dataset):
    """Dataset for 2D folded image tensors."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass
class CNNTrainConfig:
    """Training hyperparameters for folded 2D CNN."""

    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42


def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)
            ys.append(yb.numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(probs)


def train_folded_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: CNNTrainConfig,
) -> tuple[Folded2DCNN, dict[str, float]]:
    """Train folded 2D CNN and return test metrics.

    Args:
        x_train: Train folded images (N, 10, 1000).
        y_train: Train labels.
        x_val: Validation images.
        y_val: Validation labels.
        x_test: Test images.
        y_test: Test labels.
        cfg: Training configuration.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Folded2DCNN().to(device)

    train_loader = DataLoader(FoldedImageDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(FoldedImageDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(FoldedImageDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val = float("inf")

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / max(1, val_n)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_prob = _predict_probs(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true.astype(int), y_pred).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": cm,
    }
    return model, metrics
