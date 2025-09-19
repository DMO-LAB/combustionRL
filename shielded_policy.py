# shielded_policy.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitcherMLP(nn.Module):
    """Same MLP head used for training."""
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)   # 0=BDF, 1=QSS
        )
    def forward(self, x): return self.net(x)

class ShieldedPolicy:
    """
    Runtime policy wrapper:
      - Uses softmax confidence to gate decisions (fallback to BDF when uncertain)
      - Adds short hysteresis: once in BDF, stay for K steps
    """
    def __init__(self, model: nn.Module, device, pmin: float = 0.70, hold_K: int = 3):
        self.model = model.to(device).eval()
        self.device = device
        self.pmin = float(pmin)
        self.hold_K = int(hold_K)
        self._hold = 0

    @torch.no_grad()
    def __call__(self, obs_np: np.ndarray) -> int:
        x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        a = int(np.argmax(probs))

        # Confidence gate
        if probs[a] < self.pmin:
            a = 0  # BDF

        # Hysteresis: when in BDF, hold for K steps
        if self._hold > 0:
            self._hold -= 1
            a = 0
        elif a == 0:
            self._hold = self.hold_K

        return a
