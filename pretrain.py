# train_classifier_on_shards.py
# Usage:
#   python train_classifier_on_shards.py --data_dir dataset_shards --epochs 15 --batch 1024
#
# Saves: switcher_classifier.pt

import os, glob, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt

# ------------------------- Sharded dataset (lazy) -------------------------
class ShardedDataset(Dataset):
    def __init__(self, shard_dir):
        self.paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        self.index = []  # (path_idx, local_idx)
        self.sizes = []
        for pi, p in enumerate(self.paths):
            with np.load(p) as z:
                n = z["X"].shape[0]
            self.sizes.append(n)
            self.index.extend([(pi, i) for i in range(n)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pi, li = self.index[idx]
        with np.load(self.paths[pi]) as z:
            X = z["X"][li].astype(np.float32)
            y = z["y"][li].astype(np.int64)
        return X, y

# ------------------------- Model -------------------------
class SwitcherMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # 0=BDF, 1=QSS
        )
    def forward(self, x): return self.net(x)

# ------------------------- Training -------------------------
def train(shard_dir, epochs=15, batch=1024, lr=3e-4, ckpt="switcher_classifier.pt", val_frac=0.15, patience=10, min_delta=0.0):
    ds = ShardedDataset(shard_dir)
    # Peek input dim
    x0, y0 = ds[0]
    in_dim = int(x0.shape[0])

    # Split indices (shuffle once)
    N = len(ds)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = int(val_frac * N)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    train_ds = Subset(ds, tr_idx.tolist())
    val_ds   = Subset(ds, val_idx.tolist())

    loader_tr = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)
    loader_va = DataLoader(val_ds,   batch_size=max(2048, batch), shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwitcherMLP(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    hist_tr_loss = []
    hist_val_acc = []
    hist_val_loss = []
    for ep in range(1, epochs+1):
        # train
        model.train()
        tot, seen = 0.0, 0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.size(0)
            seen += xb.size(0)
        tr_loss = tot / max(1, seen)

        # val
        model.eval()
        correct, tot_va = 0, 0
        va_tot_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                va_tot_loss += crit(logits, yb).item() * xb.size(0)
                correct += (pred == yb).sum().item()
                tot_va += xb.size(0)
        acc = correct / max(1, tot_va)
        va_loss = va_tot_loss / max(1, tot_va)
        print(f"[epoch {ep:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={acc:.4f}")
        hist_tr_loss.append(tr_loss)
        hist_val_acc.append(acc)
        hist_val_loss.append(va_loss)

        # track best by validation loss for early stopping and checkpointing
        if va_loss < (best_val_loss - min_delta):
            best_val_loss = va_loss
            best_acc = acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ↳ saved best to {ckpt} (val_loss={best_val_loss:.4f}, val_acc={best_acc:.4f})")
        else:
            epochs_without_improve += 1
            print(f"  ↳ no improvement for {epochs_without_improve}/{patience} epochs")
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"[done] best val_loss={best_val_loss:.4f}  best val_acc={best_acc:.4f}")

    # ----- Save training history and plots -----
    stem = os.path.splitext(ckpt)[0]
    hist_path = f"{stem}_history.npz"
    np.savez(hist_path, train_loss=np.array(hist_tr_loss, dtype=np.float32), val_loss=np.array(hist_val_loss, dtype=np.float32), val_acc=np.array(hist_val_acc, dtype=np.float32))
    print(f"Saved history to {hist_path}")

    epochs_axis = np.arange(1, len(hist_tr_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_axis, hist_tr_loss, label="Train Loss")
    if len(hist_val_loss) == len(hist_tr_loss):
        axes[0].plot(epochs_axis, hist_val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs_axis, hist_val_acc, label="Val Accuracy", color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Validation Accuracy (best={best_acc:.3f})")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{stem}_history.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {plot_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset_shards")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", type=str, default="switcher_classifier.pt")
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()
    train(args.data_dir, args.epochs, args.batch, args.lr, args.ckpt, args.val_frac)
