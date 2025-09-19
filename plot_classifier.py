# plot_action_scatter.py
# Examples:
#   # from a single .npz
#   python plot_action_scatter.py --data oracle_dataset_ref.npz \
#       --feat_x T_norm --feat_y "log10(Y_OH)" --limit 50000 --out scatter_T_OH.png
#
#   # from shards
#   python plot_action_scatter.py --data dataset_shards \
#       --feat_x "log10(Y_OH)" --feat_y "log10(Y_H2O2)" --limit 80000 --balance \
#       --out scatter_OH_H2O2.png \
#       --mechanism large_mechanism/n-dodecane.yaml --fuel nc12h26 --oxidizer "O2:0.21, N2:0.79"
#
#   # list feature names and indices (no plot)
#   python plot_action_scatter.py --data dataset_shards --list_features \
#       --mechanism large_mechanism/n-dodecane.yaml --fuel nc12h26 --oxidizer "O2:0.21, N2:0.79"

import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: build feature names from your env
def build_feature_names(mechanism=None, fuel=None, oxidizer=None):
    """
    Returns names in the same order your dataset builder used:
      [T_norm] + [log10(Y_<key species>)] + [log10(P/atm)]
    Falls back to generic indices if Cantera/env isn't available.
    """
    try:
        import cantera as ct
        from environment import IntegratorSwitchingEnv
        from reward_model import LagrangeReward1
        reward_cfg = dict(epsilon=1e-3, lambda_init=1.0, lambda_lr=0.05,
                          target_violation=0.0, cpu_log_delta=1e-3, reward_clip=5.0)
        env = IntegratorSwitchingEnv(
            mechanism_file=mechanism or "gri30.yaml",
            fuel=fuel or "CH4:1.0",
            oxidizer=oxidizer or "N2:3.76, O2:1.0",
            precompute_reference=False, track_trajectory=False,
            reward_function=__import__("reward_model").LagrangeReward1(**reward_cfg)
        )
        names = ["T_norm"] + [f"log10(Y_{s})" for s in env.key_species] + ["log10(P/atm)"]
        return names
    except Exception:
        # fallback generic (user can still index by number)
        return None

def load_dataset_any(data_path, limit=None, balance=False, rng=None):
    """
    Load from:
      - a single .npz with arrays X [N,D], y [N]
      - a directory with shard_*.npz files (concatenate with random sampling)
    Returns (X, y)
    """
    rng = rng or np.random.default_rng(42)
    if os.path.isdir(data_path):
        shard_paths = sorted(glob.glob(os.path.join(data_path, "shard_*.npz")))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.npz in {data_path}")
        X_list, y_list = [], []
        remaining = int(1e18) if limit is None else int(limit)
        # reservoir-like uniform sampling per shard
        for p in shard_paths:
            with np.load(p) as z:
                Xs, ys = z["X"], z["y"]
            if remaining <= 0: break
            take = min(remaining, Xs.shape[0])
            idx = rng.choice(Xs.shape[0], size=take, replace=False)
            X_list.append(Xs[idx]); y_list.append(ys[idx])
            remaining -= take
        X = np.concatenate(X_list, axis=0) if len(X_list) > 1 else X_list[0]
        y = np.concatenate(y_list, axis=0) if len(y_list) > 1 else y_list[0]
    else:
        with np.load(data_path) as z:
            X, y = z["X"], z["y"]
        if limit is not None and limit < X.shape[0]:
            idx = rng.choice(X.shape[0], size=int(limit), replace=False)
            X, y = X[idx], y[idx]

    # optional class balance
    if balance:
        classes = np.unique(y)
        counts = [np.sum(y == c) for c in classes]
        m = int(np.min(counts))
        sel_idx = []
        for c in classes:
            idx_c = np.where(y == c)[0]
            sel_idx.append(rng.choice(idx_c, size=m, replace=False))
        sel_idx = np.concatenate(sel_idx, axis=0)
        X, y = X[sel_idx], y[sel_idx]

    return X, y

def parse_feature_selector(sel, feature_names):
    """
    sel can be an integer index or a name present in feature_names
    """
    # try integer
    try:
        idx = int(sel)
        return idx, f"feat[{idx}]"
    except ValueError:
        pass
    # try lookup by name
    if feature_names is None:
        raise ValueError(f"'{sel}' is not an index and feature names are unknown; "
                         f"pass an integer or provide --mechanism/--fuel/--oxidizer.")
    # accept flexible spacing/case
    canon = [s.lower().replace(" ", "") for s in feature_names]
    key = sel.lower().replace(" ", "")
    if key in canon:
        idx = canon.index(key)
        return idx, feature_names[idx]
    # also allow aliases like Y_OH, log10(Y_OH), etc.
    for i, nm in enumerate(feature_names):
        nml = nm.lower().replace(" ", "")
        if key in nml or nml in key:
            return i, feature_names[i]
    raise ValueError(f"Feature '{sel}' not found. Available: {feature_names}")

def make_pub_plot(X, y, i, j, label_i, label_j, out_path, alpha=0.35, size=8, s=10):
    # Styles
    plt.rcParams.update({
        "font.size": 12, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.labelweight": "bold",
        "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 1.2
    })
    fig = plt.figure(figsize=(size, size*0.85))
    ax = plt.gca()
    # Colors: 0=BDF (red), 1=QSS (green)
    mask_bdf = (y == 0)
    mask_qss = (y == 1)
    # unnormaliz temperature
    if label_i == "T_norm":
        X[:, i] = X[:, i] * 2000.0 + 300.0
        label_i = "Temperature [K]"
    if label_j == "T_norm":
        X[:, j] = X[:, j] * 2000.0 + 300.0
        label_j = "Temperature [K]"
  
    ax.scatter(X[mask_bdf, i], X[mask_bdf, j], s=s, c="#d62728", alpha=alpha, edgecolors="none", label="BDF (0)")
    ax.scatter(X[mask_qss, i], X[mask_qss, j], s=s, c="#2ca02c", alpha=alpha, edgecolors="none", label="QSS (1)")
    ax.set_xlabel(label_i); ax.set_ylabel(label_j)
    ax.grid(True, alpha=0.25)
    # simple legend with counts
    ax.legend(title=f"Classes (N={len(y)}):  BDF={mask_bdf.sum()} | QSS={mask_qss.sum()}", frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=400)  # high-res
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help=".npz file or directory of shards")
    ap.add_argument("--feat_x", type=str, default="T_norm", help="index or name (e.g., T_norm or log10(Y_OH))")
    ap.add_argument("--feat_y", type=str, default="log10(Y_OH)", help="index or name")
    ap.add_argument("--limit", type=int, default=60000, help="max points to load/sample")
    ap.add_argument("--balance", action="store_true", help="downsample to balance classes")
    ap.add_argument("--out", type=str, default="scatter.png")

    # only needed to auto-generate feature names
    ap.add_argument("--mechanism", type=str, default=None)
    ap.add_argument("--fuel", type=str, default=None)
    ap.add_argument("--oxidizer", type=str, default=None)
    ap.add_argument("--list_features", action="store_true", help="print feature names & exit")
    args = ap.parse_args()

    # try to build feature names from env; may be None if env/Cantera unavailable
    names = build_feature_names(args.mechanism, args.fuel, args.oxidizer)

    if args.list_features:
        if names is None:
            print("Feature names unavailable (env/Cantera not imported). You can still use numeric indices.")
        else:
            for i, n in enumerate(names):
                print(f"[{i:02d}] {n}")
        return

    # Load data
    X, y = load_dataset_any(args.data, limit=args.limit, balance=args.balance)
    if names is None:
        names = [f"feat[{i}]" for i in range(X.shape[1])]

    # Parse feature selectors
    i, label_i = parse_feature_selector(args.feat_x, names)
    j, label_j = parse_feature_selector(args.feat_y, names)

    # Make plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_pub_plot(X, y, i, j, label_i, label_j, args.out)
    print(f"[saved] {args.out}  ({label_i} vs {label_j})  N={len(y)}")

if __name__ == "__main__":
    main()
