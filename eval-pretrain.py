# eval_policy_vs_fixed.py
# Usage:
#   python eval_policy_vs_fixed.py \
#     --ckpt switcher_classifier.pt \
#     --mechanism large_mechanism/n-dodecane.yaml \
#     --fuel nc12h26 --oxidizer "O2:0.21, N2:0.79" \
#     --T 650 --P_atm 3.0 --phi 1.0 --total_time 5e-2 --dt 1e-6 \
#     --out_dir eval_outputs
#
# Outputs (in out_dir):
#   actions.png
#   cpu_time.png
#   timestep_error.png
#   temperature_vs_ref.png
#   species_vs_ref_<SPEC>.png  (for each chosen species)
#   metrics.csv

import os
import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cantera as ct
from tqdm import tqdm
from environment import IntegratorSwitchingEnv
from reward_model import LagrangeReward1

# ---------- Model (same as trainer) ----------
class SwitcherMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # 0=BDF, 1=QSS
        )
    def forward(self, x): return self.net(x)

# ---------- Robust species indexing ----------
def safe_species_index(gas: ct.Solution, name: str):
    tries = [name, name.upper(), name.capitalize(), name.lower()]
    for t in tries:
        try:
            return gas.species_index(t)
        except Exception:
            pass
    return None

# ---------- Policies ----------
def make_learned_policy(ckpt_path, obs_dim, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = SwitcherMLP(in_dim=obs_dim)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    def _policy(obs_np: np.ndarray) -> int:
        x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        return int(logits.argmax(dim=1).item())
    return _policy

def always_bdf_policy(_): return 0
def always_qss_policy(_): return 1

# ---------- Rollout ----------
def rollout_once(env: IntegratorSwitchingEnv, policy_fn, max_steps=None):
    obs, info = env.reset(temperature=env.eval_T,
                          pressure=env.eval_P,
                          phi=env.eval_phi,
                          total_time=env.eval_total_time,
                          dt=env.eval_dt,
                          etol=env.etol)
    steps = min(env.n_episodes, max_steps) if max_steps else env.n_episodes
    actions, cpu, terr, rewards = [], [], [], []
    pbar = tqdm(total=steps, desc="Rollout Step")
    for _ in range(steps):
        a = policy_fn(obs)
        obs, r, terminated, truncated, inf = env.step(a)
        actions.append(a)
        cpu.append(inf.get("cpu_time", 0.0))
        terr.append(inf.get("timestep_error", 0.0))
        rewards.append(r)
        temp = env.current_state[0]
        ref_temp = env.ref_states[env.current_episode * env.super_steps, 0]
        cpu_time = inf.get("cpu_time", 0.0)
        if terminated or truncated:
            break
        pbar.update(1)
        pbar.set_postfix({
            'T': f'{temp:.1f}K | {ref_temp:.1f}K/{env.ref_states[0, 0]:.1f}K',
            'A': a,
            'C': f'{cpu_time:.3f}s',
            'R': f'{r:.1f}'
        })
    pbar.close()
    # Trajectory time/state (env tracked at micro-step resolution)
    times = np.array(env.times_trajectory) if env.times_trajectory is not None else None
    traj = np.array(env.states_trajectory) if env.states_trajectory is not None else None

    return {
        "actions": np.array(actions, dtype=np.int32),
        "cpu": np.array(cpu, dtype=np.float64),
        "terr": np.array(terr, dtype=np.float64),
        "rewards": np.array(rewards, dtype=np.float64),
        "times": times,
        "traj": traj,
        "episodes_used": len(actions),
    }


plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5
})

# ---------- Plot helpers ----------
def save_line(y_series, labels, ylabel, out_path, x_label="Step"):
    plt.figure(figsize=(10, 4))
    for y, lab in zip(y_series, labels):
        plt.plot(y, label=lab, linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_actions(actions_dict, out_path):
    plt.figure(figsize=(10, 3))
    for name, a in actions_dict.items():
        plt.step(np.arange(len(a)), a, where='post', label=name)
    plt.yticks([0,1], ["BDF(0)","QSS(1)"])
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_temp_vs_ref(times_ref, temp_ref, rollouts, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(times_ref, temp_ref, linestyle=':', linewidth=2, label="Reference")
    for name, ro in rollouts.items():
        if ro["times"] is not None and ro["traj"] is not None:
            T = ro["traj"][:, 0]
            plt.plot(ro["times"], T, linewidth=1.5, label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_species_vs_ref(spec_name, spec_idx, times_ref, Yref, rollouts, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(times_ref, Yref, linestyle=':', linewidth=2, label="Reference")
    for name, ro in rollouts.items():
        if ro["times"] is not None and ro["traj"] is not None and spec_idx is not None:
            Y = ro["traj"][:, 1+spec_idx]
            plt.plot(ro["times"], Y, linewidth=1.5, label=name)
    plt.xlabel("Time [s]")
    plt.ylabel(f"Y({spec_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- Combined 2x2 plot helpers ----------
def save_summary_grid(ref_times, ref_temp, rollouts, learned_run, out_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Actions (only learned)
    ax = axs[0, 0]
    a = learned_run["actions"]
    ax.step(np.arange(len(a)), a, where='post', label="Learned")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BDF(0)", "QSS(1)"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Action")
    ax.set_title("Action selection (learned)")
    ax.grid(True, alpha=0.3)

    # CPU time per step (all)
    ax = axs[0, 1]
    ax.plot(rollouts["Always-BDF"]["cpu"], label="Always-BDF")
    ax.plot(rollouts["Always-QSS"]["cpu"], label="Always-QSS")
    ax.plot(rollouts["Learned"]["cpu"], label="Learned")
    ax.set_xlabel("Step")
    ax.set_ylabel("CPU time [s]")
    ax.set_title("CPU time per step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Timestep error (all)
    ax = axs[1, 0]
    ax.plot(rollouts["Always-BDF"]["terr"], label="Always-BDF")
    ax.plot(rollouts["Always-QSS"]["terr"], label="Always-QSS")
    ax.plot(rollouts["Learned"]["terr"], label="Learned")
    ax.set_xlabel("Step")
    ax.set_ylabel("Timestep error")
    ax.set_title("Timestep error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature vs reference (all)
    ax = axs[1, 1]
    for name, ro in rollouts.items():
        if ro["times"] is not None and ro["traj"] is not None:
            T = ro["traj"][:, 0]
            ax.plot(ro["times"], T, linewidth=1.5, label=name)
    ax.plot(ref_times, ref_temp, linestyle=':', linewidth=2, label="Reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature vs reference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Evaluation Summary", y=1.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_species_grid(spec_entries, ref_times, ref_states, rollouts, out_path):
    # spec_entries: list of tuples (spec_name, spec_idx)
    n = min(4, len(spec_entries))
    if n == 0:
        return
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axs.flatten()
    for i in range(n):
        spec_name, spec_idx = spec_entries[i]
        ax = axes[i]
        Yref = ref_states[:, 1 + spec_idx]
        for name, ro in rollouts.items():
            if ro["times"] is not None and ro["traj"] is not None:
                Y = ro["traj"][:, 1 + spec_idx]
                ax.plot(ro["times"], Y, linewidth=1.5, label=name)
        ax.plot(ref_times, Yref, linestyle=':', linewidth=2, label="Reference")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Y({spec_name})")
        ax.set_title(f"{spec_name} vs reference")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Hide any unused subplots
    for j in range(n, 4):
        fig.delaxes(axes[j])

    fig.suptitle("Key Species vs Reference", y=1.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ---------- Build eval env with shared ICs and reference ----------
def make_eval_env(args):
    reward_cfg = dict(
        epsilon=args.epsilon,
        lambda_init=1.0,
        lambda_lr=0.05,
        target_violation=0.0,
        cpu_log_delta=1e-3,
        reward_clip=5.0,
    )

    env = IntegratorSwitchingEnv(
        mechanism_file=args.mechanism,
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        temp_range=(args.T, args.T),
        phi_range=(args.phi, args.phi),
        pressure_range=(int(args.P_atm), int(args.P_atm)),
        time_range=(args.total_time, args.total_time),
        dt_range=(args.dt, args.dt),
        etol=args.epsilon,
        verbose=False,
        termination_count_threshold=100,
        reward_function=__import__("reward_model").LagrangeReward1(**reward_cfg),
        precompute_reference=True,     # needed for plotting vs reference
        track_trajectory=True          # needed to plot trajectories
    )

    # Solver indices: 0=BDF, 1=QSS
    env.solver_configs = [
        {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},
        {'type': 'qss',   'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False,
         'itermax': 2, 'epsmin': 0.002, 'epsmax': 100.0, 'abstol': 1e-8, 'mxsteps': 1000, 'name': 'QSS'},
    ]

    # Fix evaluation ICs on the env so we can reset multiple times identically
    env.eval_T = args.T
    env.eval_P = args.P_atm * ct.one_atm
    env.eval_phi = args.phi
    env.eval_total_time = args.total_time
    env.eval_dt = args.dt

    # Do one reset to build the reference trajectory
    env.reset(temperature=env.eval_T,
              pressure=env.eval_P,
              phi=env.eval_phi,
              total_time=env.eval_total_time,
              dt=env.eval_dt,
              etol=env.etol)

    # Copy out the reference arrays once
    ref_states = np.array(env.ref_states)
    ref_times  = np.array(env.ref_times)

    return env, ref_states, ref_times

# ---------- Metric helpers ----------
def summarize(run, epsilon):
    terr = run["terr"]
    cpu  = run["cpu"]
    actions = run["actions"]
    return {
        "mean_cpu": float(cpu.mean()) if cpu.size else 0.0,
        "viol_rate": float((terr > epsilon).mean()) if terr.size else 0.0,
        "switches": int((actions[1:] != actions[:-1]).sum()) if actions.size > 1 else 0,
        "steps": int(actions.size)
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="switcher_classifierII.pt")
    ap.add_argument("--mechanism", type=str, default="large_mechanism/n-dodecane.yaml")
    ap.add_argument("--fuel", type=str, default="nc12h26")
    ap.add_argument("--oxidizer", type=str, default="O2:0.21, N2:0.79")
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--P_atm", type=float, default=1.0)
    ap.add_argument("--phi", type=float, default=1.0)
    ap.add_argument("--total_time", type=float, default=8e-2)
    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="eval_outputs")
    ap.add_argument("--max_steps", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build one env (we'll re-use same ICs for each policy) and copy reference
    env, ref_states, ref_times = make_eval_env(args)

    # Peek obs dim for classifier
    obs0, _ = env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=args.phi,
                        total_time=args.total_time, dt=args.dt, etol=args.epsilon)
    obs_dim = int(obs0.shape[0])

    learned = make_learned_policy(args.ckpt, obs_dim)

    # Run 3 rollouts on identical ICs
    # (we re-reset the env between runs to keep them identical)
    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=args.phi,
              total_time=args.total_time, dt=args.dt, etol=args.epsilon)
    run_learned = rollout_once(env, learned, max_steps=args.max_steps)

    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=args.phi,
              total_time=args.total_time, dt=args.dt, etol=args.epsilon)
    run_bdf = rollout_once(env, always_bdf_policy, max_steps=args.max_steps)

    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=args.phi,
              total_time=args.total_time, dt=args.dt, etol=args.epsilon)
    run_qss = rollout_once(env, always_qss_policy, max_steps=args.max_steps)

    # Build comparison dicts for plotting
    ref_temp = ref_states[:, 0]
    rollouts_for_temp = {"Always-BDF": run_bdf, "Always-QSS": run_qss, "Learned": run_learned}

    # Summary 2x2 grid
    save_summary_grid(ref_times, ref_temp, rollouts_for_temp, run_learned,
                      os.path.join(args.out_dir, f"summary_2x2_{args.T}K.png"))

    # Key species vs reference
    gas_tmp = ct.Solution(args.mechanism)
    key_specs = ["OH", "H2O2", "O2"]
    # add fuel name (attempt several)
    fuel_cands = [args.fuel, "CH4", "NC12H26"]
    for f in fuel_cands:
        if safe_species_index(gas_tmp, f) is not None:
            key_specs.append(f)
            break

    spec_entries = []
    for spec in key_specs:
        idx = safe_species_index(gas_tmp, spec)
        if idx is None:
            print(f"[warn] species {spec} not in mechanism, skipping plot.")
            continue
        spec_entries.append((spec, idx))

    # Species 2x2 grid (up to four species)
    save_species_grid(spec_entries, ref_times, ref_states, rollouts_for_temp,
                      os.path.join(args.out_dir, f"species_grid_2x2_{args.T}K.png"))

    # Metrics summary
    m_learn = summarize(run_learned, args.epsilon)
    m_bdf   = summarize(run_bdf, args.epsilon)
    m_qss   = summarize(run_qss, args.epsilon)

    print("\n=== EVALUATION METRICS ===")
    for name, m in [("Learned", m_learn), ("Always-BDF", m_bdf), ("Always-QSS", m_qss)]:
        print(f"{name:12s} | steps={m['steps']:4d}  mean_cpu={m['mean_cpu']:.4e}  "
              f"viol_rate={m['viol_rate']:.3f}  switches={m['switches']}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, f"metrics_{args.T}K.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "steps", "mean_cpu", "viol_rate", "switches"])
        for name, m in [("Learned", m_learn), ("Always-BDF", m_bdf), ("Always-QSS", m_qss)]:
            w.writerow([name, m["steps"], f"{m['mean_cpu']:.6e}", f"{m['viol_rate']:.6f}", m["switches"]])
    print(f"\nSaved figures and metrics to: {args.out_dir}")

if __name__ == "__main__":
    main()
