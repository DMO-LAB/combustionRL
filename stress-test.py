# stress_test_eval.py
# Usage example:
#   python stress_test_eval.py \
#     --ckpt switcher_classifier.pt \
#     --mechanism large_mechanism/n-dodecane.yaml \
#     --fuel nc12h26 --oxidizer "O2:0.21, N2:0.79" \
#     --T_list 700,800,900,1000 --P_list 1.0,3.0,5.0 --phi_list 0.9,1.0,1.2 \
#     --total_time 5e-2 --dt 1e-6 --epsilon 1e-3 --out_dir stress_outputs
#
# It saves, for every (T,P,phi):
#   summary_2x2_T{T}_P{P}_phi{phi}.png      (actions + CPU + timestep error + temperature vs ref)
#   species_grid_2x2_T{T}_P{P}_phi{phi}.png (OH/H2O2/O2/+fuel vs ref)
# and writes a single CSV:
#   stress_metrics.csv  (mean CPU, violation rate, switches, per policy)

import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import cantera as ct
from tqdm import tqdm

from environment import IntegratorSwitchingEnv
from reward_model import LagrangeReward1
from shielded_policy import SwitcherMLP, ShieldedPolicy

# ---------- formatting ----------
plt.rcParams.update({
    'font.size': 12, 'font.weight': 'bold',
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.titlesize': 14, 'figure.titleweight': 'bold',
    'axes.linewidth': 1.2, 'grid.linewidth': 0.8, 'lines.linewidth': 2.0
})

# ---------- util ----------
def safe_species_index(gas: ct.Solution, name: str):
    for t in (name, name.upper(), name.capitalize(), name.lower()):
        try:
            return gas.species_index(t)
        except Exception:
            pass
    return None

def summarize(run, epsilon):
    terr = run["terr"]; cpu = run["cpu"]; actions = run["actions"]
    return {
        "mean_cpu": float(cpu.mean()) if cpu.size else 0.0,
        "viol_rate": float((terr > epsilon).mean()) if terr.size else 0.0,
        "switches": int((actions[1:] != actions[:-1]).sum()) if actions.size > 1 else 0,
        "steps": int(actions.size)
    }

# ---------- plotting ----------
def save_summary_grid(ref_times, ref_temp, rollouts, learned_run, title, out_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axs[0, 0]  # actions
    a = learned_run["actions"]
    ax.step(np.arange(len(a)), a, where='post', label="Learned")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["BDF(0)", "QSS(1)"])
    ax.set_xlabel("Step"); ax.set_ylabel("Action"); ax.set_title("Action selection (learned)")
    ax.grid(True, alpha=0.3)

    ax = axs[0, 1]  # CPU
    ax.plot(rollouts["Always-BDF"]["cpu"], label="Always-BDF")
    ax.plot(rollouts["Always-QSS"]["cpu"], label="Always-QSS")
    ax.plot(rollouts["Learned"]["cpu"], label="Learned", linestyle='--')
    ax.set_xlabel("Step"); ax.set_ylabel("CPU time [s]"); ax.set_title("CPU time per step")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axs[1, 0]  # timestep error
    ax.plot(rollouts["Always-BDF"]["terr"], label="Always-BDF")
    ax.plot(rollouts["Always-QSS"]["terr"], label="Always-QSS")
    ax.plot(rollouts["Learned"]["terr"], label="Learned", linestyle=':', linewidth=3)
    ax.set_xlabel("Step"); ax.set_ylabel("Timestep error"); ax.set_title("Timestep error")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axs[1, 1]  # temperature vs ref
    for name, ro in rollouts.items():
        if ro["times"] is not None and ro["traj"] is not None:
            T = ro["traj"][:, 0]
            if name == "Learned":
                ax.plot(ro["times"], T, label=name, linestyle=':', linewidth=3)
            else:
                ax.plot(ro["times"], T, label=name)
    ax.plot(ref_times, ref_temp, linestyle='--', linewidth=2, label="Reference")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]"); ax.set_title("Temperature vs reference")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=300); plt.close(fig)

def save_species_grid(spec_entries, ref_times, ref_states, rollouts, title, out_path):
    n = min(4, len(spec_entries))
    if n == 0: return
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True); axes = axs.flatten()
    for i in range(n):
        spec_name, spec_idx = spec_entries[i]; ax = axes[i]
        Yref = ref_states[:, 1 + spec_idx]
        for name, ro in rollouts.items():
            if ro["times"] is not None and ro["traj"] is not None:
                Y = ro["traj"][:, 1 + spec_idx]
                if name == "Learned":
                    ax.plot(ro["times"], Y, label=name, linestyle=':', linewidth=3)
                else:
                    ax.plot(ro["times"], Y, linewidth=1.5, label=name)
        ax.plot(ref_times, Yref, linestyle='--', linewidth=2, label="Reference")
        ax.set_xlabel("Time [s]"); ax.set_ylabel(f"Y({spec_name})"); ax.set_title(f"{spec_name} vs reference")
        ax.grid(True, alpha=0.3); 
        if i == 0: ax.legend()
    # hide unused
    for j in range(n, 4): fig.delaxes(axes[j])
    fig.suptitle(title)
    fig.savefig(out_path, dpi=300); plt.close(fig)

# ---------- rollout ----------
def rollout_once(env: IntegratorSwitchingEnv, policy_fn, max_steps=None):
    obs, _ = env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
                       total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    steps = min(env.n_episodes, max_steps) if max_steps else env.n_episodes
    actions, cpu, terr, rewards = [], [], [], []
    pbar = tqdm(range(steps), desc="Rollout")
    for _ in pbar:
        a = policy_fn(obs)
        obs, r, terminated, truncated, inf = env.step(a)
        actions.append(a)
        cpu.append(inf.get("cpu_time", 0.0))
        terr.append(inf.get("timestep_error", 0.0))
        rewards.append(r)
        if terminated or truncated: break
        pbar.set_postfix({
            'T': f'{env.current_state[0]:.1f}K | {env.ref_states[0, 0]:.1f}K',
            'A': a,
            'C': f'{inf.get("cpu_time", 0.0):.3f}s',
            'R': f'{r:.1f}'
        })
        pbar.update(1)
    pbar.close()
    times = np.array(env.times_trajectory) if env.times_trajectory is not None else None
    traj  = np.array(env.states_trajectory) if env.states_trajectory is not None else None
    return {"actions": np.array(actions, np.int32),
            "cpu": np.array(cpu, np.float64),
            "terr": np.array(terr, np.float64),
            "rewards": np.array(rewards, np.float64),
            "times": times, "traj": traj}

# ---------- env factory ----------
def make_env(args):
    reward_cfg = dict(epsilon=args.epsilon, lambda_init=1.0, lambda_lr=0.05,
                      target_violation=0.0, cpu_log_delta=1e-3, reward_clip=5.0)
    env = IntegratorSwitchingEnv(
        mechanism_file=args.mechanism, fuel=args.fuel, oxidizer=args.oxidizer,
        temp_range=(args.T, args.T), phi_range=(args.phi, args.phi),
        pressure_range=(int(args.P_atm), int(args.P_atm)),
        time_range=(args.total_time, args.total_time), dt_range=(args.dt, args.dt),
        etol=args.epsilon, verbose=False, termination_count_threshold=100,
        reward_function=LagrangeReward1(**reward_cfg),
        precompute_reference=True, track_trajectory=True
    )
    env.solver_configs = [
        {'type':'cvode','rtol':1e-6,'atol':1e-12,'mxsteps':100000,'name':'CVODE_BDF'},
        {'type':'qss', 'dtmin':1e-16,'dtmax':1e-6,'stabilityCheck':False,'itermax':2,
         'epsmin':0.002,'epsmax':100.0,'abstol':1e-8,'mxsteps':1000,'name':'QSS'}
    ]
    # Fix eval ICs
    env.eval_T = args.T; env.eval_P = args.P_atm * ct.one_atm; env.eval_phi = args.phi
    env.eval_total_time = args.total_time; env.eval_dt = args.dt
    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
              total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    return env

# ---------- one condition run ----------
def run_condition(args, device, model, T, P_atm, phi, out_dir):
    # prepare env with these ICs
    local_args = argparse.Namespace(**vars(args))
    local_args.T = float(T); local_args.P_atm = float(P_atm); local_args.phi = float(phi)
    env = make_env(local_args)
    ref_states = np.array(env.ref_states); ref_times = np.array(env.ref_times)
    ref_temp = ref_states[:, 0]

    # learned policy (shielded)
    obs0, _ = env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
                        total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    in_dim = int(obs0.shape[0])
    model = model.to(device).eval()
    shield = ShieldedPolicy(model, device, pmin=args.pmin, hold_K=args.hold_K)

    learned = lambda obs: shield(obs)
    always_bdf = lambda obs: 0
    always_qss = lambda obs: 1

    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
              total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    rL = rollout_once(env, learned, max_steps=args.max_steps)

    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
              total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    rB = rollout_once(env, always_bdf, max_steps=args.max_steps)

    env.reset(temperature=env.eval_T, pressure=env.eval_P, phi=env.eval_phi,
              total_time=env.eval_total_time, dt=env.eval_dt, etol=env.etol)
    rQ = rollout_once(env, always_qss, max_steps=args.max_steps)

    rollouts = {"Learned": rL, "Always-BDF": rB, "Always-QSS": rQ}
    title = f"T={T:.0f} K, P={P_atm:.2f} atm, φ={phi:.2f}"

    # save plots
    os.makedirs(out_dir, exist_ok=True)
    save_summary_grid(ref_times, ref_temp, rollouts, rL,
                      title, os.path.join(out_dir, f"summary_2x2_T{T:.0f}_P{P_atm:.2f}_phi{phi:.2f}.png"))

    # choose species (OH, H2O2, O2 + fuel if present)
    gas_tmp = ct.Solution(args.mechanism)
    spec_names = ["OH", "H2O2", "O2"]
    for cand in [args.fuel, "CH4", "NC12H26"]:
        if safe_species_index(gas_tmp, cand) is not None:
            spec_names.append(cand); break
    spec_entries = []
    for s in spec_names:
        idx = safe_species_index(gas_tmp, s)
        if idx is not None: spec_entries.append((s, idx))

    save_species_grid(spec_entries, ref_times, ref_states, rollouts,
                      title, os.path.join(out_dir, f"species_grid_2x2_T{T:.0f}_P{P_atm:.2f}_phi{phi:.2f}.png"))

    # metrics
    mL, mB, mQ = summarize(rL, args.epsilon), summarize(rB, args.epsilon), summarize(rQ, args.epsilon)
    return dict(T=T, P_atm=P_atm, phi=phi,
                mean_cpu_L=mL["mean_cpu"], viol_L=mL["viol_rate"], switches_L=mL["switches"],
                mean_cpu_B=mB["mean_cpu"], viol_B=mB["viol_rate"], switches_B=mB["switches"],
                mean_cpu_Q=mQ["mean_cpu"], viol_Q=mQ["viol_rate"], switches_Q=mQ["switches"])

# ---------- args ----------
def parse_list(s, cast=float):
    return [cast(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="switcher_classifierII.pt")
    ap.add_argument("--mechanism", type=str, default="large_mechanism/n-dodecane.yaml")
    ap.add_argument("--fuel", type=str, default="nc12h26")
    ap.add_argument("--oxidizer", type=str, default="O2:0.21, N2:0.79")
    ap.add_argument("--T_list", type=str, default="700,800,900,1000,1200,1400")
    ap.add_argument("--P_list", type=str, default="1.0,3.0,5.0,10.0")
    ap.add_argument("--phi_list", type=str, default="0.9,1.0,1.2")
    ap.add_argument("--total_time", type=float, default=5e-2)
    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="stress_outputs")
    ap.add_argument("--max_steps", type=int, default=1000)
    # Shield params
    ap.add_argument("--pmin", type=float, default=0.70, help="min softmax confidence to accept action")
    ap.add_argument("--hold_K", type=int, default=3, help="BDF hysteresis steps")
    args = ap.parse_args()

    Ts   = parse_list(args.T_list, float)
    Ps   = parse_list(args.P_list, float)
    Phis = parse_list(args.phi_list, float)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Peek obs dim from a temp env (to instantiate MLP)
    tmp_args = argparse.Namespace(**vars(args)); tmp_args.T=float(Ts[0]); tmp_args.P_atm=float(Ps[0]); tmp_args.phi=float(Phis[0])
    tmp_env = make_env(tmp_args); obs0, _ = tmp_env.reset(temperature=tmp_env.eval_T, pressure=tmp_env.eval_P, phi=tmp_env.eval_phi,
                                                          total_time=tmp_env.eval_total_time, dt=tmp_env.eval_dt, etol=tmp_env.etol)
    in_dim = int(obs0.shape[0]); del tmp_env
    model = SwitcherMLP(in_dim=in_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # CSV
    csv_path = os.path.join(args.out_dir, "stress_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T","P_atm","phi",
                    "cpu_learn","viol_learn","switch_learn",
                    "cpu_bdf","viol_bdf","switch_bdf",
                    "cpu_qss","viol_qss","switch_qss",
                    "speedup_vs_bdf"])
        for T in Ts:
            for P in Ps:
                for phi in Phis:
                    cond_dir = os.path.join(args.out_dir, f"T{T:.0f}_P{P:.2f}_phi{phi:.2f}")
                    os.makedirs(cond_dir, exist_ok=True)
                    res = run_condition(args, device, model, T, P, phi, cond_dir)
                    speedup = (res["mean_cpu_B"]/max(1e-12, res["mean_cpu_L"]))
                    w.writerow([T, P, phi,
                                res["mean_cpu_L"], res["viol_L"], res["switches_L"],
                                res["mean_cpu_B"], res["viol_B"], res["switches_B"],
                                res["mean_cpu_Q"], res["viol_Q"], res["switches_Q"],
                                speedup])
                    print(f"[DONE] T={T:.0f} K, P={P:.2f} atm, φ={phi:.2f} | "
                          f"CPU(L)={res['mean_cpu_L']:.3e}, viol(L)={res['viol_L']:.3f}, "
                          f"speedup_vs_BDF={speedup:.2f}x")
    print(f"\nSaved stress metrics to {csv_path}")

if __name__ == "__main__":
    main()
