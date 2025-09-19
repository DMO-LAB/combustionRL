# train_ppo_lstm.py
# From-scratch PPO + LSTM training on IntegratorSwitchingEnv
# - Optional warm start from switcher_classifier.pt
# - Uses confidence gate + hysteresis during rollouts
# - Logs CPU time & timestep error; saves publication-quality plots
# - Saves checkpoints and a final TorchScript export

import os, csv, math, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cantera as ct
from tqdm import tqdm
from environment import IntegratorSwitchingEnv
from reward_model import LagrangeReward1
from ppo_lstm_core import RunningMeanStd, BufferConfig, RolloutBufferRecurrent, PolicyLSTM, PPO, PPOConfig
from shielded_policy import SwitcherMLP, ShieldedPolicy

# ----------------------------- Matplotlib aesthetics -----------------------------
plt.rcParams.update({
    "font.size": 12, "font.weight": "bold",
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 14, "figure.titleweight": "bold",
    "axes.linewidth": 1.2, "grid.linewidth": 0.8, "lines.linewidth": 2.0
})

# ----------------------------- Utilities -----------------------------
def make_env(args):
    reward_cfg = dict(epsilon=args.epsilon, lambda_init=1.0, lambda_lr=0.05,
                      target_violation=0.0, cpu_log_delta=1e-3, reward_clip=5.0)
    env = IntegratorSwitchingEnv(
        mechanism_file=args.mechanism, fuel=args.fuel, oxidizer=args.oxidizer,
        temp_range=(args.T_low, args.T_high), phi_range=(args.phi_low, args.phi_high),
        pressure_range=(int(args.P_low), int(args.P_high)),
        time_range=(args.time_low, args.time_high),
        dt_range=(args.dt, args.dt), etol=args.epsilon, verbose=False,
        termination_count_threshold=100,
        reward_function=LagrangeReward1(**reward_cfg),
        precompute_reference=True, track_trajectory=True
    )
    env.solver_configs = [
        {'type':'cvode','rtol':1e-6,'atol':1e-12,'mxsteps':100000,'name':'CVODE_BDF'},
        {'type':'qss', 'dtmin':1e-16,'dtmax':1e-6,'stabilityCheck':False,'itermax':2,
         'epsmin':0.002,'epsmax':100.0,'abstol':1e-8,'mxsteps':1000,'name':'QSS'}
    ]
    return env

def save_training_plots(log, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    steps = np.array(log["steps"])
    def p(x, y, yl, name):
        fig = plt.figure(figsize=(7,4))
        plt.plot(x, y); plt.xlabel("Environment steps"); plt.ylabel(yl)
        plt.grid(alpha=0.3); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=300); plt.close(fig)
    p(steps, np.array(log["ep_reward"]), "Episode reward (avg)", "curve_reward.png")
    p(steps, np.array(log["mean_cpu"]), "CPU time per step (mean)", "curve_cpu.png")
    p(steps, np.array(log["viol_rate"]), "Violation rate", "curve_viol.png")
    p(steps, np.array(log["loss_pi"]), "Policy loss", "curve_loss_pi.png")
    p(steps, np.array(log["loss_v"]), "Value loss", "curve_loss_v.png")
    p(steps, np.array(log["entropy"]), "Entropy", "curve_entropy.png")
    p(steps, np.array(log["approx_kl"]), "Approx KL", "curve_kl.png")
    p(steps, np.array(log["clip_frac"]), "Clip fraction", "curve_clipfrac.png")

def maybe_load_classifier_init(policy: PolicyLSTM, classifier_path: str, obs_dim: int):
    if not classifier_path or not os.path.exists(classifier_path):
        return
    try:
        clf = SwitcherMLP(in_dim=obs_dim)
        clf.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        # copy first two Linear layers into feature
        with torch.no_grad():
            # feature[0] and feature[2] are Linear
            policy.feature[0].weight.copy_(clf.net[0].weight)
            policy.feature[0].bias.copy_(clf.net[0].bias)
            policy.feature[2].weight.copy_(clf.net[3].weight)
            policy.feature[2].bias.copy_(clf.net[3].bias)
            # initialize actor to classifier final
            policy.actor.weight.copy_(clf.net[-1].weight)
            policy.actor.bias.copy_(clf.net[-1].bias)
        print(f"[init] Warm-started policy from classifier: {classifier_path}")
    except Exception as e:
        print(f"[init] Could not warm-start from classifier ({e}). Continuing with random init.")

# ----------------------------- Training loop -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = make_env(args)

    # peek obs dim
    obs0, _ = env.reset(temperature=np.random.uniform(*env.temp_range),
                        pressure=np.random.choice(np.arange(1,6))*ct.one_atm,
                        phi=np.random.uniform(*env.phi_range),
                        total_time=np.random.uniform(*env.time_range),
                        dt=args.dt, etol=args.epsilon)
    obs_dim = int(obs0.shape[0])

    # policy, PPO
    policy = PolicyLSTM(obs_dim=obs_dim, hidden_size=args.hidden, action_dim=2)
    maybe_load_classifier_init(policy, args.init_classifier, obs_dim)
    ppo = PPO(policy, PPOConfig(
        clip_coef=args.clip_coef, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, lr=args.lr, epochs=args.epochs,
        seq_len=args.seq_len, batch_seqs=args.batch_seqs, target_kl=args.target_kl,
        device=str(device)
    ))

    # shield wrapper for rollouts (kept outside optimizer)
    shield = ShieldedPolicy(model=policy, device=device, pmin=args.pmin, hold_K=args.hold_K)

    # obs normalization
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # rollout buffer
    buf_cfg = BufferConfig(obs_dim=obs_dim, size=args.rollout_steps, gamma=args.gamma, lam=args.lam, device=str(device))
    buf_cfg.hidden_size = args.hidden  # attach hidden size
    buffer = RolloutBufferRecurrent(buf_cfg)

    # logging
    log = {"steps": [], "ep_reward": [], "mean_cpu": [], "viol_rate": [],
           "loss_pi": [], "loss_v": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["steps","ep_reward","mean_cpu","viol_rate","loss_pi","loss_v","entropy","approx_kl","clip_frac"])

    global_step = 0
    for update in range(1, args.total_updates + 1):
        # ----------------- collect rollout -----------------
        buffer.reset()
        ep_rewards = []
        cpu_times = []
        violations = []
        errors = []

        # reset env with random ICs (curriculum-friendly ranges already set)
        obs, _ = env.reset(
            temperature=np.random.uniform(*env.temp_range),
            pressure=np.random.choice(np.arange(1,6))*ct.one_atm,
            phi=np.random.uniform(*env.phi_range),
            total_time=np.random.uniform(*env.time_range),
            dt=args.dt, etol=args.epsilon
        )
        # normalize obs online (update rms with a small window to avoid bias)
        # (we’ll update at the end of rollout for stability)
        # LSTM hidden at step 0
        hx = torch.zeros(args.hidden, device=device)
        cx = torch.zeros(args.hidden, device=device)

        pbar = tqdm(total=args.rollout_steps, desc="Rollout")
        for t in range(args.rollout_steps):
            # normalize on the fly with current stats
            obs_n = obs_rms.normalize(obs)
            # model step (sampled action from policy)
            with torch.no_grad():
                # policy.step expects torch tensors; convert inside
                act_sample, logprob, value, probs, (hx_new, cx_new) = policy.step(
                    torch.from_numpy(obs_n).to(device), hx, cx
                )
                # shielded decision (confidence gate + hysteresis)
                # recompute probs on CPU np:
                # probs_np = probs.cpu().numpy()
                # a_exec = int(np.argmax(probs_np)) if probs_np.max() >= args.pmin else 0
                # # respect hysteresis: if shield changed action to 0, shield will latch internally
                # shield._hold = getattr(shield, "_hold", 0)  # ensure field exists
                # if shield._hold > 0:
                #     shield._hold -= 1
                #     a_exec = 0
                # elif a_exec == 0:
                #     shield._hold = args.hold_K
                a_exec = act_sample
                # compute logprob of the executed action from current dist
                dist = torch.distributions.Categorical(probs=probs)
                logprob_exec = dist.log_prob(torch.tensor(a_exec, device=device)).item()
                value = float(value)

            # env step
            obs_next, reward, terminated, truncated, info = env.step(a_exec)
            #print(f"INFO: Action {a_exec} took {info.get('cpu_time', 0.0):.3f}s with error {info.get('timestep_error', 0.0):.3f} and reward {reward:.3f}")
            # collect metrics
            ep_rewards.append(reward)
            cpu_times.append(info.get("cpu_time", 0.0))
            violations.append(1.0 if info.get("timestep_error", 0.0) > args.epsilon else 0.0)
            errors.append(info.get("timestep_error", 0.0))
            # store in buffer: store HIDDEN BEFORE this step (hx,cx)
            buffer.add(
                obs=obs_n.astype(np.float32),
                action=a_exec,
                reward=float(reward),
                done=bool(terminated or truncated),
                value=float(value),
                logprob=float(logprob_exec),
                hx=hx.detach().cpu().numpy(),
                cx=cx.detach().cpu().numpy()
            )

            global_step += 1
            obs = obs_next
            hx, cx = hx_new.detach(), cx_new.detach()
            if terminated or truncated:
                # reset episode within rollout if it ends early
                obs, _ = env.reset(
                    temperature=np.random.uniform(*env.temp_range),
                    pressure=np.random.choice(np.arange(1,6))*ct.one_atm,
                    phi=np.random.uniform(*env.phi_range),
                    total_time=np.random.uniform(*env.time_range),
                    dt=args.dt, etol=args.epsilon
                )
                hx.zero_(); cx.zero_()
            pbar.update(1)
            temp = env.current_state[0]
            ref_temp = env.ref_states[env.current_episode * env.super_steps, 0]
            pbar.set_postfix({
                'T': f'{temp:.1f}K | {ref_temp:.1f}K/{env.ref_states[0, 0]:.1f}K',
                'A': a_exec,
                'P': f'{env.current_pressure/ct.one_atm:.1f}bar',
                'C': f'{cpu_times[-1]:.3f}s',
                'R': f'{reward:.1f}',
                'E': f'{errors[-1]:.3f}'
            })
        pbar.close()
        # update obs rms using rollout observations
        obs_rms.update(buffer.obs[:buffer.ptr])

        # bootstrap last value for GAE
        with torch.no_grad():
            obs_last_n = obs_rms.normalize(obs).astype(np.float32)
            obs_last_t = torch.from_numpy(obs_last_n).to(device).unsqueeze(0).unsqueeze(0)
            logits_last, values_last, _ = policy(obs_last_t, hx.unsqueeze(0), cx.unsqueeze(0))
            last_value = float(values_last.squeeze().cpu().item())

        adv, ret = buffer.compute_gae(last_value)

        # ----------------- PPO update -----------------
        train_info = ppo.update(buffer, adv, ret)

        # ----------------- logging & plots -----------------
        mean_ep_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_cpu  = float(np.mean(cpu_times)) if cpu_times else 0.0
        viol_rate = float(np.mean(violations)) if violations else 0.0

        log["steps"].append(global_step)
        log["ep_reward"].append(mean_ep_r)
        log["mean_cpu"].append(mean_cpu)
        log["viol_rate"].append(viol_rate)
        for k in ["loss_pi","loss_v","entropy","approx_kl","clip_frac"]:
            log[k].append(train_info[k])

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([global_step, mean_ep_r, mean_cpu, viol_rate,
                                    train_info["loss_pi"], train_info["loss_v"],
                                    train_info["entropy"], train_info["approx_kl"],
                                    train_info["clip_frac"]])

        if update % args.plot_every == 0:
            save_training_plots(log, args.out_dir)

        if update % args.ckpt_every == 0:
            path = os.path.join(args.out_dir, f"ppo_lstm_update{update}.pt")
            torch.save(policy.state_dict(), path)
            print(f"[ckpt] saved {path}")

    # final artifacts
    save_training_plots(log, args.out_dir)
    torch.save(policy.state_dict(), os.path.join(args.out_dir, "ppo_lstm_final.pt"))
    # TorchScript for easier deployment
    example_obs = torch.from_numpy(obs0.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,D]
    h0 = torch.zeros(1, 1, args.hidden); c0 = torch.zeros(1, 1, args.hidden)
    scripted = torch.jit.trace(lambda o,h,c: policy(o,h.squeeze(0),c.squeeze(0)), (example_obs, h0, c0))
    scripted.save(os.path.join(args.out_dir, "ppo_lstm_final_scripted.pt"))
    print(f"[done] training complete. Artifacts in: {args.out_dir}")


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # env & error
    ap.add_argument("--mechanism", type=str, default="large_mechanism/n-dodecane.yaml")
    ap.add_argument("--fuel", type=str, default="nc12h26")
    ap.add_argument("--oxidizer", type=str, default="O2:0.21, N2:0.79")
    ap.add_argument("--epsilon", type=float, default=1e-3)

    # IC sampling ranges for training (curriculum-like)
    ap.add_argument("--T_low", type=float, default=600.0)
    ap.add_argument("--T_high", type=float, default=1200.0)
    ap.add_argument("--phi_low", type=float, default=0.7)
    ap.add_argument("--phi_high", type=float, default=1.6)
    ap.add_argument("--P_low", type=float, default=1.0)
    ap.add_argument("--P_high", type=float, default=10.0)
    ap.add_argument("--time_low", type=float, default=1e-3)
    ap.add_argument("--time_high", type=float, default=5e-2)
    ap.add_argument("--dt", type=float, default=1e-6)

    # PPO/LSTM
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_seqs", type=int, default=16)
    ap.add_argument("--target_kl", type=float, default=0.03)

    # rollouts & training schedule
    ap.add_argument("--rollout_steps", type=int, default=4096)
    ap.add_argument("--total_updates", type=int, default=300)
    ap.add_argument("--plot_every", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=25)

    # shield (confidence gate + hysteresis)
    ap.add_argument("--pmin", type=float, default=0.70)
    ap.add_argument("--hold_K", type=int, default=3)

    # init from classifier
    ap.add_argument("--init_classifier", type=str, default="switcher_classifierII.pt")
    ap.add_argument("--from_scratch", action="store_true")

    # misc
    ap.add_argument("--out_dir", type=str, default="ppo_runs/run1")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.from_scratch:
        args.init_classifier = None
    train(args)
