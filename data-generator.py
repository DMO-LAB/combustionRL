# import numpy as np
# import time
# import copy
# from environment import IntegratorSwitchingEnv
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from reward_model import LagrangeReward1
# import cantera as ct
# from tqdm import tqdm


# def simulate_one_superstep_from_state(env, action, start_state, dt, super_steps, next_ref_state):
#     """
#     Build a fresh solver and integrate EXACTLY one super-step horizon starting from 'start_state'.
#     Return (cpu_time, step_error) comparing the end state to 'next_ref_state'.
#     This does NOT touch the live env state.
#     """
#     cfg = env.solver_configs[action]
#     # fresh solver & a temporary gas bound to it
#     solver, gas_tmp = env._build_fresh_solver(action, start_state.copy())

#     # integrate 'super_steps' micro-steps of size dt
#     t0 = time.time()
#     try:
#         current_state = start_state.copy()
#         for _ in range(super_steps):
#             if cfg['type'] == 'cvode':
#                 solver.set_state(current_state, 0.0)
#                 current_state = solver.solve_to(dt)
#             elif cfg['type'] == 'qss':
#                 solver.setState(current_state.tolist(), 0.0)
#                 rc = solver.integrateToTime(dt)
#                 if rc != 0:
#                     raise RuntimeError(f"QSS integrateToTime failed rc={rc}")
#                 current_state = np.array(solver.y)
#             else:
#                 raise ValueError(f"Unknown solver type {cfg['type']}")
#             # keep gas_tmp coherent (not strictly required for error)
#             gas_tmp.TPY = current_state[0], env.current_pressure, current_state[1:]
#         cpu_time = time.time() - t0

#         # compute step error vs next reference state
#         step_err = env._calculate_error(current_state, next_ref_state)
#         return cpu_time, step_err

#     except Exception:
#         return np.inf, np.inf


# def reference_anchored_oracle(env, k_step):
#     """
#     k_step indexes the decision point in the episode (0-based).
#     We:
#       - take start_ref = ref_states[k_index]
#       - take next_ref  = ref_states[k_index + 1]
#       where k_index = k_step * env.super_steps
#     Evaluate both actions from start_ref and pick the fastest that satisfies epsilon.
#     Fallback to BDF (assume action 0) otherwise.

#     Returns: (best_action, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_from_start_ref)
#     """
#     # map indices in the reference arrays
#     start_idx = k_step * env.super_steps
#     next_idx  = (k_step + 1) * env.super_steps
#     if next_idx >= len(env.ref_states):
#         # out of reference horizon, default to BDF
#         start_idx = max(0, len(env.ref_states) - env.super_steps - 1)
#         next_idx  = min(len(env.ref_states) - 1, start_idx + env.super_steps)

#     start_ref = env.ref_states[start_idx]
#     next_ref  = env.ref_states[next_idx]

#     # Evaluate both actions from the same (reference) start state
#     # Ensure action indices match your solver_configs: assume 0=BDF, 1=QSS
#     cpu_bdf, err_bdf = simulate_one_superstep_from_state(env, 0, start_ref, env.dt, env.super_steps, next_ref)
#     cpu_qss, err_qss = simulate_one_superstep_from_state(env, 1, start_ref, env.dt, env.super_steps, next_ref)

#     eps = env.reward_function.epsilon
#     # choose fastest feasible
#     feasible = []
#     if err_bdf <= eps: feasible.append((0, cpu_bdf))
#     if err_qss <= eps: feasible.append((1, cpu_qss))

#     if feasible:
#         feasible.sort(key=lambda x: x[1])
#         best_action = feasible[0][0]
#     else:
#         best_action = 0  # fallback to BDF when neither meets epsilon

#     # Build features from the SAME reference state (prevents drift leakage)
#     obs_ref = env._obs_from_arbitrary_state(start_ref, set_last_obs=False)
#     return best_action, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_ref


# def build_ref_anchored_dataset(env_maker,
#                                n_episodes=150,
#                                max_steps_per_ep=200,
#                                save_path="oracle_dataset_ref.npz",
#                                verbose_every=10):
#     """
#     For each episode:
#       - env.reset() to generate a fresh reference trajectory
#       - at decision k, compute label by comparing solvers FROM ref[k] TO ref[k+1]
#       - store obs computed from ref[k]
#       - Step the LIVE env using the oracle action (so we also log realistic rollouts if desired),
#         but labels/inputs are always ref-anchored and thus drift-free.
#     """
#     X, y = [], []

#     for ep in range(n_episodes):
#         env = env_maker()
#         # pick conditions that likely include ignition (curriculum helps)
#         obs, info = env.reset(
#             temperature=np.random.uniform(*env.temp_range),
#             pressure=np.random.choice(np.arange(1, 6)) * ct.one_atm,
#             phi=np.random.uniform(*env.phi_range),
#             total_time=np.random.uniform(*env.time_range),
#             dt=np.random.uniform(*env.dt_range),
#             etol=env.etol
#         )

#         T_ref_init = env.ref_states[0][0]
#         T_ref_final = env.ref_states[-1][0]
#         pressure = env.current_pressure
#         phi = env.current_phi
#         total_time = env.total_time
#         dt = env.dt
#         print(f"[episode {ep}] - T_ref_init: {T_ref_init}, T_ref_final: {T_ref_final}, pressure: {pressure}, phi: {phi}, total_time: {total_time}, dt: {dt}")
#         # number of decision points = env.n_episodes
#         K = env.n_episodes
#         pbar = tqdm(range(min(K, max_steps_per_ep)), desc="Building reference-anchored dataset")
#         for k in pbar:
#             a_star, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_ref = reference_anchored_oracle(env, k)
#             X.append(obs_ref.astype(np.float32))
#             y.append(a_star)

#             # Advance the REAL env with the oracle action (optional but keeps trajectories sensible)
#             obs, reward, terminated, truncated, info = env.step(a_star)
#             if terminated or truncated:
#                 break
            
#             pbar.set_postfix({
#                 "episode": ep,
#                 "step": k,
#                 "err_qss": err_qss,
#                 "err_bdf": err_bdf,
#                 "cpu_qss": cpu_qss,
#                 "cpu_bdf": cpu_bdf,
#                 "Ti": T_ref_init,
#                 "Tf": T_ref_final,
#             })
#             pbar.update(1)   
            
#         if (ep + 1) % verbose_every == 0:
#             print(f"[ref-dataset] ep {ep+1}/{n_episodes}, total samples={len(X)}")

#     X = np.stack(X, axis=0).astype(np.float32)
#     y = np.array(y, dtype=np.int64)
#     np.savez_compressed(save_path, X=X, y=y)
#     print(f"[ref-dataset] saved {save_path} with X={X.shape}, y={y.shape}")





# # ---- Configure solver order: 0=BDF, 1=QSS (must match oracle_refanchored.py assumptions) ----
# SOLVER_CONFIGS = [
#     # 0 = BDF (CVODE)
#     {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},
#     # 1 = QSS
#     {'type': 'qss', 'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False, 'itermax': 2,
#      'epsmin': 0.002, 'epsmax': 100.0, 'abstol': 1e-8, 'mxsteps': 1000, 'name': 'QSS'},
# ]


# def env_maker():
#     """
#     Returns a fresh IntegratorSwitchingEnv instance each call.
#     The builder will call env.reset(...) with random ICs drawn from these ranges.
#     Bias ranges so many episodes include ignition within the horizon.
#     """
#     reward_cfg = dict(
#         epsilon=1e-4,
#         lambda_init=1.0,
#         lambda_lr=0.05,
#         target_violation=0.0,
#         cpu_log_delta=1e-3,
#         reward_clip=5.0,
#     )

#     env = IntegratorSwitchingEnv(
#         mechanism_file="large_mechanism/n-dodecane.yaml",  # or "gri30.yaml"
#         fuel="nc12h26",                                     # or "CH4:1.0" for GRI-30
#         oxidizer="O2:0.21, N2:0.79",                        # or "N2:3.76, O2:1.0"
#         # Ranges the builder will sample from at reset():
#         temp_range=(600.0, 1400.0),     # wide enough to see pre- & post-ignition
#         phi_range=(0.7, 1.6),
#         pressure_range=(1, 6),          # (ignored internally; builder passes explicit P in atm)
#         time_range=(1e-3, 1e-2),        # ensure horizon can include ignition
#         dt_range=(1e-6, 1e-6),          # fixed dt for comparability
#         etol=1e-4,
#         verbose=False,
#         termination_count_threshold=100,
#         reward_function=LagrangeReward1(**reward_cfg),
#     )

#     # Ensure solver indices: 0=BDF, 1=QSS (oracle assumes this order)
#     env.solver_configs = SOLVER_CONFIGS
#     return env



# if __name__ == "__main__":
#     build_ref_anchored_dataset(
#     env_maker=env_maker,
#     n_episodes=200,            # total episodes to sample for the dataset
#     max_steps_per_ep=200,      # cap decisions per episode
#     save_path="oracle_dataset_ref.npz",
#     verbose_every=10
#     )

#     print("Saved dataset to oracle_dataset_ref.npz")


# import numpy as np
# import time
# import copy
# from environment import IntegratorSwitchingEnv
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from reward_model import LagrangeReward1
# import cantera as ct
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import functools


# def simulate_one_superstep_from_state(env, action, start_state, dt, super_steps, next_ref_state):
#     """
#     Build a fresh solver and integrate EXACTLY one super-step horizon starting from 'start_state'.
#     Return (cpu_time, step_error) comparing the end state to 'next_ref_state'.
#     This does NOT touch the live env state.
#     """
#     cfg = env.solver_configs[action]
#     # fresh solver & a temporary gas bound to it
#     solver, gas_tmp = env._build_fresh_solver(action, start_state.copy())

#     # integrate 'super_steps' micro-steps of size dt
#     t0 = time.time()
#     try:
#         current_state = start_state.copy()
#         for _ in range(super_steps):
#             if cfg['type'] == 'cvode':
#                 solver.set_state(current_state, 0.0)
#                 current_state = solver.solve_to(dt)
#             elif cfg['type'] == 'qss':
#                 solver.setState(current_state.tolist(), 0.0)
#                 rc = solver.integrateToTime(dt)
#                 if rc != 0:
#                     raise RuntimeError(f"QSS integrateToTime failed rc={rc}")
#                 current_state = np.array(solver.y)
#             else:
#                 raise ValueError(f"Unknown solver type {cfg['type']}")
#             # keep gas_tmp coherent (not strictly required for error)
#             gas_tmp.TPY = current_state[0], env.current_pressure, current_state[1:]
#         cpu_time = time.time() - t0

#         # compute step error vs next reference state
#         step_err = env._calculate_error(current_state, next_ref_state)
#         return cpu_time, step_err

#     except Exception:
#         return np.inf, np.inf


# def reference_anchored_oracle(env, k_step):
#     """
#     k_step indexes the decision point in the episode (0-based).
#     We:
#       - take start_ref = ref_states[k_index]
#       - take next_ref  = ref_states[k_index + 1]
#       where k_index = k_step * env.super_steps
#     Evaluate both actions from start_ref and pick the fastest that satisfies epsilon.
#     Fallback to BDF (assume action 0) otherwise.

#     Returns: (best_action, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_from_start_ref)
#     """
#     # map indices in the reference arrays
#     start_idx = k_step * env.super_steps
#     next_idx  = (k_step + 1) * env.super_steps
#     if next_idx >= len(env.ref_states):
#         # out of reference horizon, default to BDF
#         start_idx = max(0, len(env.ref_states) - env.super_steps - 1)
#         next_idx  = min(len(env.ref_states) - 1, start_idx + env.super_steps)

#     start_ref = env.ref_states[start_idx]
#     next_ref  = env.ref_states[next_idx]

#     # Evaluate both actions from the same (reference) start state
#     # Ensure action indices match your solver_configs: assume 0=BDF, 1=QSS
#     cpu_bdf, err_bdf = simulate_one_superstep_from_state(env, 0, start_ref, env.dt, env.super_steps, next_ref)
#     cpu_qss, err_qss = simulate_one_superstep_from_state(env, 1, start_ref, env.dt, env.super_steps, next_ref)

#     eps = env.reward_function.epsilon
#     # choose fastest feasible
#     feasible = []
#     if err_bdf <= eps: feasible.append((0, cpu_bdf))
#     if err_qss <= eps: feasible.append((1, cpu_qss))

#     if feasible:
#         feasible.sort(key=lambda x: x[1])
#         best_action = feasible[0][0]
#     else:
#         best_action = 0  # fallback to BDF when neither meets epsilon

#     # Build features from the SAME reference state (prevents drift leakage)
#     obs_ref = env._obs_from_arbitrary_state(start_ref, set_last_obs=False)
#     return best_action, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_ref


# def process_single_episode(episode_args):
#     """
#     Process a single episode - designed to be run in parallel.
#     Returns (X_episode, y_episode) for this episode.
#     """
#     ep_id, env_maker, max_steps_per_ep, seed = episode_args
    
#     # Set random seed for this worker
#     np.random.seed(seed)
    
#     env = env_maker()
#     X_episode, y_episode = [], []
    
#     # Reset environment with random parameters
#     obs, info = env.reset(
#         temperature=np.random.uniform(*env.temp_range),
#         pressure=np.random.choice(np.arange(1, 6)) * ct.one_atm,
#         phi=np.random.uniform(*env.phi_range),
#         total_time=np.random.uniform(*env.time_range),
#         dt=np.random.uniform(*env.dt_range),
#         etol=env.etol
#     )

#     T_ref_init = env.ref_states[0][0]
#     T_ref_final = env.ref_states[-1][0]
#     pressure = env.current_pressure
#     phi = env.current_phi
#     total_time = env.total_time
#     dt = env.dt
    
#     # Process steps in this episode
#     K = env.n_episodes
#     for k in range(min(K, max_steps_per_ep)):
#         a_star, (cpu_qss, err_qss), (cpu_bdf, err_bdf), obs_ref = reference_anchored_oracle(env, k)
#         X_episode.append(obs_ref.astype(np.float32))
#         y_episode.append(a_star)

#         # Advance the REAL env with the oracle action
#         obs, reward, terminated, truncated, info = env.step(a_star)
#         if terminated or truncated:
#             break
    
#     return ep_id, X_episode, y_episode, {
#         'T_ref_init': T_ref_init,
#         'T_ref_final': T_ref_final,
#         'pressure': pressure,
#         'phi': phi,
#         'total_time': total_time,
#         'dt': dt,
#         'n_steps': len(X_episode)
#     }


# def build_ref_anchored_dataset_parallel(env_maker,
#                                        n_episodes=150,
#                                        max_steps_per_ep=200,
#                                        save_path="oracle_dataset_ref.npz",
#                                        n_workers=None,
#                                        verbose_every=10):
#     """
#     Parallel version using ProcessPoolExecutor.
#     Each episode runs in a separate process.
#     """
#     if n_workers is None:
#         n_workers = min(cpu_count(), n_episodes)
    
#     print(f"Starting parallel dataset generation with {n_workers} workers")
    
#     # Prepare arguments for each episode
#     episode_args = []
#     base_seed = np.random.randint(0, 10000)
#     for ep in range(n_episodes):
#         episode_args.append((ep, env_maker, max_steps_per_ep, base_seed + ep))
    
#     X, y = [], []
    
#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         # Submit all episodes
#         future_to_episode = {executor.submit(process_single_episode, args): args[0] 
#                            for args in episode_args}
        
#         # Collect results as they complete
#         completed = 0
#         with tqdm(total=n_episodes, desc="Processing episodes") as pbar:
#             for future in as_completed(future_to_episode):
#                 ep_id = future_to_episode[future]
#                 try:
#                     ep_id, X_episode, y_episode, info = future.result()
#                     X.extend(X_episode)
#                     y.extend(y_episode)
                    
#                     completed += 1
#                     if completed % verbose_every == 0:
#                         print(f"\nCompleted {completed}/{n_episodes} episodes")
#                         print(f"Episode {ep_id}: T_init={info['T_ref_init']:.1f}, "
#                               f"T_final={info['T_ref_final']:.1f}, "
#                               f"P={info['pressure']:.1e}, phi={info['phi']:.2f}, "
#                               f"steps={info['n_steps']}")
                    
#                     pbar.update(1)
#                     pbar.set_postfix({
#                         'total_samples': len(X),
#                         'avg_steps': len(X) / completed if completed > 0 else 0
#                     })
                    
#                 except Exception as exc:
#                     print(f'Episode {ep_id} generated an exception: {exc}')
    
#     # Convert to arrays and save
#     X = np.stack(X, axis=0).astype(np.float32)
#     y = np.array(y, dtype=np.int64)
#     np.savez_compressed(save_path, X=X, y=y)
#     print(f"\n[parallel-dataset] saved {save_path} with X={X.shape}, y={y.shape}")


# def build_ref_anchored_dataset_multiprocessing(env_maker,
#                                               n_episodes=150,
#                                               max_steps_per_ep=200,
#                                               save_path="oracle_dataset_ref.npz",
#                                               n_workers=None,
#                                               verbose_every=10):
#     """
#     Alternative parallel version using multiprocessing.Pool.
#     """
#     if n_workers is None:
#         n_workers = min(cpu_count(), n_episodes)
    
#     print(f"Starting multiprocessing dataset generation with {n_workers} workers")
    
#     # Prepare arguments for each episode
#     episode_args = []
#     base_seed = np.random.randint(0, 10000)
#     for ep in range(n_episodes):
#         episode_args.append((ep, env_maker, max_steps_per_ep, base_seed + ep))
    
#     X, y = [], []
    
#     with Pool(processes=n_workers) as pool:
#         # Use imap for better progress tracking
#         results = pool.imap(process_single_episode, episode_args)
        
#         with tqdm(total=n_episodes, desc="Processing episodes") as pbar:
#             for completed, (ep_id, X_episode, y_episode, info) in enumerate(results, 1):
#                 X.extend(X_episode)
#                 y.extend(y_episode)
                
#                 if completed % verbose_every == 0:
#                     print(f"\nCompleted {completed}/{n_episodes} episodes")
#                     print(f"Episode {ep_id}: T_init={info['T_ref_init']:.1f}, "
#                           f"T_final={info['T_ref_final']:.1f}, "
#                           f"P={info['pressure']:.1e}, phi={info['phi']:.2f}, "
#                           f"steps={info['n_steps']}")
                
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     'total_samples': len(X),
#                     'avg_steps': len(X) / completed if completed > 0 else 0
#                 })
    
#     # Convert to arrays and save
#     X = np.stack(X, axis=0).astype(np.float32)
#     y = np.array(y, dtype=np.int64)
#     np.savez_compressed(save_path, X=X, y=y)
#     print(f"\n[multiprocessing-dataset] saved {save_path} with X={X.shape}, y={y.shape}")


# # ---- Configure solver order: 0=BDF, 1=QSS (must match oracle_refanchored.py assumptions) ----
# SOLVER_CONFIGS = [
#     # 0 = BDF (CVODE)
#     {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},
#     # 1 = QSS
#     {'type': 'qss', 'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False, 'itermax': 2,
#      'epsmin': 0.002, 'epsmax': 100.0, 'abstol': 1e-8, 'mxsteps': 1000, 'name': 'QSS'},
# ]


# def env_maker():
#     """
#     Returns a fresh IntegratorSwitchingEnv instance each call.
#     The builder will call env.reset(...) with random ICs drawn from these ranges.
#     Bias ranges so many episodes include ignition within the horizon.
#     """
#     reward_cfg = dict(
#         epsilon=1e-4,
#         lambda_init=1.0,
#         lambda_lr=0.05,
#         target_violation=0.0,
#         cpu_log_delta=1e-3,
#         reward_clip=5.0,
#     )

#     env = IntegratorSwitchingEnv(
#         mechanism_file="large_mechanism/n-dodecane.yaml",  # or "gri30.yaml"
#         fuel="nc12h26",                                     # or "CH4:1.0" for GRI-30
#         oxidizer="O2:0.21, N2:0.79",                        # or "N2:3.76, O2:1.0"
#         # Ranges the builder will sample from at reset():
#         temp_range=(600.0, 1400.0),     # wide enough to see pre- & post-ignition
#         phi_range=(0.7, 1.6),
#         pressure_range=(1, 6),          # (ignored internally; builder passes explicit P in atm)
#         time_range=(1e-3, 1e-2),        # ensure horizon can include ignition
#         dt_range=(1e-6, 1e-6),          # fixed dt for comparability
#         etol=1e-4,
#         verbose=False,
#         termination_count_threshold=100,
#         reward_function=LagrangeReward1(**reward_cfg),
#     )

#     # Ensure solver indices: 0=BDF, 1=QSS (oracle assumes this order)
#     env.solver_configs = SOLVER_CONFIGS
#     return env


# if __name__ == "__main__":
#     # Choose which parallel method to use
#     USE_PROCESS_POOL_EXECUTOR = True  # Set to False to use multiprocessing.Pool instead
    
#     if USE_PROCESS_POOL_EXECUTOR:
#         build_ref_anchored_dataset_parallel(
#             env_maker=env_maker,
#             n_episodes=200,
#             max_steps_per_ep=200,
#             save_path="oracle_dataset_ref_parallel.npz",
#             n_workers=None,  # Uses all available cores by default
#             verbose_every=10
#         )
#     else:
#         build_ref_anchored_dataset_multiprocessing(
#             env_maker=env_maker,
#             n_episodes=200,
#             max_steps_per_ep=200,
#             save_path="oracle_dataset_ref_mp.npz",
#             n_workers=None,  # Uses all available cores by default
#             verbose_every=10
#         )
    
#     print("Parallel dataset generation completed!")



# make_ref_dataset_streaming.py
# Usage:
#   python make_ref_dataset_streaming.py \
#       --episodes 200 --max_steps 200 --shard_size 2000 --out_dir dataset_shards
#
# Produces shards: dataset_shards/shard_000.npz, shard_001.npz, ...

import os, glob, gc, time, argparse
import numpy as np
import cantera as ct
from tqdm import tqdm
from reward_model import LagrangeReward1
from environment import IntegratorSwitchingEnv  # <-- your env file

# ------------------------- Solver config: 0=BDF, 1=QSS -------------------------
SOLVER_CONFIGS = [
    {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},  # 0
    {'type': 'qss',   'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False,
     'itermax': 2, 'epsmin': 0.002, 'epsmax': 10.0, 'abstol': 1e-8, 'mxsteps': 1000, 'name': 'QSS'},  # 1
]

# ------------------------- Reference runner (no big arrays) -------------------------
class ReferenceRunner:
    """High-accuracy reference integrator, advancing one super-step at a time."""
    def __init__(self, env):
        gas = ct.Solution(env.mechanism_file)
        gas.TPY = env.current_state[0], env.current_pressure, env.current_state[1:]
        self.reactor = ct.IdealGasConstPressureReactor(gas)
        self.net = ct.ReactorNet([self.reactor])
        self.net.rtol = 1e-10
        self.net.atol = 1e-20
        self.t = 0.0
        self.dt = env.dt
        self.super_steps = env.super_steps
        self.P = env.current_pressure

    def current_state(self):
        return np.hstack([self.reactor.T, self.reactor.thermo.Y])

    def advance_one_super_step(self):
        for _ in range(self.super_steps):
            self.t += self.dt
            self.net.advance(self.t)
        return np.hstack([self.reactor.T, self.reactor.thermo.Y])

# ------------------------- Simulation helpers -------------------------
def simulate_one_superstep_from_state(env, action, start_state, dt, super_steps, next_ref_state):
    """Integrate exactly one super-step from start_state using solver 'action' (fresh solver)."""
    cfg = env.solver_configs[action]
    if hasattr(env, "_build_fresh_solver"):
        solver, gas_tmp = env._build_fresh_solver(action, start_state.copy())
    else:
        raise RuntimeError("Env needs _build_fresh_solver(action, init_state).")

    try:
        cur = start_state.copy()
        t0 = time.time()
        for _ in range(super_steps):
            if cfg['type'] == 'cvode':
                solver.set_state(cur, 0.0)
                cur = solver.solve_to(dt)
            elif cfg['type'] == 'qss':
                solver.setState(cur.tolist(), 0.0)
                rc = solver.integrateToTime(dt)
                if rc != 0:
                    raise RuntimeError(f"QSS failed rc={rc}")
                cur = np.array(solver.y)
            else:
                raise ValueError(cfg['type'])
            gas_tmp.TPY = cur[0], env.current_pressure, cur[1:]
        cpu = time.time() - t0
        err = env._calculate_error(cur, next_ref_state)
        return cpu, err
    except Exception:
        return np.inf, np.inf

def obs_from_state(env, state):
    """Use env._obs_from_arbitrary_state if available; else rebuild base obs here."""
    if hasattr(env, "_obs_from_arbitrary_state"):
        return env._obs_from_arbitrary_state(state, set_last_obs=False).astype(np.float32)

    # Fallback: base obs T_norm + log10(key species) + logP
    temp = state[0]
    temp_norm = (temp - 300.0) / 2000.0
    key_vals = []
    for spec in getattr(env, "key_species", ['O','H','OH','H2O','O2','H2','H2O2','N2']):
        try:
            idx = env.gas.species_index(spec)
            key_vals.append(state[1:][idx])
        except ValueError:
            key_vals.append(0.0)
    species_log = np.log10(np.maximum(key_vals, 1e-20))
    pressure_log = np.log10(env.current_pressure / ct.one_atm)
    return np.hstack([temp_norm, species_log, pressure_log]).astype(np.float32)

def reference_anchored_oracle_from_states(env, start_ref, next_ref):
    """Pick fastest feasible (≤ epsilon) between BDF (0) and QSS (1); fallback BDF."""
    eps = env.reward_function.epsilon
    cpu_bdf, err_bdf = simulate_one_superstep_from_state(env, 0, start_ref, env.dt, env.super_steps, next_ref)
    cpu_qss, err_qss = simulate_one_superstep_from_state(env, 1, start_ref, env.dt, env.super_steps, next_ref)
    feasible = []
    if err_bdf <= eps: feasible.append((0, cpu_bdf))
    if err_qss <= eps: feasible.append((1, cpu_qss))
    if feasible:
        feasible.sort(key=lambda x: x[1])
        best = feasible[0][0]
    else:
        best = 0
    return best, (cpu_bdf, err_bdf), (cpu_qss, err_qss), obs_from_state(env, start_ref), 

# ------------------------- Env factory (low-RAM) -------------------------
def make_env():
    reward_cfg = dict(epsilon=1e-3, lambda_init=1.0, lambda_lr=0.05,
                      target_violation=0.0, cpu_log_delta=1e-3, reward_clip=5.0)
    
    env = IntegratorSwitchingEnv(
        mechanism_file="large_mechanism/n-dodecane.yaml",
        fuel="nc12h26", oxidizer="O2:0.21, N2:0.79",
        temp_range=(600.0, 1400.0), phi_range=(0.7, 1.6),
        pressure_range=(1, 6), time_range=(1e-3, 5e-2),
        dt_range=(1e-6, 1e-6), etol=1e-4, verbose=False,
        termination_count_threshold=100,
        reward_function=LagrangeReward1(**reward_cfg),
        # <- the two flags below keep memory small
        precompute_reference=False,
        track_trajectory=False,
    )
    env.solver_configs = SOLVER_CONFIGS
    return env

# ------------------------- Shard writer -------------------------
def next_shard_index(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    if not files: return 0
    last = os.path.basename(files[-1])
    try:
        return int(last.split("_")[1].split(".")[0]) + 1
    except Exception:
        return 0

def build_streaming(episodes, max_steps, shard_size, out_dir, seed=123, resume=True):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    shard_idx = next_shard_index(out_dir) if resume else 0
    X_buf, y_buf = [], []
    total = 0

    for ep in range(episodes):
        env = make_env()
        # sample ICs (bias to get ignition within horizon)
        T = np.random.uniform(*env.temp_range)
        P = np.random.choice(np.arange(1, 6)) * ct.one_atm
        phi = np.random.uniform(*env.phi_range)
        total_time = np.random.uniform(*env.time_range)
        dt = env.dt_range[0]
        env.reset(temperature=T, pressure=P, phi=phi, total_time=total_time, dt=dt, etol=env.etol)
        print(f"Initial temperature: {T}, Initial pressure: {P}, Initial phi: {phi}, Total time: {total_time}, dt: {dt}")
        ref = ReferenceRunner(env)
        start_ref = ref.current_state()

        steps = min(env.n_episodes, max_steps)
        pbar = tqdm(total=steps, desc="Building reference-anchored dataset")
        for k in range(steps):
            next_ref = ref.advance_one_super_step()
            a_star, (cpu_bdf, err_bdf), (cpu_qss, err_qss), obs_ref = reference_anchored_oracle_from_states(env, start_ref, next_ref)

            X_buf.append(obs_ref)
            y_buf.append(np.int64(a_star))
            total += 1

            # advance live env with oracle action (so conditions evolve realistically)
            _, _, terminated, truncated, _ = env.step(a_star)
            if terminated or truncated:
                break
                
            pbar.update(1)
            pbar.set_postfix({
                'Ts': total,
                'E': ep,
                'a*': a_star,
                'as': total / (k + 1) if k + 1 > 0 else 0,
                'cbdf': cpu_bdf,
                'cqss': cpu_qss,
                'ebdf': err_bdf,
                'eqss': err_qss,
            })

            start_ref = next_ref

            # flush shard periodically
            if len(X_buf) >= shard_size:
                path = os.path.join(out_dir, f"shard_{shard_idx:03d}.npz")
                np.savez_compressed(path,
                                    X=np.asarray(X_buf, dtype=np.float32),
                                    y=np.asarray(y_buf, dtype=np.int64))
                print(f"[save] {path} (+{len(X_buf)} samples) total={total}")
                shard_idx += 1
                X_buf.clear(); y_buf.clear()
                gc.collect()

        del env, ref
        gc.collect()
        if (ep + 1) % 10 == 0:
            pbar.close()
            print(f"[progress] episodes {ep+1}/{episodes} | total_samples={total}")

    # final flush
    if X_buf:
        path = os.path.join(out_dir, f"shard_{shard_idx:03d}.npz")
        np.savez_compressed(path,
                            X=np.asarray(X_buf, dtype=np.float32),
                            y=np.asarray(y_buf, dtype=np.int64))
        print(f"[save] {path} (+{len(X_buf)} samples) total={total}")

    print(f"[done] wrote shards to {out_dir} | total_samples={total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--out_dir", type=str, default="dataset_shards")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    build_streaming(args.episodes, args.max_steps, args.shard_size, args.out_dir, seed=args.seed, resume=args.resume)
