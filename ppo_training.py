#!/usr/bin/env python3
"""
PPO Training Script for Combustion Integrator Switching Environment
Implements detailed logging and monitoring without TensorBoard
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import json
import os
import time
from datetime import datetime
import pandas as pd
from collections import deque, defaultdict
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from environment import IntegratorSwitchingEnv
import math
import cantera as ct
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

# Optional Neptune import (only needed when logging is enabled)
try:
    import neptune
    from neptune import init_run as neptune_init_run
    from neptune.types import File as NeptuneFile
except Exception:
    neptune = None
    neptune_init_run = None
    NeptuneFile = None


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        # x shape [N, obs_dim]
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        new_var = M2 / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class PPONetwork(nn.Module):
    """Improved PPO Actor-Critic Network for combustion integrator selection"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
        super(PPONetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers - SIMPLIFIED
        layers = []
        prev_dim = obs_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            # REMOVED: Dropout - not needed for this problem size
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy) - SIMPLIFIED
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, action_dim)  # Direct output, no extra layers
        )
        
        # Critic head (value function) - SIMPLIFIED
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 1)  # Direct output, no extra layers
        )
        
        # Initialize weights with proper variance for learning
        self._init_weights()
    
    def _init_weights(self):

        for m in self.shared_net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

        # Small logits at start -> high entropy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

        # Critic ok with gain 1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)
    
    def forward(self, x):
        # Ensure network is in correct mode
        shared_features = self.shared_net(x)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value
    
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

class PPOAgent:
    """PPO Agent for combustion integrator selection"""
    def __init__(self, obs_dim, action_dim, lr=3e-3, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, ent_coef=0.3, vf_coef=0.5, max_grad_norm=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 policy_network_arch=[256, 128, 64]):

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # === Normalizer (owned by the agent) ===
        self.obs_rms = RunningMeanStd(shape=obs_dim)

        # Exploration params
        self.initial_ent_coef = ent_coef
        self.min_ent_coef = 0.01
        self.ent_coef_decay = 0.9995
        self.exploration_bonus = 0.1
        self.action_counts = {0: 0, 1: 0}

        self.network = PPONetwork(obs_dim, action_dim, hidden_dims=policy_network_arch).to(device).float()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        print(f"PPO Agent initialized on {device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    # --- helper: normalize numpy/tensor ---
    def _norm_np(self, obs_np: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs_np)

    def _norm_t(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.obs_rms.mean, device=obs_t.device, dtype=obs_t.dtype)
        std  = torch.sqrt(torch.as_tensor(self.obs_rms.var, device=obs_t.device, dtype=obs_t.dtype) + 1e-8)
        return (obs_t - mean) / std

    # call this from training code after collecting a rollout (see §3)
    def update_obs_rms(self, obs_batch_np: np.ndarray):
        # obs_batch_np: [N, obs_dim] raw observations
        if obs_batch_np.ndim == 1:
            obs_batch_np = obs_batch_np[None, :]
        if len(obs_batch_np) > 0:
            self.obs_rms.update(obs_batch_np.astype(np.float32))

    @torch.no_grad()
    def get_action(self, obs_raw_np):
        # normalize inside the agent
        x = self._norm_np(np.asarray(obs_raw_np, dtype=np.float32))
        obs_t = torch.from_numpy(x).to(self.device, dtype=torch.float32).unsqueeze(0)
        logits, value = self.network(obs_t)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        a = int(action.item())
        self.action_counts[a] = self.action_counts.get(a, 0) + 1
        return a, float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def act_deterministic(self, obs_raw_np):
        x = self._norm_np(np.asarray(obs_raw_np, dtype=np.float32))
        obs_t = torch.from_numpy(x).to(self.device, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.network(obs_t)
        return int(torch.argmax(logits, dim=1).item())

    @torch.no_grad()
    def get_value(self, obs_raw_np):
        x = self._norm_np(np.asarray(obs_raw_np, dtype=np.float32))
        obs_t = torch.from_numpy(x).to(self.device, dtype=torch.float32).unsqueeze(0)
        _, value = self.network(obs_t)
        return value.squeeze(0).item()

    def update(self,
               rollout_data,
               update_epochs: int = 3,
               minibatch_size: int = 256,
               target_kl: float = 0.02,
               clip_coef_vf: float = 0.2,
               use_huber_vf: bool = True):
        """
        PPO update. rollouts['observations'] should be RAW obs.
        We normalize them inside using the FROZEN obs_rms.
        """
        device = self.device
        obs_raw    = torch.as_tensor(rollout_data['observations'], dtype=torch.float32, device=device)
        actions    = torch.as_tensor(rollout_data['actions'],       dtype=torch.long,   device=device)
        old_logp   = torch.as_tensor(rollout_data['log_probs'],     dtype=torch.float32, device=device)
        returns    = torch.as_tensor(rollout_data['returns'],       dtype=torch.float32, device=device)
        old_values = torch.as_tensor(rollout_data['values'],        dtype=torch.float32, device=device).squeeze(-1)
        advantages = torch.as_tensor(rollout_data['advantages'],    dtype=torch.float32, device=device)

        batch_size = obs_raw.shape[0]
        indices = np.arange(batch_size)

        logs = {"policy_loss":0.0,"value_loss":0.0,"entropy_loss":0.0,"total_loss":0.0,
                "approx_kl":0.0,"clipfrac":0.0,"explained_variance":0.0}
        count = 0
        early_stop = False

        for _ in range(update_epochs):
            if early_stop: break
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = indices[start:end]

                mb_obs_n = self._norm_t(obs_raw[mb])          # <<< normalize here
                logits, v_pred = self.network(mb_obs_n)
                v_pred = v_pred.squeeze(-1)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = advantages[mb] * ratio
                surr2 = advantages[mb] * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clip = old_values[mb] + (v_pred - old_values[mb]).clamp(-clip_coef_vf, clip_coef_vf)
                if use_huber_vf:
                    v_uncl = F.smooth_l1_loss(v_pred, returns[mb], reduction='mean')
                    v_clp  = F.smooth_l1_loss(v_clip, returns[mb], reduction='mean')
                else:
                    v_uncl = F.mse_loss(v_pred, returns[mb], reduction='mean')
                    v_clp  = F.mse_loss(v_clip, returns[mb], reduction='mean')
                value_loss = torch.max(v_uncl, v_clp)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_logp - old_logp[mb]
                    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean().clamp_min(0.0)
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                    var_y = returns[mb].var()
                    if var_y > 1e-8:
                        ev = 1.0 - F.mse_loss(v_pred, returns[mb]) / var_y
                    else:
                        ev = torch.tensor(0.0, device=device)

                if approx_kl.item() > target_kl:
                    early_stop = True

                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"]  += value_loss.item()
                logs["entropy_loss"]+= entropy.item()
                logs["total_loss"]  += loss.item()
                logs["approx_kl"]   += approx_kl.item()
                logs["clipfrac"]    += clipfrac.item()
                logs["explained_variance"] += float(ev.item())
                count += 1

        for k in logs:
            logs[k] = logs[k] / max(1, count)

        # anneal entropy
        self.ent_coef = max(self.min_ent_coef, self.ent_coef * self.ent_coef_decay)

        # keep action counts bounded
        tot = sum(self.action_counts.values())
        if tot > 10000:
            for a in self.action_counts:
                self.action_counts[a] = int(self.action_counts[a] * 0.5)

        logs['entropy_coef'] = float(self.ent_coef)
        logs['action_counts'] = self.action_counts.copy()
        return logs


class DetailedLogger:
    """Comprehensive logging system for PPO training"""
    
    def __init__(self, log_dir, experiment_name, neptune_run: Optional[object] = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.neptune_run = neptune_run
        
        # Create logging directories
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'evaluations'), exist_ok=True)
        
        # Initialize logging data
        self.episode_data = []
        self.update_data = []
        self.solver_usage_data = []
        self.combustion_metrics = []
        self.evaluation_data = []
        
        # Real-time tracking
        self.recent_rewards = deque(maxlen=100)
        self.recent_episode_lengths = deque(maxlen=100)
        self.solver_usage_count = defaultdict(int)
        
        print(f"Logger initialized: {self.experiment_dir}")
        if self.neptune_run is not None:
            print("Neptune logging is ENABLED for this run.")
    
    def log_episode(self, episode_num, episode_data, env_info):
        """Log episode-level data"""
        
        # Extract episode metrics
        episode_reward = sum(episode_data['rewards'])
        episode_length = len(episode_data['rewards'])
        
        self.recent_rewards.append(episode_reward)
        self.recent_episode_lengths.append(episode_length)
        
        # Count solver usage
        for action in episode_data['actions']:
            self.solver_usage_count[action] += 1
        
        # Detailed episode info
        log_entry = {
            'episode': episode_num,
            'timestamp': datetime.now().isoformat(),
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_reward_per_step': episode_reward / episode_length if episode_length > 0 else 0,
            'actions': episode_data['actions'],
            'rewards': episode_data['rewards'],
            'solver_usage': dict(self.solver_usage_count),
            'env_info': env_info
        }
        
        # Environment-specific metrics
        if 'episode_stats' in env_info:
            stats = env_info['episode_stats']
            log_entry.update({
                'avg_cpu_time': stats.get('avg_cpu_time', 0),
                'avg_timestep_error': stats.get('avg_timestep_error', 0),
                'avg_episode_reward': stats.get('avg_episode_reward', 0),
                'within_threshold_rate': stats.get('within_timestep_threshold_rate', 0)
            })
        
        # Combustion-specific metrics
        if 'steady_state_info' in env_info:
            ss_info = env_info['steady_state_info']
            combustion_entry = {
                'episode': episode_num,
                'ignition_detected': ss_info.get('ignition_detected', False),
                'ignition_time': ss_info.get('ignition_time', None),
                'reached_steady_state': ss_info.get('reached_steady_state', False),
                'current_time': ss_info.get('current_time', 0),
                'initial_temp': env_info['current_conditions'].get('temperature', 0),
                'initial_pressure': env_info['current_conditions'].get('pressure', 0)
            }
            self.combustion_metrics.append(combustion_entry)
        
        self.episode_data.append(log_entry)
        # Neptune logging
        if self.neptune_run is not None:
            self.neptune_run["train/episode/reward"].append(episode_reward, step=episode_num)
            self.neptune_run["train/episode/length"].append(episode_length, step=episode_num)
            if 'episode_stats' in env_info:
                stats = env_info['episode_stats']
                if isinstance(stats.get('avg_cpu_time', None), (int, float)):
                    self.neptune_run["train/episode/avg_cpu_time"].append(stats.get('avg_cpu_time', 0.0), step=episode_num)
                if isinstance(stats.get('avg_timestep_error', None), (int, float)):
                    self.neptune_run["train/episode/avg_timestep_error"].append(stats.get('avg_timestep_error', 0.0), step=episode_num)
                self.neptune_run["train/episode/within_threshold_rate"].append(stats.get('within_timestep_threshold_rate', 0.0), step=episode_num)
            if 'steady_state_info' in env_info:
                ssi = env_info['steady_state_info']
                self.neptune_run["train/episode/ignition_detected"].append(float(bool(ssi.get('ignition_detected', False))), step=episode_num)
    
    def log_update(self, update_num, update_info):
        """Log PPO update metrics"""
        
        update_entry = {
            'update': update_num,
            'timestamp': datetime.now().isoformat(),
            **update_info
        }
        
        self.update_data.append(update_entry)
        # Neptune logging
        if self.neptune_run is not None:
            for key, value in update_info.items():
                if key == 'action_counts':
                    # Log action counts separately as a namespace
                    for action_id, count in value.items():
                        self.neptune_run[f"train/update/action_counts/{action_id}"].append(count, step=update_num)
                elif isinstance(value, (int, float)):
                    self.neptune_run[f"train/update/{key}"].append(value, step=update_num)
    
    def log_evaluation(self, episode_num, eval_results):
        """Log evaluation results"""
        
        # Calculate average action distribution properly
        all_action_dists = [r['average_action_distribution'] for r in eval_results]
        avg_action_dist = {}
        if all_action_dists:
            # Get all solver names from first result
            solver_names = list(all_action_dists[0].keys())
            for solver in solver_names:
                avg_action_dist[solver] = np.mean([dist[solver] for dist in all_action_dists])
        
        eval_entry = {
            'episode': episode_num,
            'timestamp': datetime.now().isoformat(),
            'eval_episodes': len(eval_results),
            'mean_reward': np.mean([r['mean_reward'] for r in eval_results]),
            'std_reward': np.std([r['mean_reward'] for r in eval_results]),
            'mean_cpu_time': np.mean([r['mean_cpu_time'] for r in eval_results]),
            'std_cpu_time': np.std([r['mean_cpu_time'] for r in eval_results]),
            'ignition_success_rate': np.mean([r['ignition_success_rate'] for r in eval_results]),
            'average_action_distribution': avg_action_dist,
            'eval_results': eval_results
        }
        
        self.evaluation_data.append(eval_entry)
        # Neptune logging
        if self.neptune_run is not None:
            self.neptune_run["eval/mean_reward"].append(float(eval_entry['mean_reward']), step=episode_num)
            self.neptune_run["eval/std_reward"].append(float(eval_entry['std_reward']), step=episode_num)
            self.neptune_run["eval/mean_cpu_time"].append(float(eval_entry['mean_cpu_time']), step=episode_num)
            self.neptune_run["eval/std_cpu_time"].append(float(eval_entry['std_cpu_time']), step=episode_num)
            self.neptune_run["eval/ignition_success_rate"].append(float(eval_entry['ignition_success_rate']), step=episode_num)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS (Episode {episode_num})")
        print(f"{'='*60}")
        print(f"Episodes: {len(eval_results)}")
        print(f"Mean Reward: {eval_entry['mean_reward']:.3f} ± {eval_entry['std_reward']:.3f}")
        print(f"Mean CPU Time: {eval_entry['mean_cpu_time']:.3f} ± {eval_entry['std_cpu_time']:.3f}")
        print(f"Average Action Distribution: {eval_entry['average_action_distribution']}")
        print(f"{'='*60}")
    
    def print_progress(self, episode_num, total_episodes):
        """Print training progress"""
        
        if len(self.recent_rewards) == 0:
            return
        
        # Calculate statistics
        mean_reward = np.mean(self.recent_rewards)
        std_reward = np.std(self.recent_rewards) 
        mean_length = np.mean(self.recent_episode_lengths)
        
        # Solver usage percentages
        total_actions = sum(self.solver_usage_count.values())
        usage_str = ", ".join([
            f"S{solver}: {count/total_actions*100:.1f}%" 
            for solver, count in sorted(self.solver_usage_count.items())
        ]) if total_actions > 0 else "No actions yet"
        
        print(f"\n{'='*80}")
        print(f"Episode {episode_num:5d}/{total_episodes:5d} ({episode_num/total_episodes*100:.1f}%)")
        print(f"Reward: {mean_reward:8.3f} ± {std_reward:6.3f} (last 100 episodes)")
        print(f"Length: {mean_length:6.1f} steps average")
        print(f"Solver usage: {usage_str}")
        
        # Latest episode combustion metrics
        if self.combustion_metrics:
            latest = self.combustion_metrics[-1]
            if latest['ignition_detected']:
                print(f"Combustion: Ignition @ {latest['ignition_time']:.6f}s, "
                      f"T₀={latest['initial_temp']:.0f}K")
            else:
                print(f"Combustion: No ignition, T₀={latest['initial_temp']:.0f}K")
        
        print(f"{'='*80}")
    
    def save_data(self, episode_num):
        """Save all logged data to files"""
        
        # Save episode data
        episode_df = pd.DataFrame([
            {
                'episode': ep['episode'],
                'reward': ep['episode_reward'],
                'length': ep['episode_length'],
                'mean_reward_per_step': ep['mean_reward_per_step'],
                'avg_cpu_time': ep.get('avg_cpu_time', 0),
                'avg_timestep_error': ep.get('avg_timestep_error', 0),
                'within_threshold_rate': ep.get('within_threshold_rate', 0)
            }
            for ep in self.episode_data
        ])
        episode_df.to_csv(os.path.join(self.experiment_dir, 'episodes.csv'), index=False)
        
        # Save update data
        if self.update_data:
            update_df = pd.DataFrame(self.update_data)
            update_df.to_csv(os.path.join(self.experiment_dir, 'updates.csv'), index=False)
        
        # Save combustion metrics
        if self.combustion_metrics:
            combustion_df = pd.DataFrame(self.combustion_metrics)
            combustion_df.to_csv(os.path.join(self.experiment_dir, 'combustion.csv'), index=False)
        
        # Save evaluation data
        if self.evaluation_data:
            eval_rows = []
            for eval_data in self.evaluation_data:
                row = {
                    'episode': eval_data['episode'],
                    'mean_reward': eval_data['mean_reward'],
                    'std_reward': eval_data['std_reward'],
                    'mean_cpu_time': eval_data['mean_cpu_time'],
                    'std_cpu_time': eval_data['std_cpu_time']
                }
                # Add action distribution as separate columns
                action_dist = eval_data['average_action_distribution']
                if isinstance(action_dist, dict):
                    for solver, prob in action_dist.items():
                        row[f'action_prob_{solver}'] = prob
                else:
                    row['action_distribution'] = str(action_dist)
                eval_rows.append(row)
            
            eval_df = pd.DataFrame(eval_rows)
            eval_df.to_csv(os.path.join(self.experiment_dir, 'evaluations.csv'), index=False)
        
        # Save raw episode data with actions/rewards
        with open(os.path.join(self.experiment_dir, 'raw_episodes.pkl'), 'wb') as f:
            pickle.dump(self.episode_data, f)
        
        # Save raw evaluation data
        if self.evaluation_data:
            with open(os.path.join(self.experiment_dir, 'raw_evaluations.pkl'), 'wb') as f:
                pickle.dump(self.evaluation_data, f)
        
        print(f"Data saved at episode {episode_num}")
        # Upload artifacts to Neptune
        if self.neptune_run is not None:
            try:
                self.neptune_run["artifacts/episodes"].upload(os.path.join(self.experiment_dir, 'episodes.csv'))
                if self.update_data:
                    self.neptune_run["artifacts/updates"].upload(os.path.join(self.experiment_dir, 'updates.csv'))
                if self.combustion_metrics:
                    self.neptune_run["artifacts/combustion"].upload(os.path.join(self.experiment_dir, 'combustion.csv'))
                if self.evaluation_data:
                    self.neptune_run["artifacts/evaluations"].upload(os.path.join(self.experiment_dir, 'evaluations.csv'))
            except Exception as e:
                print(f"Warning: Failed to upload CSV artifacts to Neptune: {e}")
    
    def create_plots(self):
        """Generate comprehensive training plots"""
        
        if not self.episode_data:
            return
        
        # Episode rewards plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Episode rewards
        plt.subplot(2, 3, 1)
        episodes = [ep['episode'] for ep in self.episode_data]
        rewards = [ep['episode_reward'] for ep in self.episode_data]
        
        plt.plot(episodes, rewards, alpha=0.3, color='blue')
        
        # Moving average
        window = min(50, len(rewards))
        if window > 1:
            moving_avg = pd.Series(rewards).rolling(window).mean()
            plt.plot(episodes, moving_avg, color='red', linewidth=2, label=f'{window}-episode MA')
            plt.legend()
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Training Rewards')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Episode lengths
        plt.subplot(2, 3, 2)
        lengths = [ep['episode_length'] for ep in self.episode_data]
        plt.plot(episodes, lengths, alpha=0.6, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Solver usage over time
        plt.subplot(2, 3, 3)
        solver_counts = defaultdict(list)
        cumulative_counts = defaultdict(int)
        
        for ep in self.episode_data:
            for solver in ep['solver_usage']:
                cumulative_counts[solver] = ep['solver_usage'][solver]
            
            total = sum(cumulative_counts.values())
            for solver in sorted(cumulative_counts.keys()):
                percentage = cumulative_counts[solver] / total * 100 if total > 0 else 0
                solver_counts[solver].append(percentage)
        
        for solver in sorted(solver_counts.keys()):
            plt.plot(episodes[:len(solver_counts[solver])], solver_counts[solver], 
                    label=f'Solver {solver}', marker='o', markersize=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Solver Usage (%)')
        plt.title('Solver Selection Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: CPU time efficiency
        plt.subplot(2, 3, 4)
        cpu_times = [ep.get('avg_cpu_time', 0) for ep in self.episode_data]
        if any(cpu_times):
            plt.plot(episodes, cpu_times, color='orange', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Average CPU Time')
            plt.title('CPU Time Efficiency')
            plt.grid(True, alpha=0.3)
        
        # Subplot 5: Error rates
        plt.subplot(2, 3, 5)
        error_rates = [ep.get('avg_timestep_error', 0) for ep in self.episode_data]
        if any(error_rates):
            plt.semilogy(episodes, error_rates, color='red', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Average Error (log scale)')
            plt.title('Integration Errors')
            plt.grid(True, alpha=0.3)
        
        # Subplot 6: Combustion success rate
        plt.subplot(2, 3, 6)
        if self.combustion_metrics:
            combustion_df = pd.DataFrame(self.combustion_metrics)
            
            # Success rate over time
            window_size = 20
            success_rates = []
            episode_windows = []
            
            for i in range(len(combustion_df)):
                start_idx = max(0, i - window_size + 1)
                window_data = combustion_df.iloc[start_idx:i+1]
                success_rate = window_data['ignition_detected'].mean() * 100
                success_rates.append(success_rate)
                episode_windows.append(combustion_df.iloc[i]['episode'])
            
            plt.plot(episode_windows, success_rates, color='purple', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Ignition Success Rate (%)')
            plt.title(f'Ignition Success (last {window_size} episodes)')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'plots', 'training_progress.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        # Upload to Neptune
        if self.neptune_run is not None:
            try:
                self.neptune_run["plots/training_progress"].upload(os.path.join(self.experiment_dir, 'plots', 'training_progress.png'))
            except Exception as e:
                print(f"Warning: Failed to upload training_progress.png to Neptune: {e}")
        
        # PPO-specific plots
        if self.update_data:
            self._create_ppo_plots()
    
    def _create_ppo_plots(self):
        """Create PPO-specific training plots"""
        
        plt.figure(figsize=(12, 8))
        update_df = pd.DataFrame(self.update_data)
        
        # Loss plots
        plt.subplot(2, 3, 1)
        plt.plot(update_df['update'], update_df['policy_loss'], label='Policy Loss')
        plt.plot(update_df['update'], update_df['value_loss'], label='Value Loss')
        plt.plot(update_df['update'], update_df['total_loss'], label='Total Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.title('PPO Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # KL divergence
        plt.subplot(2, 3, 2)
        plt.plot(update_df['update'], update_df['approx_kl'])
        plt.xlabel('Update')
        plt.ylabel('Approximate KL')
        plt.title('KL Divergence')
        plt.grid(True, alpha=0.3)
        
        # Clip fraction
        plt.subplot(2, 3, 3)
        plt.plot(update_df['update'], update_df['clipfrac'])
        plt.xlabel('Update')
        plt.ylabel('Clip Fraction')
        plt.title('PPO Clipping Rate')
        plt.grid(True, alpha=0.3)
        
        # Entropy
        plt.subplot(2, 3, 4)
        plt.plot(update_df['update'], update_df['entropy_loss'])
        plt.xlabel('Update')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy')
        plt.grid(True, alpha=0.3)
        
        # Explained variance
        plt.subplot(2, 3, 5)
        plt.plot(update_df['update'], update_df['explained_variance'])
        plt.xlabel('Update')
        plt.ylabel('Explained Variance')
        plt.title('Value Function Fit')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'plots', 'ppo_metrics.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        if self.neptune_run is not None:
            try:
                self.neptune_run["plots/ppo_metrics"].upload(os.path.join(self.experiment_dir, 'plots', 'ppo_metrics.png'))
            except Exception as e:
                print(f"Warning: Failed to upload ppo_metrics.png to Neptune: {e}")
    
    def create_evaluation_plots(self, episode_num, eval_results):
        """Create evaluation plots for temperature, CPU time, and action trajectories"""
        
        if not eval_results:
            return
        
        # Create plots for each evaluation parameter set
        for i, result in enumerate(eval_results):
            print(f"Creating evaluation plots for episode {episode_num} - set {i+1}")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Evaluation Episode {episode_num} - Set {i+1}\n'
                        f'T={result["conditions"]["temperature"]:.0f}K, '
                        f'P={result["conditions"]["pressure"]:.1f}bar, '
                        f'φ={result["conditions"]["Z"]:.2f}', fontsize=14)
            
            # Extract data from the first episode of this condition set
            if 'episodes' in result and result['episodes']:
                episode_data = result['episodes'][0]  # Use first episode for detailed plotting
                times = episode_data['times']
                temperatures = episode_data['temperatures']
                cpu_times = episode_data['cpu_times']
                rewards = episode_data['rewards']
                actions = episode_data['actions']
                species_profile = episode_data['trajectory'][:, 1:]
                solver_names = episode_data['solver_names']
                reference_temperatures = episode_data['reference_temperatures']
                reference_species = episode_data['reference_species'] 
                reference_times = episode_data['reference_times']
            else:
                # Fallback: create dummy data if structure is different
                print(f"Warning: No detailed episode data found for evaluation set {i+1}")
                continue
            
            # Plot 1: Temperature profile
            ax1 = axes[0, 0]
            ax1.plot(times, temperatures, 'b-', linewidth=2, label='Agent')
            if 'reference_temperatures' in episode_data:
                ax1.plot(reference_times[:len(times)], reference_temperatures[:len(times)], 'r--', linewidth=2, label='Reference')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Temperature (K)')
            ax1.set_title('Temperature Profile')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: CPU time profile
            ax2 = axes[0, 1]
            # Left axis for CPU time
            ax2.plot(np.arange(len(cpu_times)), cpu_times, 'g-', linewidth=2, label='CPU Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('CPU Time (s)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Right axis for reward
            ax2_right = ax2.twinx()
            ax2_right.plot(np.arange(len(rewards)), rewards, 'r-', linewidth=2, label='Reward')
            ax2_right.set_ylabel('Reward', color='r')
            ax2_right.tick_params(axis='y', labelcolor='r')
            
            ax2.set_title('CPU Time and Reward Profile')
            ax2.grid(True, alpha=0.3)
            
            # Add both legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_right.get_legend_handles_labels()
            ax2_right.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            # Plot 3: Action trajectory
            ax3 = axes[1, 0]
            # Create action trajectory plot
            action_colors = ['red', 'blue', 'green', 'orange', 'purple']
            for j, action in enumerate(actions):
                color = action_colors[action % len(action_colors)]
                ax3.scatter(np.arange(len(actions))[j], action, c=color, s=20, alpha=0.7)
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Solver Action')
            ax3.set_title('Solver Selection Trajectory')
            ax3.set_yticks(range(len(solver_names)))
            ax3.set_yticklabels(solver_names)
            ax3.grid(True, alpha=0.3)
                        
            # Plot 4: Solver species profile of CO2, H2O, H2, OH
            ax4 = axes[1, 1]
            species_names = ['CO2', 'H2O', 'H2', 'OH']
            for s in range(len(species_names)):
                specie_index = env.gas.species_index(species_names[s])
                ax4.plot(np.arange(len(times)), species_profile[:, specie_index], label=f'{species_names[s]}')
                ax4.plot(np.arange(len(reference_times))[:len(times)], reference_species[:, specie_index][:len(times)], label=f'{species_names[s]} Reference', linestyle='--')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Species Concentration')
            ax4.set_title('Species Profile')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
           
            plt.tight_layout()
            
            print(f"Saving plot for episode {episode_num} - set {i+1}")
            # Save plot
            filename = f'evaluation_ep_{episode_num}_set{i+1}.png'
            plt.savefig(os.path.join(self.experiment_dir, 'evaluations', filename), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            if self.neptune_run is not None:
                try:
                    self.neptune_run[f"evaluations/{filename}"].upload(os.path.join(self.experiment_dir, 'evaluations', filename))
                except Exception as e:
                    print(f"Warning: Failed to upload {filename} to Neptune: {e}")
        
        # # Create summary plot for all evaluation sets
        # self._create_evaluation_summary_plot(episode_num, eval_results)
    
    def _create_evaluation_summary_plot(self, episode_num, eval_results):
        """Create summary plot comparing all evaluation sets"""
        print(f"Creating evaluation summary plot for episode {episode_num}")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evaluation Summary - Episode {episode_num}', fontsize=16)
        
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Temperature profiles comparison
        ax1 = axes[0, 0]
        for i, result in enumerate(eval_results):
            if 'episodes' in result and result['episodes']:
                episode_data = result['episodes'][0]  # Use first episode for detailed plotting
                times = episode_data['times']
                temperatures = episode_data['temperatures']
                reference_temperatures = episode_data['reference_temperatures']
                reference_times = episode_data['reference_times']
                label = f"T={result['conditions']['temperature']:.0f}K, φ={result['conditions']['Z']:.2f}"
                ax1.plot(times, temperatures, color=colors[i], linewidth=2, label=label)
                ax1.plot(reference_times[:len(times)], reference_temperatures[:len(times)], color='black', linewidth=2, label='Reference', linestyle='--')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Profiles Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: CPU time comparison
        ax2 = axes[0, 1]
        for i, result in enumerate(eval_results):
            if 'episodes' in result and result['episodes']:
                episode_data = result['episodes'][0]  # Use first episode for detailed plotting
                cpu_times = episode_data['cpu_times']
                label = f"T={result['conditions']['temperature']:.0f}K, φ={result['conditions']['Z']:.2f}"
                ax2.plot(np.arange(len(cpu_times)), cpu_times, color=colors[i], linewidth=2, label=label)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('CPU Time (s)')
        ax2.set_title('CPU Time Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Solver usage comparison
        ax3 = axes[1, 0]
        if 'episodes' in eval_results[0] and eval_results[0]['episodes']:
            solver_names = eval_results[0]['episodes'][0]['solver_names']
            x = np.arange(len(solver_names))
            width = 0.25
            
            for i, result in enumerate(eval_results):
                if 'episodes' in result and result['episodes']:
                    episode_data = result['episodes'][0]
                    action_counts = np.bincount(np.array(episode_data['actions']), minlength=len(solver_names))
                    percentages = action_counts / len(episode_data['actions']) * 100
                    ax3.bar(x + i*width, percentages, width, label=f"Set {i+1}", color=colors[i])
        
        ax3.set_xlabel('Solver')
        ax3.set_ylabel('Usage Percentage (%)')
        ax3.set_title('Solver Usage Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(solver_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        ax4 = axes[1, 1]
        metrics = ['Mean Reward', 'Mean CPU Time', 'Mean Length']
        set_labels = [f"Set {i+1}" for i in range(len(eval_results))]
        
        rewards = [r['mean_reward'] for r in eval_results]
        cpu_times = [r['mean_cpu_time'] for r in eval_results]
        lengths = [r['mean_length'] for r in eval_results]
        
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        ax4.bar(x_pos - width, rewards, width, label='Reward', color='blue', alpha=0.7)
        ax4.bar(x_pos, cpu_times, width, label='CPU Time', color='red', alpha=0.7)
        ax4.bar(x_pos + width, lengths, width, label='Length', color='green', alpha=0.7)
        
        ax4.set_xlabel('Evaluation Set')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(set_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'evaluation_summary_ep{episode_num}.png'
        plt.savefig(os.path.join(self.experiment_dir, 'evaluations', filename), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        if self.neptune_run is not None:
            try:
                self.neptune_run[f"evaluations/{filename}"].upload(os.path.join(self.experiment_dir, 'evaluations', filename))
            except Exception as e:
                print(f"Warning: Failed to upload {filename} to Neptune: {e}")

def evaluate_agent(agent, env, eval_conditions, n_eval_episodes=5, deterministic=True):
    """
    Evaluate the agent on fixed conditions
    
    Args:
        agent: Trained PPO agent
        env: Environment instance
        eval_conditions: List of dictionaries with fixed conditions
        n_eval_episodes: Number of episodes to run per condition set
        deterministic: Whether to use deterministic action selection
    
    Returns:
        List of evaluation results
    """
    
    eval_results = []
    
    print(f"\nStarting evaluation with {len(eval_conditions)} condition sets...")
    
    for cond_idx, conditions in enumerate(eval_conditions):
        print(f"Evaluating condition set {cond_idx + 1}/{len(eval_conditions)}: "
              f"T={conditions['temperature']:.0f}K, P={conditions['pressure']:.1f}bar, Z={conditions['Z']:.2f}")
        
        set_results = []
        
        for ep_idx in range(n_eval_episodes):
     
            # Reset environment
            obs, info = env.reset(temperature=conditions['temperature'], pressure=conditions['pressure']*ct.one_atm, Z=conditions['Z'], total_time=conditions['total_time'])
            print(f"Reference Max Temperature: {np.max(env.ref_states[:, 0])} - horizon: {env.horizon} - total_time: {conditions['total_time']}")
            done = False
            
            # Episode data collection
            episode_data = {
                'conditions': conditions.copy(),
                'solver_names': env.get_solver_names()
            }
            
            total_reward = 0
            total_cpu_time = 0
            
            # Create progress bar for episode steps
            pbar = tqdm(total=env.horizon, desc=f"Episode {ep_idx+1}/{n_eval_episodes}")
            while not done:
                action = agent.act_deterministic(obs) if deterministic else agent.get_action(obs)[0]
                # Step environment
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
           
                # Collect data
                
                total_reward += reward
                total_cpu_time += next_info.get('cpu_time', 0)
                
                # Update progress bar
                temp = env.current_state[0]
                ref_temp = env.ref_states[(env.current_episode-1) * env.super_steps, 0]
                #print(f"Current episode: {env.current_episode} - horizon: {env.horizon} - ref_temp: {ref_temp} - len(env.ref_states): {len(env.ref_states)}")
                cpu_time = next_info.get('cpu_time', 0)
                pbar.set_postfix({
                    'T': f'{temp:.1f}K | {ref_temp:.1f}K/{env.ref_states[0, 0]:.1f}K',
                    'I': f'{env.current_pressure/ct.one_atm:.1f}bar, Z={env.current_Z:.2f}',
                    'A': action,
                    'C': f'{cpu_time:.3f}s',
                    'R': f'{reward:.1f}',
                    'Z': f'{env.current_Z:.2f}',
                    'E': f'{env.timestep_errors[-1]:.3f}',
                    'L': f'{(env.current_episode-1)*env.super_steps}/{len(env.ref_states)}'
                })
                pbar.update(1)
                obs = next_obs
            
            pbar.close()
            
            
            
            episode_data['cpu_times'] = env.cpu_times
            episode_data['actions'] = env.action_history
            episode_data['rewards'] = env.episode_rewards
            episode_data['timestep_errors'] = env.timestep_errors
            episode_data['trajectory'] = np.array(env.states_trajectory)
            episode_data['temperatures'] = np.array(env.states_trajectory)[:, 0]
            episode_data['times'] = env.times_trajectory
            # add reference states
            episode_data['reference_temperatures'] = env.ref_states[:, 0]
            episode_data['reference_times'] = env.ref_times
            episode_data['reference_species'] = env.ref_states[:, 1:]
            # Calculate action distribution
            action_counts = {}
            for a in episode_data['actions']:
                action_counts[a] = action_counts.get(a, 0) + 1
            action_distribution = {
                solver: action_counts.get(i, 0) / len(episode_data['actions']) 
                for i, solver in enumerate(episode_data['solver_names'])
            }
            
            
            # Episode summary
            episode_summary = {
                'episode': ep_idx + 1,
                'conditions': conditions.copy(), 
                'total_reward': total_reward,
                'total_cpu_time': total_cpu_time,
                'episode_length': len(episode_data['actions']),
                'ignition_detected': info.get('steady_state_info', {}).get('ignition_detected', False),
                'ignition_time': info.get('steady_state_info', {}).get('ignition_time', None),
                'reached_steady_state': info.get('steady_state_info', {}).get('reached_steady_state', False),
                'action_distribution': action_distribution,
                **episode_data
            }
            
            set_results.append(episode_summary)
        
        # Calculate set statistics
        set_rewards = [r['total_reward'] for r in set_results]
        set_cpu_times = [r['total_cpu_time'] for r in set_results]
        set_lengths = [r['episode_length'] for r in set_results]
        ignition_rates = [r['ignition_detected'] for r in set_results]
        average_action_distribution = {
            solver: np.mean([r['action_distribution'][solver] for r in set_results])
            for solver in set_results[0]['action_distribution'].keys()
        }
        set_summary = {
            'condition_set': cond_idx + 1,
            'conditions': conditions.copy(),
            'n_episodes': n_eval_episodes,
            'mean_reward': np.mean(set_rewards),
            'std_reward': np.std(set_rewards),
            'mean_cpu_time': np.mean(set_cpu_times),
            'std_cpu_time': np.std(set_cpu_times),
            'mean_length': np.mean(set_lengths),
            'average_action_distribution': average_action_distribution,
            'ignition_success_rate': np.mean(ignition_rates) * 100,
            'episodes': set_results
        }
        
        eval_results.append(set_summary)
        
        print(f"  Set {cond_idx + 1} Results:")
        print(f"    Mean Reward: {set_summary['mean_reward']:.3f} ± {set_summary['std_reward']:.3f}")
        print(f"    Mean CPU Time: {set_summary['mean_cpu_time']:.3f} ± {set_summary['std_cpu_time']:.3f}")
        print(f"    Ignition Rate: {set_summary['ignition_success_rate']:.1f}%")
        print(f"    Action distribution {set_summary['average_action_distribution']}")

    return eval_results

def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute returns and advantages using GAE"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value_t = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value_t = values[t + 1]
        
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

def load_policy(agent, model_path):
    """Load a policy from a file"""
    model = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.network.load_state_dict(model['model_state_dict'])
    agent.network.to(agent.device)
    return agent

def train_ppo(env, total_steps=1000000, rollout_length=2048, 
              save_freq=100, plot_freq=100, eval_freq=100, eval_conditions=None, 
              n_eval_episodes=5, neptune_run: Optional[object] = None, const_pressure=None, model_path=None, **ppo_kwargs):
    """Fixed PPO training loop with proper periodic operations"""
    
    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(obs_dim, action_dim, **ppo_kwargs)

    if model_path is not None:
        agent = load_policy(agent, model_path)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ppo_integrator_switching_{timestamp}"
    logger = DetailedLogger("logs", experiment_name, neptune_run=neptune_run)
    
    print("Starting PPO training...")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim} (solvers: {env.get_solver_names()})")
    print(f"Total steps: {total_steps}")
    
    total_steps_collected = 0
    update_count = 0
    rollout_buffer = []
    raw_obs_cache = []
    
    # Initialize first episode
    if const_pressure is not None:
        obs, info = env.reset(pressure=const_pressure*ct.one_atm)
    else:
        obs, info = env.reset()
    done = False
    # rollout_data = {'actions': [], 'rewards': []}
    raw_obs_cache.append(obs.copy()) 
    # Main training loop - step-based with rollouts containing multiple episodes
    while total_steps_collected < total_steps:
        steps_collected = 0
        rollout_buffer.clear()
        raw_obs_cache.clear()
        
        # Start new rollout
        if const_pressure is not None:
            obs, info = env.reset(pressure=const_pressure*ct.one_atm)
        else:
            obs, info = env.reset()
        done = False
        episode_data = {'actions': [], 'rewards': []}
        raw_obs_cache.append(obs.copy())
        
        # Progress bar for this rollout
        pbar = tqdm(total=rollout_length, desc=f"Rollout {update_count+1} | Steps {total_steps_collected}")
        
        rollout_rewards = []
        rollout_cpu_times = []
        # Collect rollout data (may span multiple episodes)
        while steps_collected < rollout_length and total_steps_collected < total_steps:
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store transition in rollout buffer
            rollout_buffer.append({
                'obs': obs.copy(),
                'action': action,
                'reward': reward,
                'log_prob': log_prob,
                'value': value,
                'done': done
            })
            raw_obs_cache.append(next_obs.copy())
            
            # Store episode-specific data
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            rollout_rewards.append(reward)
            rollout_cpu_times.append(next_info.get('cpu_time', 0))
            steps_collected += 1
            total_steps_collected += 1
            obs = next_obs
            
            # Update progress bar
            temp = env.current_state[0]
            cpu_time = next_info.get('cpu_time', 0)
            ref_temp = env.ref_states[(env.current_episode-1) * env.super_steps, 0]
            pbar.set_postfix({
                'T': f'{temp:.1f}K | {ref_temp:.1f}K/{env.ref_states[0, 0]:.1f}K', 
                'I': f'{env.current_pressure/ct.one_atm:.1f}bar | {env.current_Z:.2f}',
                'A': action, 
                'R': f'{reward:.1f}', 
                'C': f'{cpu_time:.3f}',
                'TT': f'{env.total_time:.3f}',
                'E': f'{env.timestep_errors[-1]:.3f}',
                'L': f'{(env.current_episode-1)*env.super_steps}/{len(env.ref_states)}'
            })
            pbar.update(1)
            
            # Handle episode completion within rollout
            if done:
                # Log completed episode
                logger.log_episode(total_steps_collected, episode_data, next_info)
                
                # Reset for next episode if we need more steps
                if steps_collected < rollout_length and total_steps_collected < total_steps:
                    if const_pressure is not None:
                        obs, info = env.reset(pressure=const_pressure*ct.one_atm)
                    else:
                        obs, info = env.reset()
                    done = False
                    episode_data = {'actions': [], 'rewards': []}
                    raw_obs_cache.append(obs.copy())
        
        pbar.close()
        # rollout_data['rewards'] = np.mean(np.array(rollout_rewards))
        # rollout_data['cpu_times'] = np.mean(np.array(rollout_cpu_times))
        
        # Update observation normalization stats
        agent.update_obs_rms(np.asarray(raw_obs_cache, dtype=np.float32))
        
        # PPO update when we have enough data
        if len(rollout_buffer) >= rollout_length:
            print(f"Performing PPO update after {steps_collected} steps, {total_steps_collected} total steps")
            
            # Prepare rollout data
            rollout_data = {
                'observations': np.array([t['obs'] for t in rollout_buffer]),
                'actions': np.array([t['action'] for t in rollout_buffer]),
                'rewards': np.array([t['reward'] for t in rollout_buffer]),
                'log_probs': np.array([t['log_prob'] for t in rollout_buffer]),
                'values': np.array([t['value'] for t in rollout_buffer]),
                'dones': np.array([t['done'] for t in rollout_buffer])
            }
            
            # Bootstrap value from last state
            if not rollout_data['dones'][-1]:
                next_value = agent.get_value(obs)
            else:
                next_value = 0
            
            # Compute GAE
            returns, advantages = compute_gae(
                rollout_data['rewards'], 
                rollout_data['values'],
                rollout_data['dones'],
                next_value,
                gamma=agent.gamma,
                gae_lambda=agent.gae_lambda
            )
            
            # Normalize advantages for stability
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()
            
            rollout_data['returns'] = returns
            rollout_data['advantages'] = advantages
            
            # Update agent
            update_info = agent.update(rollout_data, update_epochs=3, minibatch_size=256,
                           target_kl=0.02, clip_coef_vf=0.2, use_huber_vf=True)

            update_count += 1
            
            update_info['rewards'] = np.mean(np.array(rollout_rewards))
            update_info['cpu_times'] = np.mean(np.array(rollout_cpu_times))
            # Print update metrics
            print(f"Update {update_count}:")
            print(f"  Policy loss: {update_info['policy_loss']:.5f}")
            print(f"  Value loss: {update_info['value_loss']:.3f}")
            print(f"  Entropy: {update_info['entropy_loss']:.3f}")
            print(f"  KL divergence: {update_info['approx_kl']:.5f}")
            print(f"  Clip fraction: {update_info['clipfrac']:.5f}")
            print(f"  Explained variance: {update_info['explained_variance']:.3f}")
            print(f"  Action counts: {update_info['action_counts']}")
            print(f"  Returns - mean: {returns.mean():.3f}, std: {returns.std():.3f}")
            print(f"  Values - mean: {rollout_data['values'].mean():.3f}, std: {rollout_data['values'].std():.3f}")
            
            # Log update
            logger.log_update(update_count, update_info)
            
            # Periodic operations after PPO update
            # Print progress periodically
            if update_count % 10 == 0:
                logger.print_progress(update_count, total_steps)
            
            # Model saving
            if update_count % save_freq == 0:
                logger.save_data(update_count)
                model_path = os.path.join(logger.experiment_dir, 'models', f'model_update_{update_count}.pt')
                torch.save({
                    'update': update_count,
                    'total_steps': total_steps_collected,
                    'model_state_dict': agent.network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'obs_mean': agent.obs_rms.mean,
                    'obs_var': agent.obs_rms.var,
                    'obs_count': agent.obs_rms.count
                }, model_path)
                print(f"Model saved at update {update_count}")
                if neptune_run is not None:
                    try:
                        neptune_run[f"models/model_update_{update_count}"].upload(model_path)
                    except Exception as e:
                        print(f"Warning: Failed to upload model to Neptune: {e}")
            
            # Plot generation
            if update_count % plot_freq == 0:
                print(f"Generating plots at update {update_count}")
                logger.create_plots()
            
            # Evaluation
            if (eval_conditions and update_count % eval_freq == 0):
                print(f"Running evaluation at update {update_count}...")
                eval_results = evaluate_agent(agent, env, eval_conditions, n_eval_episodes, deterministic=True)
                logger.log_evaluation(update_count, eval_results)
                logger.create_evaluation_plots(update_count, eval_results)
            
            # Clear rollout buffer
            rollout_buffer = []
    
    # Final save and cleanup
    print("Training completed. Saving final results...")
    logger.save_data(total_steps_collected)
    logger.create_plots()
    
    final_model_path = os.path.join(logger.experiment_dir, 'models', 'final_model.pt')
    torch.save({
        'update': update_count,
        'total_steps': total_steps_collected,
        'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'obs_mean': agent.obs_rms.mean,
        'obs_var': agent.obs_rms.var,
        'obs_count': agent.obs_rms.count
    }, final_model_path)
    if neptune_run is not None:
        try:
            neptune_run["models/final_model"].upload(final_model_path)
        except Exception as e:
            print(f"Warning: Failed to upload final model to Neptune: {e}")
    
    return agent, logger


from reward_model import LagrangeReward1, ConstrainedReward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on combustion integrator switching")
    
    # Environment arguments
    parser.add_argument('--mechanism-file', type=str, required=True,
                       help='Path to mechanism file')
    parser.add_argument('--fuel', type=str, default='nc12h26',
                       help='Fuel specification')
    parser.add_argument('--oxidizer', type=str, default='O2:0.21, N2:0.79',
                       help='Oxidizer specification')
    
    # Training arguments
    parser.add_argument('--total-steps', type=int, default=10000000,
                       help='Total training steps')
    parser.add_argument('--rollout-length', type=int, default=2048,
                       help='Rollout length for PPO updates')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Model save frequency')
    parser.add_argument('--plot-freq', type=int, default=10,
                       help='Plot generation frequency')
    
    
    # Neptune logging
    parser.add_argument('--neptune', action='store_true',
                       help='Enable Neptune logging')
    parser.add_argument('--neptune-project', type=str, default=NEPTUNE_PROJECT,
                       help='Neptune project name, e.g. user/workspace')
    parser.add_argument('--neptune-tags', nargs='*', default=None,
                       help='Optional list of tags for the Neptune run')
    parser.add_argument('--neptune-name', type=str, default=None,
                       help='Optional display name for the Neptune run')
    
    # Evaluation arguments
    parser.add_argument('--eval-freq', type=int, default=10,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--n-eval-episodes', type=int, default=1,
                       help='Number of evaluation episodes per condition set')
    parser.add_argument('--eval-temp', nargs=5, type=float, default=[650, 800, 1050, 500, 1000],
                       help='Evaluation temperatures (K)')
    parser.add_argument('--eval-pressure', nargs=5, type=float, default=[60.0, 10.0, 1, 60.0, 40],
                       help='Evaluation pressures (bar)')
    parser.add_argument('--eval-Z', nargs=5, type=float, default=[0.2, 0.34, 0.3, 0.99, 1e-8],
                       help='Evaluation mixture fractions')
    parser.add_argument('--eval-time', nargs=5, type=float, default=[5e-2, 0.007, 3e-2, 4e-2, 1e-2],
                       help='Evaluation total times (s)')


    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model to load')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help='PPO clip coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.05,
                       help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    
    # parse the policy network architecture
    parser.add_argument('--policy-network-arch', type=str, default='[128, 128]',
                       help='Policy network architecture')
    
    # Environment configuration
    parser.add_argument('--temp-range', nargs=2, type=float, default=[300, 1300],
                       help='Temperature range for environment')
    parser.add_argument('--pressure-range', nargs=2, type=float, default=[1, 60],
                       help='Pressure range (bar)')
    parser.add_argument('--time-range', nargs=2, type=float, default=[1e-2, 5e-1],
                       help='Time range for simulations')
    parser.add_argument('--dt-range', nargs=2, type=float, default=[1e-6, 1e-6],
                       help='Timestep range')
    parser.add_argument('--etol', type=float, default=1e-4,
                       help='Error tolerance')
    parser.add_argument('--super-steps', type=int, default=10,
                       help='Number of super steps per episode')
    
    # Reward function parameters
    parser.add_argument('--epsilon', type=float, default=1e-4,
                       help='Error threshold for reward function')
    parser.add_argument('--lambda-init', type=float, default=1.0,
                       help='Lambda initial value')
    parser.add_argument('--lambda-lr', type=float, default=0.05,
                       help='Lambda learning rate')
    parser.add_argument('--target-violation', type=float, default=0.0,
                       help='Target violation')
    parser.add_argument('--cpu-log-delta', type=float, default=5e-4,
                       help='CPU log delta')
    parser.add_argument('--reward-clip', type=float, default=10.0,
                       help='Reward clip')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose environment output')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")

    # Parse policy network architecture into a list of ints
    try:
        arch_arg = args.policy_network_arch
        parsed_arch = None
        if isinstance(arch_arg, str):
            try:
                parsed_arch = json.loads(arch_arg)
            except json.JSONDecodeError:
                parsed_arch = ast.literal_eval(arch_arg)
        elif isinstance(arch_arg, (list, tuple)):
            parsed_arch = list(arch_arg)

        if not isinstance(parsed_arch, (list, tuple)):
            raise ValueError("policy-network-arch must be a list or tuple of integers")

        args.policy_network_arch = [int(x) for x in parsed_arch]
    except Exception as e:
        print(f"Warning: Failed to parse --policy-network-arch ('{args.policy_network_arch}'): {e}. Using default [64, 32, 16].")
        args.policy_network_arch = [256, 128, 64]
    
    # Initialize Neptune if requested
    neptune_run = None
    if args.neptune:
        if neptune_init_run is None:
            print("Error: Neptune is not installed but --neptune flag was provided. Install with: pip install neptune")
            exit(1)
        project = args.neptune_project or os.environ.get('NEPTUNE_PROJECT')
        api_token = NEPTUNE_API_TOKEN or os.environ.get('NEPTUNE_API_TOKEN', 'anonymous')
        if project is None:
            print("Error: --neptune-project not provided and NEPTUNE_PROJECT env var is not set.")
            exit(1)
        try:
            neptune_run = neptune_init_run(project=project, api_token=api_token, name=args.neptune_name, tags=args.neptune_tags,
                                           source_files=["**/*.py", "README.md"])
            neptune_run["config/device"] = device
            neptune_run["config/timestamp"] = time.time()
        except Exception as e:
            print(f"Error initializing Neptune run: {e}")
            exit(1)
    
    # Import and create environment
    try:
        from environment import IntegratorSwitchingEnv
    except ImportError:
        print("Error: Could not import IntegratorSwitchingEnv from environment.py")
        print("Make sure environment.py is in the same directory or in your Python path")
        exit(1)
    # Reward function configuration
    reward_cfg = dict(epsilon=args.epsilon, lambda_init=1.0, lambda_lr=0.05,
                      target_violation=0.0, cpu_log_delta=1e-3, reward_clip=10.0,
                      cpu_time_baseline=0.001, soft_margin_decades=0.15, switch_penalty=0.02,
                      ema_alpha=0.3, lambda_max=1e4)
    reward_function = ConstrainedReward(**reward_cfg)
    
    # Solver configurations (matching your environment setup)
    solver_configs = [
        {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},
        {'type': 'qss', 'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False, 
         'itermax': 2, 'epsmin': 0.002, 'epsmax': 100.0, 'abstol': 1e-8, 'mxsteps': 100000, 'name': 'QSS'},
    ]
    
    # Create environment
    print("Creating environment...")
    env = IntegratorSwitchingEnv(
        mechanism_file=args.mechanism_file,
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        temp_range=args.temp_range,
        pressure_range=args.pressure_range,
        time_range=args.time_range,
        dt_range=args.dt_range,
        etol=args.etol,
        super_steps=args.super_steps,
        reward_function=reward_function,
        solver_configs=solver_configs,
        verbose=args.verbose,
        terminated_by_steady_state=False,
        precompute_reference=True,
        track_trajectory=True,
        termination_count_threshold=1000
    )
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Solvers: {env.get_solver_names()}")
    
    # PPO hyperparameters
    ppo_kwargs = {
        'lr': args.lr,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_coef': args.clip_coef,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'device': device,
        'policy_network_arch': args.policy_network_arch
    }
    
    # log the ppo hyperparameters
    try:
        if neptune_run is not None:
            neptune_run["config/ppo_kwargs_json"] = json.dumps(ppo_kwargs)
            neptune_run["reward_config_json"] = json.dumps(reward_cfg)
            neptune_run["solver_configs_json"] = json.dumps(solver_configs)
    except Exception as _:
        # Best-effort logging; ignore Neptune type issues
        pass
    
    # Test environment
    print("\nTesting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")
    
    # Take a few random actions to verify everything works
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.4f}, terminated={terminated}")
        if terminated or truncated:
            obs, info = env.reset()
            break
    
    print("Environment test passed!")
    
    # Setup evaluation conditions
    eval_conditions = []
    for i in range(5):
        eval_conditions.append({
            'temperature': args.eval_temp[i],
            'pressure': args.eval_pressure[i],
            'Z': args.eval_Z[i],
            'total_time': args.eval_time[i]
        })
    
    print(f"\nEvaluation conditions:")
    for i, cond in enumerate(eval_conditions):
        print(f"  Set {i+1}: T={cond['temperature']:.0f}K, P={cond['pressure']:.1f}bar, Z={cond['Z']:.2f}, Total time={cond['total_time']:.2e}s")
    
    # Start training
    print(f"\nStarting PPO training with {args.total_steps} steps...")
    
    try:
        agent, logger = train_ppo(
            env=env,
            total_steps=args.total_steps,
            rollout_length=args.rollout_length,
            save_freq=args.save_freq,
            plot_freq=args.plot_freq,
            eval_freq=args.eval_freq,
            eval_conditions=eval_conditions,
            n_eval_episodes=args.n_eval_episodes,
            neptune_run=neptune_run,
            const_pressure=None,
            model_path=args.model_path,
            **ppo_kwargs
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved in: {logger.experiment_dir}")
        print(f"Final model saved as: {logger.experiment_dir}/models/final_model.pt")
        print(f"Training logs: {logger.experiment_dir}/episodes.csv")
        print(f"Plots: {logger.experiment_dir}/plots/")
        
        # Print final statistics
        if logger.episode_data:
            final_rewards = [ep['episode_reward'] for ep in logger.episode_data[-100:]]
            print(f"\nFinal Performance (last 100 episodes):")
            print(f"  Mean reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
            print(f"  Best reward: {np.max(final_rewards):.3f}")
            
        if logger.combustion_metrics:
            recent_combustion = logger.combustion_metrics[-100:]
            ignition_rate = np.mean([cm['ignition_detected'] for cm in recent_combustion]) * 100
            print(f"  Ignition success rate: {ignition_rate:.1f}%")
            
        # Solver usage summary
        if logger.solver_usage_count:
            total_actions = sum(logger.solver_usage_count.values())
            print(f"\nFinal Solver Usage:")
            for solver, count in sorted(logger.solver_usage_count.items()):
                percentage = count / total_actions * 100
                solver_name = env.get_solver_names()[solver]
                print(f"  {solver_name}: {percentage:.1f}% ({count:,} actions)")
        
        # Evaluation summary
        if logger.evaluation_data:
            print(f"\nEvaluation Summary:")
            for eval_data in logger.evaluation_data:
                print(f"  Episode {eval_data['episode']}: "
                      f"Reward={eval_data['mean_reward']:.3f}±{eval_data['std_reward']:.3f}, "
                      f"CPU Time={eval_data['mean_cpu_time']:.3f}±{eval_data['std_cpu_time']:.3f}, "
                      f"Ignition Rate={eval_data['ignition_success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        if 'logger' in locals():
            print("Saving current progress...")
            logger.save_data(logger.episode_data[-1]['episode'] if logger.episode_data else 0)
            logger.create_plots()
            print(f"Progress saved in: {logger.experiment_dir}")
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if 'logger' in locals():
            print("Attempting to save partial results...")
            try:
                logger.save_data(logger.episode_data[-1]['episode'] if logger.episode_data else 0)
                print(f"Partial results saved in: {logger.experiment_dir}")
            except:
                print("Could not save partial results")
    
    finally:
        print("\nCleaning up...")
        if 'env' in locals():
            env.close() if hasattr(env, 'close') else None
        if 'neptune_run' in locals() and neptune_run is not None:
            try:
                neptune_run.stop()
            except Exception:
                pass
