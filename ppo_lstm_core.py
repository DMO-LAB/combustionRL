# ppo_lstm_core.py
# Core PPO(LSTM) pieces: RunningMeanStd, RolloutBufferRecurrent, PolicyLSTM, PPO

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Running mean/var for obs norm ---------------------------

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray):
        # x: [N, *shape]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# --------------------------- Rollout buffer (recurrent) ---------------------------

@dataclass
class BufferConfig:
    obs_dim: int
    size: int
    gamma: float = 0.99
    lam: float = 0.95
    device: str = "cpu"

class RolloutBufferRecurrent:
    """
    Stores a single-environment rollout of fixed 'size' steps.
    Also stores the LSTM hidden state (h, c) *before* each step.
    """
    def __init__(self, cfg: BufferConfig):
        self.cfg = cfg
        S = cfg.size
        D = cfg.obs_dim
        self.obs      = np.zeros((S, D), dtype=np.float32)
        self.actions  = np.zeros((S,), dtype=np.int64)
        self.rewards  = np.zeros((S,), dtype=np.float32)
        self.dones    = np.zeros((S,), dtype=np.bool_)
        self.values   = np.zeros((S,), dtype=np.float32)
        self.logprobs = np.zeros((S,), dtype=np.float32)
        self.hxs      = np.zeros((S, cfg.hidden_size), dtype=np.float32)
        self.cxs      = np.zeros((S, cfg.hidden_size), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def reset(self):
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, logprob, hx, cx):
        i = self.ptr
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.dones[i]    = done
        self.values[i]   = value
        self.logprobs[i] = logprob
        self.hxs[i]      = hx
        self.cxs[i]      = cx
        self.ptr += 1
        if self.ptr >= self.cfg.size:
            self.full = True

    def compute_gae(self, last_value):
        S = self.ptr
        adv = np.zeros(S, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(S)):
            nonterminal = 1.0 - float(self.dones[t])
            next_value = last_value if t == S-1 else self.values[t+1]
            delta = self.rewards[t] + self.cfg.gamma * next_value * nonterminal - self.values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + self.values[:S]
        # Normalize advantages (per rollout)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # For recurrent PPO we form mini-batches of sequences with stored initial hidden states
    def iter_minibatches(self, seq_len, batch_seqs, adv, ret, shuffle=True):
        S = self.ptr
        # valid sequence starts: [0, S-seq_len]
        starts = np.arange(0, S, seq_len)
        # trim last partial chunk
        if starts[-1] + seq_len > S:
            starts = starts[:-1]
        if shuffle:
            np.random.shuffle(starts)
        # group starts into minibatches
        for i in range(0, len(starts), batch_seqs):
            batch_starts = starts[i:i+batch_seqs]
            # stack sequences for this batch
            obs   = np.stack([self.obs[s:s+seq_len]   for s in batch_starts], axis=0)        # [B, T, D]
            acts  = np.stack([self.actions[s:s+seq_len] for s in batch_starts], axis=0)      # [B, T]
            oldlp = np.stack([self.logprobs[s:s+seq_len] for s in batch_starts], axis=0)     # [B, T]
            vals  = np.stack([self.values[s:s+seq_len] for s in batch_starts], axis=0)       # [B, T]
            advb  = np.stack([adv[s:s+seq_len]         for s in batch_starts], axis=0)       # [B, T]
            retb  = np.stack([ret[s:s+seq_len]         for s in batch_starts], axis=0)       # [B, T]
            h0    = np.stack([self.hxs[s]              for s in batch_starts], axis=0)       # [B, H]
            c0    = np.stack([self.cxs[s]              for s in batch_starts], axis=0)       # [B, H]
            yield obs, acts, oldlp, vals, advb, retb, h0, c0


# --------------------------- Policy (feature MLP -> LSTM -> heads) ---------------------------

def ortho_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class PolicyLSTM(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, action_dim=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.apply(lambda m: ortho_init(m, gain=math.sqrt(2)))
        ortho_init(self.actor, gain=0.01)
        ortho_init(self.critic, gain=1.0)

    def forward(self, obs_seq, h0=None, c0=None):
        """
        obs_seq: [B, T, D] or [T, D] if B=1
        h0,c0:  [B, H]
        returns: logits [B, T, A], values [B, T], (hT, cT)
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        B, T, D = obs_seq.shape
        x = self.feature(obs_seq)               # [B, T, H]
        if h0 is None or c0 is None:
            out, (hT, cT) = self.rnn(x)        # hT,cT: [1,B,H]
        else:
            h0 = h0.unsqueeze(0)               # [1,B,H]
            c0 = c0.unsqueeze(0)
            out, (hT, cT) = self.rnn(x, (h0, c0))
        logits = self.actor(out)                # [B, T, A]
        values = self.critic(out).squeeze(-1)   # [B, T]
        return logits, values, (hT.squeeze(0), cT.squeeze(0))

    @torch.no_grad()
    def step(self, obs, h, c):
        """
        Single-step inference (B=1,T=1).
        obs: [D], h,c: [H]
        returns: action, logprob, value, probs, (h_new,c_new)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # [1,1,D]
        logits, values, (hT, cT) = self.forward(obs_t, h.unsqueeze(0), c.unsqueeze(0))
        logits = logits.squeeze(0).squeeze(0)
        value = values.squeeze(0).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item(), probs, (hT.squeeze(0), cT.squeeze(0))


# --------------------------- PPO update ---------------------------

@dataclass
class PPOConfig:
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    seq_len: int = 64
    batch_seqs: int = 16
    target_kl: float = 0.03
    device: str = "cpu"

class PPO:
    def __init__(self, policy: PolicyLSTM, cfg: PPOConfig):
        self.policy = policy.to(cfg.device)
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def update(self, buffer: RolloutBufferRecurrent, adv, ret):
        policy, cfg = self.policy, self.cfg
        device = cfg.device
        log_info = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "count": 0}


        policy.train()
        for _ in range(cfg.epochs):
            # iterate recurrent mini-batches
            for obs, acts, oldlp, oldv, advb, retb, h0, c0 in buffer.iter_minibatches(
                cfg.seq_len, cfg.batch_seqs, adv, ret, shuffle=True
            ):
                obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device)     # [B,T,D]
                acts_t = torch.as_tensor(acts, dtype=torch.long, device=device)       # [B,T]
                oldlp_t= torch.as_tensor(oldlp, dtype=torch.float32, device=device)   # [B,T]
                oldv_t = torch.as_tensor(oldv, dtype=torch.float32, device=device)     # [B,T]
                ret_t  = torch.as_tensor(retb, dtype=torch.float32, device=device)    # [B,T]
                adv_t  = torch.as_tensor(advb, dtype=torch.float32, device=device)    # [B,T]
                h0_t   = torch.as_tensor(h0, dtype=torch.float32, device=device)      # [B,H]
                c0_t   = torch.as_tensor(c0, dtype=torch.float32, device=device)      # [B,H]

                logits, values, _ = policy(obs_t, h0_t, c0_t)                          # [B,T,A],[B,T]
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                newlp = dist.log_prob(acts_t)                                          # [B,T]
                entropy = dist.entropy().mean()

                ratio = torch.exp(newlp - oldlp_t)                                     # [B,T]
                # clipped policy loss
                unclipped = ratio * adv_t
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_t
                loss_pi = -torch.mean(torch.min(unclipped, clipped))
                # value loss clipping
                v_clipped = oldv_t + torch.clamp(values - oldv_t, -cfg.clip_coef, cfg.clip_coef)
                loss_v = 0.5 * torch.mean(torch.max((values - ret_t)**2, (v_clipped - ret_t)**2))
                # entropy bonus
                loss_ent = entropy

                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * loss_ent

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # stats
                approx_kl = torch.mean(oldlp_t - newlp).detach().cpu().item()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_coef).float()).cpu().item()
                log_info["loss_pi"] += loss_pi.item()
                log_info["loss_v"]  += loss_v.item()
                log_info["entropy"] += loss_ent.item()
                log_info["approx_kl"] += approx_kl
                log_info["clip_frac"] += clip_frac
                log_info["count"] += 1

                if cfg.target_kl and approx_kl > cfg.target_kl:
                    break

        # average logs
        c = max(1, log_info["count"])
        for k in list(log_info.keys()):
            if k != "count":
                log_info[k] /= c
        return log_info
