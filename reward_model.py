import numpy as np

class LagrangeReward1:
    """
    Constrained reward: minimize CPU with error <= epsilon.
    The dual variable lambda_ is adapted online (dual ascent).
    """
    def __init__(self, epsilon=1e-3, 
                 lambda_init=1.0, 
                 lambda_lr=0.05, 
                 target_violation=0.0,
                 cpu_log_delta=1e-3, 
                 reward_clip=50.0, **kwargs):
        self.epsilon = float(epsilon)
        self.lambda_ = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.target_violation = float(target_violation)
        self.cpu_log_delta = float(cpu_log_delta)
        self.reward_clip = float(reward_clip)
        self._violations = []

    def reset_episode(self):
        self._violations = []

    def step_reward(self, cpu_time, err, action=None, reached_steady_state=False):
        # CPU term: larger when faster; use log for smoothness and scale
	
        log_cpu_term = -np.log10(cpu_time + 1e-8)
        
        #print(f"Action {action} - cpu time {cpu_time} - log {log_cpu_term} - Lambda: {self.lambda_}")

        # Violation: zero if below epsilon; linear above
        violation = max(err / (self.epsilon + 1e-12) - 1.0, 0.0)
        violation = np.clip(violation, 0, 10)
        #print(f"Violation: {violation} - Error: {err} - Epsilon: {self.epsilon}")
        self._violations.append(violation)


        r = log_cpu_term - 1 * violation
        
        # if violation is 0 and action is 1 and reached_steady_state is True, add a bonus
        if violation == 0 and action == 1 and reached_steady_state:
            r_bonus = r + 5.0
            #print(f"Good action: {action} and reached_steady_state: {reached_steady_state} - reward: {r} - reward_bonus: {r_bonus} - error: {err:4f} - cpu_time: {cpu_time:4f}")
        elif violation == 0 and reached_steady_state and action == 0:
            r_bonus = -r
        else:
            r_bonus = 0.0
        
        if reached_steady_state and violation != 0:
            r -= 5.0
        
        r += r_bonus
        
        # if violation == 0 and action == 0:
        #     reward -= 0.5
        
        reward = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return reward

    def end_episode_update_lambda(self):
        if not self._violations:
            return
        avg_violation = float(np.mean(self._violations))
        # Dual ascent with nonnegativity
        self.lambda_ = max(0.0, self.lambda_ + self.lambda_lr * (avg_violation - self.target_violation))
        self._violations = []

    def get_aux(self):
        return {"lambda": self.lambda_, "epsilon": self.epsilon}

import numpy as np

class ConstrainedReward:
    """
    Speed subject to accuracy constraint err <= epsilon.
    - CPU term normalized by a baseline (e.g., BDF on this interval).
    - Smooth violation shaping around epsilon.
    - Optional switch penalty to discourage chattering.
    - EMA smoothing of violation to stabilize lambda updates.
    """
    def __init__(self,
                 epsilon=1e-3,
                 lambda_init=1.0,
                 lambda_lr=0.02,
                 target_violation=0.0,
                 cpu_time_baseline=None,   # if None, will adapt online
                 cpu_log_delta=1e-8,
                 soft_margin_decades=0.15, # +/- in log10 space around epsilon
                 switch_penalty=0.02,
                 ema_alpha=0.3,            # smoothing for violation
                 lambda_max=1e4,
                 reward_clip=50.0, **kwargs):
        self.epsilon = float(epsilon)
        self.lambda_ = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.target_violation = float(target_violation)
        self.cpu_time_baseline = cpu_time_baseline
        self.cpu_log_delta = float(cpu_log_delta)
        self.soft_margin_decades = float(soft_margin_decades)
        self.switch_penalty = float(switch_penalty)
        self.ema_alpha = float(ema_alpha)
        self.lambda_max = float(lambda_max)
        self.reward_clip = float(reward_clip)

        self._violations = []
        self._ema_violation = 0.0
        self._prev_action = None

    def reset_episode(self):
        self._violations = []
        self._ema_violation = 0.0
        self._prev_action = None

    def _safe_log10_ratio(self, num, den):
        num = float(num) if np.isfinite(num) and num > 0 else self.cpu_log_delta
        den = float(den) if (den is not None and np.isfinite(den) and den > 0) else 1.0
        return -np.log10((num + self.cpu_log_delta) / (den + self.cpu_log_delta))

    def _soft_violation(self, err):
        # work in log space for smoothness around epsilon
        # z = (log10(err) - log10(epsilon)) / soft_margin_decades
        # sigmoid(z) ~ 0 when well below eps, ~1 when well above eps
        err = float(err) if np.isfinite(err) and err > 0 else 0.0
        if err <= 0:
            return 0.0, 0.0  # no violation, comfortably safe

        log_err = np.log10(max(err, 1e-300))
        log_eps = np.log10(self.epsilon + 1e-300)
        z = (log_err - log_eps) / (self.soft_margin_decades + 1e-12)

        # smooth violation in [0, 1] via sigmoid; hinge-like but differentiable
        sig = 1.0 / (1.0 + np.exp(-z))
        # define violation as how much above epsilon we are (0 below, rises smoothly above)
        violation = max(sig - 0.5, 0.0) * 2.0  # maps to [0,1] with mid at epsilon

        # optional "comfort bonus" when comfortably under epsilon
        # distance below epsilon (clipped to margin)
        comfort = max(-z, 0.0)
        comfort = min(comfort, 1.0)  # within ~one margin decade

        return violation, comfort

    def step_reward(self, cpu_time, err, action=None, reached_steady_state=False,
                    dT_dt=None, stiffness_indicator=None):
        # Baseline CPU time: if not provided, lazily adapt (slow EMA) to the min seen so far
        if self.cpu_time_baseline is None:
            # initialize to current cpu_time to avoid huge early ratios
            self.cpu_time_baseline = max(cpu_time, 1e-8)
        else:
            # adapt baseline slowly toward the lower envelope (be conservative)
            self.cpu_time_baseline = 0.99*self.cpu_time_baseline + 0.01*max(cpu_time, 1e-8)

        # CPU term normalized by baseline
        cpu_term = self._safe_log10_ratio(cpu_time, self.cpu_time_baseline)

        # Smooth violation & comfort
        violation, comfort = self._soft_violation(err)
        # EMA violation for stability in the dual term
        self._ema_violation = (1.0 - self.ema_alpha)*self._ema_violation + self.ema_alpha*violation
        self._violations.append(self._ema_violation)

        # Base Lagrangian reward
        r = cpu_term - self.lambda_ * self._ema_violation

        # Encourage using the cheaper solver when comfortably safe
        # (comfort ~ how many soft-margin units below epsilon we are)
        if action == 1:
            r += 0.5 * comfort
        # else:
        #     r -= 0.5 * comfort
        # r += 0.25 * comfort  # small positive; keeps scale modest

        # Action chattering penalty (discourage rapid switching)
        if (self._prev_action is not None) and (action is not None) and (action != self._prev_action):
            r -= self.switch_penalty
        self._prev_action = action

        # Optional stiffness-aware shaping (proxy around ignition)
        # If user supplies dT/dt or stiffness flag, lightly encourage the "safer" choice
        if stiffness_indicator is not None:
            # Expect stiffness_indicator in [0,1], where 1=very stiff
            r += 0.15 * (stiffness_indicator if action == 0 else -stiffness_indicator)  # e.g., action 0=BDF safer
        elif dT_dt is not None and np.isfinite(dT_dt):
            # normalize by a reference slope to get ~[0,1]
            dt_ref = 1e6  # pick a ref scaling or pass one in
            stiff = np.tanh(abs(dT_dt) / dt_ref)
            r += 0.15 * (stiff if action == 0 else -stiff)

        # Light shaping at steady state: prefer cheaper solver if constraint satisfied
        if reached_steady_state and violation == 0.0:
            if action == 1:
                r += 0.5
            else:
                r -= 0.5

        # Clip for PPO stability
        reward = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return reward

    def end_episode_update_lambda(self):
        if not self._violations:
            return
        avg_v = float(np.mean(self._violations))
        self.lambda_ = float(np.clip(self.lambda_ + self.lambda_lr * (avg_v - self.target_violation),
                                     0.0, self.lambda_max))
        self._violations = []

    def get_aux(self):
        return {"lambda": self.lambda_, "epsilon": self.epsilon, "cpu_time_baseline": self.cpu_time_baseline}
