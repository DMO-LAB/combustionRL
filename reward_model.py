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
                 reward_clip=50.0):
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
        #print(f"Violation: {violation} - Error: {err} - Epsilon: {self.epsilon}")
        self._violations.append(violation)


        r = log_cpu_term - self.lambda_ * violation
        
        # if violation is 0 and action is 1 and reached_steady_state is True, add a bonus
        if violation == 0 and action == 1 and reached_steady_state:
            r_bonus = r + 5.0
            #print(f"Good action: {action} and reached_steady_state: {reached_steady_state} - reward: {r} - reward_bonus: {r_bonus} - error: {err:4f} - cpu_time: {cpu_time:4f}")
        elif violation == 0 and reached_steady_state and action == 0:
            r_bonus = -2*r
        else:
            r_bonus = 0.0
            
        r += r_bonus
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
