import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cantera as ct
import time
from utils import create_solver
from collections import deque
from reward_model import LagrangeReward1
from collections import deque

class IntegratorSwitchingEnv(gym.Env):
    """RL Environment for combustion integrator switching with steady state detection"""
    
    def __init__(self, mechanism_file='gri30.yaml', 
                 fuel='CH4:1.0', oxidizer='N2:3.76, O2:1.0',
                 temp_range=(1000, 1400), phi_range=(0.5, 2.0),
                 pressure_range=(1, 60), time_range=(1e-3, 1e-2),
                 dt_range=(1e-6, 1e-4),
                 dt=1e-6, etol=1e-3, super_steps=50,
                 fixed_n_episodes=100,
                 reward_function=None,
                 ignition_temp_threshold=1600,  # Temperature threshold for ignition detection
                 steady_temp_tolerance=1.0,     # Temperature change tolerance for steady state
                 steady_time_factor=0.2,
                 verbose=False,
                 solver_configs=None,
                 terminated_by_steady_state=False,
                 termination_count_threshold=50):       # Time factor for steady state check
        
        super().__init__()

        
        # Simulation parameters
        self.mechanism_file = mechanism_file
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.temp_range = temp_range
        self.phi_range = phi_range
        self.pressure_range = pressure_range
        self.time_range = time_range
        self.dt = dt
        self.dt_range = dt_range
        # self.super_steps = super_steps
        self.etol = etol
        self.verbose = verbose
        self.fixed_n_episodes = fixed_n_episodes
        # Steady state detection parameters
        self.ignition_temp_threshold = ignition_temp_threshold
        self.steady_temp_tolerance = steady_temp_tolerance
        self.steady_time_factor = steady_time_factor
        self.terminated_by_steady_state = terminated_by_steady_state
        self.termination_count_threshold = termination_count_threshold
        # Setup gas object
        self.gas = ct.Solution(mechanism_file)
        
        if reward_function is None:
            reward_function = LagrangeReward1()
        
        self.reward_function = reward_function
        
        # Define solver configurations
        self.solver_configs = solver_configs if solver_configs is not None else [
            # CVODE BDF with different tolerances
            {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-8, 'mxsteps': 1000, 'name': 'CVODE_BDF_loose'},
            
            # QSS with different tolerances
            {'type': 'qss', 'rtol': 1e-8, 'atol': 1e-20, 'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False, 'itermax': 1, 'epsmin': 0.02, 'epsmax': 10.0, 'abstol': 1e-12, 'mxsteps': 1000, 'name': 'QSS_tight'},
        ]
        # Action space: choose which solver to use
        self.action_space = spaces.Discrete(len(self.solver_configs))
        
        self.key_species = ['O','H','OH','H2O','O2','H2','H2O2','N2']  # fix typo & duplicates
        obs_size = 2 + len(self.key_species) # temperature + species + pressure + phi
        self.observation_space = spaces.Box(low=-25., high=10., shape=(obs_size,), dtype=np.float32)

        self.representative_species = ['ch4', 'o2', 'h2o', 'co2'] if mechanism_file == 'gri30.yaml' else ['nc12h26', 'o2', 'h2o', 'co2']
        self.representative_species_indices = [self.gas.species_index(spec) for spec in self.representative_species]
        # Initialize solver instances (will be created in reset)
        self.solvers = []
        
        self.previous_states = deque(maxlen=2)
        
        # Error tracking
        self.timestep_errors = []
        
        # Steady state tracking
        self.ignition_detected = False
        self.ignition_time = None
        self.previous_temperature = None
        self.reached_steady_state = False

    def _create_solver_instances(self):
        """Create all solver instances for current conditions"""
        self.solvers = []
        for config in self.solver_configs:
            try:
                solver = create_solver(
                    config['type'], 
                    config, 
                    self.gas, 
                    self.initial_state, 
                    0.0, 
                    self.current_pressure,
                    mxsteps=config['mxsteps']
                )
                self.solvers.append(solver)
            except Exception as e:
                print(f"Failed to create solver {config['name']}: {e}")
                self.solvers.append(None)
    
    def _setup_episode(self, use_initial_state=False, **kwargs):
        """Setup initial conditions for episode"""
        # Randomize or use provided parameters
        if use_initial_state:
            state = kwargs.get('state', None)
            if state is None:
                raise ValueError("state must be provided if use_initial_state is True")
            self.current_temp = state[-2]
            self.current_pressure = state[-1]
            self.current_species = state[:-2]
            print(f"Using initial state: {self.current_temp}, {self.current_pressure}")
        else:
            pressure_range = np.arange(1, 11, 1)
            
            self.current_temp = kwargs.get('temperature', 
                                        np.random.uniform(*self.temp_range))
            self.current_phi = kwargs.get('phi', 
                                        np.random.uniform(*self.phi_range))
            self.current_pressure = kwargs.get('pressure', 
                                            np.random.choice(pressure_range) * ct.one_atm)
        
        self.total_time = kwargs.get('total_time', 
                                   np.random.uniform(*self.time_range))
        
        self.dt = kwargs.get('dt', 
                             np.random.uniform(*self.dt_range))
        
        self.etol = kwargs.get('etol', 
                             self.etol)
        
        total_episodes = int(self.total_time / self.dt)
        self.super_steps = int(total_episodes / self.fixed_n_episodes)
        self.reward_function.cpu_log_delta = 1e-3 * (self.super_steps/50)
        if self.verbose:
            print(f"Total episodes: {total_episodes}, Super steps: {self.super_steps}")
        
        if self.verbose:
            print(f"Env Reset with temperature: {self.current_temp},  pressure: {self.current_pressure}, phi: {self.current_phi}, Total time: {self.total_time}, Dt: {self.dt}, Etol: {self.etol}")
        
        # Setup gas conditions
        self.gas = ct.Solution(self.mechanism_file)
        self.ref_gas = ct.Solution(self.mechanism_file)
        if use_initial_state:
            self.gas.TPY = self.current_temp, self.current_pressure, self.current_species
            self.ref_gas.TPY = self.current_temp, self.current_pressure, self.current_species
        else:
            self.gas.set_equivalence_ratio(self.current_phi, self.fuel, self.oxidizer)
            self.gas.TP = self.current_temp, self.current_pressure
            self.ref_gas.set_equivalence_ratio(self.current_phi, self.fuel, self.oxidizer)
            self.ref_gas.TP = self.current_temp, self.current_pressure
        
        self.current_phi = self.gas.equivalence_ratio(self.fuel, self.oxidizer)
        
        # Create initial state
        
        self.initial_state = np.hstack([self.current_temp, self.gas.Y])
        self.current_state = self.initial_state.copy()
        
        self.n_episodes = int(self.total_time / (self.dt * self.super_steps))
        # Create reference trajectory
        self._compute_reference_trajectory()
        
        # if max temperature is less than 1000K, adjust end time to 1/10 of the original end time
        if np.max(self.ref_states[:, 0]) < 600:
            self.total_time = self.total_time/10
            total_episodes = int(self.total_time / self.dt)
            self.super_steps = int(total_episodes / self.fixed_n_episodes)
            self.reward_function.cpu_log_delta = 1e-3 * (self.super_steps/50)
            if self.verbose:
                print(f"Adjusted total time to {self.total_time} because max temperature is less than 1000K")
        
        self.n_episodes = int(self.total_time / (self.dt * self.super_steps))
        # Create solver instances
        self._create_solver_instances()
    
    def _compute_reference_trajectory(self):
        """Compute high-accuracy reference trajectory"""
        # Create reference gas object
        ref_gas = self.ref_gas
        
        # High-accuracy integration
        reactor = ct.IdealGasConstPressureReactor(ref_gas)
        sim = ct.ReactorNet([reactor])
        sim.rtol = 1e-10
        sim.atol = 1e-20
        
        self.ref_states = []
        self.ref_times = []
        
        current_time = 0.0
        self.ref_states.append(np.hstack([reactor.T, reactor.thermo.Y]))
        self.ref_times.append(current_time)
        
        # Store states at each super-step
        for episode in range(self.n_episodes):
            for step in range(self.super_steps):
                current_time += self.dt
                sim.advance(current_time)
                self.ref_states.append(np.hstack([reactor.T, reactor.thermo.Y]))
                self.ref_times.append(current_time)
        
        self.ref_states = np.array(self.ref_states)
        self.ref_times = np.array(self.ref_times)
        if self.verbose:
            print(f"Max reference temperature: {np.max(self.ref_states[:, 0])}")
    
    def _get_combustion_indicators(self, temp):
        """Compute physics-based indicators for solver switching"""
        # Actual temperature gradient (dT/dt)
        if hasattr(self, 'previous_states') and len(self.previous_states) >= 1:
            temp_gradient = abs(temp - self.previous_states[-1][0])
            temp_gradient_norm = temp_gradient / (temp - self.initial_state[0])
        else:
            temp_gradient = 0.0
            temp_gradient_norm = 0.0
        
        # Temperature acceleration (d²T/dt²) - better indicator of ignition onset
        if hasattr(self, 'previous_states') and len(self.previous_states) >= 2:
            temp_accel = abs(((temp - self.previous_states[-1][0]) - 
                        (self.previous_states[-1][0] - self.previous_states[-2][0])))
            temp_accel_norm = temp_accel / (temp - self.initial_state[0])
        else:
            temp_accel = 0.0
            temp_accel_norm = 0.0
        
        # Temperature rise ratio (what the original code was actually doing)
        temp_rise_ratio = (temp - self.current_temp) / max(self.current_temp, 300)
        ignition_proximity = np.tanh(temp_rise_ratio * 5.0)
        
        # print(f"Temp gradient norm: {temp_gradient_norm}, Temp accel norm: {temp_accel_norm}, Ignition proximity: {ignition_proximity}")
        # print(f"Temp gradient: {temp_gradient}, Temp accel: {temp_accel}, Temp rise ratio: {temp_rise_ratio}")
        return [temp_gradient_norm, temp_accel_norm, ignition_proximity]
    
    def _get_observation(self, state):
        """Convert state to observation including error flags"""
        temp = state[0]
        species = state[1:]
        
        # Get key species indices
        key_species_values = []
        for spec in self.key_species:
            try:
                idx = self.gas.species_index(spec)
                key_species_values.append(species[idx])
            except ValueError:
                # Species not found in mechanism
                key_species_values.append(0.0)
        
        # Normalize and transform
        temp_norm = (temp - 300) / 2000.0  # Normalize temperature
        species_norm = np.log10(np.maximum(key_species_values, 1e-20))  # Log transform
        pressure_norm =   np.log10(self.current_pressure / ct.one_atm)
  
        obs = np.hstack([temp_norm, species_norm, pressure_norm]).astype(np.float32)
        return obs
    
    def reset(self, seed=None, options=None, use_initial_state=False, **kwargs):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Setup episode conditions
        self._setup_episode(use_initial_state=use_initial_state, **kwargs)
        
        # Reset episode variables
        self.current_episode = 0
        self.episode_start_time = 0.0
        self.action_history = []
        self.cpu_times = []
        self.episode_rewards = []
    
        self.previous_states.clear()
        # Reset error tracking
        self.cumulative_error = 0.0
        self.timestep_errors = []
        self.cumulative_errors = []
        self.states_trajectory = []
        self.current_time = 0.0
        self.times_trajectory = [self.current_time]
        self.states_trajectory.append(self.current_state.copy())

        # Reset steady state tracking
        self.ignition_detected = False
        self.ignition_time = None
        self.previous_temperature = self.current_state[0]
        self.reached_steady_state = False
        self.count_since_steady_state = 0
        
        self.gibbs_free_energy = [self.gas.gibbs_mass]
        
        # Reset reward function for new episode
        self.reward_function.reset_episode()
        
        obs = self._get_observation(self.current_state)
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        """Execute one super-step with chosen solver"""
        # Check if already reached steady state or max episodes
        if self.terminated_by_steady_state and (self.reached_steady_state or self.current_episode >= self.n_episodes):
            terminated = True
            obs = self._get_observation(self.current_state)
            info = self._get_info()
            info.update({
                'termination_reason': 'steady_state' if self.reached_steady_state else 'max_episodes',
                'reached_steady_state': self.reached_steady_state,
                'ignition_detected': self.ignition_detected,
                'ignition_time': self.ignition_time,
            })
            return obs, 0.0, terminated, False, info
    
        # Validate action
        if action >= len(self.solvers) or self.solvers[action] is None:
            # Invalid solver, give penalty
            reward = -5.0
            success = False
            cpu_time = 0.0
            timestep_error = float('inf')
        else:
            # Execute integration with chosen solver
            reward, success, cpu_time, timestep_error = self._integrate_super_step(action)
        
        # Update episode state
        self.current_episode += 1
        self.action_history.append(action)
        self.cpu_times.append(cpu_time)
        self.episode_rewards.append(reward)
        
        self._check_steady_state(self.current_state[0])
        # # Check for steady state after integration
        if self.terminated_by_steady_state or self.count_since_steady_state >= self.termination_count_threshold:
            print(f"Terminated by steady state or count since steady state >= {self.count_since_steady_state}")
            terminated = self.reached_steady_state or self.current_episode >= self.n_episodes
        else:
            terminated = self.current_episode >= self.n_episodes
        if terminated:
            if hasattr(self.reward_function, 'end_episode_update_lambda'):
                self.reward_function.end_episode_update_lambda()
     
        
        obs = self._get_observation(self.current_state)
        self.previous_states.append(self.current_state.copy())
        info = self._get_info()
        info.update({
            'action': action,
            'solver_name': self.solver_configs[action]['name'] if action < len(self.solver_configs) else 'invalid',
            'success': success,
            'cpu_time': cpu_time,
            'timestep_error': timestep_error,
            'reached_steady_state': self.reached_steady_state,
            'ignition_detected': self.ignition_detected,
            'ignition_time': self.ignition_time,
            'termination_reason': 'steady_state' if self.reached_steady_state else ('max_episodes' if terminated else 'ongoing'),
            'gibbs_free_energy': self.gibbs_free_energy[-1]
        })
        
        return obs, reward, terminated, False, info
    
    def _check_steady_state(self, current_temp):
        """Check for steady state conditions based on temperature"""
        # Check for ignition (temperature exceeds threshold)
        if current_temp > self.ignition_temp_threshold and not self.ignition_detected:
            self.ignition_detected = True
            self.ignition_time = self.current_time
            if self.verbose:
                print(f"Ignition detected at T={current_temp:.1f}K, t={self.current_time:.6f}s")
        
        # Check for steady state if ignition has occurred
        if (self.ignition_detected and self.ignition_time is not None and 
            self.previous_temperature is not None):
            
            # Temperature change is small
            temp_change = abs(current_temp - self.previous_temperature)
            temp_stable = temp_change < self.steady_temp_tolerance
            
            # Temperature is above ignition threshold
            temp_high = current_temp > self.ignition_temp_threshold
            
            # Sufficient time has passed since ignition
            time_since_ignition = self.current_time - self.ignition_time
            sufficient_time = time_since_ignition > (self.steady_time_factor * self.ignition_time)
            
            # Check steady state condition
            if temp_stable and temp_high and sufficient_time:
                self.reached_steady_state = True
                self.count_since_steady_state += 1
                if self.verbose:
                    print(f"Steady state reached at T={current_temp:.1f}K, t={self.current_time:.6f}s")
                    print(f"  Temperature change: {temp_change:.3f}K")
                    print(f"  Time since ignition: {time_since_ignition:.6f}s")

        # Update previous temperature for next check
        self.previous_temperature = current_temp
    
    def _integrate_super_step(self, action):
        """Integrate one super-step using chosen solver with CFD-like reset each timestep"""
        solver = self.solvers[action]
        config = self.solver_configs[action]
        
        start_time = time.time()
        
        try:
            # Integrate each timestep individually (like CFD)
            for step in range(self.super_steps):
                if config['type'] == 'cvode':
                    # Reset solver to current state (fresh like each CFD grid point)
                    solver.set_state(self.current_state, 0.0)
                    # Integrate for one timestep
                    self.current_state = solver.solve_to(self.dt)
                    
                elif config['type'] == 'qss':
                    # Reset solver to current state (fresh like each CFD grid point)
                    solver.setState(self.current_state.tolist(), 0.0)
                    # Integrate for one timestep
                    result = solver.integrateToTime(self.dt)
                    if result != 0:
                        raise RuntimeError(f"QSS integration failed with code {result}")
                    self.current_state = np.array(solver.y)
                
                # Update gas object with new state for next iteration
                self.gas.TPY = self.current_state[0], self.current_pressure, self.current_state[1:]
                self.gibbs_free_energy.append(self.gas.gibbs_mass)
                self.states_trajectory.append(self.current_state.copy())
                self.current_time += self.dt
                self.times_trajectory.append(self.current_time)
        
            
            cpu_time = time.time() - start_time
            
            # Calculate timestep error against reference
            ref_idx = (self.current_episode + 1) * self.super_steps
            if ref_idx < len(self.ref_states):
                timestep_error = self._calculate_error(self.current_state, self.ref_states[ref_idx])
            else:
                timestep_error = 0.0
            
            # Update error tracking
            self.timestep_errors.append(timestep_error)
            
            # Calculate reward using new reward function
            reward = self.reward_function.step_reward(cpu_time, timestep_error, action, self.reached_steady_state)
            
            return reward, True, cpu_time, timestep_error
            
        except Exception as e:
            cpu_time = time.time() - start_time
            if self.verbose:
                print(f"Solver {config['name']} failed: {e}")
            
            # Handle failed integration
            timestep_error = float('inf')
            self.timestep_errors.append(timestep_error)
            
            reward = -5.0
            
            return reward, False, cpu_time, timestep_error
    
    def _calculate_error(self, state, ref_state):
        """Calculate relative error between states"""
        try:
        
            temp_error = np.abs((state)[0] - (ref_state)[0]) / ref_state[0]
            
            # Species errors (relative where possible)
            species_errors = []
            for idx in self.representative_species_indices:
                species_error = np.abs((state)[idx+1] - (ref_state)[idx+1])
                species_errors.append(species_error)
            
            # Combined error (weighted average)
            total_error = 0.3 * temp_error + 0.7 * np.mean(species_errors)
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating error: {e}")
            total_error = 0.0
            
        return total_error
    
    def _get_info(self):
        """Get environment info"""
        return {
            'episode': self.current_episode,
            'total_episodes': self.n_episodes,
            'current_conditions': {
                'temperature': self.current_temp,
                'pressure': self.current_pressure / ct.one_atm,
                'total_time': self.total_time
            },
            'episode_stats': {
                'avg_cpu_time': np.mean(self.cpu_times) if self.cpu_times else 0.0,
                'avg_timestep_error': np.mean(self.timestep_errors) if self.timestep_errors else 0.0,
                'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                'action_counts': {i: self.action_history.count(i) for i in range(len(self.solver_configs))},
                'within_timestep_threshold_rate': np.mean([e <= self.reward_function.epsilon for e in self.timestep_errors]) if self.timestep_errors else 0.0
            },
            'reward_config': self.reward_function.get_aux(),
            'steady_state_info': {
                'ignition_detected': self.ignition_detected,
                'ignition_time': self.ignition_time,
                'reached_steady_state': self.reached_steady_state,
                'current_time': self.current_time,
            },
            'gibbs_free_energy': self.gibbs_free_energy[-1],
            'gibbs_free_energy_history': self.gibbs_free_energy
            }
    
    def get_solver_names(self):
        """Get list of solver names for reference"""
        return [config['name'] for config in self.solver_configs]
    
    def get_error_analysis(self):
        """Get detailed error analysis for current episode"""
        if not self.timestep_errors:
            return None
        
        return {
            'timestep_errors': np.array(self.timestep_errors),
            'timestep_threshold': self.reward_function.epsilon,
            'timestep_violations': np.sum(np.array(self.timestep_errors) > self.reward_function.epsilon),
            'max_timestep_error': np.max(self.timestep_errors),
        }


def get_initial_state(gas):
    """Helper function to get initial state from gas object"""
    return np.hstack([gas.T, gas.Y])


if __name__ == "__main__":
    # Create environment with custom reward configuration
    # reward_config = {
    #     'error_threshold': 1e-4,
    #     'cpu_reward_scale': 1,
    #     'error_penalty_scale': 1.0,
    #     'alpha': 0.5
    # }
    reward_config = {
        'epsilon': 1e-3,
        'lambda_init': 1.0,
        'lambda_lr': 0.05,
        'target_violation': 0.0,
        'cpu_log_delta': 1e-3,
        'reward_clip': 5.0
    }
    
    sigmoid_reward_config = {
        'epsilon': 1e-3,
        'beta': 6.0,
        'margin_decades': 0.2,
        'cpu_log_delta': 1e-3,
        'reward_clip': 50.0
    }
    
    aug_lag_reward_config = {
        'epsilon': 1e-3,
        'cpu_log_delta': 1e-3,
        'reward_clip': 10.0,
        'lambda_init': 1.0,
        'rho_init': 1.0,
        'lambda_max': 1e4,
        'rho_max': 1e4,
        'target_violation': 0.0,
        'k_update': 1,
        'ema_alpha': 0.9
    }
    #reward_function = AugLagReward(**aug_lag_reward_config)
    
    reward_function = LagrangeReward1(**reward_config)
    # training_initial_conditions = np.load("training_initial_states.npy")
    # print(training_initial_conditions.shape)
    mechanism_file = "large_mechanism/n-dodecane.yaml"
    fuel = 'nc12h26'
    oxidizer = 'O2:0.21, N2:0.79'
    temp_range = (300, 1400)
    pressure_range = (1, 6)
    time_range = (1e-3, 1e-2)
    dt_range = (1e-6, 1e-6)
    etol = reward_config['epsilon']
    super_steps = 100
    
    env = IntegratorSwitchingEnv(mechanism_file=mechanism_file, fuel=fuel, 
                                 oxidizer=oxidizer,reward_function=reward_function, 
                                 temp_range=temp_range, pressure_range=pressure_range, 
                                 time_range=time_range, dt_range=dt_range, etol=etol, 
                                 super_steps=super_steps, verbose=True, termination_count_threshold=100)

    solver_configs = [
            # CVODE BDF with different tolerances
            {'type': 'cvode', 'rtol': 1e-6, 'atol': 1e-12, 'mxsteps': 100000, 'name': 'CVODE_BDF'},
            
            {'type': 'qss', 'dtmin': 1e-16, 'dtmax': 1e-6, 'stabilityCheck': False, 'itermax': 1, 'epsmin': 0.02, 'epsmax': 10.0, 'abstol': 1e-8, 'mxsteps': 1000, 'name': 'QSS'},

        ]

    env.solver_configs = solver_configs
    
    T = 650
    P = 3*ct.one_atm
    phi = 1.0
    total_time = 5e-2
    dt = 1e-6
    etol = 1e-3
    obs, info = env.reset(temperature=T, pressure=P, phi=phi, total_time=total_time, dt=dt, etol=etol)
    max_ref_temp = np.max(env.ref_states[:, 0])
    print(f"Initial temperature: {T}, Initial pressure: {P}, Initial phi: {phi}, Max reference temperature: {max_ref_temp} - end time: {total_time} - dt: {dt} - super_steps: {super_steps}")
    # if max_ref_temp < 5000:
    #     exit()
    super_steps = env.super_steps
    

    #time.sleep(10)
    solver_names = env.get_solver_names()
    all_action_comparison = {}
    
    for action in [1]:
        cpu_times = []
        rewards = []
        cpu_rewards = []
        accuracy_rewards = []
        timestep_errors = []
        cumulative_errors = []
        print(f"Resetting environment for action: {action} -  {solver_names[action]}")
        obs, info = env.reset(temperature=T, pressure=P, phi=phi, total_time=total_time, dt=dt, etol=etol)
        from tqdm import tqdm
        
        pbar = tqdm(total=env.n_episodes, desc=f'Running solver {solver_names[action]}')
        count = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            cpu_times.append(info['cpu_time'])
            rewards.append(reward)
            timestep_errors.append(info['timestep_error'])
            count += 1
            pbar.set_postfix({
                'T': f'{env.current_state[0]:.1f}K',
                'A': action,
                'R': f'{reward:.1f}',
                'C': f'{info["cpu_time"]:.3f}s'
            })
            pbar.update(1)

            if terminated or truncated:
                pbar.close()
                break
        
        all_action_comparison[action] = {
            'cpu_times': cpu_times,
            'rewards': rewards,
            'cpu_rewards': cpu_rewards,
            'accuracy_rewards': accuracy_rewards,
            'timestep_errors': timestep_errors,
            'cumulative_errors': cumulative_errors,
            'trajectory': env.states_trajectory,
            'times_trajectory': env.times_trajectory
        }

    ref_temp = env.ref_states[:, 0]
    ref_time = env.ref_times
    # save all_action_comparison to a json file
    import pickle
    with open('all_action_comparison.pkl', 'wb') as f:
        pickle.dump(all_action_comparison, f)
    # compare all actions
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
    line_styles = ['-', '--', ':', '-.']
    for action in all_action_comparison.keys():
        times = all_action_comparison[action]['times_trajectory']
        time_step_error = all_action_comparison[action]['timestep_errors']
        reward = all_action_comparison[action]['rewards']
        ax1.plot(time_step_error, label='Action ' + str(action), linewidth=2, linestyle=line_styles[action])
        # plot reference
        
        ax2.plot(-np.log10(all_action_comparison[action]['cpu_times']), label='Action ' + str(action), linewidth=2, linestyle=line_styles[action])
        ax3.plot(reward, label='Action ' + str(action), linewidth=2, linestyle=line_styles[action])
    # ax1.plot(ref_time, ref_temp, label='Reference', linewidth=2, linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Time Step Error')
    ax1.legend()
    ax1.set_title('Time Step Error')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('CPU Time')
    ax2.legend()
    ax2.set_title('CPU Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.set_title('Reward')
    plt.savefig('reward_comparison.png')
    plt.close()
    
    # compare all actions

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    line_styles = ['-', '--', ':', '-.']
    for action in all_action_comparison.keys():
        times = all_action_comparison[action]['times_trajectory']
        temp = np.array(all_action_comparison[action]['trajectory'])[:, 0]
        ax1.plot(times, temp, label='Action ' + str(action), linewidth=2, linestyle=line_styles[action])
        # plot reference
        
        ax2.plot(all_action_comparison[action]['cpu_times'], label='Action ' + str(action), linewidth=2, linestyle=line_styles[action])
    ax1.plot(ref_time, ref_temp, label='Reference', linewidth=2, linestyle=':')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature')
    ax1.legend()

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('CPU Time')
    ax2.legend()
    plt.savefig('temperature_comparison.png')

    