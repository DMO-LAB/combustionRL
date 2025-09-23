import numpy as np
import torch
import torch.nn as nn
import cantera as ct
from torch.distributions import Categorical
import math

class PPONetwork(nn.Module):
    """Lightweight PPO Network for inference only"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[128, 128]):
        super(PPONetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, action_dim)
        )
        
        # Critic head (not needed for inference but kept for compatibility)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared_net(x)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value

class RLSolverSelector:
    """Lightweight RL policy inference for CFD integration"""
    
    def __init__(self, model_path, mechanism_file, 
                 solver_configs=None, device='cpu', use_time_left=False):
        """
        Initialize the RL solver selector
        
        Args:
            model_path: Path to saved PyTorch model (.pt file)
            mechanism_file: Path to Cantera mechanism file
            solver_configs: List of solver configurations (optional, for name mapping)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.mechanism_file = mechanism_file
        
        # Load Cantera gas object
        self.gas = ct.Solution(mechanism_file)
        
        # Key species for observation (same as training)
        self.key_species = ['O','H','OH','H2O','O2','H2','H2O2','N2']
        self.key_species_indices = np.array([self.gas.species_index(spec) for spec in self.key_species])
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract observation normalization parameters
        self.obs_mean = torch.tensor(checkpoint['obs_mean'], dtype=torch.float32, device=device)
        self.obs_var = torch.tensor(checkpoint['obs_var'], dtype=torch.float32, device=device)
        self.obs_std = torch.sqrt(self.obs_var + 1e-8)
        
        # Initialize network
        obs_dim = len(self.obs_mean)
        if solver_configs is None:
            # Default solver configs (adjust based on your setup)
            action_dim = 2  # Adjust this based on your number of solvers
        else:
            action_dim = len(solver_configs)
            
        self.network = PPONetwork(obs_dim, action_dim)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        self.network.to(device)
        
        # Store solver configs for name mapping
        self.solver_configs = solver_configs
        
        # Initialize state tracking for trend features
        self.last_obs = None
        self.last_obs_batch = None
        
        # Simulation parameters (you may need to adjust these)
        self.current_time = 0.0
        self.total_time = 1e-2  # Default, update as needed
        self.current_pressure = 1.0 * ct.one_atm  # Default pressure
        self.use_time_left = use_time_left
        print(f"RL Solver Selector initialized with {action_dim} solvers on {device}")
        
    
    def update_simulation_params(self, current_time, total_time, pressure):
        """Update simulation parameters for accurate observation calculation"""
        self.current_time = current_time
        self.total_time = total_time
        self.current_pressure = pressure
    
    def reset_state(self):
        """Reset internal state (call at start of new simulation)"""
        self.last_obs = None
        self.last_obs_batch = None
        self.current_time = 0.0
    
    def _get_observation(self, temperature, species_fractions):
        """
        Convert temperature and species to normalized observation
        
        Args:
            temperature: Temperature in K
            species_fractions: Mass fractions array (same order as gas.species_names)
        
        Returns:
            Normalized observation tensor
        """
        # Update gas state for species lookup
        self.gas.TPY = temperature, self.current_pressure, species_fractions
        
        # Key species (log10 mole fractions)
        key_vals = []
        for spec in self.key_species:
            try:
                idx = self.gas.species_index(spec)
                key_vals.append(species_fractions[idx])
            except ValueError:
                key_vals.append(0.0)
        
        # Base features (same as training)
        temp_norm = (temperature - 300.0) / 2000.0
        species_log = np.log10(np.maximum(key_vals, 1e-20))
        pressure_log = np.log10(self.current_pressure / ct.one_atm)
        
        # Time-left feature
        if self.use_time_left:
            time_left = max(0.0, self.total_time - self.current_time)
            time_left_norm = np.clip(time_left / (self.total_time + 1e-12), 0.0, 1.0)
            base_obs = np.hstack([temp_norm, species_log, pressure_log, time_left_norm]).astype(np.float32)
        else:
            base_obs = np.hstack([temp_norm, species_log, pressure_log]).astype(np.float32)
        
        # Base observation
        
        # Trend features (delta from last observation)
        if self.last_obs is None:
            trend = np.zeros_like(base_obs, dtype=np.float32)
        else:
            trend = base_obs - self.last_obs
        
        # Combined observation
        obs = np.hstack([base_obs, trend]).astype(np.float32)
        self.last_obs = base_obs.copy()
        
        return obs
    
    def _get_observation_batch(self, temperatures, species_fractions_batch):
        """
        Vectorized observation builder for a batch of grid points.
        
        Args:
            temperatures: 1D array-like of temperatures (K), shape (N,)
            species_fractions_batch: 2D array of mass fractions, shape (N, num_species)
        Returns:
            obs_batch: 2D numpy array, shape (N, obs_dim)
        """
        temperatures = np.asarray(temperatures, dtype=np.float32)
        species_fractions_batch = np.asarray(species_fractions_batch, dtype=np.float32)
        num_points = temperatures.shape[0]

        # Key species fractions (N, K)
        key_vals = species_fractions_batch[:, self.key_species_indices]
        key_vals = np.maximum(key_vals, 1e-20)
        species_log = np.log10(key_vals)

        # Base features
        temp_norm = ((temperatures - 300.0) / 2000.0).reshape(num_points, 1)
        pressure_log = np.full((num_points, 1), np.log10(self.current_pressure / ct.one_atm), dtype=np.float32)
        if self.use_time_left:
            time_left = np.clip(self.total_time - self.current_time, 0.0, None)
            time_left_norm = np.full((num_points, 1), np.clip(time_left / (self.total_time + 1e-12), 0.0, 1.0), dtype=np.float32)
            base_obs = np.hstack([temp_norm, species_log, pressure_log, time_left_norm]).astype(np.float32)
        else:
            base_obs = np.hstack([temp_norm, species_log, pressure_log]).astype(np.float32)

        

        # Trend features per point
        if self.last_obs_batch is None or (
            isinstance(self.last_obs_batch, np.ndarray) and self.last_obs_batch.shape != base_obs.shape
        ):
            trend = np.zeros_like(base_obs, dtype=np.float32)
        else:
            trend = base_obs - self.last_obs_batch

        obs = np.hstack([base_obs, trend]).astype(np.float32)
        self.last_obs_batch = base_obs.copy()

        return obs

    def _normalize_observation(self, obs):
        """Apply observation normalization (same as training)"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        normalized = (obs_tensor - self.obs_mean) / self.obs_std
        return normalized.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad()
    def select_solver(self, temperature, species_fractions, deterministic=True):
        """
        Select solver based on current state
        
        Args:
            temperature: Temperature in K
            species_fractions: Mass fractions array
            deterministic: If True, use argmax; if False, sample from distribution
        
        Returns:
            action (int): Solver index
            confidence (float): Confidence/probability of selected action
        """
        # Get observation
        obs = self._get_observation(temperature, species_fractions)
        
        # Normalize observation
        obs_normalized = self._normalize_observation(obs)
        
        # Forward pass
        logits, _ = self.network(obs_normalized)
        
        if deterministic:
            # Use most probable action
            action = torch.argmax(logits, dim=1).item()
            # Get softmax probabilities for confidence
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, action].item()
        else:
            # Sample from distribution
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            confidence = torch.exp(dist.log_prob(torch.tensor([action]))).item()
        
        return action, confidence
    
    @torch.no_grad()
    def select_solver_batch(self, temperatures, species_fractions_batch, deterministic=True):
        """
        Vectorized solver selection for multiple grid points.
        
        Args:
            temperatures: 1D array of temperatures (K) with shape (N,)
            species_fractions_batch: 2D array of mass fractions with shape (N, num_species)
            deterministic: If True, use argmax; otherwise sample per point
        Returns:
            actions: list[int] of length N
            confidences: list[float] of length N
        """
        # Build and normalize observations in batch
        obs_batch = self._get_observation_batch(temperatures, species_fractions_batch)
        obs_mean = self.obs_mean
        obs_std = self.obs_std
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        obs_normalized = (obs_tensor - obs_mean) / obs_std

        logits, _ = self.network(obs_normalized)

        if deterministic:
            actions = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            confidences = torch.exp(dist.log_prob(actions))

        return actions.tolist(), confidences.detach().cpu().tolist()

    def get_solver_name(self, action):
        """Get solver name from action index"""
        if self.solver_configs and action < len(self.solver_configs):
            return self.solver_configs[action]['name']
        else:
            return f"Solver_{action}"
    
    def __call__(self, temperature, species_fractions, deterministic=True):
        """
        Convenient callable interface
        
        Args:
            temperature: Temperature in K
            species_fractions: Mass fractions array
            deterministic: Whether to use deterministic action selection
        
        Returns:
            action (int): Selected solver index
        """
        action, _ = self.select_solver(temperature, species_fractions, deterministic)
        return action
    
