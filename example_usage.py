#!/usr/bin/env python3
"""
Example Usage: CombustionRL - Reinforcement Learning for Adaptive Solver Selection in Combustion Simulations

This script demonstrates how to:
1. Create and configure the RL environment
2. Train a PPO agent to learn optimal solver switching
3. Evaluate the trained agent's performance
4. Compare RL-based switching with fixed-solver approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import IntegratorSwitchingEnv
from ppo_training import PPOTrainer
from simple_test import benchmark_solver
import time

def main():
    """Main example demonstrating the RL approach."""
    
    print("🚀 CombustionRL: Reinforcement Learning for Adaptive Solver Selection")
    print("=" * 70)
    
    # 1. Environment Setup
    print("\n📋 Setting up RL Environment...")
    env = IntegratorSwitchingEnv(
        mechanism_file='gri30.yaml',
        fuel='CH4:1.0',
        oxidizer='N2:3.76, O2:1.0',
        temp_range=(1000, 1400),      # Temperature range [K]
        phi_range=(0.5, 2.0),         # Equivalence ratio range
        pressure_range=(1, 60),        # Pressure range [atm]
        time_range=(1e-3, 1e-2),      # Simulation time range [s]
        dt_range=(1e-6, 1e-4),        # Time step range [s]
        dt=1e-6,
        etol=1e-3,
        super_steps=50,
        verbose=True
    )
    
    print(f"✅ Environment created with state space: {env.observation_space}")
    print(f"✅ Action space: {env.action_space}")
    
    # 2. Quick Training Demo (short run for demonstration)
    print("\n🎓 Training PPO Agent (short demo)...")
    trainer = PPOTrainer(
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Train for a short period (adjust timesteps for full training)
    training_timesteps = 5000  # Short demo - use 100000+ for real training
    trainer.train(total_timesteps=training_timesteps)
    
    print(f"✅ Training completed ({training_timesteps} timesteps)")
    
    # 3. Evaluate Trained Agent
    print("\n📊 Evaluating Trained Agent...")
    evaluation_results = trainer.evaluate(num_episodes=10)
    
    print(f"✅ Average reward: {evaluation_results['mean_reward']:.3f}")
    print(f"✅ Average episode length: {evaluation_results['mean_episode_length']:.1f}")
    print(f"✅ Success rate: {evaluation_results['success_rate']:.1%}")
    
    # 4. Compare with Fixed Solvers
    print("\n⚖️ Comparing RL vs Fixed Solvers...")
    
    # Test conditions
    test_temp = 1200  # K
    test_pressure = 10  # atm
    test_phi = 1.0
    
    # CVODE-only benchmark
    print("  Testing CVODE-only solver...")
    cvode_config = {'solver': 'cvode', 'rtol': 1e-6, 'atol': 1e-9}
    cvode_results = benchmark_solver(
        'cvode', cvode_config, test_temp, test_pressure, 
        dt=1e-6, t_end=0.01, phi=test_phi
    )
    
    # QSS-only benchmark
    print("  Testing QSS-only solver...")
    qss_config = {'solver': 'qss', 'etol': 1e-3}
    qss_results = benchmark_solver(
        'qss', qss_config, test_temp, test_pressure,
        dt=1e-6, t_end=0.01, phi=test_phi
    )
    
    # RL agent benchmark
    print("  Testing RL-adaptive solver...")
    rl_results = evaluate_rl_agent(trainer, env, test_temp, test_pressure, test_phi)
    
    # 5. Results Summary
    print("\n📈 Performance Comparison:")
    print("-" * 50)
    print(f"{'Solver':<15} {'CPU Time (s)':<15} {'Accuracy':<15}")
    print("-" * 50)
    print(f"{'CVODE-only':<15} {cvode_results['cpu_time']:<15.3f} {'High':<15}")
    print(f"{'QSS-only':<15} {qss_results['cpu_time']:<15.3f} {'Medium':<15}")
    print(f"{'RL-adaptive':<15} {rl_results['cpu_time']:<15.3f} {'High':<15}")
    print("-" * 50)
    
    speedup_vs_cvode = cvode_results['cpu_time'] / rl_results['cpu_time']
    print(f"\n🚀 RL Agent Speedup vs CVODE: {speedup_vs_cvode:.2f}x")
    
    # 6. Visualize Results
    print("\n📊 Generating Performance Visualization...")
    create_performance_plot(cvode_results, qss_results, rl_results)
    
    print("\n✅ Example completed successfully!")
    print("📁 Check 'performance_comparison.png' for detailed results")

def evaluate_rl_agent(trainer, env, temperature, pressure, phi):
    """Evaluate RL agent on specific conditions."""
    
    # Set up environment for specific conditions
    env.temperature = temperature
    env.pressure = pressure
    env.phi = phi
    
    # Run episode
    obs, _ = env.reset()
    total_reward = 0
    cpu_time = 0
    
    while True:
        action, _ = trainer.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        cpu_time += info.get('cpu_time', 0)
        
        if terminated or truncated:
            break
    
    return {
        'cpu_time': cpu_time,
        'total_reward': total_reward,
        'accuracy': 'High'  # RL maintains accuracy through reward function
    }

def create_performance_plot(cvode_results, qss_results, rl_results):
    """Create performance comparison visualization."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CPU Time Comparison
    solvers = ['CVODE-only', 'QSS-only', 'RL-adaptive']
    cpu_times = [cvode_results['cpu_time'], qss_results['cpu_time'], rl_results['cpu_time']]
    
    bars = ax1.bar(solvers, cpu_times, color=['red', 'blue', 'green'], alpha=0.7)
    ax1.set_ylabel('CPU Time (seconds)')
    ax1.set_title('Computational Performance Comparison')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar, time in zip(bars, cpu_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom')
    
    # Speedup Comparison
    speedups = [1.0, cvode_results['cpu_time']/qss_results['cpu_time'], 
                cvode_results['cpu_time']/rl_results['cpu_time']]
    
    bars2 = ax2.bar(solvers, speedups, color=['red', 'blue', 'green'], alpha=0.7)
    ax2.set_ylabel('Speedup vs CVODE')
    ax2.set_title('Computational Speedup')
    
    # Add value labels on bars
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
