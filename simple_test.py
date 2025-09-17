
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import time
import SundialsPy as SP
from typing import Tuple, List, Dict, Any, Optional
import os
from tqdm import tqdm
from utils import  create_solver


mechanism = 'gri30.yaml'
fuel = 'CH4:1.0'
oxidizer = 'N2:3.76, O2:1.0'

def benchmark_solver(solver_type, config, temperature, pressure, 
                    dt=1e-6, t_end=0.5, save_states=True, phi=1):
    """
    Benchmark chemistry solvers with CFD-realistic reinitialization
    
    Parameters:
    -----------
    solver_type : str
        'cvode' or 'qss'
    config : dict
        Solver configuration
    temperature : float
        Temperature for solver
    pressure : float
        Pressure for solver
    dt : float
        Time step size (default: 1e-6)
    t_end : float
        End time (default: 0.5)
    save_states : bool
        Whether to save state history (default: True)
        
    Returns:
    --------
    dict with keys:
        'states': list of states (if save_states=True)
        'cpu_times': array of per-step CPU times
        'total_cpu_time': total CPU time
        'n_steps': number of steps
        'solver_type': solver type used
        'dt': time step used
        'final_state': final state
    """
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temperature, pressure
    Y0 = gas.Y.copy()
    initial_state = np.hstack(([temperature], Y0))
    n_steps = int(t_end / dt)
    
    print(f"Running {solver_type.upper()} benchmark: {n_steps} steps, dt={dt:.1e} temperature={temperature:.3f} K")
    
    # Initialize storage
    states = [] if save_states else None
    cpu_times = np.zeros(n_steps)
    current_state = initial_state.copy()
    
    # Create solver
    solver = create_solver(solver_type, config, gas, initial_state, 0.0, pressure)
    
    total_start_time = time.time()
    
    # Integration loop
    for step in tqdm(range(n_steps), desc=f'{solver_type.upper()} Integration (T={current_state[0]:.1f}K)'):
        step_start_time = time.time()
        
        # Update gas object with current state
        gas.TPY = current_state[0], pressure, current_state[1:]
        
        if solver_type == 'cvode':
            # Reset to fresh conditions (like each CFD grid point)
            solver.set_state(current_state, 0.0)
            
            # Integrate for one time step
            try:
                result = solver.solve_to(dt)
                current_state = result
            except Exception as e:
                print(f'CVODE Error at step {step}: {e}')
                current_state = initial_state.copy()
                
        elif solver_type == 'qss':
            # Reset to fresh conditions (like each CFD grid point)
            solver.setState(current_state.tolist(), 0.0)
            
            # Integrate for one time step
            result = solver.integrateToTime(dt)
            
            if result == 0:
                current_state = np.array(solver.y)
            else:
                print(f'QSS Error at step {step}: result={result}')
                current_state = initial_state.copy()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        # Record timing and state
        step_cpu_time = time.time() - step_start_time
        cpu_times[step] = step_cpu_time
        
        if save_states:
            states.append(current_state.copy())
    
    total_cpu_time = time.time() - total_start_time
    # Prepare results
    results = {
        'states': states,
        'cpu_times': cpu_times,
        'total_cpu_time': total_cpu_time,
        'n_steps': n_steps,
        'solver_type': solver_type,
        'dt': dt,
        't_end': t_end,
        'final_state': current_state.copy(),
        'avg_step_time': np.mean(cpu_times),
        'total_step_time': np.sum(cpu_times),
        'overhead_time': total_cpu_time - np.sum(cpu_times)
    }
    
    # Print summary
    print(f"\n{solver_type.upper()} Benchmark Results:")
    print(f"  Total steps: {n_steps}")
    print(f"  Total CPU time: {total_cpu_time:.3f} s")
    print(f"  Total step time: {np.sum(cpu_times):.3f} s")
    print(f"  Overhead time: {total_cpu_time - np.sum(cpu_times):.3f} s")
    print(f"  Average step time: {np.mean(cpu_times)*1000:.3f} ms")
    print(f"  Min step time: {np.min(cpu_times)*1000:.3f} ms")
    print(f"  Max step time: {np.max(cpu_times)*1000:.3f} ms")
    print(f"  Final temperature: {current_state[0]:.3f} K")
    return results

def compare_solvers(config, T, P, dt=1e-6, t_end=0.5):
    """
    Compare CVODE and QSS solvers
    
    Returns:
    --------
    dict with 'cvode' and 'qss' results
    """
    print("Starting solver comparison...")
    print(f"Parameters: dt={dt:.1e}, t_end={t_end}")
    
    # Benchmark CVODE
    cvode_results = benchmark_solver('cvode', config,  T, P, dt, t_end, save_states=True)
    
    print("\n" + "="*50)
    
    # Benchmark QSS
    qss_results = benchmark_solver('qss', config, T, P, dt, t_end, save_states=True)
    
    # Summary comparison
    print("\n" + "="*50)
    print("COMPARISON SUMMARY:")
    print(f"CVODE total time: {cvode_results['total_cpu_time']:.3f} s")
    print(f"QSS total time:   {qss_results['total_cpu_time']:.3f} s")
    print(f"Speedup (QSS/CVODE): {cvode_results['total_cpu_time']/qss_results['total_cpu_time']:.2f}x")
    print(f"CVODE avg step: {cvode_results['avg_step_time']*1000:.3f} ms")
    print(f"QSS avg step:   {qss_results['avg_step_time']*1000:.3f} ms")
    
    return {
        'cvode': cvode_results,
        'qss': qss_results
    }
    
def plot_results(cvode_results, qss_results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), dpi=300)
    ax1.plot(np.log10(np.maximum(cvode_results['cpu_times'], 1e-7)), label='cvode')
    ax1.plot(np.log10(np.maximum(qss_results['cpu_times'], 1e-7)), label='qss')
    ax1.legend()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Time (s)')
    ax1.set_title('CPU Time Comparison')
    ax1.grid(True)
    ax2.grid(True)
    cvode_temperatures = np.array(cvode_results['states'])[:, 0]
    qss_temperatures = np.array(qss_results['states'])[:, 0]
    ax2.plot(cvode_temperatures, label='cvode')
    ax2.plot(qss_temperatures, label='qss')
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature Comparison')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('cpu_times.png')
    plt.close()
    print(f"Comparison saved as 'cpu_times.png'")
    
    

if __name__ == "__main__":
    mechanism = 'gri30.yaml'    #"/Users/elotech/Downloads/research_code/pysundial/large_mechanism/n-dodecane.yaml"
    temp = 1200
    pressure = 6 * ct.one_atm

    phi =  1
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temp, pressure
    major_species = ['o', 'h', 'oh', 'h2o', 'o2', 'h2']
    config = {
        'rtol': 1e-8,
        'atol': 1e-10,
        'dtmin': 1e-16,
        'dtmax': 1e-6,
        'itermax': 1,
        'stabilityCheck': False,
        'epsmin': 2e-2,
        'epsmax': 10.0,
        'abstol': 1e-11,
    }



    # Setup your initial conditions

    dt = 1e-5
    t_end = 0.01

    # # Run single solver benchmark
    # cvode_results = benchmark_solver('cvode', config, gas, initial_state, pressure)

    # Or compare both solvers
    comparison = compare_solvers(config, temp, pressure, dt, t_end)

    plot_results(comparison['cvode'], comparison['qss'])