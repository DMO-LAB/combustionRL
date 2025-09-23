from ember import Config, Paths, InitialCondition, StrainParameters, General, Times, TerminationCondition, ConcreteConfig, General, Debug, \
  CvodeTolerances, Chemistry, QssTolerances, QssTolerances
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
import os
import time
from ember import _ember
from inference import RLSolverSelector
from tqdm import tqdm


def set_integrator_rl(solver, rl_selector, decision_step, super_steps=100, total_time=0.02, pressure=101325):
    """
    Set integrator type using RL agent.
    
    Args:
        solver: Ember solver instance
        rl_selector: RLSolverSelector instance
        decision_step: Current decision step (increments every super_steps timesteps)
        super_steps: Number of timesteps per RL decision (should match training)
    """
    nPoints = len(solver.T)
    
    # Map RL actions to integrator names
    action_to_integrator = {0: 'cvode', 1: 'qss'}
    
    # Update RL selector with current simulation time
    current_time = decision_step * super_steps * 1e-6  # Assuming dt = 1e-6
    rl_selector.update_simulation_params(current_time, total_time, pressure)  # 0.02s total, 1 atm
    
    try:
        temperatures = np.asarray(solver.T, dtype=np.float32)
        species_batch = np.asarray(solver.Y, dtype=np.float32).transpose(1,0)
        actions, confidences = rl_selector.select_solver_batch(temperatures, species_batch, deterministic=True)
        integ_types = [action_to_integrator.get(a, 'cvode') for a in actions]
    except Exception as e:
        print(f"RL batch selection failed: {e}")
        integ_types = ['cvode'] * nPoints
    
    return integ_types

def set_integrator_heuristic(solver, constant_int=None):
    """
    Original heuristic integrator selection (for comparison)
    """
    nPoints = len(solver.T)
    
    # Start with temperature-based decision
    integ = np.where(solver.T <= 600.0, 'qss', 'cvode')
    
    try:
        # Get equivalence ratio
        phi = solver.phi
        
        # Use qss for problematic phi regions
        integ = np.where(phi == -1, 'qss', integ)  # invalid phi
        integ = np.where(phi <= 1e-8, 'qss', integ)  # oxidizer-dominated
        integ = np.where(phi >= 1e4, 'qss', integ)   # fuel-dominated
        
        # Create boolean mask for CVODE points
        cvode_mask = (integ == 'cvode')
        
        # Shift masks to check neighbors
        cvode_mask_left = np.roll(cvode_mask, 1)
        cvode_mask_right = np.roll(cvode_mask, -1)
        
        # Fix boundary conditions
        cvode_mask_left[0] = False
        cvode_mask_right[-1] = False
        
        # If either neighbor is CVODE, use CVODE
        use_cvode = cvode_mask | cvode_mask_left | cvode_mask_right
        integ = np.where(use_cvode, 'cvode', 'qss')
        
    except Exception as e:
        print(f"Warning: Could not calculate phi for integrator selection: {e}")
        pass
    
    # Convert to list
    integ = integ.tolist()
    
    # Override with constant integrator if specified
    if constant_int:
        integ = [constant_int] * nPoints
    
    return integ

def run_simulation_with_method(integrator_method='rl', model_path=None, super_steps=100, t_fuel=300, t_oxidizer=1200, 
                               total_time=0.02, pressure=101325, center_width=0.0, slope_width=0.0, x_left=-0.02, x_right=0.02,
                               npoints=100, strain_rate=100, equilibrate_counterflow=False, dt=1e-6, fuel_choice=0, rtol=1e-10, atol=1e-20, use_time_left=False):
    """
    Run simulation with specified integrator selection method
    
    Args:
        integrator_method: 'rl', 'cvode', 'qss', or 'heuristic'
        model_path: Path to RL model (required if method='rl')
        super_steps: Number of timesteps per RL decision
    """
    # Configuration setup (using your existing config)
    fuel_info = [
        {
            'mechanism_file': "large_mechanism/n-dodecane.yaml",
            'fuel': 'nc12h26:1.0',
            'oxidizer': 'n2:3.76, o2:1.0',
        },
        {
            'mechanism_file': 'large_mechanism/ch4_53species.yaml',
            'fuel': 'CH4:1.0',
            'oxidizer': 'O2:1.0, N2:3.76',
        },
    ]
    
    choice = fuel_choice  # Using n-dodecane for testing
    fuel = fuel_info[choice]['fuel']
    oxidizer = fuel_info[choice]['oxidizer']
    mechanism_file = fuel_info[choice]['mechanism_file']
    
    output_dir = f'run/ex_diffusion_{integrator_method}'
    
    conf = Config(
        Paths(outputDir=output_dir),
        General(nThreads=1, chemistryIntegrator='cvode'),
        Chemistry(mechanismFile=mechanism_file),
        InitialCondition(Tfuel=t_fuel, Toxidizer=t_oxidizer, centerWidth=center_width,
                        equilibrateCounterflow=False, flameType='diffusion',
                        slopeWidth=slope_width, xLeft=x_left, pressure=pressure,
                        xRight=x_right, nPoints=npoints, fuel=fuel, oxidizer=oxidizer),
        StrainParameters(final=strain_rate, initial=strain_rate),
        Times(globalTimestep=dt, profileStepInterval=20),  # dt = 1e-6
        TerminationCondition(abstol=0.0, dTdtTol=0, steadyPeriod=1.0,
                           tEnd=total_time, tolerance=0.0),
        QssTolerances(abstol=1e-8, dtmin=1e-16, dtmax=1e-6,
                     epsmin=2e-3, epsmax=100, iterationCount=2,
                     stabilityCheck=False),
        CvodeTolerances(relativeTolerance=rtol, momentumAbsTol=atol,
                       energyAbsTol=atol, speciesAbsTol=atol,
                       minimumTimestep=1e-18, maximumTimestep=1e-5),
        Debug(veryVerbose=False),
    )
    
    conf = ConcreteConfig(conf)
    
    # Create output directory
    if not os.path.isdir(conf.paths.outputDir):
        os.makedirs(conf.paths.outputDir, 0o0755)
    
    # Initialize RL selector if needed
    rl_selector = None
    if integrator_method == 'rl':
        if model_path is None:
            raise ValueError("model_path is required for RL method")
        rl_selector = RLSolverSelector(model_path, mechanism_file, use_time_left=use_time_left)
        rl_selector.reset_state()
    
    print(f"Starting simulation with {integrator_method} method")
    # Run simulation
    solver = _ember.FlameSolver(conf)
    solver.initialize()
    done = False
    initial_T = solver.T.copy()
    initial_Y = solver.Y.copy()
    cpu_time_list = []
    integrator_types_history = []
    T_history = []
    Y_history = []
    decision_step = 0
    timestep_count = 0
    
    cvode_count = 0
    qss_count = 0
    count = 0
    pbar = tqdm(total=int(total_time / dt), desc="Progress")
    while not done:
        # Select integrator method
        if integrator_method == 'rl':
            # Make RL decision every super_steps timesteps
            if timestep_count % super_steps == 0:
                integrator_type_list = set_integrator_rl(solver, rl_selector, decision_step, super_steps, total_time=total_time, pressure=pressure)
                decision_step += 1
            # Use the same decision for super_steps timesteps
            solver.set_integrator_types(integrator_type_list)
            
        elif integrator_method == 'heuristic':
            integrator_type_list = set_integrator_heuristic(solver)
            solver.set_integrator_types(integrator_type_list)
            
        elif integrator_method in ['cvode', 'qss', 'cvode_tight']:
            if integrator_method == 'cvode_tight':
                integrator_method = 'cvode'
            solver.set_integrator_types([integrator_method] * npoints)
            integrator_type_list = [integrator_method] * npoints
        
        cvode_count = integrator_type_list.count('cvode')
        qss_count = integrator_type_list.count('qss')
        start_time = time.time()
        done = solver.step()
        time_taken = time.time() - start_time
        
        cpu_time_list.append(np.sum(solver.gridPointIntegrationTimes))
        
        #if count % super_steps == 0:
        integrator_types_history.append(integrator_type_list.copy())
        T_history.append(solver.T.copy())
        Y_history.append(solver.Y.copy())
        timestep_count += 1
        
        pbar.set_postfix({'T': f'{solver.T.max():.1f}K', 'T_initial': f'{initial_T.max():.1f}K', 'CVODE': f'{cvode_count}', 'QSS': f'{qss_count}'}, cpu_time=f'{np.sum(solver.gridPointIntegrationTimes):.4f}s')
        pbar.update(1)
        
        # if count % 500 == 0:
        #     print(f"Count: {count}, Max Temperature: {solver.T.max():.1f}K, Time taken: {time_taken:.4f}s")
        #     if integrator_method in ['rl', 'heuristic']:
        #         print(f"  CVODE points: {cvode_count}, QSS points: {qss_count}")

        count += 1
    
    return solver.T, initial_T, cpu_time_list, integrator_types_history, T_history, Y_history

def compare_methods(model_path, t_fuel=300, t_oxidizer=1200, total_time=0.02, pressure=101325, center_width=0.0, slope_width=0.0, x_left=-0.02, x_right=0.02,
                    npoints=100, strain_rate=100, equilibrate_counterflow=False, dt=1e-6, fuel_choice=0, use_time_left=False):
    """
    Compare different integrator selection methods
    
    Args:
        model_path: Path to trained RL model
    """
    methods = ['rl', 'qss', 'heuristic', 'cvode', 'cvode_tight']
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running simulation with {method.upper()} method")
        print(f"{'='*50}")
        
        try:
            if method == 'rl':
                T_final, T_initial, cpu_times, integrator_history, T_history, Y_history = run_simulation_with_method(
                    integrator_method=method, model_path=model_path, super_steps=100, t_fuel=t_fuel, t_oxidizer=t_oxidizer, 
                    total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                    npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, fuel_choice=fuel_choice, rtol=1e-8, atol=1e-12, use_time_left=use_time_left   
                )
            elif method == 'cvode_tight':
                T_final, T_initial, cpu_times, integrator_history, T_history, Y_history = run_simulation_with_method(
                    integrator_method=method, model_path=model_path, super_steps=100, t_fuel=t_fuel, t_oxidizer=t_oxidizer, 
                    total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                    npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, fuel_choice=fuel_choice, rtol=1e-10, atol=1e-20, use_time_left=use_time_left
                )
            else:
                T_final, T_initial, cpu_times, integrator_history, T_history, Y_history = run_simulation_with_method(
                    integrator_method=method, t_fuel=t_fuel, t_oxidizer=t_oxidizer, 
                    total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                    npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, fuel_choice=fuel_choice, rtol=1e-8, atol=1e-12, use_time_left=use_time_left
                )
            
            results[method] = {
                'T_final': T_final,
                'T_initial': T_initial,
                'cpu_times': cpu_times,
                'integrator_history': integrator_history,
                'T_history': T_history,
                'Y_history': Y_history,
                'total_cpu_time': np.sum(cpu_times),
                'mean_cpu_time': np.mean(cpu_times),
                'max_temperature': np.max(T_final),
                'T_history': T_history,
                'Y_history': Y_history
            }
            
            print(f"{method.upper()} Results:")
            print(f"  Total CPU time: {results[method]['total_cpu_time']:.4f}s")
            print(f"  Mean CPU time per step: {results[method]['mean_cpu_time']:.6f}s")
            print(f"  Max temperature: {results[method]['max_temperature']:.1f}K")
            print(f"  Number of timesteps: {len(cpu_times)}")
        
        except Exception as e:
            print(f"Error running {method}: {e}")
            results[method] = None
    
    # save all the results
    import pickle
    with open(f'results_1D_{t_fuel}_{t_oxidizer}_{pressure}__{fuel_choice}.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create comparison plots
    create_comparison_plots(results, t_fuel, t_oxidizer, pressure, fuel_choice)
    
    return results

def create_comparison_plots(results, t_fuel, t_oxidizer, pressure, fuel_choice):
    """Create comparison plots for different methods"""
    
    # Filter out failed methods
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: CPU time evolution
    ax1 = axes[0, 0]
    for method, data in valid_results.items():
        ax1.plot(data['cpu_times'], label=method.upper(), linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('CPU Time (s)')
    ax1.set_title('CPU Time Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temperature profiles
    ax2 = axes[0, 1]
    for method, data in valid_results.items():
        ax2.plot(data['T_final'], label=f'{method.upper()} Final', linewidth=2)
    
    # Plot initial temperature for reference
    if 'cvode' in valid_results:
        ax2.plot(valid_results['cvode']['T_initial'], 'k--', label='Initial', linewidth=2)
    
    ax2.set_xlabel('Grid Point')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Final Temperature Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative CPU time
    ax3 = axes[1, 0]
    for method, data in valid_results.items():
        cumulative_cpu = np.cumsum(data['cpu_times'])
        ax3.plot(cumulative_cpu, label=method.upper(), linewidth=2)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumulative CPU Time (s)')
    ax3.set_title('Cumulative CPU Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary (bar chart)
    ax4 = axes[1, 1]
    methods = list(valid_results.keys())
    total_cpu_times = [valid_results[m]['total_cpu_time'] for m in methods]
    max_temps = [valid_results[m]['max_temperature'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, total_cpu_times, width, label='Total CPU Time', alpha=0.7)
    bars2 = ax4_twin.bar(x + width/2, max_temps, width, label='Max Temperature', alpha=0.7, color='orange')
    
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Total CPU Time (s)', color='blue')
    ax4_twin.set_ylabel('Max Temperature (K)', color='orange')
    ax4.set_title('Performance Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper() for m in methods])
    
    # Add value labels on bars
    for bar, value in zip(bars1, total_cpu_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(total_cpu_times),
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2, max_temps):
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(max_temps),
                     f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'integrator_method_comparison_1D_{t_fuel}_{t_oxidizer}_{pressure}__{fuel_choice}__{total_time}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Total CPU (s)':<15} {'Mean CPU (s)':<15} {'Max Temp (K)':<12} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    baseline_cpu = valid_results['cvode']['total_cpu_time'] if 'cvode' in valid_results else None
    
    for method, data in valid_results.items():
        speedup = baseline_cpu / data['total_cpu_time'] if baseline_cpu else 1.0
        print(f"{method.upper():<12} {data['total_cpu_time']:<15.4f} "
              f"{data['mean_cpu_time']:<15.6f} {data['max_temperature']:<12.1f} "
              f"{speedup:<10.2f}x")

if __name__ == "__main__":
    # Example usage
    
    # Path to your trained RL model
    model_path = "logs/models/model_update_40.pt" 
    use_time_left = True
    super_steps = 100
    t_fuel = 300
    t_oxidizer = 800
    total_time = 0.008
    pressure = 101325 * 60
    center_width = 0.002
    slope_width = 0.001
    x_left = -0.004
    x_right = 0.004
    npoints = 100
    strain_rate = 100
    equilibrate_counterflow = 'HP'
    dt = 1e-6
    fuel_choice = 0
    MODE = 'compare' # rl or compare
    
    if MODE == 'rl':
        # Run individual simulation with RL
        try:
            print("Testing RL integration...")
            T_final, T_initial, cpu_times, integrator_history, T_history, Y_history = run_simulation_with_method(
                integrator_method='rl', 
                model_path=model_path, 
                super_steps=super_steps,
                t_fuel=t_fuel, t_oxidizer=t_oxidizer, total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, fuel_choice=fuel_choice, use_time_left=use_time_left
            )
            print(f"RL simulation completed successfully!")
            print(f"Total CPU time: {np.sum(cpu_times):.4f}s")
            print(f"Max temperature: {np.max(T_final):.1f}K")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in RL simulation: {e}")
            print("Running comparison without RL...")
            
            # Run comparison of baseline methods only
            methods = ['cvode', 'qss', 'heuristic']
            results = {}
            
            for method in methods:
                print(f"\nRunning {method.upper()}...")
                T_final, T_initial, cpu_times, integrator_history, T_history, Y_history = run_simulation_with_method(
                    integrator_method=method, t_fuel=t_fuel, t_oxidizer=t_oxidizer, total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                    npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, use_time_left=use_time_left
                )
                results[method] = {
                    'T_final': T_final,
                    'T_initial': T_initial,
                    'cpu_times': cpu_times,
                    'total_cpu_time': np.sum(cpu_times),
                    'max_temperature': np.max(T_final),
                    'T_history': T_history,
                    'Y_history': Y_history
                }
            
            create_comparison_plots(results)
    else:
        # Uncomment to run full comparison (requires trained model)
        results = compare_methods(model_path, t_fuel=t_fuel, t_oxidizer=t_oxidizer, total_time=total_time, pressure=pressure, center_width=center_width, slope_width=slope_width, x_left=x_left, x_right=x_right,
                              npoints=npoints, strain_rate=strain_rate, equilibrate_counterflow=equilibrate_counterflow, dt=dt, fuel_choice=fuel_choice, use_time_left=use_time_left)
