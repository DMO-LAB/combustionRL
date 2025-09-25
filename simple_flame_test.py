#!/usr/bin/env python3
"""
Simple counterflow diffusion flame test using CVODE only
This creates a basic methane-air diffusion flame with standard conditions
"""

from ember import Config, Paths, InitialCondition, StrainParameters, General, Times, \
    TerminationCondition, ConcreteConfig, CvodeTolerances, Chemistry, Debug, QssTolerances
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from ember import _ember
import cantera as ct
from tqdm import tqdm
from matplotlib.lines import Line2D
def run_simple_diffusion_flame(mechanism_file='gri30.yaml', fuel='CH4:1.0', oxidizer='O2:1.0, N2:3.76', tfuel=300, 
                               toxidizer=300, pressure=101325, x_left=-0.01, x_right=0.01, n_points=100, center_width=0.0005, 
                               slope_width=0.001, equilibrate_counterflow='HP', t_end=0.2, integrator='qss', plot=True, history_stride=10, strain_rate=300):
    """
    Run a simple counterflow diffusion flame simulation with methane and air
    """
    
    # Use standard test conditions
    output_dir = 'run/simple_diffusion_test'
    print(f"Running {integrator.upper()} simulation...")
    # Configuration with proper diffusion flame parameters
    conf = Config(
        Paths(outputDir=output_dir),
        
        # Use CVODE integrator throughout
        General(nThreads=1, chemistryIntegrator=integrator),
        
        # Use a simple, well-validated mechanism
        Chemistry(mechanismFile=mechanism_file),  # Standard GRI-Mech 3.0
        
        # Proper diffusion flame initial conditions
        InitialCondition(
            flameType='diffusion',
            
            # Standard ambient temperatures
            Tfuel=tfuel,      # Room temperature fuel
            Toxidizer=toxidizer,  # Room temperature oxidizer
            
            # Standard compositions
            fuel=fuel,                 # Pure methane
            oxidizer=oxidizer,     # Air (21% O2, 79% N2)
            
            # Standard pressure
            pressure=pressure,  # 1 atm
            
            # Reasonable domain size
            xLeft=x_left,      # 1 cm on fuel side
            xRight=x_right,      # 1 cm on oxidizer side
            nPoints=n_points,      # Good resolution
            
            # Narrow initial mixing region
            centerWidth=center_width,   # 0.5 mm mixing zone
            slopeWidth=slope_width,     # 1 mm transition
            
            # Don't equilibrate - let the flame develop naturally
            equilibrateCounterflow=equilibrate_counterflow
        ),
        
        # Moderate strain rate
        StrainParameters(initial=strain_rate, final=strain_rate),
        
        # Reasonable timesteps
        Times(
            globalTimestep=1e-5,        # 5 microsecond timestep
            profileStepInterval=100,    # Save profiles every 100 steps
            outputStepInterval=10       # Output data every 10 steps
        ),
        
        # Run to steady state
        TerminationCondition(abstol=0.0, dTdtTol=0, steadyPeriod=1.0,
                           tEnd=t_end, tolerance=0.0),
        
        # Standard CVODE tolerances
        CvodeTolerances(
            relativeTolerance=1e-6,
            momentumAbsTol=1e-8,
            energyAbsTol=1e-9,
            speciesAbsTol=1e-12
        ),
        
        QssTolerances(
            abstol=1e-8,
            dtmin=1e-16,
            dtmax=1e-6,
            epsmin=2e-3,
            epsmax=100,
            iterationCount=2,
            stabilityCheck=False
        ),
        
        # Minimal debug output
        Debug(veryVerbose=False, timesteps=True)
    )
    
    conf = ConcreteConfig(conf)
    
    # Create output directory
    if not os.path.isdir(conf.paths.outputDir):
        os.makedirs(conf.paths.outputDir, 0o0755)
    
    print("Starting simple diffusion flame simulation...")
    print(f"Domain: {conf.initialCondition.xLeft} to {conf.initialCondition.xRight} m")
    print(f"Fuel: {conf.initialCondition.fuel} at {conf.initialCondition.Tfuel} K")
    print(f"Oxidizer: {conf.initialCondition.oxidizer} at {conf.initialCondition.Toxidizer} K")
    print(f"Pressure: {conf.initialCondition.pressure/101325:.1f} atm")
    print(f"Strain rate: {conf.strainParameters.initial} s^-1")
    
    # Run simulation
    solver = _ember.FlameSolver(conf)
    solver.initialize()
    
    solver.set_integrator_types([integrator] * n_points)
    
    # Store initial conditions
    initial_T = solver.T.copy()
    print(f"Initial T: {initial_T.max():.1f}K")
    initial_x = solver.x.copy()
    
    start_time = time.time()
    done = False
    step_count = 0
    
    # Time-history recording
    T_history = []
    Y_history = []
    X_history = []
    time_history = []
    
    pbar = tqdm(total=int(t_end / solver.dt), desc=f'{integrator.upper()} Integration')
    while not done:
        done = solver.step()
        step_count += 1
        
        pbar.set_postfix({'Step': f'{step_count}', 'T': f'{solver.T.max():.1f}K', 'CPU time': f'{np.sum(solver.gridPointIntegrationTimes):.2f}s'})
        pbar.update(1)
        
        if step_count % 100 == 0:
            print(solver.x)
        
        if (step_count % max(1, history_stride)) == 0 or done:
            # Record snapshots
            T_history.append(solver.T.copy())
            Y_history.append(solver.Y.copy())
            X_history.append(solver.x.copy())
            time_history.append(step_count * solver.dt)
    
    end_time = time.time()
    solver.finalize()
    
    print(f"\nSimulation completed!")
    print(f"Total steps: {step_count}")
    # print(f"Final time: {solver.t:.6f} s")
    print(f"CPU time: {end_time - start_time:.2f} seconds")
    print(f"Max temperature: {np.max(solver.T):.1f} K")
    
    # Create plots
    if plot:
        create_flame_plots(solver, initial_T, initial_x, output_dir, mechanism_file, X_history)
    
    result = {
        'T_history': T_history,
        'Y_history': Y_history,
        'time_history': time_history,
        'X_history': X_history,
        'cpu_time_total': end_time - start_time
    }
    
    # save the result to a pickle file
    import pickle
    with open(os.path.join(output_dir, f'result_{integrator}.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return solver, initial_T, initial_x, result

def create_flame_plots(solver, initial_T, initial_x, output_dir, mechanism_file, X_history):
    """Create diagnostic plots for the flame"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = solver.x * 1000  # Convert to mm
    x_initial = initial_x * 1000
    # Temperature profile
    ax1 = axes[0, 0]
    ax1.plot(x_initial, initial_T, 'k--', label='Initial', linewidth=2)
    ax1.plot(x, solver.T, 'r-', label='Final', linewidth=2)
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Temperature Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Major species
    ax2 = axes[0, 1]
    gas = ct.Solution(mechanism_file)
    
    # Find key species indices
    try:
        i_CH4 = gas.species_index('CH4')
        i_O2 = gas.species_index('O2')
        i_H2O = gas.species_index('H2O')
        i_CO2 = gas.species_index('CO2')
        
        ax2.plot(x, solver.Y[i_CH4], 'b-', label='CH4', linewidth=2)
        ax2.plot(x, solver.Y[i_O2], 'g-', label='O2', linewidth=2)
        ax2.plot(x, solver.Y[i_H2O], 'c-', label='H2O', linewidth=2)
        ax2.plot(x, solver.Y[i_CO2], 'r-', label='CO2', linewidth=2)
    except ValueError as e:
        print(f"Warning: Could not find all species for plotting: {e}")
    
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Mass Fraction')
    ax2.set_title('Major Species')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Heat release rate
    ax3 = axes[1, 0]
    try:
        # Calculate heat release rate (approximate)
        hrr = solver.wdot * gas.molecular_weights  # mol/m³/s for each species
        # This is a simplified calculation - the actual HRR calculation is more complex
        ax3.plot(x, hrr.sum(axis=0), 'purple', linewidth=2)
        ax3.set_xlabel('Position (mm)')
        ax3.set_ylabel('Net reaction rate')
        ax3.set_title('Reaction Rate')
    except:
        ax3.text(0.5, 0.5, 'Heat release\ndata not available', 
                ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    
    # Velocity profile
    ax4 = axes[1, 1]
    ax4.plot(x, solver.V, 'orange', linewidth=2)
    ax4.set_xlabel('Position (mm)')
    ax4.set_ylabel('Mass flux (kg/m²/s)')
    ax4.set_title('Mass Flux Profile')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flame_profiles.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print flame location and other diagnostics
    flame_location = x[np.argmax(solver.T)]
    stagnation_point = x[np.argmin(np.abs(solver.V))]
    
    print(f"\nFlame diagnostics:")
    print(f"Flame location (max T): {flame_location:.2f} mm")
    print(f"Stagnation point: {stagnation_point:.2f} mm")
    print(f"Domain: {x[0]:.1f} to {x[-1]:.1f} mm")
    
    # Check if flame structure looks reasonable
    if flame_location < -5 or flame_location > 5:
        print("WARNING: Flame location seems unusual - check boundary conditions")
    
    if np.max(solver.T) < 1500:
        print("WARNING: Peak temperature is low - flame may not be properly established")
    elif np.max(solver.T) > 2500:
        print("WARNING: Peak temperature is very high - check for numerical issues")

def create_flame_plots_comparison(solvers_by_method, initial_T, initial_x, output_dir, mechanism_file):
    """Create comparison plots for multiple integrators against the initial state."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_initial = initial_x * 1000
    # Use x grid from the first solver
    first_solver = next(iter(solvers_by_method.values()))
    x = first_solver.x * 1000

    # Temperature profile
    ax1 = axes[0, 0]
    ax1.plot(x_initial, initial_T, 'k--', label='Initial', linewidth=2)
    colors = {
        'cvode': 'r',
        'qss': 'b'
    }
    labels = {
        'cvode': 'Final (CVODE)',
        'qss': 'Final (QSS)'
    }
    for method, solver in solvers_by_method.items():
        ax1.plot(x, solver.T, color=colors.get(method, 'gray'), label=labels.get(method, method.upper()), linewidth=2)
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Temperature Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Major species
    ax2 = axes[0, 1]
    gas = ct.Solution(mechanism_file)
    try:
        species_to_plot = ['CH4', 'O2', 'H2O', 'CO2']
        species_indices = {name: gas.species_index(name) for name in species_to_plot}
        for name, idx in species_indices.items():
            for method, solver in solvers_by_method.items():
                style = '-' if method == 'cvode' else '--'
                ax2.plot(x, solver.Y[idx], linestyle=style, label=f"{name} ({method.upper()})", linewidth=2)
    except ValueError as e:
        print(f"Warning: Could not find all species for plotting: {e}")
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Mass Fraction')
    ax2.set_title('Major Species')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Reaction rate indicator (sum of net production rates * MW)
    ax3 = axes[1, 0]
    try:
        for method, solver in solvers_by_method.items():
            hrr_like = solver.wdot * gas.molecular_weights
            style = '-' if method == 'cvode' else '--'
            ax3.plot(x, hrr_like.sum(axis=0), linestyle=style, label=method.upper(), linewidth=2)
        ax3.set_xlabel('Position (mm)')
        ax3.set_ylabel('Net reaction rate')
        ax3.set_title('Reaction Rate')
        ax3.legend()
    except Exception:
        ax3.text(0.5, 0.5, 'Heat release\ndata not available', 
                 ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    # Mass flux profile
    ax4 = axes[1, 1]
    for method, solver in solvers_by_method.items():
        style = '-' if method == 'cvode' else '--'
        ax4.plot(x, solver.V, linestyle=style, label=method.upper(), linewidth=2)
    ax4.set_xlabel('Position (mm)')
    ax4.set_ylabel('Mass flux (kg/m²/s)')
    ax4.set_title('Mass Flux Profile')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'flame_profiles_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def run_diffusion_flame_comparison(mechanism_file='gri30.yaml', fuel='CH4:1.0', oxidizer='O2:1.0, N2:3.76', tfuel=300, 
                                   toxidizer=300, pressure=101325, x_left=-0.01, x_right=0.01, n_points=100, center_width=0.0005, 
                                   slope_width=0.001, equilibrate_counterflow='HP', t_end=0.2, strain_rate=300):
    """Run both CVODE and QSS simulations and plot a comparison including the initial."""
    output_dir = 'run/simple_diffusion_test'
    solvers = {}
    results = {'cvode': None, 'qss': None}
    initial_T = None
    initial_x = None

    for integrator in ['cvode', 'qss']:
        solver, init_T, init_x, result = run_simple_diffusion_flame(
            mechanism_file, fuel, oxidizer, tfuel, toxidizer, pressure, x_left, x_right, n_points,
            center_width, slope_width, equilibrate_counterflow, t_end, integrator, plot=False, strain_rate=strain_rate
        )
        solvers[integrator] = solver
        results[integrator] = result
        if initial_T is None:
            initial_T = init_T
            initial_x = init_x

    create_flame_plots_comparison(solvers, initial_T, initial_x, output_dir, mechanism_file)
    # Also generate spatial profiles comparison plot (method vs CVODE)
    domain_length = abs(x_right - x_left)
    plot_spatial_profiles(results, method='qss', domain_length=domain_length, n_timesteps=6, mechanism_file=mechanism_file, output_dir=output_dir)
    return solvers

def plot_spatial_profiles(results, method='qss', domain_length=0.008, n_timesteps=6, mechanism_file='gri30.yaml', output_dir=None):
    """
    Plot spatial profiles comparing selected method with CVODE at different timesteps.

    results: dict with keys 'cvode' and method; each contains 'T_history' (list of [n_grid]),
             'Y_history' (list of [n_species, n_grid]).
    """
    if method not in results or results[method] is None:
        print(f"Results for method '{method}' not found")
        return
    if 'cvode' not in results or results['cvode'] is None:
        print("CVODE results not found for comparison")
        return

    T_history_method = np.array(results[method]['T_history'])
    Y_history_method = np.array(results[method]['Y_history'])
    T_history_cvode = np.array(results['cvode']['T_history'])
    Y_history_cvode = np.array(results['cvode']['Y_history'])

    n_time, n_grid = T_history_method.shape
    z = np.linspace(0, domain_length, n_grid)
    time_indices = np.linspace(0, n_time - 1, min(n_timesteps, n_time), dtype=int)

    gas = ct.Solution(mechanism_file)
    species_to_plot = ['OH', 'HO2', 'CO2', 'CH2O', 'CO']
    species_indices = []
    for spec in species_to_plot:
        try:
            species_indices.append(gas.species_index(spec))
        except ValueError:
            print(f"Warning: Species {spec} not found in mechanism")
            species_indices.append(None)

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    fig, axs = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle(f'Spatial Profiles Comparison: {method.upper()} vs CVODE', fontsize=22, fontweight='bold', y=0.98)

    subplot_data = [
        ('Temperature', 'T [K]', None, T_history_method, T_history_cvode),
        ('OH', 'Y_{OH} [-]', 0, None, None),
        ('HO2', 'Y_{HO2} [-]', 1, None, None),
        ('CO2', 'Y_{CO2} [-]', 2, None, None),
        ('CH2O', 'Y_{CH2O} [-]', 3, None, None),
        ('CO', 'Y_{CO} [-]', 4, None, None)
    ]

    for plot_idx, (title, ylabel, species_idx, T_method, T_cvode) in enumerate(subplot_data):
        row = plot_idx // 2
        col = plot_idx % 2
        ax = axs[row, col]

        for i, time_idx in enumerate(time_indices):
            if title == 'Temperature':
                profile_method = T_method[time_idx, :]
                profile_cvode = T_cvode[time_idx, :]
            else:
                if species_indices[species_idx] is None:
                    continue
                profile_method = Y_history_method[time_idx, species_indices[species_idx], :]
                profile_cvode = Y_history_cvode[time_idx, species_indices[species_idx], :]

            ax.plot(z * 1000, profile_method, color=colors[i], linewidth=3.5, linestyle='-', label=f'{method.upper()} t={time_idx}', alpha=1.0)
            ax.plot(z * 1000, profile_cvode, color='black', linewidth=2.5, linestyle='--', alpha=0.8)

        ax.set_xlabel('Z [mm]', fontweight='bold', fontsize=16)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=16)
        ax.set_title(title, fontweight='bold', fontsize=18)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(width=2.0, length=6, labelsize=14)

        if title != 'Temperature' and species_indices[species_idx] is not None:
            max_val_method = np.max(Y_history_method[:, species_indices[species_idx], :])
            max_val_cvode = np.max(Y_history_cvode[:, species_indices[species_idx], :])
            if max(max_val_method, max_val_cvode) < 1e-2:
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-8)

    legend_elements = []
    for i, time_idx in enumerate(time_indices):
        legend_elements.append(
            Line2D([0], [0], color=colors[i], linewidth=3.5, linestyle='-', label=f't = {time_idx}')
        )
    legend_elements.extend([
        Line2D([0], [0], color='black', linewidth=3.5, linestyle='-', label=f'{method.upper()} (solid)'),
        Line2D([0], [0], color='black', linewidth=2.5, linestyle='--', label='CVODE (dashed)', alpha=0.8)
    ])

    fig.legend(handles=legend_elements, bbox_to_anchor=(1.04, 0.5), loc='center left', frameon=True, fancybox=True, shadow=True, fontsize=14, title='Legend', title_fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, right=0.85, hspace=0.4, wspace=0.3)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'spatial_profiles_{method}_vs_cvode.pdf'), format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f'spatial_profiles_{method}_vs_cvode.png'), format='png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(f'spatial_profiles_{method}_vs_cvode.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'spatial_profiles_{method}_vs_cvode.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, axs

if __name__ == "__main__":
    try:
        mechanism_file = 'large_mechanism/n-dodecane.yaml'
        fuel = 'nc12h26:1.0'
        oxidizer = 'O2:1.0, N2:3.76'
        pressure = 101325 * 60
        tfuel = 300
        toxidizer = 800
        x_left = -0.01
        x_right = 0.01
        n_points = 100
        center_width = 0.0005
        slope_width = 0.001
        strain_rate = 300
        equilibrate_counterflow = False
        t_end = 0.01
        _ = run_diffusion_flame_comparison(
            mechanism_file, fuel, oxidizer, tfuel, toxidizer, pressure, x_left, x_right, n_points,
            center_width, slope_width, equilibrate_counterflow, t_end, strain_rate=strain_rate
        )
        print("\nComparison test completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()