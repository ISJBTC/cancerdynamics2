#!/usr/bin/env python3
"""
Quantum Visualization Runner - Integration with Cancer Dynamics Framework

This script shows how to integrate the quantum visualization module with your
existing cancer dynamics research framework.
"""

import numpy as np
import os
import sys
from scipy.integrate import odeint

# Import your existing framework components
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from config.parameters import get_config, get_quantum_parameters, get_quantum_initial_conditions

# First, save the quantum visualization module
def setup_quantum_visualization():
    """Setup the quantum visualization module"""
    
    # Create the visualization directory
    viz_dir = 'visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    # The quantum visualization code goes in visualization/quantum_plots.py
    quantum_viz_code = '''import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

class QuantumVisualization:
    """Enhanced visualization for quantum effects in cancer dynamics"""
    
    def __init__(self, output_dir='quantum_plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Quantum color scheme
        self.quantum_colors = {
            'active': '#FF6B6B',      # Red for active quantum effects
            'inactive': '#95E1D3',    # Green for inactive
            'threshold': '#FFD93D',   # Yellow for threshold zone
            'pressure': '#6BCF7F',    # Light green for pressure
            'tunneling': '#4ECDC4'    # Teal for tunneling
        }
        
        # Create custom colormap for quantum intensity
        self.quantum_cmap = LinearSegmentedColormap.from_list(
            'quantum', ['#95E1D3', '#FFD93D', '#FF6B6B']
        )
    
    def plot_quantum_pressure_dynamics(self, t, trajectories, quantum_threshold=1e-5, 
                                     p=0.1, q=1.0, save=True):
        """Plot quantum pressure dynamics over time with threshold visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        T_traj = trajectories[:, 0]  # Tumor trajectory
        I_traj = trajectories[:, 1]  # Immune trajectory
        
        # Calculate quantum pressures
        Q_T = np.array([-(p**2 * q)/(2*T) if T > quantum_threshold else 0.0 for T in T_traj])
        Q_I = np.array([-(p**2 * q)/(2*I) if I > quantum_threshold else 0.0 for I in I_traj])
        
        # Plot 1: Tumor population with quantum zones
        ax1.plot(t, T_traj, 'b-', linewidth=2, label='Tumor Population')
        ax1.axhline(y=quantum_threshold, color=self.quantum_colors['threshold'], 
                   linestyle='--', linewidth=2, label='Quantum Threshold')
        
        # Highlight quantum active zones
        quantum_active_T = T_traj <= quantum_threshold
        if np.any(quantum_active_T):
            ax1.fill_between(t, 0, np.max(T_traj), where=quantum_active_T, 
                           alpha=0.3, color=self.quantum_colors['active'], 
                           label='Quantum Active Zone')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Tumor Population')
        ax1.set_title('Tumor Dynamics with Quantum Effects')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if np.min(T_traj) > 0:
            ax1.set_yscale('log')
        
        # Plot 2: Immune population with quantum zones
        ax2.plot(t, I_traj, 'r-', linewidth=2, label='Immune Population')
        ax2.axhline(y=quantum_threshold, color=self.quantum_colors['threshold'], 
                   linestyle='--', linewidth=2, label='Quantum Threshold')
        
        quantum_active_I = I_traj <= quantum_threshold
        if np.any(quantum_active_I):
            ax2.fill_between(t, 0, np.max(I_traj), where=quantum_active_I, 
                           alpha=0.3, color=self.quantum_colors['active'], 
                           label='Quantum Active Zone')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Immune Population')
        ax2.set_title('Immune Dynamics with Quantum Effects')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if np.min(I_traj) > 0:
            ax2.set_yscale('log')
        
        # Plot 3: Quantum pressure for tumor
        ax3.plot(t, Q_T, color=self.quantum_colors['pressure'], linewidth=2, 
                label='Tumor Quantum Pressure Q_τ')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Quantum Pressure Q_τ')
        ax3.set_title('Tumor Quantum Pressure Dynamics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quantum pressure for immune
        ax4.plot(t, Q_I, color=self.quantum_colors['pressure'], linewidth=2, 
                label='Immune Quantum Pressure Q_i')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Quantum Pressure Q_i')
        ax4.set_title('Immune Quantum Pressure Dynamics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'quantum_pressure_dynamics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_quantum_phase_space(self, T_traj, I_traj, quantum_threshold=1e-5, save=True):
        """Plot phase space with quantum zones highlighted"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create quantum intensity map
        quantum_intensity = np.zeros_like(T_traj)
        for i in range(len(T_traj)):
            if T_traj[i] <= quantum_threshold and I_traj[i] <= quantum_threshold:
                quantum_intensity[i] = 1.0  # Both quantum active
            elif T_traj[i] <= quantum_threshold or I_traj[i] <= quantum_threshold:
                quantum_intensity[i] = 0.5  # One quantum active
            else:
                quantum_intensity[i] = 0.0  # No quantum effects
        
        # Plot trajectory with quantum coloring
        scatter = ax.scatter(T_traj, I_traj, c=quantum_intensity, cmap=self.quantum_cmap, 
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add quantum threshold lines
        ax.axvline(x=quantum_threshold, color=self.quantum_colors['threshold'], 
                  linestyle='--', linewidth=2, label=f'Quantum Threshold T = {quantum_threshold:.0e}')
        ax.axhline(y=quantum_threshold, color=self.quantum_colors['threshold'], 
                  linestyle='--', linewidth=2, label=f'Quantum Threshold I = {quantum_threshold:.0e}')
        
        # Highlight quantum exclusion zone
        quantum_zone = patches.Rectangle((0, 0), quantum_threshold, quantum_threshold, 
                                       linewidth=2, edgecolor=self.quantum_colors['active'], 
                                       facecolor=self.quantum_colors['active'], alpha=0.2,
                                       label='Quantum Exclusion Zone')
        ax.add_patch(quantum_zone)
        
        # Mark start and end points
        ax.scatter(T_traj[0], I_traj[0], color='green', s=100, marker='o', 
                  zorder=5, label='Start')
        ax.scatter(T_traj[-1], I_traj[-1], color='red', s=100, marker='x', 
                  zorder=5, label='End')
        
        # Formatting
        ax.set_xlabel('Tumor Population (T)')
        ax.set_ylabel('Immune Population (I)')
        ax.set_title('Quantum Phase Space - Tumor vs Immune Dynamics')
        if np.min(T_traj) > 0:
            ax.set_xscale('log')
        if np.min(I_traj) > 0:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar for quantum intensity
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Quantum Effect Intensity')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Inactive', 'Partial', 'Full'])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'quantum_phase_space.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_quantum_vs_classical_comparison(self, t, classical_traj, quantum_traj, 
                                           quantum_threshold=1e-5, save=True):
        """Compare classical vs quantum dynamics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        cell_labels = ['Tumor', 'Immune', 'Memory', 'Suppressor']
        
        for i, (ax, cell_label) in enumerate(zip(axes, cell_labels)):
            # Plot both trajectories
            ax.plot(t, classical_traj[:, i], 'b-', linewidth=2, 
                   label='Classical Model', alpha=0.8)
            ax.plot(t, quantum_traj[:, i], 'r-', linewidth=2, 
                   label='Quantum Model', alpha=0.8)
            
            # Add quantum threshold for tumor and immune
            if i < 2:  # Tumor and Immune
                ax.axhline(y=quantum_threshold, color=self.quantum_colors['threshold'], 
                          linestyle='--', linewidth=1, alpha=0.7, 
                          label='Quantum Threshold')
                
                # Highlight quantum active zones
                quantum_active = quantum_traj[:, i] <= quantum_threshold
                if np.any(quantum_active):
                    y_max = max(np.max(classical_traj[:, i]), np.max(quantum_traj[:, i]))
                    ax.fill_between(t, 0, y_max, where=quantum_active, 
                                   alpha=0.2, color=self.quantum_colors['active'])
            
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{cell_label} Population')
            ax.set_title(f'{cell_label} Cell Dynamics')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use log scale for tumor and immune if needed
            if i < 2 and np.min(quantum_traj[:, i]) > 0 and np.min(quantum_traj[:, i]) < 1e-3:
                ax.set_yscale('log')
        
        plt.suptitle('Classical vs Quantum-Enhanced Cancer Dynamics', fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'quantum_vs_classical.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
'''
    
    # Write the quantum visualization module
    with open(os.path.join(viz_dir, 'quantum_plots.py'), 'w', encoding='utf-8') as f:
        f.write(quantum_viz_code)
    
    print("✅ Quantum visualization module created at: visualization/quantum_plots.py")


def run_quantum_visualization_demo():
    """Run a demonstration of quantum visualization with your cancer models"""
    
    print("🔬 QUANTUM VISUALIZATION DEMO")
    print("=" * 50)
    
    # Get configuration and quantum parameters
    config = get_config()
    quantum_params = get_quantum_parameters()
    
    print(f"📊 Quantum Parameters:")
    print(f"   Momentum (p): {quantum_params['p']}")
    print(f"   Threshold: {quantum_params['quantum_threshold']:.0e}")
    print(f"   Parameter q: {quantum_params['q']}")
    
    # Create models
    print(f"\n🔄 Creating quantum-enhanced models...")
    integer_model = IntegerModel()
    fractional_model = FractionalModel()
    
    # Import quantum visualization
    try:
        from visualization.quantum_plots import QuantumVisualization
        print("✅ Quantum visualization module imported successfully")
    except ImportError:
        print("❌ Setting up quantum visualization module first...")
        setup_quantum_visualization()
        from visualization.quantum_plots import QuantumVisualization
        print("✅ Quantum visualization module created and imported")
    
    # Create visualization instance
    viz = QuantumVisualization('quantum_demo_output')
    
    # Set up simulation parameters
    t = np.linspace(0, 6, 61)  # Extended time for quantum effects
    
    # Test different initial conditions
    test_conditions = [
        [50, 10, 20, 30],        # Normal condition
        [1e-4, 1e-4, 20, 30],    # Quantum condition (both below threshold)
        [50, 1e-4, 20, 30],      # Immune quantum active
        [1e-4, 10, 20, 30],      # Tumor quantum active
    ]
    
    print(f"\n🧪 Testing {len(test_conditions)} different scenarios...")
    
    for i, init_cond in enumerate(test_conditions):
        print(f"\n📊 Scenario {i+1}: T={init_cond[0]:.0e}, I={init_cond[1]:.0e}")
        
        # Check quantum status
        quantum_status = integer_model.get_quantum_status(init_cond)
        print(f"   Quantum status: Tumor={quantum_status['tumor_quantum_active']}, "
              f"Immune={quantum_status['immune_quantum_active']}")
        
        # Run simulations
        print("   Running simulations...")
        
        # Integer model (with quantum effects)
        int_traj = odeint(integer_model, init_cond, t)
        
        # Fractional model (with quantum effects + memory)
        fractional_model.reset_history()
        frac_traj = odeint(fractional_model, init_cond, t)
        
        # Create "classical" version (without quantum effects) for comparison
        # This is simulated by using the same model but modifying quantum threshold
        original_threshold = integer_model.quantum_threshold
        integer_model.quantum_threshold = 0  # Disable quantum effects
        classical_traj = odeint(integer_model, init_cond, t)
        integer_model.quantum_threshold = original_threshold  # Restore
        
        # Create visualizations for this scenario
        scenario_dir = f'quantum_demo_output/scenario_{i+1}'
        os.makedirs(scenario_dir, exist_ok=True)
        
        viz_scenario = QuantumVisualization(scenario_dir)
        
        print("   Creating quantum pressure visualization...")
        viz_scenario.plot_quantum_pressure_dynamics(
            t, int_traj, 
            quantum_threshold=quantum_params['quantum_threshold'],
            p=quantum_params['p'],
            q=quantum_params['q']
        )
        
        print("   Creating quantum phase space...")
        viz_scenario.plot_quantum_phase_space(
            int_traj[:, 0], int_traj[:, 1],
            quantum_threshold=quantum_params['quantum_threshold']
        )
        
        print("   Creating classical vs quantum comparison...")
        viz_scenario.plot_quantum_vs_classical_comparison(
            t, classical_traj, int_traj,
            quantum_threshold=quantum_params['quantum_threshold']
        )
        
        # Print final populations
        print(f"   Final populations (Quantum Model):")
        print(f"     Tumor: {int_traj[-1, 0]:.6f}")
        print(f"     Immune: {int_traj[-1, 1]:.6f}")
        print(f"     Memory: {int_traj[-1, 2]:.6f}")
        print(f"     Suppressor: {int_traj[-1, 3]:.6f}")
        
        # Check if quantum effects were active during simulation
        quantum_events = {
            'tumor_quantum_times': np.sum(int_traj[:, 0] <= quantum_params['quantum_threshold']),
            'immune_quantum_times': np.sum(int_traj[:, 1] <= quantum_params['quantum_threshold'])
        }
        
        print(f"   Quantum events during simulation:")
        print(f"     Tumor quantum active: {quantum_events['tumor_quantum_times']}/{len(t)} time points")
        print(f"     Immune quantum active: {quantum_events['immune_quantum_times']}/{len(t)} time points")
    
    print(f"\n🎉 Quantum visualization demo completed!")
    print(f"📂 Results saved in: quantum_demo_output/")
    print(f"📊 Individual scenario results in: quantum_demo_output/scenario_X/")
    
    return viz


def run_quantum_parameter_study():
    """Run a parameter study to show quantum effects"""
    
    print("\n🔬 QUANTUM PARAMETER SENSITIVITY STUDY")
    print("=" * 50)
    
    # Import quantum visualization
    try:
        from visualization.quantum_plots import QuantumVisualization
    except ImportError:
        setup_quantum_visualization()
        from visualization.quantum_plots import QuantumVisualization
    
    viz = QuantumVisualization('quantum_param_study')
    
    # Test different quantum momentum values
    p_values = np.linspace(0.05, 0.3, 6)
    quantum_threshold = 1e-5
    
    print(f"Testing {len(p_values)} momentum parameter values...")
    
    # Create model
    model = IntegerModel()
    
    # Test condition that will trigger quantum effects
    init_cond = [1e-4, 1e-4, 20, 30]  # Both tumor and immune near quantum threshold
    t = np.linspace(0, 4, 41)
    
    outcomes = []
    
    for p in p_values:
        print(f"   Testing p = {p:.3f}")
        
        # Update quantum parameter
        model.p = p
        
        # Run simulation
        traj = odeint(model, init_cond, t)
        outcomes.append(traj[-1])  # Store final state
        
        print(f"     Final tumor: {traj[-1, 0]:.6f}")
        print(f"     Final immune: {traj[-1, 1]:.6f}")
    
    # Create parameter sensitivity plot
    print("\nCreating parameter sensitivity visualization...")
    viz.plot_quantum_parameter_sensitivity(p_values, outcomes, quantum_threshold)
    
    print(f"✅ Quantum parameter study completed!")
    print(f"📂 Results saved in: quantum_param_study/")
    
    return outcomes


def integrate_with_main_framework():
    """Show how to integrate quantum visualization with your main research framework"""
    
    print("\n🔗 INTEGRATION WITH MAIN FRAMEWORK")
    print("=" * 50)
    
    # Example of how to add quantum visualization to your existing analysis
    integration_code = '''
# Add this to your main.py or any analysis script:

from visualization.quantum_plots import QuantumVisualization

class CancerDynamicsResearch:
    def __init__(self, output_dir='research_output'):
        # ... existing initialization ...
        
        # Add quantum visualization
        self.quantum_viz = QuantumVisualization(
            os.path.join(output_dir, 'quantum_analysis')
        )
    
    def run_basic_dynamics_analysis(self):
        # ... existing analysis ...
        
        # Add quantum-specific visualization
        for i, (int_traj, frac_traj) in enumerate(zip(integer_trajectories, fractional_trajectories)):
            
            # Create quantum pressure plots
            self.quantum_viz.plot_quantum_pressure_dynamics(
                t, int_traj, 
                quantum_threshold=self.config.model_params['quantum_threshold'],
                p=self.config.model_params['p'],
                q=self.config.model_params['q']
            )
            
            # Create quantum phase space
            self.quantum_viz.plot_quantum_phase_space(
                int_traj[:, 0], int_traj[:, 1]
            )
            
            # Compare with fractional model
            self.quantum_viz.plot_quantum_vs_classical_comparison(
                t, int_traj, frac_traj
            )
        
        return results
'''
    
    print("Integration example:")
    print(integration_code)
    
    # Save integration example
    with open('quantum_integration_example.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("✅ Integration example saved as: quantum_integration_example.py")


def main():
    """Main function to run quantum visualization demos"""
    
    print("🚀 QUANTUM VISUALIZATION FOR CANCER DYNAMICS")
    print("=" * 60)
    print("This demo shows how to visualize quantum effects in your cancer models")
    print("=" * 60)
    
    try:
        # Step 1: Setup the quantum visualization module
        print("\n📦 Step 1: Setting up quantum visualization module...")
        setup_quantum_visualization()
        
        # Step 2: Run the main demo
        print("\n🎨 Step 2: Running quantum visualization demo...")
        viz = run_quantum_visualization_demo()
        
        # Step 3: Parameter sensitivity study
        print("\n📊 Step 3: Running parameter sensitivity study...")
        run_quantum_parameter_study()
        
        # Step 4: Show integration example
        print("\n🔗 Step 4: Integration with main framework...")
        integrate_with_main_framework()
        
        print(f"\n" + "🎉" * 20)
        print("QUANTUM VISUALIZATION DEMO COMPLETED!")
        print("🎉" * 20)
        
        print(f"\n📂 Check these directories for results:")
        print(f"   • quantum_demo_output/ - Main demo results")
        print(f"   • quantum_param_study/ - Parameter sensitivity")
        print(f"   • visualization/quantum_plots.py - Visualization module")
        print(f"   • quantum_integration_example.py - Integration guide")
        
    except Exception as e:
        print(f"\n❌ Error running quantum visualization demo: {e}")
        print("Make sure you have all required dependencies installed:")
        print("   pip install numpy matplotlib scipy")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 Ready to visualize quantum effects!")
        print(f"💡 Next steps:")
        print(f"   1. Run: python quantum_visualization_runner.py")
        print(f"   2. Integrate with your existing analysis scripts")
        print(f"   3. Customize visualizations for your specific research needs")
    else:
        print(f"\n🔧 Please resolve any issues and try again.")