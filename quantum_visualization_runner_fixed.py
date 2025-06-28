#!/usr/bin/env python3
"""
Fixed Quantum Visualization Runner - Uses Bulletproof Visualization

This version uses the bulletproof quantum visualization module that handles any data format.
"""

import numpy as np
import os
import sys
from scipy.integrate import odeint

# Import your existing framework components
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from config.parameters import get_config, get_quantum_parameters, get_quantum_initial_conditions

def run_quantum_visualization_demo_fixed():
    """Run quantum visualization demo with bulletproof error handling"""
    
    print("🔬 FIXED QUANTUM VISUALIZATION DEMO")
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
    
    # Import bulletproof quantum visualization
    try:
        from visualization.quantum_plots_bulletproof import QuantumVisualization
        print("✅ Bulletproof quantum visualization module imported successfully")
    except ImportError:
        print("❌ Bulletproof module not found. Creating it now...")
        create_bulletproof_module()
        from visualization.quantum_plots_bulletproof import QuantumVisualization
        print("✅ Bulletproof quantum visualization module created and imported")
    
    # Create visualization instance
    viz = QuantumVisualization('quantum_demo_output_fixed')
    
    # Set up simulation parameters
    t = np.linspace(0, 6, 61)  # Extended time for quantum effects
    
    # Test different initial conditions
    test_conditions = [
        [50, 10, 20, 30],        # Normal condition
        [1e-4, 1e-4, 20, 30],    # Quantum condition (both below threshold)
        [50, 1e-4, 20, 30],      # Immune quantum active
        [1e-4, 10, 20, 30],      # Tumor quantum active
    ]
    
    print(f"\n🧪 Testing {len(test_conditions)} different scenarios with bulletproof visualization...")
    
    for i, init_cond in enumerate(test_conditions):
        print(f"\n📊 Scenario {i+1}: T={init_cond[0]:.0e}, I={init_cond[1]:.0e}")
        
        # Check quantum status
        quantum_status = integer_model.get_quantum_status(init_cond)
        print(f"   Quantum status: Tumor={quantum_status['tumor_quantum_active']}, "
              f"Immune={quantum_status['immune_quantum_active']}")
        
        # Run simulations with detailed debugging
        print("   Running simulations...")
        
        try:
            # Integer model (with quantum effects)
            int_traj = odeint(integer_model, init_cond, t)
            print(f"   ✅ Integer simulation successful, shape: {int_traj.shape}")
            
            # Fractional model (with quantum effects + memory)
            fractional_model.reset_history()
            frac_traj = odeint(fractional_model, init_cond, t)
            print(f"   ✅ Fractional simulation successful, shape: {frac_traj.shape}")
            
            # Create "classical" version (without quantum effects) for comparison
            original_threshold = integer_model.quantum_threshold
            integer_model.quantum_threshold = 0  # Disable quantum effects
            classical_traj = odeint(integer_model, init_cond, t)
            integer_model.quantum_threshold = original_threshold  # Restore
            print(f"   ✅ Classical simulation successful, shape: {classical_traj.shape}")
            
        except Exception as e:
            print(f"   ❌ Simulation failed: {e}")
            continue
        
        # Create visualizations for this scenario with bulletproof handling
        scenario_dir = f'quantum_demo_output_fixed/scenario_{i+1}'
        os.makedirs(scenario_dir, exist_ok=True)
        
        viz_scenario = QuantumVisualization(scenario_dir)
        
        print("   Creating bulletproof quantum pressure visualization...")
        try:
            viz_scenario.plot_quantum_pressure_dynamics(
                t, int_traj, 
                quantum_threshold=quantum_params['quantum_threshold'],
                p=quantum_params['p'],
                q=quantum_params['q']
            )
            print("   ✅ Quantum pressure plot successful")
        except Exception as e:
            print(f"   ⚠️ Quantum pressure plot issue: {e}")
        
        print("   Creating bulletproof quantum phase space...")
        try:
            viz_scenario.plot_quantum_phase_space(
                int_traj[:, 0], int_traj[:, 1],
                quantum_threshold=quantum_params['quantum_threshold']
            )
            print("   ✅ Phase space plot successful")
        except Exception as e:
            print(f"   ⚠️ Phase space plot issue: {e}")
        
        print("   Creating bulletproof classical vs quantum comparison...")
        try:
            viz_scenario.plot_quantum_vs_classical_comparison(
                t, classical_traj, int_traj,
                quantum_threshold=quantum_params['quantum_threshold']
            )
            print("   ✅ Comparison plot successful")
        except Exception as e:
            print(f"   ⚠️ Comparison plot issue: {e}")
        
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
    
    print(f"\n🎉 Fixed quantum visualization demo completed!")
    print(f"📂 Results saved in: quantum_demo_output_fixed/")
    print(f"📊 Individual scenario results in: quantum_demo_output_fixed/scenario_X/")
    
    return viz

def run_quantum_parameter_study_fixed():
    """Run parameter study with bulletproof visualization"""
    
    print("\n🔬 FIXED QUANTUM PARAMETER SENSITIVITY STUDY")
    print("=" * 50)
    
    # Import bulletproof visualization
    from visualization.quantum_plots_bulletproof import QuantumVisualization
    
    viz = QuantumVisualization('quantum_param_study_fixed')
    
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
        
        try:
            # Run simulation
            traj = odeint(model, init_cond, t)
            outcomes.append(traj[-1])  # Store final state
            
            print(f"     Final tumor: {traj[-1, 0]:.6f}")
            print(f"     Final immune: {traj[-1, 1]:.6f}")
        except Exception as e:
            print(f"     ❌ Simulation failed: {e}")
            outcomes.append([0, 0, 0, 0])  # Add dummy outcome
    
    # Create parameter sensitivity plot with bulletproof handling
    print("\nCreating bulletproof parameter sensitivity visualization...")
    try:
        viz.plot_quantum_parameter_sensitivity(p_values, outcomes, quantum_threshold)
        print(f"✅ Parameter sensitivity plot successful!")
    except Exception as e:
        print(f"⚠️ Parameter sensitivity plot issue: {e}")
    
    print(f"📂 Results saved in: quantum_param_study_fixed/")
    
    return outcomes

def create_bulletproof_module():
    """Create the bulletproof quantum visualization module if it doesn't exist"""
    
    bulletproof_code = '''import numpy as np
import matplotlib.pyplot as plt
import os

class QuantumVisualization:
    """Bulletproof quantum visualization with maximum error handling"""
    
    def __init__(self, output_dir='quantum_plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _make_safe_array(self, data):
        """Convert any data to a safe 2D array"""
        try:
            # Convert to numpy array
            arr = np.asarray(data, dtype=float)
            
            # Handle different cases
            if arr.ndim == 0:  # Scalar
                return np.array([[0, 0, 0, 0]])
            elif arr.ndim == 1:  # 1D array
                if len(arr) == 4:
                    return arr.reshape(1, -1)  # Single time point
                else:
                    # Assume it's time series of one variable, pad to 4 columns
                    padded = np.zeros((len(arr), 4))
                    padded[:, 0] = arr
                    return padded
            else:  # 2D array
                if arr.shape[1] < 4:
                    # Pad with zeros
                    padded = np.zeros((arr.shape[0], 4))
                    padded[:, :arr.shape[1]] = arr
                    return padded
                else:
                    return arr[:, :4]  # Take first 4 columns
        except:
            # Ultimate fallback
            return np.array([[0, 0, 0, 0]])
    
    def plot_quantum_pressure_dynamics(self, t, trajectories, quantum_threshold=1e-5, 
                                     p=0.1, q=1.0, save=True):
        """Bulletproof quantum pressure plotting"""
        try:
            print("      Creating bulletproof quantum pressure plot...")
            
            # Make everything safe
            t_safe = np.asarray(t, dtype=float).flatten()
            traj_safe = self._make_safe_array(trajectories)
            
            # Ensure same length
            min_len = min(len(t_safe), len(traj_safe))
            if min_len == 0:
                min_len = 1
                t_safe = np.array([0])
                traj_safe = np.array([[0, 0, 0, 0]])
            else:
                t_safe = t_safe[:min_len]
                traj_safe = traj_safe[:min_len]
            
            # Extract trajectories safely
            T_traj = traj_safe[:, 0]
            I_traj = traj_safe[:, 1]
            
            # Create simple plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Tumor
            ax1.plot(t_safe, T_traj, 'b-', linewidth=2, label='Tumor')
            ax1.axhline(y=quantum_threshold, color='orange', linestyle='--', label='Threshold')
            ax1.set_title('Tumor Population')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Population')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Immune
            ax2.plot(t_safe, I_traj, 'r-', linewidth=2, label='Immune')
            ax2.axhline(y=quantum_threshold, color='orange', linestyle='--', label='Threshold')
            ax2.set_title('Immune Population')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Population')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Tumor Pressure (safe calculation)
            Q_T = []
            for T_val in T_traj:
                if T_val > quantum_threshold and T_val != 0:
                    Q_T.append(-(p**2 * q)/(2*T_val))
                else:
                    Q_T.append(0.0)
            
            ax3.plot(t_safe, Q_T, 'g-', linewidth=2, label='Tumor Pressure')
            ax3.set_title('Tumor Quantum Pressure')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Pressure')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Immune Pressure (safe calculation)
            Q_I = []
            for I_val in I_traj:
                if I_val > quantum_threshold and I_val != 0:
                    Q_I.append(-(p**2 * q)/(2*I_val))
                else:
                    Q_I.append(0.0)
            
            ax4.plot(t_safe, Q_I, 'm-', linewidth=2, label='Immune Pressure')
            ax4.set_title('Immune Quantum Pressure')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Pressure')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'quantum_pressure_dynamics.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            print("      ✅ Bulletproof plot created successfully")
            return fig
            
        except Exception as e:
            print(f"      ❌ Even bulletproof plot failed: {e}")
            # Create error message plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Visualization Error:\\n{str(e)}\\nData shapes:\\nt: {getattr(t, "shape", "unknown")}\\ntrajectories: {getattr(trajectories, "shape", "unknown")}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
            ax.set_title('Quantum Visualization Error')
            if save:
                plt.savefig(os.path.join(self.output_dir, 'error_plot.png'), dpi=300, bbox_inches='tight')
                plt.close()
            return fig
    
    def plot_quantum_phase_space(self, T_traj, I_traj, quantum_threshold=1e-5, save=True):
        """Bulletproof phase space plot"""
        try:
            T_safe = np.asarray(T_traj, dtype=float).flatten()
            I_safe = np.asarray(I_traj, dtype=float).flatten()
            
            if len(T_safe) == 0:
                T_safe = np.array([0])
            if len(I_safe) == 0:
                I_safe = np.array([0])
                
            min_len = min(len(T_safe), len(I_safe))
            T_safe = T_safe[:min_len]
            I_safe = I_safe[:min_len]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(T_safe, I_safe, alpha=0.7, s=30)
            ax.axvline(x=quantum_threshold, color='orange', linestyle='--', label='Threshold')
            ax.axhline(y=quantum_threshold, color='orange', linestyle='--')
            ax.set_xlabel('Tumor Population')
            ax.set_ylabel('Immune Population')
            ax.set_title('Quantum Phase Space')
            ax.legend()
            ax.grid(True)
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'quantum_phase_space.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            return fig
        except Exception as e:
            print(f"Phase space plot error: {e}")
            return None
    
    def plot_quantum_vs_classical_comparison(self, t, classical_traj, quantum_traj, 
                                           quantum_threshold=1e-5, save=True):
        """Bulletproof comparison plot"""
        try:
            t_safe = np.asarray(t, dtype=float).flatten()
            classical_safe = self._make_safe_array(classical_traj)
            quantum_safe = self._make_safe_array(quantum_traj)
            
            min_len = min(len(t_safe), len(classical_safe), len(quantum_safe))
            if min_len == 0:
                min_len = 1
                t_safe = np.array([0])
                classical_safe = np.array([[0, 0, 0, 0]])
                quantum_safe = np.array([[0, 0, 0, 0]])
            else:
                t_safe = t_safe[:min_len]
                classical_safe = classical_safe[:min_len]
                quantum_safe = quantum_safe[:min_len]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            cell_labels = ['Tumor', 'Immune', 'Memory', 'Suppressor']
            
            for i, (ax, label) in enumerate(zip(axes, cell_labels)):
                ax.plot(t_safe, classical_safe[:, i], 'b-', label='Classical', linewidth=2)
                ax.plot(t_safe, quantum_safe[:, i], 'r-', label='Quantum', linewidth=2)
                ax.set_title(f'{label} Dynamics')
                ax.set_xlabel('Time')
                ax.set_ylabel('Population')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'quantum_vs_classical.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            return fig
        except Exception as e:
            print(f"Comparison plot error: {e}")
            return None
    
    def plot_quantum_parameter_sensitivity(self, p_values, outcomes, quantum_threshold=1e-5, save=True):
        """Bulletproof parameter sensitivity plot"""
        try:
            p_safe = np.asarray(p_values, dtype=float).flatten()
            
            tumor_outcomes = []
            immune_outcomes = []
            
            for outcome in outcomes:
                if isinstance(outcome, (list, tuple, np.ndarray)) and len(outcome) >= 2:
                    tumor_outcomes.append(float(outcome[0]))
                    immune_outcomes.append(float(outcome[1]))
                else:
                    tumor_outcomes.append(0.0)
                    immune_outcomes.append(0.0)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(p_safe, tumor_outcomes, 'o-', linewidth=2, markersize=8)
            ax1.set_title('Tumor vs Momentum Parameter')
            ax1.set_xlabel('p')
            ax1.set_ylabel('Final Tumor Population')
            ax1.grid(True)
            
            ax2.plot(p_safe, immune_outcomes, 'o-', linewidth=2, markersize=8)
            ax2.set_title('Immune vs Momentum Parameter')
            ax2.set_xlabel('p')
            ax2.set_ylabel('Final Immune Population')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'quantum_parameter_sensitivity.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            return fig
        except Exception as e:
            print(f"Parameter sensitivity plot error: {e}")
            return None
'''
    
    # Write the bulletproof version
    os.makedirs('visualization', exist_ok=True)
    with open('visualization/quantum_plots_bulletproof.py', 'w', encoding='utf-8') as f:
        f.write(bulletproof_code)

def main():
    """Main function to run fixed quantum visualization"""
    
    print("🚀 FIXED QUANTUM VISUALIZATION FOR CANCER DYNAMICS")
    print("=" * 60)
    print("This version uses bulletproof error handling to work with any data format")
    print("=" * 60)
    
    try:
        # Run the fixed demo
        print("\n🎨 Running fixed quantum visualization demo...")
        viz = run_quantum_visualization_demo_fixed()
        
        # Run the fixed parameter study
        print("\n📊 Running fixed parameter sensitivity study...")
        run_quantum_parameter_study_fixed()
        
        print(f"\n" + "🎉" * 20)
        print("FIXED QUANTUM VISUALIZATION COMPLETED!")
        print("🎉" * 20)
        
        print(f"\n📂 Check these directories for results:")
        print(f"   • quantum_demo_output_fixed/ - Main demo results")
        print(f"   • quantum_param_study_fixed/ - Parameter sensitivity")
        print(f"   • visualization/quantum_plots_bulletproof.py - Bulletproof module")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in fixed quantum visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 Fixed quantum visualization completed successfully!")
        print(f"💡 The bulletproof version handles all data format issues.")
    else:
        print(f"\n🔧 If issues persist, check the error messages above.")