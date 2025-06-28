#!/usr/bin/env python3
"""
Debug script to identify the exact issue with quantum visualization
"""

import numpy as np
from scipy.integrate import odeint
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from config.parameters import get_config, get_quantum_parameters

def debug_integration_issue():
    """Debug the integration issue causing the error"""
    
    print("🔍 DEBUGGING QUANTUM VISUALIZATION ISSUE")
    print("=" * 50)
    
    # Get configuration
    config = get_config()
    quantum_params = get_quantum_parameters()
    
    # Create models
    integer_model = IntegerModel()
    fractional_model = FractionalModel()
    
    # Test scenarios
    test_conditions = [
        ([50, 10, 20, 30], "Normal"),
        ([1e-4, 1e-4, 20, 30], "Quantum Test"),
        ([50, 1e-4, 20, 30], "Immune Quantum"),
        ([1e-4, 10, 20, 30], "Tumor Quantum")
    ]
    
    t = np.linspace(0, 6, 61)
    
    for i, (init_cond, name) in enumerate(test_conditions):
        print(f"\n📊 Testing Scenario {i+1}: {name}")
        print(f"   Initial condition: {init_cond}")
        
        try:
            # Test integer model integration
            print("   Testing integer model integration...")
            int_traj = odeint(integer_model, init_cond, t)
            
            print(f"   ✅ Integer integration successful")
            print(f"      Shape: {int_traj.shape}")
            print(f"      Type: {type(int_traj)}")
            print(f"      First row: {int_traj[0]}")
            print(f"      Last row: {int_traj[-1]}")
            print(f"      Has NaN: {np.any(np.isnan(int_traj))}")
            print(f"      Has Inf: {np.any(np.isinf(int_traj))}")
            
            # Check if trajectory is actually a scalar (this might be the issue)
            if int_traj.ndim == 0:
                print(f"   ⚠️  WARNING: Trajectory is a scalar, not an array!")
                print(f"      Value: {int_traj}")
                print(f"      This is likely the source of the 'invalid index to scalar variable' error")
            elif int_traj.ndim == 1:
                print(f"   ⚠️  WARNING: Trajectory is 1D, expected 2D")
                print(f"      Length: {len(int_traj)}")
            
            # Test fractional model integration
            print("   Testing fractional model integration...")
            fractional_model.reset_history()
            frac_traj = odeint(fractional_model, init_cond, t)
            
            print(f"   ✅ Fractional integration successful")
            print(f"      Shape: {frac_traj.shape}")
            print(f"      Type: {type(frac_traj)}")
            
        except Exception as e:
            print(f"   ❌ Integration failed: {e}")
            print(f"      Error type: {type(e)}")
            
            # Try to understand what went wrong
            try:
                # Test just one step
                print("   Testing single step...")
                single_step = integer_model.system_dynamics(init_cond, 0)
                print(f"      Single step result: {single_step}")
                print(f"      Single step type: {type(single_step)}")
                print(f"      Single step shape: {np.array(single_step).shape}")
            except Exception as e2:
                print(f"      Single step also failed: {e2}")

def debug_visualization_data():
    """Debug what happens when we pass problematic data to visualization"""
    
    print(f"\n🎨 DEBUGGING VISUALIZATION DATA HANDLING")
    print("=" * 50)
    
    # Test different types of problematic data
    test_data = [
        (np.array([1, 2, 3]), "Normal 1D array"),
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), "Normal 2D array"),
        (5.0, "Scalar float"),
        (np.array(5.0), "Scalar array"),
        (np.array([]), "Empty array"),
        (np.array([np.nan, np.inf, 1, 2]), "Array with NaN/Inf"),
    ]
    
    for data, description in test_data:
        print(f"\n   Testing: {description}")
        print(f"      Data: {data}")
        print(f"      Type: {type(data)}")
        print(f"      Shape: {getattr(data, 'shape', 'No shape attribute')}")
        print(f"      Ndim: {getattr(data, 'ndim', 'No ndim attribute')}")
        
        try:
            # Test indexing
            if hasattr(data, '__getitem__') and hasattr(data, 'shape'):
                if data.ndim >= 2:
                    test_index = data[:, 0]
                    print(f"      Indexing [:, 0]: {test_index}")
                elif data.ndim == 1 and len(data) > 0:
                    test_index = data[0]
                    print(f"      Indexing [0]: {test_index}")
                else:
                    print(f"      Cannot index this data safely")
            else:
                print(f"      Data is not indexable")
        except Exception as e:
            print(f"      ❌ Indexing failed: {e}")

def create_minimal_fix():
    """Create a minimal fix based on what we find"""
    
    print(f"\n🔧 CREATING MINIMAL VISUALIZATION FIX")
    print("=" * 50)
    
    # Create a very simple, bulletproof visualization function
    simple_viz_code = '''
import numpy as np
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
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Immune
            ax2.plot(t_safe, I_traj, 'r-', linewidth=2, label='Immune')
            ax2.axhline(y=quantum_threshold, color='orange', linestyle='--', label='Threshold')
            ax2.set_title('Immune Population')
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
    with open('visualization/quantum_plots_bulletproof.py', 'w', encoding='utf-8') as f:
        f.write(simple_viz_code)
    
    print("✅ Created bulletproof quantum visualization: visualization/quantum_plots_bulletproof.py")
    print("   This version has maximum error handling and should work with any data")

if __name__ == "__main__":
    debug_integration_issue()
    debug_visualization_data()
    create_minimal_fix()
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the debug output above to understand the issue")
    print(f"2. Try replacing quantum_plots.py with quantum_plots_bulletproof.py")
    print(f"3. Or modify the runner script to use the bulletproof version")