import numpy as np
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
