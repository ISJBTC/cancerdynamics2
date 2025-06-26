import numpy as np
import matplotlib.pyplot as plt
import os
from config.parameters import get_config


class PhasePortraitPlotter:
    """
    Class for plotting phase portraits and phase space analysis
    """
    
    def __init__(self, output_dir='plots_output'):
        self.config = get_config()
        self.output_dir = output_dir
        self.colors = self.config.viz_params['colors']
        self.figure_size = self.config.viz_params['figure_size']
        self.dpi = self.config.viz_params['dpi']
        self.line_width = self.config.viz_params['line_width']
        self.font_sizes = self.config.viz_params['font_sizes']
        self.cell_labels = self.config.cell_labels
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_tumor_immune_phase_portrait(self, trajectories, model_name, 
                                       initial_conditions=None, save=True):
        """
        Plot tumor-immune phase portrait - exact copy from original code
        """
        if initial_conditions is None:
            initial_conditions = self.config.initial_conditions
            
        plt.figure(figsize=self.figure_size)
        
        for j, (trajectory, init_cond) in enumerate(zip(trajectories, initial_conditions)):
            color = self.colors[j % len(self.colors)]
            
            # Plot phase portrait
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, 
                    linewidth=self.line_width, 
                    label=f'{model_name} (T0={init_cond[0]}, I0={init_cond[1]})')
            
            # Mark initial and final points
            plt.scatter(init_cond[0], init_cond[1], color=color, s=30, 
                       marker='o', edgecolor='black', zorder=5)
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=30, 
                       marker='x', linewidth=1.5, zorder=5)
        
        plt.title(f'Tumor-Immune Phase Portrait ({model_name} Model)', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Tumor Cell Population', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Immune Cell Population', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.font_sizes['legend'], loc='best')
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_phase_portrait.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_3d_phase_portrait(self, trajectories, model_name, 
                              cell_indices=[0, 1, 2], initial_conditions=None, save=True):
        """
        Plot 3D phase portrait for three selected cell types
        
        Args:
            trajectories: List of trajectory arrays
            model_name: Name of the model
            cell_indices: Indices of cell types to plot [x, y, z]
            initial_conditions: Initial conditions
            save: Whether to save the plot
        """
        if initial_conditions is None:
            initial_conditions = self.config.initial_conditions
            
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        cell_names = [self.cell_labels[i] for i in cell_indices]
        
        for j, (trajectory, init_cond) in enumerate(zip(trajectories, initial_conditions)):
            color = self.colors[j % len(self.colors)]
            
            # Plot 3D trajectory
            ax.plot(trajectory[:, cell_indices[0]], 
                   trajectory[:, cell_indices[1]], 
                   trajectory[:, cell_indices[2]], 
                   '-', color=color, linewidth=self.line_width,
                   label=f'T0={init_cond[0]}, I0={init_cond[1]}')
            
            # Mark initial point
            ax.scatter(trajectory[0, cell_indices[0]], 
                      trajectory[0, cell_indices[1]], 
                      trajectory[0, cell_indices[2]], 
                      color=color, s=50, marker='o', edgecolor='black')
            
            # Mark final point
            ax.scatter(trajectory[-1, cell_indices[0]], 
                      trajectory[-1, cell_indices[1]], 
                      trajectory[-1, cell_indices[2]], 
                      color=color, s=50, marker='x')
        
        ax.set_title(f'3D Phase Portrait ({model_name} Model)', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        ax.set_xlabel(f'{cell_names[0]} Cells', fontsize=self.font_sizes['label'])
        ax.set_ylabel(f'{cell_names[1]} Cells', fontsize=self.font_sizes['label'])
        ax.set_zlabel(f'{cell_names[2]} Cells', fontsize=self.font_sizes['label'])
        ax.legend(fontsize=self.font_sizes['legend'])
        
        if save:
            filename = f'{model_name.lower()}_3d_phase_portrait.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pairwise_phase_portraits(self, trajectory, model_name, save=True):
        """
        Plot pairwise phase portraits for all cell type combinations
        """
        n_cells = len(self.cell_labels)
        fig, axes = plt.subplots(n_cells-1, n_cells-1, figsize=(12, 10))
        
        for i in range(n_cells-1):
            for j in range(n_cells-1):
                ax = axes[i, j]
                
                if j >= i:
                    # Plot phase portrait
                    x_idx, y_idx = i, j+1
                    ax.plot(trajectory[:, x_idx], trajectory[:, y_idx], 
                           '-', color=self.colors[0], linewidth=self.line_width)
                    
                    # Mark initial and final points
                    ax.scatter(trajectory[0, x_idx], trajectory[0, y_idx], 
                              color='green', s=30, marker='o', zorder=5)
                    ax.scatter(trajectory[-1, x_idx], trajectory[-1, y_idx], 
                              color='red', s=30, marker='x', zorder=5)
                    
                    ax.set_xlabel(self.cell_labels[x_idx], fontsize=8)
                    ax.set_ylabel(self.cell_labels[y_idx], fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=7)
                else:
                    # Hide lower triangle
                    ax.set_visible(False)
        
        plt.suptitle(f'Pairwise Phase Portraits ({model_name} Model)', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_pairwise_phase_portraits.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_vector_field(self, model, x_range, y_range, model_name, 
                         cell_indices=[0, 1], fixed_values=None, save=True):
        """
        Plot vector field for the dynamics
        
        Args:
            model: Dynamics model
            x_range: Range for x-axis (tuple of min, max)
            y_range: Range for y-axis (tuple of min, max)
            model_name: Name of the model
            cell_indices: Indices of cell types for x and y axes
            fixed_values: Fixed values for other cell types
            save: Whether to save the plot
        """
        if fixed_values is None:
            fixed_values = [20.0, 30.0]  # Default values for other cells
            
        # Create grid
        x = np.linspace(x_range[0], x_range[1], 20)
        y = np.linspace(y_range[0], y_range[1], 20)
        X, Y = np.meshgrid(x, y)
        
        # Initialize arrays for vector components
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Calculate vector field
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Construct state vector
                state = [0, 0, 0, 0]
                state[cell_indices[0]] = X[i, j]
                state[cell_indices[1]] = Y[i, j]
                
                # Fill in fixed values for other cells
                fixed_idx = 0
                for k in range(4):
                    if k not in cell_indices:
                        state[k] = fixed_values[fixed_idx]
                        fixed_idx += 1
                
                # Get derivatives
                try:
                    derivatives = model.system_dynamics(state, 0)
                    U[i, j] = derivatives[cell_indices[0]]
                    V[i, j] = derivatives[cell_indices[1]]
                except:
                    U[i, j] = 0
                    V[i, j] = 0
        
        # Plot vector field
        plt.figure(figsize=self.figure_size)
        plt.quiver(X, Y, U, V, alpha=0.6, color='gray', scale_units='xy', scale=1)
        
        plt.title(f'Vector Field ({model_name} Model)\n'
                 f'{self.cell_labels[cell_indices[0]]} vs {self.cell_labels[cell_indices[1]]}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel(f'{self.cell_labels[cell_indices[0]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'{self.cell_labels[cell_indices[1]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            cell_x = self.cell_labels[cell_indices[0]].lower()
            cell_y = self.cell_labels[cell_indices[1]].lower()
            filename = f'{model_name.lower()}_vector_field_{cell_x}_{cell_y}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_nullclines(self, model, x_range, y_range, model_name, 
                       cell_indices=[0, 1], fixed_values=None, save=True):
        """
        Plot nullclines for the system
        """
        if fixed_values is None:
            fixed_values = [20.0, 30.0]
            
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate derivatives on grid
        dX = np.zeros_like(X)
        dY = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = [0, 0, 0, 0]
                state[cell_indices[0]] = X[i, j]
                state[cell_indices[1]] = Y[i, j]
                
                fixed_idx = 0
                for k in range(4):
                    if k not in cell_indices:
                        state[k] = fixed_values[fixed_idx]
                        fixed_idx += 1
                
                try:
                    derivatives = model.system_dynamics(state, 0)
                    dX[i, j] = derivatives[cell_indices[0]]
                    dY[i, j] = derivatives[cell_indices[1]]
                except:
                    dX[i, j] = 0
                    dY[i, j] = 0
        
        plt.figure(figsize=self.figure_size)
        
        # Plot nullclines
        plt.contour(X, Y, dX, levels=[0], colors='red', linewidths=2, 
                   label=f'd{self.cell_labels[cell_indices[0]]}/dt = 0')
        plt.contour(X, Y, dY, levels=[0], colors='blue', linewidths=2, 
                   label=f'd{self.cell_labels[cell_indices[1]]}/dt = 0')
        
        plt.title(f'Nullclines ({model_name} Model)', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel(f'{self.cell_labels[cell_indices[0]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'{self.cell_labels[cell_indices[1]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.tight_layout()
        
        if save:
            cell_x = self.cell_labels[cell_indices[0]].lower()
            cell_y = self.cell_labels[cell_indices[1]].lower()
            filename = f'{model_name.lower()}_nullclines_{cell_x}_{cell_y}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_alpha_phase_comparison(self, trajectories_dict, model_name, 
                                   cell_indices=[0, 1], save=True):
        """
        Plot phase portraits for different alpha values
        """
        plt.figure(figsize=self.figure_size)
        
        alpha_values = sorted(trajectories_dict.keys())
        
        for i, alpha in enumerate(alpha_values):
            trajectory = trajectories_dict[alpha]
            color = plt.cm.viridis(i / (len(alpha_values) - 1)) if len(alpha_values) > 1 else self.colors[0]
            
            plt.plot(trajectory[:, cell_indices[0]], trajectory[:, cell_indices[1]], 
                    '-', color=color, linewidth=self.line_width, label=f'α = {alpha:.1f}')
            
            # Mark initial point
            plt.scatter(trajectory[0, cell_indices[0]], trajectory[0, cell_indices[1]], 
                       color=color, s=20, marker='o', edgecolor='black')
        
        plt.title(f'Phase Portrait - Alpha Comparison ({model_name})', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel(f'{self.cell_labels[cell_indices[0]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'{self.cell_labels[cell_indices[1]]} Cells', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.font_sizes['legend'], ncol=2)
        plt.tight_layout()
        
        if save:
            cell_x = self.cell_labels[cell_indices[0]].lower()
            cell_y = self.cell_labels[cell_indices[1]].lower()
            filename = f'{model_name.lower()}_phase_alpha_comparison_{cell_x}_{cell_y}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_phase_portrait_plotter(output_dir='plots_output'):
    """Factory function to create PhasePortraitPlotter"""
    return PhasePortraitPlotter(output_dir)
    