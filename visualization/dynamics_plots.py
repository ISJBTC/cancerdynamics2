import numpy as np
import matplotlib.pyplot as plt
import os
from config.parameters import get_config


class DynamicsPlotter:
    """
    Class for plotting cancer dynamics - based on original code
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
        
    def plot_individual_cell_dynamics(self, t, trajectories, labels, model_name, 
                                    initial_conditions=None, save=True):
        """
        Plot individual cell type dynamics - exact copy from original code
        
        Args:
            t: Time array
            trajectories: List of trajectory arrays for different initial conditions
            labels: List of labels for trajectories
            model_name: Name of the model (e.g., 'Integer', 'Fractional')
            initial_conditions: List of initial condition arrays
            save: Whether to save plots
        """
        if initial_conditions is None:
            initial_conditions = self.config.initial_conditions
            
        # Create individual plots for each cell type
        for i, cell_label in enumerate(self.cell_labels):
            plt.figure(figsize=self.figure_size)
            
            for j, (trajectory, label) in enumerate(zip(trajectories, labels)):
                color = self.colors[j % len(self.colors)]
                
                plt.plot(t, trajectory[:, i], '-', color=color, 
                        linewidth=self.line_width, label=label)
                
                # Mark initial and final points
                plt.scatter(t[0], trajectory[0, i], color=color, s=30, 
                           marker='o', edgecolor='black', zorder=5)
                plt.scatter(t[-1], trajectory[-1, i], color=color, s=30, 
                           marker='x', linewidth=1.5, zorder=5)
            
            plt.title(f'{cell_label} Cell Population - {model_name} Model', 
                     fontsize=self.font_sizes['title'], fontweight='bold')
            plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
            plt.ylabel('Population Size', fontsize=self.font_sizes['label'], fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=self.font_sizes['tick'])
            plt.yticks(fontsize=self.font_sizes['tick'])
            
            # Set y-axis limits for tumor plot to show promising range
            if cell_label == 'Tumor':
                plt.ylim(0, 100)
                
            plt.legend(fontsize=self.font_sizes['legend'], loc='best')
            plt.tight_layout()
            
            if save:
                filename = f'{model_name.lower()}_{cell_label.lower()}_dynamics.png'
                plt.savefig(os.path.join(self.output_dir, filename), 
                           dpi=self.dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_combined_dynamics(self, t, actual_trajectories, predicted_trajectories,
                             model_name, initial_conditions=None, save=True):
        """
        Plot actual vs predicted dynamics for each cell type
        """
        if initial_conditions is None:
            initial_conditions = self.config.initial_conditions
            
        for i, cell_label in enumerate(self.cell_labels):
            plt.figure(figsize=self.figure_size)
            
            for j, init_cond in enumerate(initial_conditions):
                color = self.colors[j % len(self.colors)]
                
                # Plot actual and predicted
                plt.plot(t, actual_trajectories[j][:, i], '-', color=color, 
                        linewidth=self.line_width, 
                        label=f'Actual (T0={init_cond[0]}, I0={init_cond[1]})')
                plt.plot(t, predicted_trajectories[j][:, i], '--', color=color, 
                        linewidth=self.line_width, 
                        label=f'Predicted (T0={init_cond[0]}, I0={init_cond[1]})')
            
            plt.title(f'{cell_label} Cell Population - {model_name} Model (Actual vs Predicted)', 
                     fontsize=self.font_sizes['title'], fontweight='bold')
            plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
            plt.ylabel('Population Size', fontsize=self.font_sizes['label'], fontweight='bold')
            plt.xticks(fontsize=self.font_sizes['tick'])
            plt.yticks(fontsize=self.font_sizes['tick'])
            
            if cell_label == 'Tumor':
                plt.ylim(0, 100)
                
            plt.legend(fontsize=self.font_sizes['legend'], loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save:
                filename = f'{model_name.lower()}_{cell_label.lower()}_comparison.png'
                plt.savefig(os.path.join(self.output_dir, filename), 
                           dpi=self.dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_all_cell_types_single_plot(self, t, trajectory, model_name, 
                                       init_condition, save=True):
        """
        Plot all cell types in a single plot
        """
        plt.figure(figsize=self.figure_size)
        
        for i, cell_label in enumerate(self.cell_labels):
            color = self.colors[i % len(self.colors)]
            plt.plot(t, trajectory[:, i], '-', color=color, 
                    linewidth=self.line_width, label=cell_label)
        
        plt.title(f'All Cell Populations - {model_name} Model\n'
                 f'Initial: T={init_condition[0]}, I={init_condition[1]}, '
                 f'M={init_condition[2]}, S={init_condition[3]}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Population Size', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        
        plt.legend(fontsize=self.font_sizes['legend'], loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_all_cells_dynamics.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_immune_tumor_ratio(self, t, trajectories, model_name, 
                               initial_conditions=None, save=True):
        """
        Plot immune/tumor ratio over time - exact copy from original code
        """
        if initial_conditions is None:
            initial_conditions = self.config.initial_conditions
            
        plt.figure(figsize=self.figure_size)
        
        for j, (trajectory, init_cond) in enumerate(zip(trajectories, initial_conditions)):
            color = self.colors[j % len(self.colors)]
            
            # Calculate ratio with protection against division by zero
            ratio = trajectory[:, 1] / (trajectory[:, 0] + 1e-10)  # I/T ratio
            ratio = np.clip(ratio, 0, 10)  # Limit for better visualization
            
            plt.plot(t, ratio, '-', color=color, linewidth=self.line_width,
                    label=f'Init: T={init_cond[0]}, I={init_cond[1]}')
        
        plt.title(f'Immune/Tumor Ratio Over Time - {model_name} Model', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('I/T Ratio', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.font_sizes['legend'], loc='best')
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_immune_tumor_ratio.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_alpha_comparison(self, t, trajectories_dict, cell_type_index=0, 
                             derivative_type='caputo', save=True):
        """
        Plot trajectories for different alpha values
        
        Args:
            t: Time array
            trajectories_dict: Dict with alpha values as keys, trajectories as values
            cell_type_index: Index of cell type to plot (0=Tumor, 1=Immune, etc.)
            derivative_type: Type of fractional derivative
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        cell_label = self.cell_labels[cell_type_index]
        
        # Sort alpha values for consistent coloring
        alpha_values = sorted(trajectories_dict.keys())
        
        for i, alpha in enumerate(alpha_values):
            trajectory = trajectories_dict[alpha]
            color = plt.cm.viridis(i / (len(alpha_values) - 1)) if len(alpha_values) > 1 else self.colors[0]
            
            plt.plot(t, trajectory[:, cell_type_index], '-', color=color, 
                    linewidth=self.line_width, label=f'α = {alpha:.1f}')
        
        plt.title(f'{cell_label} Population vs Alpha Values\n'
                 f'{derivative_type.capitalize()} Derivative', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'{cell_label} Population', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        
        if cell_label == 'Tumor':
            plt.ylim(0, 100)
            
        plt.legend(fontsize=self.font_sizes['legend'], loc='best', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{derivative_type}_{cell_label.lower()}_alpha_comparison.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_derivative_type_comparison(self, t, trajectories_dict, cell_type_index=0, 
                                      alpha=1.0, save=True):
        """
        Plot trajectories for different derivative types at fixed alpha
        
        Args:
            t: Time array
            trajectories_dict: Dict with derivative types as keys, trajectories as values
            cell_type_index: Index of cell type to plot
            alpha: Alpha value used
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        cell_label = self.cell_labels[cell_type_index]
        
        for i, (deriv_type, trajectory) in enumerate(trajectories_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            plt.plot(t, trajectory[:, cell_type_index], '-', color=color, 
                    linewidth=self.line_width, label=deriv_type.capitalize())
        
        plt.title(f'{cell_label} Population - Derivative Type Comparison\n'
                 f'α = {alpha}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'{cell_label} Population', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        
        if cell_label == 'Tumor':
            plt.ylim(0, 100)
            
        plt.legend(fontsize=self.font_sizes['legend'], loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{cell_label.lower()}_derivative_comparison_alpha_{alpha:.1f}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_time_series_grid(self, t, trajectories_dict, save=True):
        """
        Plot a grid of time series for different conditions
        """
        n_plots = len(trajectories_dict)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (condition, trajectory) in enumerate(trajectories_dict.items()):
            ax = axes[idx]
            
            for i, cell_label in enumerate(self.cell_labels):
                color = self.colors[i % len(self.colors)]
                ax.plot(t, trajectory[:, i], '-', color=color, 
                       linewidth=1.2, label=cell_label)
            
            ax.set_title(condition, fontsize=9, fontweight='bold')
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Population', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filename = 'time_series_grid.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_dynamics_plotter(output_dir='plots_output'):
    """Factory function to create DynamicsPlotter"""
    return DynamicsPlotter(output_dir)
    