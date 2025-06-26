import numpy as np
import matplotlib.pyplot as plt
import os
from config.parameters import get_config


class ComparisonPlotter:
    """
    Class for plotting comparisons between different models and conditions
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
    
    def plot_model_comparison(self, t, trajectories_dict, cell_type_index=0, 
                             initial_condition=None, save=True):
        """
        Compare different models for a specific cell type - exact copy from original
        
        Args:
            t: Time array
            trajectories_dict: Dict with model names as keys, trajectories as values
            cell_type_index: Index of cell type to plot
            initial_condition: Initial condition used
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        cell_label = self.cell_labels[cell_type_index]
        
        for i, (model_name, trajectory) in enumerate(trajectories_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            plt.plot(t, trajectory[:, cell_type_index], '-', color=color, 
                    linewidth=self.line_width, label=model_name)
        
        title = f'{cell_label} Cell Population - Model Comparison'
        if initial_condition is not None:
            title += f'\nInit: T={initial_condition[0]}, I={initial_condition[1]}'
            
        plt.title(title, fontsize=self.font_sizes['title'], fontweight='bold')
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
            filename = f'comparison_{cell_label.lower()}_dynamics.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_alpha_heatmap(self, alpha_values, cell_values, derivative_type, 
                          cell_type_index=0, time_point=-1, save=True):
        """
        Plot heatmap showing final cell values for different alpha values
        
        Args:
            alpha_values: Array of alpha values
            cell_values: 2D array [alpha_idx, cell_type] of final cell values
            derivative_type: Type of fractional derivative
            cell_type_index: Index of cell type
            time_point: Time point to analyze (-1 for final)
            save: Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        cell_label = self.cell_labels[cell_type_index]
        
        # Create heatmap data
        heatmap_data = cell_values[:, cell_type_index].reshape(-1, 1)
        
        # Plot heatmap
        im = plt.imshow(heatmap_data.T, cmap='viridis', aspect='auto', 
                       extent=[alpha_values[0], alpha_values[-1], -0.5, 0.5])
        
        plt.colorbar(im, label=f'{cell_label} Population')
        plt.title(f'{cell_label} Population vs Alpha\n{derivative_type.capitalize()} Derivative', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Alpha Value', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.yticks([])
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{derivative_type}_{cell_label.lower()}_alpha_heatmap.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_derivative_heatmap(self, derivative_types, alpha_values, final_values, 
                               cell_type_index=0, save=True):
        """
        Plot 2D heatmap: derivative types vs alpha values
        
        Args:
            derivative_types: List of derivative type names
            alpha_values: Array of alpha values
            final_values: 3D array [deriv_idx, alpha_idx, cell_type]
            cell_type_index: Index of cell type
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        cell_label = self.cell_labels[cell_type_index]
        
        # Extract data for specific cell type
        heatmap_data = final_values[:, :, cell_type_index]
        
        # Plot heatmap
        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto',
                       extent=[alpha_values[0], alpha_values[-1], 
                              -0.5, len(derivative_types)-0.5])
        
        plt.colorbar(im, label=f'{cell_label} Population')
        plt.title(f'{cell_label} Population: Derivative Types vs Alpha Values', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Alpha Value', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Derivative Type', fontsize=self.font_sizes['label'], fontweight='bold')
        
        # Set y-tick labels
        plt.yticks(range(len(derivative_types)), 
                  [dt.capitalize() for dt in derivative_types], 
                  fontsize=self.font_sizes['tick'])
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.tight_layout()
        
        if save:
            filename = f'{cell_label.lower()}_derivative_alpha_heatmap.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_performance_comparison(self, metrics_dict, metric_name='RMSE', save=True):
        """
        Plot performance comparison between different models/conditions
        
        Args:
            metrics_dict: Dict with condition names as keys, metric arrays as values
            metric_name: Name of the metric being compared
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        n_conditions = len(metrics_dict)
        x = np.arange(len(self.cell_labels))
        width = 0.8 / n_conditions
        
        for i, (condition, metrics) in enumerate(metrics_dict.items()):
            offset = (i - (n_conditions-1)/2) * width
            color = self.colors[i % len(self.colors)]
            
            
            # FIX: Ensure metrics is a list/array, not a dict
            if isinstance(metrics, dict):
                # Extract values from dict if it's a dict
                if 'per_cell_type' in metrics and metric_name.lower() in metrics['per_cell_type']:
                    metric_values = metrics['per_cell_type'][metric_name.lower()]
                elif 'overall' in metrics and metric_name.lower() in metrics['overall']:
                    # Use overall metric repeated for each cell type
                    overall_val = metrics['overall'][metric_name.lower()]
                    metric_values = [overall_val] * len(self.cell_labels)
                else:
                    # Default to zeros if structure is unexpected
                    metric_values = [0.0] * len(self.cell_labels)
            else:
                # Assume it's already a list/array
                metric_values = metrics
            
            # Ensure we have the right number of values
            if len(metric_values) != len(self.cell_labels):
                print(f"Warning: Metric values length ({len(metric_values)}) doesn't match cell labels ({len(self.cell_labels)})")
                # Pad or truncate as needed
                if len(metric_values) < len(self.cell_labels):
                    metric_values.extend([0.0] * (len(self.cell_labels) - len(metric_values)))
                else:
                    metric_values = metric_values[:len(self.cell_labels)]
            
            bars = plt.bar(x + offset, metric_values, width, label=condition, 
                          color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, fontweight='bold')
        
        plt.title(f'{metric_name} by Cell Type', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xticks(x, self.cell_labels, fontsize=self.font_sizes['tick'], fontweight='bold')
        plt.ylabel(metric_name, fontsize=self.font_sizes['label'], fontweight='bold')
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{metric_name.lower()}_comparison.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_evolution(self, t, error_dict, save=True):
        """
        Plot how prediction errors evolve over time
        
        Args:
            t: Time array
            error_dict: Dict with model names as keys, error arrays as values
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        for i, (model_name, errors) in enumerate(error_dict.items()):
            color = self.colors[i % len(self.colors)]
            plt.plot(t, errors, '-', color=color, linewidth=self.line_width, 
                    label=model_name)
        
        plt.title('Prediction Error Evolution Over Time', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Time', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Prediction Error', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'], loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = 'error_evolution.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_sensitivity_analysis(self, parameter_values, final_outcomes, 
                                 parameter_name, cell_type_index=0, save=True):
        """
        Plot sensitivity analysis for a parameter
        
        Args:
            parameter_values: Array of parameter values tested
            final_outcomes: Array of final outcomes for each parameter value
            parameter_name: Name of the parameter
            cell_type_index: Index of cell type
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        cell_label = self.cell_labels[cell_type_index]
        
        plt.plot(parameter_values, final_outcomes, 'o-', 
                color=self.colors[0], linewidth=self.line_width, markersize=6)
        
        plt.title(f'{cell_label} Population Sensitivity to {parameter_name}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel(parameter_name, fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel(f'Final {cell_label} Population', 
                  fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'sensitivity_{parameter_name.lower()}_{cell_label.lower()}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_convergence_analysis(self, step_sizes, errors, save=True):
        """
        Plot convergence analysis for numerical methods
        """
        plt.figure(figsize=self.figure_size)
        
        plt.loglog(step_sizes, errors, 'o-', color=self.colors[0], 
                  linewidth=self.line_width, markersize=6)
        
        # Add reference lines for different orders
        plt.loglog(step_sizes, step_sizes, '--', color='gray', alpha=0.7, label='Order 1')
        plt.loglog(step_sizes, step_sizes**2, '--', color='lightgray', alpha=0.7, label='Order 2')
        
        plt.title('Convergence Analysis', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Step Size', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Error', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = 'convergence_analysis.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_comparison_plotter(output_dir='plots_output'):
    """Factory function to create ComparisonPlotter"""
    return ComparisonPlotter(output_dir)

    