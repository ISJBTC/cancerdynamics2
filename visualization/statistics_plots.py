import numpy as np
import matplotlib.pyplot as plt
import os
from config.parameters import get_config


class StatisticsPlotter:
    """
    Class for plotting statistical analysis and performance metrics
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
    
    def plot_metrics_table(self, metrics_data, title="Performance Metrics", save=True):
        """
        Create a table plot for metrics - exact copy from original code style
        
        Args:
            metrics_data: List of lists for table data
            title: Title for the table
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        ax = plt.gca()
        ax.axis('off')
        
        table = plt.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Set cell colors
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[0])):
                cell = table[i, j]
                if i == 0:  # Header row
                    cell.set_facecolor('#AAD0FF')
                    cell.set_text_props(weight='bold')
                elif j == 0:  # First column
                    cell.set_text_props(weight='bold')
                elif j == 3 and i > 0:  # Difference column (if exists)
                    try:
                        value = float(metrics_data[i][j].strip('%'))
                        if (i < 3 and value < 0) or (i == 3 and value > 0):
                            cell.set_facecolor('#D0FFD0')  # Light green for good
                        else:
                            cell.set_facecolor('#FFD0D0')  # Light red for bad
                    except:
                        pass
        
        plt.title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = f'{title.lower().replace(" ", "_")}_table.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_accuracy(self, actual, predicted, model_name, save=True):
        """
        Plot prediction accuracy scatter plot
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        # Flatten arrays if multidimensional
        if actual.ndim > 1:
            actual_flat = actual.flatten()
            predicted_flat = predicted.flatten()
        else:
            actual_flat = actual
            predicted_flat = predicted
        
        # Scatter plot
        plt.scatter(actual_flat, predicted_flat, alpha=0.6, 
                   color=self.colors[0], s=20, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(actual_flat.min(), predicted_flat.min())
        max_val = max(actual_flat.max(), predicted_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        correlation_matrix = np.corrcoef(actual_flat, predicted_flat)
        r_squared = correlation_matrix[0, 1] ** 2
        
        plt.title(f'Prediction Accuracy - {model_name}\nR² = {r_squared:.4f}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Actual Values', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_prediction_accuracy.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_residuals(self, actual, predicted, model_name, save=True):
        """
        Plot residuals analysis
        """
        plt.figure(figsize=self.figure_size)
        
        # Calculate residuals
        if actual.ndim > 1:
            actual_flat = actual.flatten()
            predicted_flat = predicted.flatten()
        else:
            actual_flat = actual
            predicted_flat = predicted
            
        residuals = actual_flat - predicted_flat
        
        # Plot residuals vs predicted
        plt.scatter(predicted_flat, residuals, alpha=0.6, 
                   color=self.colors[0], s=20, edgecolors='black', linewidth=0.5)
        
        # Zero line
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        plt.title(f'Residuals Analysis - {model_name}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Predicted Values', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Residuals', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_residuals.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_distribution(self, errors, model_name, save=True):
        """
        Plot error distribution histogram
        """
        plt.figure(figsize=self.figure_size)
        
        # Flatten errors if multidimensional
        if errors.ndim > 1:
            errors_flat = errors.flatten()
        else:
            errors_flat = errors
        
        # Histogram
        plt.hist(errors_flat, bins=30, alpha=0.7, color=self.colors[0], 
                edgecolor='black', density=True)
        
        # Statistics
        mean_error = np.mean(errors_flat)
        std_error = np.std(errors_flat)
        
        plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.4f}')
        plt.axvline(mean_error + std_error, color='orange', linestyle=':', linewidth=2, 
                   label=f'±1σ: {std_error:.4f}')
        plt.axvline(mean_error - std_error, color='orange', linestyle=':', linewidth=2)
        
        plt.title(f'Error Distribution - {model_name}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Error', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Density', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_error_distribution.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_learning_curves(self, training_history, model_name, save=True):
        """
        Plot training and validation loss curves
        """
        plt.figure(figsize=self.figure_size)
        
        epochs = range(1, len(training_history['train_losses']) + 1)
        
        plt.plot(epochs, training_history['train_losses'], 'b-', 
                linewidth=self.line_width, label='Training Loss')
        
        if 'val_losses' in training_history:
            plt.plot(epochs, training_history['val_losses'], 'r-', 
                    linewidth=self.line_width, label='Validation Loss')
        
        plt.title(f'Learning Curves - {model_name}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xlabel('Epoch', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.ylabel('Loss', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.legend(fontsize=self.font_sizes['legend'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_learning_curves.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, data, labels=None, title="Correlation Matrix", save=True):
        """
        Plot correlation matrix heatmap
        """
        if labels is None:
            labels = self.cell_labels
            
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        plt.figure(figsize=(6, 5))
        
        # Plot heatmap
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label='Correlation Coefficient')
        
        # Set labels
        plt.xticks(range(len(labels)), labels, rotation=45, 
                  fontsize=self.font_sizes['tick'])
        plt.yticks(range(len(labels)), labels, fontsize=self.font_sizes['tick'])
        
        # Add correlation values as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = f'{title.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_box_plots(self, data_dict, title="Box Plot Comparison", save=True):
        """
        Plot box plots for comparing distributions
        
        Args:
            data_dict: Dict with condition names as keys, data arrays as values
            title: Title for the plot
            save: Whether to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        # Prepare data for box plot
        data_list = []
        labels = []
        
        for condition, data in data_dict.items():
            data_list.append(data.flatten() if data.ndim > 1 else data)
            labels.append(condition)
        
        # Create box plot
        box_plot = plt.boxplot(data_list, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xticks(fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.ylabel('Values', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            filename = f'{title.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_violin_plots(self, data_dict, title="Violin Plot Comparison", save=True):
        """
        Plot violin plots for comparing distributions
        """
        plt.figure(figsize=self.figure_size)
        
        # Prepare data
        data_list = []
        labels = []
        positions = []
        
        for i, (condition, data) in enumerate(data_dict.items()):
            data_list.append(data.flatten() if data.ndim > 1 else data)
            labels.append(condition)
            positions.append(i + 1)
        
        # Create violin plot
        violin_parts = plt.violinplot(data_list, positions=positions, showmeans=True)
        
        # Color the violins
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
        
        plt.title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        plt.xticks(positions, labels, fontsize=self.font_sizes['tick'])
        plt.yticks(fontsize=self.font_sizes['tick'])
        plt.ylabel('Values', fontsize=self.font_sizes['label'], fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            filename = f'{title.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_qq_plot(self, data, model_name, save=True):
        """
        Plot Q-Q plot for normality check
        """
        from scipy import stats
        
        plt.figure(figsize=self.figure_size)
        
        # Flatten data if needed
        if data.ndim > 1:
            data_flat = data.flatten()
        else:
            data_flat = data
        
        # Create Q-Q plot
        stats.probplot(data_flat, dist="norm", plot=plt)
        
        plt.title(f'Q-Q Plot - {model_name}', 
                 fontsize=self.font_sizes['title'], fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f'{model_name.lower()}_qq_plot.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_sensitivity_radar(self, parameters, sensitivities, save=True):
        """
        Plot radar chart for parameter sensitivity analysis
        """
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
        
        # Number of parameters
        n_params = len(parameters)
        
        # Compute angles for each parameter
        angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Add first value to the end to close the polygon
        sensitivities += sensitivities[:1]
        
        # Plot
        ax.plot(angles, sensitivities, 'o-', linewidth=self.line_width, 
               color=self.colors[0])
        ax.fill(angles, sensitivities, alpha=0.25, color=self.colors[0])
        
        # Add parameter labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(parameters, fontsize=self.font_sizes['tick'])
        
        # Set title
        ax.set_title('Parameter Sensitivity Analysis', 
                    fontsize=self.font_sizes['title'], fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            filename = 'parameter_sensitivity_radar.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_summary_dashboard(self, summary_data, save=True):
        """
        Create a comprehensive summary dashboard
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Model Performance Comparison
        ax1 = axes[0, 0]
        models = list(summary_data['performance'].keys())
        rmse_values = [summary_data['performance'][model]['rmse'] for model in models]
        
        bars = ax1.bar(models, rmse_values, color=self.colors[:len(models)])
        ax1.set_title('Model Performance (RMSE)', fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax1.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # Plot 2: Alpha Sensitivity
        ax2 = axes[0, 1]
        alpha_values = summary_data['alpha_analysis']['alphas']
        tumor_outcomes = summary_data['alpha_analysis']['tumor_outcomes']
        
        ax2.plot(alpha_values, tumor_outcomes, 'o-', color=self.colors[0])
        ax2.set_title('Tumor Response vs Alpha', fontweight='bold')
        ax2.set_xlabel('Alpha Value')
        ax2.set_ylabel('Final Tumor Population')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Derivative Type Comparison
        ax3 = axes[1, 0]
        deriv_types = list(summary_data['derivatives'].keys())
        final_values = [summary_data['derivatives'][dt]['final_tumor'] for dt in deriv_types]
        
        ax3.bar(deriv_types, final_values, color=self.colors[:len(deriv_types)])
        ax3.set_title('Final Tumor Population by Derivative', fontweight='bold')
        ax3.set_ylabel('Final Tumor Population')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Training Loss
        ax4 = axes[1, 1]
        if 'training_history' in summary_data:
            epochs = range(1, len(summary_data['training_history']['train_loss']) + 1)
            ax4.plot(epochs, summary_data['training_history']['train_loss'], 
                    label='Training', color=self.colors[0])
            if 'val_loss' in summary_data['training_history']:
                ax4.plot(epochs, summary_data['training_history']['val_loss'], 
                        label='Validation', color=self.colors[1])
            ax4.set_title('Training Progress', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Training Data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training Progress', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = 'summary_dashboard.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_statistics_plotter(output_dir='plots_output'):
    """Factory function to create StatisticsPlotter"""
    return StatisticsPlotter(output_dir)
