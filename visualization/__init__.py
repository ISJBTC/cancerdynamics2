"""
Visualization module for cancer dynamics research

This module provides comprehensive plotting capabilities for:
- Dynamics plots (time series, individual cell types)
- Phase portraits (2D, 3D, pairwise)
- Comparison plots (models, parameters, derivatives)
- Statistical analysis (metrics, distributions, performance)
"""

from .dynamics_plots import DynamicsPlotter, create_dynamics_plotter
from .phase_portraits import PhasePortraitPlotter, create_phase_portrait_plotter
from .comparison_plots import ComparisonPlotter, create_comparison_plotter
from .statistics_plots import StatisticsPlotter, create_statistics_plotter

import os
from config.parameters import get_config


class MasterVisualization:
    """
    Master visualization class that combines all plotting capabilities
    """
    
    def __init__(self, output_dir='plots_output'):
        self.output_dir = output_dir
        self.config = get_config()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize all plotters
        self.dynamics = create_dynamics_plotter(output_dir)
        self.phase = create_phase_portrait_plotter(output_dir)
        self.comparison = create_comparison_plotter(output_dir)
        self.statistics = create_statistics_plotter(output_dir)
        
        print(f"Master Visualization initialized with output directory: {output_dir}")
    
    def create_complete_analysis(self, results_dict, save=True):
        """
        Create a complete visual analysis from results
        
        Args:
            results_dict: Dictionary containing all analysis results
            save: Whether to save plots
        """
        print("Creating complete visual analysis...")
        
        # 1. Dynamics Plots
        if 'trajectories' in results_dict:
            print("  - Creating dynamics plots...")
            t = results_dict.get('time', self.config.get_time_array())
            
            for model_name, trajectories in results_dict['trajectories'].items():
                self.dynamics.plot_individual_cell_dynamics(
                    t, trajectories, [f"IC {i+1}" for i in range(len(trajectories))], 
                    model_name, save=save
                )
                
                self.dynamics.plot_immune_tumor_ratio(
                    t, trajectories, model_name, save=save
                )
        
        # 2. Phase Portraits
        if 'trajectories' in results_dict:
            print("  - Creating phase portraits...")
            for model_name, trajectories in results_dict['trajectories'].items():
                self.phase.plot_tumor_immune_phase_portrait(
                    trajectories, model_name, save=save
                )
                
                if len(trajectories) > 0:
                    self.phase.plot_3d_phase_portrait(
                        trajectories, model_name, save=save
                    )
        
        # 3. Comparison Plots
        if 'alpha_comparison' in results_dict:
            print("  - Creating alpha comparison plots...")
            for deriv_type, alpha_data in results_dict['alpha_comparison'].items():
                self.dynamics.plot_alpha_comparison(
                    results_dict.get('time', self.config.get_time_array()),
                    alpha_data, cell_type_index=0, derivative_type=deriv_type, save=save
                )
        
        if 'derivative_comparison' in results_dict:
            print("  - Creating derivative comparison plots...")
            for alpha, deriv_data in results_dict['derivative_comparison'].items():
                self.dynamics.plot_derivative_type_comparison(
                    results_dict.get('time', self.config.get_time_array()),
                    deriv_data, cell_type_index=0, alpha=alpha, save=save
                )
        
        # 4. Statistical Analysis
        if 'performance_metrics' in results_dict:
            print("  - Creating statistical plots...")
            metrics = results_dict['performance_metrics']
            
            # Performance comparison
            self.comparison.plot_performance_comparison(
                metrics, metric_name='RMSE', save=save
            )
            
            # Prediction accuracy plots
            if 'predictions' in results_dict:
                for model_name, pred_data in results_dict['predictions'].items():
                    if 'actual' in pred_data and 'predicted' in pred_data:
                        self.statistics.plot_prediction_accuracy(
                            pred_data['actual'], pred_data['predicted'], 
                            model_name, save=save
                        )
        
        # 5. Summary Dashboard
        if 'summary' in results_dict:
            print("  - Creating summary dashboard...")
            self.statistics.plot_summary_dashboard(
                results_dict['summary'], save=save
            )
        
        print(f"Complete analysis saved to: {self.output_dir}")
    
    def quick_comparison(self, t, integer_trajectory, fractional_trajectory, 
                        initial_condition, save=True):
        """
        Quick comparison between integer and fractional models
        """
        trajectories_dict = {
            'Integer': integer_trajectory,
            'Fractional': fractional_trajectory
        }
        
        # Plot all cell types
        for i, cell_label in enumerate(self.config.cell_labels):
            self.comparison.plot_model_comparison(
                t, trajectories_dict, cell_type_index=i, 
                initial_condition=initial_condition, save=save
            )
        
        # Phase portrait comparison
        self.phase.plot_tumor_immune_phase_portrait(
            [integer_trajectory, fractional_trajectory], 
            'Integer vs Fractional', save=save
        )
    
    def alpha_sensitivity_analysis(self, t, alpha_results, derivative_type, save=True):
        """
        Complete alpha sensitivity analysis visualization
        """
        # Time series for different alphas
        self.dynamics.plot_alpha_comparison(
            t, alpha_results, cell_type_index=0, 
            derivative_type=derivative_type, save=save
        )
        
        # Phase portraits for different alphas
        self.phase.plot_alpha_phase_comparison(
            alpha_results, derivative_type, save=save
        )
        
        # Final values heatmap
        alpha_values = sorted(alpha_results.keys())
        final_values = np.array([alpha_results[alpha][-1] for alpha in alpha_values])
        
        self.comparison.plot_alpha_heatmap(
            alpha_values, final_values, derivative_type, save=save
        )
    
    def model_performance_analysis(self, actual_data, predicted_data, model_names, save=True):
        """
        Comprehensive model performance analysis
        """
        for i, model_name in enumerate(model_names):
            # Prediction accuracy
            self.statistics.plot_prediction_accuracy(
                actual_data[i], predicted_data[i], model_name, save=save
            )
            
            # Residuals analysis
            self.statistics.plot_residuals(
                actual_data[i], predicted_data[i], model_name, save=save
            )
            
            # Error distribution
            errors = actual_data[i] - predicted_data[i]
            self.statistics.plot_error_distribution(
                errors, model_name, save=save
            )
    
    def create_publication_figures(self, results_dict, save=True):
        """
        Create high-quality figures suitable for publication
        """
        print("Creating publication-quality figures...")
        
        # Set high DPI for publication
        original_dpi = self.config.viz_params['dpi']
        self.dynamics.dpi = 600
        self.phase.dpi = 600
        self.comparison.dpi = 600
        self.statistics.dpi = 600
        
        try:
            # Main results figure
            if 'main_results' in results_dict:
                self.create_complete_analysis(results_dict['main_results'], save=save)
            
            # Key comparison figure
            if 'key_comparison' in results_dict:
                data = results_dict['key_comparison']
                self.quick_comparison(
                    data['time'], data['integer'], data['fractional'], 
                    data['initial_condition'], save=save
                )
            
        finally:
            # Restore original DPI
            self.dynamics.dpi = original_dpi
            self.phase.dpi = original_dpi
            self.comparison.dpi = original_dpi
            self.statistics.dpi = original_dpi
        
        print("Publication figures created!")


def create_master_visualization(output_dir='plots_output'):
    """Factory function to create MasterVisualization"""
    return MasterVisualization(output_dir)


# Convenience functions for quick access
def plot_dynamics(t, trajectories, model_name, output_dir='plots_output'):
    """Quick function to plot dynamics"""
    plotter = create_dynamics_plotter(output_dir)
    plotter.plot_individual_cell_dynamics(
        t, trajectories, [f"IC {i+1}" for i in range(len(trajectories))], model_name
    )


def plot_phase_portrait(trajectories, model_name, output_dir='plots_output'):
    """Quick function to plot phase portrait"""
    plotter = create_phase_portrait_plotter(output_dir)
    plotter.plot_tumor_immune_phase_portrait(trajectories, model_name)


def plot_comparison(t, trajectories_dict, output_dir='plots_output'):
    """Quick function to plot model comparison"""
    plotter = create_comparison_plotter(output_dir)
    for i in range(4):  # All cell types
        plotter.plot_model_comparison(t, trajectories_dict, cell_type_index=i)


# Export all classes and functions
__all__ = [
    'DynamicsPlotter', 'PhasePortraitPlotter', 'ComparisonPlotter', 'StatisticsPlotter',
    'MasterVisualization', 'create_master_visualization',
    'create_dynamics_plotter', 'create_phase_portrait_plotter', 
    'create_comparison_plotter', 'create_statistics_plotter',
    'plot_dynamics', 'plot_phase_portrait', 'plot_comparison'
]
