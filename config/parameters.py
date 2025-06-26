import numpy as np


class ModelConfig:
    """Configuration class for cancer dynamics research"""

    def __init__(self):
        # Alpha values from 0 to 2 step 0.1 as requested
        self.alpha_values = np.arange(0.0, 2.1, 0.1)

        # Fractional derivative types to implement
        self.fractional_derivative_types = [
            'caputo',
            'riemann_liouville',
            'grunwald_letnikov',
            'hilfer',
            'gutan'
        ]

        # Simulation parameters from original code
        self.time_params = {
            'start': 0,
            'end': 4,
            'points': 41
        }

        # Initial conditions from original code
        self.initial_conditions = [
            [50, 10, 20, 30],  # Baseline
            [70, 5, 20, 30],   # High tumor / low immune
            [30, 20, 20, 30]   # Low tumor / high immune
        ]

        # Model parameters (from original code)
        self.model_params = {
            'r1': 0.4,      # Tumor growth rate
            'r2': 0.35,     # Immune growth rate
            'r3': 0.3,      # Memory cell growth rate
            'r4': 0.2,      # Stromal cell growth rate
            'K1': 80.0,     # Tumor carrying capacity
            'K2': 100.0,    # Immune carrying capacity
            'K3': 120.0,    # Stromal carrying capacity
            'a1': 0.012,    # Tumor inhibition by immune cells
            'a2': 0.02,     # Immune stimulation by tumor
            'a3': 0.004,    # Memory-stromal interaction
            'a4': 0.002,    # Tumor-stromal interaction
            'h': 0.01,      # Tumor natural death rate
            'k': 0.1,       # Saturation constant
            'd1': 0.05,     # Immune death rate
            'c1': 0.005,    # Nonlinear coupling constant 1
            'c2': 0.005     # Nonlinear coupling constant 2
        }

        # Neural network parameters from original code
        self.nn_params = {
            'input_size': 4,
            'hidden_layers': [64, 32],
            'output_size': 4,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate_integer': 0.001,
            'learning_rate_fractional': 0.0005,
            'max_grad_norm': 1.0,
            'num_trajectories': 100
        }

        # Visualization parameters
        self.viz_params = {
            'figure_size': (4, 4),
            'dpi': 300,
            'colors': ['#0072BD', '#D95319', '#77AC30'],  # Blue, Orange, Green
            'line_width': 1.5,
            'font_sizes': {
                'title': 10,
                'label': 9,
                'tick': 7,
                'legend': 7
            }
        }

        # Output directory settings
        self.output_params = {
            'plots_dir': 'plots_output',
            'results_dir': 'results',
            'data_dir': 'data'
        }

        # Cell type labels
        self.cell_labels = ['Tumor', 'Immune', 'Memory', 'Stromal']

    def get_time_array(self):
        """Get time array for simulation"""
        return np.linspace(
            self.time_params['start'],
            self.time_params['end'],
            self.time_params['points']
        )

    def get_alpha_range(self, start=None, end=None, step=None):
        """Get custom alpha range if needed"""
        if start is not None and end is not None:
            step = step or 0.1
            return np.arange(start, end + step/2, step)
        return self.alpha_values

    def update_model_params(self, **kwargs):
        """Update model parameters"""
        for key, value in kwargs.items():
            if key in self.model_params:
                self.model_params[key] = value
            else:
                print(f"Warning: Parameter '{key}' not recognized")

    def get_derivative_type_info(self):
        """Get information about fractional derivative types"""
        info = {
            'caputo': 'Caputo fractional derivative',
            'riemann_liouville': 'Riemann-Liouville fractional derivative',
            'grunwald_letnikov': 'Grünwald-Letnikov fractional derivative',
            'hilfer': 'Hilfer fractional derivative',
            'gutan': 'Gutan fractional derivative'
        }
        return info

    def print_config_summary(self):
        """Print configuration summary"""
        print("CANCER DYNAMICS RESEARCH CONFIGURATION")
        print("=" * 50)
        print(f"Alpha values: {len(self.alpha_values)} values from "
              f"{self.alpha_values[0]:.1f} to {self.alpha_values[-1]:.1f}")
        print(f"Derivative types: {', '.join(self.fractional_derivative_types)}")
        print(f"Initial conditions: {len(self.initial_conditions)} sets")
        print(f"Time range: {self.time_params['start']} to "
              f"{self.time_params['end']} with {self.time_params['points']} points")
        print(f"Neural network epochs: {self.nn_params['epochs']}")
        print(f"Output directory: {self.output_params['plots_dir']}")


# Global configuration instance
config = ModelConfig()


# Utility functions for easy access
def get_config():
    """Get the global configuration instance"""
    return config


def get_alpha_values():
    """Get all alpha values"""
    return config.alpha_values


def get_initial_conditions():
    """Get all initial conditions"""
    return config.initial_conditions


def get_time_array():
    """Get time array for simulation"""
    return config.get_time_array()


def get_derivative_types():
    """Get all fractional derivative types"""
    return config.fractional_derivative_types
    