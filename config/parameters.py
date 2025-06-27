import numpy as np


class ModelConfig:
    """Configuration class for cancer dynamics research with quantum effects"""

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

        # Initial conditions from original code - updated for better quantum dynamics
        self.initial_conditions = [
            [50, 10, 20, 30],  # Baseline
            [70, 5, 20, 30],   # High tumor / low immune
            [30, 20, 20, 30],  # Low tumor / high immune
            [40, 15, 25, 35],  # Balanced condition
            [60, 8, 15, 25]    # Additional test condition
        ]

        # Model parameters - COMPLETE SET including quantum parameters
        self.model_params = {
            # Growth rates (from your equations)
            'r1': 0.4,      # Tumor growth rate
            'r2': 0.35,     # Immune growth rate
            'r3': 0.3,      # Memory cell growth rate
            'r4': 0.2,      # Suppressor cell growth rate
            
            # Carrying capacities (K1, K2, K3, K4)
            'K1': 80.0,     # Tumor carrying capacity
            'K2': 100.0,    # Immune carrying capacity
            'K3': 120.0,    # Memory cell carrying capacity
            'K4': 120.0,    # Suppressor carrying capacity (NEW)
            
            # Interaction coefficients (a1, a2, a3, a4)
            'a1': 0.012,    # Tumor inhibition by immune cells
            'a2': 0.02,     # Immune stimulation by tumor
            'a3': 0.004,    # Memory-suppressor interaction
            'a4': 0.002,    # Tumor-suppressor interaction
            
            # Treatment/death rate
            'h': 0.01,      # Tumor natural death rate
            
            # Saturation coefficient
            'k': 0.1,       # Saturation constant
            
            # Natural death rate
            'd1': 0.05,     # Immune death rate (NEW - from your equations)
            
            # Nonlinear coupling constants (c1, c2)
            'c1': 0.005,    # Trigonometric coupling constant 1 (sin term)
            'c2': 0.005,    # Trigonometric coupling constant 2 (cos term)
            
            # QUANTUM PARAMETERS (NEW)
            'p': 0.1,                    # Quantum momentum parameter
            'q': 1.0,                    # Quantum parameter for pressure calculation
            'quantum_threshold': 1e-5,   # 10^-5 threshold for quantum effects
            
            # Fractional model specific parameters
            'fractional_scaling_T': 0.05,  # T_frac scaling factor
            'fractional_scaling_I': 0.05,  # I_frac scaling factor
            'memory_influence': 0.15,      # Fractional influence on M and S
            
            # Stability parameters
            'min_population': 1e-10,    # Minimum population to prevent extinction
            'max_derivative': 10.0,     # Maximum derivative value for stability
            'suppressor_quadratic': 0.001  # Coefficient for (S/K4)^2 term
        }

        # Neural network parameters from original code - enhanced
        self.nn_params = {
            'input_size': 4,
            'hidden_layers': [64, 32],
            'output_size': 4,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate_integer': 0.001,
            'learning_rate_fractional': 0.0005,
            'max_grad_norm': 1.0,
            'num_trajectories': 100,
            
            # Enhanced parameters for quantum dynamics
            'validation_split': 0.2,
            'early_stopping': True,
            'patience': 15,
            'dropout_rate': 0.1
        }

        # Visualization parameters - enhanced for quantum analysis
        self.viz_params = {
            'figure_size': (4, 4),
            'dpi': 300,
            'colors': ['#0072BD', '#D95319', '#77AC30', '#EDB120', '#7E2F8E'],  # 5 colors for more conditions
            'line_width': 1.5,
            'font_sizes': {
                'title': 10,
                'label': 9,
                'tick': 7,
                'legend': 7
            },
            'quantum_colors': {
                'active': '#FF6B6B',    # Red for active quantum effects
                'inactive': '#95E1D3'   # Green for inactive quantum effects
            }
        }

        # Output directory settings
        self.output_params = {
            'plots_dir': 'plots_output',
            'results_dir': 'results',
            'data_dir': 'data',
            'quantum_analysis_dir': 'quantum_analysis'
        }

        # Cell type labels
        self.cell_labels = ['Tumor', 'Immune', 'Memory', 'Suppressor']
        
        # Quantum analysis parameters
        self.quantum_params = {
            'analyze_quantum_transitions': True,
            'track_quantum_pressure': True,
            'quantum_threshold_sensitivity': [1e-6, 1e-5, 1e-4],  # Different thresholds to test
            'momentum_parameter_range': np.arange(0.05, 0.5, 0.05),  # p values to test
            'quantum_visualization': True
        }

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
                print(f"Updated {key} = {value}")
            else:
                print(f"Warning: Parameter '{key}' not recognized")

    def update_quantum_params(self, **kwargs):
        """Update quantum-specific parameters"""
        quantum_keys = ['p', 'q', 'quantum_threshold']
        for key, value in kwargs.items():
            if key in quantum_keys:
                self.model_params[key] = value
                print(f"Updated quantum parameter {key} = {value}")
            else:
                print(f"Warning: Quantum parameter '{key}' not recognized")

    def get_derivative_type_info(self):
        """Get information about fractional derivative types"""
        info = {
            'caputo': 'Caputo fractional derivative - memory of the rate of change',
            'riemann_liouville': 'Riemann-Liouville fractional derivative - classical approach',
            'grunwald_letnikov': 'Grünwald-Letnikov fractional derivative - discrete approximation',
            'hilfer': 'Hilfer fractional derivative - interpolates between Caputo and R-L',
            'gutan': 'Gutan fractional derivative - improved convergence properties'
        }
        return info

    def get_quantum_info(self):
        """Get information about quantum parameters and their biological meaning"""
        info = {
            'quantum_threshold': f'{self.model_params["quantum_threshold"]:.0e} - Population threshold for quantum effects',
            'momentum_parameter': f'{self.model_params["p"]:.3f} - Quantum momentum affecting pressure terms',
            'quantum_pressure_T': 'Q_τ = -p²q/(2T) - Tumor quantum pressure (prevents extinction)',
            'quantum_pressure_I': 'Q_i = -p²q/(2I) - Immune quantum pressure (enables evasion)',
            'biological_meaning': 'Quantum effects explain cancer resilience, therapy resistance, and recurrence'
        }
        return info

    def print_config_summary(self):
        """Print comprehensive configuration summary"""
        print("CANCER DYNAMICS RESEARCH CONFIGURATION (with Quantum Effects)")
        print("=" * 65)
        
        # Basic configuration
        print(f"Alpha values: {len(self.alpha_values)} values from "
              f"{self.alpha_values[0]:.1f} to {self.alpha_values[-1]:.1f}")
        print(f"Derivative types: {', '.join(self.fractional_derivative_types)}")
        print(f"Initial conditions: {len(self.initial_conditions)} sets")
        print(f"Time range: {self.time_params['start']} to "
              f"{self.time_params['end']} with {self.time_params['points']} points")
        
        # Model parameters
        print(f"\nMODEL PARAMETERS:")
        print(f"Growth rates: r1={self.model_params['r1']}, r2={self.model_params['r2']}, "
              f"r3={self.model_params['r3']}, r4={self.model_params['r4']}")
        print(f"Carrying capacities: K1={self.model_params['K1']}, K2={self.model_params['K2']}, "
              f"K3={self.model_params['K3']}, K4={self.model_params['K4']}")
        print(f"Interaction coefficients: a1={self.model_params['a1']}, a2={self.model_params['a2']}, "
              f"a3={self.model_params['a3']}, a4={self.model_params['a4']}")
        print(f"Nonlinear coupling: c1={self.model_params['c1']}, c2={self.model_params['c2']}")
        
        # Quantum parameters
        print(f"\nQUANTUM PARAMETERS:")
        print(f"Momentum parameter: p = {self.model_params['p']}")
        print(f"Quantum threshold: {self.model_params['quantum_threshold']:.0e}")
        print(f"Quantum parameter: q = {self.model_params['q']}")
        
        # Analysis settings
        print(f"\nANALYSIS SETTINGS:")
        print(f"Neural network epochs: {self.nn_params['epochs']}")
        print(f"Output directory: {self.output_params['plots_dir']}")
        print(f"Quantum analysis: {self.quantum_params['analyze_quantum_transitions']}")

    def print_quantum_summary(self):
        """Print detailed quantum parameters summary"""
        print("\nQUANTUM MECHANICS IN CANCER DYNAMICS")
        print("=" * 45)
        
        quantum_info = self.get_quantum_info()
        for key, description in quantum_info.items():
            print(f"{key}: {description}")
        
        print(f"\nQuantum Exclusion Principle: Prevents complete tumor/immune extinction")
        print(f"Quantum Tunneling: Enables immune evasion and metastasis")
        print(f"Active when populations < {self.model_params['quantum_threshold']:.0e}")

    def get_test_conditions_for_quantum(self):
        """Get specific initial conditions that will trigger quantum effects"""
        # These conditions are designed to push populations near quantum threshold
        quantum_test_conditions = [
            [1e-4, 1e-4, 20, 30],    # Both T and I near quantum threshold
            [50, 1e-4, 20, 30],      # Only I near quantum threshold  
            [1e-4, 10, 20, 30],      # Only T near quantum threshold
            [1e-6, 1e-6, 20, 30],    # Both below quantum threshold
            [0.1, 0.1, 20, 30]       # Both above but close to threshold
        ]
        return quantum_test_conditions

    def validate_parameters(self):
        """Validate parameter consistency and ranges"""
        validation_results = []
        
        # Check positive parameters
        positive_params = ['r1', 'r2', 'r3', 'r4', 'K1', 'K2', 'K3', 'K4', 
                          'a1', 'a2', 'a3', 'a4', 'h', 'k', 'd1', 'p', 'q']
        for param in positive_params:
            if self.model_params[param] <= 0:
                validation_results.append(f"Warning: {param} should be positive")
        
        # Check quantum threshold
        if self.model_params['quantum_threshold'] >= 1:
            validation_results.append("Warning: quantum_threshold should be much smaller than typical populations")
        
        # Check coupling constants
        if self.model_params['c1'] > 0.1 or self.model_params['c2'] > 0.1:
            validation_results.append("Warning: Large coupling constants may cause instability")
        
        if validation_results:
            print("Parameter Validation Issues:")
            for issue in validation_results:
                print(f"  - {issue}")
        else:
            print("✓ All parameters validated successfully")
        
        return len(validation_results) == 0


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


def get_quantum_initial_conditions():
    """Get initial conditions designed to test quantum effects"""
    return config.get_test_conditions_for_quantum()


def get_time_array():
    """Get time array for simulation"""
    return config.get_time_array()


def get_derivative_types():
    """Get all fractional derivative types"""
    return config.fractional_derivative_types


def get_quantum_parameters():
    """Get quantum-specific parameters"""
    return {
        'p': config.model_params['p'],
        'q': config.model_params['q'],
        'quantum_threshold': config.model_params['quantum_threshold']
    }


def update_quantum_parameters(p=None, q=None, threshold=None):
    """Convenience function to update quantum parameters"""
    updates = {}
    if p is not None:
        updates['p'] = p
    if q is not None:
        updates['q'] = q
    if threshold is not None:
        updates['quantum_threshold'] = threshold
    
    if updates:
        config.update_quantum_params(**updates)
    
    return get_quantum_parameters()


def print_full_config():
    """Print complete configuration including quantum details"""
    config.print_config_summary()
    config.print_quantum_summary()
    print(f"\nValidating parameters...")
    config.validate_parameters()


# Quantum-specific utility functions
def is_quantum_active(population):
    """Check if quantum effects are active for a given population"""
    return population <= config.model_params['quantum_threshold']


def calculate_quantum_pressure(population, momentum_p=None, quantum_q=None):
    """Calculate quantum pressure for a given population"""
    if momentum_p is None:
        momentum_p = config.model_params['p']
    if quantum_q is None:
        quantum_q = config.model_params['q']
    
    if is_quantum_active(population):
        return 0.0  # No pressure when below threshold
    else:
        return -(momentum_p**2 * quantum_q) / (2 * population)