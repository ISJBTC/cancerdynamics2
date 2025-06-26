import numpy as np
from scipy.integrate import odeint
from visualization import create_master_visualization, plot_dynamics
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from config.parameters import get_config

print("VISUALIZATION MODULE TEST")
print("=" * 50)

# Get configuration
config = get_config()
t = config.get_time_array()
initial_conditions = config.initial_conditions

# Create models
integer_model = IntegerModel()
fractional_model = FractionalModel()

print(f"Time points: {len(t)}")
print(f"Initial conditions: {len(initial_conditions)}")

# Generate test trajectories
print("\nGenerating test trajectories...")
integer_trajectories = []
fractional_trajectories = []

for init_cond in initial_conditions:
    # Integer model
    int_traj = odeint(integer_model, init_cond, t)
    integer_trajectories.append(int_traj)
    
    # Fractional model
    fractional_model.reset_history()
    frac_traj = odeint(fractional_model, init_cond, t)
    fractional_trajectories.append(frac_traj)

print("✓ Trajectories generated successfully")

# Test individual plotters
print("\n1. TESTING INDIVIDUAL PLOTTERS:")
print("-" * 40)

try:
    # Test dynamics plotter
    print("Testing dynamics plots...")
    plot_dynamics(t, integer_trajectories, "Integer Test")
    print("✓ Dynamics plots successful")
    
    # Test master visualization
    print("Testing master visualization...")
    master_viz = create_master_visualization('test_plots')
    print("✓ Master visualization created")
    
    # Test quick comparison
    print("Testing quick comparison...")
    master_viz.quick_comparison(
        t, integer_trajectories[0], fractional_trajectories[0], 
        initial_conditions[0], save=True
    )
    print("✓ Quick comparison successful")
    
except Exception as e:
    print(f"Individual plotter test failed: {e}")

# Test complete analysis
print("\n2. TESTING COMPLETE ANALYSIS:")
print("-" * 40)

try:
    # Prepare results dictionary
    results_dict = {
        'time': t,
        'trajectories': {
            'Integer': integer_trajectories,
            'Fractional': fractional_trajectories
        },
        'alpha_comparison': {
            'caputo': {
                0.5: fractional_trajectories[0],
                0.8: fractional_trajectories[1],
                1.0: integer_trajectories[0]
            }
        },
        'derivative_comparison': {
            1.0: {
                'integer': integer_trajectories[0],
                'caputo': fractional_trajectories[0]
            }
        },
        'performance_metrics': {
            'Integer': [0.1, 0.2, 0.15, 0.12],  # RMSE for each cell type
            'Fractional': [0.12, 0.18, 0.14, 0.13]
        },
        'predictions': {
            'Integer': {
                'actual': integer_trajectories[0][:20],
                'predicted': integer_trajectories[0][:20] + 0.1 * np.random.randn(20, 4)
            }
        }
    }
    
    # Create complete analysis
    print("Creating complete analysis...")
    master_viz.create_complete_analysis(results_dict, save=True)
    print("✓ Complete analysis successful")
    
except Exception as e:
    print(f"Complete analysis test failed: {e}")

# Test alpha sensitivity
print("\n3. TESTING ALPHA SENSITIVITY:")
print("-" * 40)

try:
    # Simulate alpha results
    alpha_results = {}
    alpha_values = [0.3, 0.5, 0.8, 1.0, 1.5]
    
    for alpha in alpha_values:
        # Simulate different behavior for different alphas
        modified_traj = integer_trajectories[0] * (1 + 0.1 * (alpha - 1))
        alpha_results[alpha] = modified_traj
    
    print("Testing alpha sensitivity analysis...")
    master_viz.alpha_sensitivity_analysis(t, alpha_results, 'caputo', save=True)
    print("✓ Alpha sensitivity analysis successful")
    
except Exception as e:
    print(f"Alpha sensitivity test failed: {e}")

print(f"\n{'='*50}")
print("Visualization module test completed!")
print(f"Check output in 'test_plots' directory")
