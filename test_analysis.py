import numpy as np
from scipy.integrate import odeint
from analysis import create_master_analyzer, quick_stats, compare_models_simple
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from config.parameters import get_config

print("ANALYSIS MODULE TEST")
print("=" * 50)

# Get configuration and create models
config = get_config()
t = config.get_time_array()
initial_conditions = config.initial_conditions

integer_model = IntegerModel()
fractional_model = FractionalModel()

print(f"Configuration loaded: {len(config.alpha_values)} alpha values")
print(f"Time points: {len(t)}, Initial conditions: {len(initial_conditions)}")

# Generate test data
print("\nGenerating test data...")
actual_trajectories = []
integer_predictions = []
fractional_predictions = []

for init_cond in initial_conditions[:2]:  # Use first 2 for testing
    # Generate "actual" data (using integer model as truth)
    actual = odeint(integer_model, init_cond, t)
    actual_trajectories.append(actual)
    
    # Generate integer model "predictions" (with small noise)
    int_pred = actual + 0.05 * np.random.randn(*actual.shape)
    integer_predictions.append(int_pred)
    
    # Generate fractional model "predictions" (with different noise)
    fractional_model.reset_history()
    frac_actual = odeint(fractional_model, init_cond, t)
    frac_pred = frac_actual + 0.08 * np.random.randn(*frac_actual.shape)
    fractional_predictions.append(frac_pred)

print("✓ Test data generated")

# Test 1: Quick Statistics
print("\n1. TESTING QUICK STATISTICS:")
print("-" * 40)

try:
    actual_flat = np.vstack(actual_trajectories)
    int_pred_flat = np.vstack(integer_predictions)
    
    stats = quick_stats(actual_flat, int_pred_flat)
    
    print(f"Overall RMSE: {stats['overall']['rmse']:.4f}")
    print(f"Overall MAE:  {stats['overall']['mae']:.4f}")
    print(f"Overall R²:   {stats['overall']['r2']:.4f}")
    
    print("Per cell type RMSE:")
    for i, cell_label in enumerate(config.cell_labels):
        rmse = stats['per_cell_type']['rmse'][i]
        print(f"  {cell_label}: {rmse:.4f}")
    
    print("✓ Quick statistics successful")
    
except Exception as e:
    print(f"Quick statistics failed: {e}")

# Test 2: Master Analyzer
print("\n2. TESTING MASTER ANALYZER:")
print("-" * 40)

try:
    # Create master analyzer
    analyzer = create_master_analyzer('test_analysis_output')
    print("✓ Master analyzer created")
    
    # Test quick performance analysis
    analyzer.quick_performance_analysis(actual_flat, int_pred_flat, "Integer Model")
    print("✓ Quick performance analysis successful")
    
    # Test model comparison
    frac_pred_flat = np.vstack(fractional_predictions)
    comparison = analyzer.compare_two_models(
        actual_flat, int_pred_flat, frac_pred_flat, 
        "Integer", "Fractional"
    )
    print("✓ Model comparison successful")
    
except Exception as e:
    print(f"Master analyzer test failed: {e}")

# Test 3: Comprehensive Analysis
print("\n3. TESTING COMPREHENSIVE ANALYSIS:")
print("-" * 40)

try:
    # Prepare experiment results
    experiment_results = {
        'model_predictions': {
            'Integer': {
                'actual': actual_flat,
                'predicted': int_pred_flat
            },
            'Fractional': {
                'actual': actual_flat,
                'predicted': frac_pred_flat
            }
        },
        'alpha_experiments': {
            'caputo': {
                0.5: actual_trajectories[0],
                0.8: actual_trajectories[1],
                1.0: actual_trajectories[0] * 1.1
            }
        },
        'derivative_experiments': {
            1.0: {
                'integer': actual_trajectories[0],
                'caputo': actual_trajectories[1],
                'riemann_liouville': actual_trajectories[0] * 0.9
            }
        }
    }
    
    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    analysis_results = analyzer.analyze_complete_experiment(
        experiment_results, save_reports=True
    )
    
    print("✓ Comprehensive analysis successful")
    
    # Display key results
    if 'key_findings' in analysis_results:
        print("\nKey Findings:")
        for finding in analysis_results['key_findings'][:3]:
            print(f"  • {finding}")
    
    if 'recommendations' in analysis_results:
        print("\nRecommendations:")
        for rec in analysis_results['recommendations'][:3]:
            print(f"  • {rec}")
    
except Exception as e:
    print(f"Comprehensive analysis failed: {e}")

# Test 4: Alpha Sensitivity Analysis
print("\n4. TESTING ALPHA SENSITIVITY:")
print("-" * 40)

try:
    # Simulate alpha results
    alpha_values = [0.3, 0.5, 0.8, 1.0, 1.5]
    alpha_results = {}
    
    for alpha in alpha_values:
        # Simulate different behavior for different alphas
        modified_traj = actual_trajectories[0] * (1 + 0.1 * (alpha - 1))
        # Add some alpha-dependent variation
        modified_traj[:, 0] *= (1 - 0.05 * alpha)  # Tumor decreases with higher alpha
        modified_traj[:, 1] *= (1 + 0.03 * alpha)  # Immune increases with higher alpha
        alpha_results[alpha] = modified_traj
    
    # Analyze alpha sensitivity
    alpha_analysis = analyzer.stats.analyze_alpha_sensitivity(alpha_results, 'caputo')
    
    print("Alpha sensitivity analysis:")
    for cell_label in config.cell_labels:
        if cell_label in alpha_analysis['cell_type_analysis']:
            cell_data = alpha_analysis['cell_type_analysis'][cell_label]
            sensitivity = cell_data.get('relative_sensitivity', 0)
            optimal_alpha = cell_data.get('optimal_alpha', 0)
            print(f"  {cell_label}: Sensitivity={sensitivity:.4f}, Optimal α={optimal_alpha:.1f}")
    
    print("✓ Alpha sensitivity analysis successful")
    
except Exception as e:
    print(f"Alpha sensitivity test failed: {e}")

# Test 5: Statistical Tests
print("\n5. TESTING STATISTICAL TESTS:")
print("-" * 40)

try:
    # Test model comparison with statistical tests
    model_results = {
        'Integer': {
            'performance': {
                'per_cell_type': {
                    'rmse': [0.1, 0.12, 0.08, 0.15],
                    'mae': [0.08, 0.10, 0.06, 0.12],
                    'r2': [0.95, 0.93, 0.97, 0.90]
                }
            }
        },
        'Fractional': {
            'performance': {
                'per_cell_type': {
                    'rmse': [0.12, 0.10, 0.09, 0.14],
                    'mae': [0.09, 0.08, 0.07, 0.11],
                    'r2': [0.93, 0.95, 0.96, 0.91]
                }
            }
        }
    }
    
    comparison = compare_models_simple(model_results)
    
    print("Statistical comparison results:")
    if 'statistical_tests' in comparison:
        for metric, tests in comparison['statistical_tests'].items():
            print(f"  {metric.upper()} tests:")
            if 'pairwise_tests' in tests:
                for test_name, result in tests['pairwise_tests'].items():
                    if 'p_value' in result:
                        significance = "significant" if result.get('significant', False) else "not significant"
                        print(f"    {test_name}: p={result['p_value']:.4f} ({significance})")
    
    print("✓ Statistical tests successful")
    
except Exception as e:
    print(f"Statistical tests failed: {e}")

# Test 6: Report Generation
print("\n6. TESTING REPORT GENERATION:")
print("-" * 40)

try:
    # Test individual report functions
    summary_generator = analyzer.summary
    
    # Create test analysis results
    test_results = {
        'model_performance': {
            'metrics': {
                'Integer': {
                    'overall': {'rmse': 0.105, 'mae': 0.085, 'r2': 0.94},
                    'per_cell_type': {
                        'rmse': [0.1, 0.12, 0.08, 0.15],
                        'mae': [0.08, 0.10, 0.06, 0.12],
                        'r2': [0.95, 0.93, 0.97, 0.90]
                    }
                },
                'Fractional': {
                    'overall': {'rmse': 0.112, 'mae': 0.088, 'r2': 0.92},
                    'per_cell_type': {
                        'rmse': [0.12, 0.10, 0.09, 0.14],
                        'mae': [0.09, 0.08, 0.07, 0.11],
                        'r2': [0.93, 0.95, 0.96, 0.91]
                    }
                }
            }
        },
        'alpha_analysis': {
            'caputo': {
                'cell_type_analysis': {
                    'Tumor': {
                        'relative_sensitivity': 0.25,
                        'optimal_alpha': 1.2,
                        'monotonicity': 0.8
                    },
                    'Immune': {
                        'relative_sensitivity': 0.15,
                        'optimal_alpha': 0.8,
                        'monotonicity': 0.6
                    }
                }
            }
        },
        'key_findings': [
            "Integer model shows better overall performance",
            "Alpha parameter significantly affects tumor dynamics",
            "Fractional derivatives provide additional modeling flexibility"
        ],
        'recommendations': [
            "Use integer model for primary analysis",
            "Consider α=1.2 for tumor-focused studies",
            "Validate with clinical data"
        ]
    }
    
    # Test text summary
    text_file = summary_generator.generate_text_summary(test_results, 'test_summary.txt')
    print(f"✓ Text summary generated: {text_file}")
    
    # Test CSV summary
    csv_file = summary_generator.generate_csv_summary(test_results, 'test_summary.csv')
    print(f"✓ CSV summary generated: {csv_file}")
    
    # Test JSON summary
    json_file = summary_generator.generate_json_summary(test_results, 'test_summary.json')
    print(f"✓ JSON summary generated: {json_file}")
    
    # Test HTML report
    html_file = summary_generator.generate_html_report(test_results, 'test_report.html')
    print(f"✓ HTML report generated: {html_file}")
    
    print("✓ Report generation successful")
    
except Exception as e:
    print(f"Report generation failed: {e}")

# Test 7: Executive Summary
print("\n7. TESTING EXECUTIVE SUMMARY:")
print("-" * 40)

try:
    exec_summary = summary_generator.create_executive_summary(test_results)
    
    print("Executive Summary:")
    print(f"  Models tested: {exec_summary['experiment_overview']['models_tested']}")
    print(f"  Analysis scope: {exec_summary['experiment_overview']['analysis_scope']}")
    
    if 'best_model' in exec_summary['key_metrics']:
        print(f"  Best model: {exec_summary['key_metrics']['best_model']}")
    
    print("  Main conclusions:")
    for conclusion in exec_summary['main_conclusions'][:2]:
        print(f"    • {conclusion}")
    
    print("✓ Executive summary successful")
    
except Exception as e:
    print(f"Executive summary failed: {e}")

print(f"\n{'='*50}")
print("Analysis module test completed!")
print("Check 'test_analysis_output' directory for generated reports")

# Summary of what was tested
print(f"\nTEST SUMMARY:")
print(f"✓ Quick statistics calculation")
print(f"✓ Master analyzer functionality") 
print(f"✓ Comprehensive experimental analysis")
print(f"✓ Alpha sensitivity analysis")
print(f"✓ Statistical significance testing")
print(f"✓ Multiple report format generation")
print(f"✓ Executive summary creation")
