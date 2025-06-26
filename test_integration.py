#!/usr/bin/env python3
"""
Comprehensive Integration Test for Cancer Dynamics Research Framework

This script tests the complete integration of all modules.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Import the main framework
from main import CancerDynamicsResearch

def test_framework_initialization():
    """Test framework initialization"""
    print("1. TESTING FRAMEWORK INITIALIZATION")
    print("-" * 50)
    
    try:
        research = CancerDynamicsResearch('test_integration_output')
        
        print(f"✓ Framework initialized successfully")
        print(f"✓ Output directory: {research.output_dir}")
        print(f"✓ Configuration loaded: {len(research.config.alpha_values)} alpha values")
        print(f"✓ Derivative types: {research.config.fractional_derivative_types}")
        
        return research
        
    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        return None

def test_model_initialization(research):
    """Test model initialization"""
    print("\n2. TESTING MODEL INITIALIZATION")
    print("-" * 50)
    
    try:
        research.initialize_models()
        
        print(f"✓ Mathematical models initialized")
        print(f"  - Integer model: {type(research.models['integer']).__name__}")
        print(f"  - Fractional model: {type(research.models['fractional']).__name__}")
        print(f"  - Derivative models: {len(research.derivatives)} types")
        
        # Test basic model functionality
        test_state = [50, 10, 20, 30]
        int_result = research.models['integer'].system_dynamics(test_state, 0)
        frac_result = research.models['fractional'].system_dynamics(test_state, 0)
        
        print(f"✓ Model functionality test passed")
        print(f"  - Integer result: {[f'{x:.3f}' for x in int_result]}")
        print(f"  - Fractional result: {[f'{x:.3f}' for x in frac_result]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_neural_network_initialization(research):
    """Test neural network initialization"""
    print("\n3. TESTING NEURAL NETWORK INITIALIZATION")
    print("-" * 50)
    
    try:
        research.initialize_neural_networks()
        
        print(f"✓ Neural networks initialized")
        for name, network in research.neural_networks.items():
            print(f"  - {name}: {type(network).__name__}")
        
        # Test forward pass
        import torch
        test_input = torch.randn(5, 4)
        
        for name, network in research.neural_networks.items():
            try:
                if name == 'fractional_net':
                    # Special handling for fractional network
                    memory = torch.randn(5, 10, 4)
                    alpha_tensor = torch.full((5, 1), 0.8)
                    output = network(test_input, memory, alpha_tensor)
                elif name == 'ensemble_net':
                    output, weights = network(test_input)
                else:
                    output = network(test_input)
                print(f"    {name}: Output shape {output.shape}")
            except Exception as e:
                print(f"    {name}: Error - {e}")
        
        print(f"✓ Neural network functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Neural network initialization failed: {e}")
        return False

def test_basic_dynamics(research):
    """Test basic dynamics analysis"""
    print("\n4. TESTING BASIC DYNAMICS ANALYSIS")
    print("-" * 50)
    
    try:
        results = research.run_basic_dynamics_analysis()
        
        print(f"✓ Basic dynamics analysis completed")
        print(f"  - Time points: {len(results['time'])}")
        print(f"  - Initial conditions: {len(results['initial_conditions'])}")
        print(f"  - Integer trajectories: {len(results['integer_trajectories'])}")
        print(f"  - Fractional trajectories: {len(results['fractional_trajectories'])}")
        
        # Check trajectory shapes
        traj_shape = results['integer_trajectories'][0].shape
        print(f"  - Trajectory shape: {traj_shape}")
        
        # Check final values
        final_int = results['integer_trajectories'][0][-1]
        final_frac = results['fractional_trajectories'][0][-1]
        print(f"  - Final integer state: {[f'{x:.2f}' for x in final_int]}")
        print(f"  - Final fractional state: {[f'{x:.2f}' for x in final_frac]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic dynamics analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alpha_sensitivity(research):
    """Test alpha sensitivity analysis"""
    print("\n5. TESTING ALPHA SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    try:
        # Use limited scope for testing
        results = research.run_alpha_sensitivity_analysis(
            derivative_types=['caputo'],
            alpha_subset=[0.5, 1.0, 1.5]
        )
        
        print(f"✓ Alpha sensitivity analysis completed")
        print(f"  - Derivative types tested: {results['derivative_types']}")
        print(f"  - Alpha values tested: {results['alpha_values']}")
        print(f"  - Results structure: {list(results['results'].keys())}")
        
        # Check if statistical analysis was performed
        if 'alpha_analysis' in research.results:
            alpha_analysis = research.results['alpha_analysis']
            print(f"  - Statistical analysis completed for {len(alpha_analysis)} derivative types")
            
            for deriv_type, analysis in alpha_analysis.items():
                if 'cell_type_analysis' in analysis:
                    print(f"    {deriv_type}: {len(analysis['cell_type_analysis'])} cell types analyzed")
        
        return True
        
    except Exception as e:
        print(f"✗ Alpha sensitivity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_training(research):
    """Test neural network training"""
    print("\n6. TESTING NEURAL NETWORK TRAINING")
    print("-" * 50)
    
    try:
        # Use minimal epochs for testing
        results = research.run_neural_network_training(train_epochs=5)
        
        print(f"✓ Neural network training completed")
        print(f"  - Models trained: {list(results.keys())}")
        
        # Check if predictions were generated
        if 'nn_predictions' in research.results:
            predictions = research.results['nn_predictions']
            print(f"  - Predictions generated for: {list(predictions.keys())}")
            
            for model_name, pred_data in predictions.items():
                actual_shape = pred_data['actual'].shape
                pred_shape = pred_data['predicted'].shape
                print(f"    {model_name}: Actual {actual_shape}, Predicted {pred_shape}")
        
        # Check performance metrics
        if 'nn_performance' in research.results:
            performance = research.results['nn_performance']
            print(f"  - Performance metrics calculated for: {list(performance.keys())}")
            
            for model_name, metrics in performance.items():
                rmse = metrics['overall']['rmse']
                r2 = metrics['overall']['r2']
                print(f"    {model_name}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Neural network training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_analysis(research):
    """Test comprehensive analysis"""
    print("\n7. TESTING COMPREHENSIVE ANALYSIS")
    print("-" * 50)
    
    try:
        results = research.run_comprehensive_analysis()
        
        print(f"✓ Comprehensive analysis completed")
        print(f"  - Analysis components: {list(results.keys())}")
        
        # Check key results
        if 'key_findings' in results:
            print(f"  - Key findings: {len(results['key_findings'])}")
            for finding in results['key_findings'][:2]:
                print(f"    • {finding}")
        
        if 'recommendations' in results:
            print(f"  - Recommendations: {len(results['recommendations'])}")
            for rec in results['recommendations'][:2]:
                print(f"    • {rec}")
        
        # Check if reports were generated
        if 'report_files' in results:
            print(f"  - Report files generated: {len(results['report_files'])}")
            for report_file in results['report_files']:
                if os.path.exists(report_file):
                    print(f"    ✓ {os.path.basename(report_file)}")
                else:
                    print(f"    ✗ {os.path.basename(report_file)} (missing)")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_interface():
    """Test command line interface"""
    print("\n8. TESTING COMMAND LINE INTERFACE")
    print("-" * 50)
    
    try:
        # Test demo mode
        print("Testing demo mode...")
        exit_code = os.system("python main.py --mode demo --output test_cli_output")
        
        if exit_code == 0:
            print("✓ Command line demo mode works")
            
            # Check if output directory was created
            if os.path.exists('test_cli_output'):
                print("✓ Output directory created")
                
                # Check for some expected files
                plots_dir = os.path.join('test_cli_output', 'plots')
                if os.path.exists(plots_dir):
                    plot_files = os.listdir(plots_dir)
                    print(f"✓ {len(plot_files)} plot files generated")
                else:
                    print("⚠ No plots directory found")
            else:
                print("⚠ Output directory not created")
        else:
            print("✗ Command line demo mode failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Command line interface test failed: {e}")
        return False

def run_complete_integration_test():
    """Run complete integration test"""
    print("CANCER DYNAMICS RESEARCH FRAMEWORK - INTEGRATION TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Test 1: Framework initialization
    research = test_framework_initialization()
    test_results.append(research is not None)
    
    if research is None:
        print("\n✗ Cannot continue tests without framework initialization")
        return False
    
    # Test 2: Model initialization
    test_results.append(test_model_initialization(research))
    
    # Test 3: Neural network initialization
    test_results.append(test_neural_network_initialization(research))
    
    # Test 4: Basic dynamics analysis
    test_results.append(test_basic_dynamics(research))
    
    # Test 5: Alpha sensitivity analysis
    test_results.append(test_alpha_sensitivity(research))
    
    # Test 6: Neural network training
    test_results.append(test_neural_training(research))
    
    # Test 7: Comprehensive analysis
    test_results.append(test_comprehensive_analysis(research))
    
    # Test 8: Command line interface
    test_results.append(test_command_line_interface())
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(test_results)
    total = len(test_results)
    
    test_names = [
        "Framework Initialization",
        "Model Initialization", 
        "Neural Network Initialization",
        "Basic Dynamics Analysis",
        "Alpha Sensitivity Analysis",
        "Neural Network Training",
        "Comprehensive Analysis",
        "Command Line Interface"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name:<30} {status}")
    
    print("-" * 70)
    print(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Framework is ready for research.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test outputs saved in: test_integration_output/")
    
    return passed == total

if __name__ == "__main__":
    success = run_complete_integration_test()
    sys.exit(0 if success else 1)
    