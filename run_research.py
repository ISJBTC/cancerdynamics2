#!/usr/bin/env python3
"""
Simple run script for Cancer Dynamics Research Framework

This script provides easy access to different analysis modes without command line arguments.
"""

import sys
import os
from main import CancerDynamicsResearch

def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("CANCER DYNAMICS RESEARCH FRAMEWORK")
    print("="*60)
    print("Select analysis mode:")
    print("1. Quick Demo (5-10 minutes)")
    print("2. Basic Dynamics Analysis (10-15 minutes)")
    print("3. Alpha Sensitivity Analysis (15-20 minutes)")
    print("4. Neural Network Training (20-30 minutes)")
    print("5. Full Research Pipeline (45-60 minutes)")
    print("6. Custom Analysis")
    print("0. Exit")
    print("-"*60)

def run_demo():
    """Run quick demo"""
    print("\nRunning Quick Demo...")
    research = CancerDynamicsResearch('demo_output')
    research.quick_demo()
    print("\n✓ Demo completed! Check 'demo_output' folder for results.")

def run_basic():
    """Run basic dynamics analysis"""
    print("\nRunning Basic Dynamics Analysis...")
    research = CancerDynamicsResearch('basic_output')
    research.initialize_models()
    research.run_basic_dynamics_analysis()
    print("\n✓ Basic analysis completed! Check 'basic_output' folder for results.")

def run_alpha():
    """Run alpha sensitivity analysis"""
    print("\nRunning Alpha Sensitivity Analysis...")
    research = CancerDynamicsResearch('alpha_output')
    research.initialize_models()
    
    # Use subset for faster execution
    research.run_alpha_sensitivity_analysis(
        derivative_types=['caputo', 'riemann_liouville'],
        alpha_subset=[0.5, 0.8, 1.0, 1.2, 1.5]
    )
    print("\n✓ Alpha analysis completed! Check 'alpha_output' folder for results.")

def run_neural():
    """Run neural network training"""
    print("\nRunning Neural Network Training...")
    research = CancerDynamicsResearch('neural_output')
    research.initialize_models()
    research.initialize_neural_networks()
    research.run_basic_dynamics_analysis()  # Need data for training
    research.run_neural_network_training(train_epochs=30)
    print("\n✓ Neural network training completed! Check 'neural_output' folder for results.")

def run_full():
    """Run full research pipeline"""
    print("\nRunning Full Research Pipeline...")
    print("This will take 45-60 minutes. Continue? (y/n): ", end="")
    
    if input().lower() != 'y':
        print("Full pipeline cancelled.")
        return
    
    research = CancerDynamicsResearch('full_output')
    success = research.run_full_research_pipeline()
    
    if success:
        print("\n✓ Full research pipeline completed! Check 'full_output' folder for comprehensive results.")
    else:
        print("\n✗ Full research pipeline encountered errors. Check logs for details.")

def run_custom():
    """Run custom analysis with user choices"""
    print("\nCustom Analysis Configuration:")
    print("-"*40)
    
    # Get output directory
    output_dir = input("Output directory [custom_output]: ").strip()
    if not output_dir:
        output_dir = 'custom_output'
    
    # Get analysis components
    print("\nSelect components to run (y/n):")
    run_basic_dynamics = input("Basic dynamics analysis? [y]: ").strip().lower()
    if not run_basic_dynamics:
        run_basic_dynamics = 'y'
    
    run_alpha_analysis = input("Alpha sensitivity analysis? [y]: ").strip().lower()
    if not run_alpha_analysis:
        run_alpha_analysis = 'y'
    
    run_neural_training = input("Neural network training? [n]: ").strip().lower()
    
    # Alpha range for alpha analysis
    alpha_range = None
    if run_alpha_analysis == 'y':
        alpha_input = input("Alpha range [0.5-2.0]: ").strip()
        if alpha_input:
            try:
                parts = alpha_input.split('-')
                alpha_range = [float(parts[0]), float(parts[1])]
            except:
                print("Invalid alpha range, using default 0.5-2.0")
                alpha_range = [0.5, 2.0]
        else:
            alpha_range = [0.5, 2.0]
    
    # Derivative types
    derivative_types = ['caputo', 'riemann_liouville']
    if run_alpha_analysis == 'y':
        deriv_input = input("Derivative types [caputo,riemann_liouville]: ").strip()
        if deriv_input:
            derivative_types = [d.strip() for d in deriv_input.split(',')]
    
    print(f"\nStarting custom analysis in '{output_dir}'...")
    
    # Initialize research
    config_override = {}
    if alpha_range:
        import numpy as np
        config_override['alpha_values'] = np.arange(alpha_range[0], alpha_range[1] + 0.1, 0.2)
    
    research = CancerDynamicsResearch(output_dir, config_override)
    research.initialize_models()
    
    # Run selected components
    if run_basic_dynamics == 'y':
        print("\n➤ Running basic dynamics analysis...")
        research.run_basic_dynamics_analysis()
    
    if run_alpha_analysis == 'y':
        print("\n➤ Running alpha sensitivity analysis...")
        research.run_alpha_sensitivity_analysis(derivative_types=derivative_types)
    
    if run_neural_training == 'y':
        print("\n➤ Running neural network training...")
        research.initialize_neural_networks()
        if 'basic_dynamics' not in research.results:
            research.run_basic_dynamics_analysis()
        research.run_neural_network_training(train_epochs=20)
    
    # Run analysis if we have results
    if research.results:
        print("\n➤ Running comprehensive analysis...")
        research.run_comprehensive_analysis()
    
    print(f"\n✓ Custom analysis completed! Check '{output_dir}' folder for results.")

def main():
    """Main function"""
    while True:
        try:
            print_menu()
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print("\nExiting Cancer Dynamics Research Framework. Goodbye!")
                sys.exit(0)
            elif choice == '1':
                run_demo()
            elif choice == '2':
                run_basic()
            elif choice == '3':
                run_alpha()
            elif choice == '4':
                run_neural()
            elif choice == '5':
                run_full()
            elif choice == '6':
                run_custom()
            else:
                print("\nInvalid choice. Please enter 0-6.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
    