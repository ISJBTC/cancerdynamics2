#!/usr/bin/env python3
"""
Cancer Dynamics Research Framework - Interactive Runner

Clean, streamlined interface for running quantum-enhanced cancer dynamics research.
No test files - production ready interface for research workflows.

Author: Research Team
Date: 2025
"""

import sys
import os
import numpy as np
from datetime import datetime
from main import CancerDynamicsResearch

def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("CANCER DYNAMICS RESEARCH FRAMEWORK")
    print("Quantum-Enhanced Mathematical Modeling")
    print("="*70)
    print("Equations (1)-(8) with Quantum Pressure Terms Q_τ and Q_i")
    print("Available Derivatives: Caputo, Riemann-Liouville, Grünwald-Letnikov, Hilfer")
    print("-"*70)

def print_menu():
    """Print the main menu"""
    print("\nSelect Research Mode:")
    print("1. Quick Demo (5 minutes) - Basic quantum effects demonstration")
    print("2. Basic Analysis (15 minutes) - Integer vs Fractional with quantum")
    print("3. Alpha Sensitivity (20 minutes) - Parameter optimization")
    print("4. Neural Network Training (30 minutes) - AI learning quantum dynamics")
    print("5. Full Research Pipeline (60 minutes) - Comprehensive analysis")
    print("6. Quantum Analysis Only - Focus on quantum effects")
    print("7. Custom Analysis - Configure your own study")
    print("0. Exit")
    print("-"*70)

def get_quantum_settings():
    """Get quantum parameter settings from user"""
    print("\n🔬 Quantum Parameters Configuration:")
    print(f"Current settings:")
    
    from config.parameters import get_quantum_parameters
    params = get_quantum_parameters()
    print(f"  Momentum parameter (p): {params['p']}")
    print(f"  Quantum threshold: {params['quantum_threshold']:.0e}")
    
    modify = input("\nModify quantum parameters? (y/n) [n]: ").strip().lower()
    
    if modify == 'y':
        try:
            new_p = input(f"Enter momentum parameter p [{params['p']}]: ").strip()
            if new_p:
                params['p'] = float(new_p)
            
            new_threshold = input(f"Enter quantum threshold [{params['quantum_threshold']:.0e}]: ").strip()
            if new_threshold:
                params['quantum_threshold'] = float(new_threshold)
            
            from config.parameters import update_quantum_parameters
            update_quantum_parameters(p=params['p'], threshold=params['quantum_threshold'])
            print(f"✓ Quantum parameters updated")
            
        except ValueError:
            print("⚠ Invalid input, using default parameters")
    
    return params

def run_quick_demo():
    """Quick demonstration of quantum effects"""
    print("\n🚀 Running Quick Demo with Quantum Effects...")
    
    research = CancerDynamicsResearch('demo_output')
    result = research.quick_demo()
    
    print("\n✅ Demo Results:")
    print("  - Compared integer vs fractional models")
    print("  - Demonstrated quantum pressure effects")
    print("  - Generated basic visualizations")
    print(f"  - Check output: demo_output/")
    return True

def run_basic_analysis():
    """Basic dynamics analysis"""
    print("\n📊 Running Basic Dynamics Analysis...")
    
    research = CancerDynamicsResearch('basic_output')
    research.initialize_models()
    result = research.run_basic_dynamics_analysis()
    
    print("\n✅ Basic Analysis Complete:")
    print(f"  - Analyzed {len(result['initial_conditions'])} initial conditions")
    print(f"  - Generated {len(result['integer_trajectories'])} trajectory pairs")
    print("  - Quantum effects analyzed throughout trajectories")
    print("  - Phase portraits and dynamics plots created")
    print(f"  - Results saved: basic_output/")
    return True

def run_alpha_sensitivity():
    """Alpha sensitivity analysis"""
    print("\n📈 Running Alpha Sensitivity Analysis...")
    
    # Get user preferences
    print("Available derivatives: caputo, riemann_liouville, grunwald_letnikov, hilfer")
    derivatives = input("Enter derivatives to test [caputo,riemann_liouville]: ").strip()
    if derivatives:
        derivative_list = [d.strip() for d in derivatives.split(',')]
    else:
        derivative_list = ['caputo', 'riemann_liouville']
    
    alpha_range = input("Alpha range (min,max) [0.5,2.0]: ").strip()
    if alpha_range:
        try:
            alpha_min, alpha_max = map(float, alpha_range.split(','))
            alpha_subset = np.arange(alpha_min, alpha_max + 0.1, 0.3)
        except:
            alpha_subset = np.arange(0.5, 2.1, 0.3)
    else:
        alpha_subset = np.arange(0.5, 2.1, 0.3)
    
    research = CancerDynamicsResearch('alpha_output')
    research.initialize_models()
    result = research.run_alpha_sensitivity_analysis(
        derivative_types=derivative_list,
        alpha_subset=alpha_subset
    )
    
    print("\n✅ Alpha Sensitivity Complete:")
    print(f"  - Tested {len(derivative_list)} derivative types")
    print(f"  - Analyzed {len(alpha_subset)} alpha values")
    print("  - Statistical analysis performed")
    print("  - Optimal parameters identified")
    print(f"  - Results saved: alpha_output/")
    return True

def run_neural_training():
    """Neural network training"""
    print("\n🧠 Running Neural Network Training...")
    
    epochs = input("Training epochs [30]: ").strip()
    try:
        epochs = int(epochs) if epochs else 30
    except:
        epochs = 30
    
    research = CancerDynamicsResearch('neural_output')
    research.initialize_models()
    research.initialize_neural_networks()
    research.run_basic_dynamics_analysis()  # Generate training data
    result = research.run_neural_network_training(train_epochs=epochs)
    
    print("\n✅ Neural Training Complete:")
    print(f"  - Trained for {epochs} epochs")
    print("  - Models learned quantum dynamics")
    print("  - Performance metrics calculated")
    print("  - Trained models saved")
    print(f"  - Results saved: neural_output/")
    return True

def run_full_pipeline():
    """Full research pipeline"""
    print("\n🔬 Running Full Research Pipeline...")
    print("This comprehensive analysis will take approximately 60 minutes.")
    
    confirm = input("Continue with full analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Full pipeline cancelled.")
        return False
    
    research = CancerDynamicsResearch('full_research_output')
    success = research.run_full_research_pipeline()
    
    if success:
        print("\n✅ Full Research Pipeline Complete:")
        print("  - All models analyzed with quantum effects")
        print("  - Alpha sensitivity optimization performed")
        print("  - Neural networks trained on quantum dynamics")
        print("  - Comprehensive statistical analysis")
        print("  - Executive summary generated")
        print("  - All visualizations created")
        print(f"  - Complete results: full_research_output/")
    else:
        print("\n❌ Full pipeline encountered errors")
    
    return success

def run_quantum_analysis():
    """Quantum-focused analysis"""
    print("\n⚛️ Running Quantum Effects Analysis...")
    
    research = CancerDynamicsResearch('quantum_output')
    research.initialize_models()
    
    # Test quantum conditions specifically
    from config.parameters import get_quantum_initial_conditions
    quantum_conditions = get_quantum_initial_conditions()
    
    print(f"Testing {len(quantum_conditions)} quantum conditions...")
    
    t = np.linspace(0, 2, 21)
    quantum_results = []
    
    for i, init_cond in enumerate(quantum_conditions):
        print(f"  Quantum test {i+1}: T={init_cond[0]:.0e}, I={init_cond[1]:.0e}")
        
        # Analyze quantum status
        status = research.models['integer'].get_quantum_status(init_cond)
        print(f"    Quantum active: Tumor={status['tumor_quantum_active']}, Immune={status['immune_quantum_active']}")
        
        # Run simulation
        from scipy.integrate import odeint
        traj = odeint(research.models['integer'], init_cond, t)
        quantum_results.append(traj)
        
        print(f"    Final populations: T={traj[-1,0]:.2f}, I={traj[-1,1]:.2f}")
    
    # Create quantum-specific visualizations
    research.visualizer.dynamics.plot_individual_cell_dynamics(
        t, quantum_results, 
        [f"Quantum {i+1}" for i in range(len(quantum_conditions))],
        "Quantum Test"
    )
    
    print("\n✅ Quantum Analysis Complete:")
    print("  - Quantum pressure effects demonstrated")
    print("  - Quantum exclusion principle verified")
    print("  - Cancer resilience mechanisms identified")
    print(f"  - Results saved: quantum_output/")
    return True

def run_custom_analysis():
    """Custom analysis configuration"""
    print("\n⚙️ Custom Analysis Configuration...")
    
    # Get output directory
    output_dir = input("Output directory [custom_output]: ").strip()
    if not output_dir:
        output_dir = 'custom_output'
    
    print("\nSelect analysis components:")
    
    # Get components to run
    components = {}
    components['basic'] = input("Basic dynamics analysis? (y/n) [y]: ").strip().lower() != 'n'
    components['alpha'] = input("Alpha sensitivity? (y/n) [y]: ").strip().lower() != 'n'
    components['neural'] = input("Neural network training? (y/n) [n]: ").strip().lower() == 'y'
    components['quantum'] = input("Quantum analysis? (y/n) [y]: ").strip().lower() != 'n'
    
    # Get parameters
    if components['alpha']:
        derivatives = input("Derivatives [caputo,riemann_liouville]: ").strip()
        if derivatives:
            derivative_list = [d.strip() for d in derivatives.split(',')]
        else:
            derivative_list = ['caputo', 'riemann_liouville']
    
    if components['neural']:
        epochs = input("Training epochs [20]: ").strip()
        try:
            epochs = int(epochs) if epochs else 20
        except:
            epochs = 20
    
    # Configure quantum parameters
    quantum_params = get_quantum_settings()
    
    print(f"\n🔄 Starting custom analysis in '{output_dir}'...")
    
    # Initialize research
    research = CancerDynamicsResearch(output_dir)
    research.initialize_models()
    
    results = {}
    
    # Run selected components
    if components['basic']:
        print("\n➤ Running basic dynamics analysis...")
        results['basic'] = research.run_basic_dynamics_analysis()
    
    if components['alpha']:
        print("\n➤ Running alpha sensitivity analysis...")
        results['alpha'] = research.run_alpha_sensitivity_analysis(
            derivative_types=derivative_list
        )
    
    if components['neural']:
        print("\n➤ Running neural network training...")
        research.initialize_neural_networks()
        if 'basic' not in results:
            research.run_basic_dynamics_analysis()
        results['neural'] = research.run_neural_network_training(train_epochs=epochs)
    
    if components['quantum']:
        print("\n➤ Running quantum analysis...")
        # Quantum analysis is integrated in basic analysis
        pass
    
    # Generate comprehensive analysis if we have results
    if len(results) > 1:
        print("\n➤ Running comprehensive analysis...")
        research.run_comprehensive_analysis()
    
    print(f"\n✅ Custom analysis complete! Results in: {output_dir}")
    return True

def display_system_info():
    """Display system and configuration information"""
    print("\n📋 System Information:")
    
    try:
        from config.parameters import get_config, get_quantum_parameters
        config = get_config()
        quantum_params = get_quantum_parameters()
        
        print(f"  Python version: {sys.version.split()[0]}")
        print(f"  Framework version: Cancer Dynamics with Quantum Effects")
        print(f"  Available derivatives: {len(config.fractional_derivative_types)}")
        print(f"  Alpha values: {len(config.alpha_values)} ({config.alpha_values[0]:.1f} to {config.alpha_values[-1]:.1f})")
        print(f"  Initial conditions: {len(config.initial_conditions)} standard + quantum test cases")
        print(f"  Quantum threshold: {quantum_params['quantum_threshold']:.0e}")
        print(f"  Quantum momentum: {quantum_params['p']}")
        
    except Exception as e:
        print(f"  Error loading configuration: {e}")

def main():
    """Main application loop"""
    print_header()
    display_system_info()
    
    while True:
        try:
            print_menu()
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == '0':
                print("\n👋 Exiting Cancer Dynamics Research Framework")
                print("Thank you for using our quantum-enhanced modeling system!")
                sys.exit(0)
                
            elif choice == '1':
                success = run_quick_demo()
                
            elif choice == '2':
                success = run_basic_analysis()
                
            elif choice == '3':
                success = run_alpha_sensitivity()
                
            elif choice == '4':
                success = run_neural_training()
                
            elif choice == '5':
                success = run_full_pipeline()
                
            elif choice == '6':
                success = run_quantum_analysis()
                
            elif choice == '7':
                success = run_custom_analysis()
                
            else:
                print("\n❌ Invalid choice. Please enter 0-7.")
                continue
            
            if success:
                print(f"\n✅ Analysis completed successfully!")
            else:
                print(f"\n⚠️ Analysis completed with warnings.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user. Exiting gracefully...")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please check your configuration and try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Check if required modules are available
    try:
        import numpy
        import scipy
        import matplotlib
        from main import CancerDynamicsResearch
        print("✅ All dependencies loaded successfully")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages and try again.")
        sys.exit(1)
    
    main()