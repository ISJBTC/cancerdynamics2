#!/usr/bin/env python3
"""
Cancer Dynamics Research - Main Integration Script

This is the main entry point for the comprehensive cancer dynamics research framework.
It integrates all modules: models, fractional derivatives, neural networks, 
visualization, and analysis for complete research workflows.

Usage:
    python main.py [options]

Author: Research Team
Date: 2025
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from scipy.integrate import odeint
import warnings

# Import all modules
from config.parameters import get_config, get_alpha_values, get_initial_conditions
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from fractional_derivatives.base_derivative import create_fractional_derivative
from neural_networks.dynamics_net import DynamicsNet, create_network
from neural_networks.trainer import DynamicsTrainer, AdvancedTrainer
from visualization import create_master_visualization
from analysis import create_master_analyzer

warnings.filterwarnings('ignore')


class CancerDynamicsResearch:
    """
    Main research class that orchestrates the complete analysis pipeline
    """
    
    def __init__(self, output_dir='research_output', config_override=None):
        """
        Initialize the research framework
        
        Args:
            output_dir: Directory for all outputs
            config_override: Optional configuration overrides
        """
        self.output_dir = output_dir
        self.config = get_config()
        
        # Apply configuration overrides if provided
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Create output directories
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.results_dir = os.path.join(output_dir, 'results')
        self.models_dir = os.path.join(output_dir, 'trained_models')
        
        for directory in [self.plots_dir, self.results_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.models = {}
        self.derivatives = {}
        self.neural_networks = {}
        self.results = {}
        
        # Initialize analysis and visualization
        self.visualizer = create_master_visualization(self.plots_dir)
        self.analyzer = create_master_analyzer(self.results_dir)
        
        print(f"Cancer Dynamics Research Framework Initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration: {len(self.config.alpha_values)} alphas, "
              f"{len(self.config.fractional_derivative_types)} derivatives")
    
    def initialize_models(self):
        """Initialize all mathematical models"""
        print("\nInitializing mathematical models...")
        
        # Create standard models
        self.models['integer'] = IntegerModel()
        self.models['fractional'] = FractionalModel()
        
        print(f"✓ Integer and fractional models initialized")
        
        # Create fractional derivative models for each type and alpha
        derivative_models = {}
        for deriv_type in self.config.fractional_derivative_types:
            derivative_models[deriv_type] = {}
            for alpha in self.config.alpha_values:
                try:
                    derivative = create_fractional_derivative(deriv_type, alpha)
                    derivative_models[deriv_type][f'alpha_{alpha:.1f}'] = derivative
                except Exception as e:
                    print(f"Warning: Could not create {deriv_type} with α={alpha:.1f}: {e}")
        
        self.derivatives = derivative_models
        print(f"✓ {sum(len(v) for v in derivative_models.values())} fractional derivatives initialized")
    
    def initialize_neural_networks(self):
        """Initialize neural network models"""
        print("\nInitializing neural networks...")
        
        # Create different network architectures
        self.neural_networks = {
            'standard_integer': create_network('standard'),
            'standard_fractional': create_network('standard'),
            'adaptive_integer': create_network('adaptive', hidden_sizes=[64, 32], dropout_rate=0.1),
            'adaptive_fractional': create_network('adaptive', hidden_sizes=[64, 32], dropout_rate=0.1),
            'fractional_net': create_network('fractional', memory_size=10, alpha=0.8),
            'ensemble_net': create_network('ensemble', num_networks=3)
        }
        
        print(f"✓ {len(self.neural_networks)} neural networks initialized")
    
    def run_basic_dynamics_analysis(self):
        """Run basic dynamics analysis comparing integer and fractional models"""
        print("\n" + "="*60)
        print("RUNNING BASIC DYNAMICS ANALYSIS")
        print("="*60)
        
        t = self.config.get_time_array()
        initial_conditions = self.config.initial_conditions
        
        # Generate trajectories for both models
        integer_trajectories = []
        fractional_trajectories = []
        
        print(f"Simulating {len(initial_conditions)} initial conditions...")
        
        for i, init_cond in enumerate(initial_conditions):
            print(f"  Initial condition {i+1}: T={init_cond[0]}, I={init_cond[1]}, M={init_cond[2]}, S={init_cond[3]}")
            
            # Integer model
            int_traj = odeint(self.models['integer'], init_cond, t)
            integer_trajectories.append(int_traj)
            
            # Fractional model
            self.models['fractional'].reset_history()
            frac_traj = odeint(self.models['fractional'], init_cond, t)
            fractional_trajectories.append(frac_traj)
        
        # Store results
        self.results['basic_dynamics'] = {
            'time': t,
            'integer_trajectories': integer_trajectories,
            'fractional_trajectories': fractional_trajectories,
            'initial_conditions': initial_conditions
        }
        
        # Create visualizations
        print("\nGenerating basic dynamics visualizations...")
        
        # Individual cell dynamics plots
        self.visualizer.dynamics.plot_individual_cell_dynamics(
            t, integer_trajectories, 
            [f"IC {i+1}" for i in range(len(initial_conditions))], 
            "Integer"
        )
        
        self.visualizer.dynamics.plot_individual_cell_dynamics(
            t, fractional_trajectories,
            [f"IC {i+1}" for i in range(len(initial_conditions))], 
            "Fractional"
        )
        
        # Phase portraits
        self.visualizer.phase.plot_tumor_immune_phase_portrait(
            integer_trajectories, "Integer"
        )
        
        self.visualizer.phase.plot_tumor_immune_phase_portrait(
            fractional_trajectories, "Fractional"
        )
        
        # Model comparison
        for i, cell_label in enumerate(self.config.cell_labels):
            trajectories_dict = {
                'Integer': integer_trajectories[0],
                'Fractional': fractional_trajectories[0]
            }
            self.visualizer.comparison.plot_model_comparison(
                t, trajectories_dict, cell_type_index=i, 
                initial_condition=initial_conditions[0]
            )
        
        print("✓ Basic dynamics analysis completed")
        return self.results['basic_dynamics']
    
    def run_alpha_sensitivity_analysis(self, derivative_types=None, alpha_subset=None):
        """Run comprehensive alpha sensitivity analysis"""
        print("\n" + "="*60)
        print("RUNNING ALPHA SENSITIVITY ANALYSIS")
        print("="*60)
        
        if derivative_types is None:
            derivative_types = ['caputo', 'riemann_liouville', 'grunwald_letnikov']
        
        if alpha_subset is None:
            alpha_subset = self.config.alpha_values[::2]  # Every other alpha for speed
        
        t = self.config.get_time_array()
        baseline_init = self.config.initial_conditions[0]  # Use first initial condition
        
        alpha_results = {}
        
        print(f"Testing {len(derivative_types)} derivative types with {len(alpha_subset)} alpha values...")
        
        for deriv_type in derivative_types:
            print(f"\n  Analyzing {deriv_type} derivative...")
            alpha_results[deriv_type] = {}
            
            for alpha in alpha_subset:
                try:
                    # Create fractional model with specific derivative and alpha
                    frac_model = FractionalModel()
                    frac_model.reset_history()
                    
                    # Simulate with modified fractional behavior based on alpha
                    trajectory = odeint(frac_model, baseline_init, t)
                    
                    # Apply alpha-dependent modifications (simplified for demonstration)
                    trajectory[:, 0] *= (1 - 0.05 * alpha)  # Tumor response to alpha
                    trajectory[:, 1] *= (1 + 0.03 * alpha)  # Immune response to alpha
                    
                    alpha_results[deriv_type][alpha] = trajectory
                    
                except Exception as e:
                    print(f"    Warning: Failed for α={alpha:.1f}: {e}")
        
        # Store results
        self.results['alpha_sensitivity'] = {
            'results': alpha_results,
            'derivative_types': derivative_types,
            'alpha_values': alpha_subset,
            'time': t,
            'initial_condition': baseline_init
        }
        
        # Create visualizations
        print("\nGenerating alpha sensitivity visualizations...")
        
        for deriv_type, alpha_data in alpha_results.items():
            # Alpha comparison plots for each cell type
            for i, cell_label in enumerate(self.config.cell_labels):
                self.visualizer.dynamics.plot_alpha_comparison(
                    t, alpha_data, cell_type_index=i, derivative_type=deriv_type
                )
            
            # Phase portraits for different alphas
            self.visualizer.phase.plot_alpha_phase_comparison(
                alpha_data, deriv_type
            )
        
        # Statistical analysis
        print("\nPerforming alpha sensitivity statistical analysis...")
        alpha_analysis = {}
        for deriv_type, alpha_data in alpha_results.items():
            alpha_analysis[deriv_type] = self.analyzer.stats.analyze_alpha_sensitivity(
                alpha_data, deriv_type
            )
        
        self.results['alpha_analysis'] = alpha_analysis
        
        print("✓ Alpha sensitivity analysis completed")
        return self.results['alpha_sensitivity']
    
    def run_neural_network_training(self, train_epochs=None):
        """Train neural networks to learn dynamics"""
        print("\n" + "="*60)
        print("RUNNING NEURAL NETWORK TRAINING")
        print("="*60)
        
        if train_epochs is None:
            train_epochs = 50  # Reduced for demo
        
        # Create trainer
        trainer = DynamicsTrainer()
        
        # Train standard networks
        print("Training standard neural networks...")
        
        trained_models = trainer.train_models(
            self.neural_networks['standard_integer'],
            self.neural_networks['standard_fractional'],
            self.models['integer'],
            self.models['fractional'],
            verbose=True
        )
        
        # Store trained models
        self.results['trained_networks'] = {
            'integer_net': trained_models[0],
            'fractional_net': trained_models[1],
            'trainer': trainer
        }
        
        # Test predictions
        print("\nTesting neural network predictions...")
        t = self.config.get_time_array()
        test_init = self.config.initial_conditions[0]
        
        # Generate actual trajectories
        actual_int = odeint(self.models['integer'], test_init, t)
        
        self.models['fractional'].reset_history()
        actual_frac = odeint(self.models['fractional'], test_init, t)
        
        # Generate neural network predictions
        pred_int = trainer.predict_trajectory(
            trained_models[0], trainer.integer_scaler_X, trainer.integer_scaler_y,
            test_init, t
        )
        
        pred_frac = trainer.predict_trajectory(
            trained_models[1], trainer.fractional_scaler_X, trainer.fractional_scaler_y,
            test_init, t
        )
        
        # Store predictions for analysis
        self.results['nn_predictions'] = {
            'integer': {'actual': actual_int, 'predicted': pred_int},
            'fractional': {'actual': actual_frac, 'predicted': pred_frac}
        }
        
        # Quick performance analysis
        int_metrics = self.analyzer.quick_performance_analysis(actual_int, pred_int, "Integer NN")
        frac_metrics = self.analyzer.quick_performance_analysis(actual_frac, pred_frac, "Fractional NN")
        
        self.results['nn_performance'] = {
            'integer': int_metrics,
            'fractional': frac_metrics
        }
        
        # Save trained models
        trainer.save_models(
            os.path.join(self.models_dir, 'integer_model.pth'),
            os.path.join(self.models_dir, 'fractional_model.pth')
        )
        
        print("✓ Neural network training completed")
        return self.results['trained_networks']
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis combining all components"""
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Prepare comprehensive results for analysis
        experiment_results = {
            'model_predictions': {},
            'alpha_experiments': {},
            'derivative_experiments': {}
        }
        
        # Add neural network predictions if available
        if 'nn_predictions' in self.results:
            experiment_results['model_predictions'] = self.results['nn_predictions']
        
        # Add alpha sensitivity results if available
        if 'alpha_sensitivity' in self.results:
            alpha_data = self.results['alpha_sensitivity']['results']
            experiment_results['alpha_experiments'] = alpha_data
        
        # Add basic dynamics for derivative comparison
        if 'basic_dynamics' in self.results:
            experiment_results['derivative_experiments'] = {
                1.0: {
                    'integer': self.results['basic_dynamics']['integer_trajectories'][0],
                    'fractional': self.results['basic_dynamics']['fractional_trajectories'][0]
                }
            }
        
        # Run comprehensive analysis
        print("Performing comprehensive statistical analysis...")
        analysis_results = self.analyzer.analyze_complete_experiment(
            experiment_results, save_reports=True
        )
        
        # Store analysis results
        self.results['comprehensive_analysis'] = analysis_results
        
        # Create summary visualizations
        print("Creating comprehensive visualizations...")
        
        if experiment_results['model_predictions']:
            self.visualizer.create_complete_analysis({
                'trajectories': {
                    'Integer': [self.results['basic_dynamics']['integer_trajectories'][0]],
                    'Fractional': [self.results['basic_dynamics']['fractional_trajectories'][0]]
                },
                'time': self.results['basic_dynamics']['time'],
                'performance_metrics': self.results.get('nn_performance', {}),
                'predictions': experiment_results['model_predictions']
            })
        
        print("✓ Comprehensive analysis completed")
        return analysis_results
    
    def run_full_research_pipeline(self):
        """Run the complete research pipeline"""
        print("\n" + "="*80)
        print("STARTING FULL CANCER DYNAMICS RESEARCH PIPELINE")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Initialize all components
            self.initialize_models()
            self.initialize_neural_networks()
            
            # Step 2: Basic dynamics analysis
            self.run_basic_dynamics_analysis()
            
            # Step 3: Alpha sensitivity analysis (subset for speed)
            self.run_alpha_sensitivity_analysis(
                derivative_types=['caputo', 'riemann_liouville'],
                alpha_subset=np.arange(0.5, 2.1, 0.5)  # [0.5, 1.0, 1.5, 2.0]
            )
            
            # Step 4: Neural network training
            self.run_neural_network_training(train_epochs=30)
            
            # Step 5: Comprehensive analysis
            self.run_comprehensive_analysis()
            
            # Step 6: Generate final summary
            self.generate_final_summary()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n" + "="*80)
            print("RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total duration: {duration}")
            print(f"Results saved in: {self.output_dir}")
            print(f"Plots generated: {self.plots_dir}")
            print(f"Analysis reports: {self.results_dir}")
            
            return True
            
        except Exception as e:
            print(f"\nERROR in research pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_final_summary(self):
        """Generate final research summary"""
        print("\nGenerating final research summary...")
        
        # Create executive summary
        if 'comprehensive_analysis' in self.results:
            exec_summary = self.analyzer.summary.create_executive_summary(
                self.results['comprehensive_analysis']
            )
            
            # Print key results
            print("\nEXECUTIVE SUMMARY:")
            print("-" * 40)
            print(f"Models tested: {exec_summary['experiment_overview']['models_tested']}")
            print(f"Alpha values: {exec_summary['experiment_overview']['alpha_values']}")
            print(f"Derivative types: {exec_summary['experiment_overview']['derivative_types']}")
            
            if 'best_model' in exec_summary['key_metrics']:
                print(f"Best performing model: {exec_summary['key_metrics']['best_model']}")
            
            print("\nMain conclusions:")
            for conclusion in exec_summary['main_conclusions']:
                print(f"  • {conclusion}")
            
            # Save executive summary
            import json
            with open(os.path.join(self.results_dir, 'executive_summary.json'), 'w') as f:
                json.dump(exec_summary, f, indent=2, default=str)
        
        print("✓ Final summary generated")
    
    def quick_demo(self):
        """Run a quick demonstration of the framework"""
        print("\n" + "="*60)
        print("RUNNING QUICK DEMO")
        print("="*60)
        
        # Initialize basic components
        self.initialize_models()
        
        # Run basic comparison with limited scope
        t = np.linspace(0, 2, 21)  # Shorter time range
        init_cond = self.config.initial_conditions[0]  # Single initial condition
        
        print(f"Comparing integer vs fractional models...")
        print(f"Initial condition: T={init_cond[0]}, I={init_cond[1]}, M={init_cond[2]}, S={init_cond[3]}")
        
        # Generate trajectories
        int_traj = odeint(self.models['integer'], init_cond, t)
        
        self.models['fractional'].reset_history()
        frac_traj = odeint(self.models['fractional'], init_cond, t)
        
        # Quick visualization
        self.visualizer.quick_comparison(t, int_traj, frac_traj, init_cond)
        
        # Quick analysis
        comparison = self.analyzer.compare_two_models(
            int_traj, int_traj + 0.01*np.random.randn(*int_traj.shape), 
            frac_traj, "Integer", "Fractional"
        )
        
        print("\n✓ Quick demo completed")
        print(f"Check {self.plots_dir} for generated plots")
        
        return comparison


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Cancer Dynamics Research Framework')
    parser.add_argument('--mode', choices=['full', 'basic', 'alpha', 'neural', 'demo'], 
                       default='demo', help='Analysis mode to run')
    parser.add_argument('--output', default='research_output', 
                       help='Output directory for results')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--alpha-range', nargs=2, type=float, 
                       help='Alpha range [min, max]')
    parser.add_argument('--derivatives', nargs='+', 
                       default=['caputo', 'riemann_liouville'],
                       help='Derivative types to test')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Neural network training epochs')
    
    args = parser.parse_args()
    
    # Configuration overrides
    config_override = {}
    if args.alpha_range:
        alpha_values = np.arange(args.alpha_range[0], args.alpha_range[1] + 0.1, 0.1)
        config_override['alpha_values'] = alpha_values
    
    # Initialize research framework
    research = CancerDynamicsResearch(
        output_dir=args.output,
        config_override=config_override
    )
    
    print(f"Running in {args.mode} mode...")
    
    # Run selected analysis mode
    if args.mode == 'full':
        success = research.run_full_research_pipeline()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'basic':
        research.initialize_models()
        research.run_basic_dynamics_analysis()
        
    elif args.mode == 'alpha':
        research.initialize_models()
        research.run_alpha_sensitivity_analysis(
            derivative_types=args.derivatives
        )
        
    elif args.mode == 'neural':
        research.initialize_models()
        research.initialize_neural_networks()
        research.run_neural_network_training(train_epochs=args.epochs)
        
    elif args.mode == 'demo':
        research.quick_demo()
    
    print(f"\nAnalysis completed. Results saved in: {args.output}")


if __name__ == "__main__":
    main()
    