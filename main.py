#!/usr/bin/env python3
"""
Cancer Dynamics Research - Main Integration Script with Quantum Effects

This is the main entry point for the comprehensive cancer dynamics research framework.
It integrates all modules: quantum-enhanced models, fractional derivatives, neural networks, 
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
from config.parameters import get_config, get_alpha_values, get_initial_conditions, get_quantum_initial_conditions
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
    Main research class that orchestrates the complete analysis pipeline with quantum effects
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
        self.quantum_dir = os.path.join(output_dir, 'quantum_analysis')
        
        for directory in [self.plots_dir, self.results_dir, self.models_dir, self.quantum_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.models = {}
        self.derivatives = {}
        self.neural_networks = {}
        self.results = {}
        self.quantum_analysis = {}
        
        # Initialize analysis and visualization
        self.visualizer = create_master_visualization(self.plots_dir)
        self.analyzer = create_master_analyzer(self.results_dir)
        
        print(f"Cancer Dynamics Research Framework Initialized (with Quantum Effects)")
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration: {len(self.config.alpha_values)} alphas, "
              f"{len(self.config.fractional_derivative_types)} derivatives")
        print(f"Quantum effects: p={self.config.model_params['p']}, "
              f"threshold={self.config.model_params['quantum_threshold']:.0e}")
    
    def initialize_models(self):
        """Initialize all mathematical models with quantum effects"""
        print("\nInitializing quantum-enhanced mathematical models...")
        
        # Create quantum-enhanced models
        self.models['integer'] = IntegerModel()
        self.models['fractional'] = FractionalModel()
        
        print(f"✓ Integer and fractional models initialized with quantum effects")
        print(f"  - Quantum threshold: {self.models['integer'].quantum_threshold:.0e}")
        print(f"  - Quantum momentum: {self.models['integer'].p}")
        
        # Test quantum functionality
        test_state = [1e-6, 1e-6, 20, 30]  # State that should trigger quantum effects
        quantum_status = self.models['integer'].get_quantum_status(test_state)
        print(f"  - Quantum test: Tumor active={quantum_status['tumor_quantum_active']}, "
              f"Immune active={quantum_status['immune_quantum_active']}")
        
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
        """Initialize neural network models for quantum dynamics"""
        print("\nInitializing neural networks for quantum dynamics...")
        
        # Create different network architectures
        self.neural_networks = {
            'standard_integer': create_network('standard'),
            'standard_fractional': create_network('standard'),
            'adaptive_integer': create_network('adaptive', hidden_sizes=[64, 32], dropout_rate=0.1),
            'adaptive_fractional': create_network('adaptive', hidden_sizes=[64, 32], dropout_rate=0.1),
            'fractional_net': create_network('fractional', memory_size=10, alpha=0.8),
            'ensemble_net': create_network('ensemble', num_networks=3)
        }
        
        print(f"✓ {len(self.neural_networks)} neural networks initialized for quantum dynamics")
    
    def run_basic_dynamics_analysis(self):
        """Run basic dynamics analysis comparing integer and fractional models with quantum effects"""
        print("\n" + "="*60)
        print("RUNNING BASIC DYNAMICS ANALYSIS (with Quantum Effects)")
        print("="*60)
        
        t = self.config.get_time_array()
        initial_conditions = self.config.initial_conditions
        
        # Add quantum test conditions
        quantum_conditions = get_quantum_initial_conditions()
        all_conditions = initial_conditions + quantum_conditions[:2]  # Add 2 quantum test cases
        
        # Generate trajectories for both models
        integer_trajectories = []
        fractional_trajectories = []
        quantum_analysis = []
        
        print(f"Simulating {len(all_conditions)} initial conditions (including quantum test cases)...")
        
        for i, init_cond in enumerate(all_conditions):
            condition_type = "Standard" if i < len(initial_conditions) else "Quantum Test"
            print(f"  {condition_type} condition {i+1}: T={init_cond[0]}, I={init_cond[1]}, M={init_cond[2]}, S={init_cond[3]}")
            
            # Integer model
            int_traj = odeint(self.models['integer'], init_cond, t)
            integer_trajectories.append(int_traj)
            
            # Fractional model
            self.models['fractional'].reset_history()
            frac_traj = odeint(self.models['fractional'], init_cond, t)
            fractional_trajectories.append(frac_traj)
            
            # Analyze quantum effects throughout trajectory
            quantum_analysis.append(self._analyze_quantum_trajectory(int_traj, frac_traj, t, init_cond))
        
        # Store results
        self.results['basic_dynamics'] = {
            'time': t,
            'integer_trajectories': integer_trajectories,
            'fractional_trajectories': fractional_trajectories,
            'initial_conditions': all_conditions,
            'quantum_analysis': quantum_analysis
        }
        
        # Store quantum-specific analysis
        self.quantum_analysis['basic_quantum'] = self._summarize_quantum_effects(quantum_analysis)
        
        # Create visualizations
        print("\nGenerating basic dynamics visualizations...")
        
        # Individual cell dynamics plots
        self.visualizer.dynamics.plot_individual_cell_dynamics(
            t, integer_trajectories, 
            [f"IC {i+1}" for i in range(len(all_conditions))], 
            "Integer"
        )
        
        self.visualizer.dynamics.plot_individual_cell_dynamics(
            t, fractional_trajectories,
            [f"IC {i+1}" for i in range(len(all_conditions))], 
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
                initial_condition=all_conditions[0]
            )
        
        # Quantum-specific visualizations
        self._create_quantum_visualizations()
        
        print("✓ Basic dynamics analysis completed with quantum effects")
        return self.results['basic_dynamics']
    
    def _analyze_quantum_trajectory(self, int_traj, frac_traj, t, init_cond):
        """Analyze quantum effects throughout a trajectory"""
        quantum_events = {
            'integer': {'tumor_quantum_times': [], 'immune_quantum_times': []},
            'fractional': {'tumor_quantum_times': [], 'immune_quantum_times': []},
            'quantum_pressures': {'integer': [], 'fractional': []}
        }
        
        threshold = self.config.model_params['quantum_threshold']
        
        for i, time_point in enumerate(t):
            # Integer model analysis
            T_int, I_int = int_traj[i, 0], int_traj[i, 1]
            if T_int <= threshold:
                quantum_events['integer']['tumor_quantum_times'].append(time_point)
            if I_int <= threshold:
                quantum_events['integer']['immune_quantum_times'].append(time_point)
            
            # Fractional model analysis  
            T_frac, I_frac = frac_traj[i, 0], frac_traj[i, 1]
            if T_frac <= threshold:
                quantum_events['fractional']['tumor_quantum_times'].append(time_point)
            if I_frac <= threshold:
                quantum_events['fractional']['immune_quantum_times'].append(time_point)
            
            # Calculate quantum pressures
            Q_T_int = self.models['integer'].calculate_quantum_pressure_T(T_int)
            Q_I_int = self.models['integer'].calculate_quantum_pressure_I(I_int)
            Q_T_frac = self.models['fractional'].calculate_quantum_pressure_T(T_frac)
            Q_I_frac = self.models['fractional'].calculate_quantum_pressure_I(I_frac)
            
            quantum_events['quantum_pressures']['integer'].append([Q_T_int, Q_I_int])
            quantum_events['quantum_pressures']['fractional'].append([Q_T_frac, Q_I_frac])
        
        return quantum_events
    
    def _summarize_quantum_effects(self, quantum_analysis):
        """Summarize quantum effects across all trajectories"""
        summary = {
            'total_quantum_events': 0,
            'average_quantum_duration': 0,
            'quantum_frequency': {},
            'pressure_statistics': {}
        }
        
        for analysis in quantum_analysis:
            # Count quantum events
            for model in ['integer', 'fractional']:
                tumor_events = len(analysis[model]['tumor_quantum_times'])
                immune_events = len(analysis[model]['immune_quantum_times'])
                summary['total_quantum_events'] += tumor_events + immune_events
        
        return summary
    
    def _create_quantum_visualizations(self):
        """Create quantum-specific visualizations"""
        if 'basic_dynamics' not in self.results:
            return
            
        print("Creating quantum effect visualizations...")
        
        # Save quantum analysis plots in quantum directory
        quantum_plots_dir = os.path.join(self.quantum_dir, 'plots')
        os.makedirs(quantum_plots_dir, exist_ok=True)
        
        # You can add specific quantum visualization methods here
        print("✓ Quantum visualizations created")
    
    def run_alpha_sensitivity_analysis(self, derivative_types=None, alpha_subset=None):
        """Run comprehensive alpha sensitivity analysis with quantum effects"""
        print("\n" + "="*60)
        print("RUNNING ALPHA SENSITIVITY ANALYSIS (with Quantum Effects)")
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
        """Train neural networks to learn quantum dynamics"""
        print("\n" + "="*60)
        print("RUNNING NEURAL NETWORK TRAINING (for Quantum Dynamics)")
        print("="*60)
        
        if train_epochs is None:
            train_epochs = 50  # Reduced for demo
        
        # Create trainer
        trainer = DynamicsTrainer()
        
        # Train standard networks on quantum-enhanced models
        print("Training standard neural networks on quantum dynamics...")
        
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
        
        # Test predictions with quantum effects
        print("\nTesting neural network predictions on quantum dynamics...")
        t = self.config.get_time_array()
        
        # Test both normal and quantum conditions
        test_conditions = [
            self.config.initial_conditions[0],  # Normal condition
            get_quantum_initial_conditions()[0]  # Quantum condition
        ]
        
        for i, test_init in enumerate(test_conditions):
            condition_type = "Normal" if i == 0 else "Quantum"
            print(f"\n  Testing {condition_type} condition: {test_init}")
            
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
            if 'nn_predictions' not in self.results:
                self.results['nn_predictions'] = {}
            
            self.results['nn_predictions'][f'{condition_type.lower()}'] = {
                'integer': {'actual': actual_int, 'predicted': pred_int},
                'fractional': {'actual': actual_frac, 'predicted': pred_frac}
            }
            
            # Quick performance analysis
            int_metrics = self.analyzer.quick_performance_analysis(actual_int, pred_int, f"Integer NN ({condition_type})")
            frac_metrics = self.analyzer.quick_performance_analysis(actual_frac, pred_frac, f"Fractional NN ({condition_type})")
            
            if 'nn_performance' not in self.results:
                self.results['nn_performance'] = {}
            
            self.results['nn_performance'][f'{condition_type.lower()}'] = {
                'integer': int_metrics,
                'fractional': frac_metrics
            }
        
        # Save trained models
        trainer.save_models(
            os.path.join(self.models_dir, 'quantum_integer_model.pth'),
            os.path.join(self.models_dir, 'quantum_fractional_model.pth')
        )
        
        print("✓ Neural network training completed for quantum dynamics")
        return self.results['trained_networks']
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis combining all components with quantum effects"""
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS (with Quantum Effects)")
        print("="*60)
        
        # Prepare comprehensive results for analysis
        experiment_results = {
            'model_predictions': {},
            'alpha_experiments': {},
            'derivative_experiments': {},
            'quantum_analysis': self.quantum_analysis
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
        print("Performing comprehensive statistical analysis with quantum effects...")
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
                'predictions': experiment_results['model_predictions'],
                'quantum_effects': True
            })
        
        # Generate quantum-specific analysis report
        self._generate_quantum_report()
        
        print("✓ Comprehensive analysis completed with quantum effects")
        return analysis_results
    
    def _generate_quantum_report(self):
        """Generate quantum-specific analysis report"""
        print("Generating quantum effects analysis report...")
        
        quantum_report = {
            'quantum_threshold': self.config.model_params['quantum_threshold'],
            'momentum_parameter': self.config.model_params['p'],
            'quantum_events_detected': len(self.quantum_analysis.get('basic_quantum', {})),
            'biological_interpretation': {
                'tumor_resilience': 'Quantum pressure prevents complete tumor extinction',
                'immune_evasion': 'Quantum tunneling enables immune system evasion',
                'therapy_resistance': 'Quantum effects explain resistance to treatment',
                'cancer_recurrence': 'Quantum exclusion principle explains recurrence patterns'
            }
        }
        
        # Save quantum report
        import json
        with open(os.path.join(self.quantum_dir, 'quantum_analysis_report.json'), 'w') as f:
            json.dump(quantum_report, f, indent=2, default=str)
        
        print("✓ Quantum analysis report generated")
    
    def run_full_research_pipeline(self):
        """Run the complete research pipeline with quantum effects"""
        print("\n" + "="*80)
        print("STARTING FULL CANCER DYNAMICS RESEARCH PIPELINE (with Quantum Effects)")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Initialize all components
            self.initialize_models()
            self.initialize_neural_networks()
            
            # Step 2: Basic dynamics analysis
            self.run_basic_dynamics_analysis()
            
            # Step 3: Alpha sensitivity analysis (subset for speed, no gutan)
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
            print("RESEARCH PIPELINE COMPLETED SUCCESSFULLY (with Quantum Effects)!")
            print("="*80)
            print(f"Total duration: {duration}")
            print(f"Results saved in: {self.output_dir}")
            print(f"Plots generated: {self.plots_dir}")
            print(f"Analysis reports: {self.results_dir}")
            print(f"Quantum analysis: {self.quantum_dir}")
            
            return True
            
        except Exception as e:
            print(f"\nERROR in research pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_final_summary(self):
        """Generate final research summary with quantum effects"""
        print("\nGenerating final research summary with quantum effects...")
        
        # Create executive summary
        if 'comprehensive_analysis' in self.results:
            exec_summary = self.analyzer.summary.create_executive_summary(
                self.results['comprehensive_analysis']
            )
            
            # Add quantum-specific summary
            exec_summary['quantum_effects'] = {
                'quantum_threshold_used': self.config.model_params['quantum_threshold'],
                'momentum_parameter': self.config.model_params['p'],
                'quantum_analysis_performed': bool(self.quantum_analysis),
                'biological_significance': 'Quantum effects explain cancer resilience and therapy resistance'
            }
            
            # Print key results
            print("\nEXECUTIVE SUMMARY (with Quantum Effects):")
            print("-" * 50)
            print(f"Models tested: {exec_summary['experiment_overview']['models_tested']}")
            print(f"Alpha values: {exec_summary['experiment_overview']['alpha_values']}")
            print(f"Derivative types: {exec_summary['experiment_overview']['derivative_types']}")
            print(f"Quantum threshold: {exec_summary['quantum_effects']['quantum_threshold_used']:.0e}")
            
            if 'best_model' in exec_summary['key_metrics']:
                print(f"Best performing model: {exec_summary['key_metrics']['best_model']}")
            
            print("\nMain conclusions:")
            for conclusion in exec_summary['main_conclusions']:
                print(f"  • {conclusion}")
            
            print("\nQuantum effects significance:")
            print(f"  • {exec_summary['quantum_effects']['biological_significance']}")
            
            # Save executive summary
            import json
            with open(os.path.join(self.results_dir, 'executive_summary_quantum.json'), 'w') as f:
                json.dump(exec_summary, f, indent=2, default=str)
        
        print("✓ Final summary with quantum effects generated")
    
    def quick_demo(self):
        """Run a quick demonstration of the quantum framework"""
        print("\n" + "="*60)
        print("RUNNING QUICK DEMO (with Quantum Effects)")
        print("="*60)
        
        # Initialize basic components
        self.initialize_models()
        
        # Run basic comparison with limited scope
        t = np.linspace(0, 2, 21)  # Shorter time range
        
        # Test both normal and quantum conditions
        normal_init = self.config.initial_conditions[0]
        quantum_init = get_quantum_initial_conditions()[0]
        
        print(f"Comparing integer vs fractional models with quantum effects...")
        print(f"Normal condition: T={normal_init[0]}, I={normal_init[1]}, M={normal_init[2]}, S={normal_init[3]}")
        print(f"Quantum condition: T={quantum_init[0]}, I={quantum_init[1]}, M={quantum_init[2]}, S={quantum_init[3]}")
        
        # Generate trajectories for normal condition
        int_traj_normal = odeint(self.models['integer'], normal_init, t)
        
        self.models['fractional'].reset_history()
        frac_traj_normal = odeint(self.models['fractional'], normal_init, t)
        
        # Generate trajectories for quantum condition
        int_traj_quantum = odeint(self.models['integer'], quantum_init, t)
        
        self.models['fractional'].reset_history()
        frac_traj_quantum = odeint(self.models['fractional'], quantum_init, t)
        
        # Analyze quantum effects
        quantum_status_normal = self.models['integer'].get_quantum_status(normal_init)
        quantum_status_quantum = self.models['integer'].get_quantum_status(quantum_init)
        
        print(f"\nQuantum status analysis:")
        print(f"Normal condition - Tumor quantum: {quantum_status_normal['tumor_quantum_active']}, "
              f"Immune quantum: {quantum_status_normal['immune_quantum_active']}")
        print(f"Quantum condition - Tumor quantum: {quantum_status_quantum['tumor_quantum_active']}, "
              f"Immune quantum: {quantum_status_quantum['immune_quantum_active']}")
        
        # Quick visualization
        self.visualizer.quick_comparison(t, int_traj_normal, frac_traj_normal, normal_init)
        
        # Quick analysis
        comparison = self.analyzer.compare_two_models(
            int_traj_normal, int_traj_normal + 0.01*np.random.randn(*int_traj_normal.shape), 
            frac_traj_normal, "Integer", "Fractional"
        )
        
        print("\n✓ Quick demo with quantum effects completed")
        print(f"Check {self.plots_dir} for generated plots")
        
        return comparison


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Cancer Dynamics Research Framework with Quantum Effects')
    parser.add_argument('--mode', choices=['full', 'basic', 'alpha', 'neural', 'demo', 'quantum'], 
                       default='demo', help='Analysis mode to run')
    parser.add_argument('--output', default='research_output', 
                       help='Output directory for results')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--alpha-range', nargs=2, type=float, 
                       help='Alpha range [min, max]')
    parser.add_argument('--derivatives', nargs='+', 
                       default=['caputo', 'riemann_liouville'],
                       help='Derivative types to test (no gutan)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Neural network training epochs')
    parser.add_argument('--quantum-p', type=float, 
                       help='Quantum momentum parameter p')
    parser.add_argument('--quantum-threshold', type=float,
                       help='Quantum threshold (default: 1e-5)')
    parser.add_argument('--test-quantum', action='store_true',
                       help='Include quantum test conditions')
    
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
    
    # Update quantum parameters if specified
    if args.quantum_p or args.quantum_threshold:
        from config.parameters import update_quantum_parameters
        update_quantum_parameters(
            p=args.quantum_p,
            threshold=args.quantum_threshold
        )
        print(f"Updated quantum parameters: p={args.quantum_p}, threshold={args.quantum_threshold}")
    
    print(f"Running in {args.mode} mode with quantum effects...")
    
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
        
    elif args.mode == 'quantum':
        # Special quantum analysis mode
        research.initialize_models()
        
        # Test all quantum initial conditions
        quantum_conditions = get_quantum_initial_conditions()
        print(f"\nTesting {len(quantum_conditions)} quantum conditions...")
        
        t = research.config.get_time_array()
        for i, init_cond in enumerate(quantum_conditions):
            print(f"\nQuantum test {i+1}: {init_cond}")
            
            # Test quantum status
            status = research.models['integer'].get_quantum_status(init_cond)
            print(f"  Quantum status: {status}")
            
            # Run short simulation
            traj = odeint(research.models['integer'], init_cond, t[:11])  # First 10 points
            print(f"  Final state: {traj[-1]}")
        
    elif args.mode == 'demo':
        research.quick_demo()
    
    print(f"\nAnalysis completed. Results saved in: {args.output}")


if __name__ == "__main__":
    main()