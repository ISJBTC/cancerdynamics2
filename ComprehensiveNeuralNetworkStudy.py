#!/usr/bin/env python3
"""
Comprehensive Neural Network Study: Fractional vs Integer Models
===============================================================

This study demonstrates the superiority of fractional models over integer models
and compares all fractional derivative types using neural network learning.

Metrics: RMSE, MAE, R²
Models: Integer, Caputo, Riemann-Liouville, Grünwald-Letnikov, Hilfer
Statistical Analysis: T-tests, ANOVA, Effect sizes
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your framework components
from models.integer_model import IntegerModel
from models.fractional_model import FractionalModel
from fractional_derivatives.caputo import CaputoDerivative
from fractional_derivatives.riemann_liouville import RiemannLiouvilleDerivative
from fractional_derivatives.grunwald_letnikov import GrunwaldLetnikovDerivative
from fractional_derivatives.hilfer import HilferDerivative
from neural_networks.dynamics_net import DynamicsNet, AdaptiveDynamicsNet
from neural_networks.trainer import DynamicsTrainer
from config.parameters import get_config
from visualization.quantum_plots_bulletproof import QuantumVisualization


class FractionalModelWithDerivative:
    """Fractional model that uses specific derivative types"""
    
    def __init__(self, derivative_type='caputo', alpha=1.5):
        self.base_model = FractionalModel()
        self.derivative_type = derivative_type
        self.alpha = alpha
        
        # Create the specific derivative
        if derivative_type == 'caputo':
            self.derivative = CaputoDerivative(alpha)
        elif derivative_type == 'riemann_liouville':
            self.derivative = RiemannLiouvilleDerivative(alpha)
        elif derivative_type == 'grunwald_letnikov':
            self.derivative = GrunwaldLetnikovDerivative(alpha)
        elif derivative_type == 'hilfer':
            self.derivative = HilferDerivative(alpha, beta=0.5)
        else:
            raise ValueError(f"Unknown derivative type: {derivative_type}")
    
    def __call__(self, state, t):
        """Make model callable for odeint"""
        return self.system_dynamics(state, t)
    
    def system_dynamics(self, state, t):
        """System dynamics with specific fractional derivative"""
        # Get base fractional dynamics
        base_dynamics = self.base_model.system_dynamics(state, t)
        
        # Apply fractional derivative modification
        fractional_correction = self.derivative.compute_derivative_simple(
            lambda y, t: base_dynamics, t, state, 0.01
        )
        
        # Combine with scaling factor based on alpha
        alpha_scaling = (self.alpha - 1.0) * 0.1  # Scale the fractional effect
        modified_dynamics = base_dynamics + alpha_scaling * (fractional_correction - base_dynamics)
        
        return np.clip(modified_dynamics, -10, 10)
    
    def reset_history(self):
        """Reset history for new simulations"""
        self.base_model.reset_history()
        self.derivative.reset_history()


class ComprehensiveNeuralNetworkStudy:
    """Comprehensive study comparing all model types with neural networks"""
    
    def __init__(self, output_dir='comprehensive_nn_study'):
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, 'results')
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.models_dir = os.path.join(output_dir, 'trained_models')
        
        # Create directories
        for directory in [self.output_dir, self.results_dir, self.plots_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.model_configs = {
            'integer': {'type': 'integer', 'name': 'Integer Order'},
            'caputo': {'type': 'fractional', 'derivative': 'caputo', 'alpha': 1.5, 'name': 'Caputo Fractional'},
            'riemann_liouville': {'type': 'fractional', 'derivative': 'riemann_liouville', 'alpha': 1.5, 'name': 'Riemann-Liouville'},
            'grunwald_letnikov': {'type': 'fractional', 'derivative': 'grunwald_letnikov', 'alpha': 1.5, 'name': 'Grünwald-Letnikov'},
            'hilfer': {'type': 'fractional', 'derivative': 'hilfer', 'alpha': 1.5, 'name': 'Hilfer Fractional'}
        }
        
        # Results storage
        self.all_results = {}
        self.trained_models = {}
        self.scalers = {}
        
        print(f"🧠 Comprehensive Neural Network Study Initialized")
        print(f"📂 Output directory: {output_dir}")
        print(f"🖥️ Device: {self.device}")
        print(f"📊 Models to compare: {list(self.model_configs.keys())}")
    
    def create_models(self):
        """Create all mathematical models"""
        print("\n🔧 Creating mathematical models...")
        
        self.mathematical_models = {}
        
        # Integer model
        self.mathematical_models['integer'] = IntegerModel()
        print("  ✅ Integer model created")
        
        # Fractional models with different derivatives
        for model_name, config in self.model_configs.items():
            if config['type'] == 'fractional':
                try:
                    model = FractionalModelWithDerivative(
                        derivative_type=config['derivative'],
                        alpha=config['alpha']
                    )
                    self.mathematical_models[model_name] = model
                    print(f"  ✅ {config['name']} model created (α={config['alpha']})")
                except Exception as e:
                    print(f"  ❌ Failed to create {model_name}: {e}")
        
        print(f"📊 Total models created: {len(self.mathematical_models)}")
    
    def generate_comprehensive_training_data(self, num_trajectories=200):
        """Generate comprehensive training data for all models"""
        print(f"\n📊 Generating training data ({num_trajectories} trajectories per model)...")
        
        self.training_data = {}
        
        # Time array for simulations
        t = np.linspace(0, 4, 41)  # 4 time units, 41 points
        
        for model_name, model in self.mathematical_models.items():
            print(f"  Generating data for {model_name}...")
            
            X_data = []
            y_data = []
            
            for traj_idx in range(num_trajectories):
                # Random initial conditions with controlled ranges
                init_state = [
                    np.random.uniform(20, 80),    # Tumor: 20-80
                    np.random.uniform(5, 25),     # Immune: 5-25
                    np.random.uniform(10, 40),    # Memory: 10-40
                    np.random.uniform(15, 50)     # Suppressor: 15-50
                ]
                
                try:
                    # Reset model history for fractional models
                    if hasattr(model, 'reset_history'):
                        model.reset_history()
                    
                    # Generate trajectory
                    trajectory = odeint(model, init_state, t)
                    
                    # Create training pairs: current state -> next state change
                    for i in range(len(trajectory) - 1):
                        X_data.append(trajectory[i])
                        y_data.append(trajectory[i+1] - trajectory[i])  # State change
                
                except Exception as e:
                    print(f"    ⚠️ Trajectory {traj_idx} failed: {e}")
                    continue
            
            if len(X_data) > 0:
                self.training_data[model_name] = {
                    'X': np.array(X_data),
                    'y': np.array(y_data),
                    'samples': len(X_data)
                }
                print(f"    ✅ Generated {len(X_data)} training samples")
            else:
                print(f"    ❌ No training data generated for {model_name}")
        
        print(f"📊 Training data generation completed for {len(self.training_data)} models")
    
    def train_neural_networks(self, epochs=100):
        """Train neural networks for all models"""
        print(f"\n🧠 Training neural networks ({epochs} epochs)...")
        
        for model_name in self.training_data.keys():
            print(f"\n  Training neural network for {model_name}...")
            
            # Get training data
            X = self.training_data[model_name]['X']
            y = self.training_data[model_name]['y']
            
            # Create scalers
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)
            
            # Store scalers
            self.scalers[model_name] = {'X': scaler_X, 'y': scaler_y}
            
            # Create neural network
            network = AdaptiveDynamicsNet(
                input_size=4, 
                hidden_sizes=[128, 64, 32], 
                output_size=4,
                dropout_rate=0.1
            ).to(self.device)
            
            # Train the network
            trainer = DynamicsTrainer(device=self.device)
            
            try:
                trained_model = trainer.train_model(
                    network, X, y, model_name, 
                    epochs=epochs, verbose=False
                )
                
                self.trained_models[model_name] = {
                    'model': trained_model,
                    'trainer': trainer,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }
                
                print(f"    ✅ {model_name} network trained successfully")
                
                # Save the trained model
                model_path = os.path.join(self.models_dir, f'{model_name}_model.pth')
                torch.save(trained_model.state_dict(), model_path)
                
            except Exception as e:
                print(f"    ❌ Training failed for {model_name}: {e}")
        
        print(f"🧠 Neural network training completed for {len(self.trained_models)} models")
    
    def evaluate_model_performance(self):
        """Evaluate all models on test trajectories"""
        print(f"\n📊 Evaluating model performance...")
        
        # Generate test trajectories
        test_initial_conditions = [
            [50, 10, 20, 30],   # Standard test case
            [70, 8, 15, 25],    # High tumor
            [30, 20, 25, 35],   # High immune
            [60, 12, 30, 40],   # Balanced
            [40, 15, 35, 45]    # Alternative
        ]
        
        t_test = np.linspace(0, 4, 41)
        
        for model_name in self.trained_models.keys():
            print(f"\n  Evaluating {model_name}...")
            
            model_results = {
                'predictions': [],
                'actuals': [],
                'metrics': {'rmse': [], 'mae': [], 'r2': []},
                'per_cell_metrics': {cell: {'rmse': [], 'mae': [], 'r2': []} 
                                   for cell in ['Tumor', 'Immune', 'Memory', 'Suppressor']}
            }
            
            math_model = self.mathematical_models[model_name]
            nn_model = self.trained_models[model_name]['model']
            scaler_X = self.trained_models[model_name]['scaler_X']
            scaler_y = self.trained_models[model_name]['scaler_y']
            
            for init_cond in test_initial_conditions:
                try:
                    # Reset history for fractional models
                    if hasattr(math_model, 'reset_history'):
                        math_model.reset_history()
                    
                    # Generate actual trajectory
                    actual_traj = odeint(math_model, init_cond, t_test)
                    
                    # Generate neural network prediction
                    predicted_traj = self.predict_trajectory(
                        nn_model, scaler_X, scaler_y, init_cond, t_test
                    )
                    
                    model_results['actuals'].append(actual_traj)
                    model_results['predictions'].append(predicted_traj)
                    
                    # Calculate metrics for this trajectory
                    overall_rmse = np.sqrt(mean_squared_error(actual_traj.flatten(), predicted_traj.flatten()))
                    overall_mae = mean_absolute_error(actual_traj.flatten(), predicted_traj.flatten())
                    overall_r2 = r2_score(actual_traj.flatten(), predicted_traj.flatten())
                    
                    model_results['metrics']['rmse'].append(overall_rmse)
                    model_results['metrics']['mae'].append(overall_mae)
                    model_results['metrics']['r2'].append(overall_r2)
                    
                    # Per-cell metrics
                    cell_labels = ['Tumor', 'Immune', 'Memory', 'Suppressor']
                    for i, cell in enumerate(cell_labels):
                        cell_rmse = np.sqrt(mean_squared_error(actual_traj[:, i], predicted_traj[:, i]))
                        cell_mae = mean_absolute_error(actual_traj[:, i], predicted_traj[:, i])
                        cell_r2 = r2_score(actual_traj[:, i], predicted_traj[:, i])
                        
                        model_results['per_cell_metrics'][cell]['rmse'].append(cell_rmse)
                        model_results['per_cell_metrics'][cell]['mae'].append(cell_mae)
                        model_results['per_cell_metrics'][cell]['r2'].append(cell_r2)
                
                except Exception as e:
                    print(f"    ⚠️ Evaluation failed for initial condition {init_cond}: {e}")
                    continue
            
            # Store results
            self.all_results[model_name] = model_results
            
            # Print summary metrics
            if model_results['metrics']['rmse']:
                avg_rmse = np.mean(model_results['metrics']['rmse'])
                avg_mae = np.mean(model_results['metrics']['mae'])
                avg_r2 = np.mean(model_results['metrics']['r2'])
                
                print(f"    📊 Average RMSE: {avg_rmse:.4f}")
                print(f"    📊 Average MAE:  {avg_mae:.4f}")
                print(f"    📊 Average R²:   {avg_r2:.4f}")
        
        print(f"📊 Model evaluation completed for {len(self.all_results)} models")
    
    def predict_trajectory(self, model, scaler_X, scaler_y, init_state, t):
        """Predict trajectory using neural network"""
        model.eval()
        trajectory = [init_state]
        current_state = np.array(init_state)
        
        with torch.no_grad():
            for i in range(1, len(t)):
                # Scale current state
                state_scaled = scaler_X.transform([current_state])
                state_tensor = torch.FloatTensor(state_scaled).to(self.device)
                
                # Predict change
                delta_scaled = model(state_tensor).cpu().numpy()
                delta = scaler_y.inverse_transform(delta_scaled)[0]
                
                # Update state
                next_state = np.maximum(current_state + delta, 1e-10)  # Prevent negative values
                trajectory.append(next_state)
                current_state = next_state
        
        return np.array(trajectory)
    
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        print(f"\n📈 Performing statistical analysis...")
        
        # Prepare data for statistical tests
        metrics_data = {}
        for metric in ['rmse', 'mae', 'r2']:
            metrics_data[metric] = {}
            for model_name in self.all_results.keys():
                if self.all_results[model_name]['metrics'][metric]:
                    metrics_data[metric][model_name] = self.all_results[model_name]['metrics'][metric]
        
        # Statistical tests
        statistical_results = {}
        
        for metric in metrics_data.keys():
            print(f"\n  Analyzing {metric.upper()}...")
            metric_results = metrics_data[metric]
            model_names = list(metric_results.keys())
            
            if len(model_names) < 2:
                continue
            
            statistical_results[metric] = {
                'means': {name: np.mean(values) for name, values in metric_results.items()},
                'stds': {name: np.std(values) for name, values in metric_results.items()},
                'pairwise_tests': {},
                'anova': None
            }
            
            # Pairwise t-tests
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    data1, data2 = metric_results[model1], metric_results[model2]
                    
                    try:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2))/2)
                        
                        statistical_results[metric]['pairwise_tests'][f'{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': effect_size,
                            'interpretation': self.interpret_effect_size(effect_size)
                        }
                        
                        significance = "significant" if p_value < 0.05 else "not significant"
                        print(f"    {model1} vs {model2}: p={p_value:.4f} ({significance})")
                        
                    except Exception as e:
                        print(f"    ⚠️ T-test failed for {model1} vs {model2}: {e}")
            
            # ANOVA test
            if len(model_names) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*[metric_results[name] for name in model_names])
                    statistical_results[metric]['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    print(f"    ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                except Exception as e:
                    print(f"    ⚠️ ANOVA failed: {e}")
        
        self.statistical_results = statistical_results
        print(f"📈 Statistical analysis completed")
    
    def interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n🎨 Creating comprehensive visualizations...")
        
        # 1. Performance comparison plots
        self.plot_performance_comparison()
        
        # 2. Model prediction accuracy plots
        self.plot_prediction_accuracy()
        
        # 3. Statistical significance plots
        self.plot_statistical_results()
        
        # 4. Per-cell type analysis
        self.plot_per_cell_analysis()
        
        # 5. Learning curves (if available)
        self.plot_model_rankings()
        
        print(f"🎨 Visualization creation completed")
    
    def plot_performance_comparison(self):
        """Plot comprehensive performance comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['rmse', 'mae', 'r2']
        metric_labels = ['RMSE', 'MAE', 'R²']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            model_names = []
            means = []
            stds = []
            
            for model_name in self.all_results.keys():
                if self.all_results[model_name]['metrics'][metric]:
                    values = self.all_results[model_name]['metrics'][metric]
                    model_names.append(self.model_configs[model_name]['name'])
                    means.append(np.mean(values))
                    stds.append(np.std(values))
            
            # Create bar plot with error bars
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Color bars - Integer in blue, fractional in different colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'{label} Comparison', fontweight='bold')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.annotate(f'{mean:.3f}±{std:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_accuracy(self):
        """Plot prediction accuracy for each model"""
        n_models = len(self.all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.all_results.items()):
            ax = axes[idx]
            
            # Get all actual and predicted values
            all_actual = []
            all_predicted = []
            
            for actual, predicted in zip(results['actuals'], results['predictions']):
                all_actual.extend(actual.flatten())
                all_predicted.extend(predicted.flatten())
            
            all_actual = np.array(all_actual)
            all_predicted = np.array(all_predicted)
            
            # Scatter plot
            ax.scatter(all_actual, all_predicted, alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val = min(all_actual.min(), all_predicted.min())
            max_val = max(all_actual.max(), all_predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = r2_score(all_actual, all_predicted)
            ax.set_title(f'{self.model_configs[model_name]["name"]}\nR² = {r2:.3f}', fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'prediction_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_results(self):
        """Plot statistical significance results"""
        if not hasattr(self, 'statistical_results'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['rmse', 'mae', 'r2']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric not in self.statistical_results:
                continue
            
            pairwise_tests = self.statistical_results[metric]['pairwise_tests']
            
            # Create significance matrix
            model_names = list(self.all_results.keys())
            n_models = len(model_names)
            significance_matrix = np.ones((n_models, n_models))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        test_key1 = f"{model1}_vs_{model2}"
                        test_key2 = f"{model2}_vs_{model1}"
                        
                        if test_key1 in pairwise_tests:
                            significance_matrix[i, j] = pairwise_tests[test_key1]['p_value']
                        elif test_key2 in pairwise_tests:
                            significance_matrix[i, j] = pairwise_tests[test_key2]['p_value']
            
            # Plot heatmap
            im = ax.imshow(significance_matrix, cmap='RdYlGn', vmin=0, vmax=0.1, aspect='auto')
            
            # Set labels
            display_names = [self.model_configs[name]['name'] for name in model_names]
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(display_names, rotation=45, ha='right')
            ax.set_yticklabels(display_names)
            
            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        text = f'{significance_matrix[i, j]:.3f}'
                        color = 'white' if significance_matrix[i, j] < 0.05 else 'black'
                        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
            
            ax.set_title(f'{metric.upper()} P-values\n(Green = Significant Difference)', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('P-value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'statistical_significance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_cell_analysis(self):
        """Plot per-cell type analysis"""
        cell_labels = ['Tumor', 'Immune', 'Memory', 'Suppressor']
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(len(metrics), len(cell_labels), figsize=(16, 12))
        
        for metric_idx, metric in enumerate(metrics):
            for cell_idx, cell in enumerate(cell_labels):
                ax = axes[metric_idx, cell_idx]
                
                model_names = []
                means = []
                stds = []
                
                for model_name in self.all_results.keys():
                    cell_data = self.all_results[model_name]['per_cell_metrics'][cell][metric]
                    if cell_data:
                        model_names.append(self.model_configs[model_name]['name'])
                        means.append(np.mean(cell_data))
                        stds.append(np.std(cell_data))
                
                if means:
                    bars = ax.bar(model_names, means, yerr=stds, capsize=3, alpha=0.7)
                    
                    # Color coding
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    for bar, color in zip(bars, colors[:len(bars)]):
                        bar.set_color(color)
                
                ax.set_title(f'{cell} - {metric.upper()}', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_cell_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_rankings(self):
        """Plot overall model rankings"""
        # Calculate overall scores
        model_scores = {}
        
        for model_name in self.all_results.keys():
            results = self.all_results[model_name]
            
            # Weighted score: lower RMSE and MAE are better, higher R² is better
            if results['metrics']['rmse'] and results['metrics']['mae'] and results['metrics']['r2']:
                rmse_score = 1 / (1 + np.mean(results['metrics']['rmse']))  # Inverse for lower is better
                mae_score = 1 / (1 + np.mean(results['metrics']['mae']))    # Inverse for lower is better
                r2_score = np.mean(results['metrics']['r2'])                 # Higher is better
                
                # Weighted combination
                overall_score = 0.4 * rmse_score + 0.4 * mae_score + 0.2 * r2_score
                model_scores[model_name] = overall_score
        
        if not model_scores:
            print("No scores available for ranking")
            return
        
        # Sort models by score (descending)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create ranking plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Overall scores
        model_names = [self.model_configs[name]['name'] for name, _ in sorted_models]
        scores = [score for _, score in sorted_models]
        
        bars = ax1.bar(model_names, scores, alpha=0.7)
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_title('Overall Model Rankings', fontweight='bold')
        ax1.set_ylabel('Combined Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Metric breakdown for top 3 models
        top_3_models = [name for name, _ in sorted_models[:3]]
        metrics = ['RMSE', 'MAE', 'R²']
        
        x = np.arange(len(top_3_models))
        width = 0.25
        
        for i, metric in enumerate(['rmse', 'mae', 'r2']):
            values = []
            for model_name in top_3_models:
                if self.all_results[model_name]['metrics'][metric]:
                    values.append(np.mean(self.all_results[model_name]['metrics'][metric]))
                else:
                    values.append(0)
            
            # Normalize values for visualization (except R²)
            if metric in ['rmse', 'mae']:
                values = [1/(1+v) for v in values]  # Inverse transform
            
            ax2.bar(x + i*width, values, width, label=metrics[i], alpha=0.7)
        
        ax2.set_title('Top 3 Models - Metric Breakdown', fontweight='bold')
        ax2.set_ylabel('Normalized Score')
        ax2.set_xlabel('Models')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([self.model_configs[name]['name'] for name in top_3_models])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_rankings.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print rankings
        print(f"\n🏆 Model Rankings:")
        for i, (model_name, score) in enumerate(sorted_models, 1):
            print(f"  {i}. {self.model_configs[model_name]['name']}: {score:.4f}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive text report"""
        print(f"\n📝 Generating comprehensive report...")
        
        report_path = os.path.join(self.results_dir, 'comprehensive_study_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE NEURAL NETWORK STUDY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Study Overview
            f.write("STUDY OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Models analyzed: {len(self.all_results)}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Device used: {self.device}\n\n")
            
            # Model Performance Summary
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for model_name in self.all_results.keys():
                results = self.all_results[model_name]
                f.write(f"\n{self.model_configs[model_name]['name']}:\n")
                
                if results['metrics']['rmse']:
                    avg_rmse = np.mean(results['metrics']['rmse'])
                    avg_mae = np.mean(results['metrics']['mae'])
                    avg_r2 = np.mean(results['metrics']['r2'])
                    
                    f.write(f"  Average RMSE: {avg_rmse:.6f} ± {np.std(results['metrics']['rmse']):.6f}\n")
                    f.write(f"  Average MAE:  {avg_mae:.6f} ± {np.std(results['metrics']['mae']):.6f}\n")
                    f.write(f"  Average R²:   {avg_r2:.6f} ± {np.std(results['metrics']['r2']):.6f}\n")
            
            # Statistical Analysis
            if hasattr(self, 'statistical_results'):
                f.write("\n\nSTATISTICAL ANALYSIS\n")
                f.write("-" * 25 + "\n")
                
                for metric, stat_results in self.statistical_results.items():
                    f.write(f"\n{metric.upper()} Analysis:\n")
                    
                    # ANOVA results
                    if stat_results.get('anova'):
                        anova = stat_results['anova']
                        f.write(f"  ANOVA: F={anova['f_statistic']:.4f}, p={anova['p_value']:.6f}\n")
                        f.write(f"  Significant: {'Yes' if anova['significant'] else 'No'}\n")
                    
                    # Pairwise comparisons
                    f.write("  Pairwise comparisons:\n")
                    for test_name, test_result in stat_results['pairwise_tests'].items():
                        models = test_name.replace('_vs_', ' vs ')
                        significance = "**" if test_result['significant'] else ""
                        f.write(f"    {models}: p={test_result['p_value']:.6f} {significance}\n")
                        f.write(f"      Effect size: {test_result['effect_size']:.4f} ({test_result['interpretation']})\n")
            
            # Key Findings
            f.write("\n\nKEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            
            # Identify best performing model
            if self.all_results:
                best_model = None
                best_score = -np.inf
                
                for model_name in self.all_results.keys():
                    results = self.all_results[model_name]
                    if results['metrics']['r2']:
                        avg_r2 = np.mean(results['metrics']['r2'])
                        if avg_r2 > best_score:
                            best_score = avg_r2
                            best_model = model_name
                
                if best_model:
                    f.write(f"1. Best performing model: {self.model_configs[best_model]['name']} (R² = {best_score:.4f})\n")
                
                # Fractional vs Integer comparison
                integer_r2 = np.mean(self.all_results['integer']['metrics']['r2']) if 'integer' in self.all_results else 0
                fractional_models = [name for name in self.all_results.keys() if name != 'integer']
                
                if fractional_models:
                    avg_fractional_r2 = np.mean([
                        np.mean(self.all_results[name]['metrics']['r2']) 
                        for name in fractional_models 
                        if self.all_results[name]['metrics']['r2']
                    ])
                    
                    f.write(f"2. Integer model R²: {integer_r2:.4f}\n")
                    f.write(f"3. Average fractional R²: {avg_fractional_r2:.4f}\n")
                    
                    if avg_fractional_r2 > integer_r2:
                        f.write("4. Fractional models show superior performance on average\n")
                    else:
                        f.write("4. Integer model shows competitive performance\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 18 + "\n")
            f.write("1. Use the best-performing model for primary analysis\n")
            f.write("2. Consider ensemble approaches combining multiple models\n")
            f.write("3. Validate results on independent test datasets\n")
            f.write("4. Investigate parameter sensitivity for optimal performance\n")
            f.write("5. Consider computational efficiency vs accuracy trade-offs\n")
            
            f.write(f"\n\nReport generated successfully: {report_path}\n")
        
        print(f"📝 Report saved to: {report_path}")
        return report_path
    
    def save_results(self):
        """Save all results to files"""
        print(f"\n💾 Saving results...")
        
        # Save performance metrics as CSV
        metrics_data = []
        for model_name in self.all_results.keys():
            results = self.all_results[model_name]
            if results['metrics']['rmse']:
                metrics_data.append({
                    'Model': self.model_configs[model_name]['name'],
                    'Model_Code': model_name,
                    'RMSE_Mean': np.mean(results['metrics']['rmse']),
                    'RMSE_Std': np.std(results['metrics']['rmse']),
                    'MAE_Mean': np.mean(results['metrics']['mae']),
                    'MAE_Std': np.std(results['metrics']['mae']),
                    'R2_Mean': np.mean(results['metrics']['r2']),
                    'R2_Std': np.std(results['metrics']['r2']),
                    'Samples': len(results['metrics']['rmse'])
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            csv_path = os.path.join(self.results_dir, 'performance_metrics.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Performance metrics saved to: {csv_path}")
        
        # Save statistical results
        if hasattr(self, 'statistical_results'):
            import json
            stats_path = os.path.join(self.results_dir, 'statistical_results.json')
            
            # Convert numpy types to Python types for JSON serialization
            stats_json = {}
            for metric, results in self.statistical_results.items():
                stats_json[metric] = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        stats_json[metric][key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                                  for k, v in value.items()}
                    else:
                        stats_json[metric][key] = value
            
            with open(stats_path, 'w') as f:
                json.dump(stats_json, f, indent=2, default=str)
            print(f"📈 Statistical results saved to: {stats_path}")
        
        print(f"💾 Results saving completed")
    
    def run_complete_study(self, num_trajectories=200, epochs=100):
        """Run the complete comprehensive study"""
        print(f"\n🚀 STARTING COMPREHENSIVE NEURAL NETWORK STUDY")
        print(f"=" * 70)
        
        try:
            # Step 1: Create models
            self.create_models()
            
            # Step 2: Generate training data
            self.generate_comprehensive_training_data(num_trajectories)
            
            # Step 3: Train neural networks
            self.train_neural_networks(epochs)
            
            # Step 4: Evaluate performance
            self.evaluate_model_performance()
            
            # Step 5: Statistical analysis
            self.statistical_analysis()
            
            # Step 6: Create visualizations
            self.create_comprehensive_visualizations()
            
            # Step 7: Generate report
            self.generate_comprehensive_report()
            
            # Step 8: Save results
            self.save_results()
            
            print(f"\n✅ COMPREHENSIVE STUDY COMPLETED SUCCESSFULLY!")
            print(f"📂 All results saved in: {self.output_dir}")
            print(f"📊 Plots directory: {self.plots_dir}")
            print(f"📝 Results directory: {self.results_dir}")
            print(f"🧠 Models directory: {self.models_dir}")
            
            return {
                'all_results': self.all_results,
                'statistical_results': self.statistical_results if hasattr(self, 'statistical_results') else None,
                'trained_models': self.trained_models,
                'output_directory': self.output_dir
            }
            
        except Exception as e:
            print(f"❌ Study failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the comprehensive study"""
    print("🧠 Comprehensive Neural Network Study for Fractional vs Integer Models")
    print("=" * 80)
    
    # Create study instance
    study = ComprehensiveNeuralNetworkStudy('comprehensive_nn_study_results')
    
    # Run complete study
    results = study.run_complete_study(
        num_trajectories=150,  # Number of trajectories per model
        epochs=75              # Training epochs
    )
    
    if results:
        print(f"\n🎯 STUDY SUMMARY:")
        print(f"📊 Models analyzed: {len(results['all_results'])}")
        print(f"🧠 Neural networks trained: {len(results['trained_models'])}")
        print(f"📂 Output saved to: {results['output_directory']}")
        
        # Quick performance summary
        print(f"\n🏆 QUICK PERFORMANCE OVERVIEW:")
        for model_name, model_results in results['all_results'].items():
            if model_results['metrics']['r2']:
                avg_r2 = np.mean(model_results['metrics']['r2'])
                model_display_name = study.model_configs[model_name]['name']
                print(f"  {model_display_name}: R² = {avg_r2:.4f}")
    else:
        print("❌ Study failed. Check error messages above.")


if __name__ == "__main__":
    main()