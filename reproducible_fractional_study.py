#!/usr/bin/env python3
"""
Reproducible fractional superiority study with fixed seeds
"""
from main import CancerDynamicsResearch
import numpy as np
import torch
import random

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # For reproducible behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_reproducible_study():
    """Run reproducible study with fixed seeds"""
    
    print("🔬 REPRODUCIBLE FRACTIONAL SUPERIORITY STUDY")
    print("=" * 60)
    print("🎯 Using fixed seeds for consistent results")
    
    # SET REPRODUCIBLE SEEDS
    set_all_seeds(42)
    
    research = CancerDynamicsResearch('reproducible_results')
    
    # FIXED CONFIGURATION
    research.config.alpha_values = np.linspace(0.5, 2.0, 31)
    
    # FIXED initial conditions (no randomness)
    research.config.initial_conditions = [
        [50, 10, 20, 30],   # Baseline
        [70, 5, 20, 30],    # High tumor
        [30, 20, 20, 30],   # High immune
        [80, 3, 15, 25],    # Critical
        [25, 25, 30, 40],   # Immune dominant
        [60, 8, 18, 28],    # Moderate
        [40, 15, 25, 35],   # Balanced
        [75, 6, 12, 20],    # Severe
        [45, 12, 35, 25],   # Memory focused
        [35, 18, 20, 50],   # Stromal focused
    ]
    
    # Extended simulation
    research.config.time_params = {'start': 0, 'end': 15, 'points': 301}
    
    print(f"✅ Reproducible Configuration Applied")
    
    # Initialize with fixed seeds
    set_all_seeds(42)
    research.initialize_models()
    
    print(f"\n🔄 Running Reproducible Analysis...")
    
    # Basic dynamics
    set_all_seeds(42)
    basic_results = research.run_basic_dynamics_analysis()
    
    # Alpha sensitivity 
    set_all_seeds(42)
    alpha_results = research.run_alpha_sensitivity_analysis(
        derivative_types=['caputo', 'riemann_liouville', 'grunwald_letnikov'],
        alpha_subset=np.linspace(1.0, 2.0, 21)
    )
    
    # Neural networks with fixed seeds
    set_all_seeds(42)
    research.initialize_neural_networks()
    nn_results = research.run_neural_network_training(train_epochs=100)
    
    # Comprehensive analysis
    set_all_seeds(42)
    comprehensive_results = research.run_comprehensive_analysis()
    
    print(f"\n✅ REPRODUCIBLE STUDY COMPLETE!")
    print(f"📂 Results: ./reproducible_results/")
    
    return research, comprehensive_results

def run_multiple_trials(num_trials=5):
    """Run multiple trials to see consistency"""
    
    print(f"🔬 MULTIPLE TRIAL ANALYSIS ({num_trials} trials)")
    print("=" * 50)
    
    results_summary = []
    
    for trial in range(num_trials):
        print(f"\n🔄 Trial {trial + 1}/{num_trials}")
        
        # Use different seed for each trial
        set_all_seeds(trial * 100)
        
        research = CancerDynamicsResearch(f'trial_{trial+1}_results')
        
        # Quick configuration
        research.config.alpha_values = np.linspace(1.0, 2.0, 11)
        research.config.initial_conditions = [
            [50, 10, 20, 30],
            [70, 5, 20, 30], 
            [30, 20, 20, 30]
        ]
        research.config.time_params = {'start': 0, 'end': 10, 'points': 101}
        
        # Run analysis
        research.initialize_models()
        basic_results = research.run_basic_dynamics_analysis()
        comprehensive_results = research.run_comprehensive_analysis()
        
        # Extract key metrics
        if 'model_performance' in comprehensive_results:
            metrics = comprehensive_results['model_performance']['metrics']
            
            int_rmse = metrics.get('integer', {}).get('overall', {}).get('rmse', 0)
            frac_rmse = metrics.get('fractional', {}).get('overall', {}).get('rmse', 0)
            
            winner = 'Fractional' if frac_rmse < int_rmse else 'Integer'
            improvement = abs(int_rmse - frac_rmse) / max(int_rmse, frac_rmse) * 100
            
            results_summary.append({
                'trial': trial + 1,
                'integer_rmse': int_rmse,
                'fractional_rmse': frac_rmse,
                'winner': winner,
                'improvement_pct': improvement
            })
            
            print(f"   Integer RMSE: {int_rmse:.4f}")
            print(f"   Fractional RMSE: {frac_rmse:.4f}")
            print(f"   Winner: {winner} ({improvement:.1f}% better)")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("MULTIPLE TRIAL SUMMARY")
    print("=" * 50)
    
    fractional_wins = sum(1 for r in results_summary if r['winner'] == 'Fractional')
    integer_wins = sum(1 for r in results_summary if r['winner'] == 'Integer')
    
    print(f"Fractional wins: {fractional_wins}/{num_trials}")
    print(f"Integer wins: {integer_wins}/{num_trials}")
    
    avg_frac_rmse = np.mean([r['fractional_rmse'] for r in results_summary])
    avg_int_rmse = np.mean([r['integer_rmse'] for r in results_summary])
    
    print(f"\nAverage Integer RMSE: {avg_int_rmse:.4f}")
    print(f"Average Fractional RMSE: {avg_frac_rmse:.4f}")
    
    overall_winner = 'Fractional' if avg_frac_rmse < avg_int_rmse else 'Integer'
    overall_improvement = abs(avg_int_rmse - avg_frac_rmse) / max(avg_int_rmse, avg_frac_rmse) * 100
    
    print(f"\n🏆 OVERALL WINNER: {overall_winner}")
    print(f"📊 Average improvement: {overall_improvement:.1f}%")
    
    return results_summary

if __name__ == "__main__":
    # Run single reproducible study
    research, results = run_reproducible_study()
    
    print(f"\n" + "="*30)
    print("Want to run multiple trials? (y/n): ", end="")
    
    # For automation, let's run multiple trials
    print("y")  # Auto-answer
    trial_results = run_multiple_trials(3)
    
    print(f"\n🎯 CONCLUSION: Results may vary due to stochastic nature")
    print(f"🔬 This is normal in complex systems!")
    print(f"📊 Focus on average performance across trials")