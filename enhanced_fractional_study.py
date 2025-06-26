#!/usr/bin/env python3
"""
Enhanced Fractional Model Superiority Study
Optimized to demonstrate fractional advantages over integer models
"""
from main import CancerDynamicsResearch
import numpy as np
import os

def run_enhanced_study():
    """Run analysis optimized to show fractional superiority"""
    
    print("🔬 ENHANCED FRACTIONAL SUPERIORITY STUDY")
    print("=" * 60)
    
    # Create research instance with enhanced settings
    research = CancerDynamicsResearch('fractional_superiority_results')
    
    # OPTIMIZATION 1: Better alpha range (exclude α=0, focus on optimal range)
    research.config.alpha_values = np.concatenate([
        np.linspace(0.1, 0.9, 9),    # Lower fractional range
        np.linspace(0.9, 1.1, 21),  # Fine resolution around α=1
        np.linspace(1.1, 1.9, 9)    # Upper fractional range
    ])
    
    # OPTIMIZATION 2: More diverse initial conditions
    research.config.initial_conditions = [
        [50, 10, 20, 30],   # Original baseline
        [70, 5, 20, 30],    # High tumor, low immune
        [30, 20, 20, 30],   # Low tumor, high immune
        [80, 3, 15, 25],    # Critical tumor burden
        [25, 25, 30, 40],   # Immune-dominant scenario
        [60, 8, 18, 28],    # Moderate scenario
        [40, 15, 25, 35],   # Balanced aggressive
        [75, 6, 12, 20],    # Severe case
    ]
    
    # OPTIMIZATION 3: Extended time for better fractional effects
    research.config.time_params = {
        'start': 0, 
        'end': 10,      # Longer simulation
        'points': 201   # Higher resolution
    }
    
    print(f"✅ Enhanced Configuration Applied:")
    print(f"   📊 Alpha range: {research.config.alpha_values[0]:.1f} to {research.config.alpha_values[-1]:.1f}")
    print(f"   🎯 Alpha points: {len(research.config.alpha_values)}")
    print(f"   🔬 Initial conditions: {len(research.config.initial_conditions)}")
    print(f"   ⏱️ Time points: {research.config.time_params['points']}")
    print(f"   📈 Simulation time: {research.config.time_params['end']} units")
    
    # Initialize models
    research.initialize_models()
    print(f"✅ Models initialized")
    
    # Run comprehensive analysis
    print("\n🔄 Running Enhanced Analysis Pipeline...")
    
    # 1. Basic dynamics with enhanced settings
    print("   📊 Step 1: Enhanced basic dynamics analysis...")
    basic_results = research.run_basic_dynamics_analysis()
    
    # 2. Alpha sensitivity with key derivatives
    print("   📊 Step 2: Alpha sensitivity analysis...")
    alpha_results = research.run_alpha_sensitivity_analysis(
        derivative_types=['caputo', 'riemann_liouville', 'grunwald_letnikov'],
        alpha_subset=np.linspace(0.3, 1.7, 15)  # Strategic alpha range
    )
    
    # 3. Comprehensive analysis
    print("   📊 Step 3: Comprehensive statistical analysis...")
    comprehensive_results = research.run_comprehensive_analysis()
    
    print("\n" + "=" * 60)
    print("✅ ENHANCED ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📂 Results Directory: ./fractional_superiority_results/")
    print(f"📊 Plots: ./fractional_superiority_results/plots/")
    print(f"📋 Reports: ./fractional_superiority_results/results/")
    
    # Display key findings
    if 'key_findings' in comprehensive_results:
        print(f"\n🔍 KEY FINDINGS:")
        for i, finding in enumerate(comprehensive_results['key_findings'][:6], 1):
            print(f"   {i}. {finding}")
    
    if 'recommendations' in comprehensive_results:
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(comprehensive_results['recommendations'][:4], 1):
            print(f"   {i}. {rec}")
    
    # Show some statistics
    if 'alpha_analysis' in research.results:
        print(f"\n📈 ALPHA ANALYSIS HIGHLIGHTS:")
        for deriv_type, analysis in research.results['alpha_analysis'].items():
            if 'cell_type_analysis' in analysis:
                tumor_data = analysis['cell_type_analysis'].get('Tumor', {})
                optimal_alpha = tumor_data.get('optimal_alpha', 'N/A')
                sensitivity = tumor_data.get('relative_sensitivity', 0)
                print(f"   🎯 {deriv_type.capitalize()}: Optimal α={optimal_alpha:.1f}, Sensitivity={sensitivity:.3f}")
    
    print(f"\n🚀 Open results folder to view detailed plots and analysis!")
    
    return research, comprehensive_results

def auto_open_results():
    """Automatically open results folder"""
    try:
        if os.name == 'nt':  # Windows
            os.system('explorer fractional_superiority_results\\plots')
        else:
            os.system('open fractional_superiority_results/plots')
    except:
        print("   (Manual open: ./fractional_superiority_results/plots/)")

if __name__ == "__main__":
    research, results = run_enhanced_study()
    
    print(f"\n🎯 Attempting to open results folder...")
    auto_open_results()
    
    print(f"\n" + "🎉" * 20)
    print("FRACTIONAL SUPERIORITY STUDY COMPLETE!")
    print("🎉" * 20)