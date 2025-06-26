#!/usr/bin/env python3
"""
Comprehensive analysis using ALL fractional derivative types from the project
Focus on α=1.7 region where fractional models show optimal performance
"""
from main import CancerDynamicsResearch
import numpy as np

def run_comprehensive_all_derivatives():
    """Run analysis with all 5 fractional derivative types"""
    
    print("🔬 COMPREHENSIVE ANALYSIS - ALL FRACTIONAL DERIVATIVES")
    print("=" * 70)
    
    research = CancerDynamicsResearch('comprehensive_all_derivatives_results')
    
    # OPTIMIZATION 1: Focus on α=1.7 region with fine resolution
    research.config.alpha_values = np.concatenate([
        np.linspace(0.5, 1.0, 11),   # Lower range
        np.linspace(1.0, 1.7, 21),  # Approach optimal (fine resolution)
        np.linspace(1.7, 2.0, 16)   # Beyond optimal
    ])
    
    # OPTIMIZATION 2: Strategic initial conditions based on previous findings
    # Include scenarios where Memory and Stromal favor fractional
    research.config.initial_conditions = [
        # Original scenarios
        [50, 10, 20, 30],   # Baseline
        [70, 5, 20, 30],    # High tumor, low immune
        [30, 20, 20, 30],   # Low tumor, high immune
        
        # Memory-focused scenarios (fractional showed advantage)
        [45, 12, 35, 25],   # High memory cells
        [55, 8, 40, 20],    # Very high memory
        [35, 18, 45, 15],   # Memory dominant
        
        # Stromal-focused scenarios (fractional showed advantage)
        [40, 15, 20, 50],   # High stromal
        [60, 10, 15, 60],   # Very high stromal
        [25, 25, 25, 70],   # Stromal dominant
        
        # Extreme scenarios for better differentiation
        [85, 3, 10, 15],    # Critical tumor burden
        [15, 35, 40, 45],   # Recovery scenario
    ]
    
    # OPTIMIZATION 3: Extended simulation for better fractional effects
    research.config.time_params = {
        'start': 0,
        'end': 12,      # Extended time
        'points': 241   # High resolution
    }
    
    print(f"✅ Comprehensive Configuration:")
    print(f"   📊 Alpha range: {research.config.alpha_values[0]:.1f} to {research.config.alpha_values[-1]:.1f}")
    print(f"   🎯 Alpha points: {len(research.config.alpha_values)} (focused on α=1.7)")
    print(f"   🔬 Initial conditions: {len(research.config.initial_conditions)} (memory/stromal focused)")
    print(f"  ⏱️ Time points: {research.config.time_params['points']}")
    print(f"   📈 Simulation time: {research.config.time_params['end']} units")
    
    # Initialize models
    research.initialize_models()
    print(f"✅ All models initialized")
    
    # Run comprehensive analysis
    print(f"\n🔄 Running Comprehensive Analysis with ALL Derivatives...")
    
    # 1. Enhanced basic dynamics
    print("   📊 Step 1: Enhanced basic dynamics (11 scenarios)...")
    basic_results = research.run_basic_dynamics_analysis()
    
    # 2. ALL FRACTIONAL DERIVATIVES with focused alpha range
    print("   📊 Step 2: ALL fractional derivatives analysis...")
    alpha_results = research.run_alpha_sensitivity_analysis(
        derivative_types=['caputo', 'riemann_liouville', 'grunwald_letnikov', 'hilfer', 'gutan'],
        alpha_subset=np.linspace(1.4, 2.0, 13)  # Focus on optimal region
    )
    
    # 3. Comprehensive statistical analysis
    print("   📊 Step 3: Comprehensive statistical analysis...")
    comprehensive_results = research.run_comprehensive_analysis()
    
    print(f"\n" + "=" * 70)
    print("✅ COMPREHENSIVE ALL-DERIVATIVES ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"📂 Results: ./comprehensive_all_derivatives_results/")
    print(f"📊 Plots: ./comprehensive_all_derivatives_results/plots/")
    print(f"📋 Reports: ./comprehensive_all_derivatives_results/results/")
    
    # Enhanced results summary
    print(f"\n🔍 DETAILED ANALYSIS SUMMARY:")
    
    # Alpha analysis for all derivatives
    if 'alpha_analysis' in research.results:
        print(f"\n📈 ALPHA OPTIMIZATION RESULTS (All Derivatives):")
        print(f"{'Derivative':<20} {'Tumor α':<10} {'Immune α':<10} {'Memory α':<10} {'Stromal α':<10}")
        print(f"{'-'*70}")
        
        for deriv_type, analysis in research.results['alpha_analysis'].items():
            if 'cell_type_analysis' in analysis:
                tumor_alpha = analysis['cell_type_analysis'].get('Tumor', {}).get('optimal_alpha', 'N/A')
                immune_alpha = analysis['cell_type_analysis'].get('Immune', {}).get('optimal_alpha', 'N/A')
                memory_alpha = analysis['cell_type_analysis'].get('Memory', {}).get('optimal_alpha', 'N/A')
                stromal_alpha = analysis['cell_type_analysis'].get('Stromal', {}).get('optimal_alpha', 'N/A')
                
                print(f"{deriv_type.capitalize():<20} {tumor_alpha:<10.1f} {immune_alpha:<10.1f} {memory_alpha:<10.1f} {stromal_alpha:<10.1f}")
    
    # Sensitivity analysis
    if 'alpha_analysis' in research.results:
        print(f"\n📊 SENSITIVITY ANALYSIS (All Derivatives):")
        print(f"{'Derivative':<20} {'Tumor Sens':<12} {'Immune Sens':<12} {'Memory Sens':<12} {'Stromal Sens':<12}")
        print(f"{'-'*80}")
        
        for deriv_type, analysis in research.results['alpha_analysis'].items():
            if 'cell_type_analysis' in analysis:
                tumor_sens = analysis['cell_type_analysis'].get('Tumor', {}).get('relative_sensitivity', 0)
                immune_sens = analysis['cell_type_analysis'].get('Immune', {}).get('relative_sensitivity', 0)
                memory_sens = analysis['cell_type_analysis'].get('Memory', {}).get('relative_sensitivity', 0)
                stromal_sens = analysis['cell_type_analysis'].get('Stromal', {}).get('relative_sensitivity', 0)
                
                print(f"{deriv_type.capitalize():<20} {tumor_sens:<12.4f} {immune_sens:<12.4f} {memory_sens:<12.4f} {stromal_sens:<12.4f}")
    
    # Key findings
    if 'key_findings' in comprehensive_results:
        print(f"\n🔍 KEY FINDINGS:")
        for i, finding in enumerate(comprehensive_results['key_findings'], 1):
            print(f"   {i}. {finding}")
    
    # Recommendations
    if 'recommendations' in comprehensive_results:
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(comprehensive_results['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n🚀 Open results folder for detailed visualization!")
    print(f"🎯 Focus on Memory and Stromal cell plots - fractional advantage expected!")
    
    return research, comprehensive_results

def auto_open_results():
    """Open results folder"""
    import os
    try:
        if os.name == 'nt':  # Windows
            os.system('explorer comprehensive_all_derivatives_results\\plots')
    except:
        print("   📂 Manual: ./comprehensive_all_derivatives_results/plots/")

if __name__ == "__main__":
    research, results = run_comprehensive_all_derivatives()
    
    print(f"\n🎯 Opening results folder...")
    auto_open_results()
    
    print(f"\n" + "🎉" * 25)
    print("ALL DERIVATIVES COMPREHENSIVE ANALYSIS COMPLETE!")
    print("🎉" * 25)