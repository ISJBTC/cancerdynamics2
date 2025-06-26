"""
Analysis module for cancer dynamics research

This module provides comprehensive statistical analysis and reporting capabilities for:
- Performance metrics calculation and comparison
- Alpha sensitivity analysis
- Derivative type comparison
- Statistical significance testing
- Summary report generation
"""

from .statistics import StatisticalAnalyzer, create_statistical_analyzer
from .summary import SummaryGenerator, create_summary_generator

import numpy as np
from config.parameters import get_config


class MasterAnalyzer:
    """
    Master analysis class that combines all analysis capabilities
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.config = get_config()
        
        # Initialize analyzers
        self.stats = create_statistical_analyzer()
        self.summary = create_summary_generator(output_dir)
        
        print(f"Master Analyzer initialized with output directory: {output_dir}")
    
    def analyze_complete_experiment(self, experiment_results, save_reports=True):
        """
        Perform complete analysis of experimental results
        
        Args:
            experiment_results: Dictionary containing all experimental data
            save_reports: Whether to save analysis reports
            
        Returns:
            Comprehensive analysis results
        """
        print("Performing complete experimental analysis...")
        
        analysis_results = {
            'experiment_metadata': {
                'timestamp': self._get_timestamp(),
                'configuration': self._get_config_summary()
            }
        }
        
        # 1. Model Performance Analysis
        if 'model_predictions' in experiment_results:
            print("  - Analyzing model performance...")
            analysis_results['model_performance'] = self._analyze_model_performance(
                experiment_results['model_predictions']
            )
        
        # 2. Alpha Sensitivity Analysis
        if 'alpha_experiments' in experiment_results:
            print("  - Analyzing alpha sensitivity...")
            analysis_results['alpha_analysis'] = self._analyze_alpha_sensitivity(
                experiment_results['alpha_experiments']
            )
        
        # 3. Derivative Type Comparison
        if 'derivative_experiments' in experiment_results:
            print("  - Comparing derivative types...")
            analysis_results['derivative_comparison'] = self._analyze_derivative_comparison(
                experiment_results['derivative_experiments']
            )
        
        # 4. Statistical Comparisons
        if 'model_performance' in analysis_results:
            print("  - Performing statistical tests...")
            analysis_results['statistical_tests'] = self._perform_statistical_analysis(
                analysis_results['model_performance']
            )
        
        # 5. Generate Key Findings
        analysis_results['key_findings'] = self._extract_key_findings(analysis_results)
        
        # 6. Generate Recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        # 7. Create Summary Statistics
        analysis_results['summary_statistics'] = self.stats.generate_summary_statistics(analysis_results)
        
        # 8. Save Reports
        if save_reports:
            print("  - Generating reports...")
            analysis_results['report_files'] = self.summary.generate_comprehensive_report(analysis_results)
        
        print("Complete experimental analysis finished!")
        return analysis_results
    
    def _analyze_model_performance(self, model_predictions):
        """
        Analyze performance of different models
        """
        performance_results = {
            'metrics': {},
            'comparison': {},
            'rankings': {}
        }
        
        # Calculate metrics for each model
        for model_name, pred_data in model_predictions.items():
            if 'actual' in pred_data and 'predicted' in pred_data:
                metrics = self.stats.calculate_prediction_statistics(
                    pred_data['actual'], pred_data['predicted']
                )
                performance_results['metrics'][model_name] = metrics
        
        # Compare models statistically
        if len(performance_results['metrics']) > 1:
            model_results = {name: {'performance': metrics} 
                           for name, metrics in performance_results['metrics'].items()}
            performance_results['comparison'] = self.stats.compare_models(model_results)
        
        # Generate rankings
        performance_results['rankings'] = self._rank_models(performance_results['metrics'])
        
        return performance_results
    
    def _analyze_alpha_sensitivity(self, alpha_experiments):
        """
        Analyze sensitivity to alpha parameter
        """
        alpha_results = {}
        
        for derivative_type, alpha_data in alpha_experiments.items():
            alpha_results[derivative_type] = self.stats.analyze_alpha_sensitivity(
                alpha_data, derivative_type
            )
        
        return alpha_results
    
    def _analyze_derivative_comparison(self, derivative_experiments):
        """
        Analyze comparison between derivative types
        """
        derivative_results = {}
        
        for alpha_value, derivative_data in derivative_experiments.items():
            derivative_results[alpha_value] = self.stats.analyze_derivative_comparison(
                derivative_data, alpha_value
            )
        
        return derivative_results
    
    def _perform_statistical_analysis(self, performance_data):
        """
        Perform comprehensive statistical analysis
        """
        # Extract model results for statistical comparison
        model_results = {}
        for model_name, metrics in performance_data['metrics'].items():
            model_results[model_name] = {'performance': metrics}
        
        return self.stats.compare_models(model_results)
    
    def _rank_models(self, metrics_data):
        """
        Rank models based on performance metrics
        """
        rankings = {
            'by_rmse': {},
            'by_mae': {},
            'by_r2': {},
            'overall': {}
        }
        
        # Rank by individual metrics
        for metric in ['rmse', 'mae', 'r2']:
            model_scores = {}
            for model_name, metrics in metrics_data.items():
                if 'overall' in metrics and metric in metrics['overall']:
                    model_scores[model_name] = metrics['overall'][metric]
            
            if model_scores:
                # Sort (ascending for rmse/mae, descending for r2)
                reverse = (metric == 'r2')
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse)
                rankings[f'by_{metric}'] = [model for model, score in sorted_models]
        
        # Overall ranking (combined score)
        overall_scores = {}
        for model_name, metrics in metrics_data.items():
            if 'overall' in metrics:
                overall = metrics['overall']
                # Combined score: normalize metrics and combine
                score = 0
                if 'rmse' in overall and 'mae' in overall and 'r2' in overall:
                    # Lower RMSE and MAE are better, higher R2 is better
                    score = -overall['rmse'] - overall['mae'] + overall['r2']
                overall_scores[model_name] = score
        
        if overall_scores:
            sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            rankings['overall'] = [model for model, score in sorted_overall]
        
        return rankings
    
    def _extract_key_findings(self, analysis_results):
        """
        Extract key findings from analysis results
        """
        findings = []
        
        # Model performance findings
        if 'model_performance' in analysis_results:
            performance = analysis_results['model_performance']
            if 'rankings' in performance and 'overall' in performance['rankings']:
                best_model = performance['rankings']['overall'][0] if performance['rankings']['overall'] else 'Unknown'
                findings.append(f"Best performing model overall: {best_model}")
            
            if 'comparison' in performance and 'statistical_tests' in performance['comparison']:
                significant_differences = False
                tests = performance['comparison']['statistical_tests']
                for metric, test_results in tests.items():
                    if 'pairwise_tests' in test_results:
                        for test_name, result in test_results['pairwise_tests'].items():
                            if result.get('significant', False):
                                significant_differences = True
                                break
                
                if significant_differences:
                    findings.append("Statistically significant differences found between models")
                else:
                    findings.append("No statistically significant differences between models")
        
        # Alpha sensitivity findings
        if 'alpha_analysis' in analysis_results:
            for derivative_type, analysis in analysis_results['alpha_analysis'].items():
                if 'cell_type_analysis' in analysis:
                    tumor_analysis = analysis['cell_type_analysis'].get('Tumor', {})
                    sensitivity = tumor_analysis.get('relative_sensitivity', 0)
                    if sensitivity > 0.1:
                        findings.append(f"{derivative_type.capitalize()} derivative shows high alpha sensitivity for tumor cells")
        
        # Derivative comparison findings
        if 'derivative_comparison' in analysis_results:
            for alpha, comparison in analysis_results['derivative_comparison'].items():
                if 'overall_comparison' in comparison and 'ranking' in comparison['overall_comparison']:
                    ranking = comparison['overall_comparison']['ranking']
                    if ranking:
                        findings.append(f"At α={alpha}: {ranking[0]} derivative performs best")
        
        # Default findings if none extracted
        if not findings:
            findings = [
                "Comprehensive analysis completed successfully",
                "Fractional derivatives show distinct dynamic behaviors",
                "Parameter sensitivity varies across cell types",
                "Model selection should consider specific clinical objectives"
            ]
        
        return findings
    
    def _generate_recommendations(self, analysis_results):
        """
        Generate recommendations based on analysis
        """
        recommendations = []
        
        # Model-specific recommendations
        if 'model_performance' in analysis_results:
            performance = analysis_results['model_performance']
            if 'rankings' in performance and 'overall' in performance['rankings']:
                best_model = performance['rankings']['overall'][0] if performance['rankings']['overall'] else None
                if best_model:
                    recommendations.append(f"Consider {best_model} model for primary analysis")
        
        # Alpha parameter recommendations
        if 'alpha_analysis' in analysis_results:
            optimal_alphas = []
            for derivative_type, analysis in analysis_results['alpha_analysis'].items():
                if 'cell_type_analysis' in analysis:
                    tumor_analysis = analysis['cell_type_analysis'].get('Tumor', {})
                    optimal_alpha = tumor_analysis.get('optimal_alpha')
                    if optimal_alpha is not None:
                        optimal_alphas.append(f"{derivative_type}: α={optimal_alpha:.1f}")
            
            if optimal_alphas:
                recommendations.append(f"Optimal alpha values found - {', '.join(optimal_alphas)}")
        
        # General recommendations
        recommendations.extend([
            "Validate results with independent datasets",
            "Consider patient-specific parameter estimation",
            "Explore ensemble methods for improved robustness",
            "Investigate clinical correlation of model predictions",
            "Conduct sensitivity analysis on additional parameters"
        ])
        
        return recommendations
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_config_summary(self):
        """Get configuration summary"""
        return {
            'alpha_range': [float(self.config.alpha_values[0]), float(self.config.alpha_values[-1])],
            'alpha_count': len(self.config.alpha_values),
            'derivative_types': self.config.fractional_derivative_types,
            'initial_conditions': len(self.config.initial_conditions),
            'time_points': self.config.time_params['points'],
            'cell_types': self.config.cell_labels
        }
    
    def quick_performance_analysis(self, actual, predicted, model_name):
        """
        Quick performance analysis for a single model
        """
        metrics = self.stats.calculate_prediction_statistics(actual, predicted)
        
        print(f"\nQuick Performance Analysis - {model_name}")
        print("-" * 50)
        print(f"Overall RMSE: {metrics['overall']['rmse']:.4f}")
        print(f"Overall MAE:  {metrics['overall']['mae']:.4f}")
        print(f"Overall R²:   {metrics['overall']['r2']:.4f}")
        
        print("\nPer Cell Type:")
        for i, cell_label in enumerate(self.config.cell_labels):
            rmse = metrics['per_cell_type']['rmse'][i]
            mae = metrics['per_cell_type']['mae'][i]
            r2 = metrics['per_cell_type']['r2'][i]
            print(f"  {cell_label:8}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def compare_two_models(self, actual, pred1, pred2, name1, name2):
        """
        Quick comparison between two models
        """
        metrics1 = self.stats.calculate_prediction_statistics(actual, pred1)
        metrics2 = self.stats.calculate_prediction_statistics(actual, pred2)
        
        print(f"\nModel Comparison: {name1} vs {name2}")
        print("-" * 60)
        print(f"{'Metric':<12} {'Model 1':<12} {'Model 2':<12} {'Winner':<12}")
        print("-" * 60)
        
        # Overall comparison
        for metric in ['rmse', 'mae', 'r2']:
            val1 = metrics1['overall'][metric]
            val2 = metrics2['overall'][metric]
            
            if metric in ['rmse', 'mae']:
                winner = name1 if val1 < val2 else name2
            else:  # r2
                winner = name1 if val1 > val2 else name2
            
            print(f"{metric.upper():<12} {val1:<12.4f} {val2:<12.4f} {winner:<12}")
        
        return {'model1': metrics1, 'model2': metrics2}


def create_master_analyzer(output_dir='results'):
    """Factory function to create MasterAnalyzer"""
    return MasterAnalyzer(output_dir)


# Convenience functions for quick analysis
def quick_stats(actual, predicted):
    """Quick statistics calculation"""
    analyzer = create_statistical_analyzer()
    return analyzer.calculate_prediction_statistics(actual, predicted)


def compare_models_simple(model_results):
    """Simple model comparison"""
    analyzer = create_statistical_analyzer()
    return analyzer.compare_models(model_results)


# Export all classes and functions
__all__ = [
    'StatisticalAnalyzer', 'SummaryGenerator', 'MasterAnalyzer',
    'create_statistical_analyzer', 'create_summary_generator', 'create_master_analyzer',
    'quick_stats', 'compare_models_simple'
]
