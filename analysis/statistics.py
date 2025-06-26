import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
from config.parameters import get_config

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Statistical analysis class for cancer dynamics research
    """
    
    def __init__(self):
        self.config = get_config()
        self.cell_labels = self.config.cell_labels
    
    def calculate_prediction_statistics(self, actual, predicted):
        """
        Calculate prediction statistics including RMSE, MAE, and R² - exact copy from original
        
        Args:
            actual: Actual trajectory values
            predicted: Predicted trajectory values
            
        Returns:
            Dictionary with overall and per-cell-type metrics
        """
        # Flatten arrays for overall metrics
        actual_flat = actual.flatten()
        pred_flat = predicted.flatten()
        
        # Overall metrics
        rmse_overall = np.sqrt(mean_squared_error(actual_flat, pred_flat))
        mae_overall = mean_absolute_error(actual_flat, pred_flat)
        r2_overall = r2_score(actual_flat, pred_flat)
        
        # Per cell type metrics
        rmse_per_cell = []
        mae_per_cell = []
        r2_per_cell = []
        
        for i in range(actual.shape[1]):  # For each cell type
            rmse_cell = np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))
            mae_cell = mean_absolute_error(actual[:, i], predicted[:, i])
            r2_cell = r2_score(actual[:, i], predicted[:, i])
            
            rmse_per_cell.append(rmse_cell)
            mae_per_cell.append(mae_cell)
            r2_per_cell.append(r2_cell)
        
        return {
            'overall': {
                'rmse': rmse_overall,
                'mae': mae_overall,
                'r2': r2_overall
            },
            'per_cell_type': {
                'rmse': rmse_per_cell,
                'mae': mae_per_cell,
                'r2': r2_per_cell
            }
        }
    
    def calculate_trajectory_metrics(self, trajectories, metric_type='final_value'):
        """
        Calculate metrics from trajectory data
        
        Args:
            trajectories: List of trajectory arrays
            metric_type: Type of metric ('final_value', 'max_value', 'mean_value', 'stability')
            
        Returns:
            Dictionary with metrics for each cell type
        """
        metrics = {cell: [] for cell in self.cell_labels}
        
        for trajectory in trajectories:
            for i, cell_label in enumerate(self.cell_labels):
                if metric_type == 'final_value':
                    value = trajectory[-1, i]
                elif metric_type == 'max_value':
                    value = np.max(trajectory[:, i])
                elif metric_type == 'mean_value':
                    value = np.mean(trajectory[:, i])
                elif metric_type == 'stability':
                    # Coefficient of variation in final 20% of trajectory
                    final_portion = trajectory[int(0.8*len(trajectory)):, i]
                    value = np.std(final_portion) / (np.mean(final_portion) + 1e-10)
                else:
                    value = trajectory[-1, i]
                
                metrics[cell_label].append(value)
        
        # Convert to arrays and calculate statistics
        result = {}
        for cell_label in self.cell_labels:
            values = np.array(metrics[cell_label])
            result[cell_label] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
        
        return result
    
    def compare_models(self, model_results):
        """
        Compare multiple models statistically
        
        Args:
            model_results: Dict with model names as keys, results as values
            
        Returns:
            Statistical comparison results
        """
        comparison = {
            'model_names': list(model_results.keys()),
            'metrics_comparison': {},
            'statistical_tests': {}
        }
        
        # Extract metrics for comparison
        metrics = ['rmse', 'mae', 'r2']
        
        for metric in metrics:
            comparison['metrics_comparison'][metric] = {}
            
            for model_name, results in model_results.items():
                if 'performance' in results:
                    perf = results['performance']
                    if 'per_cell_type' in perf and metric in perf['per_cell_type']:
                        comparison['metrics_comparison'][metric][model_name] = perf['per_cell_type'][metric]
        
        # Perform statistical tests if we have multiple models
        model_names = list(model_results.keys())
        if len(model_names) >= 2:
            for metric in metrics:
                if metric in comparison['metrics_comparison']:
                    comparison['statistical_tests'][metric] = self._perform_statistical_tests(
                        comparison['metrics_comparison'][metric]
                    )
        
        return comparison
    
    def _perform_statistical_tests(self, metric_data):
        """
        Perform statistical tests between models
        """
        model_names = list(metric_data.keys())
        results = {}
        
        if len(model_names) >= 2:
            # Pairwise t-tests
            results['pairwise_tests'] = {}
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    data1, data2 = metric_data[model1], metric_data[model2]
                    
                    try:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(data1, data2)
                        
                        results['pairwise_tests'][f'{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2))/2)
                        }
                    except:
                        results['pairwise_tests'][f'{model1}_vs_{model2}'] = {
                            'error': 'Could not perform test'
                        }
            
            # ANOVA if more than 2 models
            if len(model_names) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*[metric_data[model] for model in model_names])
                    results['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    results['anova'] = {'error': 'Could not perform ANOVA'}
        
        return results
    
    def analyze_alpha_sensitivity(self, alpha_results, derivative_type):
        """
        Analyze sensitivity to alpha parameter
        
        Args:
            alpha_results: Dict with alpha values as keys, trajectories as values
            derivative_type: Type of fractional derivative
            
        Returns:
            Alpha sensitivity analysis results
        """
        alpha_values = sorted(alpha_results.keys())
        
        sensitivity_analysis = {
            'derivative_type': derivative_type,
            'alpha_values': alpha_values,
            'cell_type_analysis': {}
        }
        
        for i, cell_label in enumerate(self.cell_labels):
            # Extract final values for each alpha
            final_values = [alpha_results[alpha][-1, i] for alpha in alpha_values]
            
            # Calculate sensitivity metrics
            sensitivity_analysis['cell_type_analysis'][cell_label] = {
                'final_values': final_values,
                'alpha_range': alpha_values[-1] - alpha_values[0],
                'value_range': max(final_values) - min(final_values),
                'relative_sensitivity': (max(final_values) - min(final_values)) / (np.mean(final_values) + 1e-10),
                'monotonicity': self._calculate_monotonicity(alpha_values, final_values),
                'optimal_alpha': alpha_values[np.argmin(final_values) if cell_label == 'Tumor' else np.argmax(final_values)]
            }
            
            # Correlation with alpha
            try:
                correlation, p_value = stats.pearsonr(alpha_values, final_values)
                sensitivity_analysis['cell_type_analysis'][cell_label]['correlation'] = {
                    'coefficient': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                sensitivity_analysis['cell_type_analysis'][cell_label]['correlation'] = {
                    'error': 'Could not calculate correlation'
                }
        
        return sensitivity_analysis
    
    def _calculate_monotonicity(self, x_values, y_values):
        """
        Calculate monotonicity of a relationship
        """
        if len(y_values) < 2:
            return 0.0
            
        differences = np.diff(y_values)
        
        if len(differences) == 0:
            return 0.0
        
        # Count increasing vs decreasing
        increasing = np.sum(differences > 0)
        decreasing = np.sum(differences < 0)
        
        # Monotonicity score: 1 = perfectly monotonic, 0 = random
        monotonicity = abs(increasing - decreasing) / len(differences)
        
        return monotonicity
    
    def analyze_derivative_comparison(self, derivative_results, alpha_value):
        """
        Compare different derivative types at fixed alpha
        
        Args:
            derivative_results: Dict with derivative types as keys, trajectories as values
            alpha_value: Fixed alpha value used
            
        Returns:
            Derivative comparison analysis
        """
        derivative_types = list(derivative_results.keys())
        
        comparison = {
            'alpha_value': alpha_value,
            'derivative_types': derivative_types,
            'cell_type_analysis': {},
            'overall_comparison': {}
        }
        
        for i, cell_label in enumerate(self.cell_labels):
            # Extract final values for each derivative type
            final_values = {dt: derivative_results[dt][-1, i] for dt in derivative_types}
            
            comparison['cell_type_analysis'][cell_label] = {
                'final_values': final_values,
                'best_derivative': min(final_values.keys(), key=final_values.get) if cell_label == 'Tumor' 
                                 else max(final_values.keys(), key=final_values.get),
                'value_range': max(final_values.values()) - min(final_values.values()),
                'coefficient_of_variation': np.std(list(final_values.values())) / (np.mean(list(final_values.values())) + 1e-10)
            }
        
        # Overall ranking
        rankings = {}
        for dt in derivative_types:
            # Simple ranking based on tumor reduction and immune enhancement
            tumor_score = -derivative_results[dt][-1, 0]  # Lower tumor is better
            immune_score = derivative_results[dt][-1, 1]   # Higher immune is better
            rankings[dt] = tumor_score + immune_score
        
        comparison['overall_comparison']['ranking'] = sorted(rankings.keys(), key=rankings.get, reverse=True)
        comparison['overall_comparison']['scores'] = rankings
        
        return comparison
    
    def calculate_convergence_metrics(self, step_sizes, errors):
        """
        Calculate convergence order for numerical methods
        
        Args:
            step_sizes: Array of step sizes used
            errors: Corresponding errors
            
        Returns:
            Convergence analysis results
        """
        # Log-log regression to find convergence order
        log_h = np.log(step_sizes)
        log_e = np.log(errors)
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_e)
        
        return {
            'convergence_order': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'standard_error': std_err,
            'intercept': intercept,
            'is_significant': p_value < 0.05
        }
    
    def generate_summary_statistics(self, all_results):
        """
        Generate comprehensive summary statistics
        
        Args:
            all_results: Dictionary containing all analysis results
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'experiment_overview': {
                'total_models': len(all_results.get('models', {})),
                'alpha_values_tested': len(all_results.get('alpha_analysis', {}).get('alpha_values', [])),
                'derivative_types': len(all_results.get('derivative_analysis', {}).get('types', [])),
                'cell_types_analyzed': len(self.cell_labels)
            },
            'performance_summary': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Performance summary
        if 'model_comparison' in all_results:
            comparison = all_results['model_comparison']
            if 'metrics_comparison' in comparison:
                summary['performance_summary'] = self._summarize_performance(comparison['metrics_comparison'])
        
        # Key findings
        summary['key_findings'] = self._extract_key_findings(all_results)
        
        # Recommendations
        summary['recommendations'] = self._generate_recommendations(all_results)
        
        return summary
    
    def _summarize_performance(self, metrics_comparison):
        """Summarize performance metrics across models"""
        summary = {}
        
        for metric in ['rmse', 'mae', 'r2']:
            if metric in metrics_comparison:
                model_data = metrics_comparison[metric]
                
                # Find best performing model for each cell type
                best_models = {}
                for i, cell_label in enumerate(self.cell_labels):
                    cell_values = {model: values[i] for model, values in model_data.items()}
                    
                    if metric in ['rmse', 'mae']:
                        best_model = min(cell_values.keys(), key=cell_values.get)
                    else:  # r2
                        best_model = max(cell_values.keys(), key=cell_values.get)
                    
                    best_models[cell_label] = best_model
                
                summary[metric] = {
                    'best_models_by_cell': best_models,
                    'overall_best': max(model_data.keys(), 
                                      key=lambda m: np.mean(model_data[m]) if metric == 'r2' 
                                      else -np.mean(model_data[m]))
                }
        
        return summary
    
    def _extract_key_findings(self, all_results):
        """Extract key findings from results"""
        findings = []
        
        # Model performance findings
        if 'model_comparison' in all_results:
            findings.append("Model performance comparison completed")
        
        # Alpha sensitivity findings
        if 'alpha_analysis' in all_results:
            findings.append("Alpha parameter sensitivity analysis shows varied responses")
        
        # Derivative comparison findings
        if 'derivative_analysis' in all_results:
            findings.append("Different fractional derivatives show distinct dynamics")
        
        return findings
    
    def _generate_recommendations(self, all_results):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        recommendations.append("Use cross-validation for robust model selection")
        recommendations.append("Consider ensemble methods for improved predictions")
        recommendations.append("Investigate parameter ranges more thoroughly")
        recommendations.append("Validate results with experimental data")
        
        return recommendations


def create_statistical_analyzer():
    """Factory function to create StatisticalAnalyzer"""
    return StatisticalAnalyzer()

    