import numpy as np
import os
from datetime import datetime
from config.parameters import get_config


class SummaryGenerator:
    """
    Generate comprehensive summaries and reports for cancer dynamics research
    """
    
    def __init__(self, output_dir='results'):
        self.config = get_config()
        self.output_dir = output_dir
        self.cell_labels = self.config.cell_labels
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_text_summary(self, analysis_results, filename='analysis_summary.txt'):
        """
        Generate comprehensive text summary - similar to original code style
        
        Args:
            analysis_results: Dictionary containing all analysis results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("CANCER DYNAMICS RESEARCH - ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration Overview
            f.write("CONFIGURATION OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Alpha values tested: {len(self.config.alpha_values)}\n")
            f.write(f"Alpha range: {self.config.alpha_values[0]:.1f} to {self.config.alpha_values[-1]:.1f}\n")
            f.write(f"Derivative types: {', '.join(self.config.fractional_derivative_types)}\n")
            f.write(f"Initial conditions: {len(self.config.initial_conditions)} sets\n")
            f.write(f"Time points: {self.config.time_params['points']}\n")
            f.write(f"Cell types: {', '.join(self.cell_labels)}\n\n")
            
            # Model Performance
            if 'model_performance' in analysis_results:
                self._write_model_performance(f, analysis_results['model_performance'])
            
            # Alpha Analysis
            if 'alpha_analysis' in analysis_results:
                self._write_alpha_analysis(f, analysis_results['alpha_analysis'])
            
            # Derivative Comparison
            if 'derivative_comparison' in analysis_results:
                self._write_derivative_comparison(f, analysis_results['derivative_comparison'])
            
            # Statistical Tests
            if 'statistical_tests' in analysis_results:
                self._write_statistical_tests(f, analysis_results['statistical_tests'])
            
            # Key Findings
            if 'key_findings' in analysis_results:
                self._write_key_findings(f, analysis_results['key_findings'])
            
            # Recommendations
            if 'recommendations' in analysis_results:
                self._write_recommendations(f, analysis_results['recommendations'])
        
        print(f"Text summary saved to: {filepath}")
        return filepath
    
    def _write_model_performance(self, f, performance_data):
        """Write model performance section"""
        f.write("MODEL PERFORMANCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        if 'metrics' in performance_data:
            for model_name, metrics in performance_data['metrics'].items():
                f.write(f"\n{model_name.upper()} MODEL:\n")
                
                if 'overall' in metrics:
                    overall = metrics['overall']
                    f.write(f"  Overall RMSE: {overall.get('rmse', 'N/A'):.4f}\n")
                    f.write(f"  Overall MAE:  {overall.get('mae', 'N/A'):.4f}\n")
                    f.write(f"  Overall R²:   {overall.get('r2', 'N/A'):.4f}\n")
                
                if 'per_cell_type' in metrics:
                    f.write(f"  Cell-specific performance:\n")
                    per_cell = metrics['per_cell_type']
                    for i, cell_label in enumerate(self.cell_labels):
                        rmse = per_cell.get('rmse', [0]*4)[i] if i < len(per_cell.get('rmse', [])) else 'N/A'
                        mae = per_cell.get('mae', [0]*4)[i] if i < len(per_cell.get('mae', [])) else 'N/A'
                        r2 = per_cell.get('r2', [0]*4)[i] if i < len(per_cell.get('r2', [])) else 'N/A'
                        f.write(f"    {cell_label}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}\n")
        
        f.write("\n")
    
    def _write_alpha_analysis(self, f, alpha_data):
        """Write alpha sensitivity analysis section"""
        f.write("ALPHA SENSITIVITY ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        for derivative_type, analysis in alpha_data.items():
            f.write(f"\n{derivative_type.upper()} DERIVATIVE:\n")
            
            if 'cell_type_analysis' in analysis:
                for cell_label, cell_analysis in analysis['cell_type_analysis'].items():
                    f.write(f"  {cell_label} Cells:\n")
                    f.write(f"    Sensitivity: {cell_analysis.get('relative_sensitivity', 0):.4f}\n")
                    f.write(f"    Optimal alpha:   {cell_analysis.get('optimal_alpha', 'N/A'):.1f}\n")
                    f.write(f"    Monotonicity: {cell_analysis.get('monotonicity', 0):.4f}\n")
                    
                    if 'correlation' in cell_analysis:
                        corr = cell_analysis['correlation']
                        f.write(f"    Correlation: {corr.get('coefficient', 0):.4f} ")
                        f.write(f"(p={corr.get('p_value', 1):.4f})\n")
        
        f.write("\n")
    
    def _write_derivative_comparison(self, f, derivative_data):
        """Write derivative comparison section"""
        f.write("DERIVATIVE TYPE COMPARISON\n")
        f.write("-" * 30 + "\n")
        
        for alpha, comparison in derivative_data.items():
            f.write(f"\nAlpha = {alpha}:\n")
            
            if 'overall_comparison' in comparison and 'ranking' in comparison['overall_comparison']:
                ranking = comparison['overall_comparison']['ranking']
                f.write(f"  Overall ranking: {' > '.join(ranking)}\n")
            
            if 'cell_type_analysis' in comparison:
                for cell_label, analysis in comparison['cell_type_analysis'].items():
                    f.write(f"  {cell_label} best derivative: {analysis.get('best_derivative', 'N/A')}\n")
        
        f.write("\n")
    
    def _write_statistical_tests(self, f, test_results):
        """Write statistical test results"""
        f.write("STATISTICAL TEST RESULTS\n")
        f.write("-" * 30 + "\n")
        
        for metric, tests in test_results.items():
            f.write(f"\n{metric.upper()} Tests:\n")
            
            if 'pairwise_tests' in tests:
                f.write("  Pairwise comparisons:\n")
                for comparison, result in tests['pairwise_tests'].items():
                    if 'p_value' in result:
                        significance = "significant" if result.get('significant', False) else "not significant"
                        f.write(f"    {comparison}: p={result['p_value']:.4f} ({significance})\n")
            
            if 'anova' in tests:
                anova = tests['anova']
                if 'p_value' in anova:
                    significance = "significant" if anova.get('significant', False) else "not significant"
                    f.write(f"  ANOVA: F={anova.get('f_statistic', 0):.4f}, p={anova['p_value']:.4f} ({significance})\n")
        
        f.write("\n")
    
    def _write_key_findings(self, f, findings):
        """Write key findings section"""
        f.write("KEY FINDINGS\n")
        f.write("-" * 30 + "\n")
        
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f, recommendations):
        """Write recommendations section"""
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        for i, recommendation in enumerate(recommendations, 1):
            f.write(f"{i}. {recommendation}\n")
        
        f.write("\n")
    
    def generate_csv_summary(self, analysis_results, filename='results_summary.csv'):
        """
        Generate CSV summary for easy analysis in other tools
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Collect all results in tabular format
        rows = []
        headers = ['Model', 'Derivative', 'Alpha', 'Cell_Type', 'Metric', 'Value']
        rows.append(headers)
        
        # Process model performance data
        if 'model_performance' in analysis_results:
            perf_data = analysis_results['model_performance']
            if 'metrics' in perf_data:
                for model_name, metrics in perf_data['metrics'].items():
                    if 'per_cell_type' in metrics:
                        per_cell = metrics['per_cell_type']
                        for metric_name in ['rmse', 'mae', 'r2']:
                            if metric_name in per_cell:
                                values = per_cell[metric_name]
                                for i, cell_label in enumerate(self.cell_labels):
                                    if i < len(values):
                                        rows.append([model_name, 'N/A', 'N/A', cell_label, 
                                                   metric_name, f"{values[i]:.6f}"])
        
        # Process alpha analysis data
        if 'alpha_analysis' in analysis_results:
            for derivative_type, analysis in analysis_results['alpha_analysis'].items():
                if 'cell_type_analysis' in analysis:
                    for cell_label, cell_analysis in analysis['cell_type_analysis'].items():
                        if 'final_values' in cell_analysis and 'alpha_values' in analysis:
                            alpha_values = analysis.get('alpha_values', [])
                            final_values = cell_analysis['final_values']
                            for alpha, value in zip(alpha_values, final_values):
                                rows.append(['Fractional', derivative_type, f"{alpha:.1f}", 
                                           cell_label, 'final_value', f"{value:.6f}"])
        
        # Write CSV file
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(','.join(map(str, row)) + '\n')
        
        print(f"CSV summary saved to: {filepath}")
        return filepath
    
    def generate_latex_table(self, performance_data, filename='performance_table.tex'):
        """
        Generate LaTeX table for publications
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Performance Comparison}\n")
            f.write("\\label{tab:performance}\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Model & Tumor & Immune & Memory & Stromal \\\\\n")
            f.write("\\hline\n")
            
            if 'metrics' in performance_data:
                for model_name, metrics in performance_data['metrics'].items():
                    if 'per_cell_type' in metrics and 'rmse' in metrics['per_cell_type']:
                        rmse_values = metrics['per_cell_type']['rmse']
                        f.write(f"{model_name}")
                        for value in rmse_values:
                            f.write(f" & {value:.3f}")
                        f.write(" \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX table saved to: {filepath}")
        return filepath
    
    def generate_json_summary(self, analysis_results, filename='analysis_results.json'):
        """
        Generate JSON summary for programmatic access
        """
        import json
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_numpy_to_json(analysis_results)
        
        # Add metadata
        json_results['metadata'] = {
            'generation_time': datetime.now().isoformat(),
            'configuration': {
                'alpha_range': [float(self.config.alpha_values[0]), float(self.config.alpha_values[-1])],
                'alpha_count': len(self.config.alpha_values),
                'derivative_types': self.config.fractional_derivative_types,
                'cell_labels': self.cell_labels,
                'time_points': self.config.time_params['points']
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"JSON summary saved to: {filepath}")
        return filepath
    
    def _convert_numpy_to_json(self, obj):
        """
        Recursively convert numpy arrays to lists for JSON serialization
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        else:
            return obj
    
    def generate_html_report(self, analysis_results, filename='analysis_report.html'):
        """
        Generate HTML report with embedded plots
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._get_html_header())
            f.write(self._get_html_body(analysis_results))
            f.write(self._get_html_footer())
        
        print(f"HTML report saved to: {filepath}")
        return filepath
    
    def _get_html_header(self):
        """Get HTML header"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Dynamics Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
        .metric-table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        .metric-table th { background-color: #f2f2f2; font-weight: bold; }
        .finding { background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .recommendation { background-color: #f0f8e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .plot-container { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
"""
    
    def _get_html_body(self, analysis_results):
        """Get HTML body content"""
        html = f"""
    <div class="header">
        <h1>Cancer Dynamics Research - Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Configuration:</strong> {len(self.config.alpha_values)} alpha values, 
           {len(self.config.fractional_derivative_types)} derivative types, 
           {len(self.config.initial_conditions)} initial conditions</p>
    </div>
"""
        
        # Model Performance Section
        if 'model_performance' in analysis_results:
            html += self._get_performance_html(analysis_results['model_performance'])
        
        # Alpha Analysis Section
        if 'alpha_analysis' in analysis_results:
            html += self._get_alpha_html(analysis_results['alpha_analysis'])
        
        # Key Findings
        if 'key_findings' in analysis_results:
            html += """
    <div class="section">
        <h2>Key Findings</h2>
"""
            for finding in analysis_results['key_findings']:
                html += f'        <div class="finding">• {finding}</div>\n'
            html += "    </div>\n"
        
        # Recommendations
        if 'recommendations' in analysis_results:
            html += """
    <div class="section">
        <h2>Recommendations</h2>
"""
            for recommendation in analysis_results['recommendations']:
                html += f'        <div class="recommendation">• {recommendation}</div>\n'
            html += "    </div>\n"
        
        return html
    
    def _get_performance_html(self, performance_data):
        """Get HTML for performance section"""
        html = """
    <div class="section">
        <h2>Model Performance Analysis</h2>
        <table class="metric-table">
            <tr>
                <th>Model</th>
                <th>Overall RMSE</th>
                <th>Overall MAE</th>
                <th>Overall R²</th>
            </tr>
"""
        
        if 'metrics' in performance_data:
            for model_name, metrics in performance_data['metrics'].items():
                if 'overall' in metrics:
                    overall = metrics['overall']
                    html += f"""
            <tr>
                <td>{model_name}</td>
                <td>{overall.get('rmse', 'N/A'):.4f}</td>
                <td>{overall.get('mae', 'N/A'):.4f}</td>
                <td>{overall.get('r2', 'N/A'):.4f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
"""
        return html
    
    def _get_alpha_html(self, alpha_data):
        """Get HTML for alpha analysis section"""
        html = """
    <div class="section">
        <h2>Alpha Sensitivity Analysis</h2>
"""
        
        for derivative_type, analysis in alpha_data.items():
            html += f"        <h3>{derivative_type.capitalize()} Derivative</h3>\n"
            
            if 'cell_type_analysis' in analysis:
                html += """
        <table class="metric-table">
            <tr>
                <th>Cell Type</th>
                <th>Sensitivity</th>
                <th>Optimal alpha</th>
                <th>Monotonicity</th>
            </tr>
"""
                for cell_label, cell_analysis in analysis['cell_type_analysis'].items():
                    html += f"""
            <tr>
                <td>{cell_label}</td>
                <td>{cell_analysis.get('relative_sensitivity', 0):.4f}</td>
                <td>{cell_analysis.get('optimal_alpha', 'N/A'):.1f}</td>
                <td>{cell_analysis.get('monotonicity', 0):.4f}</td>
            </tr>
"""
                html += "        </table>\n"
        
        html += "    </div>\n"
        return html
    
    def _get_html_footer(self):
        """Get HTML footer"""
        return """
</body>
</html>
"""
    
    def generate_comprehensive_report(self, analysis_results):
        """
        Generate all report formats
        """
        print("Generating comprehensive analysis report...")
        
        files_generated = []
        
        # Text summary
        files_generated.append(self.generate_text_summary(analysis_results))
        
        # CSV summary
        files_generated.append(self.generate_csv_summary(analysis_results))
        
        # JSON summary
        files_generated.append(self.generate_json_summary(analysis_results))
        
        # HTML report
        files_generated.append(self.generate_html_report(analysis_results))
        
        # LaTeX table (if performance data exists)
        if 'model_performance' in analysis_results:
            files_generated.append(self.generate_latex_table(analysis_results['model_performance']))
        
        print(f"Comprehensive report generated. Files created:")
        for file_path in files_generated:
            print(f"  - {file_path}")
        
        return files_generated
    
    def create_executive_summary(self, analysis_results):
        """
        Create executive summary for quick overview
        """
        summary = {
            'experiment_overview': {
                'models_tested': len(analysis_results.get('model_performance', {}).get('metrics', {})),
                'alpha_values': len(self.config.alpha_values),
                'derivative_types': len(self.config.fractional_derivative_types),
                'analysis_scope': 'Comprehensive cancer dynamics modeling with fractional derivatives'
            },
            'key_metrics': {},
            'main_conclusions': [],
            'next_steps': []
        }
        
        # Extract key metrics
        if 'model_performance' in analysis_results:
            perf_data = analysis_results['model_performance']['metrics']
            best_model = None
            best_rmse = float('inf')
            
            for model_name, metrics in perf_data.items():
                if 'overall' in metrics and 'rmse' in metrics['overall']:
                    rmse = metrics['overall']['rmse']
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_name
            
            summary['key_metrics']['best_model'] = best_model
            summary['key_metrics']['best_rmse'] = best_rmse
        
        # Main conclusions
        summary['main_conclusions'] = [
            "Fractional derivatives provide enhanced modeling capabilities",
            "Alpha parameter significantly affects tumor dynamics",
            "Model selection depends on specific clinical objectives",
            "Further validation with experimental data recommended"
        ]
        
        # Next steps
        summary['next_steps'] = [
            "Conduct sensitivity analysis on additional parameters",
            "Validate predictions with clinical data",
            "Explore patient-specific parameter estimation",
            "Investigate treatment optimization strategies"
        ]
        
        return summary


def create_summary_generator(output_dir='results'):
    """Factory function to create SummaryGenerator"""
    return SummaryGenerator(output_dir)
