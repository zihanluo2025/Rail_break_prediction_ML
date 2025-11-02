"""
Enhanced Railway EDA Tool for Multi-Table Data
Performs comprehensive exploratory data analysis on real railway sensor data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from multi_table_loader import MultiTableRailwayDataLoader

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class EnhancedRailwayEDA:
    """
    Enhanced EDA tool for multi-table railway data analysis
    """
    
    def __init__(self, data_dir: str, output_dir: str = "eda_output"):
        """
        Initialize the enhanced EDA tool
        
        Args:
            data_dir (str): Directory containing CSV data files
            output_dir (str): Directory to save analysis outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.loader = MultiTableRailwayDataLoader(data_dir)
        self.data = None
        self.tables = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analysis results storage
        self.analysis_results = {}
        
    def load_and_integrate_data(self) -> pd.DataFrame:
        """
        Load all tables and integrate them for analysis
        
        Returns:
            pd.DataFrame: Integrated dataset
        """
        print("Loading and integrating railway data tables...")
        
        # Load all tables
        self.tables = self.loader.load_all_tables()
        
        # Integrate data
        self.data = self.loader.integrate_data()
        
        print(f"Data loaded successfully!")
        print(f"Integrated dataset: {len(self.data)} rows, {len(self.data.columns)} columns")
        
        return self.data
    
    def display_data_overview(self) -> None:
        """Display comprehensive data overview"""
        print("\n" + "="*80)
        print("RAILWAY DATA OVERVIEW")
        print("="*80)
        
        overview = self.loader.get_data_overview()
        print(overview)
        
        # Save overview to file
        with open(os.path.join(self.output_dir, "data_overview.txt"), "w", encoding="utf-8") as f:
            f.write(overview)
        
        print(f"\nData overview saved to: {self.output_dir}/data_overview.txt")
    
    def analyze_data_quality(self) -> None:
        """Analyze data quality across all tables with detailed insights"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("="*80)
        
        quality_report = self.loader.validate_data_quality()
        
        # Create basic quality summary
        quality_summary = []
        for table_name, metrics in quality_report.items():
            quality_summary.append({
                'Table': table_name,
                'Rows': metrics['total_rows'],
                'Missing (%)': metrics['missing_percentage'],
                'Duplicates (%)': metrics['duplicate_percentage'],
                'Memory (MB)': metrics['memory_usage_mb']
            })
        
        quality_df = pd.DataFrame(quality_summary)
        print("Basic Quality Metrics:")
        print(quality_df.to_string(index=False))
        
        # Save basic quality report
        quality_df.to_csv(os.path.join(self.output_dir, "data_quality_report.csv"), index=False)
        print(f"\nBasic quality report saved to: {self.output_dir}/data_quality_report.csv")
        
        # Perform detailed missing data analysis
        self._analyze_missing_data_patterns()
        
        # Perform duplicate data analysis
        self._analyze_duplicate_patterns()
        
        # Assess impact on prediction tasks
        self._assess_prediction_impact()
        
        # Store results
        self.analysis_results['data_quality'] = quality_report
    
    def _analyze_missing_data_patterns(self) -> None:
        """Analyze patterns and reasons for missing data"""
        print("\n" + "="*80)
        print("MISSING DATA PATTERN ANALYSIS")
        print("="*80)
        
        missing_analysis = {}
        
        for table_name, table_data in self.tables.items():
            print(f"\n📊 Analyzing missing data in '{table_name}' table...")
            
            # Get missing data statistics
            missing_counts = table_data.isnull().sum()
            missing_percentages = (missing_counts / len(table_data)) * 100
            
            # Filter columns with missing data
            missing_columns = missing_counts[missing_counts > 0]
            missing_percentages_filtered = missing_percentages[missing_counts > 0]
            
            if len(missing_columns) == 0:
                print(f"✅ No missing data found in {table_name}")
                missing_analysis[table_name] = {
                    'missing_columns': [],
                    'missing_patterns': 'No missing data',
                    'recommendations': 'No action needed'
                }
                continue
            
            # Sort by missing percentage
            missing_summary = pd.DataFrame({
                'Column': missing_columns.index,
                'Missing_Count': missing_columns.values,
                'Missing_Percentage': missing_percentages_filtered.values
            }).sort_values('Missing_Percentage', ascending=False)
            
            print(f"🔍 Found {len(missing_columns)} columns with missing data:")
            print(missing_summary.to_string(index=False))
            
            # Analyze missing data patterns
            missing_patterns = self._identify_missing_patterns(table_data, missing_columns.index)
            
            # Generate recommendations
            recommendations = self._generate_missing_data_recommendations(
                table_name, missing_summary, missing_patterns
            )
            
            missing_analysis[table_name] = {
                'missing_columns': missing_summary.to_dict('records'),
                'missing_patterns': missing_patterns,
                'recommendations': recommendations
            }
            
            # Save detailed missing data report for this table
            missing_summary.to_csv(
                os.path.join(self.output_dir, f"missing_data_{table_name}.csv"), 
                index=False
            )
            print(f"📁 Detailed missing data report saved: missing_data_{table_name}.csv")
        
        # Save comprehensive missing data analysis
        self._save_missing_data_analysis(missing_analysis)
        
        # Store results
        self.analysis_results['missing_data_analysis'] = missing_analysis
    
    def _identify_missing_patterns(self, table_data: pd.DataFrame, missing_columns: List[str]) -> Dict:
        """Identify patterns in missing data"""
        patterns = {}
        
        for col in missing_columns:
            col_patterns = {}
            
            # Check if missing data is random or systematic
            missing_indices = table_data[col].isnull()
            
            # Pattern 1: Check if missing data is clustered
            missing_runs = self._find_missing_runs(missing_indices)
            col_patterns['clustering'] = {
                'max_consecutive_missing': max(missing_runs) if missing_runs else 0,
                'total_runs': len(missing_runs),
                'avg_run_length': np.mean(missing_runs) if missing_runs else 0
            }
            
            # Pattern 2: Check if missing data is related to other columns
            correlation_with_missing = {}
            for other_col in table_data.columns:
                if other_col != col and not table_data[other_col].isnull().all():
                    # Check if missing in col is correlated with missing in other_col
                    both_missing = (table_data[col].isnull() & table_data[other_col].isnull()).sum()
                    if both_missing > 0:
                        correlation = both_missing / missing_indices.sum()
                        if correlation > 0.3:  # Threshold for meaningful correlation
                            correlation_with_missing[other_col] = correlation
            
            col_patterns['correlated_missing'] = correlation_with_missing
            
            # Pattern 3: Check if missing data is time-based (if timestamp exists)
            if 'RecordingDateTime' in table_data.columns:
                time_pattern = self._analyze_time_based_missing(table_data, col)
                col_patterns['time_based'] = time_pattern
            
            patterns[col] = col_patterns
        
        return patterns
    
    def _find_missing_runs(self, missing_series: pd.Series) -> List[int]:
        """Find consecutive runs of missing data"""
        runs = []
        current_run = 0
        
        for is_missing in missing_series:
            if is_missing:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        return runs
    
    def _analyze_time_based_missing(self, table_data: pd.DataFrame, column: str) -> Dict:
        """Analyze if missing data has time-based patterns"""
        if 'RecordingDateTime' not in table_data.columns:
            return {}
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(table_data['RecordingDateTime']):
            try:
                table_data['RecordingDateTime'] = pd.to_datetime(table_data['RecordingDateTime'])
            except:
                return {}
        
        missing_indices = table_data[column].isnull()
        missing_times = table_data.loc[missing_indices, 'RecordingDateTime']
        
        if len(missing_times) == 0:
            return {}
        
        # Check for time gaps
        missing_times_sorted = missing_times.sort_values()
        time_gaps = missing_times_sorted.diff().dropna()
        
        return {
            'missing_time_range': {
                'start': str(missing_times.min()),
                'end': str(missing_times.max()),
                'total_missing_periods': len(missing_times)
            },
            'time_gaps': {
                'min_gap': str(time_gaps.min()) if len(time_gaps) > 0 else 'N/A',
                'max_gap': str(time_gaps.max()) if len(time_gaps) > 0 else 'N/A',
                'avg_gap': str(time_gaps.mean()) if len(time_gaps) > 0 else 'N/A'
            }
        }
    
    def _generate_missing_data_recommendations(self, table_name: str, missing_summary: pd.DataFrame, 
                                             missing_patterns: Dict) -> List[str]:
        """Generate actionable recommendations for missing data"""
        recommendations = []
        
        for _, row in missing_summary.iterrows():
            col = row['Column']
            missing_pct = row['Missing_Percentage']
            patterns = missing_patterns.get(col, {})
            
            col_recommendations = []
            
            # Recommendation based on missing percentage
            if missing_pct > 50:
                col_recommendations.append("⚠️ HIGH IMPACT: Consider removing this column (>50% missing)")
            elif missing_pct > 20:
                col_recommendations.append("🔶 MODERATE IMPACT: Consider imputation strategies (20-50% missing)")
            elif missing_pct > 5:
                col_recommendations.append("🔸 LOW IMPACT: Minor imputation may be needed (5-20% missing)")
            else:
                col_recommendations.append("✅ MINIMAL IMPACT: Current missing rate is acceptable (<5%)")
            
            # Recommendation based on clustering patterns
            clustering = patterns.get('clustering', {})
            max_consecutive = clustering.get('max_consecutive_missing', 0)
            if max_consecutive > 100:
                col_recommendations.append("🚨 SYSTEMATIC MISSING: Large consecutive gaps suggest systematic issues")
            elif max_consecutive > 10:
                col_recommendations.append("⚠️ CLUSTERED MISSING: Consider if missing periods represent valid data")
            
            # Recommendation based on correlations
            correlated_missing = patterns.get('correlated_missing', {})
            if correlated_missing:
                high_corr_cols = [col for col, corr in correlated_missing.items() if corr > 0.7]
                if high_corr_cols:
                    col_recommendations.append(f"🔗 CORRELATED MISSING: Missing data correlates with {', '.join(high_corr_cols[:3])}")
            
            # Specific recommendations for different table types
            if table_name == 'wagondata':
                if 'Speed' in col:
                    col_recommendations.append("🚂 SPEED SENSOR: Critical for railway analysis - prioritize imputation")
                elif 'Twist' in col or 'Bounce' in col:
                    col_recommendations.append("🔧 TRACK SENSOR: Important for track health - consider interpolation")
            
            recommendations.extend([f"{col}: {rec}" for rec in col_recommendations])
        
        return recommendations
    
    def _analyze_duplicate_patterns(self) -> None:
        """Analyze duplicate data patterns"""
        print("\n" + "="*80)
        print("DUPLICATE DATA ANALYSIS")
        print("="*80)
        
        duplicate_analysis = {}
        
        for table_name, table_data in self.tables.items():
            print(f"\n🔍 Analyzing duplicates in '{table_name}' table...")
            
            # Find exact duplicates
            exact_duplicates = table_data.duplicated()
            exact_duplicate_count = exact_duplicates.sum()
            exact_duplicate_pct = (exact_duplicate_count / len(table_data)) * 100
            
            print(f"Exact duplicates: {exact_duplicate_count} ({exact_duplicate_pct:.2f}%)")
            
            # Find potential duplicates (based on key columns)
            potential_duplicates = self._find_potential_duplicates(table_data, table_name)
            
            duplicate_analysis[table_name] = {
                'exact_duplicates': {
                    'count': exact_duplicate_count,
                    'percentage': exact_duplicate_pct
                },
                'potential_duplicates': potential_duplicates,
                'recommendations': self._generate_duplicate_recommendations(
                    exact_duplicate_count, exact_duplicate_pct, potential_duplicates
                )
            }
            
            # Show sample duplicates if they exist
            if exact_duplicate_count > 0:
                print("Sample duplicate rows:")
                sample_duplicates = table_data[exact_duplicates].head(3)
                print(sample_duplicates.to_string())
        
        # Save duplicate analysis
        self._save_duplicate_analysis(duplicate_analysis)
        
        # Store results
        self.analysis_results['duplicate_analysis'] = duplicate_analysis
    
    def _find_potential_duplicates(self, table_data: pd.DataFrame, table_name: str) -> Dict:
        """Find potential duplicates based on business logic"""
        potential_dups = {}
        
        if table_name == 'wagondata':
            # Check for potential duplicates based on key fields
            key_columns = ['BaseCode', 'KMLocation', 'RecordingDateTime']
            available_keys = [col for col in key_columns if col in table_data.columns]
            
            if len(available_keys) >= 2:
                # Group by key columns and find groups with multiple records
                grouped = table_data.groupby(available_keys).size()
                duplicate_groups = grouped[grouped > 1]
                
                if len(duplicate_groups) > 0:
                    potential_dups['key_based'] = {
                        'duplicate_groups': len(duplicate_groups),
                        'total_duplicate_records': duplicate_groups.sum(),
                        'key_columns': available_keys
                    }
        
        elif table_name == 'trainingcontext':
            # Check for duplicate training samples
            if 'target' in table_data.columns:
                target_counts = table_data['target'].value_counts()
                potential_dups['target_distribution'] = target_counts.to_dict()
        
        return potential_dups
    
    def _generate_duplicate_recommendations(self, exact_count: int, exact_pct: float, 
                                          potential_dups: Dict) -> List[str]:
        """Generate recommendations for handling duplicates"""
        recommendations = []
        
        if exact_count > 0:
            if exact_pct > 10:
                recommendations.append("🚨 HIGH DUPLICATE RATE: Remove exact duplicates before analysis")
            elif exact_pct > 5:
                recommendations.append("⚠️ MODERATE DUPLICATES: Review and remove exact duplicates")
            else:
                recommendations.append("🔸 LOW DUPLICATES: Minor cleanup recommended")
        
        if 'key_based' in potential_dups:
            key_dups = potential_dups['key_based']
            if key_dups['total_duplicate_records'] > 100:
                recommendations.append("🔍 KEY-BASED DUPLICATES: Investigate business logic for duplicate keys")
        
        if not recommendations:
            recommendations.append("✅ No duplicate issues detected")
        
        return recommendations
    
    def _assess_prediction_impact(self) -> None:
        """Assess the impact of data quality issues on prediction tasks"""
        print("\n" + "="*80)
        print("PREDICTION TASK IMPACT ASSESSMENT")
        print("="*80)
        
        impact_analysis = {}
        
        # Analyze impact on target variable
        if 'target' in self.data.columns:
            target_quality = self._analyze_target_variable_quality()
            impact_analysis['target_quality'] = target_quality
        
        # Analyze impact on feature variables
        feature_quality = self._analyze_feature_quality_impact()
        impact_analysis['feature_quality'] = feature_quality
        
        # Generate overall recommendations
        overall_recommendations = self._generate_overall_quality_recommendations(impact_analysis)
        impact_analysis['overall_recommendations'] = overall_recommendations
        
        # Save impact assessment
        self._save_impact_assessment(impact_analysis)
        
        # Store results
        self.analysis_results['prediction_impact'] = impact_analysis
    
    def _analyze_target_variable_quality(self) -> Dict:
        """Analyze quality of target variable for prediction tasks"""
        target_col = 'target'
        target_data = self.data[target_col]
        
        missing_count = target_data.isnull().sum()
        missing_pct = (missing_count / len(target_data)) * 100
        
        # Check class balance
        value_counts = target_data.value_counts()
        class_balance = {
            'total_samples': len(target_data),
            'missing_samples': missing_count,
            'available_samples': len(target_data) - missing_count,
            'class_distribution': value_counts.to_dict()
        }
        
        # Assess impact
        if missing_pct > 20:
            impact_level = "🚨 CRITICAL: High missing rate will severely impact model training"
        elif missing_pct > 10:
            impact_level = "⚠️ HIGH: Missing data will significantly reduce training samples"
        elif missing_pct > 5:
            impact_level = "🔶 MODERATE: Some impact on model performance expected"
        else:
            impact_level = "✅ LOW: Minimal impact on prediction tasks"
        
        return {
            'missing_analysis': {
                'count': missing_count,
                'percentage': missing_pct,
                'impact_level': impact_level
            },
            'class_balance': class_balance,
            'recommendations': self._generate_target_quality_recommendations(missing_pct, value_counts)
        }
    
    def _analyze_feature_quality_impact(self) -> Dict:
        """Analyze impact of feature quality on prediction tasks"""
        sensor_cols = self.loader.get_time_series_columns()
        feature_impact = {}
        
        for col in sensor_cols[:10]:  # Analyze top 10 sensors
            if col in self.data.columns:
                col_data = self.data[col]
                missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
                
                # Assess feature importance (using variance as proxy)
                variance = col_data.var()
                
                if missing_pct > 30:
                    impact = "🚨 CRITICAL: High missing rate makes feature unreliable"
                elif missing_pct > 15:
                    impact = "⚠️ HIGH: Significant data loss affects feature quality"
                elif missing_pct > 5:
                    impact = "🔶 MODERATE: Some impact on feature reliability"
                else:
                    impact = "✅ LOW: Minimal impact on feature quality"
                
                feature_impact[col] = {
                    'missing_percentage': missing_pct,
                    'variance': variance,
                    'impact_level': impact,
                    'recommendation': self._generate_feature_recommendation(missing_pct, variance)
                }
        
        return feature_impact
    
    def _generate_target_quality_recommendations(self, missing_pct: float, value_counts: pd.Series) -> List[str]:
        """Generate recommendations for target variable quality"""
        recommendations = []
        
        if missing_pct > 20:
            recommendations.append("🚨 Remove samples with missing targets - insufficient for training")
        elif missing_pct > 10:
            recommendations.append("⚠️ Consider imputation strategies or remove missing samples")
        elif missing_pct > 5:
            recommendations.append("🔶 Minor imputation may be needed")
        else:
            recommendations.append("✅ Target quality is acceptable for modeling")
        
        # Check class balance
        if len(value_counts) == 2:  # Binary classification
            class_ratio = value_counts.iloc[0] / value_counts.iloc[1]
            if class_ratio > 10 or class_ratio < 0.1:
                recommendations.append("⚖️ SEVERE CLASS IMBALANCE: Consider resampling techniques")
            elif class_ratio > 5 or class_ratio < 0.2:
                recommendations.append("⚠️ CLASS IMBALANCE: May need balanced sampling strategies")
        
        return recommendations
    
    def _generate_feature_recommendation(self, missing_pct: float, variance: float) -> str:
        """Generate recommendation for feature quality"""
        if missing_pct > 30:
            return "Remove feature - too much missing data"
        elif missing_pct > 15:
            return "Use imputation - significant missing data"
        elif missing_pct > 5:
            return "Minor imputation recommended"
        else:
            return "Feature quality acceptable"
    
    def _generate_overall_quality_recommendations(self, impact_analysis: Dict) -> List[str]:
        """Generate overall data quality recommendations"""
        recommendations = []
        
        # Overall assessment
        target_quality = impact_analysis.get('target_quality', {})
        if target_quality:
            missing_pct = target_quality.get('missing_analysis', {}).get('percentage', 0)
            if missing_pct > 20:
                recommendations.append("🚨 OVERALL: Data quality issues will severely impact prediction tasks")
            elif missing_pct > 10:
                recommendations.append("⚠️ OVERALL: Data quality issues will significantly impact prediction tasks")
            elif missing_pct > 5:
                recommendations.append("🔶 OVERALL: Some data quality issues may affect prediction tasks")
            else:
                recommendations.append("✅ OVERALL: Data quality is acceptable for prediction tasks")
        
        # Specific actions
        recommendations.extend([
            "📊 ACTION: Review missing data patterns before model training",
            "🔍 ACTION: Investigate systematic missing data causes",
            "⚖️ ACTION: Assess class balance for classification tasks",
            "🧹 ACTION: Clean duplicates and outliers before modeling"
        ])
        
        return recommendations
    
    def _save_missing_data_analysis(self, missing_analysis: Dict) -> None:
        """Save comprehensive missing data analysis"""
        # Save summary
        summary_data = []
        for table_name, analysis in missing_analysis.items():
            for col_info in analysis['missing_columns']:
                summary_data.append({
                    'Table': table_name,
                    'Column': col_info['Column'],
                    'Missing_Count': col_info['Missing_Count'],
                    'Missing_Percentage': col_info['Missing_Percentage']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.output_dir, "missing_data_summary.csv"), index=False)
            print(f"📁 Missing data summary saved: missing_data_summary.csv")
        
        # Save detailed analysis
        with open(os.path.join(self.output_dir, "missing_data_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("COMPREHENSIVE MISSING DATA ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for table_name, analysis in missing_analysis.items():
                f.write(f"TABLE: {table_name}\n")
                f.write("-" * 30 + "\n")
                
                if analysis['missing_columns']:
                    f.write("Missing Columns:\n")
                    for col_info in analysis['missing_columns']:
                        f.write(f"  - {col_info['Column']}: {col_info['Missing_Count']} ({col_info['Missing_Percentage']:.2f}%)\n")
                    
                    f.write("\nPatterns:\n")
                    for col, patterns in analysis['missing_patterns'].items():
                        f.write(f"  {col}:\n")
                        if 'clustering' in patterns:
                            cluster = patterns['clustering']
                            f.write(f"    - Max consecutive missing: {cluster['max_consecutive_missing']}\n")
                            f.write(f"    - Total runs: {cluster['total_runs']}\n")
                    
                    f.write("\nRecommendations:\n")
                    for rec in analysis['recommendations']:
                        f.write(f"  - {rec}\n")
                else:
                    f.write("✅ No missing data found\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"📁 Detailed missing data analysis saved: missing_data_analysis.txt")
    
    def _save_duplicate_analysis(self, duplicate_analysis: Dict) -> None:
        """Save duplicate data analysis"""
        summary_data = []
        for table_name, analysis in duplicate_analysis.items():
            summary_data.append({
                'Table': table_name,
                'Exact_Duplicates': analysis['exact_duplicates']['count'],
                'Exact_Duplicate_Percentage': analysis['exact_duplicates']['percentage'],
                'Recommendations': '; '.join(analysis['recommendations'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "duplicate_data_summary.csv"), index=False)
        print(f"📁 Duplicate data summary saved: duplicate_data_summary.csv")
    
    def _save_impact_assessment(self, impact_analysis: Dict) -> None:
        """Save prediction impact assessment"""
        # Save target quality summary
        if 'target_quality' in impact_analysis:
            target_quality = impact_analysis['target_quality']
            target_summary = pd.DataFrame([{
                'Metric': 'Target Missing Rate',
                'Value': f"{target_quality['missing_analysis']['percentage']:.2f}%",
                'Impact': target_quality['missing_analysis']['impact_level']
            }])
            target_summary.to_csv(os.path.join(self.output_dir, "target_quality_summary.csv"), index=False)
            print(f"📁 Target quality summary saved: target_quality_summary.csv")
        
        # Save feature quality summary
        if 'feature_quality' in impact_analysis:
            feature_summary = []
            for col, quality in impact_analysis['feature_quality'].items():
                feature_summary.append({
                    'Feature': col,
                    'Missing_Percentage': quality['missing_percentage'],
                    'Variance': quality['variance'],
                    'Impact_Level': quality['impact_level'],
                    'Recommendation': quality['recommendation']
                })
            
            feature_df = pd.DataFrame(feature_summary)
            feature_df.to_csv(os.path.join(self.output_dir, "feature_quality_summary.csv"), index=False)
            print(f"📁 Feature quality summary saved: feature_quality_summary.csv")
        
        # Save overall recommendations
        if 'overall_recommendations' in impact_analysis:
            with open(os.path.join(self.output_dir, "overall_quality_recommendations.txt"), "w", encoding="utf-8") as f:
                f.write("OVERALL DATA QUALITY RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                
                for rec in impact_analysis['overall_recommendations']:
                    f.write(f"{rec}\n")
            
            print(f"📁 Overall quality recommendations saved: overall_quality_recommendations.txt")
    
    def analyze_sensor_distributions(self, top_n: int = 10) -> None:
        """Analyze distributions of sensor measurements"""
        print("\n" + "="*80)
        print("SENSOR DISTRIBUTION ANALYSIS")
        print("="*80)
        
        sensor_cols = self.loader.get_time_series_columns()
        print(f"Analyzing {len(sensor_cols)} sensor columns...")
        
        # Select top sensors by variance
        sensor_variance = {}
        for col in sensor_cols:
            if col in self.data.columns:
                sensor_variance[col] = self.data[col].var()
        
        # Sort by variance and take top N
        top_sensors = sorted(sensor_variance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_sensor_names = [col for col, var in top_sensors]
        
        print(f"Top {top_n} sensors by variance: {', '.join(top_sensor_names)}")
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for i, sensor in enumerate(top_sensor_names[:10]):
            if i < 10:
                ax = axes[i]
                self.data[sensor].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'{sensor}\nVariance: {sensor_variance[sensor]:.4f}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensor_distributions.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save sensor statistics
        sensor_stats = []
        for sensor in top_sensor_names:
            stats = self.data[sensor].describe()
            sensor_stats.append({
                'Sensor': sensor,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Variance': sensor_variance[sensor]
            })
        
        sensor_stats_df = pd.DataFrame(sensor_stats)
        sensor_stats_df.to_csv(os.path.join(self.output_dir, "sensor_statistics.csv"), index=False)
        print(f"Sensor statistics saved to: {self.output_dir}/sensor_statistics.csv")
        
        # Store results
        self.analysis_results['sensor_distributions'] = {
            'top_sensors': top_sensor_names,
            'sensor_variance': sensor_variance,
            'sensor_statistics': sensor_stats_df
        }
    
    def analyze_rail_break_patterns(self) -> None:
        """Analyze patterns related to rail breaks"""
        print("\n" + "="*80)
        print("RAIL BREAK PATTERN ANALYSIS")
        print("="*80)
        
        if 'target' not in self.data.columns:
            print("No target variable found. Skipping rail break analysis.")
            return
        
        # Basic statistics
        target_counts = self.data['target'].value_counts()
        print(f"Target distribution:\n{target_counts}")
        
        # Check if we have any break data
        has_break_data = 1 in target_counts.index and target_counts[1] > 0
        
        if not has_break_data:
            print("\n⚠️  No actual break events found in the dataset.")
            print("This dataset represents a 'healthy baseline' with no failures.")
            print("Creating alternative analysis for baseline establishment...")
            
            # Create baseline sensor analysis instead
            self._create_baseline_sensor_analysis()
            return
        
        # Original break analysis (when we have break data)
        sensor_cols = self.loader.get_time_series_columns()
        
        # Compare sensor distributions between break and no-break cases
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, sensor in enumerate(sensor_cols[:6]):
            if i < 6:
                ax = axes[i]
                
                # Create box plots
                break_data = self.data[self.data['target'] == 1][sensor]
                no_break_data = self.data[self.data['target'] == 0][sensor]
                
                data_to_plot = [no_break_data, break_data]
                labels = ['No Break', 'Break']
                
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(f'{sensor} by Break Status')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensor_patterns_by_break.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical significance test
        print("\nStatistical significance of sensor differences (Break vs No Break):")
        print("-" * 60)
        
        significance_results = []
        for sensor in sensor_cols[:10]:  # Test first 10 sensors
            break_data = self.data[self.data['target'] == 1][sensor].dropna()
            no_break_data = self.data[self.data['target'] == 0][sensor].dropna()
            
            if len(break_data) > 0 and len(no_break_data) > 0:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(break_data, no_break_data)
                
                significance_results.append({
                    'Sensor': sensor,
                    'Break_Mean': break_data.mean(),
                    'NoBreak_Mean': no_break_data.mean(),
                    'Difference': break_data.mean() - no_break_data.mean(),
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
        
        significance_df = pd.DataFrame(significance_results)
        print(significance_df.to_string(index=False))
        
        # Save results
        significance_df.to_csv(os.path.join(self.output_dir, "break_pattern_significance.csv"), index=False)
        print(f"\nSignificance results saved to: {self.output_dir}/break_pattern_significance.csv")
        
        # Store results
        self.analysis_results['rail_break_patterns'] = {
            'target_distribution': target_counts.to_dict(),
            'significance_results': significance_df
        }
        
        # Enhanced statistical significance testing with fracture vs normal interval comparison
        print("\nPerforming enhanced statistical significance testing...")
        enhanced_significance_results = []
        
        for sensor in sensor_cols[:10]:  # Test first 10 sensors
            break_data = self.data[self.data['target'] == 1][sensor].dropna()
            no_break_data = self.data[self.data['target'] == 0][sensor].dropna()
            
            if len(break_data) > 0 and len(no_break_data) > 0:
                from scipy import stats
                
                # T-test for mean difference
                t_stat, p_value = stats.ttest_ind(break_data, no_break_data, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(break_data) - 1) * break_data.var() + (len(no_break_data) - 1) * no_break_data.var()) / (len(break_data) + len(no_break_data) - 2))
                cohens_d = (break_data.mean() - no_break_data.mean()) / pooled_std
                
                # Variance comparison (F-test)
                f_stat, f_p_value = stats.f_oneway(break_data, no_break_data)
                
                # Levene's test for variance equality
                levene_stat, levene_p_value = stats.levene(break_data, no_break_data)
                
                enhanced_significance_results.append({
                    'Sensor': sensor,
                    'Break_Mean': break_data.mean(),
                    'NoBreak_Mean': no_break_data.mean(),
                    'Mean_Difference': break_data.mean() - no_break_data.mean(),
                    'Break_Std': break_data.std(),
                    'NoBreak_Std': no_break_data.std(),
                    'Break_Variance': break_data.var(),
                    'NoBreak_Variance': no_break_data.var(),
                    'Variance_Ratio': break_data.var() / no_break_data.var() if no_break_data.var() > 0 else float('inf'),
                    'T_Statistic': t_stat,
                    'P_Value_Mean': p_value,
                    'F_Statistic': f_stat,
                    'P_Value_Variance': f_p_value,
                    'Levene_Statistic': levene_stat,
                    'Levene_P_Value': levene_p_value,
                    'Mean_Significant': p_value < 0.05,
                    'Variance_Significant': f_p_value < 0.05,
                    'Effect_Size': cohens_d,
                    'Effect_Interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
                })
        
        # Save enhanced significance results
        enhanced_significance_df = pd.DataFrame(enhanced_significance_results)
        enhanced_significance_df.to_csv(os.path.join(self.output_dir, 'enhanced_break_pattern_significance.csv'), index=False)
        print(f"Enhanced significance results saved to: enhanced_break_pattern_significance.csv")
        
        # Display key findings
        if len(enhanced_significance_results) > 0:
            significant_mean = [r for r in enhanced_significance_results if r['Mean_Significant']]
            significant_variance = [r for r in enhanced_significance_results if r['Variance_Significant']]
            print(f"\nSignificant mean differences found in {len(significant_mean)} sensors:")
            for result in significant_mean[:5]:  # Show top 5
                print(f"- {result['Sensor']}: p={result['P_Value_Mean']:.4f}, Effect Size={result['Effect_Size']:.3f} ({result['Effect_Interpretation']})")
            
            print(f"\nSignificant variance differences found in {len(significant_variance)} sensors:")
            for result in significant_variance[:5]:  # Show top 5
                print(f"- {result['Sensor']}: p={result['P_Value_Mean']:.4f}, Variance Ratio={result['Variance_Ratio']:.3f}")
        
        # Update stored results
        self.analysis_results['rail_break_patterns']['enhanced_significance'] = enhanced_significance_results
    
    def _create_baseline_sensor_analysis(self) -> None:
        """Create baseline sensor analysis when no break data is available"""
        print("\nCreating Baseline Sensor Analysis...")
        
        # Get top sensors by variance
        sensor_cols = self.loader.get_time_series_columns()
        sensor_variance = {}
        for col in sensor_cols[:6]:  # Top 6 sensors
            if col in self.data.columns:
                sensor_variance[col] = self.data[col].var()
        
        # Sort by variance
        top_sensors = sorted(sensor_variance.items(), key=lambda x: x[1], reverse=True)
        
        # Create baseline distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (sensor, variance) in enumerate(top_sensors[:6]):
            if i < 6:
                ax = axes[i]
                
                # Create histogram of baseline sensor readings
                self.data[sensor].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black', color='green')
                ax.set_title(f'{sensor}\nBaseline Distribution\nVariance: {variance:.4f}')
                ax.set_xlabel('Sensor Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = self.data[sensor].mean()
                std_val = self.data[sensor].std()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.6, label=f'+1σ: {mean_val + std_val:.3f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.6, label=f'-1σ: {mean_val - std_val:.3f}')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensor_patterns_by_break.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Baseline sensor analysis created successfully!")
        print("This establishes the 'normal operating range' for each sensor.")
        print("Future readings can be compared against these baselines to detect anomalies.")
        
        # Store baseline analysis results
        self.analysis_results['baseline_sensor_analysis'] = {
            'top_sensors': [sensor for sensor, _ in top_sensors[:6]],
            'sensor_variance': sensor_variance,
            'sensor_statistics': self._get_detailed_sensor_statistics(top_sensors[:6])
        }
    
    def _get_detailed_sensor_statistics(self, top_sensors: List[Tuple[str, float]]) -> pd.DataFrame:
        """Get detailed statistics for top sensors including extreme values"""
        detailed_stats = []
        
        for sensor, variance in top_sensors:
            if sensor in self.data.columns:
                sensor_data = self.data[sensor].dropna()
                if len(sensor_data) > 0:
                    stats = sensor_data.describe()
                    
                    # Calculate additional statistics
                    q1 = stats['25%']
                    q3 = stats['75%']
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Find extreme values (outliers)
                    outliers = sensor_data[(sensor_data < lower_bound) | (sensor_data > upper_bound)]
                    
                    detailed_stats.append({
                        'Sensor': sensor,
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Q1': q1,
                        'Q3': q3,
                        'IQR': iqr,
                        'Variance': variance,
                        'Outlier_Count': len(outliers),
                        'Outlier_Percentage': len(outliers) / len(sensor_data) * 100,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound
                    })
        
        return pd.DataFrame(detailed_stats)
    
    def analyze_track_health_by_location(self) -> None:
        """Analyze track health patterns by location and BaseCode"""
        print("\n" + "="*80)
        print("TRACK HEALTH BY LOCATION ANALYSIS")
        print("="*80)
        
        if 'BaseCode' not in self.data.columns:
            print("No BaseCode found. Skipping location analysis.")
            return
        
        # Analyze by BaseCode
        basecode_stats = self.data.groupby('BaseCode').agg({
            'KMLocation': ['count', 'min', 'max'],
            'target': 'sum' if 'target' in self.data.columns else 'count'
        }).round(2)
        
        print("Track statistics by BaseCode:")
        print(basecode_stats)
        
        # Create location-based health map
        if 'KMLocation' in self.data.columns and 'target' in self.data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Break locations by BaseCode
            break_locations = self.data[self.data['target'] == 1]
            if len(break_locations) > 0:
                ax1.scatter(break_locations['KMLocation'], 
                           range(len(break_locations)), 
                           c=break_locations['BaseCode'].astype('category').cat.codes,
                           alpha=0.7, s=50)
                ax1.set_xlabel('KM Location')
                ax1.set_ylabel('Break Events')
                ax1.set_title('Rail Break Locations by Track Section')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Sensor readings by location
            if 'Twist14m' in self.data.columns:
                ax2.scatter(self.data['KMLocation'], self.data['Twist14m'], 
                           alpha=0.5, s=20)
                ax2.set_xlabel('KM Location')
                ax2.set_ylabel('Twist14m (sensor reading)')
                ax2.set_title('Twist Sensor Readings by Location')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "track_health_by_location.png"), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save location statistics
        basecode_stats.to_csv(os.path.join(self.output_dir, "track_health_by_location.csv"))
        print(f"\nLocation statistics saved to: {self.output_dir}/track_health_by_location.csv")
        
        # Store results
        self.analysis_results['track_health_by_location'] = {
            'basecode_statistics': basecode_stats
        }
    
    def analyze_sensor_correlations(self, method: str = 'pearson') -> None:
        """Analyze correlations between sensors"""
        print("\n" + "="*80)
        print("SENSOR CORRELATION ANALYSIS")
        print("="*80)
        
        sensor_cols = self.loader.get_time_series_columns()
        
        if len(sensor_cols) < 2:
            print("Not enough sensor columns for correlation analysis.")
            return
        
        # Select top sensors for correlation matrix
        top_sensors = sensor_cols[:15]  # Limit to 15 for readability
        
        # Calculate correlation matrix
        correlation_matrix = self.data[top_sensors].corr(method=method)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Sensor Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"sensor_correlations_{method}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated sensor pairs
        high_corr_pairs = []
        for i in range(len(top_sensors)):
            for j in range(i+1, len(top_sensors)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Sensor1': top_sensors[i],
                        'Sensor2': top_sensors[j],
                        'Correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"\nHighly correlated sensor pairs (|r| > 0.7):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df.to_string(index=False))
            
            # Save high correlations
            high_corr_df.to_csv(os.path.join(self.output_dir, "high_correlations.csv"), index=False)
            print(f"High correlations saved to: {self.output_dir}/high_correlations.csv")
        
        # Save full correlation matrix
        correlation_matrix.to_csv(os.path.join(self.output_dir, f"correlation_matrix_{method}.csv"))
        print(f"Full correlation matrix saved to: {self.output_dir}/correlation_matrix_{method}.csv")
        
        # Store results
        self.analysis_results['sensor_correlations'] = {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs if high_corr_pairs else []
        }
        
        # Analyze specific sensor correlations (BounceFrt vs Speed)
        self._analyze_specific_sensor_correlations()
    
    def _analyze_specific_sensor_correlations(self) -> None:
        """Analyze specific sensor correlations like BounceFrt vs Speed"""
        print("\n" + "="*80)
        print("SPECIFIC SENSOR CORRELATION ANALYSIS")
        print("="*80)
        
        # Define specific sensor pairs to analyze
        sensor_pairs = [
            ('BounceFrt', 'Speed'),
            ('BounceRr', 'Speed'),
            ('Twist14m', 'Speed'),
            ('BounceFrt', 'BounceRr')
        ]
        
        available_pairs = []
        for sensor1, sensor2 in sensor_pairs:
            if sensor1 in self.data.columns and sensor2 in self.data.columns:
                available_pairs.append((sensor1, sensor2))
        
        if not available_pairs:
            print("No suitable sensor pairs found for specific correlation analysis.")
            return
        
        # Create detailed correlation plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        correlation_details = []
        
        for i, (sensor1, sensor2) in enumerate(available_pairs[:4]):
            if i < 4:
                ax = axes[i]
                
                # Scatter plot
                ax.scatter(self.data[sensor1], self.data[sensor2], alpha=0.6, s=20)
                
                # Calculate correlation
                corr_data = self.data[[sensor1, sensor2]].dropna()
                if len(corr_data) > 1:
                    pearson_corr = corr_data[sensor1].corr(corr_data[sensor2], method='pearson')
                    spearman_corr = corr_data[sensor1].corr(corr_data[sensor2], method='spearman')
                    
                    # Add trend line
                    z = np.polyfit(corr_data[sensor1], corr_data[sensor2], 1)
                    p = np.poly1d(z)
                    ax.plot(corr_data[sensor1], p(corr_data[sensor1]), "r--", alpha=0.8)
                    
                    # Add correlation info to plot
                    ax.set_title(f'{sensor1} vs {sensor2}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
                    ax.set_xlabel(sensor1)
                    ax.set_ylabel(sensor2)
                    ax.grid(True, alpha=0.3)
                    
                    # Store correlation details
                    correlation_details.append({
                        'Sensor1': sensor1,
                        'Sensor2': sensor2,
                        'Pearson_Correlation': pearson_corr,
                        'Spearman_Correlation': spearman_corr,
                        'Data_Points': len(corr_data),
                        'Correlation_Strength': 'Strong' if abs(pearson_corr) > 0.7 else 'Moderate' if abs(pearson_corr) > 0.5 else 'Weak',
                        'Correlation_Direction': 'Positive' if pearson_corr > 0 else 'Negative'
                    })
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "specific_sensor_correlations.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed correlation results
        if correlation_details:
            corr_df = pd.DataFrame(correlation_details)
            corr_df.to_csv(os.path.join(self.output_dir, "specific_sensor_correlations.csv"), index=False)
            print(f"\nSpecific correlation results saved to: {self.output_dir}/specific_sensor_correlations.csv")
            
            # Display key findings
            print("\nKey Correlation Findings:")
            print("-" * 50)
            for _, row in corr_df.iterrows():
                print(f"{row['Sensor1']} vs {row['Sensor2']}:")
                print(f"  Pearson: {row['Pearson_Correlation']:.3f} ({row['Correlation_Strength']}, {row['Correlation_Direction']})")
                print(f"  Spearman: {row['Spearman_Correlation']:.3f}")
                print(f"  Data points: {row['Data_Points']}")
                print()
        
        # Store specific correlation results
        self.analysis_results['specific_sensor_correlations'] = {
            'correlation_details': correlation_details,
            'correlation_plot_file': 'specific_sensor_correlations.png'
        }
    
    def _analyze_high_correlation_sensor_groups(self) -> None:
        """Identify high-correlation sensor groups for dimensionality reduction recommendations"""
        print("\n" + "="*80)
        print("HIGH-CORRELATION SENSOR GROUPS ANALYSIS")
        print("="*80)
        
        # Get sensor columns (exclude non-sensor columns and ensure they are numeric)
        sensor_cols = []
        for col in self.data.columns:
            if col not in ['BaseCode', 'KMLocation', 'RecordingDateTime', 'target', 'rul', 'break_date', 'has_break_location', 'is_break_section', 'SectionBreakStartKM', 'SectionBreakFinishKM']:
                # Only include numeric columns
                if self.data[col].dtype in ['float64', 'int64']:
                    sensor_cols.append(col)
        
        print(f"Found {len(sensor_cols)} numeric sensor columns for correlation analysis")
        
        # Calculate correlation matrix for sensors only
        sensor_data = self.data[sensor_cols].dropna()
        if len(sensor_data) < 2:
            print("Insufficient sensor data for correlation analysis.")
            return
        
        correlation_matrix = sensor_data.corr()
        
        # Find high correlation pairs (|correlation| > 0.8)
        high_corr_pairs = []
        high_corr_groups = {}
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'Sensor1': col1,
                        'Sensor2': col2,
                        'Correlation': corr_value,
                        'Strength': 'Very Strong' if abs(corr_value) > 0.9 else 'Strong',
                        'Direction': 'Positive' if corr_value > 0 else 'Negative'
                    })
                    
                    # Group sensors by correlation strength
                    if abs(corr_value) > 0.9:
                        group_key = 'Very_Strong_Group'
                    else:
                        group_key = 'Strong_Group'
                    
                    if group_key not in high_corr_groups:
                        high_corr_groups[group_key] = []
                    
                    # Add both sensors to the group if not already present
                    if col1 not in high_corr_groups[group_key]:
                        high_corr_groups[group_key].append(col1)
                    if col2 not in high_corr_groups[group_key]:
                        high_corr_groups[group_key].append(col2)
        
        # Find sensor clusters using hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        from scipy.spatial.distance import pdist
        
        # Initialize variables
        linkage_matrix = None
        clusters = np.ones(len(correlation_matrix.columns))  # Default single cluster
        
        # Convert correlation to distance (1 - |correlation|)
        # Use pdist to create condensed distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Perform hierarchical clustering using condensed distance matrix
        try:
            # Handle NaN values in correlation matrix
            distance_matrix_clean = distance_matrix.fillna(0)
            condensed_distances = pdist(distance_matrix_clean)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Cut the dendrogram to get clusters
            clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')  # Threshold for cluster formation
        except Exception as e:
            print(f"Warning: Hierarchical clustering failed: {e}")
            print("Proceeding with correlation-based grouping only...")
        
        # Group sensors by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            sensor_name = correlation_matrix.columns[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(sensor_name)
        
        # Filter out single-sensor clusters
        cluster_groups = {k: v for k, v in cluster_groups.items() if len(v) > 1}
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: High correlation pairs
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            # Create correlation heatmap for high-correlation pairs
            high_corr_matrix = pd.DataFrame(index=range(len(high_corr_pairs)), columns=['Sensor1', 'Sensor2', 'Correlation'])
            for i, pair in enumerate(high_corr_pairs):
                high_corr_matrix.iloc[i] = [pair['Sensor1'], pair['Sensor2'], pair['Correlation']]
            
            # Create a small correlation matrix for visualization
            if len(high_corr_pairs) > 0:
                sensor_names = list(set([pair['Sensor1'] for pair in high_corr_pairs] + [pair['Sensor2'] for pair in high_corr_pairs]))
                small_corr_matrix = correlation_matrix.loc[sensor_names, sensor_names]
                
                im1 = ax1.imshow(small_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax1.set_xticks(range(len(sensor_names)))
                ax1.set_yticks(range(len(sensor_names)))
                ax1.set_xticklabels(sensor_names, rotation=45, ha='right')
                ax1.set_yticklabels(sensor_names)
                ax1.set_title('High-Correlation Sensor Pairs')
                
                # Add correlation values as text
                for i in range(len(sensor_names)):
                    for j in range(len(sensor_names)):
                        text = ax1.text(j, i, f'{small_corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
                
                plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Dendrogram
        if len(linkage_matrix) > 1:
            dendrogram(linkage_matrix, labels=correlation_matrix.columns, ax=ax2, leaf_rotation=90)
            ax2.set_title('Sensor Clustering Dendrogram')
            ax2.set_xlabel('Sensors')
            ax2.set_ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "high_correlation_sensor_groups.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate dimensionality reduction recommendations
        recommendations = []
        
        # Based on high correlation pairs
        if high_corr_pairs:
            recommendations.append("*** HIGH CORRELATION PAIRS DETECTED:")
            for pair in high_corr_pairs[:10]:  # Top 10
                recommendations.append(f"  - {pair['Sensor1']} <-> {pair['Sensor2']}: {pair['Correlation']:.3f} ({pair['Strength']}, {pair['Direction']})")
                recommendations.append(f"    -> Consider keeping only one sensor from this pair to reduce redundancy")
        
        # Based on sensor clusters
        if cluster_groups:
            recommendations.append("\n*** SENSOR CLUSTERS IDENTIFIED:")
            for cluster_id, sensors in cluster_groups.items():
                if len(sensors) > 2:  # Only show clusters with more than 2 sensors
                    recommendations.append(f"  - Cluster {cluster_id}: {', '.join(sensors)}")
                    recommendations.append(f"    -> These sensors behave similarly and may be candidates for feature selection")
        
        # Specific recommendations for common sensor types
        lp_sensors = [col for col in sensor_cols if 'LP' in col]
        acc_sensors = [col for col in sensor_cols if 'Acc' in col]
        bounce_sensors = [col for col in sensor_cols if 'Bounce' in col]
        
        if len(lp_sensors) > 1:
            recommendations.append(f"\n*** LP SENSOR GROUP ({len(lp_sensors)} sensors):")
            recommendations.append(f"  - {', '.join(lp_sensors)}")
            recommendations.append(f"    -> LP sensors often measure similar track properties - consider principal component analysis")
        
        if len(acc_sensors) > 1:
            recommendations.append(f"\n*** ACCELEROMETER GROUP ({len(acc_sensors)} sensors):")
            recommendations.append(f"  - {', '.join(acc_sensors)}")
            recommendations.append(f"    -> Accelerometers may capture similar vibration patterns - evaluate for redundancy")
        
        if len(bounce_sensors) > 1:
            recommendations.append(f"\n*** BOUNCE SENSOR GROUP ({len(bounce_sensors)} sensors):")
            recommendations.append(f"  - {', '.join(bounce_sensors)}")
            recommendations.append(f"    -> Bounce sensors measure vertical movement - check for spatial correlation")
        
        # Save recommendations
        with open(os.path.join(self.output_dir, 'dimensionality_reduction_recommendations.txt'), 'w') as f:
            f.write("DIMENSIONALITY REDUCTION RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            for rec in recommendations:
                f.write(rec + "\n")
        
        # Save high correlation pairs
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df.to_csv(os.path.join(self.output_dir, 'high_correlation_sensor_pairs.csv'), index=False)
        
        # Save cluster groups
        if cluster_groups:
            cluster_data = []
            for cluster_id, sensors in cluster_groups.items():
                cluster_data.append({
                    'Cluster_ID': cluster_id,
                    'Sensors': ', '.join(sensors),
                    'Size': len(sensors),
                    'Recommendation': 'Consider feature selection' if len(sensors) > 2 else 'Monitor for redundancy'
                })
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_csv(os.path.join(self.output_dir, 'sensor_cluster_groups.csv'), index=False)
        
        print(f"\nHigh-correlation analysis complete. Results saved to: {self.output_dir}")
        print(f"Generated {len(high_corr_pairs)} high-correlation pairs and {len(cluster_groups)} sensor clusters")
        
        # Store results for report generation
        self.analysis_results['high_correlation_groups'] = {
            'high_corr_pairs': high_corr_pairs,
            'cluster_groups': cluster_groups,
            'recommendations': recommendations,
            'correlation_matrix_file': 'high_correlation_sensor_groups.png',
            'high_corr_pairs_file': 'high_correlation_sensor_pairs.csv',
            'cluster_groups_file': 'sensor_cluster_groups.csv',
            'recommendations_file': 'dimensionality_reduction_recommendations.txt'
        }
    
    def analyze_time_series_patterns(self) -> None:
        """Analyze time series patterns in sensor data"""
        print("\n" + "="*80)
        print("TIME SERIES PATTERN ANALYSIS")
        print("="*80)
        
        if 'RecordingDateTime' not in self.data.columns:
            print("No timestamp column found. Skipping time series analysis.")
            return
        
        # Convert to datetime
        self.data['RecordingDateTime'] = pd.to_datetime(self.data['RecordingDateTime'])
        self.data = self.data.sort_values('RecordingDateTime')
        
        # Select key sensors for time series analysis
        key_sensors = ['Twist14m', 'BounceFrt', 'BounceRr', 'Speed']
        available_sensors = [s for s in key_sensors if s in self.data.columns]
        
        if not available_sensors:
            print("No suitable sensors found for time series analysis.")
            return
        
        # Create time series plots
        fig, axes = plt.subplots(len(available_sensors), 1, figsize=(15, 3*len(available_sensors)))
        if len(available_sensors) == 1:
            axes = [axes]
        
        for i, sensor in enumerate(available_sensors):
            ax = axes[i]
            
            # Plot sensor values over time
            ax.plot(self.data['RecordingDateTime'], self.data[sensor], alpha=0.7, linewidth=0.5)
            
            # Highlight break events if available
            if 'target' in self.data.columns:
                break_events = self.data[self.data['target'] == 1]
                if len(break_events) > 0:
                    ax.scatter(break_events['RecordingDateTime'], 
                             break_events[sensor], 
                             color='red', s=50, alpha=0.8, label='Break Events')
                    ax.legend()
            
            ax.set_title(f'{sensor} Over Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "time_series_patterns.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Time-based statistics
        print(f"\nTime series statistics:")
        print(f"Date range: {self.data['RecordingDateTime'].min()} to {self.data['RecordingDateTime'].max()}")
        print(f"Total time span: {self.data['RecordingDateTime'].max() - self.data['RecordingDateTime'].min()}")
        
        # Store results
        self.analysis_results['time_series_patterns'] = {
            'date_range': {
                'start': str(self.data['RecordingDateTime'].min()),
                'end': str(self.data['RecordingDateTime'].max()),
                'span': str(self.data['RecordingDateTime'].max() - self.data['RecordingDateTime'].min())
            }
        }
        
        # Analyze trends over time and position
        self._analyze_trends_over_time_and_position()
    
    def _analyze_trends_over_time_and_position(self) -> None:
        """Analyze trends over time and position for key sensors"""
        print("\n" + "="*80)
        print("TREND ANALYSIS OVER TIME AND POSITION")
        print("="*80)
        
        # Key sensors to analyze for trends
        key_sensors = ['Twist14m', 'BounceFrt', 'BounceRr', 'Speed']
        available_sensors = [s for s in key_sensors if s in self.data.columns]
        
        if not available_sensors:
            print("No suitable sensors found for trend analysis.")
            return
        
        # Create trend analysis plots
        fig, axes = plt.subplots(len(available_sensors), 2, figsize=(16, 4*len(available_sensors)))
        if len(available_sensors) == 1:
            axes = axes.reshape(1, -1)
        
        trend_results = []
        
        for i, sensor in enumerate(available_sensors):
            ax1, ax2 = axes[i, 0], axes[i, 1]
            
            # Plot 1: Trend over time
            ax1.plot(self.data['RecordingDateTime'], self.data[sensor], alpha=0.7, linewidth=0.5)
            ax1.set_title(f'{sensor} Over Time')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Trend over position (KMLocation)
            if 'KMLocation' in self.data.columns:
                ax2.scatter(self.data['KMLocation'], self.data[sensor], alpha=0.6, s=20)
                ax2.set_title(f'{sensor} Over Position (KM)')
                ax2.set_xlabel('KM Location')
                ax2.set_ylabel('Value')
                ax2.grid(True, alpha=0.3)
            
            # Calculate trend statistics
            sensor_data = self.data[sensor].dropna()
            if len(sensor_data) > 1:
                # Time trend (using index as proxy for time)
                time_indices = np.arange(len(sensor_data))
                time_slope, time_intercept, time_r_value, time_p_value, time_std_err = self._calculate_trend(sensor_data.values, time_indices)
                
                # Position trend (if available)
                if 'KMLocation' in self.data.columns:
                    km_data = self.data.loc[sensor_data.index, 'KMLocation'].dropna()
                    if len(km_data) > 1:
                        pos_slope, pos_intercept, pos_r_value, pos_p_value, pos_std_err = self._calculate_trend(sensor_data.loc[km_data.index].values, km_data.values)
                    else:
                        pos_slope = pos_r_value = pos_p_value = np.nan
                else:
                    pos_slope = pos_r_value = pos_p_value = np.nan
                
                trend_results.append({
                    'Sensor': sensor,
                    'Time_Trend_Slope': time_slope,
                    'Time_Trend_R_Squared': time_r_value**2,
                    'Time_Trend_P_Value': time_p_value,
                    'Time_Trend_Significant': time_p_value < 0.05,
                    'Position_Trend_Slope': pos_slope,
                    'Position_Trend_R_Squared': pos_r_value**2 if not np.isnan(pos_r_value) else np.nan,
                    'Position_Trend_P_Value': pos_p_value,
                    'Position_Trend_Significant': pos_p_value < 0.05 if not np.isnan(pos_p_value) else False
                })
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "trends_over_time_and_position.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save trend analysis results
        if trend_results:
            trend_df = pd.DataFrame(trend_results)
            trend_df.to_csv(os.path.join(self.output_dir, "trend_analysis_results.csv"), index=False)
            print(f"\nTrend analysis results saved to: {self.output_dir}/trend_analysis_results.csv")
            
            # Display key findings
            print("\nKey Trend Findings:")
            print("-" * 50)
            for _, row in trend_df.iterrows():
                sensor = row['Sensor']
                time_trend = "↗️ Rising" if row['Time_Trend_Slope'] > 0 else "↘️ Falling" if row['Time_Trend_Slope'] < 0 else "→ Stable"
                time_sig = "✅" if row['Time_Trend_Significant'] else "❌"
                pos_trend = "↗️ Rising" if row['Position_Trend_Slope'] > 0 else "↘️ Falling" if row['Position_Trend_Slope'] < 0 else "→ Stable"
                pos_sig = "✅" if row['Position_Trend_Significant'] else "❌"
                
                print(f"{sensor}:")
                print(f"  Time trend: {time_trend} (slope: {row['Time_Trend_Slope']:.4f}) {time_sig}")
                print(f"  Position trend: {pos_trend} (slope: {row['Position_Trend_Slope']:.4f}) {pos_sig}")
        
        # Store trend analysis results
        self.analysis_results['trend_analysis'] = {
            'trend_results': trend_results if trend_results else [],
            'trend_plot_file': 'trends_over_time_and_position.png'
        }
    
    def _calculate_trend(self, y_values: np.ndarray, x_values: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Calculate linear trend using scipy stats"""
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        return slope, intercept, r_value, p_value, std_err
    
    def generate_comprehensive_report(self) -> None:
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report_path = os.path.join(self.output_dir, "comprehensive_eda_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Railway EDA Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report provides comprehensive exploratory data analysis of railway sensor data,\n")
            f.write("including rail break patterns, track health analysis, and sensor correlations.\n\n")
            
            f.write("## Data Overview\n\n")
            f.write(self.loader.get_data_overview())
            f.write("\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Add comprehensive data quality findings
            if 'missing_data_analysis' in self.analysis_results:
                f.write("### Data Quality Analysis\n")
                missing_analysis = self.analysis_results['missing_data_analysis']
                
                f.write("#### Missing Data Patterns:\n")
                for table_name, analysis in missing_analysis.items():
                    f.write(f"- **{table_name}**: ")
                    if analysis['missing_columns']:
                        missing_cols = [f"{col['Column']} ({col['Missing_Percentage']:.1f}%)" 
                                      for col in analysis['missing_columns'][:5]]  # Top 5
                        f.write(f"{', '.join(missing_cols)}")
                        if len(analysis['missing_columns']) > 5:
                            f.write(f" and {len(analysis['missing_columns']) - 5} more columns")
                    else:
                        f.write("No missing data")
                    f.write("\n")
                
                f.write("\n#### Key Recommendations:\n")
                # Get recommendations from the first table with missing data
                for table_name, analysis in missing_analysis.items():
                    if analysis['recommendations']:
                        f.write(f"- **{table_name}**: ")
                        top_recs = analysis['recommendations'][:3]  # Top 3 recommendations
                        f.write("; ".join(top_recs))
                        f.write("\n")
                f.write("\n")
            
            if 'duplicate_analysis' in self.analysis_results:
                f.write("#### Duplicate Data Analysis:\n")
                duplicate_analysis = self.analysis_results['duplicate_analysis']
                for table_name, analysis in duplicate_analysis.items():
                    dup_pct = analysis['exact_duplicates']['percentage']
                    f.write(f"- **{table_name}**: {dup_pct:.2f}% exact duplicates")
                    if dup_pct > 5:
                        f.write(" ⚠️")
                    f.write("\n")
                f.write("\n")
            
            if 'prediction_impact' in self.analysis_results:
                f.write("#### Prediction Task Impact:\n")
                impact = self.analysis_results['prediction_impact']
                
                if 'target_quality' in impact:
                    target_quality = impact['target_quality']
                    missing_pct = target_quality['missing_analysis']['percentage']
                    f.write(f"- **Target Variable**: {missing_pct:.2f}% missing - ")
                    f.write(f"{target_quality['missing_analysis']['impact_level']}\n")
                
                if 'feature_quality' in impact:
                    feature_quality = impact['feature_quality']
                    critical_features = [col for col, qual in feature_quality.items() 
                                       if qual['missing_percentage'] > 20]
                    if critical_features:
                        f.write(f"- **Critical Features**: {', '.join(critical_features[:5])} have >20% missing data\n")
                
                f.write("\n")
            
            # Add key findings from analysis
            if 'rail_break_patterns' in self.analysis_results:
                f.write("### Rail Break Patterns\n")
                target_dist = self.analysis_results['rail_break_patterns']['target_distribution']
                if hasattr(target_dist, 'values') and target_dist is not None:
                    try:
                        total_events = sum(target_dist.values)
                        break_events = target_dist.get(1, 0)
                        no_break_events = target_dist.get(0, 0)
                        f.write(f"- Break events: {break_events}\n")
                        f.write(f"- No break events: {no_break_events}\n")
                        if total_events > 0:
                            f.write(f"- Break rate: {break_events / total_events * 100:.2f}%\n")
                        f.write("\n")
                    except (TypeError, AttributeError):
                        f.write("- Target distribution analysis not available\n\n")
                else:
                    f.write("- Target distribution analysis not available\n\n")
            
            # Add baseline sensor analysis findings
            if 'baseline_sensor_analysis' in self.analysis_results:
                f.write("### Baseline Sensor Analysis\n")
                baseline = self.analysis_results['baseline_sensor_analysis']
                if 'sensor_statistics' in baseline and not baseline['sensor_statistics'].empty:
                    stats_df = baseline['sensor_statistics']
                    f.write("#### Top Sensors by Variance:\n")
                    for _, row in stats_df.iterrows():
                        f.write(f"- **{row['Sensor']}**: Mean={row['Mean']:.3f}, Std={row['Std']:.3f}, Variance={row['Variance']:.4f}\n")
                        f.write(f"  - Range: {row['Min']:.3f} to {row['Max']:.3f}\n")
                        f.write(f"  - Outliers: {row['Outlier_Count']} ({row['Outlier_Percentage']:.1f}%)\n")
                        f.write(f"  - IQR: {row['Q1']:.3f} to {row['Q3']:.3f}\n\n")
                f.write("\n")
            
            # Add trend analysis findings
            if 'trend_analysis' in self.analysis_results:
                f.write("### Trend Analysis Over Time and Position\n")
                trend = self.analysis_results['trend_analysis']
                if 'trend_results' in trend and trend['trend_results']:
                    f.write("#### Key Trend Findings:\n")
                    for result in trend['trend_results']:
                        sensor = result['Sensor']
                        time_trend = "Rising" if result['Time_Trend_Slope'] > 0 else "Falling" if result['Time_Trend_Slope'] < 0 else "Stable"
                        time_sig = "Significant" if result['Time_Trend_Significant'] else "Not Significant"
                        pos_trend = "Rising" if result['Position_Trend_Slope'] > 0 else "Falling" if result['Position_Trend_Slope'] < 0 else "Stable"
                        pos_sig = "Significant" if result['Position_Trend_Significant'] else "Not Significant"
                        
                        f.write(f"- **{sensor}**:\n")
                        f.write(f"  - Time trend: {time_trend} (slope: {result['Time_Trend_Slope']:.4f}, {time_sig})\n")
                        f.write(f"  - Position trend: {pos_trend} (slope: {result['Position_Trend_Slope']:.4f}, {pos_sig})\n\n")
                f.write("\n")
            
            # Add specific sensor correlation findings
            if 'specific_sensor_correlations' in self.analysis_results:
                f.write("### Specific Sensor Correlations\n")
                specific_corr = self.analysis_results['specific_sensor_correlations']
                if 'correlation_details' in specific_corr and specific_corr['correlation_details']:
                    f.write("#### Key Sensor Pair Correlations:\n")
                    for corr in specific_corr['correlation_details']:
                        f.write(f"- **{corr['Sensor1']} vs {corr['Sensor2']}**:\n")
                        f.write(f"  - Pearson: {corr['Pearson_Correlation']:.3f} ({corr['Correlation_Strength']}, {corr['Correlation_Direction']})\n")
                        f.write(f"  - Spearman: {corr['Spearman_Correlation']:.3f}\n")
                        f.write(f"  - Data points: {corr['Data_Points']}\n\n")
                f.write("\n")
            
            if 'sensor_correlations' in self.analysis_results:
                f.write("### General Sensor Correlations\n")
                high_corr_count = len(self.analysis_results['sensor_correlations']['high_correlation_pairs'])
                f.write(f"- High correlation pairs (|r| > 0.7): {high_corr_count}\n\n")
            
            if 'high_correlation_groups' in self.analysis_results:
                f.write("### High-Correlation Sensor Groups for Dimensionality Reduction\n")
                high_corr_groups = self.analysis_results['high_correlation_groups']
                
                if high_corr_groups['high_corr_pairs']:
                    f.write(f"- **High correlation pairs (|r| > 0.8): {len(high_corr_groups['high_corr_pairs'])}**\n")
                    f.write("  - Top pairs for dimensionality reduction:\n")
                    for pair in high_corr_groups['high_corr_pairs'][:5]:
                        f.write(f"    • {pair['Sensor1']} ↔ {pair['Sensor2']}: {pair['Correlation']:.3f} ({pair['Strength']})\n")
                    f.write("\n")
                
                if high_corr_groups['cluster_groups']:
                    f.write(f"- **Sensor clusters identified: {len(high_corr_groups['cluster_groups'])}**\n")
                    f.write("  - Clusters with >2 sensors (candidates for feature selection):\n")
                    for cluster_id, sensors in high_corr_groups['cluster_groups'].items():
                        if len(sensors) > 2:
                            f.write(f"    • Cluster {cluster_id}: {', '.join(sensors[:5])}{'...' if len(sensors) > 5 else ''}\n")
                    f.write("\n")
                
                f.write("- **Key recommendations for dimensionality reduction:**\n")
                f.write("  • Remove redundant sensors with very high correlations (>0.9)\n")
                f.write("  • Use principal component analysis for sensor groups\n")
                f.write("  • Consider feature selection for clustered sensors\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Predictive Maintenance**: Use sensor patterns to predict rail breaks\n")
            f.write("2. **Track Monitoring**: Focus on high-risk track sections\n")
            f.write("3. **Sensor Optimization**: Consider redundant sensors with high correlations\n")
            f.write("4. **Data Quality**: Address missing data and outliers\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `data_overview.txt`: Complete data overview\n")
            f.write("- `data_quality_report.csv`: Basic data quality metrics\n")
            f.write("- `missing_data_summary.csv`: Comprehensive missing data summary\n")
            f.write("- `missing_data_analysis.txt`: Detailed missing data patterns and recommendations\n")
            f.write("- `duplicate_data_summary.csv`: Duplicate data analysis summary\n")
            f.write("- `target_quality_summary.csv`: Target variable quality assessment\n")
            f.write("- `feature_quality_summary.csv`: Feature quality impact assessment\n")
            f.write("- `overall_quality_recommendations.txt`: Overall data quality recommendations\n")
            f.write("- `sensor_statistics.csv`: Statistical summary of sensors\n")
            f.write("- `break_pattern_significance.csv`: Statistical significance of break patterns\n")
            f.write("- `track_health_by_location.csv`: Location-based health metrics\n")
            f.write("- `correlation_matrix_pearson.csv`: Sensor correlation matrix\n")
            f.write("- `high_correlations.csv`: Highly correlated sensor pairs\n")
            f.write("- `trend_analysis_results.csv`: Trend analysis over time and position\n")
            f.write("- `specific_sensor_correlations.csv`: Detailed correlation analysis for key sensor pairs\n")
            f.write("- `sensor_patterns_by_break.png`: Baseline sensor distributions (when no break data)\n")
            f.write("- `trends_over_time_and_position.png`: Trend analysis visualizations\n")
            f.write("- `specific_sensor_correlations.png`: Specific sensor correlation plots\n")
            f.write("- `high_correlation_sensor_groups.png`: High-correlation sensor groups visualization\n")
            f.write("- `high_correlation_sensor_pairs.csv`: High-correlation sensor pairs for dimensionality reduction\n")
            f.write("- `sensor_cluster_groups.csv`: Sensor cluster groups identified\n")
            f.write("- `dimensionality_reduction_recommendations.txt`: Actionable recommendations for feature reduction\n")
            f.write("- `enhanced_break_pattern_significance.csv`: Enhanced statistical significance with variance testing\n")
            f.write("- Various other visualization plots (PNG format)\n\n")
        
        print(f"Comprehensive report saved to: {report_path}")
        
        # Also save integrated data
        if self.data is not None:
            integrated_data_path = os.path.join(self.output_dir, "integrated_railway_data.csv")
            self.loader.save_integrated_data(integrated_data_path)
            print(f"Integrated data saved to: {integrated_data_path}")
    
    def run_complete_analysis(self) -> None:
        """Run the complete EDA analysis pipeline"""
        print("Starting Enhanced Railway EDA Analysis...")
        print("="*80)
        
        try:
            # Load and integrate data
            self.load_and_integrate_data()
            
            # Run all analyses
            self.display_data_overview()
            self.analyze_data_quality()
            self.analyze_sensor_distributions()
            self.analyze_rail_break_patterns()
            self.analyze_track_health_by_location()
            self.analyze_sensor_correlations()
            self.analyze_time_series_patterns()
            self._analyze_high_correlation_sensor_groups()
            
            # Generate comprehensive report
            self.generate_comprehensive_report()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)
            print(f"All results saved to: {self.output_dir}/")
            print("Check the comprehensive report for detailed findings and recommendations.")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the enhanced EDA analysis"""
    print("Enhanced Railway EDA Tool")
    print("="*50)
    
    # Get data directory
    data_dir = input("Enter the path to your data directory (or press Enter for 'datas'): ").strip()
    if not data_dir:
        data_dir = "datas"
    
    # Get output directory
    output_dir = input("Enter output directory name (or press Enter for 'real_data_output'): ").strip()
    if not output_dir:
        output_dir = "real_data_output"
    
    # Initialize and run analysis
    eda_tool = EnhancedRailwayEDA(data_dir, output_dir)
    eda_tool.run_complete_analysis()


if __name__ == "__main__":
    main()
