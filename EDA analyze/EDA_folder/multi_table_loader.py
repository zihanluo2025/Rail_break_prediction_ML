"""
Multi-Table Railway Data Loader
Handles loading and integrating multiple related railway data tables
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging

class MultiTableRailwayDataLoader:
    """
    Loads and integrates multiple railway data tables for comprehensive EDA
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader with the data directory path
        
        Args:
            data_dir (str): Path to directory containing CSV files
        """
        self.data_dir = data_dir
        self.tables = {}
        self.integrated_data = None
        self.data_info = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data directory
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of table names and their data
        """
        self.logger.info(f"Loading tables from: {self.data_dir}")
        
        # Expected table files and their descriptions
        expected_tables = {
            'wagondata': 'Main sensor data with measurements',
            'railbreaklocations': 'Rail break location mapping',
            'trainingcontext': 'Training data context with targets',
            'testcontext': 'Test data context',
            'inferencecontext': 'Inference data context',
            'basecodemap': 'Base code mapping table',
            'allrailbreaksmapped': 'Comprehensive rail break mapping'
        }
        
        for table_name, description in expected_tables.items():
            file_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if os.path.exists(file_path):
                try:
                    self.tables[table_name] = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {table_name}: {len(self.tables[table_name])} rows, {len(self.tables[table_name].columns)} columns")
                    
                    # Store basic info
                    self.data_info[table_name] = {
                        'rows': len(self.tables[table_name]),
                        'columns': len(self.tables[table_name].columns),
                        'memory_usage': self.tables[table_name].memory_usage(deep=True).sum(),
                        'description': description
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error loading {table_name}: {e}")
            else:
                self.logger.warning(f"Table file not found: {file_path}")
        
        return self.tables
    
    def get_table_info(self) -> Dict[str, dict]:
        """
        Get comprehensive information about all loaded tables
        
        Returns:
            Dict[str, dict]: Table information summary
        """
        return self.data_info
    
    def get_table_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all tables
        
        Returns:
            pd.DataFrame: Summary statistics for all tables
        """
        summary_data = []
        
        for table_name, info in self.data_info.items():
            df = self.tables[table_name]
            
            # Get data types
            dtypes = df.dtypes.value_counts().to_dict()
            
            # Get missing values
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            missing_info = f"{missing_pct.sum():.1f}% ({df.isnull().sum().sum()} total)"
            
            summary_data.append({
                'Table': table_name,
                'Rows': info['rows'],
                'Columns': info['columns'],
                'Memory (MB)': round(info['memory_usage'] / 1024 / 1024, 2),
                'Missing Data': missing_info,
                'Description': info['description']
            })
        
        return pd.DataFrame(summary_data)
    
    def integrate_data(self) -> pd.DataFrame:
        """
        Integrate all tables into a comprehensive dataset for analysis
        
        Returns:
            pd.DataFrame: Integrated dataset
        """
        self.logger.info("Integrating data tables...")
        
        if 'wagondata' not in self.tables:
            raise ValueError("wagondata table is required for integration")
        
        # Start with main sensor data
        integrated = self.tables['wagondata'].copy()
        
        # Add rail break location information
        if 'railbreaklocations' in self.tables:
            integrated = self._merge_break_locations(integrated)
        
        # Add training context (targets and RUL)
        if 'trainingcontext' in self.tables:
            integrated = self._merge_training_context(integrated)
        
        # Add base code mapping
        if 'basecodemap' in self.tables:
            integrated = self._merge_base_code_mapping(integrated)
        
        self.integrated_data = integrated
        self.logger.info(f"Integration complete. Final dataset: {len(integrated)} rows, {len(integrated.columns)} columns")
        
        return integrated
    
    def _merge_break_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge rail break location information"""
        # Since wagondata already contains the fused data with SectionBreakStartKM and SectionBreakFinishKM,
        # we can directly calculate if each row is in a break section using vectorized operations
        
        # Check if KMLocation falls within the break section range for each row
        # This is much more efficient than the previous row-by-row approach since the data is already fused
        df['has_break_location'] = (
            (df['KMLocation'] >= df['SectionBreakStartKM']) & 
            (df['KMLocation'] <= df['SectionBreakFinishKM'])
        )
        
        # Add break location details
        df['is_break_section'] = df['has_break_location']
        
        # Debug information
        break_count = df['has_break_location'].sum()
        total_count = len(df)
        self.logger.info(f"Break detection: {break_count} break locations out of {total_count} total readings")
        
        # Note: We don't create target here because it should come from trainingcontext.csv
        # The has_break_location flag indicates if the sensor reading is from a monitored break section
        
        return df
    
    def _merge_training_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge training context information"""
        training = self.tables['trainingcontext']
        
        # Merge on BaseCode and SectionBreakStartKM
        merged = pd.merge(
            df, 
            training[['BaseCode', 'SectionBreakStartKM', 'target', 'rul', 'break_date']], 
            on=['BaseCode', 'SectionBreakStartKM'], 
            how='left'
        )
        
        # If no target from training context, create a default target based on break location
        # This handles cases where we have sensor data but no training labels
        if 'target' in merged.columns:
            # Fill NaN targets with 0 (assuming no break if no training data)
            merged['target'] = merged['target'].fillna(0)
        else:
            # If no target column exists, create one based on break location
            merged['target'] = merged['has_break_location'].astype(int)
        
        return merged
    
    def _merge_base_code_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge base code mapping information"""
        base_map = self.tables['basecodemap']
        
        # Merge on BaseCode
        merged = pd.merge(
            df, 
            base_map, 
            on='BaseCode', 
            how='left'
        )
        
        return merged
    
    def get_sensor_columns(self) -> List[str]:
        """
        Get list of sensor measurement columns
        
        Returns:
            List[str]: Column names that are sensor measurements
        """
        if 'wagondata' not in self.tables:
            return []
        
        # Define sensor columns (excluding metadata)
        metadata_cols = {
            'BaseCode', 'SectionBreakStartKM', 'SectionBreakFinishKM', 'KMLocation',
            'ICWVehicle', 'FileLoadStatus', 'RecordingDateTime', 'RecordingDate',
            'p_key', 'partition_col', 'target', 'rul', 'break_date', 'r_date',
            'MappedBaseCode'
        }
        
        sensor_cols = [col for col in self.tables['wagondata'].columns if col not in metadata_cols]
        return sensor_cols
    
    def get_time_series_columns(self) -> List[str]:
        """
        Get columns suitable for time series analysis
        
        Returns:
            List[str]: Column names for time series analysis
        """
        sensor_cols = self.get_sensor_columns()
        
        # Filter to numeric columns only
        if 'wagondata' in self.tables:
            numeric_cols = self.tables['wagondata'][sensor_cols].select_dtypes(include=[np.number]).columns.tolist()
            return numeric_cols
        
        return []
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get categorical columns for analysis
        
        Returns:
            List[str]: Categorical column names
        """
        if 'wagondata' not in self.tables:
            return []
        
        categorical_cols = self.tables['wagondata'].select_dtypes(include=['object', 'category']).columns.tolist()
        return categorical_cols
    
    def validate_data_quality(self) -> Dict[str, dict]:
        """
        Perform comprehensive data quality validation
        
        Returns:
            Dict[str, dict]: Quality metrics for each table
        """
        quality_report = {}
        
        for table_name, df in self.tables.items():
            quality_metrics = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2),
                'data_types': df.dtypes.value_counts().to_dict(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            quality_report[table_name] = quality_metrics
        
        return quality_report
    
    def save_integrated_data(self, output_path: str) -> None:
        """
        Save the integrated dataset to a file
        
        Args:
            output_path (str): Path to save the integrated data
        """
        if self.integrated_data is not None:
            self.integrated_data.to_csv(output_path, index=False)
            self.logger.info(f"Integrated data saved to: {output_path}")
        else:
            self.logger.warning("No integrated data available. Run integrate_data() first.")
    
    def get_data_overview(self) -> str:
        """
        Generate a comprehensive data overview report
        
        Returns:
            str: Formatted overview report
        """
        if not self.tables:
            return "No tables loaded. Run load_all_tables() first."
        
        overview = []
        overview.append("=" * 80)
        overview.append("RAILWAY DATA OVERVIEW")
        overview.append("=" * 80)
        overview.append("")
        
        # Table summary
        overview.append("TABLE SUMMARY:")
        overview.append("-" * 40)
        summary_df = self.get_table_summary()
        overview.append(summary_df.to_string(index=False))
        overview.append("")
        
        # Data quality summary
        overview.append("DATA QUALITY SUMMARY:")
        overview.append("-" * 40)
        quality = self.validate_data_quality()
        
        for table_name, metrics in quality.items():
            overview.append(f"{table_name}:")
            overview.append(f"  - Missing data: {metrics['missing_percentage']}%")
            overview.append(f"  - Duplicates: {metrics['duplicate_percentage']}%")
            overview.append(f"  - Memory: {metrics['memory_usage_mb']} MB")
            overview.append("")
        
        # Sensor information
        if 'wagondata' in self.tables:
            sensor_cols = self.get_sensor_columns()
            overview.append(f"SENSOR COLUMNS: {len(sensor_cols)}")
            overview.append("-" * 40)
            overview.append(", ".join(sensor_cols[:10]))  # Show first 10
            if len(sensor_cols) > 10:
                overview.append(f"... and {len(sensor_cols) - 10} more")
            overview.append("")
        
        overview.append("=" * 80)
        
        return "\n".join(overview)
