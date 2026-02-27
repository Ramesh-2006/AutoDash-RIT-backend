"""
General analysis functions for AutoDash RIT
These functions work with any dataset type and provide basic data analysis capabilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(data):
    """
    Recursively converts numpy types in a data structure (dict, list, value)
    to native Python types for JSON serialization.
    """
    # Updated for NumPy 2.0 compatibility
    if isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    if isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    if isinstance(data, np.ndarray):
        return [convert_numpy_types(x) for x in data.tolist()]
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [convert_numpy_types(x) for x in data]
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    if pd.isna(data):
        return None
    return data
# ========== BASIC DATA OVERVIEW FUNCTIONS ==========

def dataset_overview(df):
    """
    Provides a comprehensive overview of the dataset
    Returns: Dictionary with dataset overview information
    """
    try:
        # Basic info
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Data types
        dtypes_count = df.dtypes.value_counts().to_dict()
        dtypes_count = {str(k): int(v) for k, v in dtypes_count.items()}
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
        
        # Column info
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float((df[col].isnull().sum() / n_rows) * 100),
                'unique_values': int(df[col].nunique())
            }
            
            # Add numeric stats if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None
                })
            
            columns_info.append(col_info)
        
        result = {
            'summary': {
                'rows': n_rows,
                'columns': n_cols,
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'missing_percentage': round(missing_percentage, 2),
                'memory_usage_mb': round(memory_usage, 2),
                'duplicate_rows': int(df.duplicated().sum())
            },
            'data_types': dtypes_count,
            'columns': columns_info
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in dataset_overview: {str(e)}'}

def data_quality_report(df):
    """
    Generates a comprehensive data quality report
    Returns: Dictionary with data quality metrics and visualizations
    """
    try:
        n_rows = len(df)
        
        # Missing data analysis
        missing_df = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percentage': (df.isnull().sum().values / n_rows * 100)
        }).sort_values('missing_percentage', ascending=False)
        
        # Create missing data visualization
        fig_missing = px.bar(
            missing_df.head(20),
            x='column',
            y='missing_percentage',
            title='Top 20 Columns by Missing Data Percentage',
            labels={'missing_percentage': 'Missing %', 'column': 'Column'},
            color='missing_percentage',
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(b=100)
        )
        
        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=[str(dt) for dt in dtype_counts.index],
            title='Data Type Distribution',
            hole=0.4
        )
        
        # Unique values analysis
        unique_analysis = []
        for col in df.columns:
            n_unique = df[col].nunique()
            unique_ratio = (n_unique / n_rows) * 100 if n_rows > 0 else 0
            unique_analysis.append({
                'column': col,
                'unique_count': n_unique,
                'unique_ratio': round(unique_ratio, 2)
            })
        
        unique_df = pd.DataFrame(unique_analysis).sort_values('unique_ratio', ascending=False)
        
        fig_unique = px.bar(
            unique_df.head(15),
            x='column',
            y='unique_ratio',
            title='Columns with Highest Unique Value Ratio',
            labels={'unique_ratio': 'Unique Values %', 'column': 'Column'},
            color='unique_ratio',
            color_continuous_scale='Viridis'
        )
        fig_unique.update_layout(xaxis_tickangle=-45)
        
        # Data quality score calculation
        completeness_score = 100 - missing_df['missing_percentage'].mean()
        uniqueness_score = 100 - (unique_df['unique_ratio'].mean() / 100)  # Lower uniqueness is better for categorical
        
        # Adjust uniqueness score for potential ID columns
        id_columns = [col for col in df.columns if 'id' in col.lower() or 'code' in col.lower()]
        if id_columns:
            uniqueness_score = min(uniqueness_score, 70)  # Penalize datasets with ID columns
        
        data_quality_score = (completeness_score * 0.7 + uniqueness_score * 0.3)
        
        result = {
            'missing_data': {
                'total_missing': int(df.isnull().sum().sum()),
                'columns_with_missing': int((df.isnull().sum() > 0).sum()),
                'missing_viz': fig_missing.to_json()
            },
            'data_types': {
                'distribution': {str(k): int(v) for k, v in dtype_counts.to_dict().items()},
                'viz': fig_dtypes.to_json()
            },
            'uniqueness': {
                'analysis': unique_analysis[:10],
                'viz': fig_unique.to_json()
            },
            'data_quality_score': round(data_quality_score, 2),
            'recommendations': []
        }
        
        # Generate recommendations
        if missing_df['missing_percentage'].max() > 50:
            result['recommendations'].append("Consider dropping columns with >50% missing data")
        if (df.isnull().sum().sum() > 0):
            result['recommendations'].append("Handle missing values through imputation or removal")
        if len([c for c in df.columns if 'id' in c.lower()]) > 0:
            result['recommendations'].append("Review ID columns - they may not be useful for analysis")
        if df.duplicated().sum() > 0:
            result['recommendations'].append(f"Remove {df.duplicated().sum()} duplicate rows")
        
        return result
    except Exception as e:
        return {'error': f'Error in data_quality_report: {str(e)}'}

# ========== STATISTICAL ANALYSIS FUNCTIONS ==========

def descriptive_statistics(df):
    """
    Calculates comprehensive descriptive statistics
    Returns: Dictionary with statistical summaries and visualizations
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'message': 'No numeric columns found for descriptive statistics'}
        
        # Basic statistics
        stats_df = df[numeric_cols].describe().round(2)
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            'skewness': df[numeric_cols].skew().round(2),
            'kurtosis': df[numeric_cols].kurtosis().round(2),
            'variance': df[numeric_cols].var().round(2),
            'range': (df[numeric_cols].max() - df[numeric_cols].min()).round(2)
        }).T
        
        # Create box plot for numeric columns
        if len(numeric_cols) <= 15:  # Limit to 15 columns for readability
            fig_box = px.box(
                df[numeric_cols],
                title='Distribution of Numeric Variables',
                points="outliers"
            )
            fig_box.update_layout(height=500)
        else:
            # If too many columns, show top 15 by variance
            top_cols = df[numeric_cols].var().sort_values(ascending=False).head(15).index.tolist()
            fig_box = px.box(
                df[top_cols],
                title='Distribution of Top 15 Numeric Variables (by variance)',
                points="outliers"
            )
            fig_box.update_layout(height=500)
        
        # Create histogram grid
        n_cols_to_plot = min(len(numeric_cols), 9)  # Max 9 plots in grid
        selected_cols = numeric_cols[:n_cols_to_plot]
        
        fig_hist = make_subplots(
            rows=3, cols=3,
            subplot_titles=selected_cols,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, col in enumerate(selected_cols):
            row = i // 3 + 1
            col_pos = i % 3 + 1
            fig_hist.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig_hist.update_layout(
            title='Distribution Histograms',
            height=600,
            showlegend=False
        )
        
        result = {
            'statistics': {
                'basic': stats_df.to_dict(),
                'additional': additional_stats.to_dict()
            },
            'visualizations': {
                'box_plot': fig_box.to_json(),
                'histograms': fig_hist.to_json()
            },
            'numeric_columns': numeric_cols,
            'summary': {
                'total_numeric': len(numeric_cols),
                'columns_with_outliers': len([col for col in numeric_cols if (df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25))).any()])
            }
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in descriptive_statistics: {str(e)}'}

def correlation_analysis(df):
    """
    Performs correlation analysis on numeric columns
    Returns: Correlation matrix and visualizations
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'message': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverinfo='text'
        ))
        
        fig_heatmap.update_layout(
            title='Correlation Matrix Heatmap',
            height=600,
            width=700,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        strong_corr = corr_df[abs(corr_df['correlation']) > 0.7].sort_values('correlation', ascending=False)
        
        # Create bar chart of top correlations
        if not strong_corr.empty:
            top_corr = strong_corr.head(10)
            top_corr['pair'] = top_corr['var1'] + ' - ' + top_corr['var2']
            
            fig_top = px.bar(
                top_corr,
                x='correlation',
                y='pair',
                orientation='h',
                title='Top 10 Strongest Correlations',
                color='correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1]
            )
            fig_top.update_layout(height=400)
        else:
            fig_top = None
        
        result = {
            'correlation_matrix': corr_matrix.round(2).to_dict(),
            'visualizations': {
                'heatmap': fig_heatmap.to_json()
            },
            'strong_correlations': strong_corr.to_dict('records') if not strong_corr.empty else [],
            'summary': {
                'total_pairs': len(corr_pairs),
                'strong_positive': len(strong_corr[strong_corr['correlation'] > 0.7]),
                'strong_negative': len(strong_corr[strong_corr['correlation'] < -0.7])
            }
        }
        
        if fig_top:
            result['visualizations']['top_correlations'] = fig_top.to_json()
        
        return result
    except Exception as e:
        return {'error': f'Error in correlation_analysis: {str(e)}'}

# ========== CATEGORICAL ANALYSIS FUNCTIONS ==========

def categorical_analysis(df):
    """
    Analyzes categorical columns in the dataset
    Returns: Frequency distributions and visualizations
    """
    try:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return {'message': 'No categorical columns found in dataset'}
        
        # Limit to 10 columns for display
        cols_to_analyze = categorical_cols[:10]
        
        distributions = {}
        visualizations = {}
        
        for col in cols_to_analyze:
            # Get value counts
            value_counts = df[col].value_counts().head(15)  # Top 15 categories
            percentages = (value_counts / len(df) * 100).round(2)
            
            dist_df = pd.DataFrame({
                'count': value_counts,
                'percentage': percentages
            })
            
            distributions[col] = dist_df.to_dict()
            
            # Create bar chart
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f'Distribution of {col}',
                labels={'x': col, 'y': 'Count'},
                color=value_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False
            )
            visualizations[col] = fig.to_json()
        
        # Summary statistics for categorical columns
        summary = []
        for col in categorical_cols:
            n_unique = df[col].nunique()
            most_common = df[col].mode().iloc[0] if not df[col].mode().empty else None
            most_common_count = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
            
            summary.append({
                'column': col,
                'unique_values': n_unique,
                'most_common': str(most_common)[:50] + '...' if most_common and len(str(most_common)) > 50 else str(most_common),
                'most_common_count': int(most_common_count),
                'most_common_percentage': round(most_common_count / len(df) * 100, 2)
            })
        
        result = {
            'categorical_columns': categorical_cols,
            'distributions': distributions,
            'visualizations': visualizations,
            'summary': summary,
            'total_categorical': len(categorical_cols)
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in categorical_analysis: {str(e)}'}

# ========== OUTLIER DETECTION FUNCTIONS ==========

def outlier_detection(df):
    """
    Detects outliers in numeric columns using IQR method
    Returns: Outlier information and visualizations
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'message': 'No numeric columns found for outlier detection'}
        
        outlier_info = []
        
        for col in numeric_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
            
            if outlier_count > 0:
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'outlier_percentage': round(outlier_percentage, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'min_outlier': round(outliers.min(), 2) if not outliers.empty else None,
                    'max_outlier': round(outliers.max(), 2) if not outliers.empty else None
                })
        
        # Create visualization for columns with outliers
        cols_with_outliers = [info['column'] for info in outlier_info]
        cols_to_plot = cols_with_outliers[:10]  # Limit to 10 columns
        
        if cols_to_plot:
            fig_box = px.box(
                df[cols_to_plot],
                title='Box Plots with Outliers',
                points="outliers"
            )
            fig_box.update_layout(height=500)
        else:
            fig_box = None
        
        # Summary statistics
        total_columns_with_outliers = len(outlier_info)
        total_outliers = sum(info['outlier_count'] for info in outlier_info)
        
        result = {
            'outlier_info': outlier_info,
            'summary': {
                'total_columns_analyzed': len(numeric_cols),
                'columns_with_outliers': total_columns_with_outliers,
                'total_outliers': total_outliers,
                'avg_outlier_percentage': round(sum(info['outlier_percentage'] for info in outlier_info) / total_columns_with_outliers if total_columns_with_outliers > 0 else 0, 2)
            },
            'recommendations': []
        }
        
        if fig_box:
            result['visualization'] = fig_box.to_json()
        
        # Generate recommendations
        if total_columns_with_outliers > 0:
            result['recommendations'].append("Consider investigating outliers for data entry errors")
            if total_columns_with_outliers > len(numeric_cols) * 0.3:
                result['recommendations'].append("High number of outliers detected - consider robust scaling methods")
        
        return result
    except Exception as e:
        return {'error': f'Error in outlier_detection: {str(e)}'}

# ========== MISSING VALUE ANALYSIS ==========

def missing_value_analysis(df):
    """
    Detailed analysis of missing values in the dataset
    Returns: Missing value patterns and recommendations
    """
    try:
        # Calculate missing values
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'column': df.columns,
            'missing_count': missing.values,
            'missing_percent': missing_percent.values.round(2)
        }).sort_values('missing_count', ascending=False)
        
        # Filter columns with missing values
        missing_with_data = missing_df[missing_df['missing_count'] > 0]
        
        if missing_with_data.empty:
            return {'message': 'No missing values found in dataset'}
        
        # Create visualizations
        fig_missing = px.bar(
            missing_with_data.head(20),
            x='column',
            y='missing_percent',
            title='Missing Values by Column',
            labels={'missing_percent': 'Missing %', 'column': 'Column'},
            color='missing_percent',
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(xaxis_tickangle=-45, height=500)
        
        # Missing value heatmap for correlation
        cols_with_missing = missing_with_data['column'].tolist()
        if len(cols_with_missing) > 1 and len(cols_with_missing) <= 20:
            # Create missing indicator matrix
            missing_matrix = pd.DataFrame()
            for col in cols_with_missing:
                missing_matrix[f"{col}_missing"] = df[col].isnull().astype(int)
            
            # Calculate correlation of missingness
            missing_corr = missing_matrix.corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=missing_corr,
                x=missing_corr.columns,
                y=missing_corr.columns,
                colorscale='Viridis',
                text=np.round(missing_corr.values, 2),
                texttemplate='%{text}'
            ))
            fig_heatmap.update_layout(
                title='Missing Value Correlation',
                height=600,
                width=600
            )
        else:
            fig_heatmap = None
        
        # Generate recommendations
        recommendations = []
        
        high_missing = missing_with_data[missing_with_data['missing_percent'] > 50]
        if not high_missing.empty:
            recommendations.append(f"Consider dropping columns with >50% missing: {', '.join(high_missing['column'].tolist())}")
        
        medium_missing = missing_with_data[(missing_with_data['missing_percent'] > 20) & (missing_with_data['missing_percent'] <= 50)]
        if not medium_missing.empty:
            recommendations.append("Consider imputation for columns with 20-50% missing values")
        
        low_missing = missing_with_data[missing_with_data['missing_percent'] <= 20]
        if not low_missing.empty:
            recommendations.append("Consider simple imputation (mean/median/mode) for columns with <20% missing")
        
        result = {
            'summary': {
                'total_missing': int(missing.sum()),
                'total_cells': int(df.size),
                'missing_percentage': round((missing.sum() / df.size) * 100, 2),
                'columns_with_missing': len(missing_with_data),
                'rows_with_missing': int(df.isnull().any(axis=1).sum()),
                'rows_with_missing_percentage': round((df.isnull().any(axis=1).sum() / len(df)) * 100, 2)
            },
            'missing_by_column': missing_with_data.to_dict('records'),
            'visualizations': {
                'missing_bar': fig_missing.to_json()
            },
            'recommendations': recommendations
        }
        
        if fig_heatmap:
            result['visualizations']['missing_correlation'] = fig_heatmap.to_json()
        
        return result
    except Exception as e:
        return {'error': f'Error in missing_value_analysis: {str(e)}'}

# ========== GENERAL INSIGHTS ==========

def general_insights_analysis(df):
    """
    Provides general insights about the dataset
    Returns: Key findings and observations
    """
    try:
        insights = []
        recommendations = []
        
        # Dataset size insight
        n_rows, n_cols = df.shape
        insights.append(f"Dataset contains {n_rows:,} rows and {n_cols} columns")
        
        if n_rows > 10000:
            insights.append("Large dataset - consider sampling for faster analysis")
        elif n_rows < 100:
            insights.append("Small dataset - statistical significance may be limited")
        
        # Data types insight
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_count = len(df.select_dtypes(include=['datetime64']).columns)
        
        insights.append(f"Data types: {numeric_count} numeric, {categorical_count} categorical, {datetime_count} datetime columns")
        
        # Missing data insight
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_percent = (missing_total / df.size) * 100
            insights.append(f"Missing values: {missing_total:,} ({missing_percent:.1f}% of total cells)")
            
            if missing_percent > 30:
                recommendations.append("High percentage of missing data - consider data quality improvements")
        else:
            insights.append("No missing values found - good data quality")
        
        # Duplicate insight
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percent = (duplicate_count / n_rows) * 100
            insights.append(f"Found {duplicate_count:,} duplicate rows ({duplicate_percent:.1f}%)")
            
            if duplicate_percent > 10:
                recommendations.append("High number of duplicates - consider deduplication")
        else:
            insights.append("No duplicate rows found")
        
        # Column name quality
        column_issues = []
        for col in df.columns:
            if ' ' in col:
                column_issues.append(f"'{col}' contains spaces")
            if any(c in col for c in '!@#$%^&*()+='):
                column_issues.append(f"'{col}' contains special characters")
        
        if column_issues:
            insights.append(f"Column name issues: {len(column_issues)} columns need cleaning")
            recommendations.append("Clean column names (remove spaces, special characters)")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        insights.append(f"Memory usage: {memory_mb:.2f} MB")
        
        if memory_mb > 100:
            recommendations.append("Large memory footprint - consider optimizing data types")
        
        result = {
            'insights': insights,
            'recommendations': recommendations,
            'quick_stats': {
                'rows': n_rows,
                'columns': n_cols,
                'numeric_features': numeric_count,
                'categorical_features': categorical_count,
                'datetime_features': datetime_count,
                'missing_cells': int(missing_total),
                'duplicate_rows': int(duplicate_count),
                'memory_mb': round(memory_mb, 2)
            }
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in general_insights_analysis: {str(e)}'}

# ========== DATA PREPROCESSING SUGGESTIONS ==========

def preprocessing_suggestions(df):
    """
    Suggests preprocessing steps based on data analysis
    Returns: Recommendations for data preprocessing
    """
    try:
        suggestions = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            suggestions.append({
                'step': 'Handle Missing Values',
                'description': f'Missing values found in {len(missing_cols)} columns',
                'actions': [
                    'Remove columns with >50% missing data',
                    'Impute numerical columns with mean/median',
                    'Impute categorical columns with mode',
                    'Consider advanced imputation techniques for important features'
                ]
            })
        
        # Check for categorical variables that need encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            suggestions.append({
                'step': 'Encode Categorical Variables',
                'description': f'Found {len(categorical_cols)} categorical columns requiring encoding',
                'actions': [
                    'Apply One-Hot Encoding for nominal categories with <10 unique values',
                    'Apply Label Encoding for ordinal categories',
                    'Consider target encoding for high-cardinality features'
                ]
            })
        
        # Check for scaling needs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Check for different scales
            ranges = df[numeric_cols].max() - df[numeric_cols].min()
            if (ranges > 100).any():
                suggestions.append({
                    'step': 'Scale Numerical Features',
                    'description': 'Features have significantly different scales',
                    'actions': [
                        'Apply StandardScaler for normally distributed features',
                        'Apply MinMaxScaler for bounded features',
                        'Apply RobustScaler if outliers are present'
                    ]
                })
        
        # Check for outliers
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                suggestions.append({
                    'step': 'Handle Outliers',
                    'description': f'Outliers detected in numerical columns',
                    'actions': [
                        'Investigate outliers for data entry errors',
                        'Consider capping/winsorizing extreme values',
                        'Use robust statistical methods'
                    ]
                })
                break
        
        # Check for datetime conversions
        date_like_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    date_like_cols.append(col)
                except:
                    pass
        
        if date_like_cols:
            suggestions.append({
                'step': 'Convert Date Columns',
                'description': f'Found {len(date_like_cols)} columns that appear to contain dates',
                'actions': [
                    'Convert to datetime format',
                    'Extract useful features: year, month, day, weekday',
                    'Consider time-based aggregations'
                ]
            })
        
        # Check for feature engineering opportunities
        if 'date' in ' '.join(df.columns).lower() and numeric_cols:
            suggestions.append({
                'step': 'Feature Engineering',
                'description': 'Opportunities for creating new features',
                'actions': [
                    'Create interaction features between important variables',
                    'Generate polynomial features for non-linear relationships',
                    'Create aggregates based on categorical groupings'
                ]
            })
        
        result = {
            'suggestions': suggestions,
            'total_suggestions': len(suggestions),
            'priority': 'High' if len(suggestions) > 3 else 'Medium' if len(suggestions) > 1 else 'Low'
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in preprocessing_suggestions: {str(e)}'}

# ========== SUMMARY REPORT ==========

def summary_report(df):
    """
    Generates a comprehensive summary report of the dataset
    Returns: Complete overview with key metrics
    """
    try:
        # Basic info
        n_rows, n_cols = df.shape
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        
        # Missing data
        missing_total = df.isnull().sum().sum()
        missing_percent = (missing_total / df.size) * 100 if df.size > 0 else 0
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percent = (duplicate_count / n_rows) * 100 if n_rows > 0 else 0
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Column statistics
        column_stats = []
        for col in df.columns:
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'missing': int(df[col].isnull().sum()),
                'missing_percent': round((df[col].isnull().sum() / n_rows) * 100, 2) if n_rows > 0 else 0,
                'unique': int(df[col].nunique())
            }
            
            # Add numeric stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None
                })
            
            column_stats.append(col_info)
        
        # Key findings
        key_findings = []
        
        if n_rows > 10000:
            key_findings.append("Large dataset - consider using sampling for interactive analysis")
        if missing_percent > 30:
            key_findings.append("High missing data percentage - data quality may be a concern")
        if duplicate_percent > 10:
            key_findings.append("Significant duplicate rows detected - consider deduplication")
        if len(df.select_dtypes(include=['object']).columns) > len(df.columns) * 0.7:
            key_findings.append("Mostly categorical data - consider encoding for analysis")
        
        result = {
            'basic_info': {
                'rows': n_rows,
                'columns': n_cols,
                'total_cells': df.size,
                'memory_mb': round(memory_mb, 2)
            },
            'data_quality': {
                'missing_values': int(missing_total),
                'missing_percent': round(missing_percent, 2),
                'duplicate_rows': int(duplicate_count),
                'duplicate_percent': round(duplicate_percent, 2)
            },
            'data_types': {str(k): int(v) for k, v in dtype_counts.to_dict().items()},
            'column_statistics': column_stats[:20],  # Limit to 20 columns
            'key_findings': key_findings,
            'analysis_readiness': {
                'score': round(100 - (missing_percent * 0.5) - (duplicate_percent * 0.3), 2),
                'level': 'High' if missing_percent < 10 and duplicate_percent < 5 else 'Medium' if missing_percent < 30 and duplicate_percent < 20 else 'Low'
            }
        }
        
        return result
    except Exception as e:
        return {'error': f'Error in summary_report: {str(e)}'}

# List of all available functions for discovery
__all__ = [
    'dataset_overview',
    'data_quality_report',
    'descriptive_statistics',
    'correlation_analysis',
    'categorical_analysis',
    'outlier_detection',
    'missing_value_analysis',
    'general_insights_analysis',
    'preprocessing_suggestions',
    'summary_report'
]