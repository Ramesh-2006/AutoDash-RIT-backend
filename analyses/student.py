import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import warnings
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========

CONFIG = {
    # Analysis thresholds
    "weak_threshold": 40,
    "strong_threshold": 75,
    "attendance_risk_threshold": 75,
    "cgpa_risk_threshold": 5.0,
    "pass_threshold": 40,
    
    # Fuzzy matching
    "fuzzy_match_threshold": 85,
    
    # Ranking
    "top_n_students": 10,
    "percentile_bins": [0, 10, 25, 50, 75, 90, 100],
    
    # Random seed for reproducibility
    "random_seed": 42,
    
    # Logging
    "log_level": logging.INFO
}

# Setup logging
logging.basicConfig(
    level=CONFIG["log_level"],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== NUMPY TO PYTHON TYPE CONVERSION UTILITY ==========

def convert_numpy_types(data):
    """
    Recursively converts numpy types in a data structure (dict, list, value)
    to native Python types for JSON serialization.
    """
    if isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    if isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
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

# ========== UTILITY FUNCTIONS ==========

def get_key_metrics(df):
    """Returns key metrics about the dataset as a dictionary"""
    df_local = df.copy()
    total_records = len(df_local)
    numeric_cols = df_local.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_local.select_dtypes(include=['object', 'category']).columns

    return {
        "total_records": total_records,
        "total_features": len(df_local.columns),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols)
    }

def get_missing_columns_message(missing_cols, matched_cols=None):
    """Returns a list of insight strings for missing columns."""
    insights = ["⚠️ Required Columns Not Found",
                "The following columns are needed for this analysis but weren't found in your data:"]
    for col in missing_cols:
        match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else ""
        insights.append(f" - {col}{match_info}")
    return insights

def fuzzy_match_column(df, target_columns):
    """
    Matches target column names to available columns in the DataFrame
    using fuzzy string matching with improved threshold.
    """
    matched = {}
    available = df.columns.tolist()
    
    for target in target_columns:
        if target in available:
            matched[target] = target
            logger.debug(f"Exact match found for '{target}'")
            continue
        
        # Case-insensitive exact match
        lower_available = {col.lower(): col for col in available}
        if target.lower() in lower_available:
            matched[target] = lower_available[target.lower()]
            logger.debug(f"Case-insensitive match found for '{target}' -> '{matched[target]}'")
            continue
            
        match, score = process.extractOne(target, available)
        if score >= CONFIG["fuzzy_match_threshold"]:
            matched[target] = match
            logger.debug(f"Fuzzy match found for '{target}' -> '{match}' (score: {score})")
        else:
            matched[target] = None
            logger.debug(f"No good match for '{target}' (best: '{match}' score: {score})")
    
    return matched

def safe_numeric_conversion(df, columns):
    """
    Safely convert columns to numeric without modifying original dataframe.
    """
    df_local = df.copy()
    for col in columns:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
    return df_local

def min_max_normalize(series):
    """
    Min-max normalize a series to [0, 1] range.
    """
    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def validate_cgpa_scale(df, cgpa_col):
    """
    Validate CGPA scale (4-point or 10-point) before analysis.
    """
    df_local = safe_numeric_conversion(df, [cgpa_col])
    max_cgpa = df_local[cgpa_col].max()
    
    if max_cgpa <= 4:
        return 4
    elif max_cgpa <= 10:
        return 10
    else:
        logger.warning(f"Unknown CGPA scale: max value {max_cgpa}")
        return None

def general_insights_analysis(df, title="General Insights Analysis"):
    """
    Show general data visualizations.
    """
    analysis_type = "general_insights_analysis"
    df_local = df.copy()
    
    try:
        visualizations = {}
        metrics = {}
        insights = [f"--- {title} ---"]

        # Key Metrics
        key_metrics = get_key_metrics(df_local)
        metrics.update(key_metrics)
        insights.append(f"Total Records: {key_metrics['total_records']}")
        insights.append(f"Total Features: {key_metrics['total_features']}")

        # Numeric columns analysis
        numeric_cols = df_local.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            selected_num_col = numeric_cols[0]
            insights.append(f"\nNumeric Features Analysis (showing first column: {selected_num_col})")

            # Histogram
            fig1 = px.histogram(df_local, x=selected_num_col,
                               title=f"Distribution of {selected_num_col}")
            visualizations["numeric_histogram"] = fig1.to_json()

            # Box Plot
            fig2 = px.box(df_local, y=selected_num_col,
                         title=f"Box Plot of {selected_num_col}")
            visualizations["numeric_boxplot"] = fig2.to_json()
        else:
            insights.append("[INFO] No numeric columns found for analysis.")

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            insights.append("\nFeature Correlations:")
            corr = df_local[numeric_cols].corr(numeric_only=True)
            fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                           title="Correlation Between Numeric Features")
            visualizations["correlation_heatmap"] = fig3.to_json()
            metrics["correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        # Categorical columns analysis
        categorical_cols = df_local.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            selected_cat_col = categorical_cols[0]
            insights.append(f"\nCategorical Features Analysis (showing first column: {selected_cat_col})")

            value_counts = df_local[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']

            fig4 = px.bar(value_counts.head(20), x='Value', y='Count',
                         title=f"Top 20 Distribution of {selected_cat_col}")
            visualizations["categorical_barchart"] = fig4.to_json()
        else:
            insights.append("[INFO] No categorical columns found for analysis.")

        logger.info(f"General insights analysis completed successfully")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "General insights generated successfully.",
            "matched_columns": {}, 
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error in general_insights_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred in general_insights_analysis: {str(e)}",
            "matched_columns": {},
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

# ========== STUDENT ANALYTICS FUNCTIONS ==========

def student_overall_performance_analysis(df):
    """Overall academic performance analysis"""
    analysis_type = "student_overall_performance_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Overall Student Performance Analysis ---"]
        
        expected = ['student_id', 'student_name', 'cgpa', 'gpa', 'total_marks', 'percentage', 'grade']
        matched = fuzzy_match_column(df_local, expected)

        found_cols = {k: v for k, v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find any performance-related columns like 'cgpa', 'gpa', or 'total_marks'.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Identify performance metric columns
        perf_cols = []
        if matched.get('cgpa'):
            perf_cols.append(matched['cgpa'])
        if matched.get('gpa'):
            perf_cols.append(matched['gpa'])
        if matched.get('total_marks'):
            perf_cols.append(matched['total_marks'])
        if matched.get('percentage'):
            perf_cols.append(matched['percentage'])

        if perf_cols:
            # Safe numeric conversion
            df_local = safe_numeric_conversion(df_local, perf_cols)
            
            primary_metric = perf_cols[0]
            valid_data = df_local[primary_metric].notna()
            df_clean = df_local[valid_data].copy()
            
            if len(df_clean) == 0:
                insights.append("No valid numeric data found for analysis.")
                return {
                    "analysis_type": analysis_type,
                    "status": "warning",
                    "message": "No valid numeric data found.",
                    "matched_columns": matched,
                    "visualizations": {},
                    "metrics": {},
                    "insights": insights
                }
            
            avg_performance = df_clean[primary_metric].mean()
            max_performance = df_clean[primary_metric].max()
            min_performance = df_clean[primary_metric].min()
            
            metrics["average_performance"] = avg_performance
            metrics["maximum_performance"] = max_performance
            metrics["minimum_performance"] = min_performance
            
            insights.append(f"Average {primary_metric}: {avg_performance:.2f}")
            insights.append(f"Highest {primary_metric}: {max_performance:.2f}")
            insights.append(f"Lowest {primary_metric}: {min_performance:.2f}")
            
            # Distribution of performance
            fig1 = px.histogram(df_clean, x=primary_metric, nbins=20,
                               title=f"Distribution of {primary_metric}")
            visualizations["performance_distribution"] = fig1.to_json()
            
            # Box plot for performance metrics
            if len(perf_cols) > 1:
                df_melted = df_clean[perf_cols].melt(var_name='Metric', value_name='Value')
                fig2 = px.box(df_melted, x='Metric', y='Value',
                             title="Comparison of Performance Metrics")
                visualizations["metrics_comparison"] = fig2.to_json()

        # Grade distribution if available
        if matched.get('grade'):
            grade_col = matched['grade']
            grade_counts = df_local[grade_col].value_counts().reset_index()
            grade_counts.columns = ['Grade', 'Count']
            
            fig3 = px.bar(grade_counts, x='Grade', y='Count',
                         title="Grade Distribution")
            visualizations["grade_distribution"] = fig3.to_json()
            
            metrics["grade_distribution"] = json.loads(grade_counts.to_json(orient="split"))

        logger.info(f"Student overall performance analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in student_overall_performance_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def subject_wise_score_analysis(df):
    """Subject marks analysis"""
    analysis_type = "subject_wise_score_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject-wise Score Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'subject_name', 'marks', 'score', 'grade']
        matched = fuzzy_match_column(df_local, expected)

        found_cols = {k: v for k, v in matched.items() if v}
        
        # Look for subject columns (could be subject names as columns or a subject column with marks)
        subject_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                       ['math', 'science', 'english', 'history', 'physics', 'chemistry', 'biology', 
                        'subject', 'marks', 'score'])]
        
        if not found_cols and not subject_cols:
            insights.append("Could not find subject or marks columns.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Case 1: Data has subject and marks columns (long format)
        if matched.get('subject') and (matched.get('marks') or matched.get('score')):
            subject_col = matched['subject']
            marks_col = matched.get('marks') or matched.get('score')
            
            df_local = safe_numeric_conversion(df_local, [marks_col])
            df_clean = df_local.dropna(subset=[subject_col, marks_col]).copy()
            
            avg_by_subject = df_clean.groupby(subject_col)[marks_col].mean().sort_values(ascending=False)
            metrics["average_marks_by_subject"] = json.loads(avg_by_subject.to_json(orient="split"))
            
            insights.append("Average Marks by Subject:")
            for subject, avg_marks in avg_by_subject.head(5).items():
                insights.append(f"  {subject}: {avg_marks:.2f}")
            
            # Box plot of marks by subject
            fig1 = px.box(df_clean, x=subject_col, y=marks_col,
                         title="Marks Distribution by Subject")
            visualizations["marks_by_subject"] = fig1.to_json()
            
            # Top performing subjects
            fig2 = px.bar(x=avg_by_subject.index, y=avg_by_subject.values,
                         title="Average Marks by Subject")
            visualizations["avg_marks_by_subject"] = fig2.to_json()
            
        # Case 2: Subjects are columns (wide format)
        elif subject_cols:
            # Only include numeric subject columns
            valid_subject_cols = []
            for col in subject_cols:
                df_local = safe_numeric_conversion(df_local, [col])
                if df_local[col].notna().any():
                    valid_subject_cols.append(col)
            
            if valid_subject_cols:
                subject_scores = {}
                for col in valid_subject_cols:
                    subject_scores[col] = df_local[col].mean()
                
                avg_df = pd.DataFrame(list(subject_scores.items()), columns=['Subject', 'Average'])
                avg_df = avg_df.sort_values('Average', ascending=False)
                
                metrics["average_scores"] = json.loads(avg_df.to_json(orient="split"))
                
                insights.append("Average Scores by Subject:")
                for _, row in avg_df.head(5).iterrows():
                    insights.append(f"  {row['Subject']}: {row['Average']:.2f}")
                
                # Box plot for all subjects
                df_melted = df_local[valid_subject_cols].melt(var_name='Subject', value_name='Marks')
                fig1 = px.box(df_melted, x='Subject', y='Marks',
                             title="Marks Distribution by Subject")
                visualizations["marks_distribution"] = fig1.to_json()
                
                # Heatmap of subject correlations
                if len(valid_subject_cols) > 1:
                    corr = df_local[valid_subject_cols].corr()
                    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                                    title="Subject Score Correlations")
                    visualizations["subject_correlations"] = fig2.to_json()
                    metrics["subject_correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        logger.info(f"Subject-wise score analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in subject_wise_score_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def attendance_percentage_analysis(df):
    """Attendance analysis"""
    analysis_type = "attendance_percentage_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance Percentage Analysis ---"]
        
        expected = ['student_id', 'student_name', 'attendance', 'attendance_percentage', 'present_days', 'total_days', 'absent_days']
        matched = fuzzy_match_column(df_local, expected)

        found_cols = {k: v for k, v in matched.items() if v}
        
        # Find attendance column
        attendance_col = None
        if matched.get('attendance'):
            attendance_col = matched['attendance']
        elif matched.get('attendance_percentage'):
            attendance_col = matched['attendance_percentage']
        else:
            # Look for columns containing attendance
            for col in df_local.columns:
                if 'attendance' in col.lower() or 'present' in col.lower():
                    attendance_col = col
                    matched['attendance'] = col
                    break

        if not attendance_col:
            insights.append("Could not find attendance-related column.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [attendance_col])
        df_clean = df_local.dropna(subset=[attendance_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid attendance data found.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid attendance data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_attendance = df_clean[attendance_col].mean()
        min_attendance = df_clean[attendance_col].min()
        max_attendance = df_clean[attendance_col].max()
        
        # Calculate attendance categories using configurable thresholds
        low_attendance_pct = (df_clean[attendance_col] < CONFIG["attendance_risk_threshold"]).mean() * 100
        moderate_attendance_pct = ((df_clean[attendance_col] >= CONFIG["attendance_risk_threshold"]) & 
                                   (df_clean[attendance_col] < 85)).mean() * 100
        good_attendance_pct = (df_clean[attendance_col] >= 85).mean() * 100

        metrics["average_attendance_percentage"] = avg_attendance
        metrics["minimum_attendance"] = min_attendance
        metrics["maximum_attendance"] = max_attendance
        metrics["low_attendance_students_percentage"] = low_attendance_pct
        metrics["moderate_attendance_students_percentage"] = moderate_attendance_pct
        metrics["good_attendance_students_percentage"] = good_attendance_pct

        insights.append(f"Average Attendance: {avg_attendance:.2f}%")
        insights.append(f"Attendance Range: {min_attendance:.2f}% - {max_attendance:.2f}%")
        insights.append(f"Students with Low Attendance (<{CONFIG['attendance_risk_threshold']}%): {low_attendance_pct:.1f}%")
        insights.append(f"Students with Good Attendance (≥85%): {good_attendance_pct:.1f}%")

        # Attendance distribution
        fig1 = px.histogram(df_clean, x=attendance_col, nbins=20,
                           title="Distribution of Attendance Percentage")
        visualizations["attendance_distribution"] = fig1.to_json()

        # Attendance categories
        categories = pd.DataFrame({
            'Category': [f'Low (<{CONFIG["attendance_risk_threshold"]}%)', 
                        f'Moderate ({CONFIG["attendance_risk_threshold"]}-85%)', 
                        'Good (≥85%)'],
            'Percentage': [low_attendance_pct, moderate_attendance_pct, good_attendance_pct]
        })
        fig2 = px.pie(categories, values='Percentage', names='Category',
                     title="Student Attendance Categories")
        visualizations["attendance_categories"] = fig2.to_json()

        logger.info(f"Attendance analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in attendance_percentage_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def internal_external_marks_comparison(df):
    """Assessment type comparison"""
    analysis_type = "internal_external_marks_comparison"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Internal vs External Marks Comparison ---"]
        
        expected = ['student_id', 'internal_marks', 'external_marks', 'internal', 'external', 'theory', 'practical', 'assignment']
        matched = fuzzy_match_column(df_local, expected)

        # Find internal and external marks columns
        internal_col = None
        external_col = None
        
        for key in ['internal_marks', 'internal']:
            if matched.get(key):
                internal_col = matched[key]
                break
        
        for key in ['external_marks', 'external', 'theory']:
            if matched.get(key):
                external_col = matched[key]
                break
        
        if not internal_col or not external_col:
            # Look for columns containing internal/external with better filtering
            for col in df_local.columns:
                col_lower = col.lower()
                if ('internal' in col_lower or 'practical' in col_lower or 'assignment' in col_lower) and not internal_col:
                    # Check if it's likely a marks column
                    df_local = safe_numeric_conversion(df_local, [col])
                    if df_local[col].notna().any():
                        internal_col = col
                if ('external' in col_lower or 'theory' in col_lower or 'final' in col_lower) and not external_col:
                    df_local = safe_numeric_conversion(df_local, [col])
                    if df_local[col].notna().any():
                        external_col = col
            
            if internal_col:
                matched['internal_marks'] = internal_col
            if external_col:
                matched['external_marks'] = external_col

        if not internal_col or not external_col:
            insights.append("Could not find both internal and external marks columns.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [internal_col, external_col])
        df_clean = df_local.dropna(subset=[internal_col, external_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid internal/external marks data found.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_internal = df_clean[internal_col].mean()
        avg_external = df_clean[external_col].mean()
        
        # Calculate performance difference
        df_clean['difference'] = df_clean[internal_col] - df_clean[external_col]
        avg_diff = df_clean['difference'].mean()
        
        # Students who perform better in internal vs external
        better_in_internal = (df_clean['difference'] > 0).sum()
        better_in_external = (df_clean['difference'] < 0).sum()
        equal_performance = (df_clean['difference'] == 0).sum()

        metrics["average_internal_marks"] = avg_internal
        metrics["average_external_marks"] = avg_external
        metrics["average_difference_internal_minus_external"] = avg_diff
        metrics["students_better_in_internal"] = int(better_in_internal)
        metrics["students_better_in_external"] = int(better_in_external)
        metrics["students_equal_performance"] = int(equal_performance)

        insights.append(f"Average Internal Marks: {avg_internal:.2f}")
        insights.append(f"Average External Marks: {avg_external:.2f}")
        insights.append(f"Average Difference (Internal - External): {avg_diff:.2f}")
        insights.append(f"Students performing better in Internals: {better_in_internal}")
        insights.append(f"Students performing better in Externals: {better_in_external}")

        # Scatter plot of internal vs external
        fig1 = px.scatter(df_clean, x=internal_col, y=external_col, trendline='ols',
                         title="Internal vs External Marks",
                         labels={internal_col: 'Internal Marks', external_col: 'External Marks'})
        
        # Add diagonal line for reference
        fig1.add_trace(go.Scatter(x=[df_clean[internal_col].min(), df_clean[internal_col].max()],
                                 y=[df_clean[internal_col].min(), df_clean[internal_col].max()],
                                 mode='lines', name='Equal Performance',
                                 line=dict(dash='dash', color='gray')))
        visualizations["internal_vs_external_scatter"] = fig1.to_json()

        # Box plot comparison
        df_melted = pd.melt(df_clean, value_vars=[internal_col, external_col],
                           var_name='Assessment Type', value_name='Marks')
        fig2 = px.box(df_melted, x='Assessment Type', y='Marks',
                     title="Internal vs External Marks Distribution")
        visualizations["assessment_comparison_box"] = fig2.to_json()
        
        # Correlation
        correlation = df_clean[internal_col].corr(df_clean[external_col])
        metrics["internal_external_correlation"] = correlation
        insights.append(f"Correlation between Internal and External Marks: {correlation:.3f}")

        logger.info(f"Internal/external marks comparison completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in internal_external_marks_comparison: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def semester_wise_performance_trend(df):
    """Performance over time"""
    analysis_type = "semester_wise_performance_trend"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Semester-wise Performance Trend Analysis ---"]
        
        expected = ['student_id', 'semester', 'sem', 'cgpa', 'gpa', 'percentage', 'year']
        matched = fuzzy_match_column(df_local, expected)

        # Find semester column
        semester_col = None
        for key in ['semester', 'sem']:
            if matched.get(key):
                semester_col = matched[key]
                break
        
        # Find performance column
        perf_col = None
        for key in ['cgpa', 'gpa', 'percentage']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not semester_col:
            # Look for semester-like columns
            for col in df_local.columns:
                if 'sem' in col.lower() or 'term' in col.lower():
                    semester_col = col
                    matched['semester'] = col
                    break

        if not perf_col:
            insights.append("Could not find semester or performance columns.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        cols_to_convert = [perf_col]
        if semester_col:
            cols_to_convert.append(semester_col)
        df_local = safe_numeric_conversion(df_local, cols_to_convert)
        
        # Drop rows with missing semester or performance
        if semester_col:
            df_clean = df_local.dropna(subset=[semester_col, perf_col]).copy()
        else:
            df_clean = df_local.dropna(subset=[perf_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid data for trend analysis.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        if not semester_col:
            insights.append("No semester column found. Cannot analyze trends over time.")
            fallback_data = general_insights_analysis(df_clean, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Semester column not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Overall trend
        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_clean.columns:
            # Multiple students, show average trend
            semester_trend = df_clean.groupby(semester_col)[perf_col].agg(['mean', 'std']).reset_index()
            semester_trend.columns = [semester_col, 'Average', 'Std_Dev']
            
            metrics["semester_trend"] = json.loads(semester_trend.to_json(orient="split"))
            
            fig1 = px.line(semester_trend, x=semester_col, y='Average',
                          error_y='Std_Dev',
                          title=f"Average {perf_col} Trend Across Semesters",
                          markers=True)
            visualizations["average_trend"] = fig1.to_json()
            
            # Individual student trends (top N students)
            if student_id_col:
                np.random.seed(CONFIG["random_seed"])
                student_ids = df_clean[student_id_col].unique()
                sample_ids = np.random.choice(student_ids, min(CONFIG["top_n_students"], len(student_ids)), replace=False)
                df_sample = df_clean[df_clean[student_id_col].isin(sample_ids)]
                
                fig2 = px.line(df_sample.sort_values(semester_col),
                              x=semester_col, y=perf_col,
                              color=student_id_col,
                              title=f"Individual Student {perf_col} Trends (Sample of {len(sample_ids)} Students)",
                              markers=True)
                visualizations["individual_trends"] = fig2.to_json()
        else:
            # Single student or aggregated data
            semester_trend = df_clean.groupby(semester_col)[perf_col].mean().reset_index()
            
            fig1 = px.line(semester_trend, x=semester_col, y=perf_col,
                          title=f"{perf_col} Trend Across Semesters",
                          markers=True)
            visualizations["performance_trend"] = fig1.to_json()

        # Calculate improvement rate
        if len(semester_trend) >= 2:
            first_sem = semester_trend.iloc[0][perf_col] if perf_col in semester_trend.columns else semester_trend.iloc[0]['Average']
            last_sem = semester_trend.iloc[-1][perf_col] if perf_col in semester_trend.columns else semester_trend.iloc[-1]['Average']
            
            improvement = last_sem - first_sem
            improvement_percentage = (improvement / first_sem) * 100 if first_sem != 0 else 0
            
            metrics["total_improvement"] = improvement
            metrics["improvement_percentage"] = improvement_percentage
            
            insights.append(f"Total Improvement from First to Last Semester: {improvement:.2f}")
            insights.append(f"Improvement Percentage: {improvement_percentage:.2f}%")

        logger.info(f"Semester-wise performance trend analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in semester_wise_performance_trend: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def weak_subject_identification(df):
    """Academic weaknesses identification"""
    analysis_type = "weak_subject_identification"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Weak Subject Identification Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'marks', 'score', 'grade', 'failed', 'status']
        matched = fuzzy_match_column(df_local, expected)

        # Use configurable threshold
        weak_threshold = CONFIG["weak_threshold"]
        
        found_subjects = False
        
        # Case 1: Long format with subject and marks columns
        if matched.get('subject') and (matched.get('marks') or matched.get('score')):
            subject_col = matched['subject']
            marks_col = matched.get('marks') or matched.get('score')
            
            df_local = safe_numeric_conversion(df_local, [marks_col])
            df_clean = df_local.dropna(subset=[subject_col, marks_col]).copy()
            
            if len(df_clean) == 0:
                insights.append("No valid subject marks data found.")
                return {
                    "analysis_type": analysis_type,
                    "status": "warning",
                    "message": "No valid data found.",
                    "matched_columns": matched,
                    "visualizations": {},
                    "metrics": {},
                    "insights": insights
                }
            
            # Identify weak subjects (below threshold)
            weak_subjects = df_clean[df_clean[marks_col] < weak_threshold].groupby(subject_col).size().reset_index(name='count')
            weak_subjects = weak_subjects.sort_values('count', ascending=False)
            
            if len(weak_subjects) > 0:
                metrics["weak_subjects"] = json.loads(weak_subjects.to_json(orient="split"))
                
                insights.append(f"Subjects where students score below {weak_threshold} marks:")
                for _, row in weak_subjects.head(5).iterrows():
                    insights.append(f"  {row[subject_col]}: {row['count']} students")
                
                # Visualize weak subjects
                fig1 = px.bar(weak_subjects.head(10), x=subject_col, y='count',
                             title=f"Top 10 Subjects Where Students Score < {weak_threshold}")
                visualizations["weak_subjects_bar"] = fig1.to_json()
                
                # Distribution of marks in weak subjects
                weak_subj_list = weak_subjects.head(5)[subject_col].tolist()
                df_weak = df_clean[df_clean[subject_col].isin(weak_subj_list)]
                
                fig2 = px.box(df_weak, x=subject_col, y=marks_col,
                             title=f"Marks Distribution in Weak Subjects")
                visualizations["weak_subjects_distribution"] = fig2.to_json()
            
            found_subjects = True
            
        # Case 2: Subject columns (wide format)
        else:
            subject_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                           ['math', 'science', 'english', 'history', 'physics', 'chemistry', 
                            'biology', 'computer', 'economics', 'accounting'])]
            
            if subject_cols:
                df_local = safe_numeric_conversion(df_local, subject_cols)
                valid_subject_cols = [col for col in subject_cols if df_local[col].notna().any()]
                
                if valid_subject_cols:
                    # Calculate average for each subject
                    subject_averages = df_local[valid_subject_cols].mean().sort_values()
                    
                    # Identify weak subjects (bottom 3 or below threshold)
                    weak_by_avg = subject_averages.head(3)
                    
                    metrics["subject_averages"] = json.loads(subject_averages.to_json(orient="split"))
                    metrics["weakest_subjects_by_avg"] = json.loads(weak_by_avg.to_json(orient="split"))
                    
                    insights.append("Weakest Subjects (by average marks):")
                    for subject, avg in weak_by_avg.items():
                        insights.append(f"  {subject}: {avg:.2f} average")
                    
                    # Students with multiple weak subjects
                    df_local['weak_subject_count'] = (df_local[valid_subject_cols] < weak_threshold).sum(axis=1)
                    students_with_multiple_weak = df_local[df_local['weak_subject_count'] >= 3].shape[0]
                    
                    metrics["students_with_3plus_weak_subjects"] = int(students_with_multiple_weak)
                    insights.append(f"Students with 3 or more weak subjects: {students_with_multiple_weak}")
                    
                    # Visualize subject averages
                    fig1 = px.bar(x=subject_averages.index, y=subject_averages.values,
                                 title="Average Marks by Subject",
                                 labels={'x': 'Subject', 'y': 'Average Marks'})
                    fig1.add_hline(y=weak_threshold, line_dash="dash", 
                                  line_color="red", annotation_text=f"Weak Threshold ({weak_threshold})")
                    visualizations["subject_averages"] = fig1.to_json()
                    
                    # Heatmap of weak subjects
                    weak_matrix = (df_local[valid_subject_cols] < weak_threshold).astype(int)
                    if weak_matrix.shape[1] > 1:
                        weak_corr = weak_matrix.corr()
                        fig2 = px.imshow(weak_corr, text_auto=True, aspect="auto",
                                        title="Correlation of Weak Subjects")
                        visualizations["weak_subject_correlation"] = fig2.to_json()
                    
                    found_subjects = True

        if not found_subjects:
            insights.append("Could not find subject or marks columns for weak subject identification.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        logger.info(f"Weak subject identification completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in weak_subject_identification: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def strong_subject_identification(df):
    """Academic strengths identification"""
    analysis_type = "strong_subject_identification"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Strong Subject Identification Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'marks', 'score', 'grade', 'rank']
        matched = fuzzy_match_column(df_local, expected)

        # Use configurable threshold
        strong_threshold = CONFIG["strong_threshold"]
        
        found_subjects = False
        
        # Case 1: Long format with subject and marks columns
        if matched.get('subject') and (matched.get('marks') or matched.get('score')):
            subject_col = matched['subject']
            marks_col = matched.get('marks') or matched.get('score')
            
            df_local = safe_numeric_conversion(df_local, [marks_col])
            df_clean = df_local.dropna(subset=[subject_col, marks_col]).copy()
            
            if len(df_clean) == 0:
                insights.append("No valid subject marks data found.")
                return {
                    "analysis_type": analysis_type,
                    "status": "warning",
                    "message": "No valid data found.",
                    "matched_columns": matched,
                    "visualizations": {},
                    "metrics": {},
                    "insights": insights
                }
            
            # Identify strong subjects (above threshold)
            strong_subjects = df_clean[df_clean[marks_col] > strong_threshold].groupby(subject_col).size().reset_index(name='count')
            strong_subjects = strong_subjects.sort_values('count', ascending=False)
            
            if len(strong_subjects) > 0:
                metrics["strong_subjects"] = json.loads(strong_subjects.to_json(orient="split"))
                
                insights.append(f"Subjects where students score above {strong_threshold} marks:")
                for _, row in strong_subjects.head(5).iterrows():
                    insights.append(f"  {row[subject_col]}: {row['count']} students")
                
                # Visualize strong subjects
                fig1 = px.bar(strong_subjects.head(10), x=subject_col, y='count',
                             title=f"Top 10 Subjects Where Students Score > {strong_threshold}")
                visualizations["strong_subjects_bar"] = fig1.to_json()
                
                # Distribution of marks in strong subjects
                strong_subj_list = strong_subjects.head(5)[subject_col].tolist()
                df_strong = df_clean[df_clean[subject_col].isin(strong_subj_list)]
                
                fig2 = px.box(df_strong, x=subject_col, y=marks_col,
                             title=f"Marks Distribution in Strong Subjects")
                visualizations["strong_subjects_distribution"] = fig2.to_json()
            
            found_subjects = True
            
        # Case 2: Subject columns (wide format)
        else:
            subject_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                           ['math', 'science', 'english', 'history', 'physics', 'chemistry', 
                            'biology', 'computer', 'economics', 'accounting'])]
            
            if subject_cols:
                df_local = safe_numeric_conversion(df_local, subject_cols)
                valid_subject_cols = [col for col in subject_cols if df_local[col].notna().any()]
                
                if valid_subject_cols:
                    # Calculate average for each subject
                    subject_averages = df_local[valid_subject_cols].mean().sort_values(ascending=False)
                    
                    # Identify strong subjects (top 3)
                    strong_by_avg = subject_averages.head(3)
                    
                    metrics["subject_averages"] = json.loads(subject_averages.to_json(orient="split"))
                    metrics["strongest_subjects_by_avg"] = json.loads(strong_by_avg.to_json(orient="split"))
                    
                    insights.append("Strongest Subjects (by average marks):")
                    for subject, avg in strong_by_avg.items():
                        insights.append(f"  {subject}: {avg:.2f} average")
                    
                    # Students excelling in multiple subjects
                    df_local['strong_subject_count'] = (df_local[valid_subject_cols] > strong_threshold).sum(axis=1)
                    top_performers = df_local[df_local['strong_subject_count'] >= 3].shape[0]
                    
                    metrics["students_with_3plus_strong_subjects"] = int(top_performers)
                    insights.append(f"Students excelling in 3 or more subjects: {top_performers}")
                    
                    # Visualize subject averages
                    fig1 = px.bar(x=subject_averages.index, y=subject_averages.values,
                                 title="Average Marks by Subject",
                                 labels={'x': 'Subject', 'y': 'Average Marks'})
                    fig1.add_hline(y=strong_threshold, line_dash="dash", 
                                  line_color="green", annotation_text=f"Strong Threshold ({strong_threshold})")
                    visualizations["subject_averages"] = fig1.to_json()
                    
                    found_subjects = True

        if not found_subjects:
            insights.append("Could not find subject or marks columns for strong subject identification.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        logger.info(f"Strong subject identification completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in strong_subject_identification: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def attendance_vs_marks_correlation(df):
    """Attendance-performance relationship"""
    analysis_type = "attendance_vs_marks_correlation"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance vs Marks Correlation Analysis ---"]
        
        expected = ['student_id', 'attendance', 'attendance_percentage', 'marks', 'total_marks', 'cgpa', 'gpa', 'percentage']
        matched = fuzzy_match_column(df_local, expected)

        # Find attendance column
        attendance_col = None
        for key in ['attendance', 'attendance_percentage']:
            if matched.get(key):
                attendance_col = matched[key]
                break
        
        if not attendance_col:
            for col in df_local.columns:
                if 'attendance' in col.lower() or 'present' in col.lower():
                    attendance_col = col
                    matched['attendance'] = col
                    break

        # Find performance column
        perf_col = None
        for key in ['marks', 'total_marks', 'cgpa', 'gpa', 'percentage']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not attendance_col or not perf_col:
            missing = []
            if not attendance_col:
                missing.append("attendance")
            if not perf_col:
                missing.append("performance")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [attendance_col, perf_col])
        df_clean = df_local.dropna(subset=[attendance_col, perf_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid attendance/marks data found.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        # Calculate correlation
        correlation = df_clean[attendance_col].corr(df_clean[perf_col])
        
        # Group attendance into categories
        attendance_bins = [0, 60, 70, 80, 90, 100]
        attendance_labels = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%']
        df_clean['attendance_category'] = pd.cut(df_clean[attendance_col], bins=attendance_bins, labels=attendance_labels)
        
        avg_marks_by_attendance = df_clean.groupby('attendance_category')[perf_col].mean().reset_index()

        metrics["attendance_marks_correlation"] = correlation
        metrics["avg_marks_by_attendance_category"] = json.loads(avg_marks_by_attendance.to_json(orient="split"))

        insights.append(f"Correlation between Attendance and {perf_col}: {correlation:.3f}")
        if correlation > 0.5:
            insights.append("Strong positive correlation: Higher attendance tends to lead to better performance.")
        elif correlation > 0.3:
            insights.append("Moderate positive correlation: Attendance positively influences performance.")
        elif correlation > 0:
            insights.append("Weak positive correlation: Attendance has some positive effect on performance.")
        else:
            insights.append("Negative or no correlation: Attendance may not strongly influence performance.")

        # Scatter plot with trendline
        fig1 = px.scatter(df_clean, x=attendance_col, y=perf_col, trendline='ols',
                         title=f"Attendance vs {perf_col}",
                         labels={attendance_col: 'Attendance (%)', perf_col: perf_col.replace('_', ' ').title()})
        visualizations["attendance_vs_marks_scatter"] = fig1.to_json()

        # Box plot by attendance category
        fig2 = px.box(df_clean, x='attendance_category', y=perf_col,
                     title=f"{perf_col} Distribution by Attendance Category")
        visualizations["marks_by_attendance_category"] = fig2.to_json()

        # Average marks by attendance
        fig3 = px.bar(avg_marks_by_attendance, x='attendance_category', y=perf_col,
                     title=f"Average {perf_col} by Attendance Category")
        visualizations["avg_marks_by_attendance"] = fig3.to_json()

        logger.info(f"Attendance vs marks correlation completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in attendance_vs_marks_correlation: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def assignment_score_analysis(df):
    """Assignment marks analysis"""
    analysis_type = "assignment_score_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Assignment Score Analysis ---"]
        
        expected = ['student_id', 'assignment', 'assignment_marks', 'assignment_score', 'submission_status', 'late_submission']
        matched = fuzzy_match_column(df_local, expected)

        # Find assignment-related columns with better filtering
        assignment_cols = []
        
        for col in df_local.columns:
            col_lower = col.lower()
            if 'assignment' in col_lower or 'assign' in col_lower or 'homework' in col_lower:
                # Check if it's numeric or can be converted
                df_local = safe_numeric_conversion(df_local, [col])
                if df_local[col].notna().any():
                    assignment_cols.append(col)
                    if 'assignment_marks' not in matched:
                        matched['assignment_marks'] = col
                    elif 'assignment_score' not in matched:
                        matched['assignment_score'] = col

        if not assignment_cols:
            insights.append("Could not find assignment-related columns with numeric data.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate assignment statistics
        assignment_stats = df_local[assignment_cols].describe()
        metrics["assignment_statistics"] = json.loads(assignment_stats.to_json(orient="split"))

        # Average assignment scores
        avg_scores = df_local[assignment_cols].mean().sort_values(ascending=False)
        
        insights.append("Assignment Score Analysis:")
        insights.append(f"  Number of assignments: {len(assignment_cols)}")
        insights.append(f"  Overall average across all assignments: {df_local[assignment_cols].mean().mean():.2f}")
        insights.append(f"  Best performed assignment: {avg_scores.index[0]} (Avg: {avg_scores.iloc[0]:.2f})")
        insights.append(f"  Worst performed assignment: {avg_scores.index[-1]} (Avg: {avg_scores.iloc[-1]:.2f})")

        # Box plot of assignment scores
        df_melted = df_local[assignment_cols].melt(var_name='Assignment', value_name='Score')
        fig1 = px.box(df_melted, x='Assignment', y='Score',
                     title="Assignment Scores Distribution")
        visualizations["assignment_scores_distribution"] = fig1.to_json()

        # Average scores bar chart
        avg_df = pd.DataFrame({'Assignment': avg_scores.index, 'Average Score': avg_scores.values})
        fig2 = px.bar(avg_df, x='Assignment', y='Average Score',
                     title="Average Score by Assignment")
        visualizations["avg_assignment_scores"] = fig2.to_json()

        # Heatmap of assignment correlations
        if len(assignment_cols) > 1:
            corr = df_local[assignment_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                            title="Assignment Score Correlations")
            visualizations["assignment_correlations"] = fig3.to_json()
            metrics["assignment_correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        # Calculate total assignment score per student if student_id exists
        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_local.columns:
            df_local['total_assignment_score'] = df_local[assignment_cols].sum(axis=1)
            student_totals = df_local.groupby(student_id_col)['total_assignment_score'].mean().reset_index()
            fig4 = px.histogram(student_totals, x='total_assignment_score',
                               title="Distribution of Total Assignment Scores")
            visualizations["total_scores_distribution"] = fig4.to_json()

        logger.info(f"Assignment score analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in assignment_score_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def cgpa_progression_analysis(df):
    """CGPA growth analysis"""
    analysis_type = "cgpa_progression_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- CGPA Progression Analysis ---"]
        
        expected = ['student_id', 'student_name', 'semester', 'sem', 'cgpa', 'gpa']
        matched = fuzzy_match_column(df_local, expected)

        # Find CGPA column
        cgpa_col = None
        for key in ['cgpa']:
            if matched.get(key):
                cgpa_col = matched[key]
                break
        
        # Find semester column
        semester_col = None
        for key in ['semester', 'sem']:
            if matched.get(key):
                semester_col = matched[key]
                break

        if not cgpa_col:
            # Look for CGPA-like columns
            for col in df_local.columns:
                if 'cgpa' in col.lower() or 'gpa' in col.lower():
                    cgpa_col = col
                    matched['cgpa'] = col
                    break

        if not cgpa_col:
            insights.append("Could not find CGPA column.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [cgpa_col])
        df_clean = df_local.dropna(subset=[cgpa_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid CGPA data found.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        # Validate CGPA scale
        cgpa_scale = validate_cgpa_scale(df_clean, cgpa_col)
        if cgpa_scale:
            insights.append(f"CGPA scale detected: {cgpa_scale}-point")

        # If semester column exists, analyze progression by semester
        if semester_col and semester_col in df_clean.columns:
            df_clean = safe_numeric_conversion(df_clean, [semester_col])
            df_clean = df_clean.dropna(subset=[semester_col]).copy()
            
            # Overall CGPA progression
            cgpa_by_semester = df_clean.groupby(semester_col)[cgpa_col].mean().reset_index()
            cgpa_by_semester = cgpa_by_semester.sort_values(semester_col)
            
            metrics["cgpa_progression"] = json.loads(cgpa_by_semester.to_json(orient="split"))
            
            insights.append("CGPA Progression Across Semesters:")
            for _, row in cgpa_by_semester.iterrows():
                insights.append(f"  Semester {int(row[semester_col])}: {row[cgpa_col]:.2f}")
            
            # Calculate growth
            if len(cgpa_by_semester) >= 2:
                start_cgpa = cgpa_by_semester.iloc[0][cgpa_col]
                end_cgpa = cgpa_by_semester.iloc[-1][cgpa_col]
                total_growth = end_cgpa - start_cgpa
                growth_percentage = (total_growth / start_cgpa) * 100 if start_cgpa > 0 else 0
                
                metrics["total_cgpa_growth"] = total_growth
                metrics["cgpa_growth_percentage"] = growth_percentage
                
                insights.append(f"Total CGPA Growth: {total_growth:+.3f}")
                insights.append(f"Growth Percentage: {growth_percentage:+.2f}%")
            
            # Line chart of CGPA progression
            fig1 = px.line(cgpa_by_semester, x=semester_col, y=cgpa_col,
                          title="Average CGPA Progression Across Semesters",
                          markers=True)
            visualizations["cgpa_progression"] = fig1.to_json()
            
            # Individual student progression (sample)
            student_id_col = matched.get('student_id')
            if student_id_col and student_id_col in df_clean.columns:
                np.random.seed(CONFIG["random_seed"])
                student_ids = df_clean[student_id_col].unique()
                sample_ids = np.random.choice(student_ids, min(10, len(student_ids)), replace=False)
                df_sample = df_clean[df_clean[student_id_col].isin(sample_ids)]
                
                fig2 = px.line(df_sample.sort_values(semester_col),
                              x=semester_col, y=cgpa_col,
                              color=student_id_col,
                              title="Individual Student CGPA Progression (Sample)",
                              markers=True)
                visualizations["individual_progression"] = fig2.to_json()
        else:
            # Just show CGPA distribution
            insights.append("No semester information found. Showing CGPA distribution only.")
            
            fig1 = px.histogram(df_clean, x=cgpa_col, nbins=20,
                               title="Distribution of CGPA")
            visualizations["cgpa_distribution"] = fig1.to_json()

        # CGPA statistics
        metrics["average_cgpa"] = df_clean[cgpa_col].mean()
        metrics["median_cgpa"] = df_clean[cgpa_col].median()
        metrics["min_cgpa"] = df_clean[cgpa_col].min()
        metrics["max_cgpa"] = df_clean[cgpa_col].max()
        metrics["std_cgpa"] = df_clean[cgpa_col].std()

        insights.append(f"CGPA Statistics:")
        insights.append(f"  Average: {metrics['average_cgpa']:.3f}")
        insights.append(f"  Median: {metrics['median_cgpa']:.3f}")
        insights.append(f"  Range: {metrics['min_cgpa']:.3f} - {metrics['max_cgpa']:.3f}")

        logger.info(f"CGPA progression analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in cgpa_progression_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def arrear_subject_detection(df):
    """Subject failures detection"""
    analysis_type = "arrear_subject_detection"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Arrear Subject Detection Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'marks', 'score', 'grade', 'result', 'status', 'failed_subjects']
        matched = fuzzy_match_column(df_local, expected)

        pass_threshold = CONFIG["pass_threshold"]
        
        found_data = False
        
        # Check for result/status column
        if matched.get('result') or matched.get('status'):
            status_col = matched.get('result') or matched.get('status')
            
            # Count failures based on status
            fail_count = df_local[status_col].astype(str).str.lower().str.contains('fail|f|arrear|backlog').sum()
            total_students = len(df_local)
            
            metrics["students_with_arrears"] = int(fail_count)
            metrics["total_students"] = total_students
            metrics["arrear_percentage"] = (fail_count / total_students * 100) if total_students > 0 else 0
            
            insights.append(f"Students with Arrears/Backlogs: {fail_count} out of {total_students} ({metrics['arrear_percentage']:.1f}%)")
            
            found_data = True
            
        # Check for failed_subjects count column
        elif matched.get('failed_subjects'):
            failed_col = matched['failed_subjects']
            df_local = safe_numeric_conversion(df_local, [failed_col])
            df_clean = df_local.dropna(subset=[failed_col]).copy()
            
            if len(df_clean) > 0:
                students_with_arrears = (df_clean[failed_col] > 0).sum()
                total_arrears = df_clean[failed_col].sum()
                
                metrics["students_with_arrears"] = int(students_with_arrears)
                metrics["total_arrear_subjects"] = int(total_arrears)
                metrics["average_arrears_per_student"] = df_clean[failed_col].mean()
                
                insights.append(f"Students with Arrears: {students_with_arrears}")
                insights.append(f"Total Arrear Subjects: {total_arrears}")
                insights.append(f"Average Arrears per Student: {df_clean[failed_col].mean():.2f}")
                
                # Distribution of arrear counts
                fig1 = px.histogram(df_clean, x=failed_col, nbins=10,
                                   title="Distribution of Arrear Subjects per Student")
                visualizations["arrear_distribution"] = fig1.to_json()
                
                found_data = True
            
        # Look for subject marks
        else:
            subject_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                           ['math', 'science', 'english', 'history', 'physics', 'chemistry'])]
            
            if subject_cols:
                df_local = safe_numeric_conversion(df_local, subject_cols)
                valid_subject_cols = [col for col in subject_cols if df_local[col].notna().any()]
                
                if valid_subject_cols:
                    # Count failures per subject
                    failures_by_subject = (df_local[valid_subject_cols] < pass_threshold).sum()
                    failures_df = pd.DataFrame({
                        'Subject': failures_by_subject.index,
                        'Failure Count': failures_by_subject.values
                    }).sort_values('Failure Count', ascending=False)
                    
                    metrics["failures_by_subject"] = json.loads(failures_df.to_json(orient="split"))
                    
                    insights.append("Failure Count by Subject:")
                    for _, row in failures_df.head(5).iterrows():
                        insights.append(f"  {row['Subject']}: {row['Failure Count']} failures")
                    
                    # Students with multiple failures
                    df_local['failure_count'] = (df_local[valid_subject_cols] < pass_threshold).sum(axis=1)
                    students_with_multiple = df_local[df_local['failure_count'] >= 2].shape[0]
                    
                    metrics["students_with_multiple_arrears"] = int(students_with_multiple)
                    insights.append(f"Students with 2 or more arrear subjects: {students_with_multiple}")
                    
                    # Bar chart of failures by subject
                    fig1 = px.bar(failures_df.head(10), x='Subject', y='Failure Count',
                                 title="Top 10 Subjects with Most Failures")
                    visualizations["failures_by_subject"] = fig1.to_json()
                    
                    # Pie chart of students with/without arrears
                    arrear_counts = pd.DataFrame({
                        'Status': ['No Arrears', 'Has Arrears'],
                        'Count': [(df_local['failure_count'] == 0).sum(), (df_local['failure_count'] > 0).sum()]
                    })
                    fig2 = px.pie(arrear_counts, values='Count', names='Status',
                                 title="Students with Arrears vs No Arrears")
                    visualizations["arrear_pie_chart"] = fig2.to_json()
                    
                    found_data = True

        if not found_data:
            insights.append("Could not find arrear or subject failure information.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        logger.info(f"Arrear subject detection completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in arrear_subject_detection: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def grade_distribution_analysis(df):
    """Grade spread analysis"""
    analysis_type = "grade_distribution_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Grade Distribution Analysis ---"]
        
        expected = ['student_id', 'grade', 'letter_grade', 'cgpa', 'percentage']
        matched = fuzzy_match_column(df_local, expected)

        # Find grade column
        grade_col = None
        for key in ['grade', 'letter_grade']:
            if matched.get(key):
                grade_col = matched[key]
                break

        if not grade_col:
            # Look for grade-like columns
            for col in df_local.columns:
                if 'grade' in col.lower():
                    grade_col = col
                    matched['grade'] = col
                    break

        if not grade_col:
            # Try to derive grades from CGPA or percentage
            perf_col = None
            for key in ['cgpa', 'percentage']:
                if matched.get(key):
                    perf_col = matched[key]
                    break
            
            if perf_col:
                df_local = safe_numeric_conversion(df_local, [perf_col])
                df_clean = df_local.dropna(subset=[perf_col]).copy()
                
                if len(df_clean) > 0:
                    # Validate CGPA scale
                    cgpa_scale = validate_cgpa_scale(df_clean, perf_col) if 'cgpa' in perf_col.lower() else None
                    
                    if 'cgpa' in perf_col.lower():
                        if cgpa_scale == 10:
                            def cgpa_to_grade(cgpa):
                                if cgpa >= 9.0:
                                    return 'A+'
                                elif cgpa >= 8.0:
                                    return 'A'
                                elif cgpa >= 7.0:
                                    return 'B+'
                                elif cgpa >= 6.0:
                                    return 'B'
                                elif cgpa >= 5.0:
                                    return 'C'
                                else:
                                    return 'F'
                        else:  # 4-point scale
                            def cgpa_to_grade(cgpa):
                                if cgpa >= 3.5:
                                    return 'A'
                                elif cgpa >= 3.0:
                                    return 'B+'
                                elif cgpa >= 2.5:
                                    return 'B'
                                elif cgpa >= 2.0:
                                    return 'C'
                                else:
                                    return 'F'
                        
                        df_clean['derived_grade'] = df_clean[perf_col].apply(cgpa_to_grade)
                    else:
                        # Assuming percentage
                        def percentage_to_grade(pct):
                            if pct >= 90:
                                return 'A+'
                            elif pct >= 80:
                                return 'A'
                            elif pct >= 70:
                                return 'B+'
                            elif pct >= 60:
                                return 'B'
                            elif pct >= 50:
                                return 'C'
                            else:
                                return 'F'
                        
                        df_clean['derived_grade'] = df_clean[perf_col].apply(percentage_to_grade)
                    
                    grade_col = 'derived_grade'
                    df_local = df_clean
                    insights.append(f"Grades derived from {perf_col} values.")
                else:
                    insights.append("Could not find grade column or derive grades.")
                    fallback_data = general_insights_analysis(df_local, "General Analysis")
                    fallback_data["analysis_type"] = analysis_type
                    fallback_data["status"] = "fallback"
                    fallback_data["message"] = "Grade column not found, returned general analysis."
                    fallback_data["matched_columns"] = matched
                    fallback_data["insights"] = insights + fallback_data["insights"]
                    return fallback_data
            else:
                insights.append("Could not find grade column or derive grades.")
                fallback_data = general_insights_analysis(df_local, "General Analysis")
                fallback_data["analysis_type"] = analysis_type
                fallback_data["status"] = "fallback"
                fallback_data["message"] = "Grade column not found, returned general analysis."
                fallback_data["matched_columns"] = matched
                fallback_data["insights"] = insights + fallback_data["insights"]
                return fallback_data

        # Calculate grade distribution
        grade_counts = df_local[grade_col].value_counts().reset_index()
        grade_counts.columns = ['Grade', 'Count']
        
        # Sort grades in logical order
        grade_order = ['A+', 'A', 'B+', 'B', 'C', 'D', 'F', 'Fail', 'Pass']
        grade_counts['Sort_Order'] = grade_counts['Grade'].apply(
            lambda x: grade_order.index(x) if x in grade_order else 999
        )
        grade_counts = grade_counts.sort_values('Sort_Order').drop('Sort_Order', axis=1)
        
        metrics["grade_distribution"] = json.loads(grade_counts.to_json(orient="split"))
        metrics["total_students"] = int(grade_counts['Count'].sum())

        insights.append("Grade Distribution:")
        for _, row in grade_counts.iterrows():
            percentage = (row['Count'] / metrics["total_students"]) * 100
            insights.append(f"  {row['Grade']}: {row['Count']} students ({percentage:.1f}%)")

        # Bar chart of grade distribution
        fig1 = px.bar(grade_counts, x='Grade', y='Count',
                     title="Grade Distribution",
                     text='Count')
        visualizations["grade_distribution_bar"] = fig1.to_json()

        # Pie chart
        fig2 = px.pie(grade_counts, values='Count', names='Grade',
                     title="Grade Distribution (Pie Chart)")
        visualizations["grade_distribution_pie"] = fig2.to_json()

        # Calculate pass percentage
        pass_grades = ['A+', 'A', 'B+', 'B', 'C', 'Pass']
        fail_grades = ['F', 'Fail']
        
        pass_count = grade_counts[grade_counts['Grade'].isin(pass_grades)]['Count'].sum()
        fail_count = grade_counts[grade_counts['Grade'].isin(fail_grades)]['Count'].sum()
        
        if pass_count + fail_count > 0:
            pass_percentage = (pass_count / (pass_count + fail_count)) * 100
            metrics["pass_percentage"] = pass_percentage
            insights.append(f"Overall Pass Percentage: {pass_percentage:.1f}%")

        logger.info(f"Grade distribution analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in grade_distribution_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def exam_attempt_performance_analysis(df):
    """Exam attempts analysis"""
    analysis_type = "exam_attempt_performance_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Exam Attempt Performance Analysis ---"]
        
        expected = ['student_id', 'exam_name', 'attempt_number', 'marks', 'score', 'improvement', 'previous_marks']
        matched = fuzzy_match_column(df_local, expected)

        # Find attempt number column
        attempt_col = None
        for key in ['attempt_number']:
            if matched.get(key):
                attempt_col = matched[key]
                break

        # Find marks column
        marks_col = None
        for key in ['marks', 'score']:
            if matched.get(key):
                marks_col = matched[key]
                break

        if not attempt_col or not marks_col:
            missing = []
            if not attempt_col:
                missing.append("attempt_number")
            if not marks_col:
                missing.append("marks/score")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [attempt_col, marks_col])
        df_clean = df_local.dropna(subset=[attempt_col, marks_col]).copy()

        if len(df_clean) == 0:
            insights.append("No valid attempt data found.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        # Performance by attempt number
        performance_by_attempt = df_clean.groupby(attempt_col)[marks_col].agg(['mean', 'std', 'count']).reset_index()
        performance_by_attempt.columns = [attempt_col, 'Average', 'Std_Dev', 'Count']
        
        metrics["performance_by_attempt"] = json.loads(performance_by_attempt.to_json(orient="split"))

        insights.append("Performance by Exam Attempt:")
        for _, row in performance_by_attempt.iterrows():
            insights.append(f"  Attempt {int(row[attempt_col])}: Avg = {row['Average']:.2f}, Students = {int(row['Count'])}")

        # Calculate improvement between attempts if student_id exists
        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_clean.columns:
            # Sort by student and attempt
            df_sorted = df_clean.sort_values([student_id_col, attempt_col])
            
            # Calculate improvement (marks - previous marks)
            df_sorted['previous_marks'] = df_sorted.groupby(student_id_col)[marks_col].shift(1)
            df_sorted['improvement'] = df_sorted[marks_col] - df_sorted['previous_marks']
            
            avg_improvement = df_sorted['improvement'].mean()
            metrics["average_improvement_between_attempts"] = avg_improvement
            
            insights.append(f"Average Improvement between Attempts: {avg_improvement:.2f}")

            # Improvement distribution
            fig3 = px.histogram(df_sorted.dropna(), x='improvement', nbins=20,
                               title="Distribution of Score Improvement Between Attempts")
            visualizations["improvement_distribution"] = fig3.to_json()

        # Box plot of marks by attempt
        fig1 = px.box(df_clean, x=attempt_col, y=marks_col,
                     title=f"Marks Distribution by Attempt Number")
        visualizations["marks_by_attempt"] = fig1.to_json()

        # Line chart of average marks by attempt
        fig2 = px.line(performance_by_attempt, x=attempt_col, y='Average',
                       error_y='Std_Dev',
                       title="Average Marks by Attempt Number",
                       markers=True)
        visualizations["avg_marks_by_attempt"] = fig2.to_json()

        # Success rate improvement
        if len(performance_by_attempt) >= 2:
            first_attempt_avg = performance_by_attempt.iloc[0]['Average']
            last_attempt_avg = performance_by_attempt.iloc[-1]['Average']
            overall_improvement = last_attempt_avg - first_attempt_avg
            
            metrics["overall_improvement"] = overall_improvement
            insights.append(f"Overall Improvement from First to Last Attempt: {overall_improvement:+.2f}")

        logger.info(f"Exam attempt performance analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in exam_attempt_performance_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def improvement_rate_analysis(df):
    """Improvement speed analysis"""
    analysis_type = "improvement_rate_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Improvement Rate Analysis ---"]
        
        expected = ['student_id', 'semester', 'cgpa', 'gpa', 'percentage', 'previous_score', 'current_score']
        matched = fuzzy_match_column(df_local, expected)

        # Find columns needed
        student_col = matched.get('student_id')
        semester_col = matched.get('semester') or matched.get('sem')
        
        # Find score column
        score_col = None
        for key in ['cgpa', 'gpa', 'percentage', 'current_score']:
            if matched.get(key):
                score_col = matched[key]
                break

        if not score_col:
            # Look for score-like columns
            for col in df_local.columns:
                if any(x in col.lower() for x in ['score', 'marks', 'cgpa', 'gpa']):
                    score_col = col
                    matched['score'] = col
                    break

        if not score_col:
            insights.append("Could not find score/performance column.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, [score_col])
        
        if semester_col and semester_col in df_local.columns:
            df_local = safe_numeric_conversion(df_local, [semester_col])
            df_clean = df_local.dropna(subset=[score_col, semester_col]).copy()
            
            if student_col and student_col in df_clean.columns:
                # Calculate improvement rate per student using groupby for efficiency
                student_improvements = []
                
                for student, student_data in df_clean.groupby(student_col):
                    student_data = student_data.sort_values(semester_col)
                    
                    if len(student_data) >= 2:
                        first_score = student_data.iloc[0][score_col]
                        last_score = student_data.iloc[-1][score_col]
                        semesters = student_data[semester_col].max() - student_data[semester_col].min()
                        
                        if semesters > 0:
                            total_improvement = last_score - first_score
                            improvement_rate = total_improvement / semesters
                            
                            student_improvements.append({
                                'student': student,
                                'first_score': first_score,
                                'last_score': last_score,
                                'total_improvement': total_improvement,
                                'improvement_rate': improvement_rate
                            })
                
                if student_improvements:
                    imp_df = pd.DataFrame(student_improvements)
                    
                    metrics["average_improvement_rate"] = imp_df['improvement_rate'].mean()
                    metrics["median_improvement_rate"] = imp_df['improvement_rate'].median()
                    metrics["fastest_improver"] = str(imp_df.loc[imp_df['improvement_rate'].idxmax(), 'student'])
                    metrics["slowest_improver"] = str(imp_df.loc[imp_df['improvement_rate'].idxmin(), 'student'])
                    
                    insights.append(f"Average Improvement Rate: {metrics['average_improvement_rate']:.3f} per semester")
                    insights.append(f"Fastest Improver Rate: {imp_df['improvement_rate'].max():.3f}")
                    insights.append(f"Slowest Improver Rate: {imp_df['improvement_rate'].min():.3f}")
                    
                    # Distribution of improvement rates
                    fig1 = px.histogram(imp_df, x='improvement_rate', nbins=20,
                                       title="Distribution of Improvement Rates")
                    visualizations["improvement_rate_distribution"] = fig1.to_json()
                    
                    # Top improvers
                    top_improvers = imp_df.nlargest(10, 'improvement_rate')[['student', 'improvement_rate']]
                    fig2 = px.bar(top_improvers, x='student', y='improvement_rate',
                                 title="Top 10 Students by Improvement Rate")
                    visualizations["top_improvers"] = fig2.to_json()
                else:
                    insights.append("Insufficient data to calculate individual improvement rates.")
            else:
                insights.append("Student ID column not found. Cannot calculate individual improvement rates.")
        else:
            # If no semester info, try to find previous/current scores
            if matched.get('previous_score') and matched.get('current_score'):
                prev_col = matched['previous_score']
                curr_col = matched['current_score']
                
                df_local = safe_numeric_conversion(df_local, [prev_col, curr_col])
                df_clean = df_local.dropna(subset=[prev_col, curr_col]).copy()
                
                if len(df_clean) > 0:
                    df_clean['improvement'] = df_clean[curr_col] - df_clean[prev_col]
                    df_clean['improvement_percentage'] = (df_clean['improvement'] / df_clean[prev_col]) * 100
                    
                    metrics["average_improvement"] = df_clean['improvement'].mean()
                    metrics["average_improvement_percentage"] = df_clean['improvement_percentage'].mean()
                    
                    insights.append(f"Average Improvement: {df_clean['improvement'].mean():.2f}")
                    insights.append(f"Average Improvement Percentage: {df_clean['improvement_percentage'].mean():.2f}%")
                    
                    # Improvement distribution
                    fig1 = px.histogram(df_clean, x='improvement', nbins=20,
                                       title="Distribution of Score Improvements")
                    visualizations["improvement_distribution"] = fig1.to_json()
                else:
                    insights.append("No valid previous/current score data found.")
            else:
                insights.append("Cannot calculate improvement rate without time dimension or before/after scores.")

        logger.info(f"Improvement rate analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in improvement_rate_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def performance_consistency_analysis(df):
    """Stability of marks analysis"""
    analysis_type = "performance_consistency_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Performance Consistency Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'semester', 'marks', 'cgpa', 'gpa']
        matched = fuzzy_match_column(df_local, expected)

        # Find student column
        student_col = matched.get('student_id')
        
        # Find performance columns
        perf_cols = []
        if matched.get('marks'):
            perf_cols.append(matched['marks'])
        if matched.get('cgpa'):
            perf_cols.append(matched['cgpa'])
        if matched.get('gpa'):
            perf_cols.append(matched['gpa'])

        # Look for subject columns if no explicit perf_cols
        if not perf_cols:
            subject_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                           ['math', 'science', 'english', 'history', 'physics', 'chemistry'])]
            if subject_cols:
                perf_cols = subject_cols

        if not perf_cols:
            insights.append("Could not find performance columns for consistency analysis.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df_local = safe_numeric_conversion(df_local, perf_cols)

        if student_col and student_col in df_local.columns:
            # Calculate consistency metrics per student using groupby
            student_consistency = []
            
            for student, student_data in df_local.groupby(student_col):
                if len(student_data) > 0:
                    if len(perf_cols) > 1:
                        # Multiple subjects - calculate across subjects
                        subject_scores = student_data[perf_cols].mean()  # Average per subject
                        if len(subject_scores) > 1 and subject_scores.mean() > 0:
                            consistency = subject_scores.std() / subject_scores.mean()
                            
                            student_consistency.append({
                                'student': student,
                                'avg_score': subject_scores.mean(),
                                'std_dev': subject_scores.std(),
                                'cv': consistency  # Coefficient of variation
                            })
                    else:
                        # Single score column - need multiple records per student
                        scores = student_data[perf_cols[0]].dropna()
                        if len(scores) >= 3 and scores.mean() > 0:  # Need at least 3 scores
                            consistency = scores.std() / scores.mean()
                            
                            student_consistency.append({
                                'student': student,
                                'avg_score': scores.mean(),
                                'std_dev': scores.std(),
                                'cv': consistency
                            })

            if student_consistency:
                cons_df = pd.DataFrame(student_consistency)
                
                metrics["average_consistency_cv"] = cons_df['cv'].mean()
                metrics["most_consistent_student"] = str(cons_df.loc[cons_df['cv'].idxmin(), 'student'])
                metrics["least_consistent_student"] = str(cons_df.loc[cons_df['cv'].idxmax(), 'student'])
                
                insights.append(f"Average Coefficient of Variation (CV): {cons_df['cv'].mean():.3f}")
                insights.append(f"Lower CV indicates more consistent performance")
                insights.append(f"Most Consistent Student CV: {cons_df['cv'].min():.3f}")
                insights.append(f"Least Consistent Student CV: {cons_df['cv'].max():.3f}")
                
                # Distribution of consistency
                fig1 = px.histogram(cons_df, x='cv', nbins=20,
                                   title="Distribution of Performance Consistency (CV)")
                visualizations["consistency_distribution"] = fig1.to_json()
                
                # Scatter plot of average vs consistency
                fig2 = px.scatter(cons_df, x='avg_score', y='cv',
                                 hover_name='student',
                                 title="Average Score vs Consistency (Lower CV = More Consistent)")
                visualizations["avg_vs_consistency"] = fig2.to_json()
            else:
                insights.append("Insufficient data per student to calculate consistency.")
        else:
            # Overall dataset consistency
            if len(perf_cols) > 1:
                # Multiple subjects - overall consistency across subjects
                overall_means = df_local[perf_cols].mean()
                if overall_means.mean() > 0:
                    overall_consistency = overall_means.std() / overall_means.mean()
                    
                    metrics["overall_subject_consistency_cv"] = overall_consistency
                    insights.append(f"Overall Consistency Across Subjects (CV): {overall_consistency:.3f}")
                    
                    # Bar chart of subject averages with error bars
                    fig1 = px.bar(x=overall_means.index, y=overall_means.values,
                                 title="Average Scores by Subject")
                    visualizations["subject_averages"] = fig1.to_json()

        logger.info(f"Performance consistency analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in performance_consistency_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def risk_of_failure_prediction_analysis(df):
    """Risk of failure analysis with proper normalization"""
    analysis_type = "risk_of_failure_prediction_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Risk of Failure Prediction Analysis ---"]
        
        expected = ['student_id', 'attendance', 'previous_marks', 'cgpa', 'failed_subjects', 'risk_level']
        matched = fuzzy_match_column(df_local, expected)

        # Define risk factors
        risk_factors = []
        factor_names = []
        
        # Attendance risk
        if matched.get('attendance'):
            att_col = matched['attendance']
            df_local = safe_numeric_conversion(df_local, [att_col])
            if df_local[att_col].notna().any():
                risk_factors.append(att_col)
                factor_names.append('Attendance')
        
        # Previous marks risk
        if matched.get('previous_marks'):
            prev_col = matched['previous_marks']
            df_local = safe_numeric_conversion(df_local, [prev_col])
            if df_local[prev_col].notna().any():
                risk_factors.append(prev_col)
                factor_names.append('Previous Marks')
        
        # CGPA risk
        if matched.get('cgpa'):
            cgpa_col = matched['cgpa']
            df_local = safe_numeric_conversion(df_local, [cgpa_col])
            if df_local[cgpa_col].notna().any():
                risk_factors.append(cgpa_col)
                factor_names.append('CGPA')
        
        # Failed subjects risk
        if matched.get('failed_subjects'):
            fail_col = matched['failed_subjects']
            df_local = safe_numeric_conversion(df_local, [fail_col])
            if df_local[fail_col].notna().any():
                risk_factors.append(fail_col)
                factor_names.append('Failed Subjects')

        if len(risk_factors) < 2:
            insights.append("Need at least 2 risk factors for meaningful analysis.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Insufficient risk factors, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Create a copy with only rows that have all risk factors
        df_risk = df_local.dropna(subset=risk_factors).copy()
        
        if len(df_risk) == 0:
            insights.append("No complete data for risk analysis.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No complete data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        # Normalize each risk factor using min-max scaling
        normalized_factors = []
        
        for i, factor in enumerate(risk_factors):
            factor_name = factor_names[i]
            
            if 'attendance' in factor_name.lower():
                # For attendance, lower is riskier (inverse relationship)
                norm_factor = 1 - min_max_normalize(df_risk[factor])
            elif 'cgpa' in factor_name.lower():
                # For CGPA, lower is riskier
                norm_factor = 1 - min_max_normalize(df_risk[factor])
            elif 'marks' in factor_name.lower():
                # For marks, lower is riskier
                norm_factor = 1 - min_max_normalize(df_risk[factor])
            else:
                # For failed subjects, higher is riskier (direct relationship)
                norm_factor = min_max_normalize(df_risk[factor])
            
            normalized_factors.append(norm_factor)

        # Calculate risk score as average of normalized factors
        df_risk['risk_score'] = np.mean(normalized_factors, axis=0)
        
        # Categorize risk levels
        def categorize_risk(score):
            if score >= 0.7:
                return 'High Risk'
            elif score >= 0.4:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        df_risk['risk_category'] = df_risk['risk_score'].apply(categorize_risk)
        
        # Risk distribution
        risk_counts = df_risk['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        metrics["risk_distribution"] = json.loads(risk_counts.to_json(orient="split"))
        metrics["average_risk_score"] = df_risk['risk_score'].mean()
        metrics["total_students_analyzed"] = len(df_risk)

        insights.append("Risk Analysis Results:")
        insights.append(f"  Average Risk Score: {df_risk['risk_score'].mean():.3f} (0=Low Risk, 1=High Risk)")
        for _, row in risk_counts.iterrows():
            percentage = (row['Count'] / len(df_risk)) * 100
            insights.append(f"  {row['Risk Level']}: {row['Count']} students ({percentage:.1f}%)")

        # Pie chart of risk categories
        fig1 = px.pie(risk_counts, values='Count', names='Risk Level',
                     title="Student Risk Categories")
        visualizations["risk_categories_pie"] = fig1.to_json()

        # Risk score distribution
        fig2 = px.histogram(df_risk, x='risk_score', nbins=20,
                           title="Distribution of Risk Scores")
        visualizations["risk_score_distribution"] = fig2.to_json()

        # Scatter matrix of risk factors
        if len(risk_factors) >= 2:
            fig3 = px.scatter_matrix(df_risk[risk_factors + ['risk_score']],
                                     dimensions=risk_factors,
                                     color='risk_score',
                                     title="Risk Factors Relationships")
            visualizations["risk_factors_matrix"] = fig3.to_json()

        # List high-risk students if student_id exists
        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_risk.columns:
            high_risk = df_risk[df_risk['risk_category'] == 'High Risk'][[student_id_col, 'risk_score']].head(10)
            if len(high_risk) > 0:
                insights.append("\nTop 10 High-Risk Students:")
                for _, row in high_risk.iterrows():
                    insights.append(f"  {row[student_id_col]}: Risk Score {row['risk_score']:.3f}")

        logger.info(f"Risk analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in risk_of_failure_prediction_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def attendance_irregularity_detection(df):
    """Attendance patterns detection"""
    analysis_type = "attendance_irregularity_detection"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance Irregularity Detection Analysis ---"]
        
        expected = ['student_id', 'date', 'attendance_status', 'present', 'absent', 'late']
        matched = fuzzy_match_column(df_local, expected)

        # Find attendance columns
        date_col = matched.get('date')
        status_col = None
        for key in ['attendance_status', 'present', 'absent']:
            if matched.get(key):
                status_col = matched[key]
                break

        if not status_col:
            # Look for attendance status columns
            for col in df_local.columns:
                if any(x in col.lower() for x in ['present', 'absent', 'status', 'attendance']):
                    status_col = col
                    matched['attendance_status'] = col
                    break

        if not status_col:
            insights.append("Could not find attendance status column.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert status to binary (1 for present, 0 for absent)
        status_str = df_local[status_col].astype(str).str.lower()
        df_local['is_present'] = status_str.str.contains('present|p|1|true|yes').astype(int)
        df_local['is_absent'] = status_str.str.contains('absent|a|0|false|no').astype(int)
        df_local['is_late'] = status_str.str.contains('late|l').astype(int)

        total_records = len(df_local)
        present_count = df_local['is_present'].sum()
        absent_count = df_local['is_absent'].sum()
        late_count = df_local['is_late'].sum()

        metrics["total_attendance_records"] = total_records
        metrics["present_count"] = int(present_count)
        metrics["absent_count"] = int(absent_count)
        metrics["late_count"] = int(late_count)
        
        if total_records > 0:
            metrics["attendance_rate"] = (present_count / total_records) * 100
            metrics["absenteeism_rate"] = (absent_count / total_records) * 100
            metrics["late_percentage"] = (late_count / total_records) * 100

        insights.append("Attendance Overview:")
        insights.append(f"  Present: {present_count} ({metrics.get('attendance_rate', 0):.1f}%)")
        insights.append(f"  Absent: {absent_count} ({metrics.get('absenteeism_rate', 0):.1f}%)")
        insights.append(f"  Late: {late_count} ({metrics.get('late_percentage', 0):.1f}%)")

        # Detect irregular patterns if date column exists
        if date_col and date_col in df_local.columns:
            df_local[date_col] = pd.to_datetime(df_local[date_col], errors='coerce')
            df_clean = df_local.dropna(subset=[date_col]).copy()
            
            if len(df_clean) > 0:
                # Sort by date
                df_clean = df_clean.sort_values(date_col)
                
                # Weekly attendance patterns
                df_clean['week'] = df_clean[date_col].dt.isocalendar().week
                weekly_attendance = df_clean.groupby('week')['is_present'].mean().reset_index()
                
                fig1 = px.line(weekly_attendance, x='week', y='is_present',
                              title="Weekly Attendance Rate Trend",
                              labels={'is_present': 'Attendance Rate', 'week': 'Week Number'})
                visualizations["weekly_attendance_trend"] = fig1.to_json()

                # Day of week patterns
                df_clean['day_of_week'] = df_clean[date_col].dt.day_name()
                dow_attendance = df_clean.groupby('day_of_week')['is_present'].mean().reset_index()
                
                # Sort days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_attendance['day_order'] = dow_attendance['day_of_week'].apply(
                    lambda x: day_order.index(x) if x in day_order else 999
                )
                dow_attendance = dow_attendance.sort_values('day_order').drop('day_order', axis=1)
                
                fig2 = px.bar(dow_attendance, x='day_of_week', y='is_present',
                             title="Attendance Rate by Day of Week")
                visualizations["attendance_by_day"] = fig2.to_json()

        # Student-wise irregularity detection using groupby
        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_local.columns:
            student_stats = df_local.groupby(student_id_col).agg({
                'is_present': 'mean',
                'is_absent': 'mean',
                'is_late': 'mean',
                status_col: 'count'
            }).reset_index()
            
            student_stats.columns = [student_id_col, 'attendance_rate', 'absent_rate', 'late_rate', 'total_records']
            
            # Identify students with irregular attendance
            low_attendance = student_stats[student_stats['attendance_rate'] < 0.75]
            high_absenteeism = student_stats[student_stats['absent_rate'] > 0.2]
            high_lateness = student_stats[student_stats['late_rate'] > 0.1]
            
            metrics["students_with_low_attendance"] = len(low_attendance)
            metrics["students_with_high_absenteeism"] = len(high_absenteeism)
            metrics["students_with_high_lateness"] = len(high_lateness)
            
            insights.append("\nAttendance Irregularities:")
            insights.append(f"  Students with low attendance (<75%): {len(low_attendance)}")
            insights.append(f"  Students with high absenteeism (>20%): {len(high_absenteeism)}")
            insights.append(f"  Students with high lateness (>10%): {len(high_lateness)}")

            # Distribution of attendance rates
            fig3 = px.histogram(student_stats, x='attendance_rate', nbins=20,
                               title="Distribution of Student Attendance Rates")
            visualizations["student_attendance_distribution"] = fig3.to_json()

        logger.info(f"Attendance irregularity detection completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in attendance_irregularity_detection: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def subject_difficulty_impact_analysis(df):
    """Subject difficulty analysis"""
    analysis_type = "subject_difficulty_impact_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject Difficulty Impact Analysis ---"]
        
        expected = ['subject', 'subject_name', 'average_marks', 'pass_rate', 'failure_rate', 'difficulty_score']
        matched = fuzzy_match_column(df_local, expected)

        # Find subject columns
        subject_col = matched.get('subject') or matched.get('subject_name')
        
        # Find marks columns
        marks_cols = [col for col in df_local.columns if any(x in col.lower() for x in 
                      ['marks', 'score', 'grade'])]

        if not subject_col and not marks_cols:
            insights.append("Could not find subject or marks columns for difficulty analysis.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # If we have subject column and marks
        if subject_col and subject_col in df_local.columns and marks_cols:
            marks_col = marks_cols[0]  # Use first marks column
            df_local = safe_numeric_conversion(df_local, [marks_col])
            df_clean = df_local.dropna(subset=[subject_col, marks_col]).copy()
            
            if len(df_clean) > 0:
                # Calculate subject statistics
                subject_stats = df_clean.groupby(subject_col)[marks_col].agg(['mean', 'std', 'count']).reset_index()
                subject_stats.columns = [subject_col, 'avg_marks', 'std_dev', 'student_count']
                
                # Calculate pass/fail rates (using configurable threshold)
                pass_threshold = CONFIG["pass_threshold"]
                subject_pass_rates = df_clean.groupby(subject_col).apply(
                    lambda x: (x[marks_col] >= pass_threshold).mean() * 100
                ).reset_index(name='pass_rate')
                
                subject_stats = subject_stats.merge(subject_pass_rates, on=subject_col)
                
                # Calculate difficulty score (inverse of avg_marks, normalized)
                max_avg = subject_stats['avg_marks'].max()
                min_avg = subject_stats['avg_marks'].min()
                
                if max_avg > min_avg:
                    subject_stats['difficulty_score'] = 1 - ((subject_stats['avg_marks'] - min_avg) / (max_avg - min_avg))
                else:
                    subject_stats['difficulty_score'] = 0.5
                
                # Sort by difficulty
                subject_stats = subject_stats.sort_values('difficulty_score', ascending=False)
                
                metrics["subject_difficulty_ranking"] = json.loads(subject_stats.to_json(orient="split"))

                insights.append("Subject Difficulty Ranking (Higher Score = More Difficult):")
                for _, row in subject_stats.head(5).iterrows():
                    insights.append(f"  {row[subject_col]}: Difficulty {row['difficulty_score']:.3f}, Avg Marks {row['avg_marks']:.1f}, Pass Rate {row['pass_rate']:.1f}%")

                # Bar chart of difficulty scores
                fig1 = px.bar(subject_stats, x=subject_col, y='difficulty_score',
                             title="Subject Difficulty Scores",
                             color='difficulty_score', color_continuous_scale='RdYlGn_r')
                visualizations["subject_difficulty"] = fig1.to_json()

                # Scatter plot of avg marks vs pass rate
                fig2 = px.scatter(subject_stats, x='avg_marks', y='pass_rate',
                                 text=subject_col,
                                 title="Subject Performance: Average Marks vs Pass Rate")
                visualizations["marks_vs_passrate"] = fig2.to_json()

        # If we have subject columns (multiple subjects as columns)
        elif marks_cols and len(marks_cols) > 1:
            subject_cols = marks_cols
            df_local = safe_numeric_conversion(df_local, subject_cols)
            valid_subject_cols = [col for col in subject_cols if df_local[col].notna().any()]
            
            if len(valid_subject_cols) > 0:
                # Calculate subject averages
                subject_averages = df_local[valid_subject_cols].mean().sort_values()
                
                # Calculate difficulty based on average (lower average = more difficult)
                max_avg = subject_averages.max()
                min_avg = subject_averages.min()
                
                if max_avg > min_avg:
                    difficulty_scores = 1 - ((subject_averages - min_avg) / (max_avg - min_avg))
                else:
                    difficulty_scores = pd.Series([0.5] * len(subject_averages), index=subject_averages.index)
                
                difficulty_df = pd.DataFrame({
                    'Subject': subject_averages.index,
                    'Average Marks': subject_averages.values,
                    'Difficulty Score': difficulty_scores.values
                }).sort_values('Difficulty Score', ascending=False)
                
                metrics["subject_difficulty"] = json.loads(difficulty_df.to_json(orient="split"))

                insights.append("Subject Difficulty Analysis (Based on Average Marks):")
                for _, row in difficulty_df.head(5).iterrows():
                    insights.append(f"  {row['Subject']}: Avg {row['Average Marks']:.1f}, Difficulty {row['Difficulty Score']:.3f}")

                # Bar chart of difficulty
                fig1 = px.bar(difficulty_df, x='Subject', y='Difficulty Score',
                             title="Subject Difficulty Scores",
                             color='Difficulty Score', color_continuous_scale='RdYlGn_r')
                visualizations["difficulty_scores"] = fig1.to_json()

                # Box plot of subject marks
                df_melted = df_local[valid_subject_cols].melt(var_name='Subject', value_name='Marks')
                fig2 = px.box(df_melted, x='Subject', y='Marks',
                             title="Marks Distribution by Subject")
                visualizations["marks_distribution"] = fig2.to_json()

        logger.info(f"Subject difficulty analysis completed with {len(visualizations)} visualizations")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in subject_difficulty_impact_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def rank_position_analysis(df):
    """Academic ranking analysis with fixed percentile calculation"""
    analysis_type = "rank_position_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Academic Rank Position Analysis ---"]
        
        expected = ['student_id', 'student_name', 'cgpa', 'gpa', 'total_marks', 'percentage', 'rank', 'position']
        matched = fuzzy_match_column(df_local, expected)

        # Find performance column for ranking
        perf_col = None
        for key in ['cgpa', 'gpa', 'total_marks', 'percentage']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not perf_col:
            # Look for numeric columns that could be used for ranking
            numeric_cols = df_local.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                perf_col = numeric_cols[0]  # Use first numeric column
                matched['performance'] = perf_col
                insights.append(f"Using '{perf_col}' for ranking analysis.")

        if not perf_col:
            insights.append("Could not find performance column for ranking.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Safe conversion and drop rows with missing performance data
        df_local = safe_numeric_conversion(df_local, [perf_col])
        df_rank = df_local.dropna(subset=[perf_col]).copy()
        
        total_students = len(df_rank)
        
        # Debug print to verify
        logger.info(f"Total rows in original: {len(df_local)}")
        logger.info(f"Rows with valid performance: {total_students}")

        if total_students == 0:
            insights.append("No valid performance data found for ranking.")
            return {
                "analysis_type": analysis_type,
                "status": "warning",
                "message": "No valid data found.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        # Calculate ranks (higher value = better rank, so descending sort)
        df_rank['rank'] = df_rank[perf_col].rank(method='dense', ascending=False).astype(int)
        
        # FIXED: Calculate percentile with ascending=False to match rank direction
        # Higher performance = higher percentile
        df_rank['percentile'] = df_rank[perf_col].rank(pct=True, ascending=False) * 100
        
        # Alternative method if above doesn't work:
        # df_rank['percentile'] = 100 - (df_rank[perf_col].rank(pct=True) * 100)
        
        # Top and bottom performers
        top_n = min(CONFIG["top_n_students"], total_students)
        top_students = df_rank.nsmallest(top_n, 'rank')  # rank 1 to top_n
        bottom_n = min(10, total_students)
        bottom_students = df_rank.nlargest(bottom_n, 'rank')  # lowest ranks
        
        metrics["total_students_ranked"] = total_students
        metrics["top_performer_score"] = top_students.iloc[0][perf_col]
        metrics["bottom_performer_score"] = bottom_students.iloc[-1][perf_col] if len(bottom_students) > 0 else None

        student_id_col = matched.get('student_id')
        if student_id_col and student_id_col in df_rank.columns:
            metrics["top_performer"] = str(top_students.iloc[0][student_id_col])

        insights.append(f"Ranking Analysis (based on {perf_col}):")
        insights.append(f"  Total Students Ranked: {total_students}")
        if metrics.get("top_performer"):
            insights.append(f"  Top Performer: {metrics['top_performer']} with {metrics['top_performer_score']:.2f}")
        else:
            insights.append(f"  Top Performer Score: {metrics['top_performer_score']:.2f}")
        
        # Rank distribution using configurable bins
        rank_bins = CONFIG["percentile_bins"]
        rank_labels = ['Top 10%', '10-25%', '25-50%', '50-75%', '75-90%', 'Bottom 10%']
        df_rank['rank_group'] = pd.cut(df_rank['percentile'], bins=rank_bins, labels=rank_labels, include_lowest=True)
        
        rank_distribution = df_rank['rank_group'].value_counts().reindex(rank_labels).reset_index()
        rank_distribution.columns = ['Rank Group', 'Count']
        rank_distribution = rank_distribution.fillna(0)
        
        metrics["rank_distribution"] = json.loads(rank_distribution.to_json(orient="split"))

        # Bar chart of rank distribution
        fig1 = px.bar(rank_distribution, x='Rank Group', y='Count',
                     title="Distribution of Students by Rank Group")
        visualizations["rank_distribution"] = fig1.to_json()

        # Score distribution with percentile markers
        fig2 = px.histogram(df_rank, x=perf_col, nbins=30,
                           title=f"Distribution of {perf_col} with Percentile Markers")
        
        # Add percentile lines
        for p in [10, 25, 50, 75, 90]:
            percentile_val = df_rank[perf_col].quantile(p/100)
            fig2.add_vline(x=percentile_val, line_dash="dash",
                          annotation_text=f"{p}th percentile")
        visualizations["score_distribution"] = fig2.to_json()

        # Top N list if student_id exists
        if student_id_col and student_id_col in df_rank.columns:
            top_list = top_students[[student_id_col, perf_col, 'percentile', 'rank']].copy()
            insights.append(f"\nTop {top_n} Students:")
            for idx, row in top_list.iterrows():
                insights.append(f"  Rank {int(row['rank'])}: {row[student_id_col]} - {row[perf_col]:.2f} ({row['percentile']:.1f}th percentile)")

            # Scatter plot of rank vs score
            fig3 = px.scatter(df_rank, x='rank', y=perf_col,
                             hover_name=student_id_col,
                             title="Rank vs Score Relationship")
            visualizations["rank_vs_score"] = fig3.to_json()

        logger.info(f"Rank analysis completed with {total_students} students ranked")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in rank_position_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def personalized_performance_recommendation_analysis(df):
    """Personalized performance recommendations for students with fixed random seed"""
    analysis_type = "personalized_performance_recommendation_analysis"
    df_local = df.copy()
    matched = {}
    
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Personalized Performance Recommendation Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'marks', 'score', 'cgpa', 'attendance', 'assignment_score', 
                   'internal_marks', 'external_marks', 'grade', 'strengths', 'weaknesses', 'learning_style']
        matched = fuzzy_match_column(df_local, expected)

        # Set random seed for reproducibility
        np.random.seed(CONFIG["random_seed"])

        # Find student identifier
        student_col = None
        for key in ['student_id', 'student_name']:
            if matched.get(key):
                student_col = matched[key]
                break

        if not student_col:
            insights.append("Could not find student identifier column.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Define performance thresholds
        thresholds = {
            'excellent': 80,
            'good': 70,
            'average': 60,
            'needs_improvement': 50,
            'critical': 40
        }

        # Collect all performance indicators
        performance_indicators = {}
        
        # Check for marks/score columns
        if matched.get('marks'):
            marks_col = matched['marks']
            df_local = safe_numeric_conversion(df_local, [marks_col])
            performance_indicators['marks'] = marks_col

        if matched.get('cgpa'):
            cgpa_col = matched['cgpa']
            df_local = safe_numeric_conversion(df_local, [cgpa_col])
            performance_indicators['cgpa'] = cgpa_col

        if matched.get('attendance'):
            att_col = matched['attendance']
            df_local = safe_numeric_conversion(df_local, [att_col])
            performance_indicators['attendance'] = att_col

        if matched.get('internal_marks'):
            int_col = matched['internal_marks']
            df_local = safe_numeric_conversion(df_local, [int_col])
            performance_indicators['internal'] = int_col

        if matched.get('external_marks'):
            ext_col = matched['external_marks']
            df_local = safe_numeric_conversion(df_local, [ext_col])
            performance_indicators['external'] = ext_col

        if not performance_indicators:
            insights.append("Could not find performance indicators for recommendations.")
            fallback_data = general_insights_analysis(df_local, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Subject-wise analysis (if subject column exists)
        subject_col = matched.get('subject')
        subject_marks_col = performance_indicators.get('marks')

        # Generate recommendations for each student
        student_recommendations = []
        recommendation_templates = {
            'excellent': [
                "🎯 You're performing excellently! Consider taking advanced courses or mentoring peers.",
                "📚 Your performance is outstanding. Explore research opportunities in your strong subjects.",
                "🌟 Excellent work! Consider participating in academic competitions or publishing papers.",
                "💡 You have mastered the fundamentals. Start exploring advanced topics and specializations.",
                "🏆 Top performer! Consider applying for scholarships and leadership programs."
            ],
            'good': [
                "📈 You're doing well! Focus on your weaker areas to reach excellence.",
                "🎓 Good performance! Try to participate in group studies to reinforce concepts.",
                "💪 You have a solid foundation. Practice more complex problems to improve further.",
                "📊 Good work! Identify patterns in your mistakes and work on them systematically.",
                "⭐ You're on the right track. Set higher targets for yourself in each subject."
            ],
            'average': [
                "📝 You're performing at an average level. Create a structured study plan with daily goals.",
                "⏰ Consider increasing your study time by 1-2 hours daily and focus on conceptual clarity.",
                "📚 Form a study group with better-performing students to improve understanding.",
                "🎯 Set specific, measurable goals for each subject and track your progress weekly.",
                "💡 Focus on understanding fundamentals before moving to advanced topics."
            ],
            'needs_improvement': [
                "⚠️ Your performance needs attention. Start with identifying your weak areas.",
                "📞 Consider seeking help from teachers during office hours or joining remedial classes.",
                "📝 Create a revision schedule and practice previous years' question papers.",
                "🎯 Focus on passing marks first, then gradually aim for higher scores.",
                "💪 Don't get discouraged. Small, consistent efforts will show improvement."
            ],
            'critical': [
                "🚨 Immediate attention needed! Please meet with your academic advisor this week.",
                "📞 Schedule a meeting with each subject teacher to discuss improvement strategies.",
                "📝 Focus on core concepts and essential topics first. Master the basics.",
                "🎯 Set realistic short-term goals. Aim to improve by 5-10% in the next assessment.",
                "💪 Consider peer tutoring or specialized coaching for challenging subjects."
            ],
            'attendance_warning': [
                "📅 Your attendance is below 75%. Regular attendance is crucial for understanding concepts.",
                "⚠️ Low attendance affects your performance. Make it a priority to attend all classes.",
                "📝 Missing classes creates knowledge gaps. Catch up on missed topics immediately.",
                "🎯 Set a goal to maintain 85%+ attendance for the remaining semester."
            ],
            'internal_external_gap': [
                "📊 You perform better in internals than externals. Practice writing answers under time constraints.",
                "📝 Your external exam performance needs improvement. Take more mock tests.",
                "🎯 Focus on exam techniques: time management, answer presentation, and revision.",
                "💡 Create concise notes for quick revision before exams."
            ],
            'assignment_focus': [
                "📚 Your assignment scores can be improved. Submit work on time and follow guidelines.",
                "📝 Assignments carry important weightage. Treat them as learning opportunities.",
                "🎯 Break down assignments into smaller tasks and start early."
            ]
        }

        # Process each student
        for student in df_local[student_col].unique():
            student_data = df_local[df_local[student_col] == student].copy()
            
            student_rec = {
                'student_id': str(student),
                'performance_metrics': {},
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'priority_level': 'Normal'
            }

            # Overall performance assessment
            if 'marks' in performance_indicators:
                avg_marks = student_data[performance_indicators['marks']].mean()
                student_rec['performance_metrics']['average_marks'] = round(avg_marks, 2)
                
                # Categorize performance
                if avg_marks >= thresholds['excellent']:
                    student_rec['performance_level'] = 'Excellent'
                    student_rec['priority_level'] = 'Low'
                    student_rec['recommendations'].extend(np.random.choice(recommendation_templates['excellent'], 2, replace=False).tolist())
                elif avg_marks >= thresholds['good']:
                    student_rec['performance_level'] = 'Good'
                    student_rec['priority_level'] = 'Low'
                    student_rec['recommendations'].extend(np.random.choice(recommendation_templates['good'], 2, replace=False).tolist())
                elif avg_marks >= thresholds['average']:
                    student_rec['performance_level'] = 'Average'
                    student_rec['priority_level'] = 'Medium'
                    student_rec['recommendations'].extend(np.random.choice(recommendation_templates['average'], 2, replace=False).tolist())
                elif avg_marks >= thresholds['needs_improvement']:
                    student_rec['performance_level'] = 'Needs Improvement'
                    student_rec['priority_level'] = 'High'
                    student_rec['recommendations'].extend(np.random.choice(recommendation_templates['needs_improvement'], 2, replace=False).tolist())
                else:
                    student_rec['performance_level'] = 'Critical'
                    student_rec['priority_level'] = 'Urgent'
                    student_rec['recommendations'].extend(np.random.choice(recommendation_templates['critical'], 2, replace=False).tolist())

            # Attendance analysis
            if 'attendance' in performance_indicators:
                attendance = student_data[performance_indicators['attendance']].mean()
                student_rec['performance_metrics']['attendance'] = round(attendance, 2)
                
                if attendance < CONFIG["attendance_risk_threshold"]:
                    student_rec['weaknesses'].append('Low Attendance')
                    student_rec['recommendations'].append(np.random.choice(recommendation_templates['attendance_warning']))
                    if student_rec['priority_level'] == 'Low':
                        student_rec['priority_level'] = 'Medium'

            # Internal vs External analysis
            if 'internal' in performance_indicators and 'external' in performance_indicators:
                internal_avg = student_data[performance_indicators['internal']].mean()
                external_avg = student_data[performance_indicators['external']].mean()
                
                student_rec['performance_metrics']['internal_avg'] = round(internal_avg, 2)
                student_rec['performance_metrics']['external_avg'] = round(external_avg, 2)
                
                gap = internal_avg - external_avg
                if gap > 10:
                    student_rec['weaknesses'].append('External Exam Performance')
                    student_rec['recommendations'].append(np.random.choice(recommendation_templates['internal_external_gap']))
                elif gap < -10:
                    student_rec['strengths'].append('Strong in External Exams')
                    student_rec['recommendations'].append("You perform well in externals. Focus on maintaining consistency in internals too.")

            # Subject-wise analysis
            if subject_col and 'marks' in performance_indicators and subject_col in student_data.columns:
                subject_performance = student_data.groupby(subject_col)[performance_indicators['marks']].mean().to_dict()
                
                # Identify strong and weak subjects
                if subject_performance:
                    sorted_subjects = sorted(subject_performance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top 2 subjects (strengths)
                    for subject, score in sorted_subjects[:2]:
                        if score >= thresholds['good']:
                            student_rec['strengths'].append(f"{subject}: {score:.1f}")
                    
                    # Bottom 2 subjects (weaknesses)
                    for subject, score in sorted_subjects[-2:]:
                        if score < thresholds['average']:
                            student_rec['weaknesses'].append(f"{subject}: {score:.1f}")
                    
                    # Subject-specific recommendations
                    for subject, score in sorted_subjects:
                        if score < thresholds['needs_improvement']:
                            student_rec['recommendations'].append(f"📚 Focus on improving {subject}. Start with basic concepts and practice regularly.")
                        elif score >= thresholds['excellent']:
                            student_rec['recommendations'].append(f"🎓 You excel in {subject}. Consider helping classmates or exploring advanced topics.")

            # Assignment score analysis
            if matched.get('assignment_score'):
                assign_col = matched['assignment_score']
                df_local = safe_numeric_conversion(df_local, [assign_col])
                assign_avg = student_data[assign_col].mean()
                
                if assign_avg < 60:
                    student_rec['weaknesses'].append('Assignment Scores')
                    student_rec['recommendations'].append(np.random.choice(recommendation_templates['assignment_focus']))

            # CGPA trend (if available)
            if 'cgpa' in performance_indicators and len(student_data) > 1:
                cgpa_values = student_data[performance_indicators['cgpa']].values
                if len(cgpa_values) >= 2:
                    trend = cgpa_values[-1] - cgpa_values[0]
                    if trend > 0.5:
                        student_rec['strengths'].append('Improving CGPA Trend')
                        student_rec['recommendations'].append("📈 Your CGPA is improving! Keep up the good work and maintain consistency.")
                    elif trend < -0.5:
                        student_rec['weaknesses'].append('Declining CGPA Trend')
                        student_rec['recommendations'].append("📉 Your CGPA is declining. Identify what changed and address those issues immediately.")

            # Add general recommendations based on performance level
            if len(student_rec['recommendations']) < 3:
                if student_rec.get('performance_level') == 'Excellent':
                    student_rec['recommendations'].append("🎯 Consider applying for research internships or advanced certification courses.")
                elif student_rec.get('performance_level') == 'Good':
                    student_rec['recommendations'].append("📊 Track your performance pattern and focus on turning weaknesses into strengths.")
                elif student_rec.get('performance_level') == 'Average':
                    student_rec['recommendations'].append("📝 Create a weekly study schedule allocating more time to challenging subjects.")
                else:
                    student_rec['recommendations'].append("💪 Remember: Every expert was once a beginner. Stay consistent and seek help when needed.")

            student_recommendations.append(student_rec)

        # Create recommendations DataFrame
        recommendations_df = pd.DataFrame(student_recommendations)
        
        # Summary statistics
        priority_counts = recommendations_df['priority_level'].value_counts().to_dict()
        performance_levels = recommendations_df['performance_level'].value_counts().to_dict() if 'performance_level' in recommendations_df.columns else {}

        metrics["total_students_analyzed"] = len(student_recommendations)
        metrics["priority_distribution"] = priority_counts
        metrics["performance_levels"] = performance_levels
        
        # Students needing urgent attention
        urgent_students = recommendations_df[recommendations_df['priority_level'] == 'Urgent']
        high_priority_students = recommendations_df[recommendations_df['priority_level'] == 'High']

        metrics["urgent_action_students"] = len(urgent_students)
        metrics["high_priority_students"] = len(high_priority_students)

        insights.append(f"✅ Personalized recommendations generated for {len(student_recommendations)} students")
        insights.append(f"\n📊 Priority Distribution:")
        for priority, count in priority_counts.items():
            insights.append(f"   {priority}: {count} students ({count/len(student_recommendations)*100:.1f}%)")

        if 'performance_levels' in locals() and performance_levels:
            insights.append(f"\n📈 Performance Levels:")
            for level, count in performance_levels.items():
                insights.append(f"   {level}: {count} students")

        if len(urgent_students) > 0:
            insights.append(f"\n🚨 {len(urgent_students)} students require URGENT attention!")
            for _, student in urgent_students.head(5).iterrows():
                insights.append(f"   - {student['student_id']}: {student.get('performance_level', 'Critical')}")

        if len(high_priority_students) > 0:
            insights.append(f"\n⚠️ {len(high_priority_students)} students are high priority for intervention")

        # Visualizations
        # Priority distribution pie chart
        if priority_counts:
            priority_df = pd.DataFrame(list(priority_counts.items()), columns=['Priority', 'Count'])
            fig1 = px.pie(priority_df, values='Count', names='Priority',
                         title="Student Priority Distribution",
                         color='Priority',
                         color_discrete_map={
                             'Urgent': 'red',
                             'High': 'orange',
                             'Medium': 'yellow',
                             'Low': 'green',
                             'Normal': 'blue'
                         })
            visualizations["priority_distribution"] = fig1.to_json()

        # Performance levels distribution
        if performance_levels:
            level_df = pd.DataFrame(list(performance_levels.items()), columns=['Level', 'Count'])
            fig2 = px.bar(level_df, x='Level', y='Count',
                         title="Student Performance Levels",
                         color='Count', color_continuous_scale='RdYlGn',
                         text='Count')
            visualizations["performance_levels"] = fig2.to_json()

        # Sample recommendations table
        sample_recs = recommendations_df[['student_id', 'performance_level', 'priority_level', 'recommendations']].head(10).copy()
        sample_recs['recommendations'] = sample_recs['recommendations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
        
        fig3 = go.Figure(data=[go.Table(
            header=dict(values=['Student ID', 'Performance Level', 'Priority', 'Sample Recommendation'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[sample_recs['student_id'], 
                              sample_recs['performance_level'],
                              sample_recs['priority_level'],
                              sample_recs['recommendations']],
                      fill_color='lavender',
                      align='left'))
        ])
        fig3.update_layout(title="Sample Student Recommendations (Top 10)")
        visualizations["sample_recommendations"] = fig3.to_json()

        # Store full recommendations in metrics
        metrics["student_recommendations"] = json.loads(recommendations_df.to_json(orient="records", default_handler=str))

        # Common patterns and insights
        common_weaknesses = []
        for rec in student_recommendations:
            common_weaknesses.extend(rec.get('weaknesses', []))
        
        if common_weaknesses:
            from collections import Counter
            weakness_counter = Counter(common_weaknesses)
            top_weaknesses = weakness_counter.most_common(3)
            
            insights.append(f"\n📋 Most Common Areas Needing Improvement:")
            for weakness, count in top_weaknesses:
                insights.append(f"   - {weakness}: {count} students")

        logger.info(f"Personalized recommendations generated for {len(student_recommendations)} students")
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Personalized recommendations generated successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        logger.error(f"Error in personalized_performance_recommendation_analysis: {str(e)}", exc_info=True)
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


# ========== MAIN APP / EXECUTION LOGIC ==========

def main():
    """Main function to run the Student Analytics script."""
    print("=" * 60)
    print("🎓 STUDENT ANALYTICS SYSTEM")
    print("=" * 60)

    # File path and encoding input
    file_path = input("Enter path to your data file (e.g., data.csv or data.xlsx): ")
    encoding = input("Enter file encoding (e.g., utf-8, latin1, cp1252, default=utf-8): ")
    if not encoding:
        encoding = 'utf-8'

    # Load data
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please provide CSV or Excel file.")
            return
    except Exception as e:
        print(f"Error loading file: {e}")
        logger.error(f"Error loading file: {e}", exc_info=True)
        return

    print("\n✅ Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # This dictionary maps the analysis names to the actual Python functions
    student_function_mapping = {
        "student_overall_performance_analysis": student_overall_performance_analysis,
        "subject_wise_score_analysis": subject_wise_score_analysis,
        "attendance_percentage_analysis": attendance_percentage_analysis,
        "internal_external_marks_comparison": internal_external_marks_comparison,
        "semester_wise_performance_trend": semester_wise_performance_trend,
        "weak_subject_identification": weak_subject_identification,
        "strong_subject_identification": strong_subject_identification,
        "attendance_vs_marks_correlation": attendance_vs_marks_correlation,
        "assignment_score_analysis": assignment_score_analysis,
        "cgpa_progression_analysis": cgpa_progression_analysis,
        "arrear_subject_detection": arrear_subject_detection,
        "grade_distribution_analysis": grade_distribution_analysis,
        "exam_attempt_performance_analysis": exam_attempt_performance_analysis,
        "improvement_rate_analysis": improvement_rate_analysis,
        "performance_consistency_analysis": performance_consistency_analysis,
        "risk_of_failure_prediction_analysis": risk_of_failure_prediction_analysis,
        "attendance_irregularity_detection": attendance_irregularity_detection,
        "subject_difficulty_impact_analysis": subject_difficulty_impact_analysis,
        "rank_position_analysis": rank_position_analysis,
        "personalized_performance_recommendation_analysis": personalized_performance_recommendation_analysis,
    }

    # --- Analysis Selection ---
    print("\n" + "=" * 60)
    print("📋 Select a Student Analytics Analysis to Perform:")
    print("=" * 60)
    
    analysis_names = list(student_function_mapping.keys())
    for i, name in enumerate(analysis_names):
        display_name = name.replace('_', ' ').title()
        print(f"{i+1:2d}. {display_name}")
    print(f"{len(analysis_names)+1:2d}. General Insights (Data Overview)")
    print("=" * 60)

    choice_str = input(f"Enter the number of your choice (1-{len(analysis_names)+1}): ")
    
    try:
        choice_idx = int(choice_str) - 1
        
        if 0 <= choice_idx < len(analysis_names):
            selected_analysis_key = analysis_names[choice_idx]
            selected_function = student_function_mapping.get(selected_analysis_key)
            
            if selected_function:
                print(f"\n🔄 Running {selected_analysis_key.replace('_', ' ').title()}...")
                print("-" * 60)
                
                try:
                    result = selected_function(df)
                    
                    print(f"\n✅ Analysis completed with status: {result['status']}")
                    print(f"📊 Found {len(result.get('visualizations', {}))} visualizations")
                    print(f"💡 Generated {len(result.get('insights', []))} insights")
                    
                    # Print insights
                    if result.get('insights'):
                        print("\n🔍 Key Insights:")
                        print("-" * 40)
                        for insight in result['insights']:
                            print(f"  {insight}")
                    
                    # Print key metrics
                    if result.get('metrics'):
                        print("\n📈 Key Metrics:")
                        print("-" * 40)
                        for key, value in result['metrics'].items():
                            if isinstance(value, (int, float)):
                                if isinstance(value, float):
                                    print(f"  {key}: {value:,.2f}")
                                else:
                                    print(f"  {key}: {value:,}")
                            elif isinstance(value, str):
                                print(f"  {key}: {value}")
                    
                    # Option to save results
                    save_choice = input("\n💾 Save results to JSON? (y/n): ").lower()
                    if save_choice == 'y':
                        output_file = f"{selected_analysis_key}_results.json"
                        with open(output_file, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                        print(f"✅ Results saved to {output_file}")
                    
                    return result
                    
                except Exception as e:
                    print(f"\n❌ Error during analysis: {e}")
                    logger.error(f"Error during analysis: {e}", exc_info=True)
                    return {
                        "analysis_type": selected_analysis_key,
                        "status": "error",
                        "error": str(e),
                        "insights": [f"Error during analysis: {str(e)}"]
                    }
            else:
                print(f"\n❌ Function for '{selected_analysis_key}' not found.")
                return None
                
        elif choice_idx == len(analysis_names):
            print("\n🔄 Running General Insights Analysis...")
            result = general_insights_analysis(df, "Student Data Overview")
            
            print(f"\n✅ General insights completed with {len(result.get('visualizations', {}))} visualizations")
            print(f"💡 Generated {len(result.get('insights', []))} insights")
            
            if result.get('insights'):
                print("\n🔍 Key Insights:")
                for insight in result['insights']:
                    print(f"  {insight}")
            
            return result
        else:
            print("\n❌ Invalid choice. Please enter a number within the given range.")
            return None
            
    except ValueError:
        print("\n❌ Invalid input. Please enter a number.")
        return None
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis cancelled by user.")
        return None


if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 60)
    print("✅ Student Analytics Complete")
    print("=" * 60)