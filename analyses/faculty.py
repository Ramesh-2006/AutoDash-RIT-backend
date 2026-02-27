import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fuzzywuzzy import process
import warnings
import json

warnings.filterwarnings('ignore')

# ========== NUMPY TO PYTHON TYPE CONVERSION UTILITY ==========

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

# ========== UTILITY FUNCTIONS ==========

def get_key_metrics(df):
    """Returns key metrics about the dataset as a dictionary"""
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    return {
        "total_records": total_records,
        "total_features": len(df.columns),
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
    using fuzzy string matching.
    """
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        # Case-insensitive exact match
        lower_available = {col.lower(): col for col in available}
        if target.lower() in lower_available:
             matched[target] = lower_available[target.lower()]
             continue
            
        match, score = process.extractOne(target, available)
        matched[target] = match if score >= 70 else None
    return matched

def general_insights_analysis(df, title="General Insights Analysis"):
    """
    Show general data visualizations.
    """
    analysis_type = "general_insights_analysis"
    try:
        visualizations = {}
        metrics = {}
        insights = [f"--- {title} ---"]

        # Key Metrics
        key_metrics = get_key_metrics(df)
        metrics.update(key_metrics)
        insights.append(f"Total Records: {key_metrics['total_records']}")
        insights.append(f"Total Features: {key_metrics['total_features']}")

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            selected_num_col = numeric_cols[0]
            insights.append(f"\nNumeric Features Analysis (showing first column: {selected_num_col})")

            # Histogram
            fig1 = px.histogram(df, x=selected_num_col,
                                title=f"Distribution of {selected_num_col}")
            visualizations["numeric_histogram"] = fig1.to_json()

            # Box Plot
            fig2 = px.box(df, y=selected_num_col,
                                title=f"Box Plot of {selected_num_col}")
            visualizations["numeric_boxplot"] = fig2.to_json()
        else:
            insights.append("[INFO] No numeric columns found for analysis.")

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            insights.append("\nFeature Correlations:")
            corr = df[numeric_cols].corr(numeric_only=True)
            fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                                   title="Correlation Between Numeric Features")
            visualizations["correlation_heatmap"] = fig3.to_json()
            metrics["correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            selected_cat_col = categorical_cols[0]
            insights.append(f"\nCategorical Features Analysis (showing first column: {selected_cat_col})")

            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']

            fig4 = px.bar(value_counts.head(20), x='Value', y='Count',
                                title=f"Top 20 Distribution of {selected_cat_col}")
            visualizations["categorical_barchart"] = fig4.to_json()
        else:
            insights.append("[INFO] No categorical columns found for analysis.")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred in general_insights_analysis: {str(e)}",
            "matched_columns": {},
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


# ========== FACULTY ANALYTICS FUNCTIONS ==========

def subject_pass_percentage_analysis(df):
    """Subject pass rates analysis"""
    analysis_type = "subject_pass_percentage_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject Pass Percentage Analysis ---"]
        
        expected = ['subject', 'subject_name', 'course_code', 'marks', 'score', 'grade', 'result', 'status', 'pass_fail']
        matched = fuzzy_match_column(df, expected)

        # Find subject column
        subject_col = None
        for key in ['subject', 'subject_name', 'course_code']:
            if matched.get(key):
                subject_col = matched[key]
                break

        if not subject_col:
            # Look for subject-like columns
            for col in df.columns:
                if any(x in col.lower() for x in ['subject', 'course', 'module']):
                    subject_col = col
                    matched['subject'] = col
                    break

        # Find result/pass status column
        result_col = None
        for key in ['result', 'status', 'pass_fail', 'grade']:
            if matched.get(key):
                result_col = matched[key]
                break

        if not subject_col or not result_col:
            missing = []
            if not subject_col:
                missing.append("subject")
            if not result_col:
                missing.append("result/pass status")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Determine pass/fail from result column
        result_str = df[result_col].astype(str).str.lower()
        
        # Try to determine pass indicators
        pass_indicators = ['pass', 'p', 'passed', 'a+', 'a', 'b+', 'b', 'c', 'd', '1', 'true', 'yes']
        fail_indicators = ['fail', 'f', 'failed', 'arrear', 'ra', '0', 'false', 'no']
        
        df['is_pass'] = result_str.apply(lambda x: 
            1 if any(ind in x for ind in pass_indicators) and not any(ind in x for ind in fail_indicators) 
            else 0 if any(ind in x for ind in fail_indicators) 
            else None)
        
        # If couldn't determine, try using marks column
        if df['is_pass'].isna().all() and matched.get('marks'):
            marks_col = matched['marks']
            df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
            pass_threshold = 40  # Default pass mark
            df['is_pass'] = (df[marks_col] >= pass_threshold).astype(int)
            insights.append(f"Using marks column with pass threshold {pass_threshold} to determine pass/fail.")

        df = df.dropna(subset=['is_pass', subject_col])

        # Calculate pass percentage by subject
        subject_pass = df.groupby(subject_col).agg(
            total_students=('is_pass', 'count'),
            passed=('is_pass', 'sum')
        ).reset_index()
        
        subject_pass['pass_percentage'] = (subject_pass['passed'] / subject_pass['total_students'] * 100).round(2)
        subject_pass = subject_pass.sort_values('pass_percentage', ascending=False)

        metrics["subject_pass_rates"] = json.loads(subject_pass.to_json(orient="split"))
        metrics["average_pass_percentage"] = subject_pass['pass_percentage'].mean()
        metrics["highest_pass_subject"] = str(subject_pass.iloc[0][subject_col])
        metrics["lowest_pass_subject"] = str(subject_pass.iloc[-1][subject_col])

        insights.append(f"Average Pass Percentage across all subjects: {metrics['average_pass_percentage']:.2f}%")
        insights.append(f"Subject with highest pass rate: {metrics['highest_pass_subject']} ({subject_pass.iloc[0]['pass_percentage']:.2f}%)")
        insights.append(f"Subject with lowest pass rate: {metrics['lowest_pass_subject']} ({subject_pass.iloc[-1]['pass_percentage']:.2f}%)")

        # Bar chart of pass percentages
        fig1 = px.bar(subject_pass, x=subject_col, y='pass_percentage',
                      title="Pass Percentage by Subject",
                      color='pass_percentage', color_continuous_scale='RdYlGn',
                      text='pass_percentage')
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        visualizations["subject_pass_rates"] = fig1.to_json()

        # Pass/Fail count by subject
        subject_pass_melted = subject_pass.melt(id_vars=[subject_col], 
                                                value_vars=['passed', 'total_students'],
                                                var_name='Metric', value_name='Count')
        
        fig2 = px.bar(subject_pass_melted, x=subject_col, y='Count', color='Metric',
                      title="Passed vs Total Students by Subject",
                      barmode='group')
        visualizations["pass_vs_total"] = fig2.to_json()

        # Identify subjects needing attention (low pass rate)
        low_pass_subjects = subject_pass[subject_pass['pass_percentage'] < 60]
        if len(low_pass_subjects) > 0:
            metrics["subjects_below_60_percent_pass"] = len(low_pass_subjects)
            insights.append(f"\n⚠️ Subjects with pass rate below 60%: {len(low_pass_subjects)}")
            for _, row in low_pass_subjects.iterrows():
                insights.append(f"  - {row[subject_col]}: {row['pass_percentage']:.1f}% pass rate")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def class_average_score_analysis(df):
    """Average scores analysis"""
    analysis_type = "class_average_score_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Class Average Score Analysis ---"]
        
        expected = ['subject', 'class', 'section', 'marks', 'score', 'internal_marks', 'external_marks', 'total_marks']
        matched = fuzzy_match_column(df, expected)

        # Find marks columns
        marks_cols = []
        for key in ['marks', 'score', 'total_marks']:
            if matched.get(key):
                marks_cols.append(matched[key])
                break
        
        # If no explicit marks column, look for numeric columns that could be scores
        if not marks_cols:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            potential_score_cols = [col for col in numeric_cols if any(x in col.lower() for x in 
                                   ['marks', 'score', 'grade', 'total', 'internal', 'external'])]
            if potential_score_cols:
                marks_cols = potential_score_cols[:3]  # Take first 3 potential score columns

        if not marks_cols:
            insights.append("Could not find marks/score columns.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert marks columns to numeric
        for col in marks_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate class averages
        class_averages = {}
        for col in marks_cols:
            class_averages[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        metrics["class_averages"] = convert_numpy_types(class_averages)

        # Primary marks column for insights
        primary_col = marks_cols[0]
        insights.append(f"Class Average ({primary_col}): {class_averages[primary_col]['mean']:.2f}")
        insights.append(f"Class Median: {class_averages[primary_col]['median']:.2f}")
        insights.append(f"Score Range: {class_averages[primary_col]['min']:.2f} - {class_averages[primary_col]['max']:.2f}")
        insights.append(f"Standard Deviation: {class_averages[primary_col]['std']:.2f}")

        # Distribution plot
        fig1 = px.histogram(df, x=primary_col, nbins=20,
                           title=f"Distribution of {primary_col}",
                           marginal='box')
        visualizations["score_distribution"] = fig1.to_json()

        # If multiple marks columns, compare them
        if len(marks_cols) > 1:
            df_melted = df[marks_cols].melt(var_name='Assessment', value_name='Score')
            fig2 = px.box(df_melted, x='Assessment', y='Score',
                         title="Score Distribution by Assessment Type")
            visualizations["assessment_comparison"] = fig2.to_json()

        # Section-wise averages if section column exists
        if matched.get('section') and matched['section'] in df.columns:
            section_col = matched['section']
            section_avgs = df.groupby(section_col)[primary_col].mean().reset_index()
            section_avgs = section_avgs.sort_values(primary_col, ascending=False)
            
            metrics["section_averages"] = json.loads(section_avgs.to_json(orient="split"))
            
            fig3 = px.bar(section_avgs, x=section_col, y=primary_col,
                         title=f"Average {primary_col} by Section",
                         color=primary_col, color_continuous_scale='Viridis')
            visualizations["section_averages"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def section_wise_performance_comparison(df):
    """Section performance comparison"""
    analysis_type = "section_wise_performance_comparison"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Section-wise Performance Comparison ---"]
        
        expected = ['section', 'class', 'batch', 'subject', 'marks', 'score', 'cgpa', 'gpa']
        matched = fuzzy_match_column(df, expected)

        # Find section column
        section_col = None
        for key in ['section', 'class', 'batch']:
            if matched.get(key):
                section_col = matched[key]
                break

        if not section_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['section', 'class', 'batch', 'group']):
                    section_col = col
                    matched['section'] = col
                    break

        # Find performance column
        perf_col = None
        for key in ['marks', 'score', 'cgpa', 'gpa']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not section_col or not perf_col:
            missing = []
            if not section_col:
                missing.append("section")
            if not perf_col:
                missing.append("performance column")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        df = df.dropna(subset=[section_col, perf_col])

        # Section-wise statistics
        section_stats = df.groupby(section_col)[perf_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
        section_stats.columns = [section_col, 'Average', 'Median', 'Std_Dev', 'Min', 'Max', 'Student_Count']
        section_stats = section_stats.sort_values('Average', ascending=False)

        metrics["section_performance"] = json.loads(section_stats.to_json(orient="split"))
        
        insights.append(f"Number of sections: {len(section_stats)}")
        insights.append(f"Top performing section: {section_stats.iloc[0][section_col]} (Avg: {section_stats.iloc[0]['Average']:.2f})")
        insights.append(f"Lowest performing section: {section_stats.iloc[-1][section_col]} (Avg: {section_stats.iloc[-1]['Average']:.2f})")
        
        # Performance gap
        performance_gap = section_stats.iloc[0]['Average'] - section_stats.iloc[-1]['Average']
        metrics["performance_gap"] = performance_gap
        insights.append(f"Performance gap between top and bottom sections: {performance_gap:.2f}")

        # Box plot comparison
        fig1 = px.box(df, x=section_col, y=perf_col,
                     title=f"Performance Distribution by Section",
                     color=section_col)
        visualizations["section_boxplot"] = fig1.to_json()

        # Bar chart of averages with error bars
        fig2 = px.bar(section_stats, x=section_col, y='Average',
                     error_y='Std_Dev',
                     title=f"Average {perf_col} by Section with Standard Deviation",
                     color='Average', color_continuous_scale='Viridis')
        visualizations["section_averages"] = fig2.to_json()

        # Radar chart for multi-subject comparison if subject column exists
        if matched.get('subject') and matched['subject'] in df.columns:
            subject_col = matched['subject']
            
            # Pivot table for radar chart
            pivot_df = df.pivot_table(values=perf_col, index=section_col, columns=subject_col, aggfunc='mean').reset_index()
            
            fig3 = go.Figure()
            for section in pivot_df[section_col].unique():
                section_data = pivot_df[pivot_df[section_col] == section].iloc[0]
                subjects = [col for col in pivot_df.columns if col != section_col]
                values = [section_data[subject] for subject in subjects]
                
                fig3.add_trace(go.Scatterpolar(
                    r=values,
                    theta=subjects,
                    fill='toself',
                    name=str(section)
                ))
            
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Section Performance Across Subjects"
            )
            visualizations["section_radar"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def top_performing_students_identification(df):
    """High performers identification"""
    analysis_type = "top_performing_students_identification"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Top Performing Students Identification ---"]
        
        expected = ['student_id', 'student_name', 'roll_no', 'cgpa', 'gpa', 'total_marks', 'percentage', 'rank']
        matched = fuzzy_match_column(df, expected)

        # Find student identifier
        student_id_col = None
        for key in ['student_id', 'student_name', 'roll_no']:
            if matched.get(key):
                student_id_col = matched[key]
                break

        # Find performance column
        perf_col = None
        for key in ['cgpa', 'gpa', 'total_marks', 'percentage', 'rank']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not perf_col:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                perf_col = numeric_cols[0]
                matched['performance'] = perf_col
                insights.append(f"Using '{perf_col}' as performance metric.")

        if not perf_col:
            insights.append("Could not find performance column for identifying top performers.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        df = df.dropna(subset=[perf_col])

        # Sort by performance (descending)
        df_sorted = df.sort_values(perf_col, ascending=False).reset_index(drop=True)
        
        # Get top performers (top 10 or top 10%)
        top_n = min(10, len(df_sorted))
        top_performers = df_sorted.head(top_n)
        
        # Calculate percentiles
        df['percentile'] = df[perf_col].rank(pct=True) * 100
        top_10_percent = df[df['percentile'] >= 90]
        
        metrics["total_students"] = len(df)
        metrics["top_10_students"] = json.loads(top_performers[[col for col in [student_id_col, perf_col] if col]].to_json(orient="split")) if student_id_col else None
        metrics["top_10_percent_count"] = len(top_10_percent)
        metrics["top_performer_score"] = top_performers.iloc[0][perf_col]
        metrics["threshold_for_top_10"] = df[perf_col].quantile(0.9)

        insights.append(f"Total students analyzed: {len(df)}")
        insights.append(f"Top performer score: {top_performers.iloc[0][perf_col]:.2f}")
        insights.append(f"Score needed for top 10%: {metrics['threshold_for_top_10']:.2f}")
        insights.append(f"Number of students in top 10%: {len(top_10_percent)}")

        if student_id_col:
            insights.append("\nTop 10 Students:")
            for idx, row in top_performers.iterrows():
                student_name = row.get(student_id_col, f"Student {idx+1}")
                insights.append(f"  Rank {idx+1}: {student_name} - {row[perf_col]:.2f}")

        # Distribution with top performers highlighted
        fig1 = px.histogram(df, x=perf_col, nbins=30,
                           title=f"Distribution of {perf_col} with Top 10% Highlighted")
        
        # Add threshold line
        fig1.add_vline(x=metrics['threshold_for_top_10'], line_dash="dash",
                      line_color="red", annotation_text="Top 10% Threshold")
        visualizations["performance_distribution"] = fig1.to_json()

        # Top performers bar chart
        if student_id_col:
            fig2 = px.bar(top_performers, x=student_id_col, y=perf_col,
                         title="Top 10 Performing Students",
                         color=perf_col, color_continuous_scale='Viridis',
                         text=perf_col)
            fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            visualizations["top_performers"] = fig2.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def low_performing_students_identification(df):
    """Low performers identification"""
    analysis_type = "low_performing_students_identification"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Low Performing Students Identification ---"]
        
        expected = ['student_id', 'student_name', 'roll_no', 'cgpa', 'gpa', 'total_marks', 'percentage', 'failed_subjects']
        matched = fuzzy_match_column(df, expected)

        # Find student identifier
        student_id_col = None
        for key in ['student_id', 'student_name', 'roll_no']:
            if matched.get(key):
                student_id_col = matched[key]
                break

        # Find performance column
        perf_col = None
        for key in ['cgpa', 'gpa', 'total_marks', 'percentage']:
            if matched.get(key):
                perf_col = matched[key]
                break

        # Define low performance threshold
        low_threshold = 40  # Default threshold for low performance

        if not perf_col:
            # Try to use failed subjects count
            if matched.get('failed_subjects'):
                fail_col = matched['failed_subjects']
                df[fail_col] = pd.to_numeric(df[fail_col], errors='coerce')
                low_performers = df[df[fail_col] > 0]
                
                metrics["students_with_failures"] = len(low_performers)
                metrics["total_failures"] = df[fail_col].sum()
                
                insights.append(f"Students with at least one failure: {len(low_performers)}")
                insights.append(f"Total failed subjects: {df[fail_col].sum()}")
                
                fig1 = px.histogram(df, x=fail_col, nbins=20,
                                   title="Distribution of Failed Subjects per Student")
                visualizations["failure_distribution"] = fig1.to_json()
                
                return {
                    "analysis_type": analysis_type,
                    "status": "success",
                    "message": "Analysis completed successfully.",
                    "matched_columns": matched,
                    "visualizations": visualizations,
                    "metrics": convert_numpy_types(metrics),
                    "insights": insights
                }
            else:
                insights.append("Could not find performance column for identifying low performers.")
                fallback_data = general_insights_analysis(df, "General Analysis")
                fallback_data["analysis_type"] = analysis_type
                fallback_data["status"] = "fallback"
                fallback_data["message"] = "Required columns not found, returned general analysis."
                fallback_data["matched_columns"] = matched
                fallback_data["insights"] = insights + fallback_data["insights"]
                return fallback_data

        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        df = df.dropna(subset=[perf_col])

        # Calculate percentiles
        df['percentile'] = df[perf_col].rank(pct=True) * 100
        
        # Identify low performers (below threshold or bottom 10%)
        low_by_threshold = df[df[perf_col] < low_threshold]
        low_by_percentile = df[df['percentile'] <= 10]
        
        # Combine or use appropriate method
        low_performers = low_by_threshold if len(low_by_threshold) > 0 else low_by_percentile
        
        metrics["total_students"] = len(df)
        metrics["low_performers_count"] = len(low_performers)
        metrics["low_performance_threshold"] = low_threshold
        metrics["bottom_10_percent_threshold"] = df[perf_col].quantile(0.1)
        metrics["lowest_score"] = df[perf_col].min()

        insights.append(f"Total students analyzed: {len(df)}")
        insights.append(f"Lowest score: {df[perf_col].min():.2f}")
        insights.append(f"Bottom 10% threshold: {df[perf_col].quantile(0.1):.2f}")
        insights.append(f"Students performing below {low_threshold}: {len(low_by_threshold)}")
        insights.append(f"Students in bottom 10%: {len(low_by_percentile)}")

        if student_id_col and len(low_performers) > 0:
            insights.append("\nLow Performing Students (sample):")
            sample_size = min(10, len(low_performers))
            for idx, row in low_performers.head(sample_size).iterrows():
                student_name = row.get(student_id_col, f"Student {idx}")
                insights.append(f"  {student_name}: {row[perf_col]:.2f} ({row['percentile']:.1f}th percentile)")

        # Distribution with low performers highlighted
        fig1 = px.histogram(df, x=perf_col, nbins=30,
                           title=f"Distribution of {perf_col} with Low Performance Region Highlighted")
        
        # Add threshold line
        fig1.add_vline(x=low_threshold, line_dash="dash",
                      line_color="red", annotation_text=f"Low Threshold ({low_threshold})")
        
        # Add bottom 10% line
        bottom_10 = df[perf_col].quantile(0.1)
        fig1.add_vline(x=bottom_10, line_dash="dot",
                      line_color="orange", annotation_text="Bottom 10%")
        
        visualizations["performance_distribution"] = fig1.to_json()

        # Box plot to identify outliers (potential low performers)
        fig2 = px.box(df, y=perf_col, title=f"Box Plot of {perf_col} with Outliers")
        visualizations["performance_boxplot"] = fig2.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def subject_difficulty_index_analysis(df):
    """Subject difficulty analysis from faculty perspective"""
    analysis_type = "subject_difficulty_index_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject Difficulty Index Analysis ---"]
        
        expected = ['subject', 'subject_name', 'course_code', 'marks', 'score', 'pass_rate', 'failure_rate', 'average_marks']
        matched = fuzzy_match_column(df, expected)

        # Find subject column
        subject_col = None
        for key in ['subject', 'subject_name', 'course_code']:
            if matched.get(key):
                subject_col = matched[key]
                break

        if not subject_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['subject', 'course', 'module']):
                    subject_col = col
                    matched['subject'] = col
                    break

        # Find marks column
        marks_col = None
        for key in ['marks', 'score', 'average_marks']:
            if matched.get(key):
                marks_col = matched[key]
                break

        if not subject_col or not marks_col:
            missing = []
            if not subject_col:
                missing.append("subject")
            if not marks_col:
                missing.append("marks")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
        df = df.dropna(subset=[subject_col, marks_col])

        # Calculate subject statistics
        subject_stats = df.groupby(subject_col)[marks_col].agg(['mean', 'std', 'count']).reset_index()
        subject_stats.columns = [subject_col, 'avg_marks', 'std_dev', 'student_count']
        
        # Calculate difficulty index (inverse of average marks, normalized 0-1)
        max_avg = subject_stats['avg_marks'].max()
        min_avg = subject_stats['avg_marks'].min()
        
        if max_avg > min_avg:
            subject_stats['difficulty_index'] = 1 - ((subject_stats['avg_marks'] - min_avg) / (max_avg - min_avg))
        else:
            subject_stats['difficulty_index'] = 0.5
        
        # Calculate pass rate if possible (assuming pass threshold 40)
        pass_threshold = 40
        subject_pass_rates = []
        for subject in subject_stats[subject_col]:
            subject_data = df[df[subject_col] == subject][marks_col]
            pass_rate = (subject_data >= pass_threshold).mean() * 100
            subject_pass_rates.append(pass_rate)
        
        subject_stats['pass_rate'] = subject_pass_rates
        subject_stats['difficulty_category'] = pd.cut(subject_stats['difficulty_index'],
                                                      bins=[0, 0.3, 0.6, 1.0],
                                                      labels=['Easy', 'Moderate', 'Difficult'])
        
        subject_stats = subject_stats.sort_values('difficulty_index', ascending=False)

        metrics["subject_difficulty"] = json.loads(subject_stats.to_json(orient="split"))
        metrics["average_difficulty_index"] = subject_stats['difficulty_index'].mean()
        metrics["most_difficult_subject"] = str(subject_stats.iloc[0][subject_col])
        metrics["easiest_subject"] = str(subject_stats.iloc[-1][subject_col])

        insights.append("Subject Difficulty Analysis:")
        insights.append(f"  Most Difficult: {metrics['most_difficult_subject']} (Index: {subject_stats.iloc[0]['difficulty_index']:.3f}, Avg: {subject_stats.iloc[0]['avg_marks']:.1f})")
        insights.append(f"  Easiest: {metrics['easiest_subject']} (Index: {subject_stats.iloc[-1]['difficulty_index']:.3f}, Avg: {subject_stats.iloc[-1]['avg_marks']:.1f})")
        
        # Category counts
        difficulty_counts = subject_stats['difficulty_category'].value_counts()
        for category, count in difficulty_counts.items():
            insights.append(f"  {category} subjects: {count}")

        # Bar chart of difficulty index
        fig1 = px.bar(subject_stats.sort_values('difficulty_index'), 
                      x=subject_col, y='difficulty_index',
                      title="Subject Difficulty Index (Higher = More Difficult)",
                      color='difficulty_index', color_continuous_scale='RdYlGn_r',
                      text=subject_stats['avg_marks'].round(1))
        fig1.update_traces(texttemplate='Avg: %{text}', textposition='outside')
        visualizations["difficulty_index"] = fig1.to_json()

        # Scatter plot of avg marks vs pass rate
        fig2 = px.scatter(subject_stats, x='avg_marks', y='pass_rate',
                         text=subject_col, size='student_count',
                         title="Average Marks vs Pass Rate by Subject",
                         labels={'avg_marks': 'Average Marks', 'pass_rate': 'Pass Rate (%)'})
        fig2.update_traces(textposition='top center')
        visualizations["marks_vs_passrate"] = fig2.to_json()

        # Difficulty distribution pie chart
        fig3 = px.pie(names=difficulty_counts.index, values=difficulty_counts.values,
                     title="Subject Difficulty Distribution")
        visualizations["difficulty_pie"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def internal_marks_variation_analysis(df):
    """Internal marks variation analysis"""
    analysis_type = "internal_marks_variation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Internal Marks Variation Analysis ---"]
        
        expected = ['subject', 'internal_marks', 'internal', 'assignment_marks', 'quiz_marks', 'lab_marks', 'student_id']
        matched = fuzzy_match_column(df, expected)

        # Find internal marks columns
        internal_cols = []
        
        # Check for explicit internal marks columns
        for key in ['internal_marks', 'internal', 'assignment_marks', 'quiz_marks', 'lab_marks']:
            if matched.get(key):
                internal_cols.append(matched[key])

        # If no explicit columns, look for columns with 'internal' or component names
        if not internal_cols:
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['internal', 'assignment', 'quiz', 'lab', 'practical', 'cia']):
                    internal_cols.append(col)

        if not internal_cols:
            insights.append("Could not find internal marks columns.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        for col in internal_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate variation statistics
        variation_stats = {}
        for col in internal_cols:
            variation_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'cv': df[col].std() / df[col].mean() if df[col].mean() > 0 else 0,  # Coefficient of variation
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
        
        metrics["variation_statistics"] = convert_numpy_types(variation_stats)

        insights.append("Internal Marks Variation Analysis:")
        for col, stats in variation_stats.items():
            insights.append(f"  {col}:")
            insights.append(f"    Mean: {stats['mean']:.2f}")
            insights.append(f"    Std Dev: {stats['std']:.2f}")
            insights.append(f"    CV: {stats['cv']:.3f}")
            insights.append(f"    Range: {stats['range']:.2f}")

        # Box plot for all internal components
        df_melted = df[internal_cols].melt(var_name='Component', value_name='Marks')
        fig1 = px.box(df_melted, x='Component', y='Marks',
                     title="Internal Marks Distribution by Component")
        visualizations["internal_marks_boxplot"] = fig1.to_json()

        # Violin plot for distribution comparison
        fig2 = px.violin(df_melted, x='Component', y='Marks',
                        box=True, points='outliers',
                        title="Internal Marks Distribution (Violin Plot)")
        visualizations["internal_marks_violin"] = fig2.to_json()

        # Heatmap of correlations between components
        if len(internal_cols) > 1:
            corr = df[internal_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, aspect='auto',
                            title="Correlation Between Internal Components")
            visualizations["component_correlations"] = fig3.to_json()
            metrics["correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        # Identify high variation components
        high_variation = [col for col, stats in variation_stats.items() if stats['cv'] > 0.3]
        if high_variation:
            insights.append(f"\n⚠️ Components with high variation (CV > 0.3): {', '.join(high_variation)}")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def attendance_pattern_analysis(df):
    """Attendance trends analysis for faculty"""
    analysis_type = "attendance_pattern_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance Pattern Analysis ---"]
        
        expected = ['date', 'student_id', 'section', 'attendance', 'present', 'absent', 'subject']
        matched = fuzzy_match_column(df, expected)

        # Find attendance column
        attendance_col = None
        for key in ['attendance', 'present', 'absent']:
            if matched.get(key):
                attendance_col = matched[key]
                break

        if not attendance_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['attendance', 'present', 'absent']):
                    attendance_col = col
                    matched['attendance'] = col
                    break

        if not attendance_col:
            insights.append("Could not find attendance column.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert attendance to binary (1 for present)
        if attendance_col in df.columns:
            att_str = df[attendance_col].astype(str).str.lower()
            df['is_present'] = att_str.str.contains('present|p|1|true|yes').astype(int)

        # Find date column for time-based patterns
        date_col = matched.get('date')
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col)
            
            # Daily attendance rate
            daily_attendance = df.groupby(date_col)['is_present'].mean().reset_index()
            daily_attendance.columns = [date_col, 'attendance_rate']
            
            fig1 = px.line(daily_attendance, x=date_col, y='attendance_rate',
                          title="Daily Attendance Rate Trend",
                          markers=True)
            visualizations["daily_attendance"] = fig1.to_json()
            
            # Day of week patterns
            df['day_of_week'] = df[date_col].dt.day_name()
            dow_attendance = df.groupby('day_of_week')['is_present'].mean().reset_index()
            
            # Sort days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_attendance['day_order'] = dow_attendance['day_of_week'].apply(
                lambda x: day_order.index(x) if x in day_order else 999
            )
            dow_attendance = dow_attendance.sort_values('day_order').drop('day_order', axis=1)
            
            fig2 = px.bar(dow_attendance, x='day_of_week', y='is_present',
                         title="Attendance Rate by Day of Week",
                         color='is_present', color_continuous_scale='RdYlGn')
            visualizations["attendance_by_day"] = fig2.to_json()

        # Section-wise attendance
        section_col = matched.get('section')
        if section_col and section_col in df.columns:
            section_attendance = df.groupby(section_col)['is_present'].agg(['mean', 'count']).reset_index()
            section_attendance.columns = [section_col, 'attendance_rate', 'total_records']
            section_attendance = section_attendance.sort_values('attendance_rate', ascending=False)
            
            metrics["section_attendance"] = json.loads(section_attendance.to_json(orient="split"))
            
            fig3 = px.bar(section_attendance, x=section_col, y='attendance_rate',
                         title="Attendance Rate by Section",
                         color='attendance_rate', color_continuous_scale='RdYlGn')
            visualizations["section_attendance"] = fig3.to_json()

        # Subject-wise attendance
        subject_col = matched.get('subject')
        if subject_col and subject_col in df.columns:
            subject_attendance = df.groupby(subject_col)['is_present'].mean().reset_index()
            subject_attendance.columns = [subject_col, 'attendance_rate']
            subject_attendance = subject_attendance.sort_values('attendance_rate', ascending=False)
            
            fig4 = px.bar(subject_attendance, x=subject_col, y='attendance_rate',
                         title="Attendance Rate by Subject",
                         color='attendance_rate', color_continuous_scale='RdYlGn')
            visualizations["subject_attendance"] = fig4.to_json()

        # Overall statistics
        overall_attendance = df['is_present'].mean() * 100
        metrics["overall_attendance_rate"] = overall_attendance
        insights.append(f"Overall Attendance Rate: {overall_attendance:.2f}%")
        
        if date_col:
            insights.append(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def exam_wise_result_trend_analysis(df):
    """Exam results trend analysis"""
    analysis_type = "exam_wise_result_trend_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Exam-wise Result Trend Analysis ---"]
        
        expected = ['exam_name', 'exam_date', 'term', 'subject', 'average_marks', 'pass_percentage', 'highest_marks', 'lowest_marks']
        matched = fuzzy_match_column(df, expected)

        # Find exam identifier
        exam_col = None
        for key in ['exam_name', 'term']:
            if matched.get(key):
                exam_col = matched[key]
                break

        # Find performance metrics
        avg_col = matched.get('average_marks')
        pass_col = matched.get('pass_percentage')
        high_col = matched.get('highest_marks')
        low_col = matched.get('lowest_marks')

        if not exam_col or not (avg_col or pass_col):
            missing = []
            if not exam_col:
                missing.append("exam identifier")
            if not (avg_col or pass_col):
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Prepare data for trend analysis
        trend_data = df.copy()
        
        # Convert date if available
        date_col = matched.get('exam_date')
        if date_col and date_col in df.columns:
            trend_data[date_col] = pd.to_datetime(trend_data[date_col], errors='coerce')
            sort_col = date_col
        else:
            # Use exam name/term as categorical
            sort_col = exam_col
            # Try to extract numeric part for sorting
            trend_data['sort_key'] = trend_data[exam_col].astype(str).str.extract('(\d+)').astype(float)
            if trend_data['sort_key'].notna().any():
                trend_data = trend_data.sort_values('sort_key')
            else:
                trend_data = trend_data.sort_values(exam_col)

        # Create trend visualizations
        if avg_col:
            avg_col = matched['average_marks']
            trend_data[avg_col] = pd.to_numeric(trend_data[avg_col], errors='coerce')
            
            fig1 = px.line(trend_data, x=sort_col, y=avg_col,
                          title=f"Average Marks Trend Across Exams",
                          markers=True)
            if high_col and low_col:
                fig1.add_scatter(x=trend_data[sort_col], y=trend_data[high_col],
                                mode='lines+markers', name='Highest', line=dict(dash='dash'))
                fig1.add_scatter(x=trend_data[sort_col], y=trend_data[low_col],
                                mode='lines+markers', name='Lowest', line=dict(dash='dot'))
            visualizations["marks_trend"] = fig1.to_json()
            
            metrics["average_marks_trend"] = json.loads(trend_data[[sort_col, avg_col]].dropna().to_json(orient="split"))

        if pass_col:
            pass_col = matched['pass_percentage']
            trend_data[pass_col] = pd.to_numeric(trend_data[pass_col], errors='coerce')
            
            fig2 = px.line(trend_data, x=sort_col, y=pass_col,
                          title="Pass Percentage Trend Across Exams",
                          markers=True)
            fig2.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Target (60%)")
            visualizations["pass_trend"] = fig2.to_json()
            
            metrics["pass_percentage_trend"] = json.loads(trend_data[[sort_col, pass_col]].dropna().to_json(orient="split"))

        # Calculate improvement
        if avg_col and len(trend_data) >= 2:
            first_exam = trend_data.iloc[0][avg_col] if pd.notna(trend_data.iloc[0][avg_col]) else None
            last_exam = trend_data.iloc[-1][avg_col] if pd.notna(trend_data.iloc[-1][avg_col]) else None
            
            if first_exam and last_exam:
                improvement = last_exam - first_exam
                improvement_pct = (improvement / first_exam) * 100 if first_exam > 0 else 0
                
                metrics["overall_improvement"] = improvement
                metrics["improvement_percentage"] = improvement_pct
                
                insights.append(f"Overall improvement from first to last exam: {improvement:+.2f} ({improvement_pct:+.1f}%)")

        # Exam performance summary
        insights.append("Exam Performance Summary:")
        if avg_col:
            insights.append(f"  Best average: {trend_data.loc[trend_data[avg_col].idxmax(), exam_col]} ({trend_data[avg_col].max():.2f})")
            insights.append(f"  Worst average: {trend_data.loc[trend_data[avg_col].idxmin(), exam_col]} ({trend_data[avg_col].min():.2f})")
        
        if pass_col:
            insights.append(f"  Highest pass rate: {trend_data.loc[trend_data[pass_col].idxmax(), exam_col]} ({trend_data[pass_col].max():.1f}%)")
            insights.append(f"  Lowest pass rate: {trend_data.loc[trend_data[pass_col].idxmin(), exam_col]} ({trend_data[pass_col].min():.1f}%)")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def grade_distribution_per_subject_analysis(df):
    """Grade distribution by subject"""
    analysis_type = "grade_distribution_per_subject_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Grade Distribution per Subject Analysis ---"]
        
        expected = ['subject', 'subject_name', 'grade', 'letter_grade', 'student_id', 'marks']
        matched = fuzzy_match_column(df, expected)

        # Find subject column
        subject_col = None
        for key in ['subject', 'subject_name']:
            if matched.get(key):
                subject_col = matched[key]
                break

        # Find grade column
        grade_col = None
        for key in ['grade', 'letter_grade']:
            if matched.get(key):
                grade_col = matched[key]
                break

        if not subject_col or not grade_col:
            # Try to derive grades from marks
            if not grade_col and matched.get('marks'):
                marks_col = matched['marks']
                df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
                
                # Define grade mapping
                def marks_to_grade(marks):
                    if marks >= 90:
                        return 'A+'
                    elif marks >= 80:
                        return 'A'
                    elif marks >= 70:
                        return 'B+'
                    elif marks >= 60:
                        return 'B'
                    elif marks >= 50:
                        return 'C'
                    elif marks >= 40:
                        return 'D'
                    else:
                        return 'F'
                
                df['derived_grade'] = df[marks_col].apply(marks_to_grade)
                grade_col = 'derived_grade'
                insights.append("Grades derived from marks column.")
            
            if not subject_col:
                missing = []
                if not subject_col:
                    missing.append("subject")
                if not grade_col:
                    missing.append("grade")
                insights += get_missing_columns_message(missing, matched)
                
                fallback_data = general_insights_analysis(df, "General Analysis")
                fallback_data["analysis_type"] = analysis_type
                fallback_data["status"] = "fallback"
                fallback_data["message"] = "Required columns not found, returned general analysis."
                fallback_data["matched_columns"] = matched
                fallback_data["insights"] = insights + fallback_data["insights"]
                return fallback_data

        # Create grade distribution by subject
        grade_dist = df.groupby([subject_col, grade_col]).size().reset_index(name='count')
        
        # Calculate percentages within each subject
        subject_totals = grade_dist.groupby(subject_col)['count'].sum().reset_index(name='total')
        grade_dist = grade_dist.merge(subject_totals, on=subject_col)
        grade_dist['percentage'] = (grade_dist['count'] / grade_dist['total'] * 100).round(2)
        
        # Sort grades logically
        grade_order = ['A+', 'A', 'B+', 'B', 'C', 'D', 'F']
        grade_dist['grade_order'] = grade_dist[grade_col].apply(
            lambda x: grade_order.index(x) if x in grade_order else 999
        )
        grade_dist = grade_dist.sort_values(['subject_col', 'grade_order'])

        metrics["grade_distribution"] = json.loads(grade_dist.to_json(orient="split"))

        insights.append(f"Grade distribution analyzed for {len(subject_totals)} subjects")

        # Stacked bar chart
        fig1 = px.bar(grade_dist, x=subject_col, y='percentage', color=grade_col,
                     title="Grade Distribution by Subject (Percentage)",
                     barmode='stack',
                     category_orders={grade_col: grade_order})
        visualizations["grade_distribution_stacked"] = fig1.to_json()

        # Heatmap of grade distribution
        pivot_df = grade_dist.pivot(index=subject_col, columns=grade_col, values='percentage').fillna(0)
        
        # Ensure all grade columns exist
        for grade in grade_order:
            if grade not in pivot_df.columns:
                pivot_df[grade] = 0
        
        pivot_df = pivot_df[grade_order]
        
        fig2 = px.imshow(pivot_df, text_auto=True, aspect='auto',
                        title="Grade Distribution Heatmap by Subject",
                        color_continuous_scale='Viridis',
                        labels=dict(x="Grade", y="Subject", color="Percentage"))
        visualizations["grade_heatmap"] = fig2.to_json()

        # Subject performance summary
        subject_summary = []
        for subject in subject_totals[subject_col]:
            subject_grades = grade_dist[grade_dist[subject_col] == subject]
            pass_grades = subject_grades[~subject_grades[grade_col].isin(['F'])]
            pass_percentage = pass_grades['percentage'].sum()
            
            top_grades = subject_grades[subject_grades[grade_col].isin(['A+', 'A'])]
            top_percentage = top_grades['percentage'].sum() if not top_grades.empty else 0
            
            subject_summary.append({
                'subject': subject,
                'pass_percentage': pass_percentage,
                'top_grade_percentage': top_percentage,
                'total_students': subject_totals[subject_totals[subject_col] == subject]['total'].values[0]
            })
        
        summary_df = pd.DataFrame(subject_summary)
        metrics["subject_summary"] = json.loads(summary_df.to_json(orient="split"))

        # Box plot of marks by subject if available
        if matched.get('marks'):
            marks_col = matched['marks']
            fig3 = px.box(df, x=subject_col, y=marks_col,
                         title="Marks Distribution by Subject")
            visualizations["marks_by_subject"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def question_difficulty_pattern_analysis(df):
    """Question difficulty pattern analysis"""
    analysis_type = "question_difficulty_pattern_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Question Difficulty Pattern Analysis ---"]
        
        expected = ['question_id', 'question_number', 'topic', 'difficulty_level', 'avg_score', 'correct_percentage', 'time_taken']
        matched = fuzzy_match_column(df, expected)

        # Find question identifier
        q_id_col = None
        for key in ['question_id', 'question_number']:
            if matched.get(key):
                q_id_col = matched[key]
                break

        # Find difficulty indicator
        diff_col = None
        for key in ['difficulty_level', 'avg_score', 'correct_percentage']:
            if matched.get(key):
                diff_col = matched[key]
                break

        if not q_id_col or not diff_col:
            missing = []
            if not q_id_col:
                missing.append("question identifier")
            if not diff_col:
                missing.append("difficulty indicator")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert difficulty metric to numeric
        df[diff_col] = pd.to_numeric(df[diff_col], errors='coerce')
        
        # If difficulty_level is categorical, map to numeric
        if diff_col == matched.get('difficulty_level'):
            difficulty_map = {
                'easy': 1, 'Easy': 1, 'EASY': 1,
                'medium': 2, 'Medium': 2, 'MEDIUM': 2,
                'hard': 3, 'Hard': 3, 'HARD': 3, 'difficult': 3
            }
            df['difficulty_numeric'] = df[diff_col].astype(str).str.lower().map(difficulty_map)
            if df['difficulty_numeric'].notna().any():
                diff_col = 'difficulty_numeric'

        # Calculate question statistics
        q_stats = df.groupby(q_id_col)[diff_col].agg(['mean', 'std', 'count']).reset_index()
        q_stats.columns = [q_id_col, 'avg_difficulty', 'std_difficulty', 'response_count']
        q_stats = q_stats.sort_values('avg_difficulty', ascending=False)

        # Categorize questions
        q_stats['difficulty_category'] = pd.cut(q_stats['avg_difficulty'],
                                               bins=[0, 0.33, 0.66, 1.0] if diff_col == 'correct_percentage' else
                                               [0, 1.5, 2.5, 3] if diff_col == 'difficulty_numeric' else
                                               [q_stats['avg_difficulty'].quantile(0.33), 
                                                q_stats['avg_difficulty'].quantile(0.67)],
                                               labels=['Easy', 'Medium', 'Hard'])

        metrics["question_difficulty"] = json.loads(q_stats.to_json(orient="split"))
        
        # Difficulty distribution
        diff_counts = q_stats['difficulty_category'].value_counts()
        insights.append("Question Difficulty Distribution:")
        for category, count in diff_counts.items():
            insights.append(f"  {category}: {count} questions ({count/len(q_stats)*100:.1f}%)")

        # Topic-wise difficulty if topic column exists
        topic_col = matched.get('topic')
        if topic_col and topic_col in df.columns:
            topic_diff = df.groupby(topic_col)[diff_col].mean().reset_index()
            topic_diff.columns = [topic_col, 'avg_difficulty']
            topic_diff = topic_diff.sort_values('avg_difficulty', ascending=False)
            
            fig1 = px.bar(topic_diff, x=topic_col, y='avg_difficulty',
                         title="Average Question Difficulty by Topic",
                         color='avg_difficulty', color_continuous_scale='RdYlGn_r')
            visualizations["topic_difficulty"] = fig1.to_json()

        # Difficulty distribution pie chart
        fig2 = px.pie(names=diff_counts.index, values=diff_counts.values,
                     title="Question Difficulty Distribution")
        visualizations["difficulty_pie"] = fig2.to_json()

        # Scatter plot of questions by difficulty
        fig3 = px.scatter(q_stats, x=q_id_col, y='avg_difficulty',
                         size='response_count', color='difficulty_category',
                         title="Question Difficulty Scatter Plot",
                         labels={'avg_difficulty': 'Average Difficulty Score'})
        visualizations["difficulty_scatter"] = fig3.to_json()

        # Identify most and least difficult questions
        hardest = q_stats.nlargest(5, 'avg_difficulty')
        easiest = q_stats.nsmallest(5, 'avg_difficulty')
        
        insights.append("\nHardest Questions (Top 5):")
        for _, row in hardest.iterrows():
            insights.append(f"  {row[q_id_col]}: {row['avg_difficulty']:.2f}")
        
        insights.append("\nEasiest Questions (Top 5):")
        for _, row in easiest.iterrows():
            insights.append(f"  {row[q_id_col]}: {row['avg_difficulty']:.2f}")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def assessment_component_weightage_analysis(df):
    """Assessment weights analysis"""
    analysis_type = "assessment_component_weightage_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Assessment Component Weightage Analysis ---"]
        
        expected = ['component', 'assessment_type', 'weightage', 'max_marks', 'obtained_marks', 'student_id']
        matched = fuzzy_match_column(df, expected)

        # Find component column
        component_col = None
        for key in ['component', 'assessment_type']:
            if matched.get(key):
                component_col = matched[key]
                break

        # Find weightage column
        weightage_col = matched.get('weightage')
        
        # Find marks columns
        marks_col = matched.get('obtained_marks')
        max_marks_col = matched.get('max_marks')

        if not component_col or not (marks_col or weightage_col):
            missing = []
            if not component_col:
                missing.append("component")
            if not (marks_col or weightage_col):
                missing.append("marks or weightage")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert columns to numeric
        if marks_col:
            df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
        if max_marks_col:
            df[max_marks_col] = pd.to_numeric(df[max_marks_col], errors='coerce')
        if weightage_col:
            df[weightage_col] = pd.to_numeric(df[weightage_col], errors='coerce')

        # Calculate component statistics
        component_stats = df.groupby(component_col).agg({
            marks_col: ['mean', 'std', 'min', 'max'] if marks_col else {},
            max_marks_col: ['mean'] if max_marks_col else {},
            weightage_col: ['mean'] if weightage_col else {},
            component_col: 'count'
        }).reset_index()
        
        # Flatten column names
        component_stats.columns = ['_'.join(col).strip('_') for col in component_stats.columns.values]
        
        # Calculate contribution if possible
        if marks_col and weightage_col:
            df['weighted_score'] = df[marks_col] * df[weightage_col] / 100
            component_stats['avg_contribution'] = df.groupby(component_col)['weighted_score'].mean().values
        
        metrics["component_statistics"] = json.loads(component_stats.to_json(orient="split"))

        # Weightage distribution pie chart
        if weightage_col:
            weightage_summary = df.groupby(component_col)[weightage_col].mean().reset_index()
            fig1 = px.pie(weightage_summary, values=weightage_col, names=component_col,
                         title="Assessment Component Weightage Distribution")
            visualizations["weightage_pie"] = fig1.to_json()

        # Component performance comparison
        if marks_col:
            fig2 = px.box(df, x=component_col, y=marks_col,
                         title=f"Marks Distribution by Assessment Component")
            visualizations["component_performance"] = fig2.to_json()

        # Radar chart for component performance
        if marks_col and len(df[component_col].unique()) >= 3:
            avg_scores = df.groupby(component_col)[marks_col].mean().reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=avg_scores[marks_col].values,
                theta=avg_scores[component_col].values,
                fill='toself',
                name='Average Scores'
            ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Component Performance Radar Chart"
            )
            visualizations["component_radar"] = fig3.to_json()

        insights.append("Assessment Component Analysis:")
        for _, row in component_stats.iterrows():
            comp_name = row[component_col + '_']
            if marks_col:
                insights.append(f"  {comp_name}: Avg {row[marks_col + '_mean']:.2f}, Range {row[marks_col + '_min']:.0f}-{row[marks_col + '_max']:.0f}")
            if weightage_col:
                insights.append(f"    Weightage: {row[weightage_col + '_mean']:.1f}%")

        # Check if weightages sum to 100
        if weightage_col:
            total_weightage = df[weightage_col].mean() if len(df[component_col].unique()) == 1 else df.groupby(component_col)[weightage_col].mean().sum()
            if abs(total_weightage - 100) > 1:
                insights.append(f"\n⚠️ Total weightage sums to {total_weightage:.1f}% (should be 100%)")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def remedial_need_identification_analysis(df):
    """Remedial support identification"""
    analysis_type = "remedial_need_identification_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Remedial Need Identification Analysis ---"]
        
        expected = ['student_id', 'student_name', 'subject', 'marks', 'grade', 'attendance', 'failed_subjects', 'remedial_needed']
        matched = fuzzy_match_column(df, expected)

        # Define thresholds for remedial need
        marks_threshold = 40  # Below this need remedial
        attendance_threshold = 75  # Below this attendance is concerning
        failure_threshold = 1  # Any failure indicates need

        # Find columns
        student_col = matched.get('student_id') or matched.get('student_name')
        subject_col = matched.get('subject')
        marks_col = matched.get('marks')
        attendance_col = matched.get('attendance')
        fail_col = matched.get('failed_subjects')

        if not student_col:
            insights.append("Could not find student identifier column.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Initialize remedial flags
        df['remedial_marks'] = 0
        df['remedial_attendance'] = 0
        df['remedial_failures'] = 0

        # Check marks-based remedial need
        if marks_col:
            df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
            df['remedial_marks'] = (df[marks_col] < marks_threshold).astype(int)

        # Check attendance-based remedial need
        if attendance_col:
            df[attendance_col] = pd.to_numeric(df[attendance_col], errors='coerce')
            df['remedial_attendance'] = (df[attendance_col] < attendance_threshold).astype(int)

        # Check failures-based remedial need
        if fail_col:
            df[fail_col] = pd.to_numeric(df[fail_col], errors='coerce')
            df['remedial_failures'] = (df[fail_col] >= failure_threshold).astype(int)

        # Calculate total remedial score
        df['remedial_score'] = df[['remedial_marks', 'remedial_attendance', 'remedial_failures']].sum(axis=1)
        df['remedial_priority'] = pd.cut(df['remedial_score'],
                                         bins=[-1, 0, 1, 3],
                                         labels=['No Need', 'Moderate Need', 'High Need'])

        # Identify students needing remedial
        remedial_students = df[df['remedial_score'] > 0]
        high_priority = df[df['remedial_priority'] == 'High Need']

        metrics["total_students"] = len(df)
        metrics["students_needing_remedial"] = len(remedial_students)
        metrics["high_priority_students"] = len(high_priority)
        metrics["remedial_by_marks"] = int(df['remedial_marks'].sum())
        metrics["remedial_by_attendance"] = int(df['remedial_attendance'].sum())
        metrics["remedial_by_failures"] = int(df['remedial_failures'].sum())

        insights.append(f"Total students analyzed: {len(df)}")
        insights.append(f"Students needing remedial support: {len(remedial_students)} ({len(remedial_students)/len(df)*100:.1f}%)")
        insights.append(f"High priority students: {len(high_priority)}")
        insights.append(f"\nRemedial need by factor:")
        insights.append(f"  Low marks (<{marks_threshold}): {df['remedial_marks'].sum()} students")
        insights.append(f"  Low attendance (<{attendance_threshold}%): {df['remedial_attendance'].sum()} students")
        insights.append(f"  Failed subjects: {df['remedial_failures'].sum()} students")

        # Priority distribution
        priority_counts = df['remedial_priority'].value_counts().reset_index()
        priority_counts.columns = ['Priority', 'Count']
        
        fig1 = px.pie(priority_counts, values='Count', names='Priority',
                     title="Remedial Need Priority Distribution")
        visualizations["remedial_pie"] = fig1.to_json()

        # List high priority students
        if len(high_priority) > 0:
            insights.append(f"\nHigh Priority Students (Top 10):")
            for idx, row in high_priority.head(10).iterrows():
                student_name = row.get(student_col, f"Student {idx}")
                reasons = []
                if row['remedial_marks']:
                    reasons.append('low marks')
                if row['remedial_attendance']:
                    reasons.append('low attendance')
                if row['remedial_failures']:
                    reasons.append('failures')
                insights.append(f"  {student_name}: {', '.join(reasons)}")

        # Subject-wise remedial need if subject column exists
        if subject_col and subject_col in df.columns:
            subject_remedial = df.groupby(subject_col)['remedial_marks'].sum().reset_index()
            subject_remedial.columns = [subject_col, 'students_needing_remedial']
            subject_remedial = subject_remedial.sort_values('students_needing_remedial', ascending=False)
            
            fig2 = px.bar(subject_remedial.head(10), x=subject_col, y='students_needing_remedial',
                         title="Top 10 Subjects Needing Remedial Support")
            visualizations["subject_remedial"] = fig2.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def performance_gap_analysis(df):
    """Performance gaps analysis"""
    analysis_type = "performance_gap_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Performance Gap Analysis ---"]
        
        expected = ['subject', 'section', 'batch', 'internal_marks', 'external_marks', 'theory_marks', 'practical_marks', 'expected_marks', 'actual_marks']
        matched = fuzzy_match_column(df, expected)

        # Find columns for gap analysis
        comparison_pairs = []
        
        # Internal vs External gap
        if matched.get('internal_marks') and matched.get('external_marks'):
            internal_col = matched['internal_marks']
            external_col = matched['external_marks']
            df[internal_col] = pd.to_numeric(df[internal_col], errors='coerce')
            df[external_col] = pd.to_numeric(df[external_col], errors='coerce')
            df['internal_external_gap'] = df[internal_col] - df[external_col]
            comparison_pairs.append(('Internal', 'External', 'internal_external_gap'))
        
        # Theory vs Practical gap
        if matched.get('theory_marks') and matched.get('practical_marks'):
            theory_col = matched['theory_marks']
            practical_col = matched['practical_marks']
            df[theory_col] = pd.to_numeric(df[theory_col], errors='coerce')
            df[practical_col] = pd.to_numeric(df[practical_col], errors='coerce')
            df['theory_practical_gap'] = df[theory_col] - df[practical_col]
            comparison_pairs.append(('Theory', 'Practical', 'theory_practical_gap'))
        
        # Expected vs Actual gap
        if matched.get('expected_marks') and matched.get('actual_marks'):
            expected_col = matched['expected_marks']
            actual_col = matched['actual_marks']
            df[expected_col] = pd.to_numeric(df[expected_col], errors='coerce')
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
            df['expected_actual_gap'] = df[expected_col] - df[actual_col]
            comparison_pairs.append(('Expected', 'Actual', 'expected_actual_gap'))

        if not comparison_pairs:
            insights.append("Could not find columns for gap analysis (need pairs like internal/external, theory/practical, etc.)")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Analyze each gap
        for name1, name2, gap_col in comparison_pairs:
            gap_stats = {
                'mean_gap': df[gap_col].mean(),
                'median_gap': df[gap_col].median(),
                'std_gap': df[gap_col].std(),
                'max_positive_gap': df[gap_col].max(),
                'max_negative_gap': df[gap_col].min(),
                'students_with_positive_gap': (df[gap_col] > 0).sum(),
                'students_with_negative_gap': (df[gap_col] < 0).sum(),
                'students_with_no_gap': (df[gap_col] == 0).sum()
            }
            
            metrics[f"{name1}_{name2}_gap"] = convert_numpy_types(gap_stats)
            
            insights.append(f"\n{name1} vs {name2} Gap Analysis:")
            insights.append(f"  Average gap: {gap_stats['mean_gap']:.2f} ({name1} - {name2})")
            insights.append(f"  Students performing better in {name1}: {gap_stats['students_with_positive_gap']}")
            insights.append(f"  Students performing better in {name2}: {gap_stats['students_with_negative_gap']}")
            
            # Distribution of gaps
            fig = px.histogram(df, x=gap_col, nbins=30,
                              title=f"Distribution of {name1} - {name2} Gap")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            visualizations[f"{name1}_{name2}_gap_dist"] = fig.to_json()

        # Subject-wise gap analysis
        subject_col = matched.get('subject')
        if subject_col and subject_col in df.columns:
            for name1, name2, gap_col in comparison_pairs:
                subject_gaps = df.groupby(subject_col)[gap_col].mean().reset_index()
                subject_gaps.columns = [subject_col, f'avg_gap']
                subject_gaps = subject_gaps.sort_values('avg_gap')
                
                fig = px.bar(subject_gaps, x=subject_col, y='avg_gap',
                            title=f"Average {name1}-{name2} Gap by Subject",
                            color='avg_gap', color_continuous_scale='RdBu')
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                visualizations[f"{name1}_{name2}_gap_by_subject"] = fig.to_json()

        # Section-wise gap analysis
        section_col = matched.get('section') or matched.get('batch')
        if section_col and section_col in df.columns:
            for name1, name2, gap_col in comparison_pairs:
                section_gaps = df.groupby(section_col)[gap_col].mean().reset_index()
                section_gaps.columns = [section_col, f'avg_gap']
                
                fig = px.bar(section_gaps, x=section_col, y='avg_gap',
                            title=f"Average {name1}-{name2} Gap by Section",
                            color='avg_gap', color_continuous_scale='RdBu')
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                visualizations[f"{name1}_{name2}_gap_by_section"] = fig.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def subject_improvement_trend_analysis(df):
    """Subject improvement trend analysis"""
    analysis_type = "subject_improvement_trend_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject Improvement Trend Analysis ---"]
        
        expected = ['subject', 'academic_year', 'semester', 'term', 'average_marks', 'pass_percentage', 'previous_avg', 'improvement']
        matched = fuzzy_match_column(df, expected)

        # Find subject column
        subject_col = None
        for key in ['subject']:
            if matched.get(key):
                subject_col = matched[key]
                break

        # Find time period column
        time_col = None
        for key in ['academic_year', 'semester', 'term']:
            if matched.get(key):
                time_col = matched[key]
                break

        # Find performance metric
        perf_col = None
        for key in ['average_marks', 'pass_percentage']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not subject_col or not time_col or not perf_col:
            missing = []
            if not subject_col:
                missing.append("subject")
            if not time_col:
                missing.append("time period")
            if not perf_col:
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        
        # Sort by time
        try:
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        except:
            # Keep as is if can't convert
            pass
        
        df = df.sort_values([subject_col, time_col])

        # Calculate improvement for each subject
        subject_improvements = []
        
        for subject in df[subject_col].unique():
            subject_data = df[df[subject_col] == subject].copy()
            
            if len(subject_data) >= 2:
                # Calculate period-over-period improvement
                subject_data['prev_value'] = subject_data[perf_col].shift(1)
                subject_data['improvement'] = subject_data[perf_col] - subject_data['prev_value']
                subject_data['improvement_pct'] = (subject_data['improvement'] / subject_data['prev_value'] * 100).where(subject_data['prev_value'] > 0, 0)
                
                first_value = subject_data.iloc[0][perf_col]
                last_value = subject_data.iloc[-1][perf_col]
                total_improvement = last_value - first_value
                total_improvement_pct = (total_improvement / first_value * 100) if first_value > 0 else 0
                
                subject_improvements.append({
                    'subject': subject,
                    'first_value': first_value,
                    'last_value': last_value,
                    'total_improvement': total_improvement,
                    'total_improvement_pct': total_improvement_pct,
                    'avg_period_improvement': subject_data['improvement'].mean(),
                    'num_periods': len(subject_data) - 1
                })
                
                # Store period data for this subject
                metrics[f"{subject}_trend"] = json.loads(subject_data[[time_col, perf_col, 'improvement']].dropna().to_json(orient="split"))

        if subject_improvements:
            imp_df = pd.DataFrame(subject_improvements)
            imp_df = imp_df.sort_values('total_improvement_pct', ascending=False)
            
            metrics["subject_improvements"] = json.loads(imp_df.to_json(orient="split"))
            
            insights.append("Subject Improvement Analysis:")
            insights.append(f"  Subjects showing improvement: {(imp_df['total_improvement'] > 0).sum()}")
            insights.append(f"  Subjects showing decline: {(imp_df['total_improvement'] < 0).sum()}")
            
            # Most improved subjects
            most_improved = imp_df.nlargest(5, 'total_improvement_pct')
            insights.append("\nMost Improved Subjects:")
            for _, row in most_improved.iterrows():
                insights.append(f"  {row['subject']}: +{row['total_improvement_pct']:.1f}% (from {row['first_value']:.1f} to {row['last_value']:.1f})")
            
            # Subjects needing attention
            declined = imp_df[imp_df['total_improvement'] < 0].nsmallest(5, 'total_improvement_pct')
            if len(declined) > 0:
                insights.append("\nSubjects Showing Decline:")
                for _, row in declined.iterrows():
                    insights.append(f"  {row['subject']}: {row['total_improvement_pct']:.1f}% (from {row['first_value']:.1f} to {row['last_value']:.1f})")

            # Line chart for top subjects
            top_subjects = most_improved['subject'].tolist()[:3]
            if top_subjects:
                top_data = df[df[subject_col].isin(top_subjects)]
                
                fig1 = px.line(top_data, x=time_col, y=perf_col, color=subject_col,
                              title="Performance Trend for Most Improved Subjects",
                              markers=True)
                visualizations["top_improvers_trend"] = fig1.to_json()

            # Overall trend across all subjects
            avg_trend = df.groupby(time_col)[perf_col].mean().reset_index()
            fig2 = px.line(avg_trend, x=time_col, y=perf_col,
                          title="Average Performance Trend Across All Subjects",
                          markers=True)
            visualizations["overall_trend"] = fig2.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def attendance_defaulter_analysis(df):
    """Attendance defaulters analysis"""
    analysis_type = "attendance_defaulter_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance Defaulter Analysis ---"]
        
        expected = ['student_id', 'student_name', 'attendance', 'attendance_percentage', 'absent_days', 'total_days', 'default_count']
        matched = fuzzy_match_column(df, expected)

        # Find attendance column
        attendance_col = None
        for key in ['attendance', 'attendance_percentage']:
            if matched.get(key):
                attendance_col = matched[key]
                break

        # Find student identifier
        student_col = matched.get('student_id') or matched.get('student_name')

        if not attendance_col or not student_col:
            missing = []
            if not attendance_col:
                missing.append("attendance")
            if not student_col:
                missing.append("student identifier")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Define attendance thresholds
        critical_threshold = 65
        warning_threshold = 75
        satisfactory_threshold = 85

        df[attendance_col] = pd.to_numeric(df[attendance_col], errors='coerce')
        df = df.dropna(subset=[attendance_col, student_col])

        # Categorize students
        df['attendance_category'] = pd.cut(df[attendance_col],
                                          bins=[0, critical_threshold, warning_threshold, satisfactory_threshold, 100],
                                          labels=['Critical', 'Warning', 'Satisfactory', 'Excellent'])
        
        # Identify defaulters
        defaulters = df[df[attendance_col] < warning_threshold]
        critical_defaulters = df[df[attendance_col] < critical_threshold]

        metrics["total_students"] = len(df)
        metrics["defaulters_count"] = len(defaulters)
        metrics["critical_defaulters_count"] = len(critical_defaulters)
        metrics["defaulters_percentage"] = (len(defaulters) / len(df)) * 100
        metrics["attendance_categories"] = json.loads(df['attendance_category'].value_counts().to_json(orient="split"))

        insights.append(f"Total students: {len(df)}")
        insights.append(f"Attendance defaulters (<{warning_threshold}%): {len(defaulters)} ({metrics['defaulters_percentage']:.1f}%)")
        insights.append(f"Critical defaulters (<{critical_threshold}%): {len(critical_defaulters)}")
        
        # Category distribution
        category_counts = df['attendance_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig1 = px.bar(category_counts, x='Category', y='Count',
                     title="Student Distribution by Attendance Category",
                     color='Count', color_continuous_scale='RdYlGn')
        visualizations["attendance_categories"] = fig1.to_json()

        # List defaulters
        if len(defaulters) > 0:
            insights.append(f"\nTop Defaulters (Lowest Attendance):")
            for idx, row in defaulters.nsmallest(10, attendance_col).iterrows():
                student_name = row.get(student_col, f"Student {idx}")
                insights.append(f"  {student_name}: {row[attendance_col]:.1f}%")

        # Distribution of attendance
        fig2 = px.histogram(df, x=attendance_col, nbins=30,
                           title="Distribution of Attendance Percentage")
        
        # Add threshold lines
        fig2.add_vline(x=warning_threshold, line_dash="dash", line_color="orange",
                      annotation_text=f"Warning ({warning_threshold}%)")
        fig2.add_vline(x=critical_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Critical ({critical_threshold}%)")
        visualizations["attendance_distribution"] = fig2.to_json()

        # Section-wise defaulter analysis if section exists
        section_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['section', 'class', 'batch']):
                section_col = col
                break
        
        if section_col:
            section_defaulters = df[df[attendance_col] < warning_threshold].groupby(section_col).size().reset_index(name='defaulters')
            section_total = df.groupby(section_col).size().reset_index(name='total')
            section_stats = section_total.merge(section_defaulters, on=section_col, how='left').fillna(0)
            section_stats['defaulter_percentage'] = (section_stats['defaulters'] / section_stats['total'] * 100).round(1)
            
            fig3 = px.bar(section_stats, x=section_col, y='defaulter_percentage',
                         title="Defaulter Percentage by Section",
                         color='defaulter_percentage', color_continuous_scale='RdYlGn_r')
            visualizations["section_defaulters"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def marks_outlier_detection_analysis(df):
    """Mark outliers detection"""
    analysis_type = "marks_outlier_detection_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Marks Outlier Detection Analysis ---"]
        
        expected = ['subject', 'marks', 'score', 'student_id', 'internal_marks', 'external_marks']
        matched = fuzzy_match_column(df, expected)

        # Find marks columns to analyze
        marks_cols = []
        for key in ['marks', 'score', 'internal_marks', 'external_marks']:
            if matched.get(key):
                marks_cols.append(matched[key])

        if not marks_cols:
            # Look for numeric columns that could be marks
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            potential_cols = [col for col in numeric_cols if any(x in col.lower() for x in 
                             ['marks', 'score', 'grade', 'total', 'internal', 'external'])]
            if potential_cols:
                marks_cols = potential_cols[:3]

        if not marks_cols:
            insights.append("Could not find marks columns for outlier detection.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        for col in marks_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        outlier_results = {}
        
        for col in marks_cols:
            # Calculate quartiles and IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            low_outliers = df[df[col] < lower_bound]
            high_outliers = df[df[col] > upper_bound]
            
            outlier_results[col] = {
                'total_outliers': len(outliers),
                'low_outliers': len(low_outliers),
                'high_outliers': len(high_outliers),
                'outlier_percentage': (len(outliers) / len(df) * 100),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            insights.append(f"\nOutlier Analysis for {col}:")
            insights.append(f"  Total outliers: {len(outliers)} ({outlier_results[col]['outlier_percentage']:.1f}%)")
            insights.append(f"  Low outliers (below {lower_bound:.2f}): {len(low_outliers)}")
            insights.append(f"  High outliers (above {upper_bound:.2f}): {len(high_outliers)}")
            
            # Box plot with outliers highlighted
            fig = px.box(df, y=col, title=f"Box Plot of {col} with Outliers",
                        points='outliers')
            visualizations[f"{col}_boxplot"] = fig.to_json()
            
            # Distribution with outlier boundaries
            fig2 = px.histogram(df, x=col, nbins=30,
                               title=f"Distribution of {col} with Outlier Boundaries")
            fig2.add_vline(x=lower_bound, line_dash="dash", line_color="red",
                          annotation_text="Lower Bound")
            fig2.add_vline(x=upper_bound, line_dash="dash", line_color="red",
                          annotation_text="Upper Bound")
            visualizations[f"{col}_distribution"] = fig2.to_json()

        metrics["outlier_analysis"] = convert_numpy_types(outlier_results)

        # Subject-wise outlier analysis
        subject_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['subject', 'course']):
                subject_col = col
                break

        if subject_col and len(marks_cols) > 0:
            primary_col = marks_cols[0]
            subject_outliers = []
            
            for subject in df[subject_col].unique():
                subject_data = df[df[subject_col] == subject][primary_col].dropna()
                if len(subject_data) > 0:
                    Q1 = subject_data.quantile(0.25)
                    Q3 = subject_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    
                    outliers = subject_data[(subject_data < lower) | (subject_data > upper)]
                    
                    subject_outliers.append({
                        'subject': subject,
                        'total_students': len(subject_data),
                        'outliers': len(outliers),
                        'outlier_percentage': (len(outliers) / len(subject_data) * 100)
                    })
            
            if subject_outliers:
                outlier_df = pd.DataFrame(subject_outliers).sort_values('outlier_percentage', ascending=False)
                fig3 = px.bar(outlier_df.head(10), x='subject', y='outlier_percentage',
                             title="Top 10 Subjects by Outlier Percentage")
                visualizations["subject_outliers"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def course_outcome_attainment_analysis(df):
    """Course outcome attainment analysis"""
    analysis_type = "course_outcome_attainment_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Course Outcome Attainment Analysis ---"]
        
        expected = ['course_code', 'course_name', 'co_number', 'co_description', 'attainment_level', 'target_percentage', 'achieved_percentage']
        matched = fuzzy_match_column(df, expected)

        # Find course identifier
        course_col = None
        for key in ['course_code', 'course_name']:
            if matched.get(key):
                course_col = matched[key]
                break

        # Find CO identifier
        co_col = None
        for key in ['co_number', 'co_description']:
            if matched.get(key):
                co_col = matched[key]
                break

        # Find attainment metric
        attainment_col = None
        for key in ['attainment_level', 'achieved_percentage']:
            if matched.get(key):
                attainment_col = matched[key]
                break

        if not course_col or not co_col or not attainment_col:
            missing = []
            if not course_col:
                missing.append("course")
            if not co_col:
                missing.append("CO")
            if not attainment_col:
                missing.append("attainment")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert attainment to numeric
        df[attainment_col] = pd.to_numeric(df[attainment_col], errors='coerce')
        
        # Define attainment thresholds (typical values)
        threshold_1 = 40  # Level 1 threshold
        threshold_2 = 60  # Level 2 threshold
        threshold_3 = 70  # Level 3 threshold

        # Calculate attainment levels
        df['attainment_level'] = pd.cut(df[attainment_col],
                                        bins=[0, threshold_1, threshold_2, threshold_3, 100],
                                        labels=['Below Level 1', 'Level 1', 'Level 2', 'Level 3'])
        
        # Course-wise summary
        course_summary = df.groupby(course_col).agg({
            attainment_col: ['mean', 'std'],
            co_col: 'count',
            'attainment_level': lambda x: (x == 'Level 3').sum()
        }).reset_index()
        
        course_summary.columns = [course_col, 'avg_attainment', 'std_attainment', 'total_cos', 'cos_at_level_3']
        course_summary['level_3_percentage'] = (course_summary['cos_at_level_3'] / course_summary['total_cos'] * 100).round(1)

        metrics["course_attainment"] = json.loads(course_summary.to_json(orient="split"))
        metrics["overall_avg_attainment"] = df[attainment_col].mean()

        insights.append(f"Overall Average Attainment: {df[attainment_col].mean():.2f}%")
        insights.append(f"Number of courses analyzed: {len(course_summary)}")
        insights.append(f"Total Course Outcomes: {len(df)}")

        # Attainment distribution
        attainment_levels = df['attainment_level'].value_counts().reset_index()
        attainment_levels.columns = ['Level', 'Count']
        
        fig1 = px.pie(attainment_levels, values='Count', names='Level',
                     title="Course Outcome Attainment Levels Distribution")
        visualizations["attainment_pie"] = fig1.to_json()

        # Bar chart of course attainment
        fig2 = px.bar(course_summary.sort_values('avg_attainment'), 
                     x=course_col, y='avg_attainment',
                     error_y='std_attainment',
                     title="Average Attainment by Course",
                     color='avg_attainment', color_continuous_scale='RdYlGn')
        
        # Add threshold lines
        fig2.add_hline(y=threshold_1, line_dash="dash", line_color="red",
                      annotation_text=f"Level 1 ({threshold_1}%)")
        fig2.add_hline(y=threshold_2, line_dash="dash", line_color="orange",
                      annotation_text=f"Level 2 ({threshold_2}%)")
        fig2.add_hline(y=threshold_3, line_dash="dash", line_color="green",
                      annotation_text=f"Level 3 ({threshold_3}%)")
        visualizations["course_attainment"] = fig2.to_json()

        # CO attainment heatmap
        pivot_df = df.pivot_table(values=attainment_col, index=course_col, columns=co_col, aggfunc='mean')
        if not pivot_df.empty:
            fig3 = px.imshow(pivot_df, text_auto=True, aspect='auto',
                            title="Course Outcome Attainment Heatmap",
                            color_continuous_scale='RdYlGn')
            visualizations["co_heatmap"] = fig3.to_json()

        # Identify courses needing attention
        low_attainment = course_summary[course_summary['avg_attainment'] < threshold_1]
        if len(low_attainment) > 0:
            insights.append(f"\n⚠️ Courses below Level 1 threshold ({threshold_1}%):")
            for _, row in low_attainment.iterrows():
                insights.append(f"  {row[course_col]}: {row['avg_attainment']:.1f}%")

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def student_engagement_score_analysis(df):
    """Student engagement score analysis"""
    analysis_type = "student_engagement_score_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Student Engagement Score Analysis ---"]
        
        expected = ['student_id', 'attendance', 'assignment_submission', 'class_participation', 'quiz_score', 'engagement_score']
        matched = fuzzy_match_column(df, expected)

        # Find engagement components
        engagement_components = []
        
        # Check for explicit engagement score
        if matched.get('engagement_score'):
            engagement_col = matched['engagement_score']
            df[engagement_col] = pd.to_numeric(df[engagement_col], errors='coerce')
            engagement_components = [engagement_col]
            insights.append("Using pre-calculated engagement score.")
        else:
            # Look for component columns
            component_keys = ['attendance', 'assignment_submission', 'class_participation', 'quiz_score']
            for key in component_keys:
                if matched.get(key):
                    col = matched[key]
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    engagement_components.append(col)

        # Find student identifier
        student_col = matched.get('student_id')

        if not engagement_components or not student_col:
            missing = []
            if not engagement_components:
                missing.append("engagement components")
            if not student_col:
                missing.append("student identifier")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate composite engagement score if not pre-calculated
        if len(engagement_components) > 1:
            # Normalize each component to 0-100 scale
            for col in engagement_components:
                if df[col].max() <= 1:  # If already normalized
                    df[f'{col}_norm'] = df[col] * 100
                elif df[col].max() <= 100:  # If on 0-100 scale
                    df[f'{col}_norm'] = df[col]
                else:
                    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100
            
            norm_cols = [f'{col}_norm' for col in engagement_components]
            df['engagement_score'] = df[norm_cols].mean(axis=1)
            engagement_col = 'engagement_score'
        else:
            engagement_col = engagement_components[0]

        # Categorize engagement levels
        df['engagement_level'] = pd.cut(df[engagement_col],
                                       bins=[0, 40, 60, 80, 100],
                                       labels=['Low', 'Moderate', 'Good', 'High'])

        # Engagement statistics
        metrics["avg_engagement_score"] = df[engagement_col].mean()
        metrics["median_engagement"] = df[engagement_col].median()
        metrics["engagement_std"] = df[engagement_col].std()
        metrics["engagement_levels"] = json.loads(df['engagement_level'].value_counts().to_json(orient="split"))

        insights.append(f"Average Engagement Score: {df[engagement_col].mean():.2f}")
        insights.append(f"Median Engagement: {df[engagement_col].median():.2f}")
        insights.append(f"Engagement Score Range: {df[engagement_col].min():.2f} - {df[engagement_col].max():.2f}")

        # Level distribution
        level_counts = df['engagement_level'].value_counts().reset_index()
        level_counts.columns = ['Level', 'Count']
        
        fig1 = px.pie(level_counts, values='Count', names='Level',
                     title="Student Engagement Level Distribution")
        visualizations["engagement_pie"] = fig1.to_json()

        # Engagement score distribution
        fig2 = px.histogram(df, x=engagement_col, nbins=30,
                           title="Distribution of Engagement Scores",
                           color_discrete_sequence=['green'])
        visualizations["engagement_distribution"] = fig2.to_json()

        # Low engagement students
        low_engagement = df[df['engagement_level'] == 'Low']
        if len(low_engagement) > 0:
            metrics["low_engagement_students"] = len(low_engagement)
            insights.append(f"\nStudents with Low Engagement: {len(low_engagement)} ({len(low_engagement)/len(df)*100:.1f}%)")
            
            if len(low_engagement) <= 20:
                insights.append("Low engagement students:")
                for idx, row in low_engagement.iterrows():
                    student_name = row.get(student_col, f"Student {idx}")
                    insights.append(f"  {student_name}: {row[engagement_col]:.1f}")

        # Component-wise engagement if multiple components
        if len(engagement_components) > 1:
            component_avgs = df[engagement_components].mean().reset_index()
            component_avgs.columns = ['Component', 'Average']
            
            fig3 = px.bar(component_avgs, x='Component', y='Average',
                         title="Average Engagement by Component",
                         color='Average', color_continuous_scale='Viridis')
            visualizations["component_engagement"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def faculty_performance_feedback_summary_analysis(df):
    """Faculty performance feedback summary"""
    analysis_type = "faculty_performance_feedback_summary_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Faculty Performance Feedback Summary Analysis ---"]
        
        expected = ['faculty_id', 'faculty_name', 'department', 'feedback_score', 'teaching_effectiveness', 'student_rating', 'peer_review', 'overall_rating']
        matched = fuzzy_match_column(df, expected)

        # Find faculty identifier
        faculty_col = None
        for key in ['faculty_id', 'faculty_name']:
            if matched.get(key):
                faculty_col = matched[key]
                break

        # Find rating/feedback columns
        rating_cols = []
        for key in ['feedback_score', 'teaching_effectiveness', 'student_rating', 'peer_review', 'overall_rating']:
            if matched.get(key):
                rating_cols.append(matched[key])

        if not faculty_col or not rating_cols:
            missing = []
            if not faculty_col:
                missing.append("faculty identifier")
            if not rating_cols:
                missing.append("feedback/rating columns")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert ratings to numeric
        for col in rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate overall rating if not present
        if 'overall_rating' not in [matched.get(key) for key in ['overall_rating']]:
            df['calculated_overall'] = df[rating_cols].mean(axis=1)
            overall_col = 'calculated_overall'
        else:
            overall_col = matched['overall_rating']

        # Faculty summary
        faculty_summary = df.groupby(faculty_col).agg({
            overall_col: ['mean', 'std', 'count'],
            **{col: 'mean' for col in rating_cols if col != overall_col}
        }).reset_index()
        
        # Flatten columns
        faculty_summary.columns = ['_'.join(col).strip('_') for col in faculty_summary.columns.values]
        faculty_summary = faculty_summary.sort_values(f'{overall_col}_mean', ascending=False)

        metrics["faculty_feedback"] = json.loads(faculty_summary.to_json(orient="split"))
        metrics["average_overall_rating"] = df[overall_col].mean()

        insights.append(f"Number of faculty evaluated: {len(faculty_summary)}")
        insights.append(f"Average Overall Rating: {df[overall_col].mean():.2f}")
        insights.append(f"Rating Range: {df[overall_col].min():.2f} - {df[overall_col].max():.2f}")

        # Top performers
        top_faculty = faculty_summary.head(5)
        insights.append("\nTop Rated Faculty:")
        for _, row in top_faculty.iterrows():
            faculty_name = row.get(faculty_col + '_', 'Unknown')
            insights.append(f"  {faculty_name}: {row[f'{overall_col}_mean']:.2f}")

        # Rating distribution
        fig1 = px.histogram(df, x=overall_col, nbins=20,
                           title="Distribution of Faculty Overall Ratings")
        visualizations["rating_distribution"] = fig1.to_json()

        # Component-wise ratings
        if len(rating_cols) > 1:
            component_avgs = df[rating_cols].mean().reset_index()
            component_avgs.columns = ['Component', 'Average']
            
            fig2 = px.bar(component_avgs, x='Component', y='Average',
                         title="Average Rating by Component",
                         color='Average', color_continuous_scale='Viridis')
            visualizations["component_ratings"] = fig2.to_json()

        # Department-wise analysis
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_summary = df.groupby(dept_col)[overall_col].mean().reset_index()
            dept_summary = dept_summary.sort_values(overall_col, ascending=False)
            
            fig3 = px.bar(dept_summary, x=dept_col, y=overall_col,
                         title="Average Faculty Rating by Department",
                         color=overall_col, color_continuous_scale='RdYlGn')
            visualizations["dept_ratings"] = fig3.to_json()

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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


# ========== MAIN FUNCTION FOR FACULTY ANALYTICS ==========

def main_faculty():
    """Main function to run Faculty Analytics"""
    print("=" * 60)
    print("👨‍🏫 FACULTY ANALYTICS SYSTEM")
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
        return

    print("\n✅ Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Faculty analytics function mapping
    faculty_function_mapping = {
        "subject_pass_percentage_analysis": subject_pass_percentage_analysis,
        "class_average_score_analysis": class_average_score_analysis,
        "section_wise_performance_comparison": section_wise_performance_comparison,
        "top_performing_students_identification": top_performing_students_identification,
        "low_performing_students_identification": low_performing_students_identification,
        "subject_difficulty_index_analysis": subject_difficulty_index_analysis,
        "internal_marks_variation_analysis": internal_marks_variation_analysis,
        "attendance_pattern_analysis": attendance_pattern_analysis,
        "exam_wise_result_trend_analysis": exam_wise_result_trend_analysis,
        "grade_distribution_per_subject_analysis": grade_distribution_per_subject_analysis,
        "question_difficulty_pattern_analysis": question_difficulty_pattern_analysis,
        "assessment_component_weightage_analysis": assessment_component_weightage_analysis,
        "remedial_need_identification_analysis": remedial_need_identification_analysis,
        "performance_gap_analysis": performance_gap_analysis,
        "subject_improvement_trend_analysis": subject_improvement_trend_analysis,
        "attendance_defaulter_analysis": attendance_defaulter_analysis,
        "marks_outlier_detection_analysis": marks_outlier_detection_analysis,
        "course_outcome_attainment_analysis": course_outcome_attainment_analysis,
        "student_engagement_score_analysis": student_engagement_score_analysis,
        "faculty_performance_feedback_summary_analysis": faculty_performance_feedback_summary_analysis,
    }

    # Analysis Selection
    print("\n" + "=" * 60)
    print("📋 Select a Faculty Analytics Analysis to Perform:")
    print("=" * 60)
    
    analysis_names = list(faculty_function_mapping.keys())
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
            selected_function = faculty_function_mapping.get(selected_analysis_key)
            
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
                    import traceback
                    traceback.print_exc()
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
            result = general_insights_analysis(df, "Faculty Data Overview")
            
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
    result = main_faculty()
    print("\n" + "=" * 60)
    print("✅ Faculty Analytics Complete")
    print("=" * 60)