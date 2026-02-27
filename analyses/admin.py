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


# ========== ADMIN ANALYTICS FUNCTIONS ==========

def department_wise_performance_analysis(df):
    """Department-wise performance analysis"""
    analysis_type = "department_wise_performance_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Department-wise Performance Analysis ---"]
        
        expected = ['department', 'dept', 'student_id', 'cgpa', 'gpa', 'percentage', 'total_marks', 'grade']
        matched = fuzzy_match_column(df, expected)

        # Find department column
        dept_col = None
        for key in ['department', 'dept']:
            if matched.get(key):
                dept_col = matched[key]
                break

        if not dept_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['dept', 'department', 'program']):
                    dept_col = col
                    matched['department'] = col
                    break

        # Find performance column
        perf_col = None
        for key in ['cgpa', 'gpa', 'percentage', 'total_marks']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not dept_col or not perf_col:
            missing = []
            if not dept_col:
                missing.append("department")
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

        df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        df = df.dropna(subset=[dept_col, perf_col])

        # Department-wise statistics
        dept_stats = df.groupby(dept_col)[perf_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
        dept_stats.columns = [dept_col, 'Average', 'Median', 'Std_Dev', 'Min', 'Max', 'Student_Count']
        dept_stats = dept_stats.sort_values('Average', ascending=False)

        metrics["department_performance"] = json.loads(dept_stats.to_json(orient="split"))
        metrics["top_department"] = str(dept_stats.iloc[0][dept_col])
        metrics["bottom_department"] = str(dept_stats.iloc[-1][dept_col])
        metrics["overall_average"] = df[perf_col].mean()

        insights.append(f"Overall Average {perf_col}: {df[perf_col].mean():.2f}")
        insights.append(f"Number of departments: {len(dept_stats)}")
        insights.append(f"Top performing department: {dept_stats.iloc[0][dept_col]} (Avg: {dept_stats.iloc[0]['Average']:.2f})")
        insights.append(f"Lowest performing department: {dept_stats.iloc[-1][dept_col]} (Avg: {dept_stats.iloc[-1]['Average']:.2f})")
        
        # Performance gap
        performance_gap = dept_stats.iloc[0]['Average'] - dept_stats.iloc[-1]['Average']
        metrics["performance_gap"] = performance_gap
        insights.append(f"Performance gap between top and bottom departments: {performance_gap:.2f}")

        # Bar chart of department averages
        fig1 = px.bar(dept_stats, x=dept_col, y='Average',
                     error_y='Std_Dev',
                     title=f"Average {perf_col} by Department",
                     color='Average', color_continuous_scale='Viridis',
                     text='Average')
        fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        visualizations["dept_averages"] = fig1.to_json()

        # Box plot comparison
        fig2 = px.box(df, x=dept_col, y=perf_col,
                     title=f"Performance Distribution by Department",
                     color=dept_col)
        visualizations["dept_boxplot"] = fig2.to_json()

        # Distribution of students across departments
        dept_sizes = dept_stats[[dept_col, 'Student_Count']].copy()
        fig3 = px.pie(dept_sizes, values='Student_Count', names=dept_col,
                     title="Student Distribution by Department")
        visualizations["dept_distribution"] = fig3.to_json()

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


def year_wise_result_trend_analysis(df):
    """Year-wise result trend analysis"""
    analysis_type = "year_wise_result_trend_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Year-wise Result Trend Analysis ---"]
        
        expected = ['academic_year', 'year', 'batch', 'pass_percentage', 'average_marks', 'total_students', 'passed_students']
        matched = fuzzy_match_column(df, expected)

        # Find year column
        year_col = None
        for key in ['academic_year', 'year', 'batch']:
            if matched.get(key):
                year_col = matched[key]
                break

        # Find performance metrics
        pass_col = matched.get('pass_percentage')
        avg_col = matched.get('average_marks')
        total_col = matched.get('total_students')
        passed_col = matched.get('passed_students')

        if not year_col or not (pass_col or avg_col):
            missing = []
            if not year_col:
                missing.append("year")
            if not (pass_col or avg_col):
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Prepare data
        trend_data = df.copy()
        
        # Convert year to string for consistent handling
        trend_data[year_col] = trend_data[year_col].astype(str)
        
        # Sort by year (try to extract numeric part for sorting)
        try:
            trend_data['year_numeric'] = trend_data[year_col].str.extract('(\d+)').astype(float)
            trend_data = trend_data.sort_values('year_numeric')
        except:
            trend_data = trend_data.sort_values(year_col)

        # Calculate pass percentage if not provided
        if not pass_col and total_col and passed_col:
            trend_data[total_col] = pd.to_numeric(trend_data[total_col], errors='coerce')
            trend_data[passed_col] = pd.to_numeric(trend_data[passed_col], errors='coerce')
            trend_data['calculated_pass_percentage'] = (trend_data[passed_col] / trend_data[total_col] * 100).round(2)
            pass_col = 'calculated_pass_percentage'
            insights.append("Pass percentage calculated from total and passed students.")

        # Convert metrics to numeric
        if pass_col:
            trend_data[pass_col] = pd.to_numeric(trend_data[pass_col], errors='coerce')
        if avg_col:
            trend_data[avg_col] = pd.to_numeric(trend_data[avg_col], errors='coerce')

        # Yearly trends
        yearly_data = trend_data.groupby(year_col).agg({
            pass_col: 'mean' if pass_col else None,
            avg_col: 'mean' if avg_col else None,
            total_col: 'sum' if total_col else None
        }).reset_index()

        metrics["yearly_trends"] = json.loads(yearly_data.to_json(orient="split"))

        # Insights
        insights.append(f"Years analyzed: {len(yearly_data)}")
        
        if pass_col:
            best_year = yearly_data.loc[yearly_data[pass_col].idxmax(), year_col]
            worst_year = yearly_data.loc[yearly_data[pass_col].idxmin(), year_col]
            insights.append(f"Best pass percentage: {yearly_data[pass_col].max():.1f}% ({best_year})")
            insights.append(f"Lowest pass percentage: {yearly_data[pass_col].min():.1f}% ({worst_year})")
            
            # Improvement trend
            if len(yearly_data) >= 2:
                first_year = yearly_data.iloc[0]
                last_year = yearly_data.iloc[-1]
                improvement = last_year[pass_col] - first_year[pass_col]
                metrics["overall_improvement"] = improvement
                insights.append(f"Overall improvement from {first_year[year_col]} to {last_year[year_col]}: {improvement:+.1f}%")

        # Line chart for pass percentage
        if pass_col:
            fig1 = px.line(yearly_data, x=year_col, y=pass_col,
                          title="Year-wise Pass Percentage Trend",
                          markers=True)
            fig1.add_hline(y=60, line_dash="dash", line_color="orange",
                          annotation_text="Target (60%)")
            visualizations["pass_trend"] = fig1.to_json()

        # Line chart for average marks
        if avg_col:
            fig2 = px.line(yearly_data, x=year_col, y=avg_col,
                          title="Year-wise Average Marks Trend",
                          markers=True)
            visualizations["avg_trend"] = fig2.to_json()

        # Combined chart if both metrics exist
        if pass_col and avg_col:
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig3.add_trace(
                go.Scatter(x=yearly_data[year_col], y=yearly_data[pass_col],
                          name="Pass %", mode='lines+markers'),
                secondary_y=False
            )
            
            fig3.add_trace(
                go.Scatter(x=yearly_data[year_col], y=yearly_data[avg_col],
                          name="Average Marks", mode='lines+markers'),
                secondary_y=True
            )
            
            fig3.update_layout(title="Year-wise Performance Trends")
            fig3.update_xaxes(title_text="Year")
            fig3.update_yaxes(title_text="Pass Percentage (%)", secondary_y=False)
            fig3.update_yaxes(title_text="Average Marks", secondary_y=True)
            
            visualizations["combined_trend"] = fig3.to_json()

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


def overall_pass_percentage_analysis(df):
    """Overall pass percentage analysis"""
    analysis_type = "overall_pass_percentage_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Overall Pass Percentage Analysis ---"]
        
        expected = ['student_id', 'result', 'status', 'grade', 'marks', 'pass_fail', 'department', 'batch']
        matched = fuzzy_match_column(df, expected)

        # Find result/pass status column
        result_col = None
        for key in ['result', 'status', 'pass_fail', 'grade']:
            if matched.get(key):
                result_col = matched[key]
                break

        # If no result column, try marks column with threshold
        marks_col = matched.get('marks') if not result_col else None

        if not result_col and not marks_col:
            insights.append("Could not find result status or marks column.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Determine pass/fail
        if result_col:
            result_str = df[result_col].astype(str).str.lower()
            pass_indicators = ['pass', 'p', 'passed', 'a+', 'a', 'b+', 'b', 'c', 'd', '1', 'true', 'yes']
            fail_indicators = ['fail', 'f', 'failed', 'arrear', 'ra', '0', 'false', 'no']
            
            df['is_pass'] = result_str.apply(lambda x: 
                1 if any(ind in x for ind in pass_indicators) and not any(ind in x for ind in fail_indicators) 
                else 0 if any(ind in x for ind in fail_indicators) 
                else None)
        else:
            df[marks_col] = pd.to_numeric(df[marks_col], errors='coerce')
            pass_threshold = 40
            df['is_pass'] = (df[marks_col] >= pass_threshold).astype(int)
            insights.append(f"Using marks column with pass threshold {pass_threshold}")

        df = df.dropna(subset=['is_pass'])

        # Overall statistics
        total_students = len(df)
        passed_students = df['is_pass'].sum()
        pass_percentage = (passed_students / total_students) * 100

        metrics["total_students"] = total_students
        metrics["passed_students"] = int(passed_students)
        metrics["failed_students"] = total_students - int(passed_students)
        metrics["overall_pass_percentage"] = pass_percentage

        insights.append(f"Total Students: {total_students}")
        insights.append(f"Passed: {passed_students}")
        insights.append(f"Failed: {total_students - passed_students}")
        insights.append(f"Overall Pass Percentage: {pass_percentage:.2f}%")

        # Pass/Fail pie chart
        pass_fail_data = pd.DataFrame({
            'Status': ['Pass', 'Fail'],
            'Count': [passed_students, total_students - passed_students]
        })
        
        fig1 = px.pie(pass_fail_data, values='Count', names='Status',
                     title="Overall Pass/Fail Distribution",
                     color='Status', color_discrete_map={'Pass': 'green', 'Fail': 'red'})
        visualizations["pass_fail_pie"] = fig1.to_json()

        # Department-wise pass percentage
        dept_col = matched.get('department') or matched.get('batch')
        if dept_col and dept_col in df.columns:
            dept_pass = df.groupby(dept_col)['is_pass'].agg(['count', 'sum']).reset_index()
            dept_pass.columns = [dept_col, 'total', 'passed']
            dept_pass['pass_percentage'] = (dept_pass['passed'] / dept_pass['total'] * 100).round(2)
            dept_pass = dept_pass.sort_values('pass_percentage', ascending=False)
            
            metrics["dept_pass_rates"] = json.loads(dept_pass.to_json(orient="split"))
            
            fig2 = px.bar(dept_pass, x=dept_col, y='pass_percentage',
                         title="Pass Percentage by Department",
                         color='pass_percentage', color_continuous_scale='RdYlGn',
                         text='pass_percentage')
            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig2.add_hline(y=pass_percentage, line_dash="dash", line_color="blue",
                          annotation_text=f"Institutional Avg: {pass_percentage:.1f}%")
            visualizations["dept_pass_rates"] = fig2.to_json()

        # Batch-wise pass percentage
        batch_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['batch', 'year', 'semester']):
                batch_col = col
                break
        
        if batch_col:
            batch_pass = df.groupby(batch_col)['is_pass'].agg(['count', 'sum']).reset_index()
            batch_pass.columns = [batch_col, 'total', 'passed']
            batch_pass['pass_percentage'] = (batch_pass['passed'] / batch_pass['total'] * 100).round(2)
            
            fig3 = px.line(batch_pass, x=batch_col, y='pass_percentage',
                          title="Pass Percentage Trend by Batch",
                          markers=True)
            visualizations["batch_pass_trend"] = fig3.to_json()

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


def cgpa_distribution_analysis(df):
    """CGPA distribution analysis"""
    analysis_type = "cgpa_distribution_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- CGPA Distribution Analysis ---"]
        
        expected = ['cgpa', 'gpa', 'student_id', 'department', 'batch', 'grade']
        matched = fuzzy_match_column(df, expected)

        # Find CGPA column
        cgpa_col = None
        for key in ['cgpa', 'gpa']:
            if matched.get(key):
                cgpa_col = matched[key]
                break

        if not cgpa_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['cgpa', 'gpa']):
                    cgpa_col = col
                    matched['cgpa'] = col
                    break

        if not cgpa_col:
            insights.append("Could not find CGPA column.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df[cgpa_col] = pd.to_numeric(df[cgpa_col], errors='coerce')
        df = df.dropna(subset=[cgpa_col])

        # CGPA statistics
        metrics["mean_cgpa"] = df[cgpa_col].mean()
        metrics["median_cgpa"] = df[cgpa_col].median()
        metrics["std_cgpa"] = df[cgpa_col].std()
        metrics["min_cgpa"] = df[cgpa_col].min()
        metrics["max_cgpa"] = df[cgpa_col].max()
        metrics["q1_cgpa"] = df[cgpa_col].quantile(0.25)
        metrics["q3_cgpa"] = df[cgpa_col].quantile(0.75)

        # Define CGPA ranges/categories
        cgpa_ranges = [0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        cgpa_labels = ['0-4.0', '4.0-5.0', '5.0-6.0', '6.0-7.0', '7.0-8.0', '8.0-9.0', '9.0-10.0']
        
        df['cgpa_range'] = pd.cut(df[cgpa_col], bins=cgpa_ranges, labels=cgpa_labels, right=False)
        range_counts = df['cgpa_range'].value_counts().sort_index().reset_index()
        range_counts.columns = ['CGPA Range', 'Count']
        
        metrics["cgpa_distribution"] = json.loads(range_counts.to_json(orient="split"))

        insights.append(f"CGPA Statistics:")
        insights.append(f"  Mean: {df[cgpa_col].mean():.2f}")
        insights.append(f"  Median: {df[cgpa_col].median():.2f}")
        insights.append(f"  Range: {df[cgpa_col].min():.2f} - {df[cgpa_col].max():.2f}")
        insights.append(f"  Q1 - Q3: {df[cgpa_col].quantile(0.25):.2f} - {df[cgpa_col].quantile(0.75):.2f}")

        # Students in top CGPA brackets
        top_students = (df[cgpa_col] >= 8.0).sum()
        excellent_students = (df[cgpa_col] >= 9.0).sum()
        
        metrics["students_above_8"] = int(top_students)
        metrics["students_above_9"] = int(excellent_students)
        
        insights.append(f"Students with CGPA ≥ 8.0: {top_students} ({top_students/len(df)*100:.1f}%)")
        insights.append(f"Students with CGPA ≥ 9.0: {excellent_students} ({excellent_students/len(df)*100:.1f}%)")

        # Histogram
        fig1 = px.histogram(df, x=cgpa_col, nbins=30,
                           title="CGPA Distribution",
                           labels={cgpa_col: 'CGPA'},
                           marginal='box')
        
        # Add mean and median lines
        fig1.add_vline(x=df[cgpa_col].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df[cgpa_col].mean():.2f}")
        fig1.add_vline(x=df[cgpa_col].median(), line_dash="dot", line_color="green",
                      annotation_text=f"Median: {df[cgpa_col].median():.2f}")
        
        visualizations["cgpa_histogram"] = fig1.to_json()

        # Bar chart of CGPA ranges
        fig2 = px.bar(range_counts, x='CGPA Range', y='Count',
                     title="Student Distribution by CGPA Range",
                     color='Count', color_continuous_scale='Viridis',
                     text='Count')
        visualizations["cgpa_ranges"] = fig2.to_json()

        # Department-wise CGPA distribution
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_cgpa = df.groupby(dept_col)[cgpa_col].agg(['mean', 'median', 'std', 'count']).reset_index()
            dept_cgpa.columns = [dept_col, 'Mean CGPA', 'Median CGPA', 'Std Dev', 'Student Count']
            dept_cgpa = dept_cgpa.sort_values('Mean CGPA', ascending=False)
            
            fig3 = px.bar(dept_cgpa, x=dept_col, y='Mean CGPA',
                         error_y='Std Dev',
                         title="Average CGPA by Department",
                         color='Mean CGPA', color_continuous_scale='Viridis')
            visualizations["dept_cgpa"] = fig3.to_json()

        # Box plot for outlier detection
        fig4 = px.box(df, y=cgpa_col, title="CGPA Box Plot")
        visualizations["cgpa_boxplot"] = fig4.to_json()

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


def department_comparison_analysis(df):
    """Department comparison analysis"""
    analysis_type = "department_comparison_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Department Comparison Analysis ---"]
        
        expected = ['department', 'dept', 'cgpa', 'gpa', 'pass_percentage', 'total_students', 'academic_year']
        matched = fuzzy_match_column(df, expected)

        # Find department column
        dept_col = None
        for key in ['department', 'dept']:
            if matched.get(key):
                dept_col = matched[key]
                break

        # Find performance metrics
        perf_cols = []
        for key in ['cgpa', 'gpa', 'pass_percentage']:
            if matched.get(key):
                perf_cols.append(matched[key])

        if not dept_col or not perf_cols:
            missing = []
            if not dept_col:
                missing.append("department")
            if not perf_cols:
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert metrics to numeric
        for col in perf_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Department comparison
        comparison_data = []
        
        for col in perf_cols:
            dept_metrics = df.groupby(dept_col)[col].agg(['mean', 'std', 'min', 'max']).reset_index()
            dept_metrics.columns = [dept_col, f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max']
            
            if comparison_data:
                comparison_data[0] = comparison_data[0].merge(dept_metrics, on=dept_col)
            else:
                comparison_data.append(dept_metrics)

        comparison_df = comparison_data[0]
        
        # Calculate rankings
        for col in perf_cols:
            comparison_df[f'{col}_rank'] = comparison_df[f'{col}_mean'].rank(ascending=False).astype(int)
        
        metrics["department_comparison"] = json.loads(comparison_df.to_json(orient="split"))

        insights.append(f"Number of departments compared: {len(comparison_df)}")
        insights.append(f"Performance metrics analyzed: {', '.join(perf_cols)}")

        # Identify top and bottom departments for each metric
        for col in perf_cols:
            top_dept = comparison_df.loc[comparison_df[f'{col}_mean'].idxmax(), dept_col]
            bottom_dept = comparison_df.loc[comparison_df[f'{col}_mean'].idxmin(), dept_col]
            insights.append(f"\n{col}:")
            insights.append(f"  Top: {top_dept} ({comparison_df[f'{col}_mean'].max():.2f})")
            insights.append(f"  Bottom: {bottom_dept} ({comparison_df[f'{col}_mean'].min():.2f})")

        # Create comparison charts
        if len(perf_cols) == 1:
            # Single metric - bar chart
            col = perf_cols[0]
            fig1 = px.bar(comparison_df.sort_values(f'{col}_mean'), 
                         x=dept_col, y=f'{col}_mean',
                         error_y=f'{col}_std',
                         title=f"Department Comparison: {col}",
                         color=f'{col}_mean', color_continuous_scale='Viridis')
            visualizations["single_metric_comparison"] = fig1.to_json()
        else:
            # Multiple metrics - grouped bar chart
            melted_data = []
            for col in perf_cols:
                temp = comparison_df[[dept_col, f'{col}_mean']].copy()
                temp.columns = [dept_col, 'value']
                temp['metric'] = col
                melted_data.append(temp)
            
            melted_df = pd.concat(melted_data, ignore_index=True)
            
            fig1 = px.bar(melted_df, x=dept_col, y='value', color='metric',
                         title="Department Performance Comparison",
                         barmode='group')
            visualizations["multi_metric_comparison"] = fig1.to_json()

        # Radar chart for multi-dimensional comparison
        if len(perf_cols) >= 3:
            # Normalize values for radar chart
            radar_data = comparison_df.copy()
            for col in perf_cols:
                col_mean = f'{col}_mean'
                min_val = radar_data[col_mean].min()
                max_val = radar_data[col_mean].max()
                if max_val > min_val:
                    radar_data[f'{col}_norm'] = (radar_data[col_mean] - min_val) / (max_val - min_val) * 100
                else:
                    radar_data[f'{col}_norm'] = 50
            
            fig2 = go.Figure()
            for idx, row in radar_data.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row[f'{col}_norm'] for col in perf_cols],
                    theta=perf_cols,
                    fill='toself',
                    name=row[dept_col]
                ))
            
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Department Performance Radar Chart"
            )
            visualizations["department_radar"] = fig2.to_json()

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


def semester_result_summary_analysis(df):
    """Semester result summary analysis"""
    analysis_type = "semester_result_summary_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Semester Result Summary Analysis ---"]
        
        expected = ['semester', 'sem', 'student_id', 'cgpa', 'gpa', 'result', 'status', 'department']
        matched = fuzzy_match_column(df, expected)

        # Find semester column
        sem_col = None
        for key in ['semester', 'sem']:
            if matched.get(key):
                sem_col = matched[key]
                break

        # Find performance/result column
        perf_col = None
        for key in ['cgpa', 'gpa', 'result', 'status']:
            if matched.get(key):
                perf_col = matched[key]
                break

        if not sem_col or not perf_col:
            missing = []
            if not sem_col:
                missing.append("semester")
            if not perf_col:
                missing.append("performance/result")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Prepare data
        df[sem_col] = pd.to_numeric(df[sem_col], errors='coerce')
        
        if perf_col in ['cgpa', 'gpa']:
            df[perf_col] = pd.to_numeric(df[perf_col], errors='coerce')
        else:
            # Handle result status
            result_str = df[perf_col].astype(str).str.lower()
            pass_indicators = ['pass', 'p', 'passed']
            df['is_pass'] = result_str.apply(lambda x: 1 if any(ind in x for ind in pass_indicators) else 0)

        df = df.dropna(subset=[sem_col])

        # Semester-wise summary
        sem_summary = df.groupby(sem_col).agg({
            perf_col: ['mean', 'std', 'min', 'max'] if perf_col in ['cgpa', 'gpa'] else None,
            'is_pass': ['mean', 'sum', 'count'] if 'is_pass' in df.columns else None,
            'student_id': 'count' if 'student_id' in df.columns else None
        }).reset_index()
        
        # Flatten columns
        sem_summary.columns = ['_'.join(col).strip('_') for col in sem_summary.columns.values]
        sem_summary = sem_summary.sort_values(sem_col)

        metrics["semester_summary"] = json.loads(sem_summary.to_json(orient="split"))

        insights.append(f"Semesters analyzed: {len(sem_summary)}")
        
        if perf_col in ['cgpa', 'gpa']:
            best_sem = sem_summary.loc[sem_summary[f'{perf_col}_mean'].idxmax(), sem_col]
            worst_sem = sem_summary.loc[sem_summary[f'{perf_col}_mean'].idxmin(), sem_col]
            insights.append(f"Best performing semester: Semester {best_sem} (Avg {perf_col}: {sem_summary[f'{perf_col}_mean'].max():.2f})")
            insights.append(f"Lowest performing semester: Semester {worst_sem} (Avg {perf_col}: {sem_summary[f'{perf_col}_mean'].min():.2f})")
        
        if 'is_pass_mean' in sem_summary.columns:
            best_pass_sem = sem_summary.loc[sem_summary['is_pass_mean'].idxmax(), sem_col]
            insights.append(f"Highest pass rate: Semester {best_pass_sem} ({sem_summary['is_pass_mean'].max()*100:.1f}%)")

        # Semester trend charts
        if perf_col in ['cgpa', 'gpa']:
            fig1 = px.line(sem_summary, x=sem_col, y=f'{perf_col}_mean',
                          error_y=f'{perf_col}_std',
                          title=f"Average {perf_col} by Semester",
                          markers=True)
            visualizations["semester_trend"] = fig1.to_json()

        if 'is_pass_mean' in sem_summary.columns:
            fig2 = px.bar(sem_summary, x=sem_col, y='is_pass_mean',
                         title="Pass Rate by Semester",
                         color='is_pass_mean', color_continuous_scale='RdYlGn',
                         text=sem_summary['is_pass_sum'])
            fig2.update_traces(texttemplate='%{text} passed', textposition='outside')
            visualizations["pass_rate_by_semester"] = fig2.to_json()

        # Student count by semester
        if 'student_id_count' in sem_summary.columns:
            fig3 = px.bar(sem_summary, x=sem_col, y='student_id_count',
                         title="Student Count by Semester",
                         color='student_id_count', color_continuous_scale='Viridis')
            visualizations["student_count"] = fig3.to_json()

        # Department-wise semester performance
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns and perf_col in ['cgpa', 'gpa']:
            dept_sem = df.groupby([dept_col, sem_col])[perf_col].mean().reset_index()
            
            fig4 = px.line(dept_sem, x=sem_col, y=perf_col, color=dept_col,
                          title=f"Semester Performance by Department",
                          markers=True)
            visualizations["dept_semester_trend"] = fig4.to_json()

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


def dropout_rate_analysis(df):
    """Dropout rate analysis"""
    analysis_type = "dropout_rate_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Dropout Rate Analysis ---"]
        
        expected = ['academic_year', 'batch', 'enrolled_students', 'continuing_students', 'dropped_students', 'dropout_rate', 'status']
        matched = fuzzy_match_column(df, expected)

        # Find enrollment/dropout columns
        enrolled_col = matched.get('enrolled_students')
        dropped_col = matched.get('dropped_students')
        continuing_col = matched.get('continuing_students')
        rate_col = matched.get('dropout_rate')
        
        # Find year/batch column
        year_col = None
        for key in ['academic_year', 'batch']:
            if matched.get(key):
                year_col = matched[key]
                break

        # Find status column as alternative
        status_col = matched.get('status')

        if not year_col or not (enrolled_col or dropped_col or continuing_col or rate_col or status_col):
            missing = []
            if not year_col:
                missing.append("year/batch")
            if not any([enrolled_col, dropped_col, continuing_col, rate_col, status_col]):
                missing.append("dropout information")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate dropout rates if needed
        if rate_col:
            df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
            df['dropout_rate'] = df[rate_col]
        elif enrolled_col and dropped_col:
            df[enrolled_col] = pd.to_numeric(df[enrolled_col], errors='coerce')
            df[dropped_col] = pd.to_numeric(df[dropped_col], errors='coerce')
            df['dropout_rate'] = (df[dropped_col] / df[enrolled_col] * 100).round(2)
        elif enrolled_col and continuing_col:
            df[enrolled_col] = pd.to_numeric(df[enrolled_col], errors='coerce')
            df[continuing_col] = pd.to_numeric(df[continuing_col], errors='coerce')
            df['dropout_rate'] = ((df[enrolled_col] - df[continuing_col]) / df[enrolled_col] * 100).round(2)

        # Use status column if available
        if status_col and 'student_id' in df.columns:
            status_str = df[status_col].astype(str).str.lower()
            dropout_indicators = ['dropout', 'dropped', 'discontinued', 'withdrawn']
            df['is_dropout'] = status_str.apply(lambda x: 1 if any(ind in x for ind in dropout_indicators) else 0)
            
            year_wise = df.groupby(year_col).agg(
                total_students=('student_id', 'count'),
                dropouts=('is_dropout', 'sum')
            ).reset_index()
            year_wise['dropout_rate'] = (year_wise['dropouts'] / year_wise['total_students'] * 100).round(2)
            
            df = year_wise

        # Year-wise dropout analysis
        if year_col in df.columns:
            df = df.sort_values(year_col)
            
            avg_dropout = df['dropout_rate'].mean()
            max_dropout_year = df.loc[df['dropout_rate'].idxmax(), year_col]
            min_dropout_year = df.loc[df['dropout_rate'].idxmin(), year_col]

            metrics["average_dropout_rate"] = avg_dropout
            metrics["max_dropout_rate"] = df['dropout_rate'].max()
            metrics["min_dropout_rate"] = df['dropout_rate'].min()
            metrics["max_dropout_year"] = str(max_dropout_year)
            metrics["min_dropout_year"] = str(min_dropout_year)

            insights.append(f"Average Dropout Rate: {avg_dropout:.2f}%")
            insights.append(f"Highest Dropout Rate: {df['dropout_rate'].max():.2f}% ({max_dropout_year})")
            insights.append(f"Lowest Dropout Rate: {df['dropout_rate'].min():.2f}% ({min_dropout_year})")

            # Dropout trend
            fig1 = px.line(df, x=year_col, y='dropout_rate',
                          title="Dropout Rate Trend Over Years",
                          markers=True)
            fig1.add_hline(y=avg_dropout, line_dash="dash", line_color="red",
                          annotation_text=f"Avg: {avg_dropout:.2f}%")
            visualizations["dropout_trend"] = fig1.to_json()

            # Bar chart of dropouts
            if 'dropouts' in df.columns:
                fig2 = px.bar(df, x=year_col, y='dropouts',
                             title="Number of Dropouts by Year",
                             color='dropouts', color_continuous_scale='Reds')
                visualizations["dropout_counts"] = fig2.to_json()

            # Identify concerning years
            high_dropout_years = df[df['dropout_rate'] > avg_dropout * 1.5]
            if len(high_dropout_years) > 0:
                insights.append(f"\n⚠️ Years with unusually high dropout rates:")
                for _, row in high_dropout_years.iterrows():
                    insights.append(f"  {row[year_col]}: {row['dropout_rate']:.1f}%")

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


def arrear_rate_analysis(df):
    """Arrear/failure rate analysis"""
    analysis_type = "arrear_rate_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Arrear Rate Analysis ---"]
        
        expected = ['academic_year', 'semester', 'department', 'subject', 'total_appeared', 'passed', 'failed', 'arrear_rate']
        matched = fuzzy_match_column(df, expected)

        # Find required columns
        year_col = matched.get('academic_year') or matched.get('semester')
        dept_col = matched.get('department')
        subject_col = matched.get('subject')
        
        total_col = matched.get('total_appeared')
        passed_col = matched.get('passed')
        failed_col = matched.get('failed')
        rate_col = matched.get('arrear_rate')

        if not (total_col or passed_col or failed_col or rate_col):
            insights.append("Could not find examination statistics columns.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate arrear rate if needed
        if rate_col:
            df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
            df['arrear_rate'] = df[rate_col]
        elif failed_col and total_col:
            df[failed_col] = pd.to_numeric(df[failed_col], errors='coerce')
            df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
            df['arrear_rate'] = (df[failed_col] / df[total_col] * 100).round(2)
        elif passed_col and total_col:
            df[passed_col] = pd.to_numeric(df[passed_col], errors='coerce')
            df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
            df['arrear_rate'] = ((df[total_col] - df[passed_col]) / df[total_col] * 100).round(2)

        # Overall statistics
        overall_arrear_rate = df['arrear_rate'].mean()
        metrics["overall_arrear_rate"] = overall_arrear_rate
        insights.append(f"Overall Arrear Rate: {overall_arrear_rate:.2f}%")

        # Subject-wise arrear rates
        if subject_col and subject_col in df.columns:
            subject_arrear = df.groupby(subject_col)['arrear_rate'].mean().reset_index()
            subject_arrear = subject_arrear.sort_values('arrear_rate', ascending=False)
            
            metrics["subject_arrear_rates"] = json.loads(subject_arrear.to_json(orient="split"))
            
            insights.append("\nSubjects with highest arrear rates:")
            for _, row in subject_arrear.head(5).iterrows():
                insights.append(f"  {row[subject_col]}: {row['arrear_rate']:.1f}%")
            
            fig1 = px.bar(subject_arrear.head(10), x=subject_col, y='arrear_rate',
                         title="Top 10 Subjects by Arrear Rate",
                         color='arrear_rate', color_continuous_scale='Reds',
                         text=subject_arrear.head(10)['arrear_rate'].round(1))
            fig1.update_traces(texttemplate='%{text}%', textposition='outside')
            visualizations["subject_arrear"] = fig1.to_json()

        # Department-wise arrear rates
        if dept_col and dept_col in df.columns:
            dept_arrear = df.groupby(dept_col)['arrear_rate'].mean().reset_index()
            dept_arrear = dept_arrear.sort_values('arrear_rate')
            
            fig2 = px.bar(dept_arrear, x=dept_col, y='arrear_rate',
                         title="Arrear Rate by Department",
                         color='arrear_rate', color_continuous_scale='RdYlGn_r')
            visualizations["dept_arrear"] = fig2.to_json()

        # Year-wise trend
        if year_col and year_col in df.columns:
            year_arrear = df.groupby(year_col)['arrear_rate'].mean().reset_index()
            year_arrear = year_arrear.sort_values(year_col)
            
            fig3 = px.line(year_arrear, x=year_col, y='arrear_rate',
                          title="Arrear Rate Trend Over Time",
                          markers=True)
            visualizations["arrear_trend"] = fig3.to_json()

        # Distribution of arrear rates
        fig4 = px.histogram(df, x='arrear_rate', nbins=20,
                           title="Distribution of Arrear Rates")
        visualizations["arrear_distribution"] = fig4.to_json()

        # Identify concerning areas
        high_arrear = df[df['arrear_rate'] > 30]
        if len(high_arrear) > 0:
            metrics["high_arrear_count"] = len(high_arrear)
            insights.append(f"\n⚠️ {len(high_arrear)} instances with arrear rate > 30%")

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


def placement_eligibility_analysis(df):
    """Placement eligibility analysis"""
    analysis_type = "placement_eligibility_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Placement Eligibility Analysis ---"]
        
        expected = ['student_id', 'cgpa', 'percentage', 'arrears', 'backlogs', 'placement_eligible', 'placed', 'company']
        matched = fuzzy_match_column(df, expected)

        # Find required columns
        student_col = matched.get('student_id')
        cgpa_col = matched.get('cgpa') or matched.get('percentage')
        arrear_col = matched.get('arrears') or matched.get('backlogs')
        eligible_col = matched.get('placement_eligible')
        placed_col = matched.get('placed')

        if not student_col or not (cgpa_col or eligible_col):
            missing = []
            if not student_col:
                missing.append("student identifier")
            if not (cgpa_col or eligible_col):
                missing.append("eligibility criteria")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Define eligibility criteria (typical)
        cgpa_threshold = 6.0
        arrear_threshold = 0

        # Calculate eligibility if not provided
        if eligible_col:
            df[eligible_col] = pd.to_numeric(df[eligible_col], errors='coerce')
            df['is_eligible'] = df[eligible_col]
        else:
            df[cgpa_col] = pd.to_numeric(df[cgpa_col], errors='coerce')
            if arrear_col:
                df[arrear_col] = pd.to_numeric(df[arrear_col], errors='coerce')
                df['is_eligible'] = ((df[cgpa_col] >= cgpa_threshold) & (df[arrear_col] <= arrear_threshold)).astype(int)
            else:
                df['is_eligible'] = (df[cgpa_col] >= cgpa_threshold).astype(int)

        # Placement status
        if placed_col:
            df[placed_col] = pd.to_numeric(df[placed_col], errors='coerce')
            df['is_placed'] = df[placed_col]

        # Eligibility statistics
        total_students = len(df)
        eligible_students = df['is_eligible'].sum()
        eligibility_rate = (eligible_students / total_students) * 100

        metrics["total_students"] = total_students
        metrics["eligible_students"] = int(eligible_students)
        metrics["eligibility_rate"] = eligibility_rate

        insights.append(f"Total Students: {total_students}")
        insights.append(f"Placement Eligible: {eligible_students} ({eligibility_rate:.1f}%)")
        insights.append(f"Not Eligible: {total_students - eligible_students}")

        if placed_col in df.columns:
            placed_students = df['is_placed'].sum()
            placement_rate = (placed_students / eligible_students * 100) if eligible_students > 0 else 0
            overall_placement_rate = (placed_students / total_students) * 100
            
            metrics["placed_students"] = int(placed_students)
            metrics["placement_rate_of_eligible"] = placement_rate
            metrics["overall_placement_rate"] = overall_placement_rate
            
            insights.append(f"Placed Students: {placed_students}")
            insights.append(f"Placement Rate (of eligible): {placement_rate:.1f}%")
            insights.append(f"Overall Placement Rate: {overall_placement_rate:.1f}%")

        # Eligibility pie chart
        elig_data = pd.DataFrame({
            'Status': ['Eligible', 'Not Eligible'],
            'Count': [eligible_students, total_students - eligible_students]
        })
        
        fig1 = px.pie(elig_data, values='Count', names='Status',
                     title="Placement Eligibility Distribution",
                     color='Status', color_discrete_map={'Eligible': 'green', 'Not Eligible': 'red'})
        visualizations["eligibility_pie"] = fig1.to_json()

        # CGPA distribution with eligibility threshold
        if cgpa_col in df.columns:
            fig2 = px.histogram(df, x=cgpa_col, nbins=30,
                               title="CGPA Distribution with Eligibility Threshold",
                               color_discrete_sequence=['blue'])
            fig2.add_vline(x=cgpa_threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Eligibility Threshold ({cgpa_threshold})")
            visualizations["cgpa_distribution"] = fig2.to_json()

        # Department-wise eligibility
        dept_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['dept', 'department', 'program']):
                dept_col = col
                break
        
        if dept_col:
            dept_elig = df.groupby(dept_col).agg(
                total=('is_eligible', 'count'),
                eligible=('is_eligible', 'sum')
            ).reset_index()
            dept_elig['eligibility_rate'] = (dept_elig['eligible'] / dept_elig['total'] * 100).round(1)
            dept_elig = dept_elig.sort_values('eligibility_rate', ascending=False)
            
            fig3 = px.bar(dept_elig, x=dept_col, y='eligibility_rate',
                         title="Placement Eligibility Rate by Department",
                         color='eligibility_rate', color_continuous_scale='RdYlGn',
                         text=dept_elig['eligibility_rate'])
            fig3.update_traces(texttemplate='%{text}%', textposition='outside')
            visualizations["dept_eligibility"] = fig3.to_json()

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


def attendance_compliance_analysis(df):
    """Attendance compliance analysis"""
    analysis_type = "attendance_compliance_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Attendance Compliance Analysis ---"]
        
        expected = ['student_id', 'attendance', 'attendance_percentage', 'required_attendance', 'compliant', 'department', 'batch']
        matched = fuzzy_match_column(df, expected)

        # Find attendance column
        attendance_col = None
        for key in ['attendance', 'attendance_percentage']:
            if matched.get(key):
                attendance_col = matched[key]
                break

        if not attendance_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['attendance', 'present']):
                    attendance_col = col
                    matched['attendance'] = col
                    break

        # Find student identifier
        student_col = matched.get('student_id')

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

        # Define compliance threshold
        required_threshold = 75  # Default required attendance

        df[attendance_col] = pd.to_numeric(df[attendance_col], errors='coerce')
        df = df.dropna(subset=[attendance_col])

        # Determine compliance
        if matched.get('compliant'):
            df['is_compliant'] = pd.to_numeric(df[matched['compliant']], errors='coerce')
        else:
            df['is_compliant'] = (df[attendance_col] >= required_threshold).astype(int)

        # Compliance statistics
        total_students = len(df)
        compliant_students = df['is_compliant'].sum()
        compliance_rate = (compliant_students / total_students) * 100

        metrics["total_students"] = total_students
        metrics["compliant_students"] = int(compliant_students)
        metrics["non_compliant_students"] = total_students - int(compliant_students)
        metrics["compliance_rate"] = compliance_rate
        metrics["required_threshold"] = required_threshold

        insights.append(f"Total Students: {total_students}")
        insights.append(f"Attendance Compliant (≥{required_threshold}%): {compliant_students} ({compliance_rate:.1f}%)")
        insights.append(f"Non-Compliant: {total_students - compliant_students}")

        # Compliance pie chart
        compliance_data = pd.DataFrame({
            'Status': ['Compliant', 'Non-Compliant'],
            'Count': [compliant_students, total_students - compliant_students]
        })
        
        fig1 = px.pie(compliance_data, values='Count', names='Status',
                     title="Attendance Compliance Distribution",
                     color='Status', color_discrete_map={'Compliant': 'green', 'Non-Compliant': 'red'})
        visualizations["compliance_pie"] = fig1.to_json()

        # Attendance distribution
        fig2 = px.histogram(df, x=attendance_col, nbins=30,
                           title="Attendance Percentage Distribution")
        fig2.add_vline(x=required_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Required ({required_threshold}%)")
        visualizations["attendance_distribution"] = fig2.to_json()

        # Department-wise compliance
        dept_col = matched.get('department') or matched.get('batch')
        if dept_col and dept_col in df.columns:
            dept_compliance = df.groupby(dept_col).agg(
                total=('is_compliant', 'count'),
                compliant=('is_compliant', 'sum')
            ).reset_index()
            dept_compliance['compliance_rate'] = (dept_compliance['compliant'] / dept_compliance['total'] * 100).round(1)
            dept_compliance = dept_compliance.sort_values('compliance_rate', ascending=False)
            
            fig3 = px.bar(dept_compliance, x=dept_col, y='compliance_rate',
                         title="Attendance Compliance Rate by Department",
                         color='compliance_rate', color_continuous_scale='RdYlGn',
                         text=dept_compliance['compliance_rate'])
            fig3.update_traces(texttemplate='%{text}%', textposition='outside')
            visualizations["dept_compliance"] = fig3.to_json()

        # List non-compliant students
        non_compliant = df[df['is_compliant'] == 0].nsmallest(20, attendance_col)
        if len(non_compliant) > 0:
            insights.append(f"\nStudents with lowest attendance (sample):")
            for idx, row in non_compliant.iterrows():
                student_name = row.get(student_col, f"Student {idx}")
                insights.append(f"  {student_name}: {row[attendance_col]:.1f}%")

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


def faculty_performance_comparison_analysis(df):
    """Faculty performance comparison analysis"""
    analysis_type = "faculty_performance_comparison_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Faculty Performance Comparison Analysis ---"]
        
        expected = ['faculty_id', 'faculty_name', 'department', 'subject', 'student_feedback', 'pass_percentage', 'average_marks', 'course_completion']
        matched = fuzzy_match_column(df, expected)

        # Find faculty identifier
        faculty_col = None
        for key in ['faculty_id', 'faculty_name']:
            if matched.get(key):
                faculty_col = matched[key]
                break

        # Find performance metrics
        perf_metrics = []
        metric_names = []
        
        metric_keys = {
            'student_feedback': 'Feedback Score',
            'pass_percentage': 'Pass %',
            'average_marks': 'Avg Marks',
            'course_completion': 'Course Completion'
        }
        
        for key, display_name in metric_keys.items():
            if matched.get(key):
                perf_metrics.append(matched[key])
                metric_names.append(display_name)

        if not faculty_col or not perf_metrics:
            missing = []
            if not faculty_col:
                missing.append("faculty identifier")
            if not perf_metrics:
                missing.append("performance metrics")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert metrics to numeric
        for col in perf_metrics:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate composite score
        # Normalize each metric to 0-100 scale
        for i, col in enumerate(perf_metrics):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val) * 100
            else:
                df[f'{col}_normalized'] = 50

        norm_cols = [f'{col}_normalized' for col in perf_metrics]
        df['composite_score'] = df[norm_cols].mean(axis=1)

        # Faculty summary
        faculty_summary = df.groupby(faculty_col).agg({
            'composite_score': 'mean',
            **{col: 'mean' for col in perf_metrics},
            faculty_col: 'count'
        }).reset_index()
        
        faculty_summary.columns = [faculty_col, 'Composite Score'] + metric_names + ['Course_Count']
        faculty_summary = faculty_summary.sort_values('Composite Score', ascending=False)

        metrics["faculty_performance"] = json.loads(faculty_summary.to_json(orient="split"))
        metrics["average_composite_score"] = faculty_summary['Composite Score'].mean()

        insights.append(f"Number of faculty evaluated: {len(faculty_summary)}")
        insights.append(f"Average Composite Score: {faculty_summary['Composite Score'].mean():.2f}")
        insights.append(f"Score Range: {faculty_summary['Composite Score'].min():.2f} - {faculty_summary['Composite Score'].max():.2f}")

        # Top performers
        top_faculty = faculty_summary.head(5)
        insights.append("\nTop Performing Faculty:")
        for _, row in top_faculty.iterrows():
            insights.append(f"  {row[faculty_col]}: {row['Composite Score']:.2f}")

        # Bar chart of composite scores
        fig1 = px.bar(faculty_summary.head(15), x=faculty_col, y='Composite Score',
                     title="Faculty Performance Comparison (Top 15)",
                     color='Composite Score', color_continuous_scale='Viridis')
        visualizations["faculty_comparison"] = fig1.to_json()

        # Radar chart for multi-metric comparison
        if len(metric_names) >= 3:
            # Select top 5 faculty for radar
            top5 = faculty_summary.head(5)
            
            fig2 = go.Figure()
            for _, row in top5.iterrows():
                values = [row[metric] for metric in metric_names]
                fig2.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metric_names,
                    fill='toself',
                    name=str(row[faculty_col])
                ))
            
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Faculty Performance Radar Chart (Top 5)"
            )
            visualizations["faculty_radar"] = fig2.to_json()

        # Department-wise comparison
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_perf = df.groupby(dept_col)['composite_score'].mean().reset_index()
            dept_perf = dept_perf.sort_values('composite_score', ascending=False)
            
            fig3 = px.bar(dept_perf, x=dept_col, y='composite_score',
                         title="Average Faculty Performance by Department",
                         color='composite_score', color_continuous_scale='Viridis')
            visualizations["dept_faculty_perf"] = fig3.to_json()

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


def subject_failure_rate_analysis(df):
    """Subject failure rate analysis"""
    analysis_type = "subject_failure_rate_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Subject Failure Rate Analysis ---"]
        
        expected = ['subject', 'subject_code', 'semester', 'department', 'total_students', 'failed_students', 'failure_rate']
        matched = fuzzy_match_column(df, expected)

        # Find subject column
        subject_col = None
        for key in ['subject', 'subject_code']:
            if matched.get(key):
                subject_col = matched[key]
                break

        # Find failure metrics
        total_col = matched.get('total_students')
        failed_col = matched.get('failed_students')
        rate_col = matched.get('failure_rate')

        if not subject_col or not (total_col or failed_col or rate_col):
            missing = []
            if not subject_col:
                missing.append("subject")
            if not (total_col or failed_col or rate_col):
                missing.append("failure information")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate failure rate if needed
        if rate_col:
            df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
            df['failure_rate'] = df[rate_col]
        elif failed_col and total_col:
            df[failed_col] = pd.to_numeric(df[failed_col], errors='coerce')
            df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
            df['failure_rate'] = (df[failed_col] / df[total_col] * 100).round(2)

        # Subject-wise analysis
        subject_failure = df.groupby(subject_col).agg({
            'failure_rate': 'mean',
            total_col: 'sum' if total_col else None,
            failed_col: 'sum' if failed_col else None
        }).reset_index()
        
        subject_failure = subject_failure.sort_values('failure_rate', ascending=False)

        metrics["subject_failure_rates"] = json.loads(subject_failure.to_json(orient="split"))
        metrics["overall_failure_rate"] = subject_failure['failure_rate'].mean()
        metrics["subject_with_highest_failure"] = str(subject_failure.iloc[0][subject_col])
        metrics["subject_with_lowest_failure"] = str(subject_failure.iloc[-1][subject_col])

        insights.append(f"Overall Average Failure Rate: {subject_failure['failure_rate'].mean():.2f}%")
        insights.append(f"Subject with highest failure rate: {subject_failure.iloc[0][subject_col]} ({subject_failure.iloc[0]['failure_rate']:.1f}%)")
        insights.append(f"Subject with lowest failure rate: {subject_failure.iloc[-1][subject_col]} ({subject_failure.iloc[-1]['failure_rate']:.1f}%)")

        # Bar chart of failure rates
        fig1 = px.bar(subject_failure.head(15), x=subject_col, y='failure_rate',
                     title="Top 15 Subjects by Failure Rate",
                     color='failure_rate', color_continuous_scale='Reds',
                     text=subject_failure.head(15)['failure_rate'].round(1))
        fig1.update_traces(texttemplate='%{text}%', textposition='outside')
        visualizations["failure_rates"] = fig1.to_json()

        # Distribution of failure rates
        fig2 = px.histogram(subject_failure, x='failure_rate', nbins=20,
                           title="Distribution of Subject Failure Rates")
        
        # Add threshold lines
        fig2.add_vline(x=30, line_dash="dash", line_color="orange",
                      annotation_text="High Risk (30%)")
        fig2.add_vline(x=50, line_dash="dash", line_color="red",
                      annotation_text="Critical (50%)")
        visualizations["failure_distribution"] = fig2.to_json()

        # Identify high-risk subjects
        high_risk = subject_failure[subject_failure['failure_rate'] > 30]
        critical = subject_failure[subject_failure['failure_rate'] > 50]

        if len(high_risk) > 0:
            metrics["high_risk_subjects"] = len(high_risk)
            insights.append(f"\n⚠️ High Risk Subjects (>30% failure): {len(high_risk)}")
            for _, row in high_risk.head(5).iterrows():
                insights.append(f"  {row[subject_col]}: {row['failure_rate']:.1f}%")

        if len(critical) > 0:
            metrics["critical_subjects"] = len(critical)
            insights.append(f"❌ Critical Subjects (>50% failure): {len(critical)}")

        # Semester-wise analysis
        sem_col = matched.get('semester')
        if sem_col and sem_col in df.columns:
            sem_failure = df.groupby(sem_col)['failure_rate'].mean().reset_index()
            sem_failure = sem_failure.sort_values(sem_col)
            
            fig3 = px.line(sem_failure, x=sem_col, y='failure_rate',
                          title="Failure Rate Trend by Semester",
                          markers=True)
            visualizations["semester_trend"] = fig3.to_json()

        # Department-wise analysis
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_failure = df.groupby(dept_col)['failure_rate'].mean().reset_index()
            dept_failure = dept_failure.sort_values('failure_rate')
            
            fig4 = px.bar(dept_failure, x=dept_col, y='failure_rate',
                         title="Average Failure Rate by Department",
                         color='failure_rate', color_continuous_scale='RdYlGn_r')
            visualizations["dept_failure"] = fig4.to_json()

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


def gender_wise_performance_analysis(df):
    """Gender-wise performance analysis"""
    analysis_type = "gender_wise_performance_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Gender-wise Performance Analysis ---"]
        
        expected = ['gender', 'student_id', 'cgpa', 'gpa', 'percentage', 'marks', 'attendance', 'result']
        matched = fuzzy_match_column(df, expected)

        # Find gender column
        gender_col = None
        for key in ['gender']:
            if matched.get(key):
                gender_col = matched[key]
                break

        if not gender_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['gender', 'sex']):
                    gender_col = col
                    matched['gender'] = col
                    break

        # Find performance columns
        perf_cols = []
        for key in ['cgpa', 'gpa', 'percentage', 'marks', 'attendance']:
            if matched.get(key):
                perf_cols.append(matched[key])

        if not gender_col or not perf_cols:
            missing = []
            if not gender_col:
                missing.append("gender")
            if not perf_cols:
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert metrics to numeric
        for col in perf_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Gender distribution
        gender_counts = df[gender_col].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        metrics["gender_distribution"] = json.loads(gender_counts.to_json(orient="split"))

        insights.append("Gender Distribution:")
        for _, row in gender_counts.iterrows():
            insights.append(f"  {row['Gender']}: {row['Count']} students ({row['Count']/len(df)*100:.1f}%)")

        # Performance by gender for each metric
        for col in perf_cols:
            gender_perf = df.groupby(gender_col)[col].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
            gender_perf.columns = [gender_col, 'Mean', 'Std', 'Min', 'Max', 'Median']
            
            metrics[f"{col}_by_gender"] = json.loads(gender_perf.to_json(orient="split"))
            
            insights.append(f"\n{col} by Gender:")
            for _, row in gender_perf.iterrows():
                insights.append(f"  {row[gender_col]}: Mean = {row['Mean']:.2f}, Median = {row['Median']:.2f}")

        # Box plot comparison
        for col in perf_cols:
            fig = px.box(df, x=gender_col, y=col,
                        title=f"{col} Distribution by Gender",
                        color=gender_col)
            visualizations[f"{col}_boxplot"] = fig.to_json()

        # Performance gap analysis
        if len(gender_counts) >= 2:
            primary_metric = perf_cols[0]
            gender_groups = df.groupby(gender_col)[primary_metric].mean()
            performance_gap = gender_groups.max() - gender_groups.min()
            
            metrics["performance_gap"] = performance_gap
            insights.append(f"\nPerformance Gap in {primary_metric}: {performance_gap:.2f}")

        # Radar chart for multi-metric comparison
        if len(perf_cols) >= 3:
            radar_data = df.groupby(gender_col)[perf_cols].mean().reset_index()
            
            fig_radar = go.Figure()
            for _, row in radar_data.iterrows():
                values = [row[col] for col in perf_cols]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=perf_cols,
                    fill='toself',
                    name=row[gender_col]
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Performance Metrics by Gender"
            )
            visualizations["gender_radar"] = fig_radar.to_json()

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


def batch_performance_comparison_analysis(df):
    """Batch performance comparison analysis"""
    analysis_type = "batch_performance_comparison_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Batch Performance Comparison Analysis ---"]
        
        expected = ['batch', 'year', 'student_id', 'cgpa', 'gpa', 'percentage', 'pass_percentage', 'placement_rate']
        matched = fuzzy_match_column(df, expected)

        # Find batch column
        batch_col = None
        for key in ['batch', 'year']:
            if matched.get(key):
                batch_col = matched[key]
                break

        if not batch_col:
            for col in df.columns:
                if any(x in col.lower() for x in ['batch', 'year', 'cohort']):
                    batch_col = col
                    matched['batch'] = col
                    break

        # Find performance metrics
        perf_metrics = []
        metric_names = []
        
        metric_keys = {
            'cgpa': 'CGPA',
            'gpa': 'GPA',
            'percentage': 'Percentage',
            'pass_percentage': 'Pass %',
            'placement_rate': 'Placement %'
        }
        
        for key, display_name in metric_keys.items():
            if matched.get(key):
                perf_metrics.append(matched[key])
                metric_names.append(display_name)

        if not batch_col or not perf_metrics:
            missing = []
            if not batch_col:
                missing.append("batch/year")
            if not perf_metrics:
                missing.append("performance metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert metrics to numeric
        for col in perf_metrics:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Batch-wise summary
        batch_summary = df.groupby(batch_col).agg({
            **{col: ['mean', 'std', 'count'] for col in perf_metrics},
            'student_id': 'count' if 'student_id' in df.columns else None
        }).reset_index()
        
        # Flatten columns
        batch_summary.columns = ['_'.join(col).strip('_') for col in batch_summary.columns.values]
        batch_summary = batch_summary.sort_values(batch_col)

        metrics["batch_performance"] = json.loads(batch_summary.to_json(orient="split"))

        insights.append(f"Number of batches compared: {len(batch_summary)}")
        
        # Performance trends
        for i, metric in enumerate(perf_metrics):
            metric_display = metric_names[i]
            col_mean = f'{metric}_mean'
            
            if col_mean in batch_summary.columns:
                best_batch = batch_summary.loc[batch_summary[col_mean].idxmax(), batch_col]
                worst_batch = batch_summary.loc[batch_summary[col_mean].idxmin(), batch_col]
                
                insights.append(f"\n{metric_display}:")
                insights.append(f"  Best: {best_batch} ({batch_summary[col_mean].max():.2f})")
                insights.append(f"  Worst: {worst_batch} ({batch_summary[col_mean].min():.2f})")

        # Create comparison charts
        for i, metric in enumerate(perf_metrics):
            metric_display = metric_names[i]
            col_mean = f'{metric}_mean'
            col_std = f'{metric}_std'
            
            if col_mean in batch_summary.columns:
                fig = px.bar(batch_summary, x=batch_col, y=col_mean,
                            error_y=col_std if col_std in batch_summary.columns else None,
                            title=f"{metric_display} by Batch",
                            color=col_mean, color_continuous_scale='Viridis')
                visualizations[f"{metric}_by_batch"] = fig.to_json()

        # Multi-metric comparison (grouped bar)
        if len(perf_metrics) > 1:
            melted_data = []
            for i, metric in enumerate(perf_metrics):
                col_mean = f'{metric}_mean'
                if col_mean in batch_summary.columns:
                    temp = batch_summary[[batch_col, col_mean]].copy()
                    temp.columns = [batch_col, 'value']
                    temp['metric'] = metric_names[i]
                    melted_data.append(temp)
            
            if melted_data:
                melted_df = pd.concat(melted_data, ignore_index=True)
                
                fig_grouped = px.bar(melted_df, x=batch_col, y='value', color='metric',
                                     title="Batch Performance Comparison",
                                     barmode='group')
                visualizations["grouped_comparison"] = fig_grouped.to_json()

        # Student count by batch
        if 'student_id_count' in batch_summary.columns:
            fig_count = px.bar(batch_summary, x=batch_col, y='student_id_count',
                              title="Student Count by Batch",
                              color='student_id_count', color_continuous_scale='Viridis')
            visualizations["batch_sizes"] = fig_count.to_json()

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


def academic_progression_analysis(df):
    """Academic progression analysis"""
    analysis_type = "academic_progression_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Academic Progression Analysis ---"]
        
        expected = ['student_id', 'joining_year', 'current_year', 'semester', 'cgpa_progression', 'credits_completed', 'credits_required', 'expected_graduation']
        matched = fuzzy_match_column(df, expected)

        # Find student identifier
        student_col = matched.get('student_id')
        
        # Find year/semester columns
        join_col = matched.get('joining_year')
        current_col = matched.get('current_year')
        sem_col = matched.get('semester')
        
        # Find progression metrics
        cgpa_prog_col = matched.get('cgpa_progression')
        credits_done_col = matched.get('credits_completed')
        credits_req_col = matched.get('credits_required')

        if not student_col:
            insights.append("Could not find student identifier column.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Calculate progression metrics
        if join_col and join_col in df.columns:
            df[join_col] = pd.to_numeric(df[join_col], errors='coerce')
            current_year = pd.Timestamp.now().year
            
            if 'current_year' not in df.columns:
                df['years_enrolled'] = current_year - df[join_col]
            else:
                df[current_col] = pd.to_numeric(df[current_col], errors='coerce')
                df['years_enrolled'] = df[current_col] - df[join_col]

            metrics["avg_years_enrolled"] = df['years_enrolled'].mean()
            insights.append(f"Average Years Enrolled: {df['years_enrolled'].mean():.1f}")

        # Credit completion analysis
        if credits_done_col and credits_req_col:
            df[credits_done_col] = pd.to_numeric(df[credits_done_col], errors='coerce')
            df[credits_req_col] = pd.to_numeric(df[credits_req_col], errors='coerce')
            
            df['credit_completion_pct'] = (df[credits_done_col] / df[credits_req_col] * 100).round(2)
            df['on_track'] = (df['credit_completion_pct'] >= 75).astype(int)
            
            metrics["avg_credit_completion"] = df['credit_completion_pct'].mean()
            metrics["students_on_track"] = int(df['on_track'].sum())
            
            insights.append(f"Average Credit Completion: {df['credit_completion_pct'].mean():.1f}%")
            insights.append(f"Students on Track: {df['on_track'].sum()} ({df['on_track'].mean()*100:.1f}%)")
            
            # Credit completion distribution
            fig1 = px.histogram(df, x='credit_completion_pct', nbins=20,
                               title="Credit Completion Distribution")
            fig1.add_vline(x=75, line_dash="dash", line_color="green",
                          annotation_text="On Track (75%)")
            visualizations["credit_completion"] = fig1.to_json()

        # CGPA progression
        if cgpa_prog_col and cgpa_prog_col in df.columns:
            df[cgpa_prog_col] = pd.to_numeric(df[cgpa_prog_col], errors='coerce')
            
            # Categorize CGPA
            df['cgpa_category'] = pd.cut(df[cgpa_prog_col],
                                        bins=[0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                        labels=['<5.0', '5.0-6.0', '6.0-7.0', '7.0-8.0', '8.0-9.0', '9.0-10.0'])
            
            category_counts = df['cgpa_category'].value_counts().sort_index().reset_index()
            category_counts.columns = ['CGPA Range', 'Count']
            
            fig2 = px.bar(category_counts, x='CGPA Range', y='Count',
                         title="CGPA Distribution",
                         color='Count', color_continuous_scale='Viridis')
            visualizations["cgpa_progression"] = fig2.to_json()

        # Identify at-risk students
        at_risk = []
        if credits_done_col and credits_req_col:
            low_credit = df[df['credit_completion_pct'] < 50]
            at_risk.append(('Low credit completion', len(low_credit)))
        
        if cgpa_prog_col:
            low_cgpa = df[df[cgpa_prog_col] < 5.0]
            at_risk.append(('Low CGPA (<5.0)', len(low_cgpa)))

        if at_risk:
            insights.append("\nAt-Risk Students:")
            for risk_type, count in at_risk:
                insights.append(f"  {risk_type}: {count} students")

        # Graduation projection
        if 'expected_graduation' in df.columns:
            grad_col = matched['expected_graduation']
            grad_years = df[grad_col].value_counts().sort_index().reset_index()
            grad_years.columns = ['Year', 'Count']
            
            fig3 = px.bar(grad_years, x='Year', y='Count',
                         title="Expected Graduation Years",
                         color='Count', color_continuous_scale='Viridis')
            visualizations["graduation_projection"] = fig3.to_json()

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


def result_improvement_trend_analysis(df):
    """Result improvement trend analysis"""
    analysis_type = "result_improvement_trend_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Result Improvement Trend Analysis ---"]
        
        expected = ['academic_year', 'semester', 'pass_percentage', 'average_marks', 'previous_pass_percentage', 'improvement', 'department']
        matched = fuzzy_match_column(df, expected)

        # Find time column
        time_col = None
        for key in ['academic_year', 'semester']:
            if matched.get(key):
                time_col = matched[key]
                break

        # Find result metric
        result_col = None
        for key in ['pass_percentage', 'average_marks']:
            if matched.get(key):
                result_col = matched[key]
                break

        # Find previous/improvement columns
        prev_col = matched.get('previous_pass_percentage')
        imp_col = matched.get('improvement')

        if not time_col or not result_col:
            missing = []
            if not time_col:
                missing.append("time period")
            if not result_col:
                missing.append("result metric")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Prepare data
        df[result_col] = pd.to_numeric(df[result_col], errors='coerce')
        
        # Sort by time
        try:
            df['time_numeric'] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.sort_values('time_numeric')
        except:
            df = df.sort_values(time_col)

        # Calculate improvement if not provided
        if imp_col:
            df[imp_col] = pd.to_numeric(df[imp_col], errors='coerce')
            df['improvement'] = df[imp_col]
        elif prev_col:
            df[prev_col] = pd.to_numeric(df[prev_col], errors='coerce')
            df['improvement'] = df[result_col] - df[prev_col]
        else:
            df['improvement'] = df[result_col].diff()

        # Overall statistics
        overall_improvement = df['improvement'].mean()
        metrics["average_improvement"] = overall_improvement
        metrics["max_improvement"] = df['improvement'].max()
        metrics["min_improvement"] = df['improvement'].min()

        insights.append(f"Average Improvement per Period: {overall_improvement:+.2f}")
        insights.append(f"Largest Improvement: +{df['improvement'].max():.2f}")
        insights.append(f"Largest Decline: {df['improvement'].min():.2f}")

        # Periods with improvement vs decline
        improving_periods = (df['improvement'] > 0).sum()
        declining_periods = (df['improvement'] < 0).sum()
        stable_periods = (df['improvement'] == 0).sum()

        metrics["improving_periods"] = int(improving_periods)
        metrics["declining_periods"] = int(declining_periods)

        insights.append(f"Periods with Improvement: {improving_periods}")
        insights.append(f"Periods with Decline: {declining_periods}")

        # Line chart of results
        fig1 = px.line(df, x=time_col, y=result_col,
                      title=f"{result_col} Trend Over Time",
                      markers=True)
        visualizations["result_trend"] = fig1.to_json()

        # Bar chart of improvements
        fig2 = px.bar(df, x=time_col, y='improvement',
                     title="Period-over-Period Improvement",
                     color='improvement', color_continuous_scale='RdYlGn',
                     text=df['improvement'].round(1))
        fig2.update_traces(texttemplate='%{text}', textposition='outside')
        fig2.add_hline(y=0, line_dash="solid", line_color="black")
        visualizations["improvement_bars"] = fig2.to_json()

        # Department-wise analysis
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_imp = df.groupby(dept_col)['improvement'].mean().reset_index()
            dept_imp = dept_imp.sort_values('improvement', ascending=False)
            
            fig3 = px.bar(dept_imp, x=dept_col, y='improvement',
                         title="Average Improvement by Department",
                         color='improvement', color_continuous_scale='RdYlGn')
            visualizations["dept_improvement"] = fig3.to_json()

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


def risk_student_population_analysis(df):
    """At-risk student population analysis"""
    analysis_type = "risk_student_population_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- At-Risk Student Population Analysis ---"]
        
        expected = ['student_id', 'cgpa', 'attendance', 'arrears', 'backlogs', 'risk_score', 'risk_category']
        matched = fuzzy_match_column(df, expected)

        # Find student identifier
        student_col = matched.get('student_id')
        
        # Find risk factors
        risk_factors = []
        factor_names = []
        
        factor_keys = {
            'cgpa': 'Low CGPA',
            'attendance': 'Low Attendance',
            'arrears': 'Has Arrears',
            'backlogs': 'Has Backlogs'
        }
        
        for key, display_name in factor_keys.items():
            if matched.get(key):
                risk_factors.append(matched[key])
                factor_names.append(display_name)

        if not student_col or not risk_factors:
            missing = []
            if not student_col:
                missing.append("student identifier")
            if not risk_factors:
                missing.append("risk factors")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Define risk thresholds
        risk_thresholds = {
            'cgpa': 5.0,
            'attendance': 75,
            'arrears': 1,
            'backlogs': 1
        }

        # Calculate individual risk indicators
        for i, col in enumerate(risk_factors):
            factor_key = list(factor_keys.keys())[i]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            threshold = risk_thresholds.get(factor_key, 0)
            
            if factor_key in ['cgpa', 'attendance']:
                # Lower is riskier
                df[f'risk_{factor_key}'] = (df[col] < threshold).astype(int)
            else:
                # Higher is riskier
                df[f'risk_{factor_key}'] = (df[col] >= threshold).astype(int)

        # Calculate composite risk score
        risk_cols = [f'risk_{key}' for key in factor_keys.keys() if f'risk_{key}' in df.columns]
        df['risk_score'] = df[risk_cols].sum(axis=1)
        
        # Categorize risk level
        df['risk_level'] = pd.cut(df['risk_score'],
                                  bins=[-1, 0, 1, 2, len(risk_cols)],
                                  labels=['No Risk', 'Low Risk', 'Moderate Risk', 'High Risk'])

        # Risk statistics
        total_students = len(df)
        risk_counts = df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        metrics["total_students"] = total_students
        metrics["risk_distribution"] = json.loads(risk_counts.to_json(orient="split"))
        metrics["average_risk_score"] = df['risk_score'].mean()

        insights.append(f"Total Students Analyzed: {total_students}")
        insights.append("\nRisk Distribution:")
        for _, row in risk_counts.iterrows():
            percentage = (row['Count'] / total_students) * 100
            insights.append(f"  {row['Risk Level']}: {row['Count']} ({percentage:.1f}%)")

        # Students at risk
        at_risk = df[df['risk_level'].isin(['Moderate Risk', 'High Risk'])]
        high_risk = df[df['risk_level'] == 'High Risk']

        metrics["at_risk_students"] = len(at_risk)
        metrics["high_risk_students"] = len(high_risk)

        # Risk pie chart
        fig1 = px.pie(risk_counts, values='Count', names='Risk Level',
                     title="Student Risk Distribution",
                     color='Risk Level',
                     color_discrete_map={
                         'No Risk': 'green',
                         'Low Risk': 'yellow',
                         'Moderate Risk': 'orange',
                         'High Risk': 'red'
                     })
        visualizations["risk_pie"] = fig1.to_json()

        # Risk factor contribution
        risk_factor_counts = df[risk_cols].sum().reset_index()
        risk_factor_counts.columns = ['Risk Factor', 'Count']
        
        # Clean factor names
        risk_factor_counts['Risk Factor'] = risk_factor_counts['Risk Factor'].str.replace('risk_', '')
        
        fig2 = px.bar(risk_factor_counts, x='Risk Factor', y='Count',
                     title="Number of Students by Risk Factor",
                     color='Count', color_continuous_scale='Reds')
        visualizations["risk_factors"] = fig2.to_json()

        # Department-wise risk
        dept_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ['dept', 'department']):
                dept_col = col
                break
        
        if dept_col:
            dept_risk = df.groupby(dept_col)['risk_score'].mean().reset_index()
            dept_risk = dept_risk.sort_values('risk_score', ascending=False)
            
            fig3 = px.bar(dept_risk, x=dept_col, y='risk_score',
                         title="Average Risk Score by Department",
                         color='risk_score', color_continuous_scale='Reds')
            visualizations["dept_risk"] = fig3.to_json()

        # List high-risk students
        if len(high_risk) > 0:
            insights.append(f"\n⚠️ High Risk Students ({len(high_risk)}):")
            for idx, row in high_risk.head(10).iterrows():
                student_name = row.get(student_col, f"Student {idx}")
                insights.append(f"  {student_name}: Risk Score {row['risk_score']}")

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


def resource_utilization_academic_analysis(df):
    """Resource utilization academic analysis"""
    analysis_type = "resource_utilization_academic_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Resource Utilization Academic Analysis ---"]
        
        expected = ['resource_type', 'resource_name', 'total_capacity', 'utilized', 'utilization_rate', 'department', 'semester']
        matched = fuzzy_match_column(df, expected)

        # Find resource columns
        resource_col = None
        for key in ['resource_type', 'resource_name']:
            if matched.get(key):
                resource_col = matched[key]
                break

        # Find capacity/utilization columns
        capacity_col = matched.get('total_capacity')
        utilized_col = matched.get('utilized')
        rate_col = matched.get('utilization_rate')

        if not resource_col or not (capacity_col or utilized_col or rate_col):
            missing = []
            if not resource_col:
                missing.append("resource")
            if not (capacity_col or utilized_col or rate_col):
                missing.append("utilization data")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        if capacity_col:
            df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
        if utilized_col:
            df[utilized_col] = pd.to_numeric(df[utilized_col], errors='coerce')

        # Calculate utilization rate if needed
        if rate_col:
            df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
            df['utilization_rate'] = df[rate_col]
        elif capacity_col and utilized_col:
            df['utilization_rate'] = (df[utilized_col] / df[capacity_col] * 100).round(2)

        # Resource summary
        resource_summary = df.groupby(resource_col).agg({
            'utilization_rate': 'mean',
            capacity_col: 'sum' if capacity_col else None,
            utilized_col: 'sum' if utilized_col else None,
            resource_col: 'count'
        }).reset_index()
        
        resource_summary.columns = [resource_col, 'Avg Utilization %', 'Total Capacity', 'Total Utilized', 'Instance Count']
        resource_summary = resource_summary.sort_values('Avg Utilization %', ascending=False)

        metrics["resource_utilization"] = json.loads(resource_summary.to_json(orient="split"))
        metrics["overall_utilization"] = df['utilization_rate'].mean()

        insights.append(f"Overall Average Utilization: {df['utilization_rate'].mean():.1f}%")
        insights.append(f"Number of resource types: {len(resource_summary)}")

        # Identify underutilized and overutilized resources
        underutilized = resource_summary[resource_summary['Avg Utilization %'] < 50]
        overutilized = resource_summary[resource_summary['Avg Utilization %'] > 90]

        if len(underutilized) > 0:
            metrics["underutilized_resources"] = len(underutilized)
            insights.append(f"\n⚠️ Underutilized Resources (<50%): {len(underutilized)}")
            for _, row in underutilized.head(5).iterrows():
                insights.append(f"  {row[resource_col]}: {row['Avg Utilization %']:.1f}%")

        if len(overutilized) > 0:
            metrics["overutilized_resources"] = len(overutilized)
            insights.append(f"\n📈 Overutilized Resources (>90%): {len(overutilized)}")

        # Bar chart of utilization
        fig1 = px.bar(resource_summary.head(15), x=resource_col, y='Avg Utilization %',
                     title="Resource Utilization Rates (Top 15)",
                     color='Avg Utilization %', color_continuous_scale='RdYlGn',
                     text=resource_summary.head(15)['Avg Utilization %'].round(1))
        fig1.update_traces(texttemplate='%{text}%', textposition='outside')
        fig1.add_hline(y=50, line_dash="dash", line_color="orange",
                      annotation_text="Underutilized Threshold")
        fig1.add_hline(y=90, line_dash="dash", line_color="red",
                      annotation_text="Overutilized Threshold")
        visualizations["utilization_chart"] = fig1.to_json()

        # Utilization distribution
        fig2 = px.histogram(df, x='utilization_rate', nbins=20,
                           title="Distribution of Utilization Rates")
        visualizations["utilization_distribution"] = fig2.to_json()

        # Department-wise analysis
        dept_col = matched.get('department')
        if dept_col and dept_col in df.columns:
            dept_util = df.groupby(dept_col)['utilization_rate'].mean().reset_index()
            dept_util = dept_util.sort_values('utilization_rate', ascending=False)
            
            fig3 = px.bar(dept_util, x=dept_col, y='utilization_rate',
                         title="Average Utilization by Department",
                         color='utilization_rate', color_continuous_scale='Viridis')
            visualizations["dept_utilization"] = fig3.to_json()

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


def program_outcome_achievement_analysis(df):
    """Program outcome achievement analysis"""
    analysis_type = "program_outcome_achievement_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Program Outcome Achievement Analysis ---"]
        
        expected = ['po_code', 'po_description', 'target_percentage', 'achieved_percentage', 'attainment_level', 'assessment_method', 'academic_year']
        matched = fuzzy_match_column(df, expected)

        # Find PO columns
        po_col = None
        for key in ['po_code', 'po_description']:
            if matched.get(key):
                po_col = matched[key]
                break

        # Find attainment columns
        target_col = matched.get('target_percentage')
        achieved_col = matched.get('achieved_percentage')
        level_col = matched.get('attainment_level')

        if not po_col or not (achieved_col or level_col):
            missing = []
            if not po_col:
                missing.append("PO identifier")
            if not (achieved_col or level_col):
                missing.append("attainment data")
            insights += get_missing_columns_message(missing, matched)
            
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Convert to numeric
        if achieved_col:
            df[achieved_col] = pd.to_numeric(df[achieved_col], errors='coerce')
            df['achieved'] = df[achieved_col]
        if target_col:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df['target'] = df[target_col]
        if level_col:
            df[level_col] = pd.to_numeric(df[level_col], errors='coerce')
            df['attainment_level'] = df[level_col]

        # Calculate attainment metrics
        df['attainment_gap'] = df['achieved'] - df['target'] if 'target' in df.columns else 0
        df['achieved_target_ratio'] = (df['achieved'] / df['target'] * 100) if 'target' in df.columns else 100

        # PO summary
        po_summary = df.groupby(po_col).agg({
            'achieved': 'mean',
            'target': 'mean' if 'target' in df.columns else None,
            'attainment_level': 'mean' if 'attainment_level' in df.columns else None,
            'achieved_target_ratio': 'mean',
            po_col: 'count'
        }).reset_index()
        
        po_summary.columns = [po_col, 'Avg Achieved %', 'Avg Target %', 'Avg Attainment Level', 'Achievement Ratio %', 'Assessment Count']
        po_summary = po_summary.sort_values('Avg Achieved %', ascending=False)

        metrics["po_attainment"] = json.loads(po_summary.to_json(orient="split"))
        metrics["overall_achievement"] = df['achieved'].mean()

        insights.append(f"Number of Program Outcomes analyzed: {len(po_summary)}")
        insights.append(f"Overall Average Achievement: {df['achieved'].mean():.1f}%")

        # Best and worst performing POs
        best_po = po_summary.iloc[0][po_col]
        worst_po = po_summary.iloc[-1][po_col]
        insights.append(f"Best Performing PO: {best_po} ({po_summary.iloc[0]['Avg Achieved %']:.1f}%)")
        insights.append(f"PO Needing Attention: {worst_po} ({po_summary.iloc[-1]['Avg Achieved %']:.1f}%)")

        # Bar chart of PO achievement
        fig1 = px.bar(po_summary, x=po_col, y='Avg Achieved %',
                     title="Program Outcome Achievement Levels",
                     color='Avg Achieved %', color_continuous_scale='RdYlGn',
                     text=po_summary['Avg Achieved %'].round(1))
        fig1.update_traces(texttemplate='%{text}%', textposition='outside')
        
        if 'Avg Target %' in po_summary.columns:
            fig1.add_scatter(x=po_summary[po_col], y=po_summary['Avg Target %'],
                            mode='markers', name='Target',
                            marker=dict(color='red', size=10, symbol='diamond'))
        visualizations["po_achievement"] = fig1.to_json()

        # Attainment level distribution
        if 'Avg Attainment Level' in po_summary.columns:
            level_counts = df['attainment_level'].value_counts().sort_index().reset_index()
            level_counts.columns = ['Attainment Level', 'Count']
            
            fig2 = px.bar(level_counts, x='Attainment Level', y='Count',
                         title="Distribution of Attainment Levels",
                         color='Count', color_continuous_scale='Viridis')
            visualizations["attainment_levels"] = fig2.to_json()

        # Gap analysis
        if 'attainment_gap' in df.columns:
            gap_summary = df.groupby(po_col)['attainment_gap'].mean().reset_index()
            gap_summary = gap_summary.sort_values('attainment_gap')
            
            fig3 = px.bar(gap_summary, x=po_col, y='attainment_gap',
                         title="Attainment Gap (Achieved - Target)",
                         color='attainment_gap', color_continuous_scale='RdBu')
            fig3.add_hline(y=0, line_dash="solid", line_color="black")
            visualizations["attainment_gap"] = fig3.to_json()

        # Year-wise trend
        year_col = matched.get('academic_year')
        if year_col and year_col in df.columns:
            year_achievement = df.groupby(year_col)['achieved'].mean().reset_index()
            year_achievement = year_achievement.sort_values(year_col)
            
            fig4 = px.line(year_achievement, x=year_col, y='achieved',
                          title="PO Achievement Trend Over Years",
                          markers=True)
            visualizations["po_trend"] = fig4.to_json()

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


def institutional_academic_health_index_analysis(df):
    """Institutional academic health index analysis"""
    analysis_type = "institutional_academic_health_index_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Institutional Academic Health Index Analysis ---"]
        
        expected = ['academic_year', 'pass_percentage', 'average_cgpa', 'placement_rate', 'retention_rate', 'graduation_rate', 'faculty_student_ratio']
        matched = fuzzy_match_column(df, expected)

        # Find metrics
        metrics_found = {}
        metric_weights = {
            'pass_percentage': 0.20,
            'average_cgpa': 0.15,
            'placement_rate': 0.25,
            'retention_rate': 0.15,
            'graduation_rate': 0.15,
            'faculty_student_ratio': 0.10
        }

        for key in metric_weights.keys():
            if matched.get(key):
                col = matched[key]
                df[col] = pd.to_numeric(df[col], errors='coerce')
                metrics_found[key] = col

        if not metrics_found:
            insights.append("Could not find academic health metrics.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # Normalize each metric to 0-100 scale
        for key, col in metrics_found.items():
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{key}_normalized'] = (df[col] - min_val) / (max_val - min_val) * 100
            else:
                df[f'{key}_normalized'] = 50

        # Calculate composite health index
        norm_cols = [f'{key}_normalized' for key in metrics_found.keys()]
        weights = [metric_weights[key] for key in metrics_found.keys()]
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        df['health_index'] = sum(df[col] * w for col, w in zip(norm_cols, weights))

        # Year-wise analysis
        year_col = matched.get('academic_year')
        if year_col and year_col in df.columns:
            year_health = df.groupby(year_col)['health_index'].mean().reset_index()
            year_health = year_health.sort_values(year_col)
            
            metrics["yearly_health"] = json.loads(year_health.to_json(orient="split"))
            
            current_year = year_health.iloc[-1]
            first_year = year_health.iloc[0]
            improvement = current_year['health_index'] - first_year['health_index']
            
            insights.append(f"Current Academic Health Index: {current_year['health_index']:.1f}")
            insights.append(f"Health Index Trend: {improvement:+.1f} since {first_year[year_col]}")

            # Health index trend
            fig1 = px.line(year_health, x=year_col, y='health_index',
                          title="Institutional Academic Health Index Trend",
                          markers=True)
            
            # Add threshold lines
            fig1.add_hline(y=75, line_dash="dash", line_color="green",
                          annotation_text="Excellent (75)")
            fig1.add_hline(y=50, line_dash="dash", line_color="orange",
                          annotation_text="Good (50)")
            fig1.add_hline(y=25, line_dash="dash", line_color="red",
                          annotation_text="Needs Attention (25)")
            
            visualizations["health_trend"] = fig1.to_json()
        else:
            # Single year or aggregated data
            avg_health = df['health_index'].mean()
            metrics["current_health_index"] = avg_health
            
            insights.append(f"Institutional Academic Health Index: {avg_health:.1f}")

        # Component breakdown
        component_scores = []
        for key, col in metrics_found.items():
            component_scores.append({
                'Metric': key.replace('_', ' ').title(),
                'Score': df[col].mean()
            })
        
        component_df = pd.DataFrame(component_scores)
        
        fig2 = px.bar(component_df, x='Metric', y='Score',
                     title="Academic Health Component Scores",
                     color='Score', color_continuous_scale='RdYlGn',
                     text=component_df['Score'].round(1))
        fig2.update_traces(texttemplate='%{text}', textposition='outside')
        visualizations["component_scores"] = fig2.to_json()

        # Radar chart for current year
        if year_col and year_col in df.columns:
            latest_year = df[year_col].max()
            latest_data = df[df[year_col] == latest_year].iloc[0]
            
            fig3 = go.Figure(data=go.Scatterpolar(
                r=[latest_data[col] for col in metrics_found.values()],
                theta=[key.replace('_', ' ').title() for key in metrics_found.keys()],
                fill='toself'
            ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=f"Academic Health Radar ({latest_year})"
            )
            visualizations["health_radar"] = fig3.to_json()

        # Health index distribution
        fig4 = px.histogram(df, x='health_index', nbins=20,
                           title="Distribution of Health Index")
        visualizations["health_distribution"] = fig4.to_json()

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


# ========== MAIN FUNCTION FOR ADMIN ANALYTICS ==========

def main_admin():
    """Main function to run Admin Analytics"""
    print("=" * 60)
    print("👔 ADMIN ANALYTICS SYSTEM")
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

    # Admin analytics function mapping
    admin_function_mapping = {
        "department_wise_performance_analysis": department_wise_performance_analysis,
        "year_wise_result_trend_analysis": year_wise_result_trend_analysis,
        "overall_pass_percentage_analysis": overall_pass_percentage_analysis,
        "cgpa_distribution_analysis": cgpa_distribution_analysis,
        "department_comparison_analysis": department_comparison_analysis,
        "semester_result_summary_analysis": semester_result_summary_analysis,
        "dropout_rate_analysis": dropout_rate_analysis,
        "arrear_rate_analysis": arrear_rate_analysis,
        "placement_eligibility_analysis": placement_eligibility_analysis,
        "attendance_compliance_analysis": attendance_compliance_analysis,
        "faculty_performance_comparison_analysis": faculty_performance_comparison_analysis,
        "subject_failure_rate_analysis": subject_failure_rate_analysis,
        "gender_wise_performance_analysis": gender_wise_performance_analysis,
        "batch_performance_comparison_analysis": batch_performance_comparison_analysis,
        "academic_progression_analysis": academic_progression_analysis,
        "result_improvement_trend_analysis": result_improvement_trend_analysis,
        "risk_student_population_analysis": risk_student_population_analysis,
        "resource_utilization_academic_analysis": resource_utilization_academic_analysis,
        "program_outcome_achievement_analysis": program_outcome_achievement_analysis,
        "institutional_academic_health_index_analysis": institutional_academic_health_index_analysis,
    }

    # Analysis Selection
    print("\n" + "=" * 60)
    print("📋 Select an Admin Analytics Analysis to Perform:")
    print("=" * 60)
    
    analysis_names = list(admin_function_mapping.keys())
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
            selected_function = admin_function_mapping.get(selected_analysis_key)
            
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
            result = general_insights_analysis(df, "Admin Data Overview")
            
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
    result = main_admin()
    print("\n" + "=" * 60)
    print("✅ Admin Analytics Complete")
    print("=" * 60)