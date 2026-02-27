import os
import sys
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
import joblib
from rapidfuzz import fuzz
from groq import Groq
from dotenv import load_dotenv
import warnings
from datetime import datetime
from io import BytesIO, StringIO
import base64
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import traceback
import aiofiles
from fastapi.encoders import jsonable_encoder
import importlib
from sklearn.exceptions import InconsistentVersionWarning
import warnings
from pydantic import BaseModel
from functools import lru_cache
import hashlib

# Filter out specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="errors='ignore' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)

# Set pandas options for better performance
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
# Monkey patch for NumPy 2.0 compatibility
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'float64'):
    np.float64 = float
# Cache for dataset analysis results
analysis_cache = {}

def get_cache_key(df, analysis_name, dataset_type):
    """Generate a cache key for analysis results"""
    df_hash = hashlib.md5(
        f"{df.shape}_{list(df.columns)}_{df.head(3).to_string()}".encode()
    ).hexdigest()
    return f"{dataset_type}_{analysis_name}_{df_hash}"

@lru_cache(maxsize=128)
def cached_classify_dataset(df_hash, columns_str):
    """Cached dataset classification"""
    return "unknown"

# Fix for numpy compatibility
try:
    np.bool = np.bool_
except AttributeError:
    np.bool = bool

try:
    np.int = np.int_
except AttributeError:
    np.int = int

try:
    np.float = np.float_
except AttributeError:
    np.float = float

# Initialize FastAPI app
app = FastAPI(title="AutoDash RIT Backend", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3004", "http://localhost:5000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None
    print("Warning: GROQ_API_KEY not found in environment variables")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    print(f"Warning: Could not mount static files or templates: {str(e)}")
    templates = None

# Load student performance ML model
student_performance_model = None

def load_student_performance_model():
    """Load the trained ML model for student performance rating"""
    try:
        # Adjust the path to where your model is stored
        model_path = "models/student_performance_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                student_performance_model = pickle.load(f)
            print("Student performance model loaded successfully")
            return student_performance_model
        else:
            print(f"Model not found at {model_path}")
            return None
    except Exception as e:
        print(f"Failed to load student performance model: {str(e)}")
        return None

student_performance_model = load_student_performance_model()

# Role-based analysis function counts
ROLE_ANALYSIS_COUNTS = {
    "student": 20,
    "faculty": 20,
    "admin": 20
}

# ========== HELPER FUNCTIONS ==========

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def is_year_or_time_column(series):
    """Check if a column represents years, time values, or phone numbers that shouldn't be averaged"""
    if is_numeric_dtype(series):
        if series.dropna().between(1900, 2100).all():
            return True
        if series.dropna().between(0, 2400).all():
            return True
        if series.dropna().between(1000000000, 999999999999999).all():
            return True
    elif series.dtype == 'object':
        if series.str.match(r'^\+?[\d\s-]{10,15}$').all():
            return True
    return False

def detect_categorical_columns(df, threshold=0.05):
    """Enhanced categorical column detection"""
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'category' or df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                string_patterns = any(
                    isinstance(val, str) and 
                    (len(val) < 50 or val in ['Yes', 'No', 'True', 'False', 'Male', 'Female'])
                    for val in sample_values
                )
                
                if unique_ratio < threshold or string_patterns:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
    
    return categorical_cols

def detect_datetime_columns(df):
    """Enhanced datetime column detection"""
    datetime_cols = []
    
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
            
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    datetime_cols.append(col)
            except:
                try:
                    sample_str = str(sample.iloc[0]) if len(sample) > 0 else ""
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',
                        r'\d{2}/\d{2}/\d{4}',
                        r'\d{2}-\d{2}-\d{4}',
                        r'\d{4}/\d{2}/\d{2}',
                        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                    ]
                    
                    if any(re.search(pattern, sample_str) for pattern in date_patterns):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            if not df[col].isna().all():
                                datetime_cols.append(col)
                        except:
                            pass
                except:
                    pass
    return datetime_cols

def show_key_metrics(df):
    """Display key metrics about the dataset"""
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    return {
        "total_records": total_records,
        "total_features": len(df.columns),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols)
    }

def get_available_analyses_for_role(role: str) -> List[str]:
    """Get all available analysis functions for a specific role"""
    try:
        if role in ["student", "faculty", "admin"]:
            try:
                module = importlib.import_module(f"analyses.{role}")
            except ImportError:
                # Fallback to general analyses
                from analyses import general as module
                print(f"Using general analyses for role {role}")
        else:
            from analyses import general as module
        
        available_functions = []
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            if callable(attr) and hasattr(attr, '__name__'):
                if (attr.__module__ and 
                    (attr.__module__.startswith(f'analyses.{role}') or 
                     attr.__module__ == 'analyses.general')):
                    available_functions.append(attr_name)
        
        print(f"Found {len(available_functions)} available analyses for role {role}")
        return available_functions
        
    except Exception as e:
        print(f"Error getting available analyses for role {role}: {str(e)}")
        return ["general_insights_analysis", "data_quality_report"]

# ========== GROQ ANALYZER CLASS ==========

class GroqAnalyzer:
    def __init__(self):
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.client = None
    
    def analyze_dataset_and_suggest_functions(self, df: pd.DataFrame, available_analyses: List[str], role: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "analysis_recommendations": [],
                "confidence_score": 0
            }
        
        sample_data = df.head(5).to_string()
        column_names = df.columns.tolist()
        
        available_analyses_info = []
        for analysis in available_analyses:
            clean_name = analysis.replace('_', ' ').title()
            available_analyses_info.append(f"- {analysis} ({clean_name})")
        
        available_analyses_str = "\n".join(available_analyses_info)
        
        role_context = {
            "student": "Focus on student performance, learning patterns, attendance impact, and academic progress",
            "faculty": "Focus on teaching effectiveness, class performance, assessment analysis, and student engagement",
            "admin": "Focus on institutional metrics, resource allocation, overall performance trends, and strategic planning"
        }
        
        prompt = f"""
You are a senior data analysis expert for an educational institution. Analyze the following dataset sample and recommend 5-6 of the most relevant analyses to perform for a {role} user.

CRITICAL RULES:
1. You MUST choose analysis names ONLY from the provided list of available functions
2. Each recommended analysis MUST exactly match one of the function names in the available list
3. Recommend analyses that are most appropriate for the {role} role and dataset structure
4. Consider the data types and patterns in the sample data
5. You MUST suggest analysis functions that should match with the columns available in the dataset

ROLE-SPECIFIC CONTEXT:
{role_context[role]}

AVAILABLE ANALYSIS FUNCTIONS (use exact names):
{available_analyses_str}

DATASET INFORMATION:
- Columns: {column_names}
- Sample Data (first 5 rows):
{sample_data}

For each recommended analysis:
- Use the EXACT function name from the available list
- Provide a brief description explaining why it's relevant for a {role}
- Suggest specific columns from the dataset that would be most useful

RESPONSE FORMAT (JSON):
{{
    "analysis_recommendations": [
        {{
            "name": "exact_function_name_from_list",
            "description": "Brief explanation of why this analysis is relevant for a {role}",
            "columns": ["column1", "column2", "column3"]
        }}
    ],
    "confidence_score": 0.85
}}

Focus on analyses that:
- Match the {role} role's needs
- The columns suggested MUST exist in the dataset
- Use available columns effectively
- Provide meaningful insights for this specific data in an educational context
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            response = json.loads(chat_completion.choices[0].message.content)
            
            valid_recommendations = []
            for rec in response.get('analysis_recommendations', []):
                analysis_name = rec.get('name', '')
                
                if analysis_name in available_analyses:
                    suggested_columns = rec.get('columns', [])
                    valid_columns = [col for col in suggested_columns if col in column_names]
                    
                    if not valid_columns and column_names:
                        valid_columns = column_names[:3]
                    
                    valid_recommendations.append({
                        'name': analysis_name,
                        'description': rec.get('description', 'No description provided'),
                        'columns': valid_columns
                    })
                else:
                    print(f"Warning: Analysis '{analysis_name}' not found in available analyses")
            
            if not valid_recommendations and available_analyses:
                fallback_analysis = available_analyses[0]
                valid_recommendations.append({
                    'name': fallback_analysis,
                    'description': f"General analysis of the dataset using {fallback_analysis}",
                    'columns': column_names[:3] if column_names else []
                })
                response['confidence_score'] = 0.5
            
            response['analysis_recommendations'] = valid_recommendations
            return response
            
        except Exception as e:
            print(f"Error analyzing dataset with Groq: {str(e)}")
            fallback_recommendations = []
            if available_analyses:
                for analysis in available_analyses[:3]:
                    fallback_recommendations.append({
                        'name': analysis,
                        'description': f"Analysis using {analysis} function",
                        'columns': df.columns.tolist()[:3] if not df.empty else []
                    })
            
            return {
                "analysis_recommendations": fallback_recommendations,
                "confidence_score": 0.3
            }

# Initialize Groq Analyzer
groq_analyzer = GroqAnalyzer()

# ========== AI INSIGHTS FUNCTIONS ==========

def get_ai_insights(df, role):
    """Get AI insights about the dataset based on user role"""
    if not client:
        return "Error: Groq client not initialized. Please check your API key."
        
    data_summary = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    User Role: {role}
    
    Numeric Columns Summary:
    {df.describe().to_string()}
    
    Categorical Columns Summary:
    {df.select_dtypes(include=['object', 'category']).describe().to_string()}
    
    Sample Data:
    {df.head(3).to_string()}
    """
    
    role_context = {
        "student": "as a student looking to understand my academic performance",
        "faculty": "as a faculty member analyzing student performance and teaching effectiveness",
        "admin": "as an administrator evaluating institutional metrics and strategic planning"
    }
    
    prompt = f"""
You are a senior data analyst in an educational institution. Analyze this dataset and provide detailed insights in a professional report format for a user {role_context[role]}.

IMPORTANT FORMATTING RULES:
1. DO NOT use markdown symbols like #, *, **, __, or any other markdown formatting
2. Use clear section headings with descriptive titles
3. Use bullet points with • symbol for lists
4. Write in professional business language
5. Use proper paragraph breaks and spacing
6. Structure your response with clear sections

Please structure your analysis as follows:

DATA OVERVIEW
- Summarize the key characteristics of the dataset
- Highlight important metrics relevant to a {role}

KEY FINDINGS
- Identify 3-5 most interesting patterns or trends
- Point out any unexpected findings that warrant investigation

ACTIONABLE INSIGHTS
- Provide specific recommendations based on the {role} perspective
- Suggest areas for further analysis or investigation

Data Summary:
{data_summary}

Provide the response in clean, professional language without any markdown symbols.
"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def chat_with_data(df, question, role=None, analysis_context=None):
    """Generate a professional response to user questions"""
    if not client:
        return "Error: Groq client not initialized. Please check your API key."

    if df is None or df.empty or df.shape[1] == 0:
        return "The dataset has no valid columns after cleaning. Please upload a proper file."
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    data_summary = f"""
    Dataset Information:
    • Shape: {df.shape[0]} rows × {df.shape[1]} columns
    • User Role: {role or 'General'}
    • Columns: {list(df.columns)}
    • Numeric Columns: {list(numeric_cols)}
    • Categorical Columns: {list(categorical_cols)}
    • Missing Values: {df.isnull().sum().sum()} total
    
    Sample Data (first 3 rows):
    {df.head(3).to_string()}
    """

    role_context = f"as a {role}" if role else "generally"

    prompt = f"""
You are an expert educational data analyst assistant. Provide clear, professional responses to user questions about their dataset {role_context}.

IMPORTANT RESPONSE GUIDELINES:
1. Structure your response with clear sections and bullet points
2. Use • for bullet points instead of markdown
3. Provide specific, actionable insights relevant to the user's role
4. Reference actual column names and data from the dataset
5. Use professional business language
6. Break down complex concepts into understandable points

Dataset Summary:
{data_summary}

User Question: {question}

Provide a comprehensive, well-structured response that directly addresses the user's question while being professional and easy to understand.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try rephrasing your question."

# ========== VISUALIZATION FUNCTIONS ==========

def analyze_relationships(df, theme):
    """Analyze feature relationships with fixed layout"""
    results = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hoverinfo="text"
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text'],
                height=600,
                width=800,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            results['correlation_matrix'] = fig.to_json()
        except Exception as e:
            results['correlation_error'] = f"Error creating correlation matrix: {str(e)}"
    
    if len(numeric_cols) > 1:
        try:
            selected_cols = numeric_cols[:4]
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            color_col = categorical_cols[0] if categorical_cols else None
            
            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                color=color_col,
                title="Scatter Plot Matrix",
                height=450,
                width=800
            )
            
            fig.update_layout(
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text'],
                margin=dict(l=50, r=50, t=80, b=50),
                autosize=False
            )
            
            fig.update_traces(
                diagonal_visible=True, 
                showupperhalf=True, 
                showlowerhalf=True,
                marker=dict(size=4, opacity=0.6)
            )
            
            results['scatter_matrix'] = fig.to_json()
        except Exception as e:
            results['scatter_error'] = f"Error creating scatter matrix: {str(e)}"
    
    return results

def analyze_time_series(df, theme, date_col=None, num_col=None, period=30):
    """Enhanced time series analysis with better detection and error handling"""
    results = {}
    
    def find_datetime_columns(df):
        datetime_cols = []
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                continue
            
            if df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().head(5)
                    if len(sample) > 0:
                        test_series = pd.to_datetime(sample, errors='coerce')
                        if not test_series.isna().all():
                            datetime_cols.append(col)
                except:
                    sample_str = str(sample.iloc[0]) if len(sample) > 0 else ""
                    import re
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',
                        r'\d{2}/\d{2}/\d{4}',
                        r'\d{2}-\d{2}-\d{4}',
                        r'\d{4}/\d{2}/\d{2}',
                        r'\d{1,2}/\d{1,2}/\d{4}',
                    ]
                    
                    if any(re.search(pattern, sample_str) for pattern in date_patterns):
                        try:
                            converted = pd.to_datetime(df[col], errors='coerce')
                            if not converted.isna().all():
                                datetime_cols.append(col)
                        except:
                            pass
        return datetime_cols
    
    if not date_col or not num_col:
        date_cols = find_datetime_columns(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return {
            "date_cols": date_cols,
            "numeric_cols": numeric_cols,
            "message": "Please select date and numeric columns",
            "suggestions": {
                "best_date_col": date_cols[0] if date_cols else None,
                "best_numeric_col": numeric_cols[0] if numeric_cols else None
            }
        }
    
    try:
        df_clean = df.copy()
        
        if not is_datetime64_any_dtype(df_clean[date_col]):
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        
        df_clean = df_clean.dropna(subset=[date_col, num_col])
        
        if len(df_clean) == 0:
            return {"error": "No valid data after cleaning"}
        
        df_clean = df_clean.sort_values(date_col)
        time_df = df_clean.set_index(date_col)
        
        time_stats = {
            "start_date": str(time_df.index.min()),
            "end_date": str(time_df.index.max()),
            "total_periods": int(len(time_df)),
            "mean_value": float(time_df[num_col].mean()),
            "std_value": float(time_df[num_col].std()),
            "min_value": float(time_df[num_col].min()),
            "max_value": float(time_df[num_col].max())
        }
        
        fig_line = px.line(
            time_df,
            y=num_col,
            title=f'{num_col} Over Time',
            markers=True,
            line_shape='linear'
        )
        
        fig_line.update_layout(
            plot_bgcolor=theme.get('chart_bg', '#1a183c'),
            paper_bgcolor=theme.get('bg', '#0A071E'),
            font_color=theme.get('text', '#ffffff'),
            xaxis_title=date_col,
            yaxis_title=num_col,
            hovermode="x unified",
            height=500,
            showlegend=True
        )
        
        try:
            from scipy import stats
            x_numeric = np.arange(len(time_df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, time_df[num_col])
            trend_line = slope * x_numeric + intercept
            
            fig_line.add_trace(go.Scatter(
                x=time_df.index,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        except:
            pass
        
        results['line_chart'] = fig_line.to_json()
        results['time_stats'] = time_stats
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            clean_series = time_df[num_col].dropna()
            
            if len(clean_series) < period * 2:
                if len(clean_series) > 12:
                    period = min(12, len(clean_series) // 2)
                else:
                    results['decomposition_warning'] = "Not enough data for time series decomposition"
                    return results
            
            decomposition = seasonal_decompose(clean_series, period=period, model='additive')
            
            fig_dec = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                vertical_spacing=0.05
            )
            
            fig_dec.add_trace(
                go.Scatter(x=clean_series.index, y=decomposition.observed, name="Observed", line=dict(color='blue')),
                row=1, col=1
            )
            
            fig_dec.add_trace(
                go.Scatter(x=clean_series.index, y=decomposition.trend, name="Trend", line=dict(color='green')),
                row=2, col=1
            )
            
            fig_dec.add_trace(
                go.Scatter(x=clean_series.index, y=decomposition.seasonal, name="Seasonal", line=dict(color='orange')),
                row=3, col=1
            )
            
            fig_dec.add_trace(
                go.Scatter(x=clean_series.index, y=decomposition.resid, name="Residual", line=dict(color='red')),
                row=4, col=1
            )
            
            fig_dec.update_layout(
                height=800,
                plot_bgcolor=theme.get('chart_bg', '#1a183c'),
                paper_bgcolor=theme.get('bg', '#0A071E'),
                font_color=theme.get('text', '#ffffff'),
                showlegend=False,
                title=f"Time Series Decomposition (Period: {period})"
            )
            
            results['decomposition'] = fig_dec.to_json()
            
            decomposition_stats = {
                "trend_strength": float(abs(decomposition.trend.dropna().std() / decomposition.observed.dropna().std())),
                "seasonal_strength": float(abs(decomposition.seasonal.dropna().std() / decomposition.observed.dropna().std())),
                "residual_strength": float(abs(decomposition.resid.dropna().std() / decomposition.observed.dropna().std()))
            }
            results['decomposition_stats'] = decomposition_stats
            
        except Exception as e:
            results['decomposition_error'] = f"Time series decomposition failed: {str(e)}"
    
    except Exception as e:
        results['timeseries_error'] = f"Time series analysis failed: {str(e)}"
    
    return convert_numpy_types(results)

def generate_visualizations(df, theme):
    """Generate dashboard visualizations for the report"""
    visualizations = {}
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        for i, col in enumerate(numeric_cols[:2]):
            try:
                fig_hist = px.histogram(df, x=col, nbins=50, title=f'Distribution of {col}')
                fig_hist.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'histogram_{col}'] = fig_hist.to_json()
                
                fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
                fig_box.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'boxplot_{col}'] = fig_box.to_json()
            except Exception as e:
                print(f"Error creating visualization for {col}: {str(e)}")
    
    relationships = analyze_relationships(df, theme)
    visualizations.update(relationships)
    
    timeseries = analyze_time_series(df, theme)
    visualizations.update(timeseries)
    
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                               title="Correlation Between Numeric Features")
            fig_corr.update_layout(
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text']
            )
            visualizations['correlation_matrix'] = fig_corr.to_json()
        except Exception as e:
            print(f"Error creating correlation matrix: {str(e)}")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        for i, col in enumerate(categorical_cols[:2]):
            try:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                fig_bar = px.bar(value_counts.head(10), x='Value', y='Count', 
                               title=f"Distribution of {col}")
                fig_bar.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'barchart_{col}'] = fig_bar.to_json()
            except Exception as e:
                print(f"Error creating bar chart for {col}: {str(e)}")
    
    return visualizations

# ========== REPORT GENERATION FUNCTIONS ==========

def generate_pdf_report(df, role, insights, analysis_results=None, theme=None):
    """Generate a PDF report with insights, analysis, and dashboard visualizations"""
    if theme is None:
        theme = {
            "bg": "#FFFFFF",
            "text": "#31333F",
            "primary": "#4A90E2",
            "secondary": "#F0F2F6",
            "chart_bg": "#FFFFFF",
            "grid": "#E5E5E5"
        }
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#4A90E2')
    )
    story.append(Paragraph(f"AutoDash RIT Analysis Report - {role.capitalize()}", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    key_metrics = show_key_metrics(df)
    dataset_info = [
        ["User Role", role.capitalize()],
        ["Number of Rows", str(df.shape[0])],
        ["Number of Columns", str(df.shape[1])],
        ["Numeric Features", str(key_metrics['numeric_features'])],
        ["Categorical Features", str(key_metrics['categorical_features'])],
        ["Missing Values", str(df.isnull().sum().sum())]
    ]
    table = Table(dataset_info, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F2F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("AI Insights & Recommendations", styles['Heading2']))
    insights_paragraphs = insights.split('\n\n')
    for para in insights_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 12))
    
    story.append(Paragraph("Data Preview (First 5 rows)", styles['Heading2']))
    preview_data = [df.columns.tolist()] + df.head().values.tolist()
    preview_table = Table(preview_data, repeatRows=1)
    preview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(preview_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_excel_report(df, role, insights):
    """Generate an Excel report with multiple sheets"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_data = {
            'Metric': ['User Role', 'Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
            'Value': [
                role.capitalize(),
                len(df),
                len(df.columns),
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(include=['object', 'category']).columns),
                df.isnull().sum().sum()
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        df.head(100).to_excel(writer, sheet_name='Data Preview', index=False)
        
        if not df.select_dtypes(include=[np.number]).empty:
            df.describe().to_excel(writer, sheet_name='Statistics')
        
        insights_data = {'Insights': insights.split('\n') if insights else ['No insights available']}
        pd.DataFrame(insights_data).to_excel(writer, sheet_name='AI Insights', index=False)
    
    buffer.seek(0)
    return buffer

def generate_html_report(df, role, insights, theme=None):
    """Generate an HTML report with styling"""
    if theme is None:
        theme = {
            "bg": "#0A071E",
            "text": "#ffffff",
            "primary": "#8a2be2",
            "secondary": "#1a183c"
        }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoDash RIT Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: {theme['bg']};
                color: {theme['text']};
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: {theme['secondary']};
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: {theme['primary']};
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{
                color: {theme['primary']};
                border-bottom: 2px solid {theme['primary']};
                padding-bottom: 10px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background-color: {theme['bg']};
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid {theme['primary']};
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: {theme['primary']};
            }}
            .metric-label {{
                margin-top: 10px;
                opacity: 0.8;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: {theme['secondary']};
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid {theme['primary']};
            }}
            th {{
                background-color: {theme['primary']};
                color: white;
            }}
            .insights {{
                background-color: {theme['bg']};
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid {theme['primary']};
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AutoDash RIT Analysis Report - {role.capitalize()}</h1>
            <p style="text-align: center; opacity: 0.8;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Dataset Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(numeric_cols)}</div>
                    <div class="metric-label">Numeric Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(categorical_cols)}</div>
                    <div class="metric-label">Categorical Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{df.isnull().sum().sum()}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{role.capitalize()}</div>
                    <div class="metric-label">User Role</div>
                </div>
            </div>
            
            <h2>Data Preview</h2>
            {df.head(10).to_html(classes='data-table', table_id='preview-table', escape=False)}
            
            <h2>AI Insights</h2>
            <div class="insights">
                {insights.replace(chr(10), '<br>') if insights else 'No insights available'}
            </div>
            
            <h2>Column Information</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Non-Null Count</th>
                        <th>Null Count</th>
                        <th>Unique Values</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for col in df.columns:
        html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{df[col].dtype}</td>
                        <td>{df[col].count()}</td>
                        <td>{df[col].isnull().sum()}</td>
                        <td>{df[col].nunique()}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ========== STUDENT PERFORMANCE RATING FUNCTIONS ==========

def predict_student_rating(input_data):
    """Predict student performance rating using trained ML model"""
    if student_performance_model is None:
        # Fallback to simple calculation if model not available
        return calculate_fallback_rating(input_data)
    
    try:
        # Prepare input features in the order expected by the model
        features = np.array([[
            input_data['attendance'],
            1 if input_data['extracurricular'].upper() == 'YES' else 0,
            input_data['previous_score_1'],
            input_data['previous_score_2'],
            input_data['assignment_score'],
            input_data['previous_semester_cgpa']
        ]])
        
        # Predict rating (1-10)
        rating = student_performance_model.predict(features)[0]
        
        # Ensure rating is within 1-10 range
        rating = max(1, min(10, rating))
        
        return float(rating)
    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return calculate_fallback_rating(input_data)

def calculate_fallback_rating(input_data):
    """Fallback rating calculation if ML model is not available"""
    # Simple weighted average calculation
    weights = {
        'attendance': 0.15,
        'extracurricular': 0.05,
        'previous_score_1': 0.2,
        'previous_score_2': 0.2,
        'assignment_score': 0.15,
        'previous_semester_cgpa': 0.25
    }
    
    # Normalize inputs to 1-10 scale
    attendance_norm = min(10, input_data['attendance'] / 10)  # Assuming attendance is percentage
    extracurricular_norm = 10 if input_data['extracurricular'].upper() == 'YES' else 5
    prev_score_1_norm = min(10, input_data['previous_score_1'] / 10)  # Assuming scores are out of 100
    prev_score_2_norm = min(10, input_data['previous_score_2'] / 10)
    assignment_norm = min(10, input_data['assignment_score'] / 10)
    cgpa_norm = min(10, input_data['previous_semester_cgpa'])  # Assuming CGPA is on 10-point scale
    
    weighted_sum = (
        attendance_norm * weights['attendance'] +
        extracurricular_norm * weights['extracurricular'] +
        prev_score_1_norm * weights['previous_score_1'] +
        prev_score_2_norm * weights['previous_score_2'] +
        assignment_norm * weights['assignment_score'] +
        cgpa_norm * weights['previous_semester_cgpa']
    ) / sum(weights.values())
    
    return weighted_sum

def generate_student_performance_report(subject_ratings, overall_rating, extracurricular_activity=None):
    """Generate AI-powered report based on student performance ratings"""
    if not client:
        return "Error: Groq client not initialized. Please check your API key."
    
    # Prepare data for the prompt
    subjects_data = []
    for subject in subject_ratings:
        subjects_data.append(f"- {subject['name']}: Rating {subject['rating']:.1f}/10")
    
    subjects_str = "\n".join(subjects_data)
    
    # Determine strengths and weaknesses
    strong_subjects = [s['name'] for s in subject_ratings if s['rating'] >= 7]
    weak_subjects = [s['name'] for s in subject_ratings if s['rating'] < 5]
    average_subjects = [s['name'] for s in subject_ratings if 5 <= s['rating'] < 7]
    
    extracurricular_info = f"Extracurricular Activity: {extracurricular_activity}" if extracurricular_activity else "No extracurricular activity specified"
    
    prompt = f"""
You are an expert academic advisor and mentor. Generate a comprehensive, personalized student performance report based on the following data:

STUDENT PERFORMANCE DATA:
Overall Rating: {overall_rating:.1f}/10

Subject-wise Ratings:
{subjects_str}

{extracurricular_info}

Strong Subjects (Rating ≥ 7): {strong_subjects if strong_subjects else 'None'}
Average Subjects (Rating 5-7): {average_subjects if average_subjects else 'None'}
Weak Subjects (Rating < 5): {weak_subjects if weak_subjects else 'None'}

Please generate a detailed report with the following sections:

1. EXECUTIVE SUMMARY
- Brief overview of overall academic performance
- Highlight key strengths and areas of concern

2. SUBJECT-WISE ANALYSIS
- For each subject, provide specific feedback based on the rating
- Suggest improvement strategies for weaker subjects
- Recommend how to leverage strengths in strong subjects

3. EXTRACURRICULAR IMPACT
- Analyze how extracurricular activities (if any) complement academic performance
- Suggest how to balance academics and extracurriculars

4. AREAS FOR IMPROVEMENT
- Specific recommendations for subjects needing attention
- Study techniques and resources for weaker areas
- Time management and study schedule suggestions

5. STRENGTHS AND OPPORTUNITIES
- Acknowledge strong areas and suggest how to build on them
- Recommend advanced topics or competitions in strong subjects
- Suggest leadership opportunities based on performance

6. ACTION PLAN
- Specific steps to improve overall rating
- Short-term goals (next semester)
- Long-term academic/career recommendations

Write in a supportive, encouraging tone while being honest and constructive. Use professional language and avoid markdown formatting.
"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

def generate_student_pdf_report(subject_ratings, overall_rating, report_text, extracurricular_activity=None):
    """Generate PDF report for student performance"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#4A90E2'),
        alignment=1  # Center alignment
    )
    
    story.append(Paragraph("Student Performance Rating Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Overall Rating
    story.append(Paragraph("Overall Performance", styles['Heading2']))
    overall_style = ParagraphStyle(
        'Overall',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.HexColor('#4A90E2'),
        alignment=1,
        spaceAfter=20
    )
    story.append(Paragraph(f"Overall Rating: {overall_rating:.1f}/10", overall_style))
    
    # Subject Ratings Table
    story.append(Paragraph("Subject-wise Ratings", styles['Heading2']))
    
    table_data = [["Subject", "Rating (out of 10)"]]
    for subject in subject_ratings:
        table_data.append([subject['name'], f"{subject['rating']:.1f}"])
    
    table = Table(table_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    if extracurricular_activity:
        story.append(Paragraph("Extracurricular Activity", styles['Heading2']))
        story.append(Paragraph(f"• {extracurricular_activity}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Detailed Report
    story.append(Paragraph("Detailed Analysis & Recommendations", styles['Heading2']))
    
    # Split report into paragraphs and add
    report_paragraphs = report_text.split('\n\n')
    for para in report_paragraphs:
        if para.strip():
            # Check if it's a heading
            if para.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.')) or 'SUMMARY' in para.upper() or 'ANALYSIS' in para.upper():
                story.append(Paragraph(para.strip(), styles['Heading3']))
            else:
                story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ========== PYDANTIC MODELS ==========

class RoleSelection(BaseModel):
    role: str

class ChatRequest(BaseModel):
    message: str
    preview: List[Dict]
    history: Optional[List[Dict]] = []
    role: Optional[str] = None
    analysis_context: Optional[str] = None

class ExportRequest(BaseModel):
    preview: List[Dict]
    format: str
    insights: str = ""
    role: str = "general"
    theme: Dict = None
    include_charts: bool = True
    include_raw_data: bool = True

class StudentRatingRequest(BaseModel):
    num_subjects: int
    subjects: List[Dict]  # Each dict: {"name": "subject_name", "inputs": {...}}
    extracurricular: str
    extracurricular_activity: Optional[str] = None

# ========== API ENDPOINTS ==========

@app.post("/api/select_role")
async def select_role(request: RoleSelection):
    """Endpoint for role selection"""
    try:
        role = request.role.lower()
        if role not in ["student", "faculty", "admin"]:
            return {
                "status": "error",
                "message": "Invalid role. Please select student, faculty, or admin."
            }
        
        return {
            "status": "success",
            "role": role,
            "message": f"Role selected: {role}",
            "analysis_count": ROLE_ANALYSIS_COUNTS.get(role, 20)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/upload")
async def upload_file(role: str, file: UploadFile = File(...)):
    """Handle file upload and initial processing with role context"""
    try:
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.file.seek(0)
                    df = pd.read_csv(file.file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if df is None:
                file.file.seek(0)
                content = await file.read()
                try:
                    content_str = content.decode('utf-8', errors='ignore')
                    df = pd.read_csv(StringIO(content_str))
                except:
                    file.file.seek(0)
                    df = pd.read_csv(file.file, encoding='utf-8', errors='ignore')
                    
        elif file_extension in ['xlsx', 'xls']:
            file.file.seek(0)
            df = pd.read_excel(file.file, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        categorical_cols = detect_categorical_columns(df)
        datetime_cols = detect_datetime_columns(df)
        
        for col in datetime_cols:
            if not is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        preview_data = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "is_categorical": col in categorical_cols,
                "is_datetime": col in datetime_cols,
                "is_numeric": df[col].dtype in ['int64', 'float64'],
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum())
            }
            columns_info.append(col_info)
        
        response = {
            "status": "success",
            "filename": file.filename,
            "role": role,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "preview": preview_data,
            "columns_info": columns_info,
            "numeric_columns": [str(col) for col in df.select_dtypes(include=[np.number]).columns.tolist()],
            "categorical_columns": [str(col) for col in categorical_cols],
            "date_columns": [str(col) for col in datetime_cols],
            "data_quality": {
                "total_missing": int(df.isnull().sum().sum()),
                "completeness": float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                "duplicate_rows": int(df.duplicated().sum())
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/api/get_ai_insights")
async def get_insights(request: Request):
    """Get AI insights about the dataset based on user role"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        role = data.get("role", "general")
        
        if not df_data or len(df_data) == 0:
            return {
                "status": "error", 
                "message": "No data available for analysis. Please upload a file first."
            }
        
        df = pd.DataFrame(df_data)
        
        if df.empty or df.shape[1] == 0:
            return {
                "status": "error", 
                "message": "Dataset is empty or has no valid columns."
            }
        
        insights = get_ai_insights(df, role)
        return {"status": "success", "insights": insights}
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error generating insights: {str(e)}"
        }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chat with the data using AI with role context"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        question = data.get("message", "")
        role = data.get("role", "general")
        
        if not df_data or len(df_data) == 0:
            return {
                "status": "error",
                "message": "No data available for analysis. Please upload a file first."
            }
        
        if not question:
            return {
                "status": "error",
                "message": "Please provide a question."
            }
        
        df = pd.DataFrame(df_data)
        
        if df.empty or df.shape[1] == 0:
            return {
                "status": "error",
                "message": "Dataset is empty or has no valid columns."
            }
        
        response = chat_with_data(df, question, role)
        return {"status": "success", "response": response}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Chat failed: {str(e)}"
        }

@app.post("/api/analyze_with_ai")
async def analyze_with_ai(request: Request):
    """Analyze dataset with AI and get recommendations based on role"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        role = data.get("role", "general")
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        
        available_analyses = get_available_analyses_for_role(role)
        
        if not available_analyses:
            available_analyses = ["general_insights_analysis", "data_quality_report"]
        
        print(f"Using {len(available_analyses)} available analyses for role {role}")
        
        if available_analyses and groq_analyzer.client:
            analysis_result = groq_analyzer.analyze_dataset_and_suggest_functions(df, available_analyses, role)
        else:
            analysis_result = {
                "analysis_recommendations": [
                    {
                        "name": available_analyses[0] if available_analyses else "general_insights_analysis",
                        "description": f"General data overview and analysis for {role}",
                        "columns": df.columns.tolist()[:3] if not df.empty else []
                    }
                ],
                "confidence_score": 0.7
            }
        
        return {
            "status": "success",
            "analysis_result": analysis_result,
            "available_analyses": available_analyses,
            "role": role
        }
            
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        return {
            "status": "success",
            "analysis_result": {
                "analysis_recommendations": [
                    {
                        "name": "general_insights_analysis",
                        "description": "Basic data overview and quality assessment",
                        "columns": []
                    }
                ],
                "confidence_score": 0.5
            },
            "available_analyses": ["general_insights_analysis"],
            "role": "general",
            "message": f"Using fallback analysis due to: {str(e)}"
        }

@app.post("/api/run_analysis_function")
async def run_analysis_function(request: Request):
    """Run a specific analysis function based on role"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        analysis_name = data.get("analysis_name")
        role = data.get("role", "general")
        
        print(f"Running analysis: {analysis_name} for role: {role}")
        
        if not df_data or not analysis_name:
            return {
                "status": "error",
                "message": "Missing data or analysis name",
                "details": "No preview data or analysis name provided"
            }
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return {
                "status": "error",
                "message": "Empty dataset",
                "details": "The provided dataset is empty"
            }
        
        try:
            try:
                if role in ["student", "faculty", "admin"]:
                    module = importlib.import_module(f"analyses.{role}")
                else:
                    from analyses import general as module
            except ImportError as e:
                print(f"Analysis module not found: {e}")
                from analyses import general as module
                role = "general"
            
            if hasattr(module, analysis_name):
                analysis_function = getattr(module, analysis_name)
                
                if callable(analysis_function):
                    print(f"Executing analysis function: {analysis_name}")
                    
                    try:
                        result = analysis_function(df)
                        
                        processed_result = {}
                        if isinstance(result, dict):
                            for key, value in result.items():
                                if hasattr(value, 'to_json'):
                                    processed_result[key] = value.to_json()
                                elif hasattr(value, 'show'):
                                    processed_result[key] = f"Visualization available for {key}"
                                else:
                                    processed_result[key] = convert_numpy_types(value)
                        else:
                            processed_result = {"analysis_output": convert_numpy_types(result)}
                        
                        print(f"Analysis completed successfully: {analysis_name}")
                        
                        return {
                            "status": "success",
                            "analysis_name": analysis_name,
                            "role": role,
                            "result": processed_result
                        }
                        
                    except Exception as e:
                        print(f"Error executing analysis function: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Error executing analysis '{analysis_name}': {str(e)}",
                            "traceback": traceback.format_exc()
                        }
                else:
                    return {
                        "status": "error",
                        "message": f"Analysis function '{analysis_name}' is not callable"
                    }
            else:
                available_functions = [attr for attr in dir(module) if callable(getattr(module, attr)) and not attr.startswith('_')]
                return {
                    "status": "error",
                    "message": f"Analysis function '{analysis_name}' not found in {role} module",
                    "available_functions": available_functions
                }
                
        except Exception as e:
            print(f"Module import error: {str(e)}")
            return {
                "status": "error",
                "message": f"Error accessing analysis module: {str(e)}"
            }
            
    except Exception as e:
        print(f"General analysis error: {str(e)}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.post("/api/get_time_series_data")
async def get_time_series_data(request: Request):
    """Get time series analysis data"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        date_col = data.get("date_col")
        num_col = data.get("num_col")
        period = data.get("period", 30)
        theme = data.get("theme", {})
        
        if not df_data:
            return {
                "status": "error",
                "message": "Missing preview data"
            }
        
        df = pd.DataFrame(df_data)
        
        date_cols = []
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif df[col].dtype == 'object':
                sample = df[col].dropna().head(3)
                if len(sample) > 0:
                    try:
                        test_conversion = pd.to_datetime(sample, errors='coerce')
                        if not test_conversion.isna().all():
                            date_cols.append(col)
                    except:
                        pass
        
        numeric_cols = []
        for col in df.columns:
            if (is_numeric_dtype(df[col]) and 
                col not in date_cols and
                not is_year_or_time_column(df[col])):
                numeric_cols.append(col)
        
        if not date_col or not num_col:
            return {
                "status": "success",
                "date_cols": date_cols,
                "numeric_cols": numeric_cols,
                "message": "Please select date and numeric columns"
            }
        
        results = analyze_time_series(df, theme, date_col, num_col, period)
        return {"status": "success", "results": results}
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Time series analysis failed: {str(e)}"
        }

@app.post("/api/get_relationships_data")
async def get_relationships_data(request: Request):
    """Get relationships analysis data"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        theme = data.get("theme", {})
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        results = analyze_relationships(df, theme)
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate_visualizations")
async def generate_dashboard_visualizations(request: Request):
    """Generate dashboard visualizations for the dataset"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        theme = data.get("theme", {
            "bg": "#0A071E",
            "text": "#ffffff",
            "primary": "#8a2be2",
            "secondary": "#1a183c",
            "chart_bg": "#1a183c",
            "grid": "#3a3863"
        })
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        visualizations = generate_visualizations(df, theme)
        key_metrics = show_key_metrics(df)
        
        return {
            "status": "success",
            "visualizations": visualizations,
            "key_metrics": key_metrics
        }
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat_with_data")
async def chat_with_data_endpoint(request: ChatRequest):
    """Enhanced chat endpoint"""
    try:
        df = pd.DataFrame(request.preview)
        
        role = getattr(request, 'role', None)
        analysis_context = getattr(request, 'analysis_context', None)
        
        response = chat_with_data(df, request.message, role, analysis_context)
        
        return {
            "status": "success", 
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "role": role
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/export_dashboard")
async def export_dashboard(request: ExportRequest):
    """Export dashboard with multiple format support"""
    try:
        df = pd.DataFrame(request.preview)
        
        if request.format == "pdf":
            pdf_buffer = generate_pdf_report(
                df, 
                request.role, 
                request.insights, 
                theme=request.theme
            )
            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=autodash_rit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
            )
        elif request.format == "excel":
            excel_buffer = generate_excel_report(df, request.role, request.insights)
            return StreamingResponse(
                excel_buffer,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=autodash_rit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
            )
        elif request.format == "html":
            html_content = generate_html_report(df, request.role, request.insights, request.theme)
            return HTMLResponse(
                content=html_content,
                headers={"Content-Disposition": f"attachment; filename=autodash_rit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"}
            )
        else:
            return {"status": "error", "message": "Supported formats: pdf, excel, html"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/student_performance_rating")
async def student_performance_rating(request: StudentRatingRequest):
    """Calculate student performance ratings and generate report"""
    try:
        
        subject_ratings = []
        
        for subject_data in request.subjects:
            inputs = subject_data.get("inputs", {})
            
            # Validate required inputs
            required_inputs = ["Attendance", "Extracurricular_Activities", 
                             "Previous_Scores_1", "previous_Score_2", 
                             "Assignment_Score", "Previous_Semester_CGPA"]
            
            missing_inputs = [req for req in required_inputs if req not in inputs]
            if missing_inputs:
                return {
                    "status": "error",
                    "message": f"Missing inputs for subject {subject_data['name']}: {missing_inputs}"
                }
            
            # Prepare input data for prediction
            prediction_input = {
                'attendance': float(inputs["Attendance"]),
                'extracurricular': inputs["Extracurricular_Activities"],
                'previous_score_1': float(inputs["Previous_Scores_1"]),
                'previous_score_2': float(inputs["previous_Score_2"]),
                'assignment_score': float(inputs["Assignment_Score"]),
                'previous_semester_cgpa': float(inputs["Previous_Semester_CGPA"])
            }
            
            # Get rating from model
            rating = predict_student_rating(prediction_input)
            
            subject_ratings.append({
                "name": subject_data["name"],
                "rating": rating
            })
        
        # Calculate overall rating (mean of subject ratings)
        overall_rating = sum(s["rating"] for s in subject_ratings) / len(subject_ratings)
        
        # Generate AI report
        report = generate_student_performance_report(
            subject_ratings, 
            overall_rating, 
            request.extracurricular_activity if request.extracurricular.upper() == 'YES' else None
        )
        
        return {
            "status": "success",
            "subject_ratings": subject_ratings,
            "overall_rating": overall_rating,
            "report": report
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error calculating performance rating: {str(e)}"
        }

@app.post("/api/export_student_report")
async def export_student_report(request: Request):
    """Export student performance report as PDF"""
    try:
        data = await request.json()
        subject_ratings = data.get("subject_ratings", [])
        overall_rating = data.get("overall_rating", 0)
        report_text = data.get("report", "")
        extracurricular_activity = data.get("extracurricular_activity")
        
        pdf_buffer = generate_student_pdf_report(
            subject_ratings,
            overall_rating,
            report_text,
            extracurricular_activity
        )
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=student_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
        )
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/get_available_analyses")
async def get_available_analyses(role: str = "general"):
    """Get available analysis functions for a role"""
    try:
        available_analyses = get_available_analyses_for_role(role)
        
        return {
            "status": "success",
            "available_analyses": available_analyses,
            "role": role,
            "total_count": len(available_analyses),
            "role_limit": ROLE_ANALYSIS_COUNTS.get(role, 20)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/debug/available_analyses")
async def debug_available_analyses(role: str = "student"):
    """Debug endpoint to check available analyses for a role"""
    try:
        available_functions = get_available_analyses_for_role(role)
        
        return {
            "status": "success",
            "role": role,
            "available_functions": available_functions,
            "count": len(available_functions)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the frontend HTML"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content="<h1>AutoDash RIT Backend is Running</h1><p>Templates not configured</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)