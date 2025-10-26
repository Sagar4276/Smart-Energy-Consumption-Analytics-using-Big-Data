#!/usr/bin/env python3
"""
Smart Energy Analytics Dashboard - Complete Storytelling Experience
Comprehensive visualization following the 7-layer roadmap for big data insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="⚡ Smart-Energy-Consumption-Analytics-using-Big-Data",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Sagar4276/Smart-Energy-Consumption-Analytics-using-Big-Data',
        'Report a bug': 'https://github.com/Sagar4276/Smart-Energy-Consumption-Analytics-using-Big-Data/issues',
        'About': '''
        **Smart Energy Analytics Dashboard**
        *Complete Big Data Pipeline Intelligence*

        Built with: Apache Spark 3.5.0, PySpark ML, Streamlit
        Optimized for: 8GB RAM systems with Docker containers
        '''
    }
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; font-weight: 700;}
    .section-header {font-size: 2rem; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 0.5rem;}
    .metric-value {font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;}
    .metric-label {font-size: 0.9rem; opacity: 0.9;}
    .insight-box {background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;}
    .warning-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {padding: 1rem 2rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "model"

# Memory-safe default sample size (but user can increase to full dataset)
DEFAULT_SAMPLE_SIZE = 50000
MAX_SAMPLE_SIZE = 500000  # Allow up to 500k for powerful filtering

# ============================================================================
# DATA LOADING FUNCTIONS (NO SPARK FOR STREAMLIT - DIRECT PARQUET)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_parquet_optimized(file_path: Path, nrows: int = None) -> pd.DataFrame:
    """
    Load Parquet file directly with pandas (faster for samples)
    For Big Data Analytics, allows loading up to 500k records
    """
    try:
        if not file_path.exists():
            st.error(f"❌ File not found: {file_path}")
            return pd.DataFrame()
        
        # Use pandas for direct Parquet reading
        df = pd.read_parquet(file_path)
        
        total_rows = len(df)
        
        # Sample if requested (for dashboard performance)
        if nrows and nrows < total_rows:
            df = df.sample(n=nrows, random_state=42).reset_index(drop=True)
            st.info(f"📊 Loaded {nrows:,} / {total_rows:,} records (sampled for performance)")
        else:
            st.success(f"📊 Loaded FULL DATASET: {total_rows:,} records - TRUE BIG DATA!")
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file_path.name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_model_metadata() -> dict:
    """Load model metadata from text file"""
    metadata_file = MODEL_PATH / "model_metadata.txt"
    metadata = {}
    
    try:
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
    except Exception as e:
        st.warning(f"⚠️ Could not load model metadata: {e}")
    
    return metadata


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_energy_distribution(df: pd.DataFrame, column='daily_energy_kwh'):
    """Plot energy consumption distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if column in df.columns:
        ax.hist(df[column].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Energy (kWh)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{column.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Column "{column}" not found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def plot_energy_boxplot(df: pd.DataFrame, column='daily_energy_kwh'):
    """Plot energy consumption boxplot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if column in df.columns:
        ax.boxplot(df[column].dropna(), vert=True, patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Energy (kWh)', fontsize=12)
        ax.set_title(f'{column.replace("_", " ").title()} Box Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, f'Column "{column}" not found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def plot_seasonal_patterns(df: pd.DataFrame):
    """Plot seasonal patterns (monthly and weekly)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if 'date' in df.columns and 'daily_energy_kwh' in df.columns:
        df_plot = df.copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'])
        df_plot['month'] = df_plot['date'].dt.month
        df_plot['weekday'] = df_plot['date'].dt.day_name()
        
        # Monthly patterns
        monthly_avg = df_plot.groupby('month')['daily_energy_kwh'].mean()
        ax1.bar(monthly_avg.index, monthly_avg.values, color='lightgreen', alpha=0.7)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Average Energy (kWh)', fontsize=12)
        ax1.set_title('Average Energy by Month', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, 13))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Weekly patterns
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = df_plot.groupby('weekday')['daily_energy_kwh'].mean().reindex(weekday_order)
        ax2.bar(range(len(weekly_avg)), weekly_avg.values, color='coral', alpha=0.7)
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Average Energy (kWh)', fontsize=12)
        ax2.set_title('Average Energy by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(weekday_order)))
        ax2.set_xticklabels(weekday_order, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'Date or energy data not available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'Date or energy data not available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns=None):
    """Plot correlation heatmap for selected columns"""
    if columns is None:
        # Select numeric columns, limit to avoid overcrowding
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = numeric_cols[:15]  # Limit to 15 columns for readability
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_multiple_households(df: pd.DataFrame, household_ids: list):
    """Plot time series comparison for multiple households"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10.colors
    
    for i, hid in enumerate(household_ids):
        household_data = df[df['LCLid'] == hid].copy()
        if not household_data.empty and 'date' in household_data.columns:
            household_data['date'] = pd.to_datetime(household_data['date'])
            household_data = household_data.sort_values('date')
            
            ax.plot(household_data['date'], household_data['daily_energy_kwh'], 
                    linewidth=1.5, color=colors[i % len(colors)], 
                    alpha=0.8, label=f'{hid}')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Energy (kWh)', fontsize=12)
    ax.set_title(f'Energy Consumption - {len(household_ids)} Households', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(features_data: pd.DataFrame):
    """Plot feature importance if available (simplified version)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # For demonstration, show correlation with target
    if 'daily_energy_kwh' in features_data.columns:
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns
        correlations = features_data[numeric_cols].corr()['daily_energy_kwh'].abs().sort_values(ascending=True)
        correlations = correlations.drop('daily_energy_kwh')  # Remove self-correlation
        
        ax.barh(range(len(correlations)), correlations.values, color='skyblue', alpha=0.7)
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(correlations.index, fontsize=10)
        ax.set_xlabel('Absolute Correlation with Energy', fontsize=12)
        ax.set_title('Feature Correlation with Target (Proxy for Importance)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'Target variable not available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def plot_prediction_scatter(actual: pd.Series, predicted: pd.Series):
    """Plot predictions vs actual values scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, color='blue', edgecolors='black', s=30)

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Energy (kWh)', fontsize=12)
    ax.set_ylabel('Predicted Energy (kWh)', fontsize=12)
    ax.set_title('Prediction vs Actual Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_error_distribution(actual: pd.Series, predicted: pd.Series):
    """Plot distribution of prediction errors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    errors = predicted - actual

    # Histogram
    ax1.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
    ax1.set_xlabel('Prediction Error (kWh)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution Histogram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot (simplified)
    errors_sorted = np.sort(errors)
    theoretical_quantiles = np.random.normal(errors.mean(), errors.std(), len(errors_sorted))
    theoretical_quantiles = np.sort(theoretical_quantiles)

    ax2.scatter(theoretical_quantiles, errors_sorted, alpha=0.6, color='green', edgecolors='black', s=20)
    min_val = min(theoretical_quantiles.min(), errors_sorted.min())
    max_val = max(theoretical_quantiles.max(), errors_sorted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Normal Reference')
    ax2.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
    ax2.set_ylabel('Sample Quantiles (Errors)', fontsize=12)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance_detailed(features_data: pd.DataFrame, model_metadata: dict = None):
    """Plot detailed feature importance analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    if 'daily_energy_kwh' in features_data.columns:
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns
        correlations = features_data[numeric_cols].corr()['daily_energy_kwh'].abs().sort_values(ascending=False)
        correlations = correlations.drop('daily_energy_kwh')  # Remove self-correlation

        # Top 10 features
        top_features = correlations.head(10)

        # Bar chart
        ax1.barh(range(len(top_features)), top_features.values, color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index, fontsize=10)
        ax1.set_xlabel('Absolute Correlation', fontsize=12)
        ax1.set_title('Top 10 Feature Correlations', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Feature categories
        lag_features = [f for f in correlations.index if 'lag_' in f]
        rolling_features = [f for f in correlations.index if 'rolling_' in f]
        seasonal_features = [f for f in correlations.index if any(x in f for x in ['is_', 'month', 'day', 'season'])]
        other_features = [f for f in correlations.index if f not in lag_features + rolling_features + seasonal_features]

        categories = ['Lag Features', 'Rolling Features', 'Seasonal Features', 'Other Features']
        counts = [len(lag_features), len(rolling_features), len(seasonal_features), len(other_features)]
        colors = ['lightcoral', 'lightgreen', 'lightblue', 'gold']

        ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Categories Distribution', fontsize=14, fontweight='bold')

        # Correlation heatmap for top features
        top_feature_names = top_features.index.tolist()[:8]  # Top 8 for heatmap
        if len(top_feature_names) > 1:
            corr_matrix = features_data[top_feature_names + ['daily_energy_kwh']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3,
                       fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
            ax3.set_title('Correlation Heatmap (Top Features)', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Insufficient features for heatmap', ha='center', va='center', transform=ax3.transAxes)

        # Feature vs target scatter for top feature
        if len(top_feature_names) > 0:
            top_feature = top_feature_names[0]
            ax4.scatter(features_data[top_feature], features_data['daily_energy_kwh'],
                       alpha=0.6, color='purple', edgecolors='black', s=20)
            ax4.set_xlabel(f'{top_feature.replace("_", " ").title()}', fontsize=12)
            ax4.set_ylabel('Daily Energy (kWh)', fontsize=12)
            ax4.set_title(f'Top Feature vs Target Scatter', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No features available', ha='center', va='center', transform=ax4.transAxes)

    else:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Target variable not available', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return fig


def plot_anomaly_distribution(df: pd.DataFrame):
    """Plot anomaly distribution and statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    if 'is_anomaly' in df.columns and 'daily_energy_kwh' in df.columns:
        # Anomaly vs Normal distribution
        normal_data = df[df['is_anomaly'] == 0]['daily_energy_kwh']
        anomaly_data = df[df['is_anomaly'] == 1]['daily_energy_kwh']

        ax1.hist([normal_data, anomaly_data], bins=50, alpha=0.7, label=['Normal', 'Anomaly'],
                color=['blue', 'red'], edgecolor='black')
        ax1.set_xlabel('Daily Energy (kWh)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Energy Distribution: Normal vs Anomalies', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        data_to_plot = [normal_data.dropna(), anomaly_data.dropna()]
        ax2.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
        ax2.set_ylabel('Daily Energy (kWh)', fontsize=12)
        ax2.set_title('Box Plot: Normal vs Anomalies', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Anomaly rate over time (if date available)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            monthly_anomalies = df.groupby('month')['is_anomaly'].mean() * 100

            monthly_anomalies.plot(kind='line', ax=ax3, marker='o', color='darkgreen')
            ax3.set_xlabel('Month', fontsize=12)
            ax3.set_ylabel('Anomaly Rate (%)', fontsize=12)
            ax3.set_title('Monthly Anomaly Rate Trend', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Date column not available', ha='center', va='center', transform=ax3.transAxes)

        # Anomaly statistics
        total_records = len(df)
        anomaly_count = (df['is_anomaly'] == 1).sum()
        anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0

        stats_text = f"""
        Total Records: {total_records:,}
        Anomalies: {anomaly_count:,}
        Anomaly Rate: {anomaly_rate:.2f}%

        Normal Mean: {normal_data.mean():.2f} kWh
        Normal Std: {normal_data.std():.2f} kWh
        Anomaly Mean: {anomaly_data.mean():.2f} kWh
        Anomaly Std: {anomaly_data.std():.2f} kWh
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Anomaly Statistics', fontsize=14, fontweight='bold')
        ax4.axis('off')

    else:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Required columns not available', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return fig


def plot_anomaly_time_series(df: pd.DataFrame, household_id: str):
    """Plot time series with anomalies highlighted"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    household_data = df[df['LCLid'] == household_id].copy()
    
    if not household_data.empty and 'date' in household_data.columns and 'is_anomaly' in household_data.columns:
        household_data['date'] = pd.to_datetime(household_data['date'])
        household_data = household_data.sort_values('date')
        
        # Normal points
        normal_data = household_data[household_data['is_anomaly'] == 0]
        ax.scatter(normal_data['date'], normal_data['daily_energy_kwh'], 
                  color='blue', alpha=0.6, s=20, label='Normal')
        
        # Anomalies
        anomaly_data = household_data[household_data['is_anomaly'] == 1]
        ax.scatter(anomaly_data['date'], anomaly_data['daily_energy_kwh'], 
                  color='red', alpha=0.8, s=40, marker='x', label='Anomaly')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Energy (kWh)', fontsize=12)
        ax.set_title(f'Anomaly Detection - {household_id}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, 'Required data not available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, household_id: str):
    """Plot time series for a specific household"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    household_data = df[df['LCLid'] == household_id].copy()
    
    if not household_data.empty and 'date' in household_data.columns:
        household_data['date'] = pd.to_datetime(household_data['date'])
        household_data = household_data.sort_values('date')
        
        ax.plot(household_data['date'], household_data['daily_energy_kwh'], 
                linewidth=1.5, color='steelblue', alpha=0.8)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Energy (kWh)', fontsize=12)
        ax.set_title(f'Energy Consumption - {household_id}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def plot_anomaly_distribution(df: pd.DataFrame):
    """Plot anomaly distribution by household"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'is_anomaly' in df.columns:
        # Anomaly counts
        anomaly_counts = df['is_anomaly'].value_counts()
        ax1.pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%', 
                startangle=90, colors=['lightgreen', 'salmon'])
        ax1.set_title('Anomaly Distribution', fontsize=14, fontweight='bold')
        
        # Top households with anomalies
        if 'LCLid' in df.columns:
            anomalies = df[df['is_anomaly'] == 1]
            top_households = anomalies['LCLid'].value_counts().head(10)
            ax2.barh(range(len(top_households)), top_households.values, color='coral')
            ax2.set_yticks(range(len(top_households)))
            ax2.set_yticklabels(top_households.index, fontsize=10)
            ax2.set_xlabel('Anomaly Count', fontsize=12)
            ax2.set_title('Top 10 Households with Anomalies', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax1.text(0.5, 0.5, 'No anomaly data', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No anomaly data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-header">⚡ Smart Energy Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Optimized for 8GB Systems** | Powered by Apache Spark & Machine Learning")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Dashboard Settings")
    
    st.sidebar.markdown("### 📊 Data Sample Size")
    sample_size = st.sidebar.select_slider(
        "Choose sample size",
        options=[10000, 25000, 50000, 100000, 250000, 500000, "FULL"],
        value=50000,
        help="Larger = more data but slower. 'FULL' loads entire dataset (millions of rows)!"
    )
    
    # Convert "FULL" to None (load everything)
    if sample_size == "FULL":
        sample_size_value = None
        st.sidebar.warning("🚀 FULL DATASET MODE - This is TRUE Big Data! May take 1-2 minutes to load.")
    else:
        sample_size_value = sample_size
        st.sidebar.success(f"✓ Loading {sample_size:,} records")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Recommendation:**")
    st.sidebar.markdown("- **Fast preview**: 50k records")
    st.sidebar.markdown("- **Detailed analysis**: 250k records")
    st.sidebar.markdown("- **Complete dataset**: FULL mode")
    
    # System info
    with st.sidebar.expander("📊 System Info"):
        dataset_info = f"""
Project: {PROJECT_ROOT.name}
Data Path: {BASE_PATH.relative_to(PROJECT_ROOT)}
Sample Mode: {'FULL DATASET' if sample_size_value is None else f'{sample_size_value:,} records'}
Spark Config: Optimized for 8GB (see spark_config.py)

📈 Full Dataset Sizes:
  - Daily: ~3.5 million records
  - Features: ~2 million records  
  - Anomalies: ~2 million records
  - Households: 5,567
        """
        st.code(dataset_info)
    
    # Load data
    with st.spinner(f"📂 Loading {'FULL DATASET (millions of rows)' if sample_size_value is None else f'{sample_size_value:,} records'}..."):
        daily_data = load_parquet_optimized(BASE_PATH / "daily", nrows=sample_size_value)
        features_data = load_parquet_optimized(BASE_PATH / "energy_features", nrows=sample_size_value)
        anomaly_data = load_parquet_optimized(BASE_PATH / "anomalies", nrows=sample_size_value)
        model_metadata = load_model_metadata()
    
    # Data status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Daily Records", f"{len(daily_data):,}" if not daily_data.empty else "0")
    with col2:
        st.metric("🔮 Features", f"{len(features_data):,}" if not features_data.empty else "0")
    with col3:
        st.metric("🚨 Anomalies", f"{len(anomaly_data):,}" if not anomaly_data.empty else "0")
    with col4:
        households = daily_data['LCLid'].nunique() if not daily_data.empty and 'LCLid' in daily_data.columns else 0
        st.metric("🏠 Households", f"{households:,}")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔮 Forecasting", "🚨 Anomalies"])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.header("📊 Data Overview")
        
        if daily_data.empty:
            st.error("❌ No daily data available. Please run the pipeline.")
            st.code("docker exec energy-analytics python3 /app/scripts/data_ingestion.py")
            return
        
        st.success(f"✓ Loaded {len(daily_data):,} records | {households:,} households")
        
        # Energy distribution
        st.subheader("Energy Consumption Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_energy_distribution(daily_data)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig = plot_energy_boxplot(daily_data)
            st.pyplot(fig)
            plt.close()
        
        # Seasonal patterns
        st.subheader("Seasonal Consumption Patterns")
        fig = plot_seasonal_patterns(daily_data)
        st.pyplot(fig)
        plt.close()
        
        # Correlation heatmap (if features available)
        if not features_data.empty:
            st.subheader("Feature Correlations")
            # Select key features for heatmap
            key_features = ['daily_energy_kwh', 'rolling_avg_7d', 'rolling_avg_30d', 
                          'lag_1_day', 'lag_7_day', 'is_weekend', 'is_winter', 'is_summer']
            available_features = [f for f in key_features if f in features_data.columns]
            if len(available_features) > 2:
                fig = plot_correlation_heatmap(features_data, available_features)
                st.pyplot(fig)
                plt.close()
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Statistics")
            if 'daily_energy_kwh' in daily_data.columns:
                stats_df = daily_data['daily_energy_kwh'].describe().to_frame()
                stats_df.columns = ['Daily Energy (kWh)']
                st.dataframe(stats_df, width=400)
        
        with col2:
            st.subheader("📅 Date Range")
            if 'date' in daily_data.columns:
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                st.write(f"**Start:** {daily_data['date'].min().strftime('%Y-%m-%d')}")
                st.write(f"**End:** {daily_data['date'].max().strftime('%Y-%m-%d')}")
                st.write(f"**Days:** {(daily_data['date'].max() - daily_data['date'].min()).days}")
        
        # Data preview
        with st.expander("📄 View Sample Data"):
            display_cols = ['LCLid', 'date', 'daily_energy_kwh']
            available_cols = [c for c in display_cols if c in daily_data.columns]
            st.dataframe(daily_data[available_cols].head(50), width='stretch')
    
    # ========================================================================
    # TAB 2: TRENDS
    # ========================================================================
    with tab2:
        st.header("📈 Energy Consumption Trends")
        
        if daily_data.empty or 'LCLid' not in daily_data.columns:
            st.warning("⚠️ No household data available")
            return
        
        # Household selector
        all_households = sorted(daily_data['LCLid'].unique())
        selected_household = st.selectbox("Select Household", options=all_households, index=0)
        
        # Time series plot
        st.subheader("Individual Household Time Series")
        fig = plot_time_series(daily_data, selected_household)
        st.pyplot(fig)
        plt.close()
        
        # Multiple households comparison
        st.subheader("Multi-Household Comparison")
        num_households = st.slider("Number of households to compare", 2, 10, 5)
        random_households = np.random.choice(all_households, size=min(num_households, len(all_households)), replace=False)
        
        fig = plot_multiple_households(daily_data, random_households.tolist())
        st.pyplot(fig)
        plt.close()
        
        # Household statistics
        household_data = daily_data[daily_data['LCLid'] == selected_household]
        
        if not household_data.empty and 'daily_energy_kwh' in household_data.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average", f"{household_data['daily_energy_kwh'].mean():.2f} kWh")
            with col2:
                st.metric("Median", f"{household_data['daily_energy_kwh'].median():.2f} kWh")
            with col3:
                st.metric("Max", f"{household_data['daily_energy_kwh'].max():.2f} kWh")
            with col4:
                st.metric("Std Dev", f"{household_data['daily_energy_kwh'].std():.2f} kWh")
    
    # ========================================================================
    # TAB 3: FORECASTING
    # ========================================================================
    with tab3:
        st.header("🔮 Energy Demand Forecasting")
        
        if model_metadata:
            st.success("✓ ML Model Available")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance")
                for key, value in model_metadata.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.subheader("📈 Model Details")
                st.info("""
                **Model Type:** Linear Regression with Cross-Validation
                
                **Features Used:** 35+ features including:
                - Lag features (1-30 days)
                - Rolling averages (7d, 30d)
                - Seasonal indicators
                - Tariff information
                """)
            
            # Feature importance visualization
            if not features_data.empty:
                st.subheader("🔍 Feature Importance Analysis")
                fig = plot_feature_importance_detailed(features_data, model_metadata)
                st.pyplot(fig)
                plt.close()
                
                # Additional model insights
                st.subheader("📊 Model Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    **Key Findings:**
                    - Lag features (1-7 days) show strongest correlation
                    - Rolling averages help capture trends
                    - Seasonal features improve accuracy
                    - Weekend patterns are significant
                    """)
                
                with col2:
                    st.info("""
                    **Performance Notes:**
                    - R² indicates good fit for forecasting
                    - MAE shows typical prediction error
                    - Model handles temporal dependencies well
                    - Cross-validation prevents overfitting
                    """)
        else:
            st.warning("⚠️ Model metadata not found. Train the model first.")
            st.code("docker exec energy-analytics python3 /app/scripts/forecasting_model.py")
        
        if not features_data.empty:
            st.subheader("🔍 Feature Sample")
            display_cols = ['LCLid', 'date', 'daily_energy_kwh', 'rolling_avg_7d', 'lag_1_day']
            available_cols = [c for c in display_cols if c in features_data.columns]
            st.dataframe(features_data[available_cols].head(20), width='stretch')
    
    # ========================================================================
    # TAB 4: ANOMALIES
    # ========================================================================
    with tab4:
        st.header("🚨 Anomaly Detection")
        
        if anomaly_data.empty or 'is_anomaly' not in anomaly_data.columns:
            st.warning("⚠️ No anomaly data available. Run anomaly detection first.")
            st.code("docker exec energy-analytics python3 /app/scripts/anomaly_detection.py")
            return
        
        # Anomaly metrics
        total_records = len(anomaly_data)
        anomaly_count = (anomaly_data['is_anomaly'] == 1).sum()
        anomaly_pct = (anomaly_count / total_records * 100) if total_records > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Anomalies Detected", f"{anomaly_count:,}")
        with col3:
            st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
        
        # Anomaly visualizations
        st.subheader("Anomaly Analysis Dashboard")
        fig = plot_anomaly_distribution(anomaly_data)
        st.pyplot(fig)
        plt.close()
        
        # Individual household anomaly view
        st.subheader("Individual Household Anomalies")
        all_households_anomaly = sorted(anomaly_data['LCLid'].unique())
        selected_household_anomaly = st.selectbox("Select Household for Anomaly View", 
                                                 options=all_households_anomaly, 
                                                 index=0, key="anomaly_household")
        
        fig = plot_anomaly_time_series(anomaly_data, selected_household_anomaly)
        st.pyplot(fig)
        plt.close()
        
        # Anomaly time series for specific household
        st.subheader("Anomaly Time Series")
        anomaly_households = sorted(anomaly_data[anomaly_data['is_anomaly'] == 1]['LCLid'].unique())
        if anomaly_households:
            selected_anomaly_household = st.selectbox("Select household with anomalies", options=anomaly_households, key="anomaly_select")
            fig = plot_anomaly_time_series(anomaly_data, selected_anomaly_household)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No households with detected anomalies in this sample.")
        
        # Top anomalies
        st.subheader("🔍 Recent Anomalies")
        anomalies_only = anomaly_data[anomaly_data['is_anomaly'] == 1].copy()
        
        if not anomalies_only.empty:
            if 'date' in anomalies_only.columns:
                anomalies_only['date'] = pd.to_datetime(anomalies_only['date'])
                anomalies_only = anomalies_only.sort_values('date', ascending=False)
            
            display_cols = ['LCLid', 'date', 'daily_energy_kwh', 'rolling_avg_7d']
            if 'distance_to_center' in anomalies_only.columns:
                display_cols.append('distance_to_center')
            
            available_cols = [c for c in display_cols if c in anomalies_only.columns]
            st.dataframe(anomalies_only[available_cols].head(20), width='stretch')
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Smart Energy Analytics Dashboard</strong> | Built with Streamlit & Apache Spark</p>
        <p>Optimized for 8GB Systems | Using spark_config.py for memory management</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
