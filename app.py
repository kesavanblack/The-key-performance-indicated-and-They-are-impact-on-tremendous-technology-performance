import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import io
import json

# Page configuration
st.set_page_config(page_title="Ultra Advanced Technology KPI Dashboard", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# Auto-refresh configuration
import time
refresh_interval = 5  # seconds

# Custom CSS with more styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for custom features
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'custom_thresholds' not in st.session_state:
    st.session_state.custom_thresholds = {
        'System_Uptime': 99.5,
        'Response_Time': 200,
        'CPU_Utilization': 80,
        'Memory_Usage': 85,
        'Security_Score': 80
    }
if 'favorite_metrics' not in st.session_state:
    st.session_state.favorite_metrics = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'notes' not in st.session_state:
    st.session_state.notes = []

# Enhanced Helper Functions
def detect_anomalies_ml(data):
    """Detect anomalies using Isolation Forest (ML-based)"""
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(data.values.reshape(-1, 1))
        return predictions == -1
    except:
        return np.zeros(len(data), dtype=bool)

def detect_anomalies(data, threshold=2):
    """Detect anomalies using Z-score method"""
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def predict_trend(data, days_ahead=7):
    """Predict future values using linear regression"""
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Calculate confidence interval
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    confidence = 1.96 * std_error  # 95% confidence
    
    return predictions, confidence

def calculate_sla_compliance(uptime_data, threshold=99.5):
    """Calculate SLA compliance percentage"""
    return (uptime_data >= threshold).sum() / len(uptime_data) * 100

def calculate_trend_direction(data, window=7):
    """Calculate if metric is trending up or down"""
    recent = data.tail(window).mean()
    previous = data.iloc[-2*window:-window].mean()
    return "up" if recent > previous else "down"

def generate_ai_insights(kpi_data):
    """Generate AI-powered insights from the data"""
    insights = []
    
    # Uptime analysis
    uptime_trend = calculate_trend_direction(kpi_data['System_Uptime'])
    if uptime_trend == "down":
        insights.append({
            'type': 'warning',
            'title': 'System Uptime Declining',
            'message': f'System uptime has decreased by {abs(kpi_data["System_Uptime"].iloc[-1] - kpi_data["System_Uptime"].iloc[-7]):.2f}% over the last week.',
            'recommendation': 'Consider reviewing recent deployments and infrastructure changes.'
        })
    
    # Response time analysis
    if kpi_data['Response_Time'].iloc[-1] > kpi_data['Response_Time'].mean() * 1.5:
        insights.append({
            'type': 'critical',
            'title': 'Response Time Spike Detected',
            'message': f'Current response time is {kpi_data["Response_Time"].iloc[-1]:.0f}ms, which is 50% above average.',
            'recommendation': 'Investigate database queries, API calls, and server load immediately.'
        })
    
    # Correlation analysis
    corr = kpi_data['CPU_Utilization'].corr(kpi_data['Response_Time'])
    if abs(corr) > 0.7:
        insights.append({
            'type': 'info',
            'title': 'Strong CPU-Response Time Correlation',
            'message': f'CPU utilization and response time are {abs(corr):.2f} correlated.',
            'recommendation': 'Consider horizontal scaling or optimizing CPU-intensive operations.'
        })
    
    # Security trend
    security_trend = calculate_trend_direction(kpi_data['Security_Score'])
    if security_trend == "up":
        insights.append({
            'type': 'success',
            'title': 'Security Posture Improving',
            'message': f'Security score has improved to {kpi_data["Security_Score"].iloc[-1]:.1f}.',
            'recommendation': 'Continue current security practices and monitoring.'
        })
    
    # Deployment success
    if kpi_data['Deployment_Success_Rate'].iloc[-7:].mean() < 90:
        insights.append({
            'type': 'warning',
            'title': 'Low Deployment Success Rate',
            'message': f'Only {kpi_data["Deployment_Success_Rate"].iloc[-7:].mean():.1f}% deployment success in the last week.',
            'recommendation': 'Review deployment pipeline, add more automated tests, and improve rollback procedures.'
        })
    
    return insights

def generate_alerts(kpi_data, custom_thresholds):
    """Generate real-time alerts based on custom thresholds"""
    alerts = []
    
    # Check System Uptime
    if kpi_data['System_Uptime'].iloc[-1] < custom_thresholds['System_Uptime']:
        alerts.append({
            'type': 'critical',
            'metric': 'System Uptime',
            'message': f"System Uptime below threshold: {kpi_data['System_Uptime'].iloc[-1]:.2f}% < {custom_thresholds['System_Uptime']}%",
            'timestamp': datetime.now()
        })
    
    # Check Response Time
    if kpi_data['Response_Time'].iloc[-1] > custom_thresholds['Response_Time']:
        alerts.append({
            'type': 'warning',
            'metric': 'Response Time',
            'message': f"High Response Time: {kpi_data['Response_Time'].iloc[-1]:.0f}ms > {custom_thresholds['Response_Time']}ms",
            'timestamp': datetime.now()
        })
    
    # Check CPU Utilization
    if kpi_data['CPU_Utilization'].iloc[-1] > custom_thresholds['CPU_Utilization']:
        alerts.append({
            'type': 'warning',
            'metric': 'CPU Utilization',
            'message': f"High CPU Utilization: {kpi_data['CPU_Utilization'].iloc[-1]:.1f}% > {custom_thresholds['CPU_Utilization']}%",
            'timestamp': datetime.now()
        })
    
    # Check Memory Usage
    if kpi_data['Memory_Usage'].iloc[-1] > custom_thresholds['Memory_Usage']:
        alerts.append({
            'type': 'critical',
            'metric': 'Memory Usage',
            'message': f"Critical Memory Usage: {kpi_data['Memory_Usage'].iloc[-1]:.1f}% > {custom_thresholds['Memory_Usage']}%",
            'timestamp': datetime.now()
        })
    
    # Check Security Score
    if kpi_data['Security_Score'].iloc[-1] < custom_thresholds['Security_Score']:
        alerts.append({
            'type': 'critical',
            'metric': 'Security Score',
            'message': f"Low Security Score: {kpi_data['Security_Score'].iloc[-1]:.1f} < {custom_thresholds['Security_Score']}",
            'timestamp': datetime.now()
        })
    
    return alerts

def export_to_excel(kpi_data, summary_df, alerts_df=None):
    """Export dashboard data to Excel with multiple sheets"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        kpi_data.to_excel(writer, sheet_name='Raw Data', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add statistics sheet
        stats_df = kpi_data.describe()
        stats_df.to_excel(writer, sheet_name='Statistics')
        
        # Add alerts if available
        if alerts_df is not None and len(alerts_df) > 0:
            alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
        
        # Add correlation matrix
        corr_df = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                            'Deployment_Success_Rate', 'Security_Score', 'CPU_Utilization']].corr()
        corr_df.to_excel(writer, sheet_name='Correlations')
    
    return output.getvalue()

def export_to_json(kpi_data):
    """Export data to JSON format"""
    return kpi_data.to_json(orient='records', date_format='iso')

def calculate_health_score(kpi_data):
    """Calculate overall system health score"""
    weights = {
        'System_Uptime': 0.25,
        'Response_Time': 0.15,  # Inverse (lower is better)
        'Code_Quality_Score': 0.15,
        'Deployment_Success_Rate': 0.15,
        'Security_Score': 0.20,
        'API_Success_Rate': 0.10
    }
    
    # Normalize response time (inverse)
    normalized_response = 100 - (kpi_data['Response_Time'].iloc[-1] / 5)  # Assuming max 500ms
    normalized_response = max(0, min(100, normalized_response))
    
    health_score = (
        kpi_data['System_Uptime'].iloc[-1] * weights['System_Uptime'] +
        normalized_response * weights['Response_Time'] +
        kpi_data['Code_Quality_Score'].iloc[-1] * weights['Code_Quality_Score'] +
        kpi_data['Deployment_Success_Rate'].iloc[-1] * weights['Deployment_Success_Rate'] +
        kpi_data['Security_Score'].iloc[-1] * weights['Security_Score'] +
        kpi_data['API_Success_Rate'].iloc[-1] * weights['API_Success_Rate']
    )
    
    return health_score

# Title with dynamic status
st.title("📊 Ultra Advanced Technology Performance KPI Dashboard")
st.markdown("*Real-time monitoring with AI insights, predictive analytics, and advanced anomaly detection*")

# Quick status indicator
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("### Live Dashboard")
with col2:
    st.success("🟢 Systems Operational")
with col3:
    st.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")

# Sidebar for filters and controls
st.sidebar.header("🎛️ Dashboard Controls")

# Time period filter
time_period = st.sidebar.selectbox(
    "📅 Select Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "Custom Range"]
)

if time_period == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

# Department filter
department = st.sidebar.multiselect(
    "🏢 Select Department",
    ["All", "Development", "Operations", "Security", "Data Science", "Infrastructure", "QA", "DevOps"],
    default=["All"]
)

# Environment filter
environment = st.sidebar.selectbox(
    "🌍 Environment",
    ["Production", "Staging", "Development", "All Environments"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Advanced Features")

# Feature toggles
enable_predictions = st.sidebar.checkbox("🔮 Enable Predictive Analytics", value=True)
enable_anomaly_detection = st.sidebar.checkbox("🔍 Enable Anomaly Detection", value=True)
enable_ml_anomalies = st.sidebar.checkbox("🤖 ML-Based Anomaly Detection", value=False)
enable_alerts = st.sidebar.checkbox("🚨 Enable Real-time Alerts", value=True)
enable_ai_insights = st.sidebar.checkbox("🧠 Enable AI Insights", value=True)
show_confidence_intervals = st.sidebar.checkbox("📊 Show Confidence Intervals", value=True)
enable_data_filtering = st.sidebar.checkbox("🔬 Enable Data Filtering", value=False)

prediction_days = st.sidebar.slider("📈 Prediction Days Ahead", 1, 30, 7)

# Auto-refresh controls
st.sidebar.markdown("---")
st.sidebar.header("🔄 Refresh Controls")
auto_refresh = st.sidebar.checkbox("⚡ Auto-Refresh", value=False)
if auto_refresh:
    refresh_rate = st.sidebar.slider("⏱️ Refresh Rate (seconds)", 5, 60, 10)
    st.sidebar.success(f"Dashboard will refresh every {refresh_rate} seconds")
    time.sleep(refresh_rate)
    st.rerun()

if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.rerun()

# Custom threshold settings
st.sidebar.markdown("---")
st.sidebar.header("⚠️ Custom Alert Thresholds")
with st.sidebar.expander("⚙️ Configure Thresholds"):
    st.session_state.custom_thresholds['System_Uptime'] = st.slider(
        "System Uptime (%)", 95.0, 100.0, 99.5, 0.1
    )
    st.session_state.custom_thresholds['Response_Time'] = st.slider(
        "Response Time (ms)", 100, 500, 200, 10
    )
    st.session_state.custom_thresholds['CPU_Utilization'] = st.slider(
        "CPU Utilization (%)", 50, 100, 80, 5
    )
    st.session_state.custom_thresholds['Memory_Usage'] = st.slider(
        "Memory Usage (%)", 50, 100, 85, 5
    )
    st.session_state.custom_thresholds['Security_Score'] = st.slider(
        "Security Score", 50, 100, 80, 5
    )

# Export options
st.sidebar.markdown("---")
st.sidebar.header("📥 Export Options")

# Theme selector
st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization Theme")
chart_theme = st.sidebar.selectbox(
    "Chart Theme",
    ["plotly", "plotly_white", "plotly_dark", "seaborn", "simple_white"]
)

# Generate sample data - Dynamic seed based on current time
random_seed = int(time.time()) % 10000
np.random.seed(random_seed)

days = 90
dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

# KPI Data with more realistic patterns
base_trend = np.linspace(0, 5, days)
kpi_data = pd.DataFrame({
    'Date': dates,
    'System_Uptime': 98 + 2 * np.sin(np.linspace(0, 4*np.pi, days)) + np.random.normal(0, 0.5, days),
    'Response_Time': 250 + 100 * np.sin(np.linspace(0, 3*np.pi, days)) + np.random.normal(0, 30, days),
    'Code_Quality_Score': 80 + base_trend + np.random.normal(0, 3, days),
    'Deployment_Success_Rate': 92 + 5 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 2, days),
    'Bug_Fix_Time': 24 - 10 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 5, days),
    'CPU_Utilization': 60 + 20 * np.sin(np.linspace(0, 6*np.pi, days)) + np.random.normal(0, 5, days),
    'Memory_Usage': 65 + 15 * np.sin(np.linspace(0, 5*np.pi, days)) + np.random.normal(0, 4, days),
    'Security_Score': 85 + base_trend * 1.5 + np.random.normal(0, 2, days),
    'User_Satisfaction': 4.0 + 0.5 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 0.2, days),
    'API_Success_Rate': 97 + 2 * np.sin(np.linspace(0, 3*np.pi, days)) + np.random.normal(0, 1, days),
    'Incident_Count': np.random.poisson(3, days),
    'Active_Users': 1000 + 200 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 50, days),
    'Request_Count': 50000 + 10000 * np.sin(np.linspace(0, 4*np.pi, days)) + np.random.normal(0, 2000, days)
})

# Clip values to realistic ranges
kpi_data['System_Uptime'] = kpi_data['System_Uptime'].clip(95, 100)
kpi_data['Response_Time'] = kpi_data['Response_Time'].clip(100, 500)
kpi_data['Code_Quality_Score'] = kpi_data['Code_Quality_Score'].clip(70, 95)
kpi_data['Deployment_Success_Rate'] = kpi_data['Deployment_Success_Rate'].clip(85, 100)
kpi_data['CPU_Utilization'] = kpi_data['CPU_Utilization'].clip(40, 95)
kpi_data['Memory_Usage'] = kpi_data['Memory_Usage'].clip(50, 90)
kpi_data['Security_Score'] = kpi_data['Security_Score'].clip(75, 98)
kpi_data['User_Satisfaction'] = kpi_data['User_Satisfaction'].clip(3.5, 5.0)
kpi_data['API_Success_Rate'] = kpi_data['API_Success_Rate'].clip(95, 100)
kpi_data['Incident_Count'] = kpi_data['Incident_Count'].clip(0, 10)
kpi_data['Active_Users'] = kpi_data['Active_Users'].clip(800, 1500)

# Apply data filtering if enabled
if enable_data_filtering:
    st.info("🔬 Data filtering is enabled. Outliers are being removed from visualizations.")
    # Remove extreme outliers for better visualization
    for col in ['Response_Time', 'CPU_Utilization', 'Memory_Usage']:
        Q1 = kpi_data[col].quantile(0.25)
        Q3 = kpi_data[col].quantile(0.75)
        IQR = Q3 - Q1
        kpi_data[col] = kpi_data[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# Calculate health score
health_score = calculate_health_score(kpi_data)

# AI Insights Section
if enable_ai_insights:
    insights = generate_ai_insights(kpi_data)
    
    if insights:
        st.header("🧠 AI-Powered Insights & Recommendations")
        
        # Create tabs for different insight types
        insight_tabs = st.tabs(["All Insights", "Critical", "Warnings", "Info", "Success"])
        
        with insight_tabs[0]:
            for insight in insights:
                icon_map = {'critical': '🔴', 'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}
                with st.expander(f"{icon_map.get(insight['type'], '📌')} {insight['title']}", expanded=True):
                    st.markdown(f"**Analysis:** {insight['message']}")
                    st.markdown(f"**Recommendation:** {insight['recommendation']}")
        
        for idx, tab_type in enumerate(['critical', 'warning', 'info', 'success']):
            with insight_tabs[idx + 1]:
                filtered_insights = [i for i in insights if i['type'] == tab_type]
                if filtered_insights:
                    for insight in filtered_insights:
                        icon_map = {'critical': '🔴', 'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}
                        with st.expander(f"{icon_map.get(insight['type'], '📌')} {insight['title']}", expanded=True):
                            st.markdown(f"**Analysis:** {insight['message']}")
                            st.markdown(f"**Recommendation:** {insight['recommendation']}")
                else:
                    st.info(f"No {tab_type} insights at this time.")
        
        st.markdown("---")

# Real-time Alerts Section
if enable_alerts:
    alerts = generate_alerts(kpi_data, st.session_state.custom_thresholds)
    
    # Add to alert history
    for alert in alerts:
        if alert not in st.session_state.alert_history:
            st.session_state.alert_history.append(alert)
    
    if alerts:
        st.header("🚨 Active Real-time Alerts")
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        with col1:
            critical_count = len([a for a in alerts if a['type'] == 'critical'])
            st.metric("Critical Alerts", critical_count, delta=None if critical_count == 0 else f"+{critical_count}")
        with col2:
            warning_count = len([a for a in alerts if a['type'] == 'warning'])
            st.metric("Warning Alerts", warning_count, delta=None if warning_count == 0 else f"+{warning_count}")
        with col3:
            st.metric("Total Active Alerts", len(alerts))
        
        # Display alerts
        cols = st.columns(min(len(alerts), 3))
        for idx, alert in enumerate(alerts[:6]):  # Show max 6 alerts
            with cols[idx % 3]:
                alert_class = f"alert-{alert['type']}"
                icon = "🔴" if alert['type'] == 'critical' else "⚠️"
                st.markdown(f"""
                    <div class="alert-box {alert_class}">
                        <strong>{icon} {alert['type'].upper()}: {alert['metric']}</strong><br>
                        {alert['message']}<br>
                        <small>{alert['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                """, unsafe_allow_html=True)
        
        # Alert history
        with st.expander("📜 Alert History (Last 20)"):
            if st.session_state.alert_history:
                alert_df = pd.DataFrame(st.session_state.alert_history[-20:])
                st.dataframe(alert_df, use_container_width=True)
            else:
                st.info("No alert history available yet.")
        
        st.markdown("---")

# System Health Overview
st.header("🏥 System Health Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #667eea; margin: 0;">Overall Health Score</h3>
            <h1 style="margin: 10px 0;">{health_score:.1f}/100</h1>
            <p style="margin: 0; color: #666;">{'🟢 Excellent' if health_score > 90 else '🟡 Good' if health_score > 80 else '🟠 Fair' if health_score > 70 else '🔴 Poor'}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    incident_count = kpi_data['Incident_Count'].iloc[-7:].sum()
    st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #f093fb; margin: 0;">Incidents (7d)</h3>
            <h1 style="margin: 10px 0;">{int(incident_count)}</h1>
            <p style="margin: 0; color: #666;">📉 {((kpi_data['Incident_Count'].iloc[-14:-7].sum() - incident_count) / max(kpi_data['Incident_Count'].iloc[-14:-7].sum(), 1) * 100):.0f}% vs prev week</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    active_users = int(kpi_data['Active_Users'].iloc[-1])
    st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #00cc96; margin: 0;">Active Users</h3>
            <h1 style="margin: 10px 0;">{active_users:,}</h1>
            <p style="margin: 0; color: #666;">👥 Current active sessions</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    request_count = int(kpi_data['Request_Count'].iloc[-1])
    st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #ffa15a; margin: 0;">Requests Today</h3>
            <h1 style="margin: 10px 0;">{request_count:,}</h1>
            <p style="margin: 0; color: #666;">📊 API requests handled</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Key Metrics Row with Advanced Stats
st.header("🎯 Key Performance Indicators")

# Metric selection
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Primary Metrics")
with col2:
    view_mode = st.selectbox("View Mode", ["Compact", "Detailed"], label_visibility="collapsed")

if view_mode == "Detailed":
    # Detailed view with 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_uptime = kpi_data['System_Uptime'].iloc[-1]
        delta_uptime = kpi_data['System_Uptime'].iloc[-1] - kpi_data['System_Uptime'].iloc[-7]
        sla_compliance = calculate_sla_compliance(kpi_data['System_Uptime'])
        st.metric(
            label="System Uptime",
            value=f"{current_uptime:.2f}%",
            delta=f"{delta_uptime:.2f}%",
            help=f"SLA Compliance: {sla_compliance:.1f}% | Target: {st.session_state.custom_thresholds['System_Uptime']}%"
        )
        # Mini sparkline
        fig_spark = go.Figure(go.Scatter(y=kpi_data['System_Uptime'].tail(30), mode='lines', line=dict(color='#00CC96', width=1)))
        fig_spark.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

    with col2:
        current_response = kpi_data['Response_Time'].iloc[-1]
        delta_response = kpi_data['Response_Time'].iloc[-1] - kpi_data['Response_Time'].iloc[-7]
        avg_response = kpi_data['Response_Time'].tail(7).mean()
        st.metric(
            label="Avg Response Time",
            value=f"{current_response:.0f}ms",
            delta=f"{delta_response:.0f}ms",
            delta_color="inverse",
            help=f"7-day avg: {avg_response:.0f}ms | Target: <{st.session_state.custom_thresholds['Response_Time']}ms"
        )
        fig_spark = go.Figure(go.Scatter(y=kpi_data['Response_Time'].tail(30), mode='lines', line=dict(color='#EF553B', width=1)))
        fig_spark.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

    with col3:
        current_quality = kpi_data['Code_Quality_Score'].iloc[-1]
        delta_quality = kpi_data['Code_Quality_Score'].iloc[-1] - kpi_data['Code_Quality_Score'].iloc[-7]
        trend_quality = "📈" if delta_quality > 0 else "📉"
        st.metric(
            label="Code Quality",
            value=f"{current_quality:.1f}",
            delta=f"{delta_quality:.1f}",
            help=f"Trend: {trend_quality} | Target: >85"
        )
        fig_spark = go.Figure(go.Scatter(y=kpi_data['Code_Quality_Score'].tail(30), mode='lines', line=dict(color='#636EFA', width=1)))
        fig_spark.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

    with col4:
        current_deploy = kpi_data['Deployment_Success_Rate'].iloc[-1]
        delta_deploy = kpi_data['Deployment_Success_Rate'].iloc[-1] - kpi_data['Deployment_Success_Rate'].iloc[-7]
        total_deploys = np.random.randint(50, 100)
        st.metric(
            label="Deployment Success",
            value=f"{current_deploy:.1f}%",
            delta=f"{delta_deploy:.1f}%",
            help=f"Total deploys: {total_deploys} | Target: >95%"
        )
        fig_spark = go.Figure(go.Scatter(y=kpi_data['Deployment_Success_Rate'].tail(30), mode='lines', line=dict(color='#00CC96', width=1)))
        fig_spark.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

    with col5:
        current_security = kpi_data['Security_Score'].iloc[-1]
        delta_security = kpi_data['Security_Score'].iloc[-1] - kpi_data['Security_Score'].iloc[-7]
        vulnerabilities = np.random.randint(0, 5)
        st.metric(
            label="Security Score",
            value=f"{current_security:.1f}",
            delta=f"{delta_security:.1f}",
            help=f"Open vulnerabilities: {vulnerabilities} | Target: >{st.session_state.custom_thresholds['Security_Score']}"
        )
        fig_spark = go.Figure(go.Scatter(y=kpi_data['Security_Score'].tail(30), mode='lines', line=dict(color='#FFA15A', width=1)))
        fig_spark.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

else:
    # Compact view
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("System Uptime", f"{kpi_data['System_Uptime'].iloc[-1]:.2f}%", 
                 f"{kpi_data['System_Uptime'].iloc[-1] - kpi_data['System_Uptime'].iloc[-7]:.2f}%")
    with col2:
        st.metric("Response Time", f"{kpi_data['Response_Time'].iloc[-1]:.0f}ms",
                 f"{kpi_data['Response_Time'].iloc[-1] - kpi_data['Response_Time'].iloc[-7]:.0f}ms", delta_color="inverse")
    with col3:
        st.metric("Code Quality", f"{kpi_data['Code_Quality_Score'].iloc[-1]:.1f}",
                 f"{kpi_data['Code_Quality_Score'].iloc[-1] - kpi_data['Code_Quality_Score'].iloc[-7]:.1f}")
    with col4:
        st.metric("Deployment Success", f"{kpi_data['Deployment_Success_Rate'].iloc[-1]:.1f}%",
                 f"{kpi_data['Deployment_Success_Rate'].iloc[-1] - kpi_data['Deployment_Success_Rate'].iloc[-7]:.1f}%")
    with col5:
        st.metric("Security Score", f"{kpi_data['Security_Score'].iloc[-1]:.1f}",
                 f"{kpi_data['Security_Score'].iloc[-1] - kpi_data['Security_Score'].iloc[-7]:.1f}")

st.markdown("---")

# Performance Trends with Advanced Visualizations
st.header("📈 Performance Trends & Analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖥️ System Performance", 
    "💻 Development Metrics", 
    "📊 Resource Utilization", 
    "🔮 Predictive Analytics",
    "📉 Trend Analysis"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_uptime = go.Figure()
        
        # Historical data
        fig_uptime.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=kpi_data['System_Uptime'],
            mode='lines',
            name='Actual',
            line=dict(color='#00CC96', width=2),
            hovertemplate='%{y:.2f}%<br>%{x}<extra></extra>'
        ))
        
        # Anomaly detection
        if enable_anomaly_detection:
            if enable_ml_anomalies:
                anomalies = detect_anomalies_ml(kpi_data['System_Uptime'])
            else:
                anomalies = detect_anomalies(kpi_data['System_Uptime'])
            
            if anomalies.sum() > 0:
                anomaly_dates = kpi_data['Date'][anomalies]
                anomaly_values = kpi_data['System_Uptime'][anomalies]
                
                fig_uptime.add_trace(go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate='Anomaly: %{y:.2f}%<extra></extra>'
                ))
        
        # Predictions with confidence intervals
        if enable_predictions:
            predictions, confidence = predict_trend(kpi_data['System_Uptime'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_uptime.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color='#00CC96', width=2, dash='dash'),
                hovertemplate='Forecast: %{y:.2f}%<extra></extra>'
            ))
            
            if show_confidence_intervals:
                fig_uptime.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(predictions + confidence) + list(predictions - confidence)[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 204, 150, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        fig_uptime.add_hline(y=st.session_state.custom_thresholds['System_Uptime'], 
                            line_dash="dash", line_color="red", 
                            annotation_text=f"SLA Target: {st.session_state.custom_thresholds['System_Uptime']}%")
        
        fig_uptime.update_layout(
            title='System Uptime Trend & Forecast',
            xaxis_title='Date',
            yaxis_title='Uptime (%)',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_uptime, use_container_width=True)
    
    with col2:
        fig_response = go.Figure()
        
        fig_response.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=kpi_data['Response_Time'],
            mode='lines',
            name='Actual',
            line=dict(color='#EF553B', width=2),
            fill='tonexty',
            fillcolor='rgba(239, 85, 59, 0.1)',
            hovertemplate='%{y:.0f}ms<br>%{x}<extra></extra>'
        ))
        
        if enable_anomaly_detection:
            if enable_ml_anomalies:
                anomalies = detect_anomalies_ml(kpi_data['Response_Time'])
            else:
                anomalies = detect_anomalies(kpi_data['Response_Time'])
            
            if anomalies.sum() > 0:
                anomaly_dates = kpi_data['Date'][anomalies]
                anomaly_values = kpi_data['Response_Time'][anomalies]
                
                fig_response.add_trace(go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate='Anomaly: %{y:.0f}ms<extra></extra>'
                ))
        
        if enable_predictions:
            predictions, confidence = predict_trend(kpi_data['Response_Time'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_response.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color='#EF553B', width=2, dash='dash'),
                hovertemplate='Forecast: %{y:.0f}ms<extra></extra>'
            ))
            
            if show_confidence_intervals:
                fig_response.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(predictions + confidence) + list(predictions - confidence)[::-1],
                    fill='toself',
                    fillcolor='rgba(239, 85, 59, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        fig_response.add_hline(y=st.session_state.custom_thresholds['Response_Time'], 
                              line_dash="dash", line_color="green", 
                              annotation_text=f"Target: <{st.session_state.custom_thresholds['Response_Time']}ms")
        
        fig_response.update_layout(
            title='Response Time Trend & Forecast',
            xaxis_title='Date',
            yaxis_title='Response Time (ms)',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    # API Success Rate
    fig_api = go.Figure()
    fig_api.add_trace(go.Scatter(
        x=kpi_data['Date'],
        y=kpi_data['API_Success_Rate'],
        mode='lines+markers',
        name='API Success Rate',
        line=dict(color='#AB63FA', width=2),
        marker=dict(size=3),
        hovertemplate='%{y:.2f}%<br>%{x}<extra></extra>'
    ))
    
    if enable_predictions:
        predictions, _ = predict_trend(kpi_data['API_Success_Rate'], prediction_days)
        future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                    periods=prediction_days, freq='D')
        fig_api.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Forecast',
            line=dict(color='#AB63FA', width=2, dash='dash')
        ))
    
    fig_api.add_hline(y=99, line_dash="dash", line_color="orange", annotation_text="Target: >99%")
    fig_api.update_layout(
        title='API Success Rate',
        xaxis_title='Date',
        yaxis_title='Success Rate (%)',
        hovermode='x unified',
        template=chart_theme,
        height=400
    )
    st.plotly_chart(fig_api, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Code Quality with Moving Average
        fig_quality = go.Figure()
        
        fig_quality.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=kpi_data['Code_Quality_Score'],
            mode='lines',
            name='Daily Score',
            line=dict(color='#636EFA', width=1),
            opacity=0.5,
            hovertemplate='Daily: %{y:.1f}<extra></extra>'
        ))
        
        # Multiple moving averages
        ma_7 = kpi_data['Code_Quality_Score'].rolling(window=7).mean()
        ma_30 = kpi_data['Code_Quality_Score'].rolling(window=30).mean()
        
        fig_quality.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=ma_7,
            mode='lines',
            name='7-Day MA',
            line=dict(color='#636EFA', width=2),
            hovertemplate='7-Day MA: %{y:.1f}<extra></extra>'
        ))
        
        fig_quality.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=ma_30,
            mode='lines',
            name='30-Day MA',
            line=dict(color='#00CC96', width=2),
            hovertemplate='30-Day MA: %{y:.1f}<extra></extra>'
        ))
        
        if enable_predictions:
            predictions, _ = predict_trend(kpi_data['Code_Quality_Score'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_quality.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color='#636EFA', width=2, dash='dash')
            ))
        
        fig_quality.update_layout(
            title='Code Quality Score with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Quality Score',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Deployment Success Rate
        fig_deployment = go.Figure()
        
        fig_deployment.add_trace(go.Bar(
            x=kpi_data['Date'].tail(30),
            y=kpi_data['Deployment_Success_Rate'].tail(30),
            name='Success Rate',
            marker_color=kpi_data['Deployment_Success_Rate'].tail(30).apply(
                lambda x: '#00CC96' if x >= 95 else '#FFA15A' if x >= 90 else '#EF553B'
            ),
            hovertemplate='%{y:.1f}%<br>%{x}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(range(30), kpi_data['Deployment_Success_Rate'].tail(30), 1)
        p = np.poly1d(z)
        
        fig_deployment.add_trace(go.Scatter(
            x=kpi_data['Date'].tail(30),
            y=p(range(30)),
            mode='lines',
            name='Trend',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        if enable_predictions:
            predictions, _ = predict_trend(kpi_data['Deployment_Success_Rate'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_deployment.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#00CC96', width=2, dash='dash'),
                marker=dict(size=5)
            ))
        
        fig_deployment.add_hline(y=95, line_dash="dash", line_color="orange", 
                                annotation_text="Target: >95%")
        
        fig_deployment.update_layout(
            title='Deployment Success Rate (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Success Rate (%)',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_deployment, use_container_width=True)
    
    # Bug Fix Time
    fig_bugfix = go.Figure()
    fig_bugfix.add_trace(go.Box(
        y=kpi_data['Bug_Fix_Time'],
        name='Bug Fix Time',
        marker_color='#FFA15A',
        boxmean='sd'
    ))
    fig_bugfix.update_layout(
        title='Bug Fix Time Distribution',
        yaxis_title='Time (hours)',
        template=chart_theme,
        height=400
    )
    st.plotly_chart(fig_bugfix, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Utilization with threshold zones
        fig_cpu = go.Figure()
        
        fig_cpu.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=kpi_data['CPU_Utilization'],
            mode='lines',
            name='CPU Usage',
            line=dict(color='#FFA15A', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 161, 90, 0.2)',
            hovertemplate='%{y:.1f}%<br>%{x}<extra></extra>'
        ))
        
        if enable_predictions:
            predictions, confidence = predict_trend(kpi_data['CPU_Utilization'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_cpu.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color='#FFA15A', width=2, dash='dash')
            ))
            
            if show_confidence_intervals:
                fig_cpu.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(predictions + confidence) + list(predictions - confidence)[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 161, 90, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        fig_cpu.add_hrect(y0=st.session_state.custom_thresholds['CPU_Utilization'], y1=100, 
                         fillcolor="red", opacity=0.1, 
                         annotation_text="Critical Zone", annotation_position="top left")
        fig_cpu.add_hrect(y0=60, y1=st.session_state.custom_thresholds['CPU_Utilization'], 
                         fillcolor="yellow", opacity=0.1, 
                         annotation_text="Warning Zone", annotation_position="top left")
        
        fig_cpu.update_layout(
            title='CPU Utilization with Forecast',
            xaxis_title='Date',
            yaxis_title='CPU Usage (%)',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Memory Usage
        fig_memory = go.Figure()
        
        fig_memory.add_trace(go.Scatter(
            x=kpi_data['Date'],
            y=kpi_data['Memory_Usage'],
            mode='lines',
            name='Memory Usage',
            line=dict(color='#AB63FA', width=2),
            fill='tozeroy',
            fillcolor='rgba(171, 99, 250, 0.2)',
            hovertemplate='%{y:.1f}%<br>%{x}<extra></extra>'
        ))
        
        if enable_predictions:
            predictions, confidence = predict_trend(kpi_data['Memory_Usage'], prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            fig_memory.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color='#AB63FA', width=2, dash='dash')
            ))
            
            if show_confidence_intervals:
                fig_memory.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(predictions + confidence) + list(predictions - confidence)[::-1],
                    fill='toself',
                    fillcolor='rgba(171, 99, 250, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        fig_memory.add_hrect(y0=st.session_state.custom_thresholds['Memory_Usage'], y1=100, 
                            fillcolor="red", opacity=0.1, 
                            annotation_text="Critical Zone", annotation_position="top left")
        fig_memory.add_hrect(y0=70, y1=st.session_state.custom_thresholds['Memory_Usage'], 
                            fillcolor="yellow", opacity=0.1, 
                            annotation_text="Warning Zone", annotation_position="top left")
        
        fig_memory.update_layout(
            title='Memory Usage with Forecast',
            xaxis_title='Date',
            yaxis_title='Memory (%)',
            hovermode='x unified',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_memory, use_container_width=True)
    
    # Combined Resource Usage
    fig_combined = go.Figure()
    
    fig_combined.add_trace(go.Scatter(
        x=kpi_data['Date'],
        y=kpi_data['CPU_Utilization'],
        mode='lines',
        name='CPU',
        line=dict(color='#FFA15A', width=2)
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=kpi_data['Date'],
        y=kpi_data['Memory_Usage'],
        mode='lines',
        name='Memory',
        line=dict(color='#AB63FA', width=2)
    ))
    
    fig_combined.update_layout(
        title='Combined Resource Usage',
        xaxis_title='Date',
        yaxis_title='Usage (%)',
        hovermode='x unified',
        template=chart_theme,
        height=400
    )
    st.plotly_chart(fig_combined, use_container_width=True)

with tab4:
    st.subheader("🔮 Predictive Analytics Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-metric forecast comparison
        fig_forecast = go.Figure()
        
        metrics_to_forecast = {
            'System Uptime': kpi_data['System_Uptime'],
            'Code Quality': kpi_data['Code_Quality_Score'],
            'Security Score': kpi_data['Security_Score'],
            'API Success': kpi_data['API_Success_Rate']
        }
        
        colors = ['#00CC96', '#636EFA', '#FFA15A', '#AB63FA']
        
        for idx, (metric_name, metric_data) in enumerate(metrics_to_forecast.items()):
            # Normalize to 0-100 scale for comparison
            normalized = (metric_data - metric_data.min()) / (metric_data.max() - metric_data.min()) * 100
            
            predictions, _ = predict_trend(pd.Series(normalized), prediction_days)
            future_dates = pd.date_range(start=kpi_data['Date'].iloc[-1] + timedelta(days=1), 
                                        periods=prediction_days, freq='D')
            
            # Historical (last 30 days)
            fig_forecast.add_trace(go.Scatter(
                x=kpi_data['Date'].tail(30),
                y=normalized.tail(30),
                mode='lines',
                name=f'{metric_name} (Historical)',
                line=dict(color=colors[idx], width=2),
                legendgroup=metric_name
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name=f'{metric_name} (Forecast)',
                line=dict(color=colors[idx], width=2, dash='dash'),
                legendgroup=metric_name
            ))
        
        fig_forecast.update_layout(
            title=f'Multi-Metric Forecast Comparison ({prediction_days} Days Ahead)',
            xaxis_title='Date',
            yaxis_title='Normalized Score (0-100)',
            hovermode='x unified',
            template=chart_theme,
            height=500
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Forecast Summary")
        st.markdown(f"**Prediction Period:** {prediction_days} days")
        st.markdown("---")
        
        # Calculate trend directions
        for metric_name, metric_data in metrics_to_forecast.items():
            predictions, _ = predict_trend(metric_data, prediction_days)
            current_value = metric_data.iloc[-1]
            predicted_value = predictions[-1]
            trend = "↗️ Improving" if predicted_value > current_value else "↘️ Declining"
            change = ((predicted_value - current_value) / current_value * 100)
            
            color = "green" if change > 0 else "red"
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>{metric_name}</strong><br>
                    {trend}<br>
                    <span style="color: {color}; font-size: 1.2em;">{change:+.1f}%</span><br>
                    <small>Current: {current_value:.1f} → Predicted: {predicted_value:.1f}</small>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 Predictions are based on linear regression analysis of historical trends.")

with tab5:
    st.subheader("📉 Comprehensive Trend Analysis")
    
    # Trend comparison table
    trend_data = []
    for col in ['System_Uptime', 'Response_Time', 'Code_Quality_Score', 'Deployment_Success_Rate', 
                'Security_Score', 'CPU_Utilization', 'Memory_Usage']:
        current = kpi_data[col].iloc[-1]
        week_ago = kpi_data[col].iloc[-7]
        month_ago = kpi_data[col].iloc[-30]
        
        week_change = ((current - week_ago) / week_ago * 100) if week_ago != 0 else 0
        month_change = ((current - month_ago) / month_ago * 100) if month_ago != 0 else 0
        
        trend_icon = "📈" if week_change > 0 else "📉"
        
        trend_data.append({
            'Metric': col.replace('_', ' '),
            'Current': f"{current:.2f}",
            '7d Change': f"{week_change:+.1f}%",
            '30d Change': f"{month_change:+.1f}%",
            'Trend': trend_icon,
            'Status': '✅ Good' if abs(week_change) < 5 else '⚠️ Monitor'
        })
    
    trend_df = pd.DataFrame(trend_data)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)
    
    # Trend heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily change heatmap
        daily_changes = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                                   'Deployment_Success_Rate', 'Security_Score']].pct_change() * 100
        
        fig_heatmap = px.imshow(
            daily_changes.tail(30).T,
            labels=dict(x="Day", y="Metric", color="% Change"),
            title='Daily Percentage Change (Last 30 Days)',
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig_heatmap.update_layout(template=chart_theme, height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Volatility analysis
        volatility_data = []
        for col in ['System_Uptime', 'Response_Time', 'Code_Quality_Score', 'Security_Score']:
            volatility = kpi_data[col].std()
            volatility_data.append({
                'Metric': col.replace('_', ' '),
                'Volatility': volatility
            })
        
        vol_df = pd.DataFrame(volatility_data)
        fig_vol = px.bar(
            vol_df,
            x='Metric',
            y='Volatility',
            title='Metric Volatility (Standard Deviation)',
            color='Volatility',
            color_continuous_scale='Reds'
        )
        fig_vol.update_layout(template=chart_theme, height=400, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# Advanced Analytics Section
st.header("🔍 Advanced KPI Analytics & Insights")

tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Distribution Analysis", "Performance Score"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced correlation matrix
        correlation_cols = ['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                           'Deployment_Success_Rate', 'Security_Score', 'CPU_Utilization',
                           'Memory_Usage', 'API_Success_Rate']
        correlation_data = kpi_data[correlation_cols].corr()
        
        fig_heatmap = px.imshow(
            correlation_data,
            text_auto='.2f',
            title='Comprehensive KPI Correlation Matrix',
            color_continuous_scale='RdYlGn',
            aspect='auto',
            zmin=-1, zmax=1,
            labels=dict(color="Correlation")
        )
        fig_heatmap.update_layout(template=chart_theme, height=600)
        fig_heatmap.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.markdown("### 🔗 Key Correlations")
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_data.columns)):
            for j in range(i+1, len(correlation_data.columns)):
                correlations.append({
                    'Pair': f"{correlation_data.columns[i]} ↔ {correlation_data.columns[j]}",
                    'Correlation': correlation_data.iloc[i, j]
                })
        
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.reindex(correlations_df['Correlation'].abs().sort_values(ascending=False).index)
        
        st.markdown("**Strongest Positive:**")
        for idx, row in correlations_df.head(3).iterrows():
            if row['Correlation'] > 0:
                st.success(f"{row['Pair'].split('↔')[0].strip()[:15]}... ↔ {row['Pair'].split('↔')[1].strip()[:15]}...\n\n**{row['Correlation']:.3f}**")
        
        st.markdown("**Strongest Negative:**")
        for idx, row in correlations_df.tail(3).iterrows():
            if row['Correlation'] < 0:
                st.error(f"{row['Pair'].split('↔')[0].strip()[:15]}... ↔ {row['Pair'].split('↔')[1].strip()[:15]}...\n\n**{row['Correlation']:.3f}**")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution comparison
        selected_metric = st.selectbox(
            "Select Metric for Distribution Analysis",
            ['Response_Time', 'CPU_Utilization', 'Memory_Usage', 'Code_Quality_Score']
        )
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=kpi_data[selected_metric],
            name='Distribution',
            nbinsx=30,
            marker_color='#636EFA',
            opacity=0.7
        ))
        
        fig_dist.add_vline(x=kpi_data[selected_metric].mean(), 
                          line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {kpi_data[selected_metric].mean():.1f}")
        
        fig_dist.add_vline(x=kpi_data[selected_metric].median(), 
                          line_dash="dash", line_color="green", 
                          annotation_text=f"Median: {kpi_data[selected_metric].median():.1f}")
        
        fig_dist.update_layout(
            title=f'{selected_metric.replace("_", " ")} Distribution',
            xaxis_title='Value',
            yaxis_title='Frequency',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Box plot comparison
        fig_box = go.Figure()
        
        metrics_for_box = ['Response_Time', 'CPU_Utilization', 'Memory_Usage']
        for metric in metrics_for_box:
            fig_box.add_trace(go.Box(
                y=kpi_data[metric],
                name=metric.replace('_', ' '),
                boxmean='sd'
            ))
        
        fig_box.update_layout(
            title='Multi-Metric Distribution Comparison',
            yaxis_title='Value',
            template=chart_theme,
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    stats_df = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                         'Deployment_Success_Rate', 'Security_Score']].describe().T
    stats_df['coefficient_of_variation'] = (stats_df['std'] / stats_df['mean']) * 100
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Overall performance gauge with custom styling
        overall_score = (
            kpi_data['System_Uptime'].mean() * 0.25 +
            kpi_data['Code_Quality_Score'].mean() * 0.20 +
            kpi_data['Deployment_Success_Rate'].mean() * 0.20 +
            kpi_data['Security_Score'].mean() * 0.25 +
            kpi_data['API_Success_Rate'].mean() * 0.10
        )
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Technology Performance Score", 'font': {'size': 20}},
            delta={'reference': 90, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue", 'thickness': 0.75},
                'steps': [
                    {'range': [0, 60], 'color': "#ffcdd2"},
                    {'range': [60, 70], 'color': "#ffecb3"},
                    {'range': [70, 80], 'color': "#fff9c4"},
                    {'range': [80, 90], 'color': "#dcedc8"},
                    {'range': [90, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_gauge.update_layout(template=chart_theme, height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Performance breakdown
        st.markdown("### 📈 Performance Breakdown")
        
        breakdown = {
            'System Uptime (25%)': kpi_data['System_Uptime'].mean() * 0.25,
            'Security Score (25%)': kpi_data['Security_Score'].mean() * 0.25,
            'Code Quality (20%)': kpi_data['Code_Quality_Score'].mean() * 0.20,
            'Deployment Success (20%)': kpi_data['Deployment_Success_Rate'].mean() * 0.20,
            'API Success (10%)': kpi_data['API_Success_Rate'].mean() * 0.10
        }
        
        fig_breakdown = go.Figure(go.Bar(
            x=list(breakdown.values()),
            y=list(breakdown.keys()),
            orientation='h',
            marker_color=['#00CC96', '#FFA15A', '#636EFA', '#AB63FA', '#EF553B'],
            text=[f"{v:.1f}" for v in breakdown.values()],
            textposition='auto'
        ))
        
        fig_breakdown.update_layout(
            title='Weighted Score Contribution',
            xaxis_title='Contribution to Overall Score',
            template=chart_theme,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

st.markdown("---")

# Comparative Analysis
st.header("📊 Comparative Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    # Week-over-week comparison with more detail
    current_week = kpi_data.tail(7)
    previous_week = kpi_data.iloc[-14:-7]
    two_weeks_ago = kpi_data.iloc[-21:-14] if len(kpi_data) >= 21 else None
    
    comparison_metrics = ['System_Uptime', 'Code_Quality_Score', 'Deployment_Success_Rate', 
                         'Security_Score', 'API_Success_Rate']
    
    current_avg = [current_week[m].mean() for m in comparison_metrics]
    previous_avg = [previous_week[m].mean() for m in comparison_metrics]
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Previous Week',
        x=[m.replace('_', ' ') for m in comparison_metrics],
        y=previous_avg,
        marker_color='#636EFA',
        text=[f"{v:.1f}" for v in previous_avg],
        textposition='auto'
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='Current Week',
        x=[m.replace('_', ' ') for m in comparison_metrics],
        y=current_avg,
        marker_color='#00CC96',
        text=[f"{v:.1f}" for v in current_avg],
        textposition='auto'
    ))
    
    if two_weeks_ago is not None:
        two_weeks_avg = [two_weeks_ago[m].mean() for m in comparison_metrics]
        fig_comparison.add_trace(go.Bar(
            name='Two Weeks Ago',
            x=[m.replace('_', ' ') for m in comparison_metrics],
            y=two_weeks_avg,
            marker_color='#FFA15A',
            text=[f"{v:.1f}" for v in two_weeks_avg],
            textposition='auto'
        ))
    
    fig_comparison.update_layout(
        title='Multi-Week Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Average Score',
        barmode='group',
        hovermode='x unified',
        template=chart_theme
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

with col2:
    # Performance trends radar chart with multiple time periods
    categories = ['Uptime', 'Quality', 'Deployment', 'Security', 'API Success']
    
    fig_radar = go.Figure()
    
    # Current performance
    fig_radar.add_trace(go.Scatterpolar(
        r=[
            kpi_data['System_Uptime'].mean(),
            kpi_data['Code_Quality_Score'].mean(),
            kpi_data['Deployment_Success_Rate'].mean(),
            kpi_data['Security_Score'].mean(),
            kpi_data['API_Success_Rate'].mean()
        ],
        theta=categories,
        fill='toself',
        name='Current (90d avg)',
        line=dict(color='#00CC96', width=2)
    ))
    
    # Last 30 days
    fig_radar.add_trace(go.Scatterpolar(
        r=[
            kpi_data['System_Uptime'].tail(30).mean(),
            kpi_data['Code_Quality_Score'].tail(30).mean(),
            kpi_data['Deployment_Success_Rate'].tail(30).mean(),
            kpi_data['Security_Score'].tail(30).mean(),
            kpi_data['API_Success_Rate'].tail(30).mean()
        ],
        theta=categories,
        fill='toself',
        name='Last 30 days',
        line=dict(color='#636EFA', width=2)
    ))
    
    # Target
    fig_radar.add_trace(go.Scatterpolar(
        r=[99.5, 90, 95, 95, 99],
        theta=categories,
        fill='toself',
        name='Target',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title='Performance vs Target (Radar View)',
        template=chart_theme
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# Incident Analysis
st.header("🚨 Incident & Event Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Incident trend
    fig_incidents = go.Figure()
    fig_incidents.add_trace(go.Bar(
        x=kpi_data['Date'].tail(30),
        y=kpi_data['Incident_Count'].tail(30),
        name='Incident Count',
        marker_color=kpi_data['Incident_Count'].tail(30).apply(
            lambda x: '#00CC96' if x <= 2 else '#FFA15A' if x <= 4 else '#EF553B'
        )
    ))
    fig_incidents.update_layout(
        title='Daily Incident Count (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Count',
        template=chart_theme,
        height=300
    )
    st.plotly_chart(fig_incidents, use_container_width=True)

with col2:
    # User activity
    fig_users = go.Figure()
    fig_users.add_trace(go.Scatter(
        x=kpi_data['Date'].tail(30),
        y=kpi_data['Active_Users'].tail(30),
        mode='lines+markers',
        name='Active Users',
        line=dict(color='#AB63FA', width=2),
        marker=dict(size=4)
    ))
    fig_users.update_layout(
        title='Active Users Trend (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Users',
        template=chart_theme,
        height=300
    )
    st.plotly_chart(fig_users, use_container_width=True)

with col3:
    # Request volume
    fig_requests = go.Figure()
    fig_requests.add_trace(go.Scatter(
        x=kpi_data['Date'].tail(30),
        y=kpi_data['Request_Count'].tail(30),
        mode='lines',
        name='Requests',
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.2)'
    ))
    fig_requests.update_layout(
        title='API Request Volume (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Requests',
        template=chart_theme,
        height=300
    )
    st.plotly_chart(fig_requests, use_container_width=True)

st.markdown("---")

# Detailed KPI Summary Table with Enhanced Features
st.header("📋 Detailed KPI Summary & Statistics")

# Tab for different views
summary_tab1, summary_tab2, summary_tab3 = st.tabs(["📊 Summary View", "📈 Detailed Stats", "🎯 Target vs Actual"])

with summary_tab1:
    summary_data = []
    for col in ['System_Uptime', 'Response_Time', 'Code_Quality_Score', 'Deployment_Success_Rate', 
                'Bug_Fix_Time', 'Security_Score', 'API_Success_Rate', 'User_Satisfaction']:
        current_val = kpi_data[col].iloc[-1]
        week_ago = kpi_data[col].iloc[-7]
        change = ((current_val - week_ago) / week_ago * 100) if week_ago != 0 else 0
        
        summary_data.append({
            'KPI': col.replace('_', ' '),
            'Current': f"{current_val:.2f}",
            '7d Change': f"{change:+.1f}%",
            'Mean': f"{kpi_data[col].mean():.2f}",
            'Min': f"{kpi_data[col].min():.2f}",
            'Max': f"{kpi_data[col].max():.2f}",
            'Trend': "📈" if change > 0 else "📉" if change < 0 else "➡️",
            'Status': '✅' if abs(change) < 10 else '⚠️'
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Trend": st.column_config.TextColumn("Trend", width="small"),
        }
    )

with summary_tab2:
    # Comprehensive statistics
    detailed_stats = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                               'Deployment_Success_Rate', 'Security_Score', 'CPU_Utilization',
                               'Memory_Usage', 'API_Success_Rate']].describe()
    
    # Add additional statistics
    detailed_stats.loc['variance'] = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                                                'Deployment_Success_Rate', 'Security_Score', 'CPU_Utilization',
                                                'Memory_Usage', 'API_Success_Rate']].var()
    detailed_stats.loc['cv%'] = (detailed_stats.loc['std'] / detailed_stats.loc['mean']) * 100
    
    st.dataframe(detailed_stats.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=1), 
                use_container_width=True)

with summary_tab3:
    # Target vs Actual comparison
    targets = {
        'System Uptime': 99.5,
        'Response Time': 200,  # Lower is better
        'Code Quality Score': 85,
        'Deployment Success Rate': 95,
        'Security Score': 90,
        'API Success Rate': 99
    }
    
    actual_values = {
        'System Uptime': kpi_data['System_Uptime'].mean(),
        'Response Time': kpi_data['Response_Time'].mean(),
        'Code Quality Score': kpi_data['Code_Quality_Score'].mean(),
        'Deployment Success Rate': kpi_data['Deployment_Success_Rate'].mean(),
        'Security Score': kpi_data['Security_Score'].mean(),
        'API Success Rate': kpi_data['API_Success_Rate'].mean()
    }
    
    target_comparison = []
    for metric in targets.keys():
        target = targets[metric]
        actual = actual_values[metric]
        
        # For response time, lower is better
        if metric == 'Response Time':
            gap = target - actual
            status = '✅ Meeting Target' if actual <= target else '❌ Above Target'
            gap_pct = (gap / target * 100)
        else:
            gap = actual - target
            status = '✅ Meeting Target' if actual >= target else '❌ Below Target'
            gap_pct = (gap / target * 100)
        
        target_comparison.append({
            'Metric': metric,
            'Target': f"{target:.1f}",
            'Actual': f"{actual:.1f}",
            'Gap': f"{gap:+.1f}",
            'Gap %': f"{gap_pct:+.1f}%",
            'Status': status
        })
    
    target_df = pd.DataFrame(target_comparison)
    st.dataframe(target_df, use_container_width=True, hide_index=True)

# Export functionality with more options
st.markdown("---")
st.header("📥 Export & Download Options")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Export to Excel
    alerts_df = pd.DataFrame(st.session_state.alert_history) if st.session_state.alert_history else None
    excel_data = export_to_excel(kpi_data, summary_df, alerts_df)
    st.download_button(
        label="📥 Export to Excel",
        data=excel_data,
        file_name=f"kpi_dashboard_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col2:
    # Export to CSV
    csv_data = kpi_data.to_csv(index=False)
    st.download_button(
        label="📥 Export to CSV",
        data=csv_data,
        file_name=f"kpi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col3:
    # Export to JSON
    json_data = export_to_json(kpi_data)
    st.download_button(
        label="📥 Export to JSON",
        data=json_data,
        file_name=f"kpi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

with col4:
    # Export summary report
    summary_text = f"""
    KPI Dashboard Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Overall Health Score: {health_score:.1f}/100
    
    Key Metrics:
    - System Uptime: {kpi_data['System_Uptime'].iloc[-1]:.2f}%
    - Response Time: {kpi_data['Response_Time'].iloc[-1]:.0f}ms
    - Code Quality: {kpi_data['Code_Quality_Score'].iloc[-1]:.1f}
    - Security Score: {kpi_data['Security_Score'].iloc[-1]:.1f}
    
    Active Alerts: {len(alerts) if enable_alerts else 0}
    """
    
    st.download_button(
        label="📥 Export Summary",
        data=summary_text,
        file_name=f"kpi_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# Notes section
st.markdown("---")
with st.expander("📝 Dashboard Notes & Annotations"):
    note_text = st.text_area("Add notes about current metrics or observations:", height=100)
    if st.button("Save Note"):
        if note_text:
            st.session_state.notes.append({
                'timestamp': datetime.now(),
                'note': note_text
            })
            st.success("Note saved!")
    
    if st.session_state.notes:
        st.markdown("### Recent Notes:")
        for note in reversed(st.session_state.notes[-5:]):
            st.info(f"**{note['timestamp'].strftime('%Y-%m-%d %H:%M')}**: {note['note']}")

# Footer with enhanced information
st.markdown("---")
refresh_status = "🟢 Auto-refresh enabled" if auto_refresh else "⚪ Manual refresh mode"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px;'>
            <strong>Last Updated</strong><br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <small>{refresh_status}</small>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px;'>
            <strong>Data Points</strong><br>
            {len(kpi_data)} records<br>
            <small>Random Seed: {random_seed}</small>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px;'>
            <strong>Forecast Period</strong><br>
            {prediction_days} days ahead<br>
            <small>95% Confidence Interval</small>
        </div>
    """, unsafe_allow_html=True)

anomaly_method = 'ML-Based (Isolation Forest)' if enable_ml_anomalies else 'Statistical (Z-Score)'
st.markdown(f"""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        <p><em>💡 Tip: Click 'Refresh Now' button or enable auto-refresh to see live data updates</em></p>
        <p><small>Anomaly Detection: {anomaly_method} | 
        Chart Theme: {chart_theme.replace('_', ' ').title()}</small></p>
    </div>
""", unsafe_allow_html=True)