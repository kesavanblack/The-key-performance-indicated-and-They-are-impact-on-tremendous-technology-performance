import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(page_title="Technology KPI Dashboard", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Technology Performance KPI Dashboard")
st.markdown("---")

# Sidebar for filters
st.sidebar.header("Dashboard Controls")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"]
)

department = st.sidebar.multiselect(
    "Select Department",
    ["Development", "Operations", "Security", "Data Science", "Infrastructure"],
    default=["Development", "Operations"]
)

# Generate sample data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

# KPI Data
kpi_data = pd.DataFrame({
    'Date': dates,
    'System_Uptime': np.random.uniform(95, 100, 90),
    'Response_Time': np.random.uniform(100, 500, 90),
    'Code_Quality_Score': np.random.uniform(70, 95, 90),
    'Deployment_Success_Rate': np.random.uniform(85, 100, 90),
    'Bug_Fix_Time': np.random.uniform(2, 48, 90),
    'CPU_Utilization': np.random.uniform(40, 90, 90),
    'Memory_Usage': np.random.uniform(50, 85, 90),
    'Security_Score': np.random.uniform(75, 98, 90),
    'User_Satisfaction': np.random.uniform(3.5, 5.0, 90),
    'API_Success_Rate': np.random.uniform(95, 100, 90)
})

# Key Metrics Row
st.header("🎯 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_uptime = kpi_data['System_Uptime'].iloc[-1]
    delta_uptime = kpi_data['System_Uptime'].iloc[-1] - kpi_data['System_Uptime'].iloc[-7]
    st.metric(
        label="System Uptime",
        value=f"{current_uptime:.2f}%",
        delta=f"{delta_uptime:.2f}%"
    )

with col2:
    current_response = kpi_data['Response_Time'].iloc[-1]
    delta_response = kpi_data['Response_Time'].iloc[-1] - kpi_data['Response_Time'].iloc[-7]
    st.metric(
        label="Avg Response Time",
        value=f"{current_response:.0f}ms",
        delta=f"{delta_response:.0f}ms",
        delta_color="inverse"
    )

with col3:
    current_quality = kpi_data['Code_Quality_Score'].iloc[-1]
    delta_quality = kpi_data['Code_Quality_Score'].iloc[-1] - kpi_data['Code_Quality_Score'].iloc[-7]
    st.metric(
        label="Code Quality",
        value=f"{current_quality:.1f}",
        delta=f"{delta_quality:.1f}"
    )

with col4:
    current_deploy = kpi_data['Deployment_Success_Rate'].iloc[-1]
    delta_deploy = kpi_data['Deployment_Success_Rate'].iloc[-1] - kpi_data['Deployment_Success_Rate'].iloc[-7]
    st.metric(
        label="Deployment Success",
        value=f"{current_deploy:.1f}%",
        delta=f"{delta_deploy:.1f}%"
    )

with col5:
    current_security = kpi_data['Security_Score'].iloc[-1]
    delta_security = kpi_data['Security_Score'].iloc[-1] - kpi_data['Security_Score'].iloc[-7]
    st.metric(
        label="Security Score",
        value=f"{current_security:.1f}",
        delta=f"{delta_security:.1f}"
    )

st.markdown("---")

# Performance Trends
st.header("📈 Performance Trends Over Time")

tab1, tab2, tab3 = st.tabs(["System Performance", "Development Metrics", "Resource Utilization"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_uptime = px.line(
            kpi_data, 
            x='Date', 
            y='System_Uptime',
            title='System Uptime Trend',
            labels={'System_Uptime': 'Uptime (%)'}
        )
        fig_uptime.update_traces(line_color='#00CC96')
        fig_uptime.add_hline(y=99.5, line_dash="dash", line_color="red", 
                            annotation_text="Target: 99.5%")
        st.plotly_chart(fig_uptime, use_container_width=True)
    
    with col2:
        fig_response = px.line(
            kpi_data, 
            x='Date', 
            y='Response_Time',
            title='Response Time Trend',
            labels={'Response_Time': 'Response Time (ms)'}
        )
        fig_response.update_traces(line_color='#EF553B')
        fig_response.add_hline(y=200, line_dash="dash", line_color="green", 
                              annotation_text="Target: <200ms")
        st.plotly_chart(fig_response, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = px.area(
            kpi_data, 
            x='Date', 
            y='Code_Quality_Score',
            title='Code Quality Score',
            labels={'Code_Quality_Score': 'Quality Score'}
        )
        fig_quality.update_traces(fillcolor='rgba(99, 110, 250, 0.3)', line_color='#636EFA')
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        fig_deployment = px.line(
            kpi_data, 
            x='Date', 
            y='Deployment_Success_Rate',
            title='Deployment Success Rate',
            labels={'Deployment_Success_Rate': 'Success Rate (%)'}
        )
        fig_deployment.update_traces(line_color='#00CC96')
        st.plotly_chart(fig_deployment, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cpu = px.line(
            kpi_data, 
            x='Date', 
            y='CPU_Utilization',
            title='CPU Utilization',
            labels={'CPU_Utilization': 'CPU Usage (%)'}
        )
        fig_cpu.update_traces(line_color='#FFA15A')
        fig_cpu.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="Warning Threshold: 80%")
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        fig_memory = px.line(
            kpi_data, 
            x='Date', 
            y='Memory_Usage',
            title='Memory Usage',
            labels={'Memory_Usage': 'Memory (%)'}
        )
        fig_memory.update_traces(line_color='#AB63FA')
        fig_memory.add_hline(y=85, line_dash="dash", line_color="red", 
                            annotation_text="Critical: 85%")
        st.plotly_chart(fig_memory, use_container_width=True)

st.markdown("---")

# KPI Impact Analysis
st.header("🔍 KPI Impact Analysis")

col1, col2 = st.columns(2)

with col1:
    # Correlation matrix
    correlation_data = kpi_data[['System_Uptime', 'Response_Time', 'Code_Quality_Score', 
                                  'Deployment_Success_Rate', 'Security_Score']].corr()
    
    fig_heatmap = px.imshow(
        correlation_data,
        text_auto='.2f',
        title='KPI Correlation Matrix',
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Gauge chart for overall performance
    overall_score = (
        kpi_data['System_Uptime'].mean() * 0.3 +
        kpi_data['Code_Quality_Score'].mean() * 0.2 +
        kpi_data['Deployment_Success_Rate'].mean() * 0.2 +
        kpi_data['Security_Score'].mean() * 0.3
    )
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Technology Performance Score"},
        delta={'reference': 90},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Performance Summary Table
st.header("📋 Detailed KPI Summary")

summary_df = pd.DataFrame({
    'KPI': ['System Uptime', 'Response Time', 'Code Quality', 'Deployment Success', 
            'Bug Fix Time', 'Security Score', 'API Success Rate', 'User Satisfaction'],
    'Current': [
        f"{kpi_data['System_Uptime'].iloc[-1]:.2f}%",
        f"{kpi_data['Response_Time'].iloc[-1]:.0f}ms",
        f"{kpi_data['Code_Quality_Score'].iloc[-1]:.1f}",
        f"{kpi_data['Deployment_Success_Rate'].iloc[-1]:.1f}%",
        f"{kpi_data['Bug_Fix_Time'].iloc[-1]:.1f}h",
        f"{kpi_data['Security_Score'].iloc[-1]:.1f}",
        f"{kpi_data['API_Success_Rate'].iloc[-1]:.1f}%",
        f"{kpi_data['User_Satisfaction'].iloc[-1]:.2f}/5.0"
    ],
    'Target': ['99.5%', '<200ms', '>85', '>95%', '<24h', '>90', '>99%', '>4.5/5'],
    'Status': ['✅', '⚠️', '✅', '✅', '✅', '⚠️', '✅', '✅']
})

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Dashboard last updated: {}</p>
        <p>Data refreshes every 5 minutes</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)