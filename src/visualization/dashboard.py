#!/usr/bin/env python3
"""
AccidentIQ Interactive Dashboard
Modern dark theme with glassmorphism design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import h3
from sqlalchemy import create_engine
import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AccidentIQ - Road Safety Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern dark theme CSS with glassmorphism
st.markdown("""
<style>
    /* Import SF Pro Display font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global variables */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-glass: rgba(25, 25, 25, 0.8);
        --bg-card: rgba(35, 35, 35, 0.6);
        --accent-blue: #007AFF;
        --accent-purple: #5856D6;
        --accent-green: #34C759;
        --accent-orange: #FF9500;
        --accent-red: #FF3B30;
        --text-primary: #FFFFFF;
        --text-secondary: #A0A0A0;
        --border-color: rgba(255, 255, 255, 0.1);
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --blur: blur(20px);
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Glass cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(45, 45, 45, 0.7);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 122, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background: rgba(52, 199, 89, 0.2);
        color: var(--accent-green);
        display: inline-block;
    }
    
    .metric-delta.negative {
        background: rgba(255, 59, 48, 0.2);
        color: var(--accent-red);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue);
        color: white;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-glass);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar-content {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    .stMultiSelect > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    .stDateInput > div > div > input {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 122, 255, 0.4);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Chart containers */
    .chart-container {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        background: rgba(45, 45, 45, 0.7);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* Insight boxes */
    .insight-box {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-blue);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        background: rgba(45, 45, 45, 0.7);
        transform: translateX(4px);
    }
    
    .insight-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-message {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .insight-priority {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .priority-critical { background: rgba(255, 59, 48, 0.2); color: var(--accent-red); }
    .priority-high { background: rgba(255, 149, 0, 0.2); color: var(--accent-orange); }
    .priority-medium { background: rgba(255, 204, 0, 0.2); color: #FFCC00; }
    .priority-low { background: rgba(52, 199, 89, 0.2); color: var(--accent-green); }
    
    /* Map legend */
    .map-legend {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        color: var(--text-secondary);
    }
    
    .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 0.75rem;
    }
    
    /* Data tables */
    .dataframe {
        background: var(--bg-card);
        backdrop-filter: var(--blur);
        -webkit-backdrop-filter: var(--blur);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .metric-card {
            margin: 0.5rem 0;
        }
        
        .glass-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Dark theme for plotly charts
plotly_theme = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#FFFFFF', 'family': 'Inter'},
        'colorway': ['#007AFF', '#5856D6', '#34C759', '#FF9500', '#FF3B30', '#00C7BE', '#5AC8FA'],
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'linecolor': 'rgba(255,255,255,0.2)',
            'zerolinecolor': 'rgba(255,255,255,0.2)'
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'linecolor': 'rgba(255,255,255,0.2)',
            'zerolinecolor': 'rgba(255,255,255,0.2)'
        }
    }
}


class AccidentDashboard:
    def __init__(self):
        self.db_url = "postgresql://postgres:password123@localhost:5432/accidentiq"
        self.engine = create_engine(self.db_url)

    @st.cache_data(ttl=600)
    def load_data(_self):
        """Load accident data from database"""
        try:
            query = """
            SELECT * FROM accidents 
            WHERE latitude IS NOT NULL 
            AND longitude IS NOT NULL
            ORDER BY accident_datetime DESC
            """
            df = pd.read_sql(query, _self.engine)
            df['accident_datetime'] = pd.to_datetime(df['accident_datetime'])
            return df
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=600)
    def load_analysis_results(_self):
        """Load pre-computed analysis results"""
        results = {}
        tables = ['accident_clusters_spark', 'rush_hour_patterns',
                  'seasonal_patterns', 'road_risk_scores']

        for table in tables:
            try:
                results[table] = pd.read_sql(
                    f"SELECT * FROM {table}", _self.engine)
            except:
                results[table] = pd.DataFrame()

        return results

    def create_overview_metrics(self, df):
        """Create modern metric cards"""
        if df.empty:
            return

        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = [
            {
                'label': 'Total Accidents',
                'value': f"{len(df):,}",
                'delta': f"+{len(df[df['accident_datetime'] > datetime.datetime.now() - datetime.timedelta(days=30)]):,} last 30 days",
                'positive': True
            },
            {
                'label': 'Fatal Accidents',
                'value': f"{len(df[df['severity'] == 3]):,}",
                'delta': f"{(len(df[df['severity'] == 3])/len(df)*100):.1f}% of total",
                'positive': False
            },
            {
                'label': 'Cities Covered',
                'value': f"{df['city'].nunique()}",
                'delta': f"Across {df['state'].nunique()} states",
                'positive': True
            },
            {
                'label': 'Avg Severity',
                'value': f"{df['severity'].mean():.2f}",
                'delta': "Scale 1-3",
                'positive': False
            },
            {
                'label': 'Total Casualties',
                'value': f"{df['casualties'].sum():,}",
                'delta': f"{df['fatalities'].sum():,} fatalities",
                'positive': False
            }
        ]

        for col, metric in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                delta_class = "" if metric['positive'] else "negative"
                st.markdown(f"""
                <div class="metric-card fade-in">
                    <div class="metric-label">{metric['label']}</div>
                    <div class="metric-value">{metric['value']}</div>
                    <div class="metric-delta {delta_class}">{metric['delta']}</div>
                </div>
                """, unsafe_allow_html=True)

    def create_accident_heatmap(self, df, city_filter=None):
        """Create modern styled accident heatmap"""
        if df.empty:
            return None

        if city_filter and city_filter != "All Cities":
            df = df[df['city'] == city_filter]

        if len(df) > 5000:
            df = df.sample(5000)

        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()

        # Create map with dark theme
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=None
        )

        # Add dark theme tile layer
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Dark Theme',
            overlay=False,
            control=True
        ).add_to(m)

        # Add heatmap with modern colors
        from folium.plugins import HeatMap
        heat_data = [[row['latitude'], row['longitude'], row['severity']]
                     for _, row in df.iterrows()]

        HeatMap(
            heat_data,
            radius=15,
            blur=10,
            gradient={
                0.2: '#007AFF',
                0.4: '#34C759',
                0.6: '#FF9500',
                1: '#FF3B30'
            }
        ).add_to(m)

        # Add markers with modern styling
        sample_df = df.sample(min(100, len(df)))
        for _, row in sample_df.iterrows():
            color = {1: '#34C759', 2: '#FF9500', 3: '#FF3B30'}[row['severity']]
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fillColor=color,
                popup=f"""
                <div style="font-family: Inter; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 8px;">
                    <b>Accident Details</b><br>
                    City: {row['city']}<br>
                    Severity: {row['severity_label']}<br>
                    Date: {row['date']}<br>
                    Vehicle: {row['vehicle_type']}<br>
                    Road: {row['road_type']}
                </div>
                """,
                fillOpacity=0.8,
                weight=2
            ).add_to(m)

        return m

    def create_temporal_analysis(self, df):
        """Create modern temporal analysis charts"""
        if df.empty:
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.markdown(
                '<h3 class="section-header">Accidents by Hour</h3>', unsafe_allow_html=True)

            hourly_accidents = df.groupby(
                'hour').size().reset_index(name='count')

            fig = px.bar(
                hourly_accidents,
                x='hour',
                y='count',
                title="",
                color='count',
                color_continuous_scale=['#007AFF', '#5856D6', '#FF3B30']
            )
            fig.update_layout(plotly_theme['layout'])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.markdown(
                '<h3 class="section-header">Monthly Trends</h3>', unsafe_allow_html=True)

            monthly_data = df.groupby(
                ['year', 'month']).size().reset_index(name='count')
            monthly_data['date'] = pd.to_datetime(
                monthly_data[['year', 'month']].assign(day=1))

            fig = px.line(
                monthly_data,
                x='date',
                y='count',
                title="",
                markers=True,
                line_shape='spline'
            )
            fig.update_traces(line_color='#007AFF', marker_color='#007AFF')
            fig.update_layout(plotly_theme['layout'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def create_severity_analysis(self, df):
        """Create modern severity analysis charts"""
        if df.empty:
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.markdown(
                '<h3 class="section-header">Severity Distribution</h3>', unsafe_allow_html=True)

            severity_counts = df['severity_label'].value_counts()

            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="",
                color_discrete_map={
                    'Minor': '#34C759',
                    'Serious': '#FF9500',
                    'Fatal': '#FF3B30'
                },
                hole=0.4
            )
            fig.update_layout(plotly_theme['layout'])
            fig.update_traces(textfont_size=12, textfont_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.markdown(
                '<h3 class="section-header">Vehicle Type Analysis</h3>', unsafe_allow_html=True)

            vehicle_severity = df.groupby(
                ['vehicle_type', 'severity_label']).size().unstack(fill_value=0)

            fig = px.bar(
                vehicle_severity.reset_index(),
                x='vehicle_type',
                y=['Minor', 'Serious', 'Fatal'],
                title="",
                color_discrete_map={
                    'Minor': '#34C759',
                    'Serious': '#FF9500',
                    'Fatal': '#FF3B30'
                }
            )
            fig.update_layout(plotly_theme['layout'])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def create_city_comparison(self, df):
        """Create modern city comparison"""
        if df.empty:
            return

        st.markdown('<h2 class="section-header">City Analysis</h2>',
                    unsafe_allow_html=True)

        city_stats = df.groupby('city').agg({
            'accident_id': 'count',
            'severity': 'mean',
            'fatalities': 'sum',
            'casualties': 'sum'
        }).round(2)
        city_stats.columns = [
            'Total Accidents', 'Avg Severity', 'Total Fatalities', 'Total Casualties']
        city_stats = city_stats.sort_values('Total Accidents', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            fig = px.bar(
                city_stats.reset_index().head(10),
                x='city',
                y='Total Accidents',
                title="Top 10 Cities by Accident Count",
                color='Avg Severity',
                color_continuous_scale=['#007AFF', '#FF9500', '#FF3B30']
            )
            fig.update_layout(plotly_theme['layout'])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<h4 style="color: #FFFFFF; margin-bottom: 1rem;">City Statistics</h4>', unsafe_allow_html=True)
            st.dataframe(city_stats.head(10), height=400)
            st.markdown('</div>', unsafe_allow_html=True)

    def create_recommendations(self, df, analysis_results):
        """Create modern AI recommendations"""
        st.markdown(
            '<h2 class="section-header">AI-Powered Recommendations</h2>', unsafe_allow_html=True)

        if df.empty:
            return

        insights = []

        # Peak hour analysis
        hourly_accidents = df.groupby('hour').size()
        peak_hour = hourly_accidents.idxmax()
        peak_accidents = hourly_accidents.max()

        insights.append({
            'title': 'Peak Hour Alert',
            'message': f"Hour {peak_hour}:00 has the highest accident rate ({peak_accidents} accidents). Consider increased traffic monitoring during this time.",
            'priority': 'High'
        })

        # Fatal accident hotspots
        fatal_cities = df[df['severity'] == 3]['city'].value_counts().head(3)
        if not fatal_cities.empty:
            insights.append({
                'title': 'Fatal Accident Hotspots',
                'message': f"Cities with highest fatal accidents: {', '.join(fatal_cities.index[:3])}. Immediate infrastructure review recommended.",
                'priority': 'Critical'
            })

        # Vehicle type recommendations
        risky_vehicles = df.groupby('vehicle_type')[
            'severity'].mean().sort_values(ascending=False).head(2)
        insights.append({
            'title': 'Vehicle Safety Focus',
            'message': f"Focus safety campaigns on {risky_vehicles.index[0]} and {risky_vehicles.index[1]} drivers - they have higher average accident severity.",
            'priority': 'Medium'
        })

        # Weather-based insights
        weather_accidents = df.groupby(
            'weather').size().sort_values(ascending=False)
        if len(weather_accidents) > 1:
            insights.append({
                'title': 'Weather-Based Prevention',
                'message': f"Most accidents occur during {weather_accidents.index[0]} conditions. Deploy weather-specific safety measures.",
                'priority': 'Medium'
            })

        # Display modern insights
        for insight in insights:
            priority_class = f"priority-{insight['priority'].lower()}"

            st.markdown(f"""
            <div class="insight-box fade-in">
                <div class="insight-title">{insight['title']}</div>
                <div class="insight-message">{insight['message']}</div>
                <span class="insight-priority {priority_class}">Priority: {insight['priority']}</span>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Modern header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>AccidentIQ</h1>
        <p>Advanced Road Safety Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize dashboard
    dashboard = AccidentDashboard()

    # Modern sidebar
    st.sidebar.markdown('<div class="sidebar-content">',
                        unsafe_allow_html=True)
    st.sidebar.markdown(
        '<h3 style="color: #FFFFFF; margin-bottom: 1rem;">Filters & Controls</h3>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        df = dashboard.load_data()
        analysis_results = dashboard.load_analysis_results()

    if df.empty:
        st.error(
            "No data available. Please run the data collection and processing scripts first.")
        st.stop()

    # Filters
    cities = ["All Cities"] + sorted(df['city'].unique())
    selected_city = st.sidebar.selectbox("Select City", cities)

    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    severity_options = st.sidebar.multiselect(
        "Severity Levels",
        options=df['severity_label'].unique(),
        default=df['severity_label'].unique()
    )

    vehicle_options = st.sidebar.multiselect(
        "Vehicle Types",
        options=df['vehicle_type'].unique(),
        default=df['vehicle_type'].unique()
    )

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    filtered_df = df.copy()

    if selected_city != "All Cities":
        filtered_df = filtered_df[filtered_df['city'] == selected_city]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) &
            (filtered_df['date'] <= date_range[1])
        ]

    if severity_options:
        filtered_df = filtered_df[filtered_df['severity_label'].isin(
            severity_options)]

    if vehicle_options:
        filtered_df = filtered_df[filtered_df['vehicle_type'].isin(
            vehicle_options)]

    # Modern tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Geospatial Analysis",
        "Temporal Patterns",
        "Risk Analysis",
        "Recommendations"
    ])

    with tab1:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        dashboard.create_overview_metrics(filtered_df)
        st.markdown("---")
        dashboard.create_city_comparison(filtered_df)
        st.markdown("---")
        dashboard.create_severity_analysis(filtered_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(
            '<h2 class="section-header">Geospatial Analysis</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<div class="chart-container">',
                        unsafe_allow_html=True)
            st.markdown(
                '<h3 style="color: #FFFFFF; margin-bottom: 1rem;">Accident Heatmap</h3>', unsafe_allow_html=True)
            if not filtered_df.empty:
                map_obj = dashboard.create_accident_heatmap(
                    filtered_df, selected_city)
                if map_obj:
                    st_folium(map_obj, width=700, height=500)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="map-legend">
                <h4 style="color: #FFFFFF; margin-bottom: 1rem;">Map Legend</h4>
                
                <div style="margin-bottom: 1rem;">
                    <h5 style="color: #A0A0A0; font-size: 0.9rem; margin-bottom: 0.5rem;">HEATMAP COLORS</h5>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #007AFF;"></div>
                        Low severity areas
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #34C759;"></div>
                        Moderate risk zones
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF9500;"></div>
                        High severity areas
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF3B30;"></div>
                        Critical risk zones
                    </div>
                </div>
                
                <div>
                    <h5 style="color: #A0A0A0; font-size: 0.9rem; margin-bottom: 0.5rem;">ACCIDENT MARKERS</h5>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #34C759;"></div>
                        Minor accidents
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF9500;"></div>
                        Serious accidents
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF3B30;"></div>
                        Fatal accidents
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Spatial statistics
            if not filtered_df.empty:
                st.markdown("""
                <div class="glass-card" style="margin-top: 1rem;">
                    <h4 style="color: #FFFFFF; margin-bottom: 1rem;">Spatial Statistics</h4>
                </div>
                """, unsafe_allow_html=True)

                if 'h3_index' in filtered_df.columns:
                    h3_stats = filtered_df.groupby('h3_index').agg({
                        'accident_id': 'count',
                        'severity': 'mean'
                    }).round(2)
                    h3_stats.columns = ['Accidents', 'Avg Severity']
                    top_h3 = h3_stats.sort_values(
                        'Accidents', ascending=False).head(5)
                    st.dataframe(top_h3, use_container_width=True)

        # Hotspot analysis
        if 'accident_clusters_spark' in analysis_results and not analysis_results['accident_clusters_spark'].empty:
            st.markdown("---")
            st.markdown(
                '<h3 class="section-header">AI-Detected Hotspots</h3>', unsafe_allow_html=True)

            clusters_df = analysis_results['accident_clusters_spark']
            if selected_city != "All Cities":
                clusters_df = clusters_df[clusters_df['city'] == selected_city]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(
                clusters_df[['city', 'accident_count',
                             'avg_severity', 'total_fatalities']].head(10),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown(
            '<h2 class="section-header">Temporal Patterns</h2>', unsafe_allow_html=True)
        dashboard.create_temporal_analysis(filtered_df)

        # Rush hour patterns
        if 'rush_hour_patterns' in analysis_results and not analysis_results['rush_hour_patterns'].empty:
            st.markdown("---")
            st.markdown(
                '<h3 class="section-header">Rush Hour Analysis</h3>', unsafe_allow_html=True)

            rush_df = analysis_results['rush_hour_patterns']
            if selected_city != "All Cities":
                rush_df = rush_df[rush_df['city'] == selected_city]

            if not rush_df.empty:
                st.markdown('<div class="chart-container">',
                            unsafe_allow_html=True)
                fig = px.bar(
                    rush_df.groupby('time_category')[
                        'accident_count'].sum().reset_index(),
                    x='time_category',
                    y='accident_count',
                    title="Accidents by Time Category",
                    color='accident_count',
                    color_continuous_scale=['#007AFF', '#5856D6', '#FF3B30']
                )
                fig.update_layout(plotly_theme['layout'])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Seasonal patterns
        if 'seasonal_patterns' in analysis_results and not analysis_results['seasonal_patterns'].empty:
            st.markdown("---")
            st.markdown(
                '<h3 class="section-header">Seasonal Analysis</h3>', unsafe_allow_html=True)

            seasonal_df = analysis_results['seasonal_patterns']
            if selected_city != "All Cities":
                seasonal_df = seasonal_df[seasonal_df['city'] == selected_city]

            if not seasonal_df.empty:
                seasonal_summary = seasonal_df.groupby(
                    'season')['accident_count'].sum().reset_index()

                st.markdown('<div class="chart-container">',
                            unsafe_allow_html=True)
                fig = px.pie(
                    seasonal_summary,
                    values='accident_count',
                    names='season',
                    title="Seasonal Distribution of Accidents",
                    color_discrete_sequence=[
                        '#007AFF', '#34C759', '#FF9500', '#FF3B30'],
                    hole=0.4
                )
                fig.update_layout(plotly_theme['layout'])
                fig.update_traces(textfont_size=12, textfont_color='white')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<h2 class="section-header">Risk Analysis</h2>',
                    unsafe_allow_html=True)

        # Risk analysis from Spark results
        if 'road_risk_scores' in analysis_results and not analysis_results['road_risk_scores'].empty:
            risk_df = analysis_results['road_risk_scores']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="chart-container">',
                            unsafe_allow_html=True)
                st.markdown(
                    '<h4 style="color: #FFFFFF; margin-bottom: 1rem;">Risk by Road Type</h4>', unsafe_allow_html=True)
                road_risk = risk_df.groupby('road_type')[
                    'risk_score'].mean().sort_values(ascending=False)

                fig = px.bar(
                    x=road_risk.index,
                    y=road_risk.values,
                    title="",
                    color=road_risk.values,
                    color_continuous_scale=['#007AFF', '#FF9500', '#FF3B30']
                )
                fig.update_layout(plotly_theme['layout'])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="chart-container">',
                            unsafe_allow_html=True)
                st.markdown(
                    '<h4 style="color: #FFFFFF; margin-bottom: 1rem;">Weather Impact</h4>', unsafe_allow_html=True)
                weather_risk = risk_df.groupby(
                    'weather')['risk_score'].mean().sort_values(ascending=False)

                fig = px.bar(
                    x=weather_risk.index,
                    y=weather_risk.values,
                    title="",
                    color=weather_risk.values,
                    color_continuous_scale=['#007AFF', '#FF9500', '#FF3B30']
                )
                fig.update_layout(plotly_theme['layout'])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Risk metrics
        if not filtered_df.empty:
            st.markdown("---")
            st.markdown('<h3 class="section-header">Risk Metrics</h3>',
                        unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                fatality_rate = (
                    filtered_df['fatalities'].sum() / len(filtered_df) * 100)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fatality Rate</div>
                    <div class="metric-value">{fatality_rate:.2f}%</div>
                    <div class="metric-delta">Per 100 accidents</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                avg_casualties = filtered_df['casualties'].mean()
                overall_avg = df['casualties'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Casualties</div>
                    <div class="metric-value">{avg_casualties:.1f}</div>
                    <div class="metric-delta">vs {overall_avg:.1f} overall</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                high_severity = len(filtered_df[filtered_df['severity'] >= 2])
                high_severity_rate = (high_severity / len(filtered_df) * 100)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Severity Rate</div>
                    <div class="metric-value">{high_severity_rate:.1f}%</div>
                    <div class="metric-delta">Serious + Fatal</div>
                </div>
                """, unsafe_allow_html=True)

    with tab5:
        dashboard.create_recommendations(filtered_df, analysis_results)

        # Infrastructure recommendations
        st.markdown("---")
        st.markdown(
            '<h3 class="section-header">Infrastructure Recommendations</h3>', unsafe_allow_html=True)

        if not filtered_df.empty:
            location_accidents = filtered_df.groupby(['city', 'road_type']).agg({
                'accident_id': 'count',
                'severity': 'mean',
                'fatalities': 'sum'
            }).reset_index()
            location_accidents.columns = [
                'City', 'Road Type', 'Accident Count', 'Avg Severity', 'Fatalities']
            location_accidents = location_accidents.sort_values(
                'Accident Count', ascending=False)

            for i, (_, row) in enumerate(location_accidents.head(5).iterrows()):
                recommendations = []

                if row['Avg Severity'] > 2:
                    recommendations.append("Install speed cameras")
                if row['Road Type'] == 'Highway':
                    recommendations.append("Add median barriers")
                if row['Fatalities'] > 0:
                    recommendations.append("Improve emergency response")
                if row['Accident Count'] > 10:
                    recommendations.append("Traffic signal optimization")

                st.markdown(f"""
                <div class="glass-card fade-in">
                    <h4 style="color: #FFFFFF; margin-bottom: 0.5rem;">{row['City']} - {row['Road Type']}</h4>
                    <p style="color: #A0A0A0; margin-bottom: 1rem;">
                        {int(row['Accident Count'])} accidents • {int(row['Fatalities'])} fatalities
                    </p>
                    <div style="color: #007AFF;">
                        <strong>Recommended:</strong> {', '.join(recommendations)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Modern footer with export options
    st.markdown("---")
    st.markdown('<h3 class="section-header">Export Options</h3>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Filtered Data", type="primary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"accident_data_{selected_city}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Generate Report", type="primary"):
            # Create summary report
            report = f"""# AccidentIQ Analysis Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Filter: {selected_city}

## Summary Statistics
- Total Accidents: {len(filtered_df):,}
- Fatal Accidents: {len(filtered_df[filtered_df['severity'] == 3]):,}
- Average Severity: {filtered_df['severity'].mean():.2f}
- Total Casualties: {filtered_df['casualties'].sum():,}

## Top Risk Areas
{location_accidents.head(5).to_string(index=False) if not filtered_df.empty else 'No data available'}
            """

            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"accident_report_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

    with col3:
        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
