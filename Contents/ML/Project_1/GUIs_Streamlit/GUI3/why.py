import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import math
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Housing Market Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
  :root { --bg: #1b1126; --card: #2a1630; --accent: #9b59b6; --muted: #d9cfe8; --panel: #241028; }
  .stApp, .main, body { background-color: var(--bg) !important; color: var(--muted) !important; }
  .block-container { padding: 1rem 1.5rem; }
  h1, h2, h3 { color: var(--accent) !important; }
  .main-header { font-size: 28px; font-weight:700; text-align:center; margin-bottom:10px; color:var(--accent) !important; }
  .chart-container { background-color: var(--card); padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); margin-bottom: 1rem; }
  .metric-card { background-color: var(--panel); padding: 0.6rem 0.8rem; border-radius: 8px; text-align:center; }
  .subheader-custom { color: var(--accent); font-weight:600; font-size:18px; margin-bottom:6px; }
  .stButton>button { background-color: #7b2cbf; color: white; border: none; }
  .plotly-graph-div { background: transparent !important; }
  /* header color */
  .header-title { color: #9b59b6 !important; font-weight:700; font-size:28px; text-align:center; }
</style>
""", unsafe_allow_html=True)

class StreamlitHousingAnalyzer:
    """Housing data analyzer for Streamlit dashboard"""
    
    def __init__(self):
        self.data = self.load_data()
        self.processed_data = self.preprocess_data()
    
    def load_data(self):
        """Load housing data from CSV"""
        try:
            data = []
            with open(r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_1\GUIs_Streamlit\GUI3\housing.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data.append({
                        'RM': float(row['RM']),
                        'LSTAT': float(row['LSTAT']),
                        'PTRATIO': float(row['PTRATIO']),
                        'MEDV': float(row['MEDV'])
                    })
            return data
        except FileNotFoundError:
            st.error("❌ housing.csv file not found!")
            return []
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return []
    
    def preprocess_data(self):
        """Preprocess data for dashboard"""
        if not self.data:
            return []
        
        processed = []
        for i, row in enumerate(self.data):
            # Create synthetic time series data
            base_date = datetime(2020, 1, 1)
            month_offset = i % 48
            current_date = base_date + timedelta(days=month_offset * 30)
            
            # Create categories
            if row['MEDV'] < 300000:
                price_category = 'Budget'
            elif row['MEDV'] < 500000:
                price_category = 'Mid-Range'
            elif row['MEDV'] < 700000:
                price_category = 'Premium'
            else:
                price_category = 'Luxury'
            
            if row['RM'] < 5:
                room_category = 'Small'
            elif row['RM'] < 7:
                room_category = 'Medium'
            else:
                room_category = 'Large'
            
            if row['LSTAT'] < 10:
                socio_category = 'High Income'
            elif row['LSTAT'] < 20:
                socio_category = 'Middle Income'
            else:
                socio_category = 'Low Income'
            
            if row['PTRATIO'] < 15:
                school_category = 'Excellent'
            elif row['PTRATIO'] < 20:
                school_category = 'Good'
            else:
                school_category = 'Average'
            
            processed.append({
                'RM': row['RM'],
                'LSTAT': row['LSTAT'],
                'PTRATIO': row['PTRATIO'],
                'MEDV': row['MEDV'],
                'Date': current_date,
                'Year': current_date.year,
                'Month': current_date.month,
                'Quarter': (current_date.month - 1) // 3 + 1,
                'Price_Category': price_category,
                'Room_Category': room_category,
                'Socio_Category': socio_category,
                'School_Category': school_category,
                'Price_Per_Room': row['MEDV'] / row['RM'],
                'Value_Score': (row['RM'] * 0.3) + ((100 - row['LSTAT']) * 0.4) + ((25 - row['PTRATIO']) * 0.3)
            })
        
        return processed
    
    def get_filtered_data(self, price_filter, room_filter, socio_filter, school_filter):
        """Filter data based on selected criteria"""
        filtered = self.processed_data
        
        if price_filter and 'All' not in price_filter:
            filtered = [row for row in filtered if row['Price_Category'] in price_filter]
        
        if room_filter and 'All' not in room_filter:
            filtered = [row for row in filtered if row['Room_Category'] in room_filter]
        
        if socio_filter and 'All' not in socio_filter:
            filtered = [row for row in filtered if row['Socio_Category'] in socio_filter]
        
        if school_filter and 'All' not in school_filter:
            filtered = [row for row in filtered if row['School_Category'] in school_filter]
        
        return filtered

def create_metrics_section(analyzer, filtered_data):
    """Create key metrics section"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">💜 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = statistics.mean([row['MEDV'] for row in filtered_data])
        st.metric("Average House Price", f"${avg_price:,.0f}")
    
    with col2:
        avg_rooms = statistics.mean([row['RM'] for row in filtered_data])
        st.metric("Average Rooms", f"{avg_rooms:.1f}")
    
    with col3:
        avg_lstat = statistics.mean([row['LSTAT'] for row in filtered_data])
        st.metric("Average LSTAT (Poverty %)", f"{avg_lstat:.1f}%")
    
    with col4:
        total_properties = len(filtered_data)
        st.metric("Total Properties", f"{total_properties:,}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_line_charts(analyzer, filtered_data):
    """Create interactive line charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">📈 Trend Analysis</div>', unsafe_allow_html=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(filtered_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price trends over time
        monthly_avg = df.groupby(['Year', 'Month'])['MEDV'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))
        
        fig = px.line(monthly_avg, x='Date', y='MEDV', 
                     title='Average Price Over Time',
                     labels={'MEDV': 'Average Price ($)', 'Date': 'Date'},
                     color_discrete_sequence=['#2E86AB'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Room count vs Price
        fig = px.scatter(df, x='RM', y='MEDV', color='Price_Category',
                        title='Price vs Room Count',
                        labels={'RM': 'Average Rooms', 'MEDV': 'Price ($)'},
                        color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_bar_charts(analyzer, filtered_data):
    """Create interactive bar charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">📊 Category & Distribution</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price categories
        price_counts = df['Price_Category'].value_counts()
        fig = px.bar(x=price_counts.index, y=price_counts.values,
                    title='Properties by Price Category',
                    labels={'x': 'Price Category', 'y': 'Count'},
                    color=price_counts.index,
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Room categories
        room_counts = df['Room_Category'].value_counts()
        fig = px.bar(x=room_counts.index, y=room_counts.values,
                    title='Properties by Room Category',
                    labels={'x': 'Room Category', 'y': 'Count'},
                    color=room_counts.index,
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution histogram
    fig = px.histogram(df, x='MEDV', nbins=20, title='Price Distribution',
                      labels={'MEDV': 'Price ($)', 'count': 'Frequency'},
                      color_discrete_sequence=['#2E86AB'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_pie_charts(analyzer, filtered_data):
    """Create interactive pie charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">🥧 Share by Category</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price categories pie chart
        price_counts = df['Price_Category'].value_counts()
        fig = px.pie(values=price_counts.values, names=price_counts.index,
                    title='Distribution by Price Category',
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Socioeconomic categories pie chart
        socio_counts = df['Socio_Category'].value_counts()
        fig = px.pie(values=socio_counts.values, names=socio_counts.index,
                    title='Distribution by Socioeconomic Category',
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_variable_width_charts(analyzer, filtered_data):
    """Create variable width charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">📏 Sample-Weighted Views</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    # Room categories with variable width based on count
    room_stats = df.groupby('Room_Category').agg({
        'MEDV': ['mean', 'count']
    }).round(0)
    room_stats.columns = ['avg_price', 'count']
    room_stats = room_stats.reset_index()
    
    fig = px.bar(room_stats, x='Room_Category', y='avg_price',
                title='Average Price by Room Category (Width = Sample Size)',
                labels={'avg_price': 'Average Price ($)', 'Room_Category': 'Room Category'},
                color='Room_Category',
                color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
    
    # Add count as text on bars
    fig.update_traces(texttemplate='%{y:,.0f}<br>Count: %{customdata}', 
                     textposition='outside',
                     customdata=room_stats['count'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_time_series_charts(analyzer, filtered_data):
    """Create time series charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">⏰ Temporal Analysis</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    # Monthly trends
    monthly_data = df.groupby(['Year', 'Month']).agg({
        'MEDV': 'mean',
        'RM': 'mean',
        'LSTAT': 'mean'
    }).reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Price Trends', 'Room Count Trends'))
    
    # Price trends
    fig.add_trace(go.Scatter(x=monthly_data['Date'], y=monthly_data['MEDV'],
                            mode='lines+markers', name='Average Price',
                            line=dict(color='#2E86AB', width=3)),
                 row=1, col=1)
    
    # Room count trends
    fig.add_trace(go.Scatter(x=monthly_data['Date'], y=monthly_data['RM'],
                            mode='lines+markers', name='Average Rooms',
                            line=dict(color='#A23B72', width=3)),
                 row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_stacked_bar_charts(analyzer, filtered_data):
    """Create stacked bar charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">📚 Comparative Stacks</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    # Price categories by room category
    cross_tab = pd.crosstab(df['Room_Category'], df['Price_Category'])
    
    fig = px.bar(cross_tab, title='Price Categories by Room Category',
                labels={'value': 'Count', 'index': 'Room Category'},
                color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    fig.update_layout(barmode='stack')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_donut_charts(analyzer, filtered_data):
    """Create concentric donut charts"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">🍩 Composition Overview</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price categories donut
        price_counts = df['Price_Category'].value_counts()
        fig = px.pie(values=price_counts.values, names=price_counts.index,
                    title='Price Categories Distribution',
                    hole=0.4,
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Room categories donut
        room_counts = df['Room_Category'].value_counts()
        fig = px.pie(values=room_counts.values, names=room_counts.index,
                    title='Room Categories Distribution',
                    hole=0.4,
                    color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_correlation_heatmap(analyzer, filtered_data):
    """Create correlation heatmap"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader-custom">🔥 Relationship Matrix</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(filtered_data)
    numeric_cols = ['RM', 'LSTAT', 'PTRATIO', 'MEDV', 'Price_Per_Room', 'Value_Score']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   title='Correlation Matrix',
                   color_continuous_scale='RdBu')
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Market Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = StreamlitHousingAnalyzer()
    
    if not analyzer.processed_data:
        st.error("No data available for analysis")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Get unique values for filters
    price_categories = ['All'] + list(set([row['Price_Category'] for row in analyzer.processed_data]))
    room_categories = ['All'] + list(set([row['Room_Category'] for row in analyzer.processed_data]))
    socio_categories = ['All'] + list(set([row['Socio_Category'] for row in analyzer.processed_data]))
    school_categories = ['All'] + list(set([row['School_Category'] for row in analyzer.processed_data]))
    
    # Filter controls
    price_filter = st.sidebar.multiselect("Price Category", price_categories, default=['All'])
    room_filter = st.sidebar.multiselect("Room Category", room_categories, default=['All'])
    socio_filter = st.sidebar.multiselect("Socioeconomic Category", socio_categories, default=['All'])
    school_filter = st.sidebar.multiselect("School Quality", school_categories, default=['All'])
    
    # Apply filters
    filtered_data = analyzer.get_filtered_data(price_filter, room_filter, socio_filter, school_filter)
    
    if not filtered_data:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Display filtered data count
    st.sidebar.success(f"Showing {len(filtered_data)} properties")
    
    # Create visualizations
    create_metrics_section(analyzer, filtered_data)
    create_line_charts(analyzer, filtered_data)
    create_bar_charts(analyzer, filtered_data)
    create_pie_charts(analyzer, filtered_data)
    create_variable_width_charts(analyzer, filtered_data)
    create_time_series_charts(analyzer, filtered_data)
    create_stacked_bar_charts(analyzer, filtered_data)
    create_donut_charts(analyzer, filtered_data)
    create_correlation_heatmap(analyzer, filtered_data)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2024 Housing Market Analysis Dashboard") 

if __name__ == "__main__":
    main()
