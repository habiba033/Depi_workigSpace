import streamlit as st
import pandas as pd
import numpy as np
import time # Used for simulated motion/sequential loading
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION AND DATA LOADING ---

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Wholesale Customer Segmentation Dashboard (Plotly)", page_icon="📊")

# Define spending columns for consistency
SPENDING_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

@st.cache_data
def load_data():
    """Loads, cleans, and prepares the customer data."""
    
    # 1. ISOLATE FILE READ (Fixes UnboundLocalError)
    try:
        df = pd.read_csv(r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_3\customers.csv')
    except Exception as e:
        # Immediate fallback return path on file error
        st.error(f"Error loading customers.csv: {e}. Please ensure the file is in the correct directory.")
        return pd.DataFrame(), None, None, None, None 

    # --- Data Processing (Guaranteed to run only if file was read successfully) ---

    # Rename Channel and Region for better labels
    df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})
    df['Region'] = df['Region'].replace({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})

    # Calculate Total_Spending 
    df['Total_Spending'] = df[SPENDING_COLS].sum(axis=1)

    # Calculate 99th percentile for clipping outliers in distributions
    q99 = df[SPENDING_COLS + ['Total_Spending']].quantile(0.99)

    # Aggregations for Channel and Region
    channel_agg = df.groupby('Channel')[SPENDING_COLS].mean().reset_index()
    region_agg = df.groupby('Region')[SPENDING_COLS].mean().reset_index()

    # Prepare data for Marry Categories Line Chart (includes the previous fix)
    multi_cat_df = region_agg.set_index('Region')[SPENDING_COLS].T.reset_index()
    multi_cat_df = multi_cat_df.rename(columns={'index': 'Category'})
    multi_cat_df = multi_cat_df.melt(id_vars='Category', var_name='Region', value_name='Average Spending')
    
    # Final return
    return df, q99, channel_agg, region_agg, multi_cat_df

df, q99, channel_agg, region_agg, multi_cat_df = load_data()

# Check if data loaded successfully. If load_data returned None for q99, stop execution.
if df.empty or q99 is None:
    st.stop()

# --- HELPER FUNCTION FOR DISPLAYING CHARTS WITH ANIMATION ---
def safe_display_chart(col, func, data, q99_val=None, delay=0.05):
    """Function to safely display a chart in a column with a small delay for 'motion'."""
    
    # Extract the title from the function's docstring
    title = func.__doc__.split(' - ')[1].strip()

    with col:
        with st.expander(f"{title}", expanded=True):
            try:
                # Add a small delay for the 'motion' effect (simulated load)
                time.sleep(delay)
                
                # Handle functions that take q99
                if q99_val is not None:
                    fig = func(data, q99_val)
                else:
                    fig = func(data)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"Error generating {func.__name__}: {e}")

# --- 20 VISUALIZATION FUNCTIONS (Plotly Versions) ---

# Group 1: Comparison and Distribution 
def viz_1_bar_vertical(data):
    """1. Vertical Bar Chart - Average Fresh Product Spending by Customer Channel"""
    fig = px.bar(data, x='Channel', y='Fresh', color='Channel', 
                 title='Avg Fresh Spending by Channel', 
                 labels={'Fresh': 'Average Fresh Spending (USD)'},
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(xaxis_title='Customer Channel', showlegend=False)
    return fig

def viz_5_bar_histogram(data, q99_val):
    """5. Bar Histogram - Distribution of Milk Product Spending (Clipped)"""
    clipped_milk = data['Milk'].clip(upper=q99_val['Milk'])
    fig = px.histogram(x=clipped_milk, nbins=30, 
                       title='Distribution of Milk Spending', 
                       labels={'x': 'Milk Spending (Capped at 99th Pctl)', 'count': 'Frequency (Count)'},
                       color_discrete_sequence=['skyblue'])
    fig.update_layout(bargap=0.05)
    return fig

def viz_6_line_histogram(data, q99_val):
    """6. Line Histogram (Frequency Polygon) - Frequency Distribution of Grocery Spending (Clipped)"""
    clipped_grocery = data['Grocery'].clip(upper=q99_val['Grocery'])
    hist, bins = np.histogram(clipped_grocery, bins=25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig = go.Figure(data=[
        go.Scatter(x=bin_centers, y=hist, mode='lines+markers', name='Frequency', 
                   line=dict(color='red', width=2), marker=dict(size=5))
    ])
    fig.add_trace(go.Scatter(x=bin_centers, y=hist, fill='tozeroy', fillcolor='rgba(255,0,0,0.1)', mode='none'))

    fig.update_layout(title='Frequency Distribution of Grocery Spending',
                      xaxis_title='Grocery Spending (Capped at 99th Pctl)',
                      yaxis_title='Frequency')
    return fig

def viz_17_bar_horizontal(agg_data):
    """17. Horizontal Bar Chart - Average Delicatessen Spending by Region"""
    region_delicatessen_avg = agg_data[['Region', 'Delicatessen']].sort_values('Delicatessen', ascending=True)
    fig = px.bar(region_delicatessen_avg, x='Delicatessen', y='Region', color='Region', orientation='h',
                 title='Avg Delicatessen Spending by Region',
                 labels={'Delicatessen': 'Average Delicatessen Spending (USD)'},
                 color_discrete_sequence=px.colors.sequential.Sunset_r)
    fig.update_layout(yaxis_title='Region', showlegend=False)
    return fig

def viz_14_overlaid_histogram(data, q99_val):
    """14. Overlaid Histogram - Comparison of Fresh Spending Distribution by Channel (Clipped)"""
    data_clipped = data.copy()
    data_clipped['Fresh'] = data_clipped['Fresh'].clip(upper=q99_val['Fresh'])

    # Fixed color scale 
    fig = px.histogram(data_clipped, x='Fresh', color='Channel', nbins=30, histnorm='density',
                       title='Fresh Spending Distribution by Channel', 
                       labels={'Fresh': 'Fresh Spending (Capped at 99th Pctl)', 'count': 'Density'},
                       opacity=0.6, barmode='overlay', color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(xaxis_range=[0, q99_val['Fresh']])
    return fig

# Group 2: Categorical and Proportional Charts
def viz_2_stacked_bar(agg_data):
    """2. Stacked Bar Chart - Average Spending on Fresh, Milk, & Grocery by Channel"""
    temp_df = agg_data.set_index('Channel')[['Fresh', 'Milk', 'Grocery']].reset_index()
    df_melted = temp_df.melt(id_vars='Channel', var_name='Category', value_name='Average Spending')
    
    fig = px.bar(df_melted, x='Channel', y='Average Spending', color='Category', 
                 title='Avg Spending on Key Categories by Channel', 
                 color_discrete_sequence=px.colors.sequential.Plasma,
                 labels={'Average Spending': 'Average Spending (USD)'})
    fig.update_layout(barmode='stack', xaxis_title='Channel', legend_title='Category')
    return fig

def viz_3_pie_chart(data):
    """3. Pie Chart - Customer Count Distribution by Channel"""
    channel_counts = data['Channel'].value_counts().reset_index()
    channel_counts.columns = ['Channel', 'Count']
    
    fig = px.pie(channel_counts, values='Count', names='Channel', 
                 title='Customer Distribution by Channel', 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def viz_4_concentric_donut(data):
    """4. Concentric Donut Chart - Customer Count Distribution by Region"""
    region_counts = data['Region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    
    fig = px.pie(region_counts, values='Count', names='Region', hole=0.55,
                 title='Customer Distribution by Region', 
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='inside', textinfo='percent', hovertemplate='%{label}<br>Count: %{value}<extra></extra>')
    return fig

def viz_20_double_donut(data):
    """20. Double Donut Chart - Hierarchical Customer Proportion by Region (Outer) and Channel (Inner)"""
    
    # Prepare data for sunburst/double donut hierarchy: Overall -> Region -> Channel
    df_sunburst = data.groupby(['Region', 'Channel']).size().reset_index(name='Count')
    
    fig = px.sunburst(df_sunburst, path=['Region', 'Channel'], values='Count',
                      title='Hierarchical Customer Proportion by Region and Channel',
                      color='Region',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig


def viz_18_grouped_bar_chart(agg_data):
    """18. Grouped Bar Chart - Average Milk and Grocery Spending by Channel"""
    temp_df = agg_data.set_index('Channel')[['Milk', 'Grocery']].reset_index()
    df_melted = temp_df.melt(id_vars='Channel', var_name='Category', value_name='Average Spending')
    
    fig = px.bar(df_melted, x='Channel', y='Average Spending', color='Category', 
                 barmode='group',
                 title='Avg Milk and Grocery Spending by Channel', 
                 color_discrete_sequence=['sandybrown', 'darkkhaki'],
                 labels={'Average Spending': 'Average Spending (USD)'})
    fig.update_layout(xaxis_title='Channel', legend_title='Category')
    return fig

# Group 3: Trend and Time Series (Line/Area/Cumulative)

def viz_7_line_cumulative(data):
    """7. Line Chart - Cumulative Total Fresh Spending (Sorted by Customer Index)"""
    temp_df = data.sort_values('Fresh').reset_index(drop=True)
    temp_df['Fresh_Cumulative'] = temp_df['Fresh'].cumsum()
    
    fig = px.line(temp_df, x=temp_df.index, y='Fresh_Cumulative', 
                  title='Cumulative Total Fresh Spending', 
                  labels={'x': 'Customer Index (Sorted by Fresh Spending)', 'Fresh_Cumulative': 'Cumulative Fresh Spending (USD)'},
                  line_shape='linear', color_discrete_sequence=['green'])
    fig.update_traces(mode='lines', line=dict(width=2))
    return fig

def viz_13_time_series_line(data):
    """13. Time Series Line Chart - Cumulative Total Fresh Spending Trend (Sorted by Customer Index)"""
    temp_df = data.sort_values('Fresh').reset_index(drop=True)
    temp_df['Fresh_Cumulative'] = temp_df['Fresh'].cumsum()
    
    fig = px.line(temp_df, x=temp_df.index, y='Fresh_Cumulative', 
                  title='Cumulative Total Fresh Spending Trend', 
                  labels={'x': 'Customer ID (as Time/Sequence Index)', 'Fresh_Cumulative': 'Cumulative Fresh Spending (USD)'},
                  line_shape='linear', color_discrete_sequence=['darkblue'])
    fig.update_traces(line=dict(dash='dash', width=1.5), mode='lines+markers', marker=dict(size=4))
    return fig

def viz_15_stacked_area(agg_data):
    """15. Stacked Area Chart - Average Spending Profile (Fresh, Milk, Grocery) Across Regions"""
    temp_agg = agg_data.set_index('Region')[['Fresh', 'Milk', 'Grocery']].T.reset_index().rename(columns={'index': 'Category'})
    df_melted = temp_agg.melt(id_vars='Category', var_name='Region', value_name='Average Spending')
    
    # FIX: Removed groupnorm='first' as it caused the Invalid Value error.
    fig = px.area(df_melted, x='Category', y='Average Spending', color='Region', 
                  title='Avg Spending Profile Across Regions', 
                  color_discrete_sequence=px.colors.qualitative.Plotly,
                  labels={'Average Spending': 'Average Spending (USD)'})
    fig.update_layout(xaxis_title='Category', yaxis_title='Average Spending (USD)', legend_title='Region')
    return fig

def viz_16_marry_categories_line(multi_cat_data):
    """16. Marry Categories Line Chart - Average Spending across all Categories by Region"""
    fig = px.line(multi_cat_data, x='Category', y='Average Spending', color='Region', 
                  title='Avg Spending across all Categories by Region', 
                  labels={'Average Spending': 'Average Spending (USD)'},
                  markers=True, line_shape='linear', color_discrete_sequence=px.colors.qualitative.D3) 
    fig.update_traces(line=dict(width=2), marker=dict(size=5))
    fig.update_layout(xaxis_title='Spending Category')
    return fig

# Group 4: Advanced Relationship and Distribution 
def viz_8_scatter(data, q99_val):
    """8. Scatter Plot - Relationship between Milk Spending and Grocery Spending (Clipped)"""
    data_clipped = data.copy()
    data_clipped['Milk'] = data_clipped['Milk'].clip(upper=q99_val['Milk'])
    data_clipped['Grocery'] = data_clipped['Grocery'].clip(upper=q99_val['Grocery'])

    fig = px.scatter(data_clipped, x='Milk', y='Grocery', 
                     title='Relationship: Milk Spending vs. Grocery Spending', 
                     labels={'Milk': 'Milk Spending (USD)', 'Grocery': 'Grocery Spending (USD)'},
                     opacity=0.6, color_discrete_sequence=['purple'])
    return fig

def viz_9_box_plot(data, q99_val):
    """9. Box Plot - Distribution of Frozen Spending by Channel (Clipped)"""
    df_frozen_clipped = data.copy()
    df_frozen_clipped['Frozen'] = data['Frozen'].clip(upper=q99_val['Frozen'])

    # Fixed color scale 
    fig = px.box(df_frozen_clipped, x='Channel', y='Frozen', color='Channel',
                 title='Frozen Spending Distribution by Channel', 
                 labels={'Frozen': 'Frozen Spending (Capped at 99th Pctl)'},
                 color_discrete_sequence=px.colors.diverging.RdBu)
    fig.update_layout(xaxis_title='Channel', showlegend=False)
    return fig

def viz_10_violin_plot(data, q99_val):
    """10. Violin Plot - Distribution of Detergents/Paper Spending by Region (Clipped)"""
    df_detergents_clipped = data.copy()
    df_detergents_clipped['Detergents_Paper'] = df_detergents_clipped['Detergents_Paper'].clip(upper=q99_val['Detergents_Paper'])

    # Fixed color scale
    fig = px.violin(df_detergents_clipped, x='Region', y='Detergents_Paper', color='Region',
                    title='Detergents/Paper Spending Distribution by Region', 
                    labels={'Detergents_Paper': 'Detergents_Paper Spending (Capped at 99th Pctl)'},
                    box=True, points='outliers', color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(xaxis_title='Region', showlegend=False)
    return fig

def viz_11_heatmap(data):
    """11. Heatmap - Correlation Matrix of All Spending Categories"""
    corr_matrix = data[SPENDING_COLS].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto",
                    color_continuous_scale='cividis', 
                    title="Correlation Matrix of All Spending Categories")
    fig.update_xaxes(tickangle=45, side="top")
    return fig

def viz_12_variable_width_chart(data):
    """12. Vertical Bar Chart - Total Spending Magnitude by Category"""
    category_total_spending = data[SPENDING_COLS].sum().sort_values(ascending=False).reset_index()
    category_total_spending.columns = ['Category', 'Total Spending']
    
    fig = px.bar(category_total_spending, x='Category', y='Total Spending', color='Category',
                 title='Total Spending Magnitude by Category', 
                 labels={'Total Spending': 'Total Spending (USD)'},
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(xaxis_title='Category', yaxis_tickformat='.2s')
    return fig

def viz_19_bubble_chart(data, q99_val):
    """19. Bubble Chart - Relationship between Frozen and Delicatessen (Sized by Detergents/Paper)"""
    clipped_df = data.copy()
    for col in ['Frozen', 'Delicatessen', 'Detergents_Paper']:
        clipped_df[col] = clipped_df[col].clip(upper=q99_val[col])
    
    fig = px.scatter(clipped_df, x='Frozen', y='Delicatessen', size='Detergents_Paper', color='Region',
                     hover_name='Region', 
                     title='Frozen vs Delicatessen (Size by Detergents/Paper)',
                     labels={'Frozen': 'Frozen Spending (USD)', 'Delicatessen': 'Delicatessen Spending (USD)'},
                     size_max=40, opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set1)
    return fig


# --- CHART GROUPING FOR NEW PAGES ---
CHARTS_HOME = [
    (viz_3_pie_chart, df, None),
    (viz_4_concentric_donut, df, None),
    (viz_11_heatmap, df, None),
    (viz_12_variable_width_chart, df, None),
]

CHARTS_COMPARISON = [
    (viz_1_bar_vertical, channel_agg, None),
    (viz_17_bar_horizontal, region_agg, None),
    (viz_2_stacked_bar, channel_agg, None),
    (viz_18_grouped_bar_chart, channel_agg, None),
    (viz_14_overlaid_histogram, df, q99),
    (viz_9_box_plot, df, q99),
    (viz_10_violin_plot, df, q99),
]

CHARTS_RELATIONSHIP = [
    (viz_8_scatter, df, q99),
    (viz_19_bubble_chart, df, q99),
    (viz_20_double_donut, df, None),
    (viz_15_stacked_area, region_agg, None),
    (viz_16_marry_categories_line, multi_cat_df, None),
]

CHARTS_DISTRIBUTION_TREND = [
    (viz_5_bar_histogram, df, q99),
    (viz_6_line_histogram, df, q99),
    (viz_7_line_cumulative, df, None),
    (viz_13_time_series_line, df, None),
]


# --- PAGE RENDER FUNCTIONS ---

def render_home():
    """Renders the Home page with metrics and navigation."""
    st.header("Home: Dashboard Overview")
    st.markdown("This page provides *key business metrics* and *navigational controls* for the detailed analysis sections.")
    st.divider()

    # 1. Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Customers", value=f"{len(df):,}")
    with col2:
        total_spending = df['Total_Spending'].sum()
        st.metric(label="Total Recorded Spending (USD)", value=f"${total_spending:,.0f}")
    with col3:
        st.metric(label="Average Spending per Customer (USD)", value=f"${df['Total_Spending'].mean():,.0f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Navigation Buttons
    st.subheader("Explore the Dashboard Sections")
    col_nav_1, col_nav_2, col_nav_3 = st.columns(3)
    
    def nav_button(col, label, page):
        with col:
            if st.button(label, use_container_width=True, help=f"Go to {label} page"):
                st.session_state.page = page
                st.rerun()

    nav_button(col_nav_1, "📊 Segment Comparison", "Comparison")
    nav_button(col_nav_2, "🔗 Relationships & Proportion", "Relationship")
    nav_button(col_nav_3, "📈 Distribution & Trends", "Trend")
    
    st.divider()
    
    # 3. Main Overview Charts
    st.subheader("Top-Level Insights (Plotly Charts)")
    
    chart_cols = st.columns(2)
    
    # Use the safe_display_chart with a small delay for a subtle fade-in effect
    safe_display_chart(chart_cols[0], CHARTS_HOME[0][0], CHARTS_HOME[0][1], CHARTS_HOME[0][2], delay=0.01)
    safe_display_chart(chart_cols[1], CHARTS_HOME[1][0], CHARTS_HOME[1][1], CHARTS_HOME[1][2], delay=0.02)
    
    chart_cols = st.columns(2)
    safe_display_chart(chart_cols[0], CHARTS_HOME[2][0], CHARTS_HOME[2][1], CHARTS_HOME[2][2], delay=0.03)
    safe_display_chart(chart_cols[1], CHARTS_HOME[3][0], CHARTS_HOME[3][1], CHARTS_HOME[3][2], delay=0.04)
    

def render_comparison_page():
    """Renders the Segment Comparison page."""
    st.header("📊 Segment Comparison: Channel vs. Region Performance")
    st.markdown("Analyze how the two *Channels* (Horeca/Retail) and three *Regions* (Lisbon/Oporto/Other) compare across key spending categories.")
    st.divider()
    
    # Display charts with simulated animation
    # The 'motion' is achieved by the charts appearing sequentially via time.sleep
    with st.spinner('Loading Segment Comparison Charts with Motion Effect...'):
        time.sleep(0.5) # Initial large delay for the section load
        
        # Display charts in a 2-column layout
        for i in range(0, len(CHARTS_COMPARISON), 2):
            col1, col2 = st.columns(2)

            chart1 = CHARTS_COMPARISON[i]
            safe_display_chart(col1, chart1[0], chart1[1], chart1[2], delay=0.05)

            if i + 1 < len(CHARTS_COMPARISON):
                chart2 = CHARTS_COMPARISON[i+1]
                safe_display_chart(col2, chart2[0], chart2[1], chart2[2], delay=0.05)


def render_relationship_page():
    """Renders the Relationships and Proportion page."""
    st.header("🔗 Relationships & Proportion: Correlation and Segment Breakdown")
    st.markdown("Examine the *correlation* between product spending and the *proportional* breakdown of customers and average spending.")
    st.divider()
    
    with st.spinner('Loading Relationship and Proportion Charts with Motion Effect...'):
        time.sleep(0.5)
        
        for i in range(0, len(CHARTS_RELATIONSHIP), 2):
            col1, col2 = st.columns(2)

            chart1 = CHARTS_RELATIONSHIP[i]
            safe_display_chart(col1, chart1[0], chart1[1], chart1[2], delay=0.05)

            if i + 1 < len(CHARTS_RELATIONSHIP):
                chart2 = CHARTS_RELATIONSHIP[i+1]
                safe_display_chart(col2, chart2[0], chart2[1], chart2[2], delay=0.05)


def render_trend_page():
    """Renders the Distribution and Trends page."""
    st.header("📈 Distribution & Trends: Feature Spread and Cumulative Spending")
    st.markdown("View the *distribution* (spread and shape) of individual spending categories and *cumulative trends* across the customer base.")
    st.divider()
    
    with st.spinner('Loading Distribution and Trend Charts with Motion Effect...'):
        time.sleep(0.5)
        
        for i in range(0, len(CHARTS_DISTRIBUTION_TREND), 2):
            col1, col2 = st.columns(2)

            chart1 = CHARTS_DISTRIBUTION_TREND[i]
            safe_display_chart(col1, chart1[0], chart1[1], chart1[2], delay=0.05)

            if i + 1 < len(CHARTS_DISTRIBUTION_TREND):
                chart2 = CHARTS_DISTRIBUTION_TREND[i+1]
                safe_display_chart(col2, chart2[0], chart2[1], chart2[2], delay=0.05)

# --- DASHBOARD LAYOUT & NAVIGATION (Main execution logic) ---

# 1. Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# 2. Sidebar Navigation (FIXED Duplicate Key Error)
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    
    # FIX: We set the on_change callback to update the session state directly, 
    # preventing the duplicate key error on reruns.
    def set_page():
        st.session_state.page = st.session_state.sidebar_page_select_value

    page_options = ["Home", "Comparison", "Relationship", "Trend"]
    
    st.radio(
        "Select a Section",
        page_options,
        index=page_options.index(st.session_state.page),
        key='sidebar_page_select_value', # Unique key for the radio buttons
        on_change=set_page
    )
    
    st.markdown("---")
    st.subheader("Data Summary")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)
    st.markdown(f"*Total Customers:* {len(df)}")
    st.markdown("Developed with Streamlit and Plotly.")


# 3. Main Content Router
if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "Comparison":
    render_comparison_page()
elif st.session_state.page == "Relationship":
    render_relationship_page()
elif st.session_state.page == "Trend":
    render_trend_page()