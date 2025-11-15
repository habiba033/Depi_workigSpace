import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge

# --- CONFIGURATION AND DATA LOADING ---

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Wholesale Customer Segmentation Dashboard", page_icon="📊")

# Define spending columns for consistency
SPENDING_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

@st.cache_data
def load_data():
    """Loads, cleans, and prepares the customer data."""
    try:
        # Assuming the 'customers.csv' file is accessible
        df = pd.read_csv(r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_3\customers.csv')
    except FileNotFoundError:
        st.error("Error: 'customers.csv' not found. Please upload the file.")
        # Return a structure that can be unpacked but stops execution later
        return pd.DataFrame(), None, None, None, None 

    # --- Data Processing ---

    # Rename Channel and Region for better labels
    df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})
    df['Region'] = df['Region'].replace({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})

    # Calculate Total_Spending (Crucial for the fix)
    df['Total_Spending'] = df[SPENDING_COLS].sum(axis=1)

    # Calculate 99th percentile for clipping outliers in distributions
    q99 = df[SPENDING_COLS + ['Total_Spending']].quantile(0.99)

    # Aggregations for Channel and Region
    channel_agg = df.groupby('Channel')[SPENDING_COLS].mean().reset_index()
    region_agg = df.groupby('Region')[SPENDING_COLS].mean().reset_index()

    # Prepare data for Marry Categories Line Chart
    multi_cat_df = region_agg.set_index('Region')[SPENDING_COLS].T
    multi_cat_df.index.name = 'Category'

    return df, q99, channel_agg, region_agg, multi_cat_df

df, q99, channel_agg, region_agg, multi_cat_df = load_data()

# Check if data loaded successfully (df.empty handles FileNotFoundError case)
if df.empty or q99 is None:
    st.stop()

# --- 20 VISUALIZATION FUNCTIONS (using Matplotlib/Seaborn) ---

# Group 1: Comparison and Distribution (Vertical/Horizontal/Histogram)
def viz_1_bar_vertical(data):
    """1. Vertical Bar Chart - Avg Fresh Spending by Channel"""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x='Channel', y='Fresh', data=data, errorbar=None, palette='viridis', hue='Channel', legend=False, ax=ax)
    ax.set_title('1. Avg Fresh Spending by Channel')
    ax.set_ylabel('Average Fresh Spending (USD)')
    ax.set_xlabel('Customer Channel')
    return fig

def viz_5_bar_histogram(data, q99_val):
    """5. Bar Histogram - Distribution of Milk Spending"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['Milk'].clip(upper=q99_val['Milk']), bins=30, kde=False, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('5. Bar Histogram: Distribution of Milk Spending')
    ax.set_xlabel('Milk Spending (Capped at 99th Pctl)')
    ax.set_ylabel('Frequency (Count)')
    return fig

def viz_6_line_histogram(data, q99_val):
    """6. Line Histogram (Frequency Polygon) - Distribution of Grocery Spending"""
    fig, ax = plt.subplots(figsize=(8, 6))
    clipped_grocery = data['Grocery'].clip(upper=q99_val['Grocery'])
    hist, bins = np.histogram(clipped_grocery, bins=25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, hist, marker='o', linestyle='-', color='red', linewidth=2, markersize=5)
    ax.fill_between(bin_centers, hist, color='red', alpha=0.1)
    ax.set_title('6. Line Histogram: Distribution of Grocery Spending')
    ax.set_xlabel('Grocery Spending (Capped at 99th Pctl)')
    ax.set_ylabel('Frequency')
    return fig

def viz_17_bar_horizontal(agg_data):
    """17. Horizontal Bar Chart - Avg Delicatessen Spending by Region"""
    region_delicatessen_avg = agg_data[['Region', 'Delicatessen']].sort_values('Delicatessen', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x='Delicatessen', y='Region', data=region_delicatessen_avg, errorbar=None, palette='rocket', hue='Region', legend=False, ax=ax)
    ax.set_title('17. Horizontal Bar Chart: Avg Delicatessen Spending by Region')
    ax.set_xlabel('Average Delicatessen Spending (USD)')
    ax.set_ylabel('Region')
    return fig

def viz_14_overlaid_histogram(data, q99_val):
    """14. Overlaid Histogram - Fresh Spending Distribution by Channel"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=data, x='Fresh', hue='Channel', bins=30, kde=True,
                 stat='density', common_norm=False, palette='mako', alpha=0.5, ax=ax)
    ax.set_title('14. Overlaid Histogram: Fresh Spending by Channel')
    ax.set_xlabel('Fresh Spending (Capped at 99th Pctl)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, q99_val['Fresh'])
    return fig

# Group 2: Categorical and Proportional Charts (Pie/Donut/Stacked Bar)
def viz_2_stacked_bar(agg_data):
    """2. Stacked Bar Chart - Avg Spending on Key Categories by Channel"""
    temp_df = agg_data.set_index('Channel')[['Fresh', 'Milk', 'Grocery']]
    fig, ax = plt.subplots(figsize=(8, 6))
    temp_df.plot(kind='bar', stacked=True, colormap='plasma', ax=ax)
    ax.set_title('2. Stacked Bar: Avg Spending on Key Categories by Channel')
    ax.set_ylabel('Average Spending (USD)')
    ax.set_xlabel('Channel')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Category', loc='upper left')
    return fig

def viz_3_pie_chart(data):
    """3. Pie Chart - Customer Distribution by Channel"""
    channel_counts = data['Channel'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=90,
           colors=sns.color_palette('Pastel1'), wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
    ax.set_title('3. Pie Chart: Customer Distribution by Channel')
    ax.axis('equal')
    return fig

def viz_4_concentric_donut(data):
    """4. Concentric Donut Chart - Customer Distribution by Region"""
    region_counts = data['Region'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(region_counts, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                      colors=sns.color_palette('Set2'),
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'width': 0.3})
    centre_circle = plt.Circle((0,0), 0.55, fc='white', edgecolor='black', linewidth=1.5)
    ax.add_artist(centre_circle)
    ax.set_title('4. Concentric Donut Chart: Customer Distribution by Region')
    ax.legend(wedges, region_counts.index, title="Region", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')
    return fig

def viz_18_proportional_stacked_bar(agg_data):
    """18. Proportional Stacked Bar Chart - Category Share by Region"""
    temp_df = agg_data.set_index('Region')[SPENDING_COLS]
    temp_df_prop = temp_df.div(temp_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    temp_df_prop.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)
    ax.set_title('18. Proportional Stacked Bar: Category Share by Region')
    ax.set_ylabel('Percentage Share (%)')
    ax.set_xlabel('Region')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Category', loc='center left', bbox_to_anchor=(1.0, 0.5))
    return fig

def viz_20_double_donut(data):
    """20. Double Concentric Donut Chart - Region (Outer) and Channel (Inner)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    outer_counts = data['Region'].value_counts()
    outer_labels = outer_counts.index
    outer_colors = sns.color_palette('pastel', len(outer_counts))

    inner_counts = data['Channel'].value_counts()
    inner_labels = inner_counts.index
    inner_colors = sns.color_palette('Set1', len(inner_counts))

    ax.pie(outer_counts, radius=1.0, colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='white'))
    ax.pie(inner_counts, radius=0.6, colors=inner_colors, wedgeprops=dict(width=0.3, edgecolor='white'))

    legend_elements = [Wedge((0,0), 1.0, 0, 0, width=0.3, facecolor=c) for c in outer_colors] + \
                      [Wedge((0,0), 0.6, 0, 0, width=0.3, facecolor=c) for c in inner_colors]
    legend_labels = [f"Region: {l}" for l in outer_labels] + [f"Channel: {l}" for l in inner_labels]

    ax.legend(legend_elements, legend_labels, title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('20. Double Donut: Region (Outer) and Channel (Inner)')
    ax.axis('equal')
    return fig

# Group 3: Trend and Time Series (Line/Area/Cumulative)
def viz_7_line_cumulative(data):
    """7. Line Chart - Cumulative Sum of Fresh Spending (Sorted Index)"""
    temp_df = data.sort_values('Fresh').reset_index(drop=True)
    temp_df['Fresh_Cumulative'] = temp_df['Fresh'].cumsum()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(temp_df.index, temp_df['Fresh_Cumulative'], color='green', linewidth=2)
    ax.set_title('7. Line Chart: Cumulative Sum of Fresh Spending')
    ax.set_xlabel('Customer Index (Sorted by Fresh Spending)')
    ax.set_ylabel('Cumulative Fresh Spending (USD)')
    return fig

def viz_13_time_series_line(data):
    """13. Time Series Line Chart - Cumulative Fresh Spending Trend (Dashed)"""
    temp_df = data.sort_values('Fresh').reset_index(drop=True)
    temp_df['Fresh_Cumulative'] = temp_df['Fresh'].cumsum()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(temp_df.index, temp_df['Fresh_Cumulative'], color='darkblue', linestyle='--', marker='.', markersize=4, linewidth=1.5)
    ax.set_title('13. Time Series Line Chart: Cumulative Fresh Spending Trend')
    ax.set_xlabel('Customer ID (as Time/Sequence Index)')
    ax.set_ylabel('Cumulative Fresh Spending (USD)')
    return fig

def viz_15_stacked_area(agg_data):
    """15. Stacked Area Chart - Avg Key Spending by Region (Time interpretation)"""
    temp_agg = agg_data.set_index('Region')[['Fresh', 'Milk', 'Grocery']]
    fig, ax = plt.subplots(figsize=(9, 6))
    temp_agg.T.plot.area(stacked=True, alpha=0.6, colormap='Accent', ax=ax)
    ax.set_title('15. Stacked Area Chart: Avg Key Spending by Region')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Spending (USD)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Region', loc='upper right')
    return fig

def viz_16_marry_categories_line(multi_cat_data):
    """16. Marry Categories Line Chart - Spending Profile Across Regions (Style 1)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    multi_cat_data.plot(kind='line', style='-o', colormap='Dark2', linewidth=2, ax=ax)
    ax.set_title('16. Marry Categories Line Chart: Spending Profile Across Regions')
    ax.set_xlabel('Spending Category')
    ax.set_ylabel('Average Spending (USD)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Region')
    ax.grid(True, linestyle='--')
    return fig

def viz_19_unstacked_area(data):
    """19. Area Chart (Unstacked) - Cumulative Milk vs. Grocery Spending Comparison"""
    temp_df = data.sort_values('Total_Spending').reset_index(drop=True)
    temp_df['Milk_Cumulative'] = temp_df['Milk'].cumsum()
    temp_df['Grocery_Cumulative'] = temp_df['Grocery'].cumsum()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(temp_df.index, temp_df['Milk_Cumulative'], color='gold', alpha=0.3, label='Milk Cumulative')
    ax.fill_between(temp_df.index, temp_df['Grocery_Cumulative'], color='darkcyan', alpha=0.3, label='Grocery Cumulative')
    ax.plot(temp_df.index, temp_df['Milk_Cumulative'], color='gold', linewidth=1.5)
    ax.plot(temp_df.index, temp_df['Grocery_Cumulative'], color='darkcyan', linewidth=1.5)

    ax.set_title('19. Unstacked Area Chart: Cumulative Milk vs. Grocery Spending')
    ax.set_xlabel('Customer Index (Sorted by Total Spending)')
    ax.set_ylabel('Cumulative Spending (USD)')
    ax.legend(loc='upper left')
    return fig

# Group 4: Advanced Relationship and Distribution (Scatter/Box/Violin/Heatmap)
def viz_8_scatter(data, q99_val):
    """8. Scatter Plot - Milk Spending vs. Grocery Spending (Simple)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['Milk'].clip(upper=q99_val['Milk']),
               data['Grocery'].clip(upper=q99_val['Grocery']),
               s=50, alpha=0.6, color='purple')
    ax.set_title('8. Scatter Plot: Milk Spending vs. Grocery Spending')
    ax.set_xlabel('Milk Spending (USD)')
    ax.set_ylabel('Grocery Spending (USD)')
    return fig

def viz_9_box_plot(data, q99_val):
    """9. Box Plot - Distribution of Frozen Spending by Channel (Clipped)"""
    df_frozen_clipped = data.copy()
    df_frozen_clipped['Frozen'] = data['Frozen'].clip(upper=q99_val['Frozen'])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x='Channel', y='Frozen', data=df_frozen_clipped, palette='coolwarm', ax=ax)
    ax.set_title('9. Box Plot: Frozen Spending by Channel')
    ax.set_ylabel('Frozen Spending (Capped at 99th Pctl)')
    ax.set_xlabel('Channel')
    return fig

def viz_10_violin_plot(data, q99_val):
    """10. Violin Plot - Distribution of Detergents_Paper Spending by Region (Clipped)"""
    df_detergents_clipped = data.copy()
    df_detergents_clipped['Detergents_Paper'] = data['Detergents_Paper'].clip(upper=q99_val['Detergents_Paper'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='Region', y='Detergents_Paper', data=df_detergents_clipped, palette='Spectral', ax=ax)
    ax.set_title('10. Violin Plot: Detergents_Paper Spending by Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Detergents_Paper Spending (Capped at 99th Pctl)')
    return fig

def viz_11_heatmap(data):
    """11. Heatmap - Correlation Matrix of Spending Categories"""
    corr_matrix = data[SPENDING_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='cividis', fmt=".2f", linewidths=.5, linecolor='black',
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title('11. Heatmap: Correlation Matrix of Spending Categories')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    return fig

def viz_12_variable_width_chart(data):
    """12. Variable Width Chart - Total Spending by Category (Magnitude Plot)"""
    category_total_spending = data[SPENDING_COLS].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(category_total_spending.index, category_total_spending.values,
            color=sns.color_palette("tab10", len(category_total_spending)))
    ax.set_title('12. Variable Width Chart (Total Spending by Category)')
    ax.set_xlabel('Category')
    ax.set_ylabel('Total Spending (Millions USD)')
    ax.tick_params(axis='x', rotation=45)
    return fig

def viz_19_bubble_chart(data, q99_val):
    """19. Bubble Chart - Frozen vs Delicatessen (Size by Detergents_Paper)"""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Clip data for cleaner view
    clipped_df = data.copy()
    for col in ['Frozen', 'Delicatessen', 'Detergents_Paper']:
        clipped_df[col] = clipped_df[col].clip(upper=q99_val[col])

    # Convert Region to categorical codes for coloring
    region_codes = clipped_df['Region'].astype('category').cat.codes

    scatter = ax.scatter(x=clipped_df['Frozen'], y=clipped_df['Delicatessen'],
                         s=clipped_df['Detergents_Paper'] / 10, # Scale size for visualization
                         c=region_codes, cmap='Set1', alpha=0.6,
                         edgecolors='w', linewidths=0.5)

    ax.set_title('19. Bubble Chart: Frozen vs Delicatessen (Size by Detergents_Paper)')
    ax.set_xlabel('Frozen Spending (USD)')
    ax.set_ylabel('Delicatessen Spending (USD)')

    # Create legend for color (Region)
    unique_regions = clipped_df['Region'].unique()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=region,
                                 markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                      for i, region in enumerate(unique_regions)]
    ax.legend(handles=legend_handles, title="Region", loc="upper right")

    return fig

def viz_20_ecdf_plot(data, q99_val):
    """20. Empirical CDF Plot - Total Spending by Channel (Clipped)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.ecdfplot(data=data, x="Total_Spending", hue="Channel", palette="Set1", ax=ax)
    ax.set_title('20. Empirical CDF Plot: Total Spending by Channel')
    ax.set_xlabel('Total Spending (Capped at 99th Pctl)')
    ax.set_ylabel('Proportion')
    ax.set_xlim(0, q99_val['Total_Spending'])
    return fig

def viz_18_grouped_bar_chart(agg_data):
    """18. Grouped Bar Chart - Avg Milk and Grocery Spending by Channel"""
    temp_df = channel_agg.set_index('Channel')[['Milk', 'Grocery']]
    fig, ax = plt.subplots(figsize=(8, 6))
    temp_df.plot(kind='bar', color=['sandybrown', 'darkkhaki'], ax=ax)
    ax.set_title('18. Grouped Bar Chart: Avg Milk and Grocery Spending by Channel')
    ax.set_ylabel('Average Spending (USD)')
    ax.set_xlabel('Channel')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Category')
    return fig


# List of all visualization functions for display
VIZ_FUNCTIONS = [
    (viz_1_bar_vertical, channel_agg, "Distribution & Comparison"),
    (viz_2_stacked_bar, channel_agg, "Categorical & Proportional"),
    (viz_3_pie_chart, df, "Categorical & Proportional"),
    (viz_4_concentric_donut, df, "Categorical & Proportional"),
    (viz_5_bar_histogram, df, "Distribution & Comparison"),
    (viz_6_line_histogram, df, "Distribution & Comparison"),
    (viz_7_line_cumulative, df, "Trend & Time Series"),
    (viz_8_scatter, df, "Advanced Relationship & Distribution"),
    (viz_9_box_plot, df, "Advanced Relationship & Distribution"),
    (viz_10_violin_plot, df, "Advanced Relationship & Distribution"),
    (viz_11_heatmap, df, "Advanced Relationship & Distribution"),
    (viz_12_variable_width_chart, df, "Advanced Relationship & Distribution"),
    (viz_13_time_series_line, df, "Trend & Time Series"),
    (viz_14_overlaid_histogram, df, "Distribution & Comparison"),
    (viz_15_stacked_area, region_agg, "Trend & Time Series"),
    (viz_16_marry_categories_line, multi_cat_df, "Trend & Time Series"),
    (viz_17_bar_horizontal, region_agg, "Distribution & Comparison"),
    (viz_18_grouped_bar_chart, channel_agg, "Categorical & Proportional"),
    (viz_19_bubble_chart, df, "Advanced Relationship & Distribution"),
    (viz_20_double_donut, df, "Categorical & Proportional"),
]

# Pass q99 dictionary to all functions that need it (identified via inspection)
for i in range(len(VIZ_FUNCTIONS)):
    func, data, group = VIZ_FUNCTIONS[i]
    if func in [viz_5_bar_histogram, viz_6_line_histogram, viz_8_scatter, viz_9_box_plot, viz_10_violin_plot, viz_14_overlaid_histogram, viz_19_bubble_chart, viz_20_ecdf_plot]:
        VIZ_FUNCTIONS[i] = (func, (data, q99), group)


# --- DASHBOARD LAYOUT ---

st.title("📊 20-Chart Wholesale Customer Segmentation Analysis")
st.markdown("This dashboard presents *20 unique visualizations* covering customer spending distribution, segment comparisons, and internal feature correlations.")

# Sidebar for Filtering and Information
with st.sidebar:
    st.header("Dashboard Controls")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)
    st.markdown(f"*Total Customers:* {len(df)}")

    # FIX: Check for the 'Total_Spending' column before summing to avoid KeyError
    if 'Total_Spending' in df.columns:
        total_spending = df['Total_Spending'].sum()
        st.markdown(f"*Total Spending:* ${total_spending:,.0f}")
    else:
        # Fallback if the column is missing
        st.markdown("*Total Spending:* N/A (Data processing incomplete)")


    # Grouping logic for filter
    chart_groups = sorted(list(set(group for _, _, group in VIZ_FUNCTIONS)))
    selected_group = st.selectbox(
        "Filter Charts by Type:",
        ["All Charts"] + chart_groups
    )

# Main Content Area
if selected_group == "All Charts":
    charts_to_display = VIZ_FUNCTIONS
else:
    charts_to_display = [v for v in VIZ_FUNCTIONS if v[2] == selected_group]

st.subheader(f"Charts ({selected_group})")
st.divider()

# Display charts in a 2-column layout
for i in range(0, len(charts_to_display), 2):
    col1, col2 = st.columns(2)

    # Function to safely display a chart
    def safe_display_chart(col, chart_tuple):
        func, data_or_tuple, _ = chart_tuple
        # The docstring check is robust now as all function definitions are consistent.
        title = func.__doc__.split(' - ')[1].strip()

        with col:
            with st.expander(f"{title}", expanded=True):
                try:
                    # Handle functions that take tuple of (data, q99)
                    if isinstance(data_or_tuple, tuple):
                        fig = func(*data_or_tuple)
                    else:
                        fig = func(data_or_tuple)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating {func.__name__}: {e}")

    # Process first chart in the pair
    safe_display_chart(col1, charts_to_display[i])

    # Process second chart in the pair (if exists)
    if i + 1 < len(charts_to_display):
        safe_display_chart(col2, charts_to_display[i+1])

# Footer (optional)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed using Streamlit, Matplotlib, and Seaborn.")