import csv
import math
import statistics
import random
from datetime import datetime, timedelta
from collections import defaultdict
import os

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Wedge, Rectangle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib available - creating comprehensive visualizations")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ Matplotlib not available - creating text-based analysis")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    print("✓ Seaborn available - enhanced styling")
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠ Seaborn not available - using basic matplotlib")

class HousingDataProcessor:
    """Data preprocessing and feature engineering for housing analysis"""
    
    def __init__(self, data_file='C:/ami_ai/masine/housing.csv'):
        self.data_file = 'C:/ami_ai/masine/housing.csv'
        self.raw_data = []
        self.processed_data = []
        self.load_data()
        self.preprocess_data()
    
    def load_data(self):
        """Load raw housing data from CSV file"""
        try:
            with open(self.data_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.raw_data.append({
                        'RM': float(row['RM']),
                        'LSTAT': float(row['LSTAT']),
                        'PTRATIO': float(row['PTRATIO']),
                        'MEDV': float(row['MEDV'])
                    })
            print(f"✓ Loaded {len(self.raw_data)} raw housing records")
        except FileNotFoundError:
            print(f"❌ Error: {self.data_file} not found!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_data(self):
        """Preprocess and engineer features for comprehensive analysis"""
        print("🔧 Preprocessing data and engineering features...")
        
        for i, row in enumerate(self.raw_data):
            # Create synthetic time series data (simulating monthly data over 4 years)
            base_date = datetime(2020, 1, 1)
            month_offset = i % 48  # 48 months = 4 years
            current_date = base_date + timedelta(days=month_offset * 30)
            
            # Create price categories
            if row['MEDV'] < 300000:
                price_category = 'Budget'
            elif row['MEDV'] < 500000:
                price_category = 'Mid-Range'
            elif row['MEDV'] < 700000:
                price_category = 'Premium'
            else:
                price_category = 'Luxury'
            
            # Create room categories
            if row['RM'] < 5:
                room_category = 'Small'
            elif row['RM'] < 7:
                room_category = 'Medium'
            else:
                room_category = 'Large'
            
            # Create socioeconomic categories
            if row['LSTAT'] < 10:
                socio_category = 'High Income'
            elif row['LSTAT'] < 20:
                socio_category = 'Middle Income'
            else:
                socio_category = 'Low Income'
            
            # Create school quality categories
            if row['PTRATIO'] < 15:
                school_category = 'Excellent'
            elif row['PTRATIO'] < 20:
                school_category = 'Good'
            else:
                school_category = 'Average'
            
            # Add synthetic market trends (price appreciation)
            market_trend = 1 + (0.02 * month_offset / 12)  # 2% annual appreciation
            adjusted_price = row['MEDV'] * market_trend
            
            # Add seasonal variation
            seasonal_factor = 1 + 0.1 * math.sin(2 * math.pi * month_offset / 12)
            seasonal_price = adjusted_price * seasonal_factor
            
            processed_row = {
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
                'Adjusted_Price': adjusted_price,
                'Seasonal_Price': seasonal_price,
                'Price_Per_Room': row['MEDV'] / row['RM'],
                'Value_Score': (row['RM'] * 0.3) + ((100 - row['LSTAT']) * 0.4) + ((25 - row['PTRATIO']) * 0.3),
                'Market_Index': i / len(self.raw_data) * 100
            }
            
            self.processed_data.append(processed_row)
        
        print(f"✓ Processed {len(self.processed_data)} records with engineered features")
        print("  - Added time series data (4 years)")
        print("  - Created categorical variables")
        print("  - Added market trends and seasonal variations")
        print("  - Calculated derived metrics")
    
    def get_column_data(self, column):
        """Extract data for a specific column"""
        return [row[column] for row in self.processed_data]
    
    def get_data_by_category(self, category_col, value_col):
        """Get data grouped by category"""
        categories = defaultdict(list)
        for row in self.processed_data:
            categories[row[category_col]].append(row[value_col])
        return dict(categories)

class ComprehensiveVisualizer:
    """Create comprehensive visualizations for housing data"""
    
    def __init__(self, processor):
        self.processor = processor
        self.data = processor.processed_data
        self.setup_styling()
    
    def setup_styling(self):
        """Set up professional styling for all visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#17A2B8',
            'warning': '#FFC107',
            'light': '#F8F9FA',
            'dark': '#343A40',
            'text': '#2C3E50'
        }
        
        # Color schemes for different chart types
        self.color_schemes = {
            'categorical': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#17A2B8', '#6F42C1'],
            'sequential': ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3'],
            'diverging': ['#D32F2F', '#F57C00', '#FBC02D', '#689F38', '#388E3C', '#1976D2']
        }
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': self.colors['dark'],
            'axes.linewidth': 1.2,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_line_charts(self):
        """Create various line charts"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Line Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Price trends over time
        dates = self.processor.get_column_data('Date')
        prices = self.processor.get_column_data('Seasonal_Price')
        
        axes[0, 0].plot(dates, prices, color=self.colors['primary'], linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Housing Price Trends Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Room count vs Price relationship
        rm_data = self.processor.get_column_data('RM')
        medv_data = self.processor.get_column_data('MEDV')
        
        # Sort by RM for smooth line
        sorted_data = sorted(zip(rm_data, medv_data))
        rm_sorted, medv_sorted = zip(*sorted_data)
        
        axes[0, 1].plot(rm_sorted, medv_sorted, color=self.colors['secondary'], 
                       linewidth=2, marker='o', markersize=4, alpha=0.7)
        axes[0, 1].set_title('Price vs Room Count Relationship', fontweight='bold')
        axes[0, 1].set_xlabel('Average Number of Rooms')
        axes[0, 1].set_ylabel('Median Home Value ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. LSTAT vs Price relationship
        lstat_data = self.processor.get_column_data('LSTAT')
        
        sorted_lstat = sorted(zip(lstat_data, medv_data))
        lstat_sorted, medv_sorted2 = zip(*sorted_lstat)
        
        axes[1, 0].plot(lstat_sorted, medv_sorted2, color=self.colors['accent'], 
                       linewidth=2, marker='s', markersize=4, alpha=0.7)
        axes[1, 0].set_title('Price vs Lower Status Population', fontweight='bold')
        axes[1, 0].set_xlabel('Lower Status Population (%)')
        axes[1, 0].set_ylabel('Median Home Value ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Value Score over Market Index
        value_scores = self.processor.get_column_data('Value_Score')
        market_index = self.processor.get_column_data('Market_Index')
        
        axes[1, 1].plot(market_index, value_scores, color=self.colors['success'], 
                       linewidth=2, marker='^', markersize=4, alpha=0.7)
        axes[1, 1].set_title('Value Score Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Market Index')
        axes[1, 1].set_ylabel('Value Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('line_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_bar_charts(self):
        """Create vertical bar charts and histograms"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bar Charts and Histograms Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Price categories bar chart
        price_categories = self.processor.get_data_by_category('Price_Category', 'MEDV')
        categories = list(price_categories.keys())
        avg_prices = [statistics.mean(prices) for prices in price_categories.values()]
        
        bars = axes[0, 0].bar(categories, avg_prices, color=self.color_schemes['categorical'][:len(categories)])
        axes[0, 0].set_title('Average Price by Category', fontweight='bold')
        axes[0, 0].set_xlabel('Price Category')
        axes[0, 0].set_ylabel('Average Price ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_prices):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                           f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Room categories histogram
        room_categories = self.processor.get_data_by_category('Room_Category', 'MEDV')
        room_cats = list(room_categories.keys())
        room_counts = [len(prices) for prices in room_categories.values()]
        
        bars2 = axes[0, 1].bar(room_cats, room_counts, color=self.color_schemes['categorical'][:len(room_cats)])
        axes[0, 1].set_title('Property Count by Room Category', fontweight='bold')
        axes[0, 1].set_xlabel('Room Category')
        axes[0, 1].set_ylabel('Number of Properties')
        
        for bar, value in zip(bars2, room_counts):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Price distribution histogram
        medv_data = self.processor.get_column_data('MEDV')
        axes[1, 0].hist(medv_data, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='white')
        axes[1, 0].axvline(statistics.mean(medv_data), color=self.colors['accent'], 
                          linestyle='--', linewidth=2, label=f'Mean: ${statistics.mean(medv_data):,.0f}')
        axes[1, 0].set_title('Price Distribution Histogram', fontweight='bold')
        axes[1, 0].set_xlabel('Median Home Value ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Quarterly analysis
        quarterly_data = self.processor.get_data_by_category('Quarter', 'MEDV')
        quarters = sorted(quarterly_data.keys())
        quarterly_avg = [statistics.mean(prices) for prices in [quarterly_data[q] for q in quarters]]
        
        bars3 = axes[1, 1].bar([f'Q{q}' for q in quarters], quarterly_avg, 
                              color=self.color_schemes['categorical'][:len(quarters)])
        axes[1, 1].set_title('Average Price by Quarter', fontweight='bold')
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Average Price ($)')
        
        for bar, value in zip(bars3, quarterly_avg):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                           f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('bar_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_pie_charts(self):
        """Create pie charts for categorical analysis"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pie Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Price categories pie chart
        price_categories = self.processor.get_data_by_category('Price_Category', 'MEDV')
        categories = list(price_categories.keys())
        counts = [len(prices) for prices in price_categories.values()]
        
        wedges, texts, autotexts = axes[0, 0].pie(counts, labels=categories, autopct='%1.1f%%',
                                                 colors=self.color_schemes['categorical'][:len(categories)],
                                                 startangle=90)
        axes[0, 0].set_title('Property Distribution by Price Category', fontweight='bold')
        
        # 2. Room categories pie chart
        room_categories = self.processor.get_data_by_category('Room_Category', 'MEDV')
        room_cats = list(room_categories.keys())
        room_counts = [len(prices) for prices in room_categories.values()]
        
        wedges2, texts2, autotexts2 = axes[0, 1].pie(room_counts, labels=room_cats, autopct='%1.1f%%',
                                                    colors=self.color_schemes['categorical'][:len(room_cats)],
                                                    startangle=90)
        axes[0, 1].set_title('Property Distribution by Room Category', fontweight='bold')
        
        # 3. Socioeconomic categories pie chart
        socio_categories = self.processor.get_data_by_category('Socio_Category', 'MEDV')
        socio_cats = list(socio_categories.keys())
        socio_counts = [len(prices) for prices in socio_categories.values()]
        
        wedges3, texts3, autotexts3 = axes[1, 0].pie(socio_counts, labels=socio_cats, autopct='%1.1f%%',
                                                    colors=self.color_schemes['categorical'][:len(socio_cats)],
                                                    startangle=90)
        axes[1, 0].set_title('Property Distribution by Socioeconomic Category', fontweight='bold')
        
        # 4. School quality categories pie chart
        school_categories = self.processor.get_data_by_category('School_Category', 'MEDV')
        school_cats = list(school_categories.keys())
        school_counts = [len(prices) for prices in school_categories.values()]
        
        wedges4, texts4, autotexts4 = axes[1, 1].pie(school_counts, labels=school_cats, autopct='%1.1f%%',
                                                    colors=self.color_schemes['categorical'][:len(school_cats)],
                                                    startangle=90)
        axes[1, 1].set_title('Property Distribution by School Quality', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pie_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_variable_width_charts(self):
        """Create variable width charts"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variable Width Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Variable width bar chart based on room count
        room_categories = self.processor.get_data_by_category('Room_Category', 'MEDV')
        categories = list(room_categories.keys())
        avg_prices = [statistics.mean(prices) for prices in room_categories.values()]
        counts = [len(prices) for prices in room_categories.values()]
        
        # Width proportional to count
        max_count = max(counts)
        widths = [count / max_count * 0.8 for count in counts]
        
        x_pos = np.arange(len(categories))
        bars = axes[0, 0].bar(x_pos, avg_prices, width=widths, 
                             color=self.color_schemes['categorical'][:len(categories)])
        axes[0, 0].set_title('Average Price by Room Category\\n(Width = Sample Size)', fontweight='bold')
        axes[0, 0].set_xlabel('Room Category')
        axes[0, 0].set_ylabel('Average Price ($)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(categories)
        
        # 2. Variable width based on price range
        price_ranges = [(0, 300000), (300000, 500000), (500000, 700000), (700000, float('inf'))]
        range_labels = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
        range_data = []
        
        for min_price, max_price in price_ranges:
            if max_price == float('inf'):
                matching = [row for row in self.data if row['MEDV'] >= min_price]
            else:
                matching = [row for row in self.data if min_price <= row['MEDV'] < max_price]
            range_data.append(matching)
        
        range_avg_prices = [statistics.mean([row['MEDV'] for row in data]) if data else 0 for data in range_data]
        range_counts = [len(data) for data in range_data]
        
        widths2 = [count / max(range_counts) * 0.8 for count in range_counts]
        x_pos2 = np.arange(len(range_labels))
        
        bars2 = axes[0, 1].bar(x_pos2, range_avg_prices, width=widths2,
                              color=self.color_schemes['categorical'][:len(range_labels)])
        axes[0, 1].set_title('Average Price by Price Range\\n(Width = Sample Size)', fontweight='bold')
        axes[0, 1].set_xlabel('Price Range')
        axes[0, 1].set_ylabel('Average Price ($)')
        axes[0, 1].set_xticks(x_pos2)
        axes[0, 1].set_xticklabels(range_labels)
        
        # 3. Variable width based on LSTAT ranges
        lstat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
        lstat_labels = ['0-10%', '10-20%', '20-30%', '30-40%']
        lstat_data = []
        
        for min_lstat, max_lstat in lstat_ranges:
            matching = [row for row in self.data if min_lstat <= row['LSTAT'] < max_lstat]
            lstat_data.append(matching)
        
        lstat_avg_prices = [statistics.mean([row['MEDV'] for row in data]) if data else 0 for data in lstat_data]
        lstat_counts = [len(data) for data in lstat_data]
        
        widths3 = [count / max(lstat_counts) * 0.8 for count in lstat_counts]
        x_pos3 = np.arange(len(lstat_labels))
        
        bars3 = axes[1, 0].bar(x_pos3, lstat_avg_prices, width=widths3,
                              color=self.color_schemes['categorical'][:len(lstat_labels)])
        axes[1, 0].set_title('Average Price by LSTAT Range\\n(Width = Sample Size)', fontweight='bold')
        axes[1, 0].set_xlabel('LSTAT Range (%)')
        axes[1, 0].set_ylabel('Average Price ($)')
        axes[1, 0].set_xticks(x_pos3)
        axes[1, 0].set_xticklabels(lstat_labels)
        
        # 4. Variable width based on PTRATIO ranges
        ptratio_ranges = [(10, 15), (15, 20), (20, 25)]
        ptratio_labels = ['10-15', '15-20', '20-25']
        ptratio_data = []
        
        for min_ptratio, max_ptratio in ptratio_ranges:
            matching = [row for row in self.data if min_ptratio <= row['PTRATIO'] < max_ptratio]
            ptratio_data.append(matching)
        
        ptratio_avg_prices = [statistics.mean([row['MEDV'] for row in data]) if data else 0 for data in ptratio_data]
        ptratio_counts = [len(data) for data in ptratio_data]
        
        if ptratio_counts:
            widths4 = [count / max(ptratio_counts) * 0.8 for count in ptratio_counts]
            x_pos4 = np.arange(len(ptratio_labels))
            
            bars4 = axes[1, 1].bar(x_pos4, ptratio_avg_prices, width=widths4,
                                  color=self.color_schemes['categorical'][:len(ptratio_labels)])
            axes[1, 1].set_title('Average Price by PTRATIO Range\\n(Width = Sample Size)', fontweight='bold')
            axes[1, 1].set_xlabel('PTRATIO Range')
            axes[1, 1].set_ylabel('Average Price ($)')
            axes[1, 1].set_xticks(x_pos4)
            axes[1, 1].set_xticklabels(ptratio_labels)
        
        plt.tight_layout()
        plt.savefig('variable_width_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_time_series_charts(self):
        """Create time series line charts"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # Group data by month
        monthly_data = defaultdict(list)
        for row in self.data:
            month_key = f"{row['Year']}-{row['Month']:02d}"
            monthly_data[month_key].append(row)
        
        # Sort months
        sorted_months = sorted(monthly_data.keys())
        
        # 1. Average price over time
        monthly_avg_prices = []
        for month in sorted_months:
            prices = [row['MEDV'] for row in monthly_data[month]]
            monthly_avg_prices.append(statistics.mean(prices))
        
        axes[0, 0].plot(range(len(sorted_months)), monthly_avg_prices, 
                       color=self.colors['primary'], linewidth=2, marker='o')
        axes[0, 0].set_title('Average Price Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Price ($)')
        axes[0, 0].set_xticks(range(0, len(sorted_months), 6))
        axes[0, 0].set_xticklabels([sorted_months[i] for i in range(0, len(sorted_months), 6)], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Price trends by category
        price_categories = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
        for i, category in enumerate(price_categories):
            category_prices = []
            for month in sorted_months:
                category_data = [row for row in monthly_data[month] if row['Price_Category'] == category]
                if category_data:
                    prices = [row['MEDV'] for row in category_data]
                    category_prices.append(statistics.mean(prices))
                else:
                    category_prices.append(0)
            
            if any(category_prices):
                axes[0, 1].plot(range(len(sorted_months)), category_prices, 
                               color=self.color_schemes['categorical'][i], linewidth=2, 
                               marker='o', label=category, alpha=0.8)
        
        axes[0, 1].set_title('Price Trends by Category', fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Price ($)')
        axes[0, 1].set_xticks(range(0, len(sorted_months), 6))
        axes[0, 1].set_xticklabels([sorted_months[i] for i in range(0, len(sorted_months), 6)], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Room count trends
        monthly_avg_rooms = []
        for month in sorted_months:
            rooms = [row['RM'] for row in monthly_data[month]]
            monthly_avg_rooms.append(statistics.mean(rooms))
        
        axes[1, 0].plot(range(len(sorted_months)), monthly_avg_rooms, 
                       color=self.colors['secondary'], linewidth=2, marker='s')
        axes[1, 0].set_title('Average Room Count Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Room Count')
        axes[1, 0].set_xticks(range(0, len(sorted_months), 6))
        axes[1, 0].set_xticklabels([sorted_months[i] for i in range(0, len(sorted_months), 6)], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. LSTAT trends
        monthly_avg_lstat = []
        for month in sorted_months:
            lstat_values = [row['LSTAT'] for row in monthly_data[month]]
            monthly_avg_lstat.append(statistics.mean(lstat_values))
        
        axes[1, 1].plot(range(len(sorted_months)), monthly_avg_lstat, 
                       color=self.colors['accent'], linewidth=2, marker='^')
        axes[1, 1].set_title('Average LSTAT Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average LSTAT (%)')
        axes[1, 1].set_xticks(range(0, len(sorted_months), 6))
        axes[1, 1].set_xticklabels([sorted_months[i] for i in range(0, len(sorted_months), 6)], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_series_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_stacked_bar_charts(self):
        """Create stacked bar charts"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stacked Bar Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Price categories by room category
        room_categories = list(set([row['Room_Category'] for row in self.data]))
        price_categories = list(set([row['Price_Category'] for row in self.data]))
        
        # Create stacked data
        stacked_data = {}
        for room_cat in room_categories:
            stacked_data[room_cat] = {}
            for price_cat in price_categories:
                count = len([row for row in self.data if row['Room_Category'] == room_cat and row['Price_Category'] == price_cat])
                stacked_data[room_cat][price_cat] = count
        
        # Plot stacked bars
        bottom = np.zeros(len(room_categories))
        for i, price_cat in enumerate(price_categories):
            values = [stacked_data[room_cat][price_cat] for room_cat in room_categories]
            axes[0, 0].bar(room_categories, values, bottom=bottom, 
                          label=price_cat, color=self.color_schemes['categorical'][i])
            bottom += values
        
        axes[0, 0].set_title('Price Categories by Room Category', fontweight='bold')
        axes[0, 0].set_xlabel('Room Category')
        axes[0, 0].set_ylabel('Number of Properties')
        axes[0, 0].legend()
        
        # 2. Socioeconomic categories by price category
        socio_categories = list(set([row['Socio_Category'] for row in self.data]))
        
        stacked_data2 = {}
        for price_cat in price_categories:
            stacked_data2[price_cat] = {}
            for socio_cat in socio_categories:
                count = len([row for row in self.data if row['Price_Category'] == price_cat and row['Socio_Category'] == socio_cat])
                stacked_data2[price_cat][socio_cat] = count
        
        bottom2 = np.zeros(len(price_categories))
        for i, socio_cat in enumerate(socio_categories):
            values = [stacked_data2[price_cat][socio_cat] for price_cat in price_categories]
            axes[0, 1].bar(price_categories, values, bottom=bottom2, 
                          label=socio_cat, color=self.color_schemes['categorical'][i])
            bottom2 += values
        
        axes[0, 1].set_title('Socioeconomic Categories by Price Category', fontweight='bold')
        axes[0, 1].set_xlabel('Price Category')
        axes[0, 1].set_ylabel('Number of Properties')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. School quality by room category
        school_categories = list(set([row['School_Category'] for row in self.data]))
        
        stacked_data3 = {}
        for room_cat in room_categories:
            stacked_data3[room_cat] = {}
            for school_cat in school_categories:
                count = len([row for row in self.data if row['Room_Category'] == room_cat and row['School_Category'] == school_cat])
                stacked_data3[room_cat][school_cat] = count
        
        bottom3 = np.zeros(len(room_categories))
        for i, school_cat in enumerate(school_categories):
            values = [stacked_data3[room_cat][school_cat] for room_cat in room_categories]
            axes[1, 0].bar(room_categories, values, bottom=bottom3, 
                          label=school_cat, color=self.color_schemes['categorical'][i])
            bottom3 += values
        
        axes[1, 0].set_title('School Quality by Room Category', fontweight='bold')
        axes[1, 0].set_xlabel('Room Category')
        axes[1, 0].set_ylabel('Number of Properties')
        axes[1, 0].legend()
        
        # 4. Quarterly analysis by price category
        quarters = sorted(list(set([row['Quarter'] for row in self.data])))
        
        stacked_data4 = {}
        for quarter in quarters:
            stacked_data4[quarter] = {}
            for price_cat in price_categories:
                count = len([row for row in self.data if row['Quarter'] == quarter and row['Price_Category'] == price_cat])
                stacked_data4[quarter][price_cat] = count
        
        bottom4 = np.zeros(len(quarters))
        for i, price_cat in enumerate(price_categories):
            values = [stacked_data4[quarter][price_cat] for quarter in quarters]
            axes[1, 1].bar([f'Q{q}' for q in quarters], values, bottom=bottom4, 
                          label=price_cat, color=self.color_schemes['categorical'][i])
            bottom4 += values
        
        axes[1, 1].set_title('Price Categories by Quarter', fontweight='bold')
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Number of Properties')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('stacked_bar_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_concentric_donut_charts(self):
        """Create concentric donut charts"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Concentric Donut Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # 1. Price categories (outer) and room categories (inner)
        price_categories = self.processor.get_data_by_category('Price_Category', 'MEDV')
        room_categories = self.processor.get_data_by_category('Room_Category', 'MEDV')
        
        # Outer ring - Price categories
        price_counts = [len(prices) for prices in price_categories.values()]
        price_labels = list(price_categories.keys())
        
        # Inner ring - Room categories
        room_counts = [len(prices) for prices in room_categories.values()]
        room_labels = list(room_categories.keys())
        
        # Create concentric donut
        wedges1, texts1 = axes[0, 0].pie(price_counts, labels=price_labels, 
                                        colors=self.color_schemes['categorical'][:len(price_counts)],
                                        radius=1, startangle=90)
        
        wedges2, texts2 = axes[0, 0].pie(room_counts, labels=room_labels,
                                        colors=self.color_schemes['categorical'][:len(room_counts)],
                                        radius=0.6, startangle=90)
        
        # Create donut effect
        centre_circle = plt.Circle((0, 0), 0.3, fc='white')
        axes[0, 0].add_artist(centre_circle)
        axes[0, 0].set_title('Price Categories (Outer) vs Room Categories (Inner)', fontweight='bold')
        
        # 2. Socioeconomic categories (outer) and school categories (inner)
        socio_categories = self.processor.get_data_by_category('Socio_Category', 'MEDV')
        school_categories = self.processor.get_data_by_category('School_Category', 'MEDV')
        
        socio_counts = [len(prices) for prices in socio_categories.values()]
        socio_labels = list(socio_categories.keys())
        
        school_counts = [len(prices) for prices in school_categories.values()]
        school_labels = list(school_categories.keys())
        
        wedges3, texts3 = axes[0, 1].pie(socio_counts, labels=socio_labels,
                                        colors=self.color_schemes['categorical'][:len(socio_counts)],
                                        radius=1, startangle=90)
        
        wedges4, texts4 = axes[0, 1].pie(school_counts, labels=school_labels,
                                        colors=self.color_schemes['categorical'][:len(school_counts)],
                                        radius=0.6, startangle=90)
        
        centre_circle2 = plt.Circle((0, 0), 0.3, fc='white')
        axes[0, 1].add_artist(centre_circle2)
        axes[0, 1].set_title('Socioeconomic (Outer) vs School Quality (Inner)', fontweight='bold')
        
        # 3. Price ranges (outer) and quarters (inner)
        price_ranges = [(0, 300000), (300000, 500000), (500000, 700000), (700000, float('inf'))]
        range_labels = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
        range_counts = []
        
        for min_price, max_price in price_ranges:
            if max_price == float('inf'):
                count = len([row for row in self.data if row['MEDV'] >= min_price])
            else:
                count = len([row for row in self.data if min_price <= row['MEDV'] < max_price])
            range_counts.append(count)
        
        quarterly_data = self.processor.get_data_by_category('Quarter', 'MEDV')
        quarter_counts = [len(prices) for prices in quarterly_data.values()]
        quarter_labels = [f'Q{q}' for q in sorted(quarterly_data.keys())]
        
        wedges5, texts5 = axes[1, 0].pie(range_counts, labels=range_labels,
                                        colors=self.color_schemes['categorical'][:len(range_counts)],
                                        radius=1, startangle=90)
        
        wedges6, texts6 = axes[1, 0].pie(quarter_counts, labels=quarter_labels,
                                        colors=self.color_schemes['categorical'][:len(quarter_counts)],
                                        radius=0.6, startangle=90)
        
        centre_circle3 = plt.Circle((0, 0), 0.3, fc='white')
        axes[1, 0].add_artist(centre_circle3)
        axes[1, 0].set_title('Price Ranges (Outer) vs Quarters (Inner)', fontweight='bold')
        
        # 4. LSTAT ranges (outer) and PTRATIO ranges (inner)
        lstat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
        lstat_range_labels = ['0-10%', '10-20%', '20-30%', '30-40%']
        lstat_range_counts = []
        
        for min_lstat, max_lstat in lstat_ranges:
            count = len([row for row in self.data if min_lstat <= row['LSTAT'] < max_lstat])
            lstat_range_counts.append(count)
        
        ptratio_ranges = [(10, 15), (15, 20), (20, 25)]
        ptratio_range_labels = ['10-15', '15-20', '20-25']
        ptratio_range_counts = []
        
        for min_ptratio, max_ptratio in ptratio_ranges:
            count = len([row for row in self.data if min_ptratio <= row['PTRATIO'] < max_ptratio])
            ptratio_range_counts.append(count)
        
        wedges7, texts7 = axes[1, 1].pie(lstat_range_counts, labels=lstat_range_labels,
                                        colors=self.color_schemes['categorical'][:len(lstat_range_counts)],
                                        radius=1, startangle=90)
        
        wedges8, texts8 = axes[1, 1].pie(ptratio_range_counts, labels=ptratio_range_labels,
                                        colors=self.color_schemes['categorical'][:len(ptratio_range_counts)],
                                        radius=0.6, startangle=90)
        
        centre_circle4 = plt.Circle((0, 0), 0.3, fc='white')
        axes[1, 1].add_artist(centre_circle4)
        axes[1, 1].set_title('LSTAT Ranges (Outer) vs PTRATIO Ranges (Inner)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('concentric_donut_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        print("📊 Creating comprehensive dashboard...")
        
        # Create all visualizations
        self.create_line_charts()
        self.create_bar_charts()
        self.create_pie_charts()
        self.create_variable_width_charts()
        self.create_time_series_charts()
        self.create_stacked_bar_charts()
        self.create_concentric_donut_charts()
        
        print("✅ All visualizations created successfully!")
        print("📁 Generated files:")
        print("  - line_charts.png")
        print("  - bar_charts.png")
        print("  - pie_charts.png")
        print("  - variable_width_charts.png")
        print("  - time_series_charts.png")
        print("  - stacked_bar_charts.png")
        print("  - concentric_donut_charts.png")

def main():
    """Main execution function"""
    print("🏠 COMPREHENSIVE HOUSING DATA VISUALIZATION SUITE")
    print("="*70)
    print("Creating professional visualizations with all chart types...")
    print()
    
    # Initialize data processor
    processor = HousingDataProcessor('housing.csv')
    
    if not processor.processed_data:
        print("❌ No data available for analysis")
        return
    
    # Create visualizer
    visualizer = ComprehensiveVisualizer(processor)
    
    # Create all visualizations
    if MATPLOTLIB_AVAILABLE:
        visualizer.create_comprehensive_dashboard()
    else:
        print("⚠ Matplotlib not available - skipping visualizations")
    
    print("\\n🎉 Comprehensive visualization suite completed!")
    print("Ready to build interactive dashboards!")

if __name__ == "__main__":
             main()
#"""Create line charts for various relationships"""
#if not MATPLOTLIB_AVAILABLE:
 #  return

#fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#fig.suptitle('Line Charts Analysis', fontsize=16, fontweight='bold', color=self.colors['text'])

# 1. Price vs RM relationship
#medv_data = self.processor.get_column_data('MEDV')
#rm_data = self.processor.get_column_data('RM')

#sorted_rm = sorted(zip(rm_data, medv_data))
#rm_sorted, medv_sorted = zip(*sorted_rm)

#axes[0, 0].plot(rm_sorted, medv_sorted, color=self.colors['primary'], 
#            linewidth=2, marker='o', markersize=4, alpha=0.7)
#axes[0, 0].set_title('Price vs Average Number of Rooms', fontweight='bold')
#axes[0, 0].set_xlabel('Average Number of Rooms')
#axes[0, 0].set_ylabel('Median Home Value ($)')
#axes[0, 0].grid(True, alpha=0.3)

# 2. Price vs RM relationship (reversed)
#axes[0, 1].plot(rm_sorted, medv_sorted, color=self.colors['secondary'], 
 #           linewidth=2, marker='o', markersize=4, alpha=0.7)
#axes[0, 1].set_title('Average Number of Rooms vs Price (Reverse)', fontweight='bold')
#axes[0, 1].set_xlabel('Average Number of Rooms')  