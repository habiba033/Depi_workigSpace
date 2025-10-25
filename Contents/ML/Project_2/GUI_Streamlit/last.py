import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score
import warnings
import time
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎯 Smart Donation Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING FUNCTION TO ENSURE SPEED (PERFORMANCE FIX) ---
@st.cache_resource
def load_data_and_train_model():
    """Loads data, preprocesses, trains the model, and returns all necessary objects.
    Cached resource prevents reloading/retraining on every user interaction."""
    
    # Define the path to your data file
    data_path = r'C:\Users\habib\OneDrive\المستندات\Depi_workingSpace\Depi_workigSpace\Contents\ML\Project_2\GUI_Streamlit\census.csv'
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        return None 

    # 1. Data Cleaning: Strip whitespace
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # 2. Target Encoding and Labels
    df['donation'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    df['income_label'] = df['income'].apply(lambda x: '>50K' if x == '>50K' else '<=50K')

    # 3. Feature Engineering for Visualizations
    bins = [17, 30, 45, 60, 90]
    labels = ['Youth (17-29)', 'Adult (30-44)', 'Middle-Aged (45-59)', 'Senior (60+)']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # Ensure 'education_level' is correctly set up
    if 'education_level' not in df.columns:
        df['education_level'] = df['education'] 
    
    # 4. Model Training Preparation
    model_features_list = [
        'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 
        'workclass_Federal-gov', 'marital-status_Married-civ-spouse', 
        'occupation_Exec-managerial', 'relationship_Husband', 
        'race_White', 'sex_Male', 'native-country_United-States'
    ]

    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=False, prefix_sep='_')
    
    # Prepare X
    X = df_model[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].copy()
    
    for feature in model_features_list:
        if feature not in X.columns:
            if feature in df_model.columns:
                X[feature] = df_model[feature]
            else:
                X[feature] = 0

    X = X[model_features_list]
    y = df_model['donation'] 

    # Visualization DataFrame for complex plots (Sunburst, Violin)
    df_model_viz = pd.concat([df[['income_label', 'donation', 'workclass', 'education_level', 'sex']], X], axis=1)

    # 5. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Store results
    predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_pred_proba
    })
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        'df': df, 
        'df_model_viz': df_model_viz,
        'model': model, 
        'scaler': scaler, 
        'accuracy': accuracy, 
        'roc_auc': roc_auc,
        'predictions': predictions,
        'feature_importance': feature_importance
    }
# --- END CACHING FUNCTION ---

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling - matching HTML exactly */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
    }
    
    /* App container styling */
    .app-container {
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-radius: 1rem;
        max-width: 1200px;
        margin: 40px auto;
    }
    
    /* Sidebar styling - exact match */
    .sidebar {
        background-color: #1f2937;
        color: white;
        border-radius: 1rem 0 0 1rem;
    }
    
    .nav-button {
        padding: 1rem 1.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        width: 100%;
        background: none;
        border: none;
        color: white;
        font-size: 1rem;
        margin: 0.25rem 0;
    }
    
    .nav-button:hover, .nav-button.active {
        background-color: #374151;
        color: #38bdf8;
    }
    
    /* Main content styling */
    .main-content {
        padding: 2.5rem;
    }
    
    /* Custom metric cards - exact colors from HTML */
    .metric-card-blue {
        background: #3b82f6;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card-green {
        background: #10b981;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card-yellow {
        background: #f59e0b;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Prediction result styling - exact match */
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-success {
        background: #10b981;
        color: white;
    }
    
    .prediction-failure {
        background: #6b7280;
        color: white;
    }
    
    /* Custom buttons - exact styling */
    .stButton > button {
        background: #0ea5e9;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background: #0284c7;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    /* Input form styling */
    .stSelectbox > div > div {
        background-color: #1f2937; 
        color: white; 
        border: 1px solid #374151;
        border-radius: 0.375rem;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1f2937; 
        color: white; 
        border: 1px solid #374151; 
        border-radius: 0.375rem;
    }
    
    /* Quick actions styling */
    .quick-actions {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Form styling */
    .form-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Result display styling */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid #e5e7eb;
    }
    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 0;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: #f0f4f8; /* Light gray background for inactive tabs */
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: 600;
        border: 1px solid #e5e7eb;
        color: #4b5563; /* Dark gray text */
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff; /* White background for active tab */
        border-top: 3px solid #0ea5e9; /* Blue accent line */
        color: #1f2937; /* Darker text for active tab */
        border-bottom: 1px solid #ffffff; /* Hide border at bottom to blend with content */
    }
</style>
""", unsafe_allow_html=True)

class AdvancedDonationDashboard:
    def __init__(self, data_results):
        """Initialize the dashboard using cached data/model objects."""
        
        if data_results is None or self.check_data_integrity(data_results) is False:
            self.df = pd.DataFrame()
            self.df_model_viz = pd.DataFrame()
            self.model = None
            self.scaler = None
            self.accuracy = 0.0
            self.roc_auc = 0.0
            self.predictions = pd.DataFrame()
            self.feature_importance = pd.DataFrame()
            st.error("Dashboard failed to initialize. Check data file path or structure.")
            return

        self.df = data_results['df']
        self.df_model_viz = data_results['df_model_viz']
        self.model = data_results['model']
        self.scaler = data_results['scaler']
        self.accuracy = data_results['accuracy']
        self.roc_auc = data_results['roc_auc']
        self.predictions = data_results['predictions']
        self.feature_importance = data_results['feature_importance']

    def check_data_integrity(self, data_results):
        """Simple check to ensure core dataframes were created."""
        return not data_results['df'].empty and not data_results['predictions'].empty and not data_results['feature_importance'].empty

    # --- PLOTTING FUNCTIONS ---

    def create_age_income_line_chart(self):
        """Create line chart showing high income percentage by age."""
        age_income_data = self.df.groupby(['age', 'income_label']).size().unstack(fill_value=0)
        age_income_data['total'] = age_income_data.sum(axis=1)
        age_income_data['high_income_pct'] = (age_income_data.get('>50K', 0) / age_income_data['total'] * 100).fillna(0)
        
        fig = px.line(
            age_income_data.reset_index(),
            x='age',
            y='high_income_pct',
            title='High Income Percentage Trend by Age',
            labels={'age': 'Age', 'high_income_pct': 'High Income Percentage (%)'},
            color_discrete_sequence=['#0ea5e9']
        )
        fig.update_traces(mode='lines+markers', marker={'size': 4})
        fig.update_layout(height=400)
        return fig

    def create_education_bar_chart(self):
        """Create horizontal bar chart for high income percentage by education."""
        income_by_education = pd.crosstab(self.df['education_level'], self.df['income_label'])
        income_by_education['total'] = income_by_education.sum(axis=1)
        income_by_education['high_income_pct'] = (income_by_education.get('>50K', 0) / income_by_education['total'] * 100).fillna(0)
        
        sorted_data = income_by_education.reset_index().sort_values('high_income_pct', ascending=True)
        
        fig = px.bar(
            sorted_data,
            x='high_income_pct',
            y='education_level',
            orientation='h',
            title='High Income % by Education Level (Horizontal Bar Chart)',
            color='high_income_pct',
            color_continuous_scale='Mint',
            labels={'high_income_pct': 'High Income %', 'education_level': 'Education Level'}
        )
        fig.update_layout(height=400)
        return fig
        
    def create_race_stacked_bar_chart(self):
        """Create stacked bar chart for income distribution by race."""
        race_income = pd.crosstab(self.df['race'], self.df['income_label'])
        race_income_pct = race_income.div(race_income.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            race_income_pct.reset_index(),
            x='race',
            y=['<=50K', '>50K'],
            title='Income Distribution by Race (%) (Stacked Bar Chart)',
            barmode='stack',
            color_discrete_map={'<=50K':"#131561", '>50K':  "#4384f4"},
            labels={'value': 'Percentage (%)', 'variable': 'Income Class'}
        )
        fig.update_layout(yaxis_title="Percentage (%)", height=400)
        return fig

    def create_gender_income_bar_chart(self):
        """Create vertical bar chart (histogram) for income distribution based on sex/gender."""
        
        fig = px.histogram(
            self.df, 
            x='income_label', 
            color='sex', 
            title='Income Distribution based on Gender (Vertical Bar Chart)', 
            barmode='group',
            color_discrete_map={'Female': '#ec4899', 'Male': '#0ea5e9'},
            labels={'income_label': 'Income Class', 'sex': 'Gender', 'count': 'Count'}
        )
        
        fig.update_layout(height=400, bargap=0.2)
        return fig

    def create_sex_education_income_chart(self):
        """NEW: Create a grouped/stacked bar chart for Income distribution across Gender and Education."""
        
        # Calculate normalized counts (Percentage based on Education level)
        pivot_data = pd.crosstab([self.df['education_level'], self.df['sex']], self.df['income_label']).apply(lambda x: x / x.sum(), axis=1) * 100
        pivot_data = pivot_data.reset_index().rename(columns={'<=50K': 'Percent <=50K', '>50K': 'Percent >50K'})
        
        # Melt data for grouped bar chart visualization
        df_melted = pivot_data.melt(
            id_vars=['education_level', 'sex'], 
            value_vars=['Percent <=50K', 'Percent >50K'],
            var_name='Income Class',
            value_name='Percentage'
        )

        fig = px.bar(
            df_melted,
            x='education_level',
            y='Percentage',
            color='Income Class',
            facet_col='sex', # Group by sex
            title='Income Distribution by Education and Gender (Stacked Bar Chart)',
            color_discrete_map={'Percent <=50K': '#ef4444', 'Percent >50K': '#10b981'},
            labels={'education_level': 'Education Level'}
        )

        fig.update_layout(
            height=600, 
            barmode='relative', # Stacks the bars
            xaxis={'categoryorder':'total ascending'}
        )
        fig.for_each_xaxis(lambda x: x.update(showgrid=False))
        return fig
        
    def create_sunburst_chart(self):
        """NEW: Create a sunburst chart for hierarchical visualization (Workclass > Education > Income)."""
        
        # Prepare hierarchical data: Workclass -> Education -> Income
        df_sunburst = self.df_model_viz.groupby(['workclass', 'education_level', 'income_label']).size().reset_index(name='count')
        
        fig = px.sunburst(
            df_sunburst,
            path=['workclass', 'education_level', 'income_label'],
            values='count',
            color='income_label',
            color_discrete_map={'<=50K': '#ef4444', '>50K': '#10b981'},
            title='Hierarchical Income Distribution (Sunburst/Cocentric Chart)'
        )
        
        fig.update_layout(height=500)
        return fig

    def create_feature_importance_chart(self):
        """Create top 10 feature importance chart."""
        top_features = self.feature_importance.head(10)
        
        # *COLOR CHANGE APPLIED HERE* (Using the strong blue color for visibility)
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 10 Feature Importances (Model Drivers)', 
            color='importance',
            color_continuous_scale='Blues', # Changed from 'Teal' to 'Blues' for better visibility against white/gray background
            labels={'importance': 'Feature Importance Score', 'feature': 'Feature Name'}
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def create_correlation_heatmap(self):
        """Create a correlation heatmap for numerical features."""
        
        numerical_df = self.df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'donation']]
        
        corr_matrix = numerical_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='RdBu', 
            title="Correlation Heatmap of Numerical Features"
        )
        
        fig.update_xaxes(side="top")
        fig.update_layout(height=500)
        return fig

    def create_confusion_matrix(self):
        """Create confusion matrix."""
        cm = confusion_matrix(self.predictions['actual'], self.predictions['predicted'])
        
        fig = px.imshow(
            cm, 
            text_auto=True, 
            aspect="auto",
            title="Model Confusion Matrix",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="True Label"),
            x=['<=50K', '>50K'],
            y=['<=50K', '>50K']
        )
        
        fig.update_layout(height=400)
        return fig

    def create_roc_curve(self):
        """Create the ROC Curve."""
        
        fpr, tpr, thresholds = roc_curve(self.predictions['actual'], self.predictions['probability'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr, 
            mode='lines', 
            name=f'ROC Curve (AUC = {self.roc_auc:.4f})',
            line=dict(color='#0ea5e9', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            mode='lines', 
            name='Random Guess',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=400
        )
        return fig

    def create_top_features_distribution_violin_chart(self, top_n=5):
        """Create violin plot for the distribution of the top N features."""
        
        top_features_names = self.feature_importance['feature'].head(top_n).tolist()
        
        df_violin = self.df_model_viz[['income_label'] + top_features_names].copy()
        
        df_melted = df_violin.melt(
            id_vars=['income_label'], 
            value_vars=top_features_names, 
            var_name='Feature', 
            value_name='Value'
        )

        fig = px.violin(
            df_melted, 
            y='Value', 
            x='Feature', 
            color='income_label',
            box=True, 
            points='outliers', 
            title=f'Distribution of Top {top_n} Features by Income Class (Violin Plot)',
            color_discrete_map={'<=50K': '#fca5a5', '>50K': '#10b981'},
            labels={'income_label': 'Income Class'}
        )
        
        fig.update_layout(height=450, showlegend=True, xaxis_title='Top Features')
        return fig

    def create_smart_donation_gauge(self, value):
        """Create the Smart Donation Index gauge."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Smart Donation Index"},
            gauge = {
                'axis': {'range': [None, 90]},
                'bar': {'color': "#1f2937"},
                'steps': [
                    {'range': [0, 20], 'color': "#fca5a5"},
                    {'range': [20, 40], 'color': "#fcd34d"},
                    {'range': [40, 70], 'color': "#a7f3d0"},
                    {'range': [70, 90], 'color': "#10b981"}
                ],
                'threshold': {
                    'line': {"color": "#ef4444", "width": 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Inter"})
        return fig
    
    # --- PAGE RENDERERS ---

    def render_sidebar(self):
        """Render sidebar navigation and handle state."""
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.5rem; font-weight: bold; color: #38bdf8; margin-bottom: 0.5rem;">
                🎯 CharityML
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        def nav_button(icon, label, page_name):
            if st.sidebar.button(f"{icon} {label}", use_container_width=True, key=f"nav_{page_name}"):
                st.session_state.page = page_name
                st.rerun()

        nav_button("🏠", "Dashboard", "🏠 Dashboard")
        nav_button("💸", "Predict Income", "💸 Predict Income")
        nav_button("📊", "Visualize Insights", "📊 Visualize Insights")
        nav_button("⚙", "Train Model", "🔧 Train Model")
        
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("### 📈 Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Records", f"{len(self.df):,}")
        with col2:
            st.metric("Accuracy", f"{self.accuracy*100:.1f}%")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; color: #9ca3af; font-size: 0.8rem;">
            ML Engineer Nanodegree Project
        </div>
        """, unsafe_allow_html=True)
        
        return st.session_state.get('page', "🏠 Dashboard")

    def render_dashboard(self):
        """Render the main dashboard page with Quick Actions and Metrics."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                🎯 Comprehensive Dashboard – Overview
            </h1>
            <p style="color: #6b7280; font-size: 1rem; margin: 1rem 0 2rem 0;">
                Welcome to the predictive donation modeling application. Review key metrics and model insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card-blue">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">👥</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">{len(self.df):,}</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Total Records</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card-green">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏆</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">{self.accuracy*100:.1f}%</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card-yellow">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💰</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0;">{self.roc_auc:.4f}</div>
                <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">ROC AUC Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>---<br>", unsafe_allow_html=True)

        # --- QUICK ACTIONS (AS REQUESTED) ---
        st.header("Quick Actions")
        col1_qa, col2_qa, col3_qa = st.columns(3)
        
        with col1_qa:
            if st.button("💸 Predict Income", use_container_width=True, key="qa_predict"):
                st.session_state.page = "💸 Predict Income"
                st.rerun()
        
        with col2_qa:
            if st.button("📊 Visualize Insights", use_container_width=True, key="qa_visualize"):
                st.session_state.page = "📊 Visualize Insights"
                st.rerun()
        
        with col3_qa:
            if st.button("⚙ Train Model", use_container_width=True, key="qa_train"):
                st.session_state.page = "🔧 Train Model"
                st.rerun()
        
        st.markdown("<br>---<br>", unsafe_allow_html=True)

        # --- CONTENT BELOW QUICK ACTIONS ---
        st.header("Model Performance and Key Drivers")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Feature Importances")
            st.plotly_chart(self.create_feature_importance_chart(), use_container_width=True)
        with col2:
            st.subheader("Overall Income Distribution (Donut Chart)")
            donation_counts = self.df['donation'].value_counts()
            names = ['Unlikely Donor (<=50K)', 'Likely Donor (>50K)']
            values = [donation_counts.get(0, 0), donation_counts.get(1, 0)]
            st.plotly_chart(px.pie(
                names=names,
                values=values,
                title='Overall Donation Distribution (Donut Chart)',
                color_discrete_sequence=['#ef4444', '#10b981'],
                hole=0.4
            ), use_container_width=True)
            
    def render_prediction_page(self):
        """Render the prediction page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                💸 Predict Donor Potential
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        # Define the actual education levels and their 'education-num' mapping
        education_levels_map = {
            "Doctorate": 16.0, "Masters": 14.0, "Bachelors": 13.0, "Assoc-acdm": 12.0, 
            "Assoc-voc": 11.0, "Some-college": 10.0, "HS-grad": 9.0, "11th": 7.0, "9th": 5.0
        }
        
        with col1:
            st.markdown("""
            <div class="form-container">
                <h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem;">Donor Profile Input (Features)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                
                # --- Numerical Inputs ---
                col_age, col_edu = st.columns(2)
                age = col_age.number_input("Age", min_value=17, max_value=90, value=45, step=1, key="input_age")
                education_level_display = col_edu.selectbox(
                    "Education Level", list(education_levels_map.keys()), index=2, key="input_education_level"
                )
                
                col_work, col_hours = st.columns(2)
                workclass = col_work.selectbox(
                    "Workclass (Federal-gov indicator)", self.df['workclass'].unique().tolist(), key="input_workclass"
                )
                hours_per_week = col_hours.number_input("Hours/Week", min_value=1, max_value=99, value=40, step=1, key="input_hours_per_week")
                
                col_gain, col_loss = st.columns(2)
                capital_gain = col_gain.number_input("Capital Gain (USD)", min_value=0, max_value=100000, value=5000, step=100, key="input_capital_gain")
                capital_loss = col_loss.number_input("Capital Loss (USD)", min_value=0, max_value=10000, value=0, step=100, key="input_capital_loss")
                
                # --- Categorical Inputs ---
                marital_status = st.selectbox("Marital Status (Married-civ-spouse indicator)", self.df['marital-status'].unique().tolist(), key="input_marital_status")
                occupation = st.selectbox("Occupation (Exec-managerial indicator)", self.df['occupation'].unique().tolist(), key="input_occupation")
                relationship = st.selectbox("Relationship (Husband indicator)", self.df['relationship'].unique().tolist(), key="input_relationship")
                race = st.selectbox("Race (White indicator)", self.df['race'].unique().tolist(), key="input_race")
                sex = st.selectbox("Sex (Male indicator)", self.df['sex'].unique().tolist(), key="input_sex")
                native_country = st.selectbox("Native Country (US indicator)", self.df['native-country'].unique().tolist(), key="input_native_country")
                
                submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)
                
                if submitted:
                    education_num = education_levels_map.get(education_level_display, 9.0)

                    input_data = np.array([[
                        age, education_num, capital_gain, capital_loss, hours_per_week, 
                        1 if workclass == "Federal-gov" else 0, 
                        1 if marital_status == "Married-civ-spouse" else 0, 
                        1 if occupation == "Exec-managerial" else 0, 
                        1 if relationship == "Husband" else 0, 
                        1 if race == "White" else 0, 
                        1 if sex == "Male" else 0, 
                        1 if native_country == "United-States" else 0
                    ]])
                    
                    input_scaled = self.scaler.transform(input_data)
                    prob = self.model.predict_proba(input_scaled)[0][1]
                    prediction = self.model.predict(input_scaled)[0]
                    
                    st.session_state.prediction_result = {
                        'probability': prob,
                        'prediction': prediction,
                        'smart_index': prob * 90
                    }
                    st.rerun() 
        
        with col2:
            st.markdown("""
            <div class="result-container">
                <h2 style="color: #1f2937; margin-bottom: 1.5rem; text-align: center; font-size: 1.25rem;">Prediction Output</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if 'prediction_result' in st.session_state:
                result = st.session_state.prediction_result
                gauge_fig = self.create_smart_donation_gauge(result['smart_index'])
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-result prediction-success">
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 0;">&gt; $50K (Likely Donor)</div>
                        <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">High confidence score in donor prediction.</div>
                        <div style="margin: 0.75rem 0 0 0; font-size: 0.75rem; opacity: 0.8; display: flex; align-items: center; justify-content: center;">
                            🧪 Model Probability: {result['probability']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result prediction-failure">
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 0;">&lt;= $50K (Unlikely Donor)</div>
                        <div style="font-size: 0.875rem; margin: 0.5rem 0 0 0;">Low confidence score in donor prediction.</div>
                        <div style="margin: 0.75rem 0 0 0; font-size: 0.75rem; opacity: 0.8; display: flex; align-items: center; justify-content: center;">
                            🧪 Model Probability: {result['probability']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; color: #6b7280; padding: 2rem;">
                    <h3>Input details and click "Run Prediction"</h3>
                    <p>The Smart Donation Index will be displayed here.</p>
                </div>
                """, unsafe_allow_html=True)

    def render_visualization_page(self):
        """Render the visualization page (now focusing on deep-dive charts)."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                📊 Deep-Dive Visualizations & Exploratory Analysis
            </h1>
            <p style="color: #6b7280; font-size: 0.875rem;">
                Explore complex demographic relationships, feature distributions, and correlations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- TABBED VISUALIZATION CONTENT ---
        tab_demographics, tab_relationships, tab_diagnostics = st.tabs(["👥 Demographics & Trends", "🔗 Correlations & Distributions", "🔬 Model Diagnostics"])
        
        with tab_demographics:
            st.header("📈 Income Trends")
            st.plotly_chart(self.create_age_income_line_chart(), use_container_width=True)
            
            # *Combined Chart*
            st.subheader("Combined Income Distribution by Gender and Education")
            st.plotly_chart(self.create_sex_education_income_chart(), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Income by Education (Horizontal Bar Chart)")
                st.plotly_chart(self.create_education_bar_chart(), use_container_width=True)
            with col2:
                st.subheader("Income by Gender (Vertical Bar Chart/Histogram)")
                st.plotly_chart(self.create_gender_income_bar_chart(), use_container_width=True)
            
            st.subheader("🌍 Income by Race Distribution (Stacked Bar Chart)")
            st.plotly_chart(self.create_race_stacked_bar_chart(), use_container_width=True)

        with tab_relationships:
            st.header("🔑 Complex Data Relationships")
            
            st.subheader("Hierarchical Income Distribution (Sunburst Chart / Cocentric Donut)")
            st.plotly_chart(self.create_sunburst_chart(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Correlation Heatmap (Numerical Features)")
                st.plotly_chart(self.create_correlation_heatmap(), use_container_width=True)
            with col2:
                st.subheader("Top 5 Feature Value Distribution (Violin Plot)")
                st.plotly_chart(self.create_top_features_distribution_violin_chart(top_n=5), use_container_width=True)
            
            # --- MESSAGE REMOVED ---
            # Removed the conditional markdown block containing the time series note.


        with tab_diagnostics:
            st.header("🔬 Model Assessment")
            
            st.subheader("Top 10 Feature Importances")
            st.plotly_chart(self.create_feature_importance_chart(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ROC Curve")
                st.plotly_chart(self.create_roc_curve(), use_container_width=True)
            with col2:
                st.subheader("Confusion Matrix")
                st.plotly_chart(self.create_confusion_matrix(), use_container_width=True)


    def render_train_model_page(self):
        """Render the train model page."""
        st.markdown("""
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="color: #1f2937; font-size: 2rem; font-weight: 700; margin: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;">
                ⚙ Model Retraining
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb;">
            <h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem;">Model Training Configuration</h2>
            <p style="color: #6b7280; margin-bottom: 1.5rem;">Configure and simulate retraining your model with advanced parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛 Model Parameters")
            algorithm = st.selectbox("Algorithm", ["Random Forest (Current)", "XGBoost", "AdaBoost", "SVM"])
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        
        with col2:
            st.markdown("### 📊 Training Configuration")
            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        if st.button("🚀 Start Retraining Simulation", use_container_width=True):
            with st.spinner(f"Training {algorithm} with {n_estimators} estimators..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Simulate new metrics
                new_acc = self.accuracy + np.random.uniform(-0.02, 0.02)
                new_auc = self.roc_auc + np.random.uniform(-0.01, 0.01)
                self.accuracy = max(0.75, min(0.95, new_acc))
                self.roc_auc = max(0.75, min(0.99, new_auc))
                
                # Re-run feature importance simulation (must maintain the same features)
                new_importance = np.random.dirichlet(np.ones(len(self.feature_importance)), size=1)[0]
                self.feature_importance['importance'] = new_importance / new_importance.sum()
                self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)

                st.success("✅ Model retraining completed successfully! Metrics updated.")
                st.balloons()
                st.session_state.page = "🏠 Dashboard"
                st.rerun()

    def run_dashboard(self):
        """Run the main dashboard application."""
        if self.df is not None and not self.df.empty:
            if 'page' not in st.session_state:
                st.session_state.page = "🏠 Dashboard"
            
            selected_page = self.render_sidebar()
            
            st.markdown("""<div class="app-container">""", unsafe_allow_html=True)
            st.markdown("""<div class="main-content">""", unsafe_allow_html=True)
            
            if selected_page == "🏠 Dashboard":
                self.render_dashboard()
            elif selected_page == "💸 Predict Income":
                self.render_prediction_page()
            elif selected_page == "📊 Visualize Insights":
                self.render_visualization_page()
            elif selected_page == "🔧 Train Model":
                self.render_train_model_page()

            st.markdown("""</div>""", unsafe_allow_html=True)
            st.markdown("""</div>""", unsafe_allow_html=True)
        else:
            st.error("Cannot run the dashboard. Data loading failed.")

def main():
    """Main function to run the dashboard."""
    # Load all data and model only once using the cache
    data_results = load_data_and_train_model()

    dashboard = AdvancedDonationDashboard(data_results)
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()