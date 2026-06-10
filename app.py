import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Matatu ML - Demand & Pricing Prediction",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚌 Matatu ML Dashboard</h1>
    <p>Machine Learning System for Demand & Pricing Prediction in Kenya</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Insights", "Model Performance", "Predictions", "Surge Analysis", "About"]
)

# Load data
@st.cache_data
def load_data():
    data = {}
    try:
        data['predictions'] = pd.read_csv('predictions_test.csv')
        data['simulation'] = pd.read_csv('simulation_results.csv')
        data['surge_ranking'] = pd.read_csv('route_surge_ranking_with_names.csv')
        data['surge_multipliers'] = pd.read_csv('surge_multipliers.csv')
        data['ab_comparison'] = pd.read_csv('ab_comparison.csv')
    except FileNotFoundError as e:
        st.warning(f"Some data files not found: {e}")
    return data

data = load_data()

# Page: Home
if page == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Matatu ML System
        
        This project develops a **machine learning system** that predicts:
        - **Demand**: How many passengers will use matatu services
        - **Pricing**: Optimal pricing strategies based on demand patterns
        
        ### Key Features:
        - 📊 **Data-driven insights** from real matatu operations in Kenya
        - 🤖 **Advanced ML models** including XGBoost and neural networks
        - 📈 **Demand forecasting** with surge pricing analysis
        - 💰 **Revenue optimization** strategies
        
        ### Project Components:
        1. Data ingestion and cleaning
        2. Exploratory data analysis (EDA)
        3. Feature engineering and preprocessing
        4. Model training and evaluation
        5. Surge pricing simulation
        """)
    
    with col2:
        st.metric("Project Type", "Final Year Project")
        st.metric("Focus Area", "ML & Pricing")

# Page: Data Insights
elif page == "Data Insights":
    st.header("📊 Data Insights")
    
    if 'predictions' in data and not data['predictions'].empty:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data['predictions']))
        with col2:
            st.metric("Features", data['predictions'].shape[1])
        with col3:
            st.metric("Data Points Available", f"{len(data['predictions']):,}")
        
        st.subheader("Data Sample")
        st.dataframe(data['predictions'].head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(data['predictions'].describe(), use_container_width=True)
    else:
        st.info("Prediction data not yet loaded. Check if predictions_test.csv exists.")

# Page: Model Performance
elif page == "Model Performance":
    st.header("🤖 Model Performance")
    
    st.markdown("""
    ### Models Developed:
    1. **Traditional ML Models** (Random Forest, Gradient Boosting)
    2. **XGBoost Model** - Enhanced gradient boosting with optimized hyperparameters
    3. **Neural Networks** - Deep learning approach for demand prediction
    
    ### Model Evaluation:
    The models were evaluated using:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
    """)
    
    if 'ab_comparison' in data and not data['ab_comparison'].empty:
        st.subheader("Model Comparison Results")
        st.dataframe(data['ab_comparison'], use_container_width=True)
        
        # Visualize if columns are available
        if len(data['ab_comparison'].columns) > 1:
            fig = px.bar(
                data['ab_comparison'],
                x=data['ab_comparison'].columns[0],
                y=data['ab_comparison'].columns[1:],
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison data not available.")

# Page: Predictions
elif page == "Predictions":
    st.header("🔮 Predictions & Results")
    
    if 'predictions' in data and not data['predictions'].empty:
        st.subheader("Test Set Predictions")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(data['predictions'].head(20), use_container_width=True)
        
        with col2:
            if 'actual' in data['predictions'].columns or data['predictions'].shape[1] >= 2:
                st.write("### Prediction Statistics")
                for col in data['predictions'].columns[:3]:
                    st.metric(col.replace('_', ' ').title(), f"{data['predictions'][col].mean():.2f}")
        
        # Download predictions
        csv = data['predictions'].to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions_test.csv",
            mime="text/csv"
        )
    else:
        st.info("Predictions data not available.")

# Page: Surge Analysis
elif page == "Surge Analysis":
    st.header("⚡ Surge Pricing Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'surge_ranking' in data and not data['surge_ranking'].empty:
            st.subheader("Routes by Surge Potential")
            fig = px.bar(
                data['surge_ranking'].head(15),
                x=data['surge_ranking'].columns[1] if len(data['surge_ranking'].columns) > 1 else data['surge_ranking'].columns[0],
                y=data['surge_ranking'].columns[0],
                title="Top 15 Routes by Surge Multiplier",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'surge_multipliers' in data and not data['surge_multipliers'].empty:
            st.subheader("Surge Multiplier Distribution")
            fig = px.histogram(
                data['surge_multipliers'],
                nbins=30,
                title="Distribution of Surge Multipliers",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Simulation results
    if 'simulation' in data and not data['simulation'].empty:
        st.subheader("Simulation Results")
        st.dataframe(data['simulation'].head(10), use_container_width=True)
        
        # Revenue comparison
        if len(data['simulation'].columns) > 1:
            fig = px.scatter(
                data['simulation'],
                x=data['simulation'].columns[0],
                y=data['simulation'].columns[1],
                title="Simulation Outcomes",
                size=data['simulation'].columns[2] if len(data['simulation'].columns) > 2 else None
            )
            st.plotly_chart(fig, use_container_width=True)

# Page: About
elif page == "About":
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        This is a **final year university project** focused on developing a machine learning 
        system that predicts:
        - **Demand** for matatu (mini-bus) services
        - **Optimal pricing** strategies
        
        ### Technologies Used:
        - **Python** - Core programming language
        - **Scikit-learn** - Machine learning models
        - **XGBoost** - Gradient boosting
        - **TensorFlow/Keras** - Neural networks
        - **Pandas & NumPy** - Data manipulation
        - **Streamlit** - Web interface
        - **Plotly** - Interactive visualizations
        """)
    
    with col2:
        st.markdown("""
        ### Dataset
        The project uses real-world data from matatu operations in Kenya, including:
        - Historical demand patterns
        - Pricing information
        - Route and timing data
        - Passenger behavior patterns
        
        ### Objectives
        1. Build accurate demand prediction models
        2. Optimize pricing strategies for revenue
        3. Identify surge pricing opportunities
        4. Provide actionable insights for operators
        
        ### Author
        **Joshua Okoth**
        
        For more details, visit the 
        [GitHub Repository](https://github.com/okothjosh/matatu_ml)
        """)
    
    st.divider()
    
    st.markdown("""
    ### Documentation
    The project includes comprehensive analysis across multiple stages:
    - **01_data_ingestion.ipynb** - Data collection and preparation
    - **02_preprocessing.ipynb** - Feature engineering
    - **03_model_training_evaluation.ipynb** - Model development
    - **04_XGBoost_model_training.ipynb** - Advanced gradient boosting
    - **05_surge_simulation.ipynb** - Pricing simulation
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚌 Matatu ML - Demand & Pricing Prediction System</p>
    <p>Final Year Project | Machine Learning for Transportation</p>
    <p><a href="https://github.com/okothjosh/matatu_ml">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
