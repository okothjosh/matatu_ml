import streamlit as st

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

# Main landing page
st.markdown("""
<div class="main-header">
    <h1>🚌 Matatu ML Dashboard</h1>
    <p>Machine Learning System for Demand & Pricing Prediction in Kenya</p>
</div>
""", unsafe_allow_html=True)

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
    
    ### Getting Started:
    Use the navigation menu on the left to explore:
    1. **Data Insights** - Explore the dataset
    2. **Model Performance** - View model metrics
    3. **Predictions** - See prediction results
    4. **Surge Analysis** - Route rankings and pricing
    5. **About** - Project details
    """)

with col2:
    st.metric("Project Type", "Final Year Project")
    st.metric("Focus Area", "ML & Pricing")
    st.metric("Framework", "Streamlit")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚌 Matatu ML - Demand & Pricing Prediction System</p>
    <p>Final Year Project | Machine Learning for Transportation</p>
    <p><a href="https://github.com/okothjosh/matatu_ml">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
